from jax.numpy import ndarray
from numpy import ndarray
from jaxrl.agents.sac.sac_learner import CoTASPLearner
import jax.numpy as jnp
from jaxrl.datasets.dataset import Batch
from jaxrl.networks.policies_PRE import Decoder_PRE
from jaxrl.networks.rnd_net import rnd_network
import jax
from flax.core import FrozenDict, unfreeze, freeze
from jaxrl.networks.common import TrainState, PRNGKey, Params, InfoDict, MPNTrainState
from optax import global_norm
import jaxrl.networks.common as utils_fn
from typing import Any, Tuple
import functools
from jaxrl.agents.sac.sac_learner import _update_critic, _update_temp
import numpy as np
from jaxrl.dict_learning.task_dict import OnlineDictLearnerV2
from jax import custom_jvp
from jax import vmap

@custom_jvp
def clip_fn(x):
    return jnp.minimum(jnp.maximum(x, 0), 1.0)

@clip_fn.defjvp
def f_jvp(primals, tangents):
    # Custom derivative rule for clip_fn 
    # x' = 1, when 0 < x < 1;
    # x' = 0, otherwise.
    x, = primals
    x_dot, = tangents
    ans = clip_fn(x)
    ans_dot = jnp.where(x >= 1.0, 0, jnp.where(x <= 0, 0, 1.0)) * x_dot
    return ans, ans_dot

def ste_step_fn(x):
    # Create an exactly-zero expression with Sterbenz lemma that has
    # an exactly-one gradient.
    # Straight-through estimator of step function
    # its derivative is equal to 1 when 0 < x < 1, 0 otherwise.
    zero = clip_fn(x) - jax.lax.stop_gradient(clip_fn(x))
    return zero + jax.lax.stop_gradient(jnp.heaviside(x, 0))

rnd_rate = 0.01
def embedding(actor, task_id):
    output_list = []
    embedding_name_list = ['embeds_bb_0', 'embeds_bb_1', 'embeds_bb_2', 'embeds_bb_3']
    for embedding_name in embedding_name_list:
        if embedding_name in actor.params.keys():
            output_list.append(actor.params[embedding_name]['embedding'][task_id])
    embed = jnp.stack(output_list)
    embed = vmap(ste_step_fn)(embed)
    embed = jax.lax.expand_dims(embed, dimensions=(0, 1)) # add batch and channel
    return embed

def _update_theta(
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, temp: TrainState, 
    batch: Batch, decoder: TrainState, rnd_net: TrainState) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_decoder_loss_fn(actor_params: Params, decoder_params: Params, rnd_net_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params': actor_params}, batch.observations, jnp.array([task_id])
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        phi_st = dicts['encoder_output']
        #print(f'phi_st.shape is {phi_st.shape}')
        #print(f'batch.actions.shape is {batch.actions.shape}')
        pre_input = jnp.concatenate([phi_st, batch.actions], -1)
        phi_next_st = decoder.apply_fn({'params': decoder_params}, pre_input)
        #recon_loss = jnp.mean((pre_next_st - batch.next_observations)**2)
        embedding_vector = embedding(actor, task_id)
        # rnd_net
        target_next_st = rnd_net.apply_fn({'params': rnd_net_params}, batch.next_observations, embedding_vector)
        rnd_loss = jnp.mean((phi_next_st - target_next_st)**2)

        _info = {
            'rnd_loss': rnd_loss,
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'means': dicts['means'].mean()
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        return actor_loss + rnd_rate * rnd_loss, _info
        
    # grads of actor
    # NOTE I need to deep understanding the jax.grad function
    grads_actor_decoder, actor_info = jax.grad(actor_decoder_loss_fn, has_aux=True, argnums=[0, 1])(actor.params, decoder.params, rnd_net.params)
    grads_actor, grads_decoder = grads_actor_decoder
    # recording info
    g_norm = global_norm(grads_actor)
    actor_info['g_norm_actor'] = g_norm
    for p, v in param_mask.items():
        if p[-1] == 'kernel':
            actor_info['used_capacity_'+p[0]] = 1.0 - jnp.mean(v)

    # Masking gradients according to cumulative binary masks
    unfrozen_grads = unfreeze(grads_actor)
    for path, value in param_mask.items():
        cursor = unfrozen_grads
        for key in path[:-1]:
            if key in cursor:
                cursor = cursor[key]
        cursor[path[-1]] *= value
    
    # only update policy parameters (theta)
    new_actor = actor.apply_grads_theta(grads=freeze(unfrozen_grads))
    new_decoder = decoder.apply_gradients(grads=grads_decoder)

    return rng, new_actor, actor_info, new_decoder

def _update_alpha(
    rng: PRNGKey, task_id: int, param_mask: FrozenDict[str, Any],
    actor: MPNTrainState, critic: TrainState, temp: TrainState, 
    batch: Batch, decoder: TrainState, rnd_net: TrainState) -> Tuple[PRNGKey, MPNTrainState, InfoDict]:

    rng, key = jax.random.split(rng)
    def actor_decoder_loss_fn(actor_params: Params, decoder_params: Params, rnd_net_params: Params) -> Tuple[jnp.ndarray, InfoDict]:
        dist, dicts = actor.apply_fn(
            {'params': actor_params}, batch.observations, jnp.array([task_id])
        )
        actions = dist.sample(seed=key)
        log_probs = dist.log_prob(actions)
        q1, q2 = critic(batch.observations, actions)
        q = jnp.minimum(q1, q2)
        actor_loss = (log_probs * temp() - q).mean()

        phi_st = dicts['encoder_output']
        #print(f'phi_st.shape is {phi_st.shape}')
        #print(f'batch.actions.shape is {batch.actions.shape}')
        pre_input = jnp.concatenate([phi_st, batch.actions], -1)
        phi_next_st = decoder.apply_fn({'params': decoder_params}, pre_input)
        #recon_loss = jnp.mean((pre_next_st - batch.next_observations)**2)
        embedding_vector = embedding(actor, task_id)
        # rnd_net
        target_next_st = rnd_net.apply_fn({'params': rnd_net_params}, batch.next_observations, embedding_vector)
        rnd_loss = jnp.mean((phi_next_st - target_next_st)**2)

        _info = {
            'rnd_loss': rnd_loss,
            'hac_sac_loss': actor_loss,
            'entropy': -log_probs.mean(),
            'means': dicts['means'].mean()
        }
        for k in dicts['masks']:
            _info[k+'_rate_act'] = jnp.mean(dicts['masks'][k])

        return actor_loss + rnd_rate * rnd_loss, _info
    
    # grads of actor
    print("updating alpha")
    grads_actor_decoder, actor_info = jax.grad(actor_decoder_loss_fn, has_aux=True, argnums=[0, 1])(actor.params, decoder.params, rnd_net.params)
    grads_actor, grads_decoder = grads_actor_decoder
    # recording info
    g_norm = global_norm(grads_actor)
    actor_info['g_norm_actor'] = g_norm
    for p, v in param_mask.items():
        if p[-1] == 'kernel':
            actor_info['used_capacity_'+p[0]] = 1.0 - jnp.mean(v)

    # only update coefficients (alpha)
    new_actor = actor.apply_grads_alpha(grads=grads_actor)
    new_decoder = decoder.apply_gradients(grads=grads_decoder)

    return rng, new_actor, actor_info, new_decoder
 

@functools.partial(jax.jit, static_argnames=('update_target'))
def _update_cotasp_jit(rng: PRNGKey, task_id: int, tau: float, discount: float, 
    target_entropy: float, optimize_alpha: bool, param_mask: FrozenDict[str, Any], 
    actor: MPNTrainState, critic: TrainState, target_critic: TrainState, 
    update_target, temp: TrainState, batch: Batch, decoder: TrainState, rnd_net: TrainState
    ) -> Tuple[PRNGKey, MPNTrainState, TrainState, TrainState, TrainState, InfoDict]:
    # optimizing critics 
    new_rng, new_critic, new_target_critic, critic_info = _update_critic(
        rng, task_id, actor, critic, target_critic, update_target, temp, 
        batch, discount, tau
    )

    # optimizing either alpha or theta
    new_rng, new_actor, actor_info, new_decoder = jax.lax.cond(
        optimize_alpha,
        _update_alpha,
        _update_theta,
        new_rng, task_id, param_mask, actor, new_critic, temp, batch, decoder, rnd_net
    )

    # updating temperature coefficient
    new_temp, temp_info = _update_temp(
        temp, actor_info['entropy'], target_entropy
    )

    return new_rng, new_actor, new_temp, new_critic, new_target_critic, {
        **actor_info,
        **temp_info,
        **critic_info
    }, new_decoder

class RNDLearner(CoTASPLearner):
    def __init__(self, seed: int, observations: jnp.ndarray, actions: jnp.ndarray, task_num: int, load_policy_dir: str | None = None, load_dict_dir: str | None = None, update_dict=True, update_coef=True, dict_configs: dict = ..., pi_opt_configs: dict = ..., q_opt_configs: dict = ..., t_opt_configs: dict = ..., actor_configs: dict = ..., critic_configs: dict = ..., tau: float = 0.005, discount: float = 0.99, target_update_period: int = 1, target_entropy: float | None = None, init_temperature: float = 1):
        super().__init__(seed, observations, actions, task_num, load_policy_dir, load_dict_dir, update_dict, update_coef, dict_configs, pi_opt_configs, q_opt_configs, t_opt_configs, actor_configs, critic_configs, tau, discount, target_update_period, target_entropy, init_temperature)
        decoder_key, rnd_key = jax.random.split(jax.random.PRNGKey(110), 2)

        decoder_def = Decoder_PRE()
        decoder_params = FrozenDict(decoder_def.init(decoder_key, jnp.ones((1, 1028))).pop('params'))
        decoder_network = TrainState.create(
            apply_fn=decoder_def.apply,
            params=decoder_params,
            tx=utils_fn.set_optimizer(**t_opt_configs)
        )
        self.t_opt_configs = t_opt_configs
        self.decoder = decoder_network

        rnd_net_def = rnd_network()
        rnd_net_params = FrozenDict(rnd_net_def.init(rnd_key, jnp.ones((1,12)), jnp.ones((1, 1, 4, 1024))).pop('params'))
        rnd_net_ = TrainState.create(
            apply_fn=rnd_net_def.apply,
            params=rnd_net_params,
            tx=utils_fn.set_optimizer(**t_opt_configs)
        )
        self.rnd_net = rnd_net_

        # dict4layer_rnd
        # preset dict learner for each layer:
        self.dict4layers_rnd = {}
        for id_layer, hidn in enumerate(actor_configs['hidden_dims']):
            dict_learner = OnlineDictLearnerV2(
                384,
                hidn,
                seed+id_layer+1,
                None, # whether using svd dictionary initialization
                **dict_configs)
            self.dict4layers_rnd[f'embeds_bb_{id_layer}'] = dict_learner
        
        if load_dict_dir is not None:
            for k in self.dict4layers_rnd.keys():
                self.dict4layers_rnd[k].load(f'{load_dict_dir}/{k}.pkl')
    
    def start_task(self, task_id: int, description: str):
        task_e = self.task_encoder.encode(description)[np.newaxis, :]
        self.task_embeddings.append(task_e)

        # set initial alpha for each layer of MPN
        actor_params = unfreeze(self.actor.params)
        for k in self.actor.params.keys():
            if k.startswith('embeds'):
                alpha_l = self.dict4layers[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                actor_params[k]['embedding'] = actor_params[k]['embedding'].at[task_id].set(alpha_l)
        self.actor = self.actor.update_params(freeze(actor_params))

        # modification
        rnd_net_params = unfreeze(self.rnd_net.params)
        for k in self.rnd_net.params.keys():
            if k.startswith('embeds'):
                alpha_l = self.dict4layers_rnd[k].get_alpha(task_e)
                alpha_l = jnp.asarray(alpha_l.flatten())
                # Replace the i-th row
                rnd_net_params[k]['embedding'] = rnd_net_params[k]['embedding'].at[task_id].set(alpha_l)
        self.rnd_net = self.rnd_net.update_params(freeze(rnd_net_params))

    def end_task(self, task_id: int, save_actor_dir: str, save_dict_dir: str):
        decoder_def = Decoder_PRE()
        decoder_key = jax.random.split(jax.random.PRNGKey(110), 1)
        decoder_params = FrozenDict(decoder_def.init(decoder_key[0], jnp.ones((1, 1028))).pop('params'))
        decoder_network = TrainState.create(
            apply_fn=decoder_def.apply,
            params=decoder_params,
            tx=utils_fn.set_optimizer(**self.t_opt_configs)
        )

        self.decoder = decoder_network

        # update dictionary learners
        dict_rnd_stats = {}
        if self.update_dict:
            for k in self.rnd_net.params.keys():
                if k.startswith('embeds'):
                    optimal_alpha_l = self.rnd_net.params[k]['embedding'][task_id]
                    optimal_alpha_l = np.array([optimal_alpha_l.flatten()])
                    task_e = self.task_embeddings[task_id]
                    # online update dictionary via CD
                    self.dict4layers[k].update_dict(optimal_alpha_l, task_e)
                    dict_rnd_stats[k] = {
                        'sim_mat': self.dict4layers[k]._compute_overlapping(),
                        'change_of_d': np.array(self.dict4layers[k].change_of_dict)
                    }
        else:
            for k in self.rnd_net.params.keys():
                if k.startswith('embeds'):
                    dict_rnd_stats[k] = {
                        'sim_mat': self.dict4layers[k]._compute_overlapping(),
                        'change_of_d': 0
                    }

        return super().end_task(task_id, save_actor_dir, save_dict_dir), dict_rnd_stats
    
    def update(self, task_id: int, batch: Batch, optimize_alpha: bool = False) -> utils_fn.Dict[str, float]:
        if not self.update_coef:
            optimize_alpha = False
            
        update_target = self.step % self.target_update_period == 0

        new_rng, new_actor, new_temp, new_critic, new_target_critic, info, new_decoder = _update_cotasp_jit(
            self.rng, task_id, self.tau, self.discount, self.target_entropy, optimize_alpha, 
            self.param_masks, self.actor, self.critic, self.target_critic, update_target,
            self.temp, batch, self.decoder, self.rnd_net
        )

        self.step += 1
        self.rng = new_rng
        self.actor = new_actor
        self.temp = new_temp  
        self.critic = new_critic
        self.target_critic = new_target_critic   
        self.decoder = new_decoder

        return info

    
