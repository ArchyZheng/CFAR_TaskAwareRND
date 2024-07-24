from jax.numpy import ndarray
from jaxrl.networks.policies import MetaPolicy
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
import jax
from jaxrl.networks.common import default_init, activation_fn
from flax.core import FrozenDict
from jaxrl.networks.common import TrainState
import jaxrl.networks.common as utils_fn
from jax import custom_jvp
from jaxrl.networks.common import InfoDict, TrainState, PRNGKey, Params, \
    MPNTrainState

class RND_CNN(nn.Module):
    def setup(self):
        #// TODO - kernel_init should written as np.sqrt(2)
        self.model = nn.Sequential([
            nn.Conv(features=32, kernel_size=(8, 8), strides=(4, 4), name='conv1', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu'),
            nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2), name='conv2', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu'),
            nn.Conv(features=64, kernel_size=(4, 4), strides=(1, 1), name='conv3', kernel_init=default_init(jnp.sqrt(2))),
            activation_fn('relu')])
        self.mlp = nn.Sequential([
            nn.Dense(features=512, name='fc1'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=512, name='fc2'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=256, name='fc3')])
    @nn.compact
    def __call__(self, x):
        x = self.model(x)
        x = jnp.reshape(x, (x.shape[0], -1)) # flatten
        x = self.mlp(x)
        return x
    
class rnd_network(nn.Module):
    """
    This is created by Chengqi. 
    Input the features
    Output the x. 
    """
    features = [1024, 256, 64, 16, 12]

    def setup(self):
        self.cnn_key = jax.random.PRNGKey(110)
        self.hidden_dims = [1024, 1024, 1024, 1024]
        self.task_num = 10
        # CNN setup
        self.rnd_cnn = RND_CNN()
        self.rnd_cnn_params = FrozenDict(self.rnd_cnn.init(self.cnn_key, jnp.ones((1, 4, 1024, 1))).pop('params')) # was [10, 256, 1024]
        # MLP setup
        self.mlp_obs = nn.Sequential([
            nn.Dense(features=256, name='fc1'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=256, name='fc2'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=256, name='fc3')])
        self.mlp_output = nn.Sequential([
            nn.Dense(features=256, name='output_fc1'),
            nn.LayerNorm(),
            activation_fn('leaky_relu'),
            nn.Dense(features=64, name='output_fc2')])
    @nn.compact
    def __call__(self, 
                 x: jnp.ndarray,
                 mask_t: jnp.ndarray):

        rnd_cnn_output = self.rnd_cnn.apply({'params': self.rnd_cnn_params}, mask_t)
        rnd_cnn_output = jnp.tile(rnd_cnn_output, (x.shape[0], 1))
        phi_next_st = self.mlp_obs(x)
        target_next_st = jnp.multiply(phi_next_st, rnd_cnn_output)
        target_next_st = self.mlp_output(target_next_st)
        return target_next_st