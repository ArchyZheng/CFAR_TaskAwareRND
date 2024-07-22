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

class RND_CNN(nn.Module):
    def setup(self):
        self.conv1 = nn.Conv(features=32, kernel_size=(3, 3), strides=(1, 1))
        self.conv2 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1))
        #self.conv3 = nn.Conv(features=128, kernel_size=(3, 3), strides=(1, 1))
        #self.conv4 = nn.Conv(features=256, kernel_size=(3, 3), strides=(1, 1))
        self.mlp = nn.Dense(features=256)

    def __call__(self, x):
        #print(f'before cnn x.shape is {x.shape}')
        x = self.conv1(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        #print(f'conv1 x.shape is {x.shape}')
        
        x = self.conv2(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        #print(f'conv2 x.shape is {x.shape}')
        '''
        x = self.conv3(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        print(f'conv3 x.shape is {x.shape}')
        
        x = self.conv4(x)
        x = nn.relu(x)
        print(f'conv4 x.shape is {x.shape}')
        '''
        x = x.reshape((256,64))
        x = self.mlp(x)
        #print(f'mlp x.shape is {x.shape}')
        # Reshape to the desired output shape
        x = x.reshape((256,256))
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
        self.embeds_bb = [nn.Embed(self.task_num, hidn, embedding_init=default_init()) \
            for hidn in self.hidden_dims]
        # CNN setup
        self.rnd_cnn = RND_CNN()
        self.rnd_cnn_params = FrozenDict(self.rnd_cnn.init(self.cnn_key, jnp.ones((4, 256, 1024))).pop('params')) # was [10, 256, 1024]
        # MLP setup
        self.mlp1_hidden_dims = [256, 256, 256, 256]
        self.mlp1 = [nn.Dense(hidn, kernel_init=default_init()) \
            for hidn in self.mlp1_hidden_dims]
        self.mlp2_hidden_dims = [128, 64]
        self.mlp2 = [nn.Dense(hidn, kernel_init=default_init()) \
            for hidn in self.mlp2_hidden_dims]

    def __call__(self, 
                 x: jnp.ndarray,
                 t: jnp.ndarray):
        combined_list = []
        for i, layer in enumerate(self.embeds_bb):
            phi_l = ste_step_fn(self.embeds_bb[i](t))
            #print(f'the dimension of phi_l is {phi_l.shape}')
            mask_l = jnp.broadcast_to(phi_l, [x.shape[0], 1024])
            combined_list.append(mask_l)
        mask_t = jnp.stack(combined_list, axis=0)
        #print(f'the dimension of mask_t is {mask_t.shape}')

        # CNN for phi(task_embedding)
        rnd_cnn_output = jnp.ones((256, 256))
        if mask_t.shape[1] == 256:   # batch size is 256
            rnd_cnn_output = self.rnd_cnn.apply({'params': self.rnd_cnn_params}, mask_t)
            #print(f'the dimension of mask_t is {rnd_cnn_output.shape}')

        # MLP for phi(st+1)
        for i, layer in enumerate(self.mlp1):
            x = layer(x)
            if i < len(self.mlp1) - 1:
                x = nn.relu(x)
        phi_next_st = x  #
        #print(f'phi_next_st.shape is {phi_next_st.shape}')
        target_next_st = jnp.multiply(phi_next_st, rnd_cnn_output)
        for i, layer in enumerate(self.mlp2):
            target_next_st = layer(target_next_st)
            if i < len(self.mlp1) - 1:
                target_next_st = nn.relu(target_next_st)
        

        return target_next_st

