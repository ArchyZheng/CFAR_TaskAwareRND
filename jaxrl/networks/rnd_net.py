from jax.numpy import ndarray
from jaxrl.networks.policies import MetaPolicy
import flax.linen as nn
import jax.numpy as jnp
from typing import Sequence
import jax
from jaxrl.networks.common import default_init, activation_fn, MLP
from flax.core import FrozenDict
from jaxrl.networks.common import TrainState
import jaxrl.networks.common as utils_fn
from jax import custom_jvp
from jaxrl.networks.common import InfoDict, TrainState, PRNGKey, Params, \
    MPNTrainState
from jax import lax
from jax.example_libraries import stax


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

class RND_MLP_1(nn.Module):
    hidden_dims: Sequence[int] = (256, 256, 256)
    name_activation: str = 'leaky_relu'
    output_dim: int=256
    use_layer_norm: bool = True
    @nn.compact
    def __call__(self, x) -> jnp.ndarray:
        inputs = x
        rnd_out = MLP(
            (*self.hidden_dims, self.output_dim), 
            activations=activation_fn(self.name_activation),
            use_layer_norm=self.use_layer_norm,
            activate_final=False)(inputs)
        return rnd_out


class RND_CNN(nn.Module):
    def setup(self):
        key = jax.random.PRNGKey(42)
        self.conv1_kernel = jax.random.normal(key, (1, 3, 3, 3)) # IOHW <- initial kernel conv1
        key = jax.random.PRNGKey(43)
        self.conv2_kernel = jax.random.normal(key, (3, 1, 3, 3)) # IOHW <- initial kernel conv2

        self.mlp_def = RND_MLP_1()
        rnd_mlp_params = self.mlp_def.init(
            jax.random.PRNGKey(44), jax.random.normal(jax.random.PRNGKey(44), (1, 256))
        )
        _, self.rnd_mlp_params = FrozenDict(rnd_mlp_params).pop('params')


    def __call__(self, x):
        #NOTE - careful with the shapes. for maxpool it will use NHWC format.
        #x -> (N, 1, 4, 1024)
        dn_1 = lax.conv_dimension_numbers(x.shape, self.conv1_kernel.shape, ('NCHW', 'IOHW', 'NCHW'))
        out_1 = lax.conv_general_dilated(x, self.conv1_kernel, (1, 1), 'SAME', (1, 1), (1, 1), dn_1)
        out_1 = nn.relu(out_1)
        out_2 = nn.max_pool(jnp.transpose(out_1, (0, 2, 3, 1)), window_shape=(2, 2), strides=(2, 2)) # out_3 -> (N, 2, 512, 3)
        out_2 = jnp.transpose(out_2, (0, 3, 1, 2)) # out_3 -> (N, 3, 2, 512)

        dn_2 = lax.conv_dimension_numbers(out_2.shape, self.conv2_kernel.shape, ('NCHW', 'IOHW', 'NCHW'))
        out_3 = lax.conv_general_dilated(out_2, self.conv2_kernel, (1, 1), 'SAME', (1, 1), (1, 1), dn_2)
        out_3 = nn.relu(out_3) # out_3 -> (N, 1, 2, 512)
        out_4 = nn.max_pool(jnp.transpose(out_3, (0, 2, 3, 1)), window_shape=(2, 2), strides=(2, 2)) # out_3 -> (N, 1, 256, 1)
        out_4 = jnp.transpose(out_4, (0, 3, 1, 2)) # out_3 -> (N, 1, 1, 256)

        out_5 = out_4.squeeze([1, 2]) # out_5 -> (N, 256) squeeze the dimensions
        final_out = self.mlp_def.apply({'params': self.rnd_mlp_params}, out_5)
        #print(f'mlp x.shape is {x.shape}')
        # Reshape to the desired output shape
        return final_out # final_out -> (N, 256)
    
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
        # self.rnd_cnn_params = FrozenDict(self.rnd_cnn.init(self.cnn_key, jax.random.normal(jax.random.PRNGKey(110), (1, 1, 4, 1024))).pop('params'))
        # MLP setup
        # self.mlp_next_observation_dims = [256, 256, 256, 256]
        self.mlp1_hidden_dims = [256, 256, 256, 256]
        self.mlp_final_layers_dims = [128, 64]
        self.mlp1 = [nn.Dense(hidn, kernel_init=default_init()) \
            for hidn in self.mlp1_hidden_dims]
        self.mlp2_hidden_dims = [128, 64]
        self.mlp2 = [nn.Dense(hidn, kernel_init=default_init()) \
            for hidn in self.mlp2_hidden_dims]

    def __call__(self, 
                 x: jnp.ndarray,
                #  t: jnp.ndarray,
                 embedding):
        #print(f'the dimension of mask_t is {mask_t.shape}')

        # CNN for phi(task_embedding)
        rnd_cnn_output = self.rnd_cnn(embedding)
        #print(f'the dimension of mask_t is {rnd_cnn_output.shape}')

        # MLP for phi(st+1)
        for i, layer in enumerate(self.mlp1):
            x = layer(x)
            if i < len(self.mlp1) - 1:
                x = nn.relu(x)
        phi_next_st = x  
        #print(f'phi_next_st.shape is {phi_next_st.shape}')
        rnd_cnn_output = jnp.tile(rnd_cnn_output, (phi_next_st.shape[0], 1))
        target_next_st = jnp.multiply(phi_next_st, rnd_cnn_output)
        for i, layer in enumerate(self.mlp2):
            target_next_st = layer(target_next_st)
            if i < len(self.mlp1) - 1:
                target_next_st = nn.relu(target_next_st)
        

        return target_next_st

