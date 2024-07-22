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

    
class Decoder_PRE(nn.Module):
    """
    This is created by Chengqi. 
    Input the features
    Output the x. 
    """
    features = [1028, 256, 64, 16, 12]

    def setup(self):
        # where to add non-linearity function?
        self.decoder1 = [nn.Dense(hidn, kernel_init=default_init()) for hidn in self.features]

    def __call__(self, x):
        for i, layer in enumerate(self.decoder1): # BUG: why we should use self.decoder[0]???
            x = layer(x)
            if i < len(self.decoder1) - 1:
                x = activation_fn('relu')(x)
        return x

class PolicyPRE(MetaPolicy):

    def setup(self):
        super().setup()
        self.decoder_def = Decoder_PRE() # use default settings
        decoder_key = jax.random.split(jax.random.PRNGKey(110), 1)
        self.decoder_params = FrozenDict(self.decoder_def.init(decoder_key[0], jnp.ones((1, 1028))).pop('params'))
        self.decoder = TrainState(self.decoder_def, self.decoder_params)
        
    
    def __call__(self, x: jnp.ndarray, t: jnp.ndarray, temperature: float = 1):
        """

        Ouput:
        @base_dist: the base distribution
        @original_output: the original output. including masks, mean and stddev

        NOTE @reconstructed_x: the reconstructed x, but this part will be add into the original_output dict
        """
        base_dist, original_output = super().__call__(x, t, temperature)
        assert 'encoder_output' in original_output
        reconstructed_x = self.decoder(original_output['encoder_output'])
        original_output['reconstructed_x'] = reconstructed_x
        
        return base_dist, original_output
    
    def reset_decoder(self):
        decoder_key = jax.random.split(jax.random.PRNGKey(110), 1)
        self.decoder.init(decoder_key[0], jnp.ones((1, 1028)))
