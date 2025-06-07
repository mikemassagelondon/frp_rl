import jax.numpy as jnp
import jax
import flax.linen as nn
from flax import struct
import chex
from typing import Tuple, Optional
from flax.linen.initializers import constant, orthogonal



class PaddingAugmentatoin(nn.Module):
    out_size: int = 4
    in_size: int = 2

    @nn.compact
    def __call__(self, x):
        weight =  jnp.eye(self.out_size)[:self.in_size,:]
        return nn.Dense(self.out_size,  kernel_init=lambda *_: weight, bias_init=constant(0.0))(x)

def create_periodic_weight(input_dim: int, output_dim: int, period: int) -> jnp.ndarray:
    """Create a weight matrix that periodically copies input and normalizes by sqrt(copies).
    
    Args:
        input_dim: Dimension of input vector
        output_dim: Dimension of output vector 
        period: Number of elements to copy before repeating
        
    Returns:
        Weight matrix of shape (input_dim, output_dim)
    """
    # Calculate number of complete copies that will fit
    num_copies = output_dim // input_dim
    
    # Create base identity matrix for one copy
    base = jnp.eye(input_dim)
    
    # Stack the identity matrix num_copies times horizontally
    repeated = jnp.tile(base, (1, num_copies))
    
    # Pad with zeros if needed to reach output_dim
    remaining = output_dim - (num_copies * input_dim)
    if remaining > 0:
        padding = jnp.zeros((input_dim, remaining))
        weight = jnp.concatenate([repeated, padding], axis=1)
    else:
        weight = repeated
        
    # Normalize by sqrt(num_copies) to maintain variance
    weight = weight / jnp.sqrt(num_copies)
    
    return weight