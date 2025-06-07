import jax
import jax.numpy as jnp
import numpy as np
import pytest
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from envs.environments.popgym_minesweeper import MineSweeper
from algorithms.s5 import StackedEncoderModel, init_S5SSM, make_DPLR_HiPPO

class EncoderModel(nn.Module):
    """Encoder model to match the structure in ppo_s5.py"""
    d_model: int
    
    def setup(self):
        self.encoder_0 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))
        self.encoder_1 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))  # Match internal S5 dimension
    
    def __call__(self, x):
        x = self.encoder_0(x)
        x = nn.leaky_relu(x)
        x = self.encoder_1(x)
        x = nn.leaky_relu(x)
        return x

def test_convolution_dtype_consistency():
    """Test convolution with explicit dtype handling and jax-native kernel."""
    env = MineSweeper(dims=(4, 4), num_mines=2)
    key = jax.random.PRNGKey(0)
    
    # Create test grid with explicit dtype
    hidden_grid = jnp.zeros((4, 4), dtype=jnp.int8)
    hidden_grid = hidden_grid.at[1, 1].set(1)
    hidden_grid = hidden_grid.at[2, 2].set(1)
    
    # Create kernel using jnp instead of np
    kernel = jnp.ones((3, 3), dtype=jnp.int8)
    
    # Test convolution with explicit casting
    neighbor_grid = jax.scipy.signal.convolve2d(
        jnp.asarray(hidden_grid, dtype=jnp.float32),  # Cast to float32 first
        jnp.asarray(kernel, dtype=jnp.float32),  # Cast to float32 first
        mode="same"
    )
    
    # Cast back to int8
    neighbor_grid = jnp.array(neighbor_grid, dtype=jnp.int8)
    
    print("\nHidden grid:")
    print(hidden_grid)
    print("\nNeighbor grid:")
    print(neighbor_grid)
    
    # Verify values make sense
    assert jnp.all(neighbor_grid >= 0), "Negative values in neighbor grid"
    assert jnp.all(neighbor_grid <= 9), "Too large values in neighbor grid"

def test_minesweeper_s5_integration():
    """Test MineSweeper environment with S5 architecture integration."""
    # Initialize environment
    env = MineSweeper(dims=(4, 4), num_mines=2)
    key = jax.random.PRNGKey(0)
    
    # S5 configuration matching ppo_s5.py
    d_model = 128  # Hidden dimension - match encoder output
    ssm_size = 256  # State dimension
    blocks = 1
    block_size = int(ssm_size / blocks)
    
    # Get DPLR HiPPO matrices with full size first
    Lambda, _, _, V, _ = make_DPLR_HiPPO(ssm_size)
    
    # Then halve the sizes for SSM initialization
    block_size = block_size // 2
    ssm_size = ssm_size // 2
    Lambda = Lambda[:block_size]
    V = V[:, :block_size]
    Vinv = V.conj().T
    
    # Initialize S5 SSM
    ssm = init_S5SSM(
        H=d_model,
        P=ssm_size,  # Already halved above
        Lambda_re_init=Lambda.real,
        Lambda_im_init=Lambda.imag,
        V=V,
        Vinv=Vinv,
        C_init="lecun_normal",  # Match ppo_s5.py
        discretization="zoh",
        dt_min=0.001,
        dt_max=0.1,
        conj_sym=True,
        clip_eigs=False,  # Match ppo_s5.py
        bidirectional=False
    )
    
    # Create encoder model
    encoder = EncoderModel(d_model=d_model)
    
    # Create S5 model
    s5_model = StackedEncoderModel(
        ssm=ssm,
        d_model=d_model,
        n_layers=1,
        activation="full_glu",  # Match ppo_s5.py
        do_norm=True,
        prenorm=True,
        do_gtrxl_norm=True
    )
    
    # Initialize parameters
    init_key1, init_key2, init_key3 = jax.random.split(key, 3)
    
    # Match input structure from ppo_s5.py
    num_envs = 1
    obs_shape = (2,)  # num_mines
    seq_len = 1
    
    # Create inputs with consistent dimensions
    dummy_obs = jnp.zeros((seq_len, num_envs, *obs_shape))  # (time, batch, features)
    dummy_reset = jnp.zeros((seq_len, num_envs), dtype=bool)  # (time, batch)
    
    # Initialize encoder
    encoder_params = encoder.init(init_key1, dummy_obs[0])
    
    # Initialize S5 with encoded input
    encoded_dummy = encoder.apply(encoder_params, dummy_obs[0])
    encoded_dummy = encoded_dummy[None, :]  # Add time dimension back
    
    hidden = s5_model.initialize_carry(batch_size=num_envs, hidden_size=d_model, n_layers=1)
    s5_params = s5_model.init({'params': init_key2, 'dropout': init_key3}, hidden, encoded_dummy, dummy_reset)
    
    # Test environment reset
    obs, state = env.reset_env(key, env.default_params)
    
    # Print dtypes for debugging
    print("\nInitial state dtypes:")
    print("mine_grid dtype:", state.mine_grid.dtype)
    print("neighbor_grid dtype:", state.neighbor_grid.dtype)
    print("observation dtype:", obs.dtype)
    
    # Test forward pass through S5
    @jax.jit
    def forward(encoder_params, s5_params, hidden, obs, reset):
        # Reshape inputs to match expected format
        obs = obs[None, None, :]  # (time=1, batch=1, features)
        reset = jnp.zeros((1, 1), dtype=bool)  # (time=1, batch=1)
        
        # Encode input
        encoded = encoder.apply(encoder_params, obs[0])
        encoded = encoded[None, :]  # Add time dimension back
        
        # Pass through S5
        return s5_model.apply(s5_params, hidden, encoded, reset, rngs={'dropout': jax.random.PRNGKey(0)})
    
    # Run forward pass
    reset = jnp.zeros((), dtype=bool)  # Scalar
    new_hidden, output = forward(encoder_params, s5_params, hidden, obs, reset)
    
    print("\nS5 output shape:", output.shape)
    print("S5 output dtype:", output.dtype)
    
    # Verify dtypes remain consistent
    assert state.mine_grid.dtype == jnp.int8, "Mine grid dtype changed"
    assert state.neighbor_grid.dtype == jnp.int8, "Neighbor grid dtype changed"
    assert output.dtype == jnp.float32, "S5 output dtype should be float32"

if __name__ == "__main__":
    test_convolution_dtype_consistency()
    test_minesweeper_s5_integration()
