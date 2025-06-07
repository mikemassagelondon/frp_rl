import pytest
import jax
import jax.numpy as jnp
from envs.environments.popgym_concentration import Concentration
from envs.meta_environment import MetaEnvironment, create_meta_environment

def test_base_concentration_env():
    """Test the base concentration environment without meta wrapper."""
    env = Concentration()
    params = env.default_params
    
    # Test observation space
    obs_space = env.observation_space(params)
    print(f"\nBase env observation space shape: {obs_space.shape}")
    assert len(obs_space.shape) == 1
    
    # Test reset
    key = jax.random.PRNGKey(0)
    obs, state = env.reset_env(key, params)
    print(f"Reset observation shape: {obs.shape}")
    assert obs.shape == obs_space.shape
    
    # Test step
    action = 0
    obs, state, reward, done, info = env.step_env(key, state, action, params)
    print(f"Step observation shape: {obs.shape}")
    assert obs.shape == obs_space.shape

def test_meta_concentration_env():
    """Test the concentration environment with meta wrapper."""
    # Test with different meta_dim values
    meta_dims = [64, 128, 256]
    
    for meta_dim in meta_dims:
        print(f"\nTesting with meta_dim: {meta_dim}")
        
        # Create environment
        env_kwargs = {}
        meta_kwargs = {
            'meta_dim': meta_dim,
            'meta_depth': 1,
            'meta_max_depth': 2,
            'meta_with_adjoint': False,
            'num_trials_per_episode': 16,
            'rng': jax.random.PRNGKey(42),
            'meta_const_aug': 'tiling'  # Use tiling for dimension reduction
        }
        
        try:
            env = create_meta_environment("concentration", env_kwargs, meta_kwargs)
            print("Successfully created meta environment")
            
            # Get and print dimensions
            input_dim = env.input_dim
            print(f"Input dimension: {input_dim}")
            print(f"Word tensor shape: {env.words.shape}")
            
            # Test reset
            key = jax.random.PRNGKey(0)
            params = env.default_params
            obs, state = env.reset_env(key, params)
            print(f"Meta env reset observation shape: {obs.shape}")
            
            # Test step
            action = 0
            obs, state, reward, done, info = env.step_env(key, state, action, params)
            print(f"Meta env step observation shape: {obs.shape}")
            
            print(f"Test passed for meta_dim={meta_dim}")
            
        except Exception as e:
            print(f"Error with meta_dim={meta_dim}: {str(e)}")
            raise

if __name__ == "__main__":
    test_base_concentration_env()
    test_meta_concentration_env()
