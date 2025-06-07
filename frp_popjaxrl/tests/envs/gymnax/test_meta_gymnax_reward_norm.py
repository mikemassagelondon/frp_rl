import jax
import jax.numpy as jnp
import pytest
#from gymnax import gymnax_make
from envs.registration import make
from envs.wrappers import GymnaxRewardNormWrapper

def test_meta_gymnax_reward_normalization():
    """Test that rewards are normalized when using MetaGymnax environments."""
    # Create a MetaGymnax environment
    env, env_params = make("MetaGymnaxCartPole-v1", meta_dim=32, meta_depth=1)
    
    # Initialize the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, env_params)
    
    # Take a few steps and check that rewards are normalized
    rewards = []
    original_rewards = []
    
    for _ in range(10):
        key, subkey = jax.random.split(key)
        # In CartPole, action 1 means push right
        action = 1
        obs, state, reward, done, info = env.step(subkey, state, action, env_params)
        rewards.append(reward)
        original_rewards.append(info["original_reward"])
        
        if done:
            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey, env_params)
    
    # Check that original rewards are all 1.0 (CartPole gives +1 for each step)
    assert all(r == 1.0 for r in original_rewards), "Original CartPole rewards should all be 1.0"
    
    # Check that normalized rewards are different from original rewards
    assert not all(jnp.isclose(r, 1.0) for r in rewards), "Normalized rewards should differ from original rewards"
    
    # Check that normalized rewards have reasonable values (not too large or small)
    assert all(jnp.abs(r) < 10.0 for r in rewards), "Normalized rewards should have reasonable magnitudes"

def test_meta_gymnax_pendulum_reward_normalization():
    """Test that rewards are normalized when using MetaGymnax Pendulum environment."""
    # Create a MetaGymnax Pendulum environment
    env, env_params = make("MetaGymnaxPendulum-v1", meta_dim=32, meta_depth=1)
    
    # Initialize the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, env_params)
    
    # Take a few steps and check that rewards are normalized
    rewards = []
    original_rewards = []
    
    for _ in range(10):
        key, subkey = jax.random.split(key)
        # In Pendulum, action is a continuous value between -2 and 2
        action = jnp.array([0.0])
        obs, state, reward, done, info = env.step(subkey, state, action, env_params)
        rewards.append(reward)
        original_rewards.append(info["original_reward"])
        
        if done:
            key, subkey = jax.random.split(key)
            obs, state = env.reset(subkey, env_params)
    
    # Check that original rewards are negative (Pendulum gives negative rewards)
    assert all(r < 0 for r in original_rewards), "Original Pendulum rewards should be negative"
    
    # Check that normalized rewards are different from original rewards
    assert not all(jnp.isclose(r, o) for r, o in zip(rewards, original_rewards)), "Normalized rewards should differ from original rewards"
    
    # Check that normalized rewards have reasonable values (not too large or small)
    assert all(jnp.abs(r) < 10.0 for r in rewards), "Normalized rewards should have reasonable magnitudes"

def test_meta_gymnax_reward_normalization_state_update():
    """Test that the normalization state is properly updated in MetaGymnax environments."""
    # Create a MetaGymnax environment
    env, env_params = make("MetaGymnaxCartPole-v1", meta_dim=32, meta_depth=1)
    
    # Initialize the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = env.reset(subkey, env_params)
    
    # Extract the normalization state from the MetaEnvironment state
    # The MetaEnvironment state structure is:
    # state.env_state = GymnaxRewardNormState(env_state=CartPoleState(...), running_mean=..., running_var=..., discounted_return=...)
    norm_state = state.env_state
    
    # Check initial state values
    assert norm_state.running_mean == 0.0, "Initial running mean should be 0.0"
    assert norm_state.running_var == 1.0, "Initial running variance should be 1.0"
    assert norm_state.discounted_return == 0.0, "Initial discounted return should be 0.0"
    
    # Take a step and check that state is updated
    key, subkey = jax.random.split(key)
    action = 1
    max_num_step=200
    obs, new_state, reward, done, info = env.step(subkey, state, action, env_params)

    
    # Extract the new normalization state
    new_norm_state = new_state.env_state
    # [For Cline: here we get assert the following]
    # Check that running statistics are updated
    assert new_norm_state.running_var != norm_state.running_var, "Running variance should be updated"
    assert new_norm_state.running_mean != norm_state.running_mean, "Running mean should be updated"

def test_direct_gymnax_vs_meta_gymnax_reward_normalization():
    """Test that direct Gymnax and MetaGymnax environments have similar reward normalization."""
    # Create both types of environments
    direct_env, direct_params = make("GymnaxCartPole-v1")
    meta_env, meta_params = make("MetaGymnaxCartPole-v1", meta_dim=32, meta_depth=1)
    
    # Initialize the environments
    key = jax.random.PRNGKey(0)
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    meta_params_vec = jnp.zeros((32,))
    
    direct_obs, direct_state = direct_env.reset(subkey1, direct_params)
    meta_obs, meta_state = meta_env.reset(subkey2, meta_params)
    
    # Take a step in both environments with the same action
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    action = 1
    
    direct_obs, direct_state, direct_reward, direct_done, direct_info = direct_env.step(
        subkey1, direct_state, action, direct_params
    )
    
    meta_obs, meta_state, meta_reward, meta_done, meta_info = meta_env.step(
        subkey2, meta_state, action, meta_params
    )
    
    # Check that original rewards are the same
    assert direct_info["original_reward"] == meta_info["original_reward"], "Original rewards should be the same"
    
    # Check that normalized rewards are similar (they might not be exactly the same due to different initialization)
    # But they should be in the same ballpark
    assert jnp.abs(direct_reward - meta_reward) < 0.5, "Normalized rewards should be similar"
