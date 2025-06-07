import jax
import jax.numpy as jnp
import pytest
from gymnax import make
from envs.wrappers import GymnaxRewardNormWrapper

def test_gymnax_reward_norm_wrapper_dynamic():
    """Test that the GymnaxRewardNormWrapper correctly normalizes rewards using the dynamic strategy."""
    # Create a CartPole environment
    env, env_params = make("CartPole-v1")
    
    # Apply the reward normalization wrapper with dynamic strategy
    norm_env = GymnaxRewardNormWrapper(env, strategy='dynamic')
    
    # Initialize the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = norm_env.reset(subkey, env_params)
    
    # Take a few steps and check that rewards are normalized
    rewards = []
    original_rewards = []
    
    for _ in range(10):
        key, subkey = jax.random.split(key)
        # In CartPole, action 1 means push right
        action = 1
        obs, state, reward, done, info = norm_env.step(subkey, state, action, env_params)
        rewards.append(reward)
        original_rewards.append(info["original_reward"])
        
        if done:
            key, subkey = jax.random.split(key)
            obs, state = norm_env.reset(subkey, env_params)
    
    # Check that original rewards are all 1.0 (CartPole gives +1 for each step)
    assert all(r == 1.0 for r in original_rewards), "Original CartPole rewards should all be 1.0"
    
    # Check that normalized rewards are different from original rewards
    assert not all(jnp.isclose(r, 1.0) for r in rewards), "Normalized rewards should differ from original rewards"
    
    # Check that normalized rewards have reasonable values (not too large or small)
    assert all(jnp.abs(r) < 10.0 for r in rewards), "Normalized rewards should have reasonable magnitudes"

def test_gymnax_reward_norm_wrapper_minmax():
    """Test that the GymnaxRewardNormWrapper correctly normalizes rewards using the minmax strategy."""
    # Create a Pendulum environment (minmax strategy is designed for Pendulum)
    env, env_params = make("Pendulum-v1")
    
    # Apply the reward normalization wrapper with minmax strategy
    norm_env = GymnaxRewardNormWrapper(env, strategy='minmax', reward_range=(-16.2736044, 0), max_steps=200)
    
    # Initialize the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = norm_env.reset(subkey, env_params)
    
    # Take a few steps and check that rewards are normalized
    rewards = []
    original_rewards = []
    
    for _ in range(10):
        key, subkey = jax.random.split(key)
        # In Pendulum, action is a continuous value between -2 and 2
        action = jnp.array([0.0])
        obs, state, reward, done, info = norm_env.step(subkey, state, action, env_params)
        rewards.append(reward)
        original_rewards.append(info["original_reward"])
        
        if done:
            key, subkey = jax.random.split(key)
            obs, state = norm_env.reset(subkey, env_params)
    
    # Check that normalized rewards are different from original rewards
    assert not all(jnp.isclose(r, o) for r, o in zip(rewards, original_rewards)), "Normalized rewards should differ from original rewards"
    
    # Check that normalized rewards have reasonable values (between -1/max_steps and 1/max_steps)
    assert all(jnp.abs(r) <= 1.0/norm_env.max_steps for r in rewards), "Normalized rewards should be within reasonable bounds"

def test_gymnax_reward_norm_wrapper_fixed():
    """Test that the GymnaxRewardNormWrapper correctly normalizes rewards using the fixed strategy."""
    # Create a CartPole environment
    env, env_params = make("CartPole-v1")
    
    # Apply the reward normalization wrapper with fixed strategy
    norm_env = GymnaxRewardNormWrapper(env, strategy='fixed', max_steps=200)
    
    # Initialize the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = norm_env.reset(subkey, env_params)
    
    # Take a few steps and check that rewards are normalized
    rewards = []
    
    for _ in range(10):
        key, subkey = jax.random.split(key)
        # In CartPole, action 1 means push right
        action = 1
        obs, state, reward, done, info = norm_env.step(subkey, state, action, env_params)
        rewards.append(reward)
        
        if done:
            key, subkey = jax.random.split(key)
            obs, state = norm_env.reset(subkey, env_params)
    
    # Check that normalized rewards match the fixed strategy
    # For CartPole, reward is 1.0, which should be mapped to 1.0/max_steps
    expected_reward = 1.0 / 200.0
    assert all(jnp.isclose(r, expected_reward) for r in rewards), "Normalized rewards should follow fixed strategy"

def test_gymnax_reward_norm_wrapper_state_update():
    """Test that the GymnaxRewardNormWrapper correctly updates its state."""
    # Create a CartPole environment
    env, env_params = make("CartPole-v1")
    
    # Apply the reward normalization wrapper with dynamic strategy
    norm_env = GymnaxRewardNormWrapper(env, strategy='dynamic', update_rate=0.1)
    
    # Initialize the environment
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    obs, state = norm_env.reset(subkey, env_params)
    
    # Check initial state values
    assert state.running_mean == 0.0, "Initial running mean should be 0.0"
    assert state.running_var == 1.0, "Initial running variance should be 1.0"
    assert state.discounted_return == 0.0, "Initial discounted return should be 0.0"
    
    # Take a step and check that state is updated
    key, subkey = jax.random.split(key)
    action = 1
    obs, new_state, reward, done, info = norm_env.step(subkey, state, action, env_params)
    
    # Check that discounted return is updated
    assert new_state.discounted_return > 0.0, "Discounted return should be updated"
    
    # Check that running statistics are updated
    assert new_state.running_var != state.running_var, "Running variance should be updated"
    assert new_state.running_mean != state.running_mean, "Running mean should be updated"
