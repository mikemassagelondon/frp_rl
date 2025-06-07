import unittest
import jax
import jax.numpy as jnp
import numpy as np
from gymnax.environments import spaces
from envs.meta_environment import create_meta_environment

class TestMetaPendulum(unittest.TestCase):
    def setUp(self):
        # Common parameters for meta environment
        self.meta_kwargs = {
            'meta_depth': 1,
            'meta_dim': 32,
            'meta_max_depth': 2,
            'meta_with_adjoint': False,
            'rng': jax.random.PRNGKey(42),
            'num_trials_per_episode': 16
        }
        
        # Create environment instance
        self.env = create_meta_environment(
            "pendulum", 
            env_kwargs={'max_steps_in_episode': 200, 'noise_sigma': 0.0}, 
            meta_kwargs=self.meta_kwargs
        )

    def test_continuous_action_space(self):
        """Test if continuous action space is handled correctly"""
        action_space = self.env.action_space(self.env.default_params)
        
        # Verify it's a Box space (continuous)
        self.assertTrue(isinstance(action_space, spaces.Box))
        
        # Check action space shape and bounds
        self.assertEqual(action_space.shape, (1,))
        self.assertEqual(action_space.dtype, jnp.float32)
        self.assertEqual(self.env.num_actions, 1)  # Should be 1 for pendulum

    def test_observation_space(self):
        """Test observation space properties"""
        obs_space = self.env.observation_space(self.env.default_params)
        
        # Meta observation includes augmented state plus 3 additional values
        expected_dim = self.meta_kwargs['meta_dim'] + 3
        self.assertEqual(obs_space.shape, (expected_dim,))
        self.assertEqual(obs_space.dtype, jnp.float32)
        
        # Check bounds
        np.testing.assert_array_equal(obs_space.low, -jnp.ones(expected_dim))
        np.testing.assert_array_equal(obs_space.high, jnp.ones(expected_dim))

    def test_reset_behavior(self):
        """Test environment reset behavior"""
        key = jax.random.PRNGKey(0)
        
        # Reset environment
        obs, state = self.env.reset_env(key, self.env.default_params)
        
        # Check observation shape and type
        self.assertEqual(obs.shape, (self.meta_kwargs['meta_dim'] + 3,))
        self.assertEqual(obs.dtype, jnp.float32)
        
        # Check initial state values
        self.assertEqual(state.trial_num, 0)
        self.assertEqual(state.total_steps, 0)
        
        # Check if last 3 values are correct (action=0, done=0, reset=1)
        np.testing.assert_array_equal(obs[-3:], jnp.array([0.0, 0.0, 1.0]))

    def test_step_behavior(self):
        """Test environment step behavior with continuous actions"""
        key = jax.random.PRNGKey(0)
        key_reset, key_step = jax.random.split(key)
        
        # Reset environment
        obs, state = self.env.reset_env(key_reset, self.env.default_params)
        
        # Take a step with a continuous action
        action = jnp.array([0.5])  # Use a continuous action value
        next_obs, next_state, reward, done, info = \
            self.env.step_env(key_step, state, action, self.env.default_params)
        
        # Check observation shape and type
        self.assertEqual(next_obs.shape, (self.meta_kwargs['meta_dim'] + 3,))
        self.assertEqual(next_obs.dtype, jnp.float32)
        
        # Check state updates
        self.assertEqual(next_state.total_steps, state.total_steps + 1)
        
        # Check if last 3 values are correct (action, done, reset)
        np.testing.assert_array_equal(next_obs[-3:], jnp.array([0.5, 0.0, 0.0]))

    def test_trial_boundaries(self):
        """Test if trial boundaries are handled correctly"""
        key = jax.random.PRNGKey(0)
        state = self.env.reset_env(key, self.env.default_params)[1]
        
        # Run until environment signals done for the current trial
        max_steps = 5000  # Safety limit (increased for Pendulum which may take longer)
        step = 0
        trial_num = 0
        
        while step < max_steps:
            key, subkey = jax.random.split(key)
            action = jnp.array([0.0])
            
            _, state, _, done, _ = \
                self.env.step_env(subkey, state, action, self.env.default_params)
            
            if state.trial_num > trial_num:
                # Trial boundary detected
                trial_num = state.trial_num
                # Check if trial count is within expected range
                self.assertLessEqual(trial_num, self.meta_kwargs['num_trials_per_episode'])
            
            if done:
                # Episode should end after specified number of trials
                self.assertEqual(state.trial_num, self.meta_kwargs['num_trials_per_episode'])
                break
            
            step += 1
        
        # Ensure we didn't hit the safety limit
        self.assertLess(step, max_steps, "Episode did not complete within safety limit")

    def test_reward_range(self):
        """Test if rewards are properly scaled"""
        key = jax.random.PRNGKey(0)
        state = self.env.reset_env(key, self.env.default_params)[1]
        
        rewards = []
        for _ in range(100):  # Collect some sample rewards
            key, subkey = jax.random.split(key)
            action = jnp.array([jax.random.uniform(subkey, minval=-2.0, maxval=2.0)])
            _, state, reward, done, _ = \
                self.env.step_env(subkey, state, action, self.env.default_params)
            rewards.append(reward)
            
            if done:
                state = self.env.reset_env(key, self.env.default_params)[1]
        
        rewards = jnp.array(rewards)
        # Check if rewards are properly bounded
        self.assertTrue(jnp.all(rewards >= -1.0))
        self.assertTrue(jnp.all(rewards <= 1.0))

    def test_auto_reset(self):
        """Test if environment auto-resets correctly when done"""
        key = jax.random.PRNGKey(0)
        state = self.env.reset_env(key, self.env.default_params)[1]
        
        # Run until we get a done signal
        max_steps = 5000
        step = 0
        done = False
        last_obs = None
        last_state = None
        
        while not done and step < max_steps:
            key, subkey = jax.random.split(key)
            action = jnp.array([0.0])
            
            obs, state, _, done, _ = \
                self.env.step(subkey, state, action, self.env.default_params)
            
            if done:
                last_obs = obs
                last_state = state
            
            step += 1
        
        # Verify we got a done signal
        self.assertTrue(done, "Environment never reached done state")
        
        # After done, the next step should call reset_env
        key, subkey = jax.random.split(key)
        action = jnp.array([0.0])
        next_obs, next_state = self.env.reset_env(subkey, self.env.default_params)
        
        # Verify the environment reset
        # - State should be reset (trial_num back to 0)
        self.assertEqual(next_state.trial_num, 0)
        # - Last 3 values of obs should indicate reset (action=0, done=0, reset=1)
        np.testing.assert_array_equal(next_obs[-3:], jnp.array([0.0, 0.0, 1.0]))

if __name__ == '__main__':
    unittest.main()
