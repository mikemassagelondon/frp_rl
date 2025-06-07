import unittest
import jax
import jax.numpy as jnp
import numpy as np
from envs.environments.meta_cartpole import NoisyStatelessMetaCartPole
from envs.meta_environment import create_meta_environment

class TestMetaCartpole(unittest.TestCase):
    def setUp(self):
        # Common parameters for both implementations
        self.meta_kwargs = {
            'meta_depth': 1,
            'meta_dim': 32,
            'meta_max_depth': 2,
            'meta_with_adjoint': False,
            'rng': jax.random.PRNGKey(42)
        }
        
        # Create both environment instances
        self.direct_env = NoisyStatelessMetaCartPole(**self.meta_kwargs)
        self.meta_env = create_meta_environment("cartpole", 
                                              env_kwargs={'max_steps_in_episode': 200, 'noise_sigma': 0.0}, 
                                              meta_kwargs=self.meta_kwargs)

    def test_spaces_match(self):
        """Test if observation and action spaces match between implementations"""
        # Test observation space
        direct_obs_space = self.direct_env.observation_space(self.direct_env.default_params)
        meta_obs_space = self.meta_env.observation_space(self.meta_env.default_params)
        
        self.assertEqual(direct_obs_space.shape, meta_obs_space.shape)
        self.assertEqual(direct_obs_space.dtype, meta_obs_space.dtype)
        np.testing.assert_array_equal(direct_obs_space.low, meta_obs_space.low)
        np.testing.assert_array_equal(direct_obs_space.high, meta_obs_space.high)

        # Test action space
        self.assertEqual(self.direct_env.num_actions, self.meta_env.num_actions)
        
    def test_reset_behavior(self):
        """Test if reset behavior matches between implementations"""
        key = jax.random.PRNGKey(0)
        
        # Reset both environments
        direct_obs, direct_state = self.direct_env.reset_env(key, self.direct_env.default_params)
        meta_obs, meta_state = self.meta_env.reset_env(key, self.meta_env.default_params)
        
        # Check observation shapes and types
        self.assertEqual(direct_obs.shape, meta_obs.shape)
        self.assertEqual(direct_obs.dtype, meta_obs.dtype)
        
        # Check state attributes
        self.assertEqual(direct_state.trial_num, meta_state.trial_num)
        self.assertEqual(direct_state.total_steps, meta_state.total_steps)
        
    def test_step_behavior(self):
        """Test if step behavior matches between implementations"""
        key = jax.random.PRNGKey(0)
        key_reset, key_step = jax.random.split(key)
        
        # Reset both environments
        direct_obs, direct_state = self.direct_env.reset_env(key_reset, self.direct_env.default_params)
        meta_obs, meta_state = self.meta_env.reset_env(key_reset, self.meta_env.default_params)
        
        # Take a step in both environments
        action = 0  # Choose action 0 for test
        direct_next_obs, direct_next_state, direct_reward, direct_done, direct_info = \
            self.direct_env.step_env(key_step, direct_state, action, self.direct_env.default_params)
            
        meta_next_obs, meta_next_state, meta_reward, meta_done, meta_info = \
            self.meta_env.step_env(key_step, meta_state, action, self.meta_env.default_params)
        
        # Check observation shapes and types
        self.assertEqual(direct_next_obs.shape, meta_next_obs.shape)
        self.assertEqual(direct_next_obs.dtype, meta_next_obs.dtype)
        
        # Check state updates
        self.assertEqual(direct_next_state.trial_num, meta_next_state.trial_num)
        self.assertEqual(direct_next_state.total_steps, meta_next_state.total_steps)
        
        # Check reward and done flag
        self.assertEqual(direct_reward, meta_reward)
        self.assertEqual(direct_done, meta_done)

    def test_episode_completion(self):
        """Test if episode completion behavior matches between implementations"""
        key = jax.random.PRNGKey(0)
        
        # Run through a full episode for both implementations
        direct_state = self.direct_env.reset_env(key, self.direct_env.default_params)[1]
        meta_state = self.meta_env.reset_env(key, self.meta_env.default_params)[1]
        
        # Run until either environment signals done
        max_steps = 1000  # Safety limit
        step = 0
        direct_done = False
        meta_done = False
        
        while not (direct_done or meta_done) and step < max_steps:
            key, subkey = jax.random.split(key)
            action = 0  # Fixed action for testing
            
            _, direct_state, _, direct_done, _ = \
                self.direct_env.step_env(subkey, direct_state, action, self.direct_env.default_params)
            _, meta_state, _, meta_done, _ = \
                self.meta_env.step_env(subkey, meta_state, action, self.meta_env.default_params)
            
            # Check if both environments end episodes at the same time
            self.assertEqual(direct_done, meta_done)
            
            step += 1
        
        # Ensure we didn't hit the safety limit
        self.assertLess(step, max_steps, "Episode did not complete within safety limit")

    def test_reward_range(self):
        """Test if rewards are properly scaled"""
        key = jax.random.PRNGKey(0)
        
        # Test both implementations
        direct_state = self.direct_env.reset_env(key, self.direct_env.default_params)[1]
        meta_state = self.meta_env.reset_env(key, self.meta_env.default_params)[1]
        
        direct_rewards = []
        meta_rewards = []
        
        for _ in range(100):  # Collect some sample rewards
            key, subkey = jax.random.split(key)
            action = jax.random.randint(subkey, shape=(), minval=0, maxval=2)  # Random action (0 or 1)
            
            _, direct_state, direct_reward, direct_done, _ = \
                self.direct_env.step_env(subkey, direct_state, action, self.direct_env.default_params)
            _, meta_state, meta_reward, meta_done, _ = \
                self.meta_env.step_env(subkey, meta_state, action, self.meta_env.default_params)
            
            direct_rewards.append(direct_reward)
            meta_rewards.append(meta_reward)
            
            if direct_done or meta_done:
                direct_state = self.direct_env.reset_env(key, self.direct_env.default_params)[1]
                meta_state = self.meta_env.reset_env(key, self.meta_env.default_params)[1]
        
        direct_rewards = jnp.array(direct_rewards)
        meta_rewards = jnp.array(meta_rewards)
        
        # Check if rewards are properly bounded
        self.assertTrue(jnp.all(direct_rewards >= -1.0))
        self.assertTrue(jnp.all(direct_rewards <= 1.0))
        self.assertTrue(jnp.all(meta_rewards >= -1.0))
        self.assertTrue(jnp.all(meta_rewards <= 1.0))
        
        # Check if rewards match between implementations
        np.testing.assert_array_almost_equal(direct_rewards, meta_rewards)

    def test_trial_boundaries(self):
        """Test if trial boundaries are handled correctly"""
        key = jax.random.PRNGKey(0)
        
        # Test both implementations
        direct_state = self.direct_env.reset_env(key, self.direct_env.default_params)[1]
        meta_state = self.meta_env.reset_env(key, self.meta_env.default_params)[1]
        
        # Run until either environment signals done
        max_steps = 1000  # Safety limit
        step = 0
        direct_trial_num = 0
        meta_trial_num = 0
        
        while step < max_steps:
            key, subkey = jax.random.split(key)
            action = 1  # Use right action
            
            _, direct_state, _, direct_done, _ = \
                self.direct_env.step_env(subkey, direct_state, action, self.direct_env.default_params)
            _, meta_state, _, meta_done, _ = \
                self.meta_env.step_env(subkey, meta_state, action, self.meta_env.default_params)
            
            # Check trial boundaries
            if direct_state.trial_num > direct_trial_num:
                direct_trial_num = direct_state.trial_num
                self.assertLessEqual(direct_trial_num, self.meta_kwargs.get('num_trials_per_episode', 16))
            
            if meta_state.trial_num > meta_trial_num:
                meta_trial_num = meta_state.trial_num
                self.assertLessEqual(meta_trial_num, self.meta_kwargs.get('num_trials_per_episode', 16))
            
            # Check if trial numbers match between implementations
            self.assertEqual(direct_state.trial_num, meta_state.trial_num)
            
            if direct_done or meta_done:
                # Both should complete at the same time
                self.assertEqual(direct_done, meta_done)
                # Episode should end after specified number of trials
                self.assertEqual(direct_state.trial_num, self.meta_kwargs.get('num_trials_per_episode', 16))
                self.assertEqual(meta_state.trial_num, self.meta_kwargs.get('num_trials_per_episode', 16))
                break
            
            step += 1
        
        # Ensure we didn't hit the safety limit
        self.assertLess(step, max_steps, "Episode did not complete within safety limit")

    def test_auto_reset(self):
        """Test if environment auto-resets correctly when done"""
        key = jax.random.PRNGKey(0)
        
        # Test both implementations
        direct_state = self.direct_env.reset_env(key, self.direct_env.default_params)[1]
        meta_state = self.meta_env.reset_env(key, self.meta_env.default_params)[1]
        
        # Run until we get a done signal
        max_steps = 1000
        step = 0
        done = False
        last_direct_obs = None
        last_direct_state = None
        last_meta_obs = None
        last_meta_state = None
        
        while not done and step < max_steps:
            key, subkey = jax.random.split(key)
            action = 0  # Fixed action for testing
            
            direct_obs, direct_state, _, direct_done, _ = \
                self.direct_env.step(subkey, direct_state, action, self.direct_env.default_params)
            meta_obs, meta_state, _, meta_done, _ = \
                self.meta_env.step(subkey, meta_state, action, self.meta_env.default_params)
            
            # Both implementations should reach done state together
            self.assertEqual(direct_done, meta_done)
            done = direct_done or meta_done
            
            if done:
                last_direct_obs = direct_obs
                last_direct_state = direct_state
                last_meta_obs = meta_obs
                last_meta_state = meta_state
            
            step += 1
        
        # Verify we got a done signal
        self.assertTrue(done, "Environment never reached done state")
        
        # After done, the next step should call reset_env
        key, subkey = jax.random.split(key)
        
        next_direct_obs, next_direct_state = self.direct_env.reset_env(subkey, self.direct_env.default_params)
        next_meta_obs, next_meta_state = self.meta_env.reset_env(subkey, self.meta_env.default_params)
        
        # Verify both implementations reset correctly
        # - States should be reset (trial_num back to 0)
        self.assertEqual(next_direct_state.trial_num, 0)
        self.assertEqual(next_meta_state.trial_num, 0)
        
        # - Last 3 values of obs should indicate reset (action=0, done=0, reset=1)
        np.testing.assert_array_equal(next_direct_obs[-3:], jnp.array([0.0, 0.0, 1.0]))
        np.testing.assert_array_equal(next_meta_obs[-3:], jnp.array([0.0, 0.0, 1.0]))

if __name__ == '__main__':
    unittest.main()
