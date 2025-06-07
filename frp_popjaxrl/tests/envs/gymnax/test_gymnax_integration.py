import unittest
import jax
import jax.numpy as jnp
from envs.registration import make

class TestGymnaxIntegration(unittest.TestCase):
    def setUp(self):
        self.rng = jax.random.PRNGKey(42)
    
    def test_direct_gymnax_environment(self):
        """Test creating and using a direct gymnax environment."""
        env_id = "GymnaxCartPole-v1"  # Note: This will be parsed as "CartPole-v1"
        env, env_params = make(env_id)
        
        # Test reset
        key, subkey = jax.random.split(self.rng)
        obs, state = env.reset(subkey, env_params)
        
        # Check observation shape
        self.assertEqual(obs.shape, (7,))  # CartPole has 4 observation dimensions + 3 flag dimension
        
        # Test step
        key, subkey = jax.random.split(key)
        action = 0  # CartPole has 2 actions (0 or 1)
        next_obs, next_state, reward, done, info = env.step(subkey, state, action, env_params)
        
        # Check observation shape after step
        self.assertEqual(next_obs.shape, (7,))
        
        # Check reward is a scalar
        self.assertTrue(jnp.isscalar(reward))
        
        # Check done is a boolean
        self.assertIn(done, [True, False])
    
    def test_meta_gymnax_environment(self):
        """Test creating and using a meta-gymnax environment."""
        env_id = "MetaGymnaxCartPole-v1"
        meta_kwargs = {
            'meta_depth': 1,
            'meta_dim': 32,
            'meta_max_depth': 2,
            'meta_with_adjoint': False,
            'meta_rng': jax.random.PRNGKey(42)
        }
        env, env_params = make(env_id, **meta_kwargs)
        # Test reset
        key, subkey = jax.random.split(self.rng)
        obs, state = env.reset_env(subkey, env_params)
        
        # Check observation shape (input_dim + 3 for action, done, reset flags)
        # because before obs param
        self.assertEqual(obs.shape, (35,))  # 32 + 3
        
        # Test step
        key, subkey = jax.random.split(key)
        action = 0  # CartPole has 2 actions (0 or 1)
        next_obs, next_state, reward, done, info = env.step_env(subkey, state, action, env_params)
        
        # Check observation shape after step
        self.assertEqual(next_obs.shape, (35,))
        
        # Check reward is a scalar
        self.assertTrue(jnp.isscalar(reward))
        
        # Check done is a boolean
        self.assertIn(done, [True, False])
    
    def test_meta_popgym_environment(self):
        """Test creating and using a meta-popgym environment."""
        env_id = "MetaCartpole"
        meta_kwargs = {
            'meta_depth': 1,
            'meta_dim': 32,
            'meta_max_depth': 2,
            'meta_with_adjoint': False,
            'meta_rng': jax.random.PRNGKey(42)
        }
        env, env_params = make(env_id, **meta_kwargs)
        
        # Test reset
        key, subkey = jax.random.split(self.rng)
        obs, state = env.reset_env(subkey, env_params)
        
        # Check observation shape (meta_dim + 3 for action, done, reset flags)
        self.assertEqual(obs.shape, (35,))  # 32 + 3
        
        # Test step
        key, subkey = jax.random.split(key)
        action = 0  # CartPole has 2 actions (0 or 1)
        next_obs, next_state, reward, done, info = env.step_env(subkey, state, action, env_params)
        
        # Check observation shape after step
        self.assertEqual(next_obs.shape, (35,))
        
        # Check reward is a scalar
        self.assertTrue(jnp.isscalar(reward))
        
        # Check done is a boolean
        self.assertIn(done, [True, False])

if __name__ == '__main__':
    unittest.main()
