import unittest
import jax
import jax.numpy as jnp
import numpy as np
from envs.environments.popgym_higherlower import HigherLowerEasy, HigherLowerMedium, HigherLowerHard

class TestHigherLower(unittest.TestCase):
    def setUp(self):
        """Initialize environments for all difficulty levels"""
        self.easy_env = HigherLowerEasy()
        self.medium_env = HigherLowerMedium()
        self.hard_env = HigherLowerHard()
        self.key = jax.random.PRNGKey(0)

    def test_initialization(self):
        """Test environment initialization and configuration"""
        # Test deck sizes
        self.assertEqual(self.easy_env.num_decks, 1)
        self.assertEqual(self.medium_env.num_decks, 2)
        self.assertEqual(self.hard_env.num_decks, 3)

        # Test number of ranks
        self.assertEqual(self.easy_env.num_ranks, 13)
        self.assertEqual(self.medium_env.num_ranks, 13)
        self.assertEqual(self.hard_env.num_ranks, 13)

        # Test deck sizes
        self.assertEqual(self.easy_env.decksize, 52)
        self.assertEqual(self.medium_env.decksize, 52)
        self.assertEqual(self.hard_env.decksize, 52)

    def test_reset_behavior(self):
        """Test environment reset behavior"""
        for env in [self.easy_env, self.medium_env, self.hard_env]:
            key, subkey = jax.random.split(self.key)
            obs, state = env.reset_env(subkey, env.default_params)

            # Test observation shape and type
            self.assertEqual(obs.shape, (env.num_ranks,))
            self.assertEqual(obs.dtype, jnp.float32)

            # Test one-hot encoding
            self.assertEqual(jnp.sum(obs), 1.0)  # Only one 1, rest 0s
            self.assertTrue(jnp.all((obs == 0) | (obs == 1)))  # Only contains 0s and 1s

            # Test initial state
            self.assertEqual(state.timestep, 0)
            self.assertEqual(state.cards.shape, (env.decksize * env.num_decks,))

            # Test card distribution
            unique_cards = jnp.unique(state.cards)
            self.assertTrue(jnp.all(unique_cards >= 0))
            self.assertTrue(jnp.all(unique_cards < env.num_ranks))

    def test_step_mechanics(self):
        """Test step behavior and reward calculation"""
        for env in [self.easy_env, self.medium_env, self.hard_env]:
            key, subkey = jax.random.split(self.key)
            obs, state = env.reset_env(subkey, env.default_params)

            # Create a sequence that will reach the termination condition
            num_cards = env.decksize * env.num_decks
            test_sequence = jnp.array([0, 1, 1, 2] + [0] * (num_cards - 4), dtype=jnp.int32)
            state = state.replace(cards=test_sequence)

            # Test correct guess (0->1, guess higher)
            obs, state, reward, done, _ = env.step_env(subkey, state, 0, env.default_params)
            self.assertGreater(reward, 0)  # Should get positive reward
            self.assertEqual(state.timestep, 1)
            self.assertFalse(done)

            # Test equal cards (1->1, guess higher)
            obs, state, reward, done, _ = env.step_env(subkey, state, 0, env.default_params)
            self.assertEqual(reward, 0)  # Should get zero reward
            self.assertEqual(state.timestep, 2)
            self.assertFalse(done)

            # Test incorrect guess (1->2, guess lower)
            obs, state, reward, done, _ = env.step_env(subkey, state, 1, env.default_params)
            self.assertLess(reward, 0)  # Should get negative reward
            self.assertEqual(state.timestep, 3)
            self.assertFalse(done)  # Not done yet, need to reach num_cards

            # Step until the end to verify termination
            while not done:
                obs, state, reward, done, _ = env.step_env(subkey, state, 0, env.default_params)
            self.assertEqual(state.timestep, num_cards)
            self.assertTrue(done)

    def test_reward_scaling(self):
        """Test reward scaling based on deck size"""
        envs = {
            1: self.easy_env,
            2: self.medium_env,
            3: self.hard_env
        }

        for num_decks, env in envs.items():
            key, subkey = jax.random.split(self.key)
            obs, state = env.reset_env(subkey, env.default_params)

            # Force a known sequence for testing
            state = state.replace(cards=jnp.array([0, 1]))  # 0->1
            
            # Test correct guess (higher)
            _, _, reward, _, _ = env.step_env(subkey, state, 0, env.default_params)
            expected_reward = 1.0 / (env.decksize * num_decks)
            self.assertAlmostEqual(reward, expected_reward)

    def test_observation_encoding(self):
        """Test observation space and encoding"""
        for env in [self.easy_env, self.medium_env, self.hard_env]:
            # Test observation space
            obs_space = env.observation_space(env.default_params)
            self.assertEqual(obs_space.shape, (env.num_ranks,))
            self.assertEqual(obs_space.dtype, jnp.float32)
            np.testing.assert_array_equal(obs_space.low, jnp.zeros((env.num_ranks,)))
            np.testing.assert_array_equal(obs_space.high, jnp.ones((env.num_ranks,)))

            # Test observation encoding
            key, subkey = jax.random.split(self.key)
            obs, state = env.reset_env(subkey, env.default_params)
            
            # Verify one-hot encoding
            self.assertEqual(jnp.sum(obs), 1.0)
            self.assertTrue(jnp.all((obs == 0) | (obs == 1)))
            self.assertEqual(obs.shape, (env.num_ranks,))

    def test_episode_completion(self):
        """Test episode completion and termination"""
        for env in [self.easy_env, self.medium_env, self.hard_env]:
            key, subkey = jax.random.split(self.key)
            obs, state = env.reset_env(subkey, env.default_params)
            
            num_steps = 0
            done = False
            while not done:
                key, subkey = jax.random.split(key)
                action = jax.random.randint(subkey, shape=(), minval=0, maxval=2)
                obs, state, reward, done, _ = env.step_env(subkey, state, action, env.default_params)
                num_steps += 1

            # Verify episode length matches deck size
            # We can take steps until timestep equals num_cards
            expected_steps = env.decksize * env.num_decks
            self.assertEqual(num_steps, expected_steps)
            self.assertEqual(state.timestep, expected_steps)
            self.assertTrue(done)  # Verify we're done at the end

    def test_action_space(self):
        """Test action space configuration"""
        for env in [self.easy_env, self.medium_env, self.hard_env]:
            # Test action space
            action_space = env.action_space()
            self.assertEqual(action_space.n, 2)
            self.assertEqual(env.num_actions, 2)

            # Test both valid actions
            key, subkey = jax.random.split(self.key)
            obs, state = env.reset_env(subkey, env.default_params)
            
            for action in [0, 1]:
                _, _, reward, _, _ = env.step_env(subkey, state, action, env.default_params)
                self.assertTrue(isinstance(reward, float) or isinstance(reward, jnp.ndarray))

    def test_card_distribution(self):
        """Test card distribution and shuffling"""
        for env in [self.easy_env, self.medium_env, self.hard_env]:
            key, subkey = jax.random.split(self.key)
            obs, state = env.reset_env(subkey, env.default_params)

            # Test card count
            self.assertEqual(len(state.cards), env.decksize * env.num_decks)

            # Test rank distribution
            unique, counts = jnp.unique(state.cards, return_counts=True)
            self.assertEqual(len(unique), env.num_ranks)
            self.assertTrue(jnp.all(counts == 4 * env.num_decks))  # 4 cards per rank per deck

            # Test shuffling
            key, subkey = jax.random.split(key)
            _, state2 = env.reset_env(subkey, env.default_params)
            self.assertFalse(jnp.array_equal(state.cards, state2.cards))

if __name__ == '__main__':
    unittest.main()
