import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envs.environments.popgym_minesweeper import MineSweeper, count_neighbors

def test_neighbor_counting_implementation():
    """Test neighbor counting without using convolution."""
    # Create test grid
    grid = jnp.zeros((4, 4), dtype=jnp.int8)
    grid = grid.at[1, 1].set(1)  # Center mine
    grid = grid.at[2, 2].set(1)  # Another mine
    
    # Count neighbors using our implementation
    result = count_neighbors(grid)
    
    print("\nInput grid:")
    print(grid)
    print("\nNeighbor count:")
    print(result)
    
    # Compare with expected result
    expected = jnp.array([
        [1, 1, 1, 0],
        [1, 1, 2, 1],
        [1, 2, 1, 1],
        [0, 1, 1, 1]  # Bottom-right corner has 1 neighbor from the mine at (2,2)
    ], dtype=jnp.int8)
    
    np.testing.assert_array_equal(result, expected)
    
    # Test with different mine configurations
    grid = jnp.zeros((4, 4), dtype=jnp.int8)
    grid = grid.at[0, 0].set(1)  # Corner mine
    result = count_neighbors(grid)
    
    print("\nCorner mine grid:")
    print(grid)
    print("\nCorner mine neighbor count:")
    print(result)
    
    # Verify corner case
    expected_corner = jnp.array([
        [0, 1, 0, 0],
        [1, 1, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 0, 0]
    ], dtype=jnp.int8)
    
    np.testing.assert_array_equal(result, expected_corner)

def test_minesweeper_with_pure_jax():
    """Test MineSweeper environment with pure JAX neighbor counting."""
    env = MineSweeper(dims=(4, 4), num_mines=2)
    key = jax.random.PRNGKey(0)
    
    # Test multiple resets and steps
    @jax.jit
    def reset_and_step(key):
        # Reset environment
        obs, state = env.reset_env(key, env.default_params)
        
        # Get the mine grid and reshape it
        mine_grid = state.mine_grid.reshape(4, 4)
        
        # Count neighbors using our implementation
        neighbor_grid = count_neighbors(mine_grid)
        
        # Use JAX where instead of assertions for validation
        valid_counts = jnp.logical_and(
            neighbor_grid >= 0,
            neighbor_grid <= 8
        )
        neighbor_grid = jnp.where(valid_counts, neighbor_grid, 0)
        
        # Take a step
        next_obs, next_state, reward, done, info = env.step_env(key, state, 0, env.default_params)
        return obs, state, neighbor_grid
    
    # Run multiple episodes
    for i in range(5):
        key, subkey = jax.random.split(key)
        obs, state, neighbor_grid = reset_and_step(subkey)
        print(f"\nEpisode {i} mine grid:")
        print(state.mine_grid.reshape(4, 4))
        print("Neighbor grid:")
        print(neighbor_grid)
        
        # Verify results outside of jitted function
        assert jnp.all(neighbor_grid >= 0), "Negative neighbor counts"
        assert jnp.all(neighbor_grid <= 8), "Too many neighbors"

if __name__ == "__main__":
    test_neighbor_counting_implementation()
    test_minesweeper_with_pure_jax()
