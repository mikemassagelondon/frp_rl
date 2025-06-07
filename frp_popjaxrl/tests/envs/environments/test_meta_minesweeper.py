import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envs.environments.popgym_minesweeper import MineSweeper, count_neighbors

def test_neighbor_counting_center_mine():
    """Test neighbor counting with a single mine in the center."""
    #env = MineSweeper(dims=(5, 5), num_mines=1)
    #key = jax.random.PRNGKey(0)
    
    # Create a grid with one mine in the center
    hidden_grid = jnp.zeros((5, 5), dtype=jnp.int8)
    hidden_grid = hidden_grid.at[2, 2].set(1)
    
    # Create kernel that excludes center cell
    kernel = jnp.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=jnp.int8)
    # Calculate neighbors using the same convolution as the environment
    neighbor_grid = jax.scipy.signal.convolve2d(hidden_grid, kernel, mode="same")
    # Expected: all 8 surrounding cells should have count=1, center should be 0
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 1, 0, 1, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.int8)
    
    np.testing.assert_array_equal(neighbor_grid, expected)

    np.testing.assert_array_equal(count_neighbors(hidden_grid), expected)

def test_neighbor_counting_corner_mine():
    """Test neighbor counting with a mine in the corner."""
    #env = MineSweeper(dims=(5, 5), num_mines=1)
    #key = jax.random.PRNGKey(0)
    
    # Create a grid with one mine in the top-left corner
    hidden_grid = jnp.zeros((5, 5), dtype=jnp.int8)
    hidden_grid = hidden_grid.at[0, 0].set(1)
    
    # Create kernel that excludes center cell
    kernel = jnp.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=jnp.int8)
    # Calculate neighbors using the same convolution as the environment
    neighbor_grid = jax.scipy.signal.convolve2d(hidden_grid, kernel, mode="same")
    
    # Expected: 3 adjacent cells should have count=1, corner should be 0
    expected = np.array([
        [0, 1, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.int8)
    
    np.testing.assert_array_equal(neighbor_grid, expected)
    np.testing.assert_array_equal(count_neighbors(hidden_grid), expected)

def test_neighbor_counting_adjacent_mines():
    """Test neighbor counting with adjacent mines."""
    env = MineSweeper(dims=(5, 5), num_mines=2)
    key = jax.random.PRNGKey(0)
    
    # Create a grid with two adjacent mines
    hidden_grid = jnp.zeros((5, 5), dtype=jnp.int8)
    hidden_grid = hidden_grid.at[2, 2].set(1)  # Center mine
    hidden_grid = hidden_grid.at[2, 3].set(1)  # Adjacent mine
    
    # Create kernel that excludes center cell
    kernel = jnp.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=jnp.int8)
    # Calculate neighbors using the same convolution as the environment
    neighbor_grid = jax.scipy.signal.convolve2d(hidden_grid, kernel, mode="same")
    
    # Expected: overlapping neighbor counts, mine positions should be 0
    expected = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 2, 2, 1],
        [0, 1, 1, 1, 1],
        [0, 1, 2, 2, 1],
        [0, 0, 0, 0, 0]
    ], dtype=np.int8)
    np.testing.assert_array_equal(neighbor_grid, expected)
    np.testing.assert_array_equal(count_neighbors(hidden_grid), expected)

def test_reset_env_mine_placement():
    """Test that reset_env places the correct number of mines."""
    env = MineSweeper(dims=(4, 4), num_mines=3)
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    obs, state = env.reset_env(key, env.default_params)
    
    # Check mine count
    mine_count = jnp.sum(state.mine_grid == 1)
    assert mine_count == 3, f"Expected 3 mines, got {mine_count}"
    
    # Check mine grid shape after flattening
    assert state.mine_grid.shape == (16,), f"Expected shape (16,), got {state.mine_grid.shape}"
    
    # Check neighbor grid shape after flattening
    assert state.neighbor_grid.shape == (16,), f"Expected shape (16,), got {state.neighbor_grid.shape}"

def test_convolution_kernel():
    """Test that the convolution kernel correctly counts neighbors without counting the mine itself."""
    # Create a simple 3x3 grid with a mine in the center
    grid = jnp.zeros((3, 3), dtype=jnp.int8)
    grid = grid.at[1, 1].set(1)
    
    # Create kernel that excludes center cell
    kernel = jnp.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=jnp.int8)
    
    # Perform convolution
    result = jax.scipy.signal.convolve2d(grid, kernel, mode="same")
    
    # Expected result: all surrounding cells should have count=1, center should be 0
    expected = jnp.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]], dtype=jnp.int8)
    
    np.testing.assert_array_equal(result, expected)

def test_neighbor_grid_dtype():
    """Test that neighbor grid maintains int8 dtype throughout computation."""
    env = MineSweeper(dims=(4, 4), num_mines=2)
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    obs, state = env.reset_env(key, env.default_params)
    
    # Check dtypes
    assert state.mine_grid.dtype == jnp.int8, f"Mine grid dtype is {state.mine_grid.dtype}, expected int8"
    assert state.neighbor_grid.dtype == jnp.int8, f"Neighbor grid dtype is {state.neighbor_grid.dtype}, expected int8"

def test_neighbor_counting_implementation():
    """Test that the actual implementation correctly counts neighbors without counting the mine itself."""
    env = MineSweeper(dims=(5, 5), num_mines=1)
    key = jax.random.PRNGKey(0)
    
    # Reset environment with a fixed seed that places a mine
    obs, state = env.reset_env(key, env.default_params)
    
    # Reshape grids back to 2D for easier comparison
    mine_grid_2d = state.mine_grid.reshape(5, 5)
    neighbor_grid_2d = state.neighbor_grid.reshape(5, 5)
    
    # Find mine location
    mine_location = jnp.where(mine_grid_2d == 1)
    mine_x, mine_y = mine_location[0][0], mine_location[1][0]
    
    # The mine location itself should have neighbor count 0
    assert neighbor_grid_2d[mine_x, mine_y] == 0, f"Mine location should have neighbor count 0, got {neighbor_grid_2d[mine_x, mine_y]}"
    
    # All cells adjacent to the mine should have count 1
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx == 0 and dy == 0:  # Skip mine location
                continue
            x, y = mine_x + dx, mine_y + dy
            if 0 <= x < 5 and 0 <= y < 5:  # Check bounds
                assert neighbor_grid_2d[x, y] == 1, f"Adjacent cell at ({x},{y}) should have count 1, got {neighbor_grid_2d[x, y]}"
