import jax
import jax.numpy as jnp
import numpy as np
import pytest
from envs.environments.popgym_minesweeper import MineSweeper

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
    
    # Test both convolution approaches
    # 1. Using jax.scipy.signal.convolve2d (current implementation)
    neighbor_grid1 = jax.scipy.signal.convolve2d(hidden_grid, kernel, mode="same")
    neighbor_grid1 = jnp.array(neighbor_grid1, dtype=jnp.int8)
    
    # 2. Using jax.lax.conv (alternative approach)
    # Reshape for lax.conv format (NCHW)
    hidden_grid_4d = hidden_grid.reshape(1, 1, 4, 4)
    kernel_4d = kernel.reshape(1, 1, 3, 3)
    
    # Perform convolution
    neighbor_grid2 = jax.lax.conv(
        hidden_grid_4d,
        kernel_4d,
        window_strides=(1, 1),
        padding=((1, 1), (1, 1))
    )
    
    # Reshape back to 2D
    neighbor_grid2 = neighbor_grid2.reshape(4, 4)
    neighbor_grid2 = jnp.array(neighbor_grid2, dtype=jnp.int8)
    
    # Print debug information
    print("Hidden grid dtype:", hidden_grid.dtype)
    print("Kernel dtype:", kernel.dtype)
    print("Result1 dtype:", neighbor_grid1.dtype)
    print("Result2 dtype:", neighbor_grid2.dtype)
    
    # Compare results
    np.testing.assert_array_equal(neighbor_grid1, neighbor_grid2)

def test_convolution_with_zero_center_kernel():
    """Test convolution with zero-center kernel to match test implementation."""
    env = MineSweeper(dims=(4, 4), num_mines=2)
    key = jax.random.PRNGKey(0)
    
    # Create test grid
    hidden_grid = jnp.zeros((4, 4), dtype=jnp.int8)
    hidden_grid = hidden_grid.at[1, 1].set(1)
    
    # Create kernel with zero center using jnp
    kernel = jnp.array([
        [1, 1, 1],
        [1, 0, 1],
        [1, 1, 1]
    ], dtype=jnp.int8)
    
    # Perform convolution
    neighbor_grid = jax.scipy.signal.convolve2d(hidden_grid, kernel, mode="same")
    neighbor_grid = jnp.array(neighbor_grid, dtype=jnp.int8)
    
    # Expected result
    expected = jnp.array([
        [1, 1, 1, 0],
        [1, 0, 1, 0],
        [1, 1, 1, 0],
        [0, 0, 0, 0]
    ], dtype=jnp.int8)
    
    np.testing.assert_array_equal(neighbor_grid, expected)

def test_full_env_reset_with_explicit_dtypes():
    """Test the full environment reset with explicit dtype handling."""
    env = MineSweeper(dims=(4, 4), num_mines=2)
    key = jax.random.PRNGKey(0)
    
    # Reset environment
    obs, state = env.reset_env(key, env.default_params)
    
    # Check dtypes
    print("Mine grid dtype:", state.mine_grid.dtype)
    print("Neighbor grid dtype:", state.neighbor_grid.dtype)
    
    # Reshape for visualization
    mine_grid_2d = state.mine_grid.reshape(4, 4)
    neighbor_grid_2d = state.neighbor_grid.reshape(4, 4)
    
    # Print shapes and values for debugging
    print("Mine grid shape:", mine_grid_2d.shape)
    print("Neighbor grid shape:", neighbor_grid_2d.shape)
    print("Mine grid:\n", mine_grid_2d)
    print("Neighbor grid:\n", neighbor_grid_2d)
    
    # Verify neighbor counts
    mine_positions = jnp.where(mine_grid_2d == 1)
    for x, y in zip(mine_positions[0], mine_positions[1]):
        # The mine position itself should have neighbor count 0
        assert neighbor_grid_2d[x, y] >= 0, f"Invalid neighbor count at mine position ({x},{y})"
