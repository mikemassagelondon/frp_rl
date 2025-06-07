"""
Tests for the visualization utilities in the LS-MDP package.
"""

import numpy as np
import pytest
import matplotlib.pyplot as plt
from lsmdp.visualization import (
    visualize_grid_value,
    visualize_grid_policy,
    visualize_tree_value,
    visualize_tree_policy,
    visualize_results
)

# Check if pygraphviz is available
#try:
#    import pygraphviz
#    PYGRAPHVIZ_AVAILABLE = True
#except ImportError:
PYGRAPHVIZ_AVAILABLE = False


@pytest.fixture
def simple_grid_data():
    """Create simple grid data for testing visualization."""
    # 3x3 grid (9 states)
    n = 9
    grid_shape = (3, 3)
    
    # Value function (decreasing from top-left to bottom-right)
    V = np.array([5.0, 4.0, 3.0, 4.0, 3.0, 2.0, 3.0, 2.0, 0.0])
    
    # Desirability function (increasing from top-left to bottom-right)
    z = np.exp(-V)
    
    # Policy (always move toward bottom-right)
    policy = {
        0: {1: 0.5, 3: 0.5},
        1: {2: 0.5, 4: 0.5},
        2: {5: 1.0},
        3: {4: 0.5, 6: 0.5},
        4: {5: 0.5, 7: 0.5},
        5: {8: 1.0},
        6: {7: 1.0},
        7: {8: 1.0},
        8: {8: 1.0}  # Goal state (absorbing)
    }
    
    return V, z, policy, grid_shape


@pytest.fixture
def simple_tree_data():
    """Create simple tree data for testing visualization."""
    # Binary tree with 7 nodes (depth 2)
    n = 7
    depth = 2
    
    # Value function (decreasing from root to leaves)
    V = np.array([3.0, 2.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    
    # Desirability function (increasing from root to leaves)
    z = np.exp(-V)
    
    # Policy (equal probability to children, leaves are absorbing)
    policy = {
        0: {1: 0.5, 2: 0.5},
        1: {3: 0.5, 4: 0.5},
        2: {5: 0.5, 6: 0.5},
        3: {3: 1.0},
        4: {4: 1.0},
        5: {5: 1.0},
        6: {6: 1.0}
    }
    
    return V, z, policy, depth


def test_visualize_grid_value(simple_grid_data):
    """Test that visualize_grid_value runs without errors."""
    V, _, _, grid_shape = simple_grid_data
    
    # Test with default parameters
    fig = visualize_grid_value(V, grid_shape)
    assert fig is not None, "visualize_grid_value should return a figure"
    plt.close(fig)
    
    # Test with custom parameters
    fig = visualize_grid_value(V, grid_shape, cmap='hot', title='Test Value Function')
    assert fig is not None, "visualize_grid_value should return a figure"
    plt.close(fig)
    
    # Test with provided axes
    fig, ax = plt.subplots()
    fig_result = visualize_grid_value(V, grid_shape, ax=ax)
    assert fig_result is fig, "visualize_grid_value should return the provided figure"
    plt.close(fig)


def test_visualize_grid_policy(simple_grid_data):
    """Test that visualize_grid_policy runs without errors."""
    _, _, policy, grid_shape = simple_grid_data
    
    # Test with default parameters
    fig = visualize_grid_policy(policy, grid_shape)
    assert fig is not None, "visualize_grid_policy should return a figure"
    plt.close(fig)
    
    # Test with custom parameters
    fig = visualize_grid_policy(policy, grid_shape, title='Test Policy')
    assert fig is not None, "visualize_grid_policy should return a figure"
    plt.close(fig)
    
    # Test with provided axes
    fig, ax = plt.subplots()
    fig_result = visualize_grid_policy(policy, grid_shape, ax=ax)
    assert fig_result is fig, "visualize_grid_policy should return the provided figure"
    plt.close(fig)


@pytest.mark.skipif(not PYGRAPHVIZ_AVAILABLE, reason="pygraphviz not available")
def test_visualize_tree_value(simple_tree_data):
    """Test that visualize_tree_value runs without errors."""
    V, _, _, depth = simple_tree_data
    
    # Test with depth parameter
    fig = visualize_tree_value(V, depth=depth)
    assert fig is not None, "visualize_tree_value should return a figure"
    plt.close(fig)
    
    # Test with n_nodes parameter
    n_nodes = len(V)
    fig = visualize_tree_value(V, n_nodes=n_nodes)
    assert fig is not None, "visualize_tree_value should return a figure"
    plt.close(fig)
    
    # Test with custom parameters
    fig = visualize_tree_value(V, depth=depth, title='Test Tree Value')
    assert fig is not None, "visualize_tree_value should return a figure"
    plt.close(fig)
    
    # Test with provided axes
    fig, ax = plt.subplots()
    fig_result = visualize_tree_value(V, depth=depth, ax=ax)
    assert fig_result is fig, "visualize_tree_value should return the provided figure"
    plt.close(fig)


@pytest.mark.skipif(not PYGRAPHVIZ_AVAILABLE, reason="pygraphviz not available")
def test_visualize_tree_policy(simple_tree_data):
    """Test that visualize_tree_policy runs without errors."""
    _, _, policy, depth = simple_tree_data
    
    # Test with depth parameter
    fig = visualize_tree_policy(policy, depth=depth)
    assert fig is not None, "visualize_tree_policy should return a figure"
    plt.close(fig)
    
    # Test with n_nodes parameter
    n_nodes = len(policy)
    fig = visualize_tree_policy(policy, n_nodes=n_nodes)
    assert fig is not None, "visualize_tree_policy should return a figure"
    plt.close(fig)
    
    # Test with custom parameters
    fig = visualize_tree_policy(policy, depth=depth, title='Test Tree Policy')
    assert fig is not None, "visualize_tree_policy should return a figure"
    plt.close(fig)
    
    # Test with provided axes
    fig, ax = plt.subplots()
    fig_result = visualize_tree_policy(policy, depth=depth, ax=ax)
    assert fig_result is fig, "visualize_tree_policy should return the provided figure"
    plt.close(fig)


def test_visualize_results_grid(simple_grid_data):
    """Test that visualize_results runs without errors for grid data."""
    V, z, policy, grid_shape = simple_grid_data
    
    # Test with grid structure
    fig = visualize_results(z, V, policy, "lattice", grid_shape=grid_shape)
    assert fig is not None, "visualize_results should return a figure"
    plt.close(fig)
    
    # Test with "grid" structure type
    fig = visualize_results(z, V, policy, "grid", grid_shape=grid_shape)
    assert fig is not None, "visualize_results should return a figure"
    plt.close(fig)
    
    # Test with inferred grid shape
    fig = visualize_results(z, V, policy, "lattice")
    assert fig is not None, "visualize_results should return a figure"
    plt.close(fig)


@pytest.mark.skipif(not PYGRAPHVIZ_AVAILABLE, reason="pygraphviz not available")
def test_visualize_results_tree(simple_tree_data):
    """Test that visualize_results runs without errors for tree data."""
    V, z, policy, depth = simple_tree_data
    
    # Test with tree structure and depth
    fig = visualize_results(z, V, policy, "tree", depth=depth)
    assert fig is not None, "visualize_results should return a figure"
    plt.close(fig)
    
    # Test with "binary_tree" structure type
    fig = visualize_results(z, V, policy, "binary_tree", depth=depth)
    assert fig is not None, "visualize_results should return a figure"
    plt.close(fig)
    
    # Test with n_nodes
    n_nodes = len(V)
    fig = visualize_results(z, V, policy, "tree", n_nodes=n_nodes)
    assert fig is not None, "visualize_results should return a figure"
    plt.close(fig)
    
    # Test with inferred n_nodes
    fig = visualize_results(z, V, policy, "tree")
    assert fig is not None, "visualize_results should return a figure"
    plt.close(fig)


def test_visualize_results_invalid_structure():
    """Test that visualize_results raises an error for an invalid structure type."""
    V = np.zeros(10)
    z = np.ones(10)
    policy = {i: {i: 1.0} for i in range(10)}
    
    with pytest.raises(ValueError):
        visualize_results(z, V, policy, "invalid_structure")
