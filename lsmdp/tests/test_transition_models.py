"""
Tests for the transition matrix models in the LS-MDP package.
"""

import numpy as np
import pytest
from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions


def test_lattice_transitions_shape():
    """Test that the lattice transition matrix has the correct shape."""
    n = 16
    P = create_lattice_transitions(n)
    assert P.shape == (n, n), f"Expected shape ({n}, {n}), got {P.shape}"


def test_lattice_transitions_row_sum():
    """Test that each row in the lattice transition matrix sums to 1."""
    n = 16
    P = create_lattice_transitions(n)
    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums should be 1.0, got {row_sums}"


def test_lattice_transitions_neighbors():
    """Test that transitions are only to neighboring states in the lattice."""
    n = 9  # 3x3 grid
    P = create_lattice_transitions(n)
    
    # Check a few specific states
    # Center state (4) should transition to states 1, 3, 5, 7 (up, left, right, down)
    center_transitions = np.where(P[4] > 0)[0]
    expected_neighbors = np.array([1, 3, 5, 7])
    assert np.all(np.isin(center_transitions, expected_neighbors)), \
        f"Center state should transition to {expected_neighbors}, got {center_transitions}"
    
    # Corner state (0) should transition to states 1 and 3 (right and down)
    corner_transitions = np.where(P[0] > 0)[0]
    expected_neighbors = np.array([1, 3])
    assert np.all(np.isin(corner_transitions, expected_neighbors)), \
        f"Corner state should transition to {expected_neighbors}, got {corner_transitions}"


def test_lattice_transitions_diagonal():
    """Test that diagonal transitions are included when allow_diagonal=True."""
    n = 9  # 3x3 grid
    P = create_lattice_transitions(n, allow_diagonal=True)
    
    # Center state (4) should transition to states 0, 1, 2, 3, 5, 6, 7, 8
    # (all surrounding states including diagonals)
    center_transitions = np.where(P[4] > 0)[0]
    expected_neighbors = np.array([0, 1, 2, 3, 5, 6, 7, 8])
    assert np.all(np.isin(center_transitions, expected_neighbors)), \
        f"Center state with diagonals should transition to {expected_neighbors}, got {center_transitions}"


def test_binary_tree_transitions_shape():
    """Test that the binary tree transition matrix has the correct shape."""
    depth = 2
    n_nodes = 7  # Complete binary tree of depth 2
    P = create_binary_tree_transitions(depth=depth)
    assert P.shape == (n_nodes, n_nodes), f"Expected shape ({n_nodes}, {n_nodes}), got {P.shape}"


def test_binary_tree_transitions_row_sum():
    """Test that each row in the binary tree transition matrix sums to 1."""
    depth = 2
    P = create_binary_tree_transitions(depth=depth)
    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums should be 1.0, got {row_sums}"


def test_binary_tree_transitions_structure():
    """Test that transitions follow the binary tree structure."""
    depth = 2
    P = create_binary_tree_transitions(depth=depth,bidirectional=False)
    
    # Root (0) should transition to children (1, 2)
    root_transitions = np.where(P[0] > 0)[0]
    expected_children = np.array([1, 2])
    assert np.all(np.isin(root_transitions, expected_children)), \
        f"Root should transition to {expected_children}, got {root_transitions}"
    
    # Node 1 should transition to children (3, 4) and its parent 0
    node1_transitions = np.where(P[1] > 0)[0]
    expected_children = np.array([3, 4])
    assert np.all(np.isin(node1_transitions, expected_children)), \
        f"Node 1 should transition to {expected_children}, got {node1_transitions}"
    
    # Leaf nodes (3, 4, 5, 6) should be absorbing (transition to themselves)
    for leaf in [3, 4, 5, 6]:
        leaf_transitions = np.where(P[leaf] > 0)[0]
        assert leaf_transitions[0] == leaf, f"Leaf {leaf} should be absorbing, got {leaf_transitions}"


def test_binary_tree_transitions_probabilities():
    """Test that transition probabilities are correct in the binary tree."""
    depth = 2
    P = create_binary_tree_transitions(depth=depth,bidirectional=False)
    
    # Root (0) should transition to children (1, 2) with probability 0.5 each
    assert np.isclose(P[0, 1], 0.5), f"P[0, 1] should be 0.5, got {P[0, 1]}"
    assert np.isclose(P[0, 2], 0.5), f"P[0, 2] should be 0.5, got {P[0, 2]}"
    
    # Leaf nodes (3, 4, 5, 6) should transition to themselves with probability 1.0
    for leaf in [3, 4, 5, 6]:
        assert np.isclose(P[leaf, leaf], 1.0), f"P[{leaf}, {leaf}] should be 1.0, got {P[leaf, leaf]}"


def test_binary_tree_transitions_n_nodes():
    """Test creating a binary tree with a specific number of nodes."""
    n_nodes = 5  # Not a complete binary tree
    P = create_binary_tree_transitions(n_nodes=n_nodes,bidirectional=False)
    assert P.shape == (n_nodes, n_nodes), f"Expected shape ({n_nodes}, {n_nodes}), got {P.shape}"
    
    # Root (0) should transition to children (1, 2) with probability 0.5 each
    assert np.isclose(P[0, 1], 0.5), f"P[0, 1] should be 0.5, got {P[0, 1]}"
    assert np.isclose(P[0, 2], 0.5), f"P[0, 2] should be 0.5, got {P[0, 2]}"
    
    # Node 1 should transition to child (3) with probability 0.5
    # and to child (4) with probability 0.5 if it exists, otherwise 0.0
    if 4 < n_nodes:
        assert np.isclose(P[1, 3], 0.5), f"P[1, 3] should be 0.5, got {P[1, 3]}"
        assert np.isclose(P[1, 4], 0.5), f"P[1, 4] should be 0.5, got {P[1, 4]}"
    else:
        assert np.isclose(P[1, 3], 1.0), f"P[1, 3] should be 1.0, got {P[1, 3]}"
    
    # Leaf nodes (3, 4 if it exists) should be absorbing
    assert np.isclose(P[3, 3], 1.0), f"P[3, 3] should be 1.0, got {P[3, 3]}"
    if 4 < n_nodes:
        assert np.isclose(P[4, 4], 1.0), f"P[4, 4] should be 1.0, got {P[4, 4]}"


def test_binary_tree_transitions_invalid_params():
    """Test that an error is raised when neither depth nor n_nodes is provided."""
    with pytest.raises(ValueError):
        create_binary_tree_transitions()


def test_binary_tree_transitions_bidirectional_shape():
    """Test that the bidirectional binary tree transition matrix has the correct shape."""
    depth = 2
    n_nodes = 7  # Complete binary tree of depth 2
    P = create_binary_tree_transitions(depth=depth, bidirectional=True)
    assert P.shape == (n_nodes, n_nodes), f"Expected shape ({n_nodes}, {n_nodes}), got {P.shape}"


def test_binary_tree_transitions_bidirectional_row_sum():
    """Test that each row in the bidirectional binary tree transition matrix sums to 1."""
    depth = 2
    P = create_binary_tree_transitions(depth=depth, bidirectional=True)
    row_sums = np.sum(P, axis=1)
    assert np.allclose(row_sums, 1.0), f"Row sums should be 1.0, got {row_sums}"


def test_binary_tree_transitions_bidirectional_structure():
    """Test that transitions follow the bidirectional binary tree structure."""
    depth = 2
    P = create_binary_tree_transitions(depth=depth, bidirectional=True)
    
    # Root (0) should transition to children (1, 2) with equal probability
    root_transitions = np.where(P[0] > 0)[0]
    expected_children = np.array([1, 2])
    assert np.all(np.isin(root_transitions, expected_children)), \
        f"Root should transition to {expected_children}, got {root_transitions}"
    
    # Node 1 should transition to parent (0) and children (3, 4) with equal probability
    node1_transitions = np.where(P[1] > 0)[0]
    expected_connections = np.array([0, 3, 4])
    assert np.all(np.isin(node1_transitions, expected_connections)), \
        f"Node 1 should transition to {expected_connections}, got {node1_transitions}"
    
    # Leaf nodes (3, 4, 5, 6) should transition to their parent with probability 1.0
    for leaf, parent in [(3, 1), (4, 1), (5, 2), (6, 2)]:
        leaf_transitions = np.where(P[leaf] > 0)[0]
        assert leaf_transitions[0] == parent, \
            f"Leaf {leaf} should transition to parent {parent}, got {leaf_transitions}"
        assert np.isclose(P[leaf, parent], 1.0), \
            f"P[{leaf}, {parent}] should be 1.0, got {P[leaf, parent]}"


def test_binary_tree_transitions_bidirectional_probabilities():
    """Test that transition probabilities are correct in the bidirectional binary tree."""
    depth = 2
    P = create_binary_tree_transitions(depth=depth, bidirectional=True)
    
    # Root (0) should transition to children (1, 2) with probability 0.5 each
    assert np.isclose(P[0, 1], 0.5), f"P[0, 1] should be 0.5, got {P[0, 1]}"
    assert np.isclose(P[0, 2], 0.5), f"P[0, 2] should be 0.5, got {P[0, 2]}"
    
    # Node 1 should transition to parent (0) and children (3, 4) with probability 1/3 each
    assert np.isclose(P[1, 0], 1/3), f"P[1, 0] should be 1/3, got {P[1, 0]}"
    assert np.isclose(P[1, 3], 1/3), f"P[1, 3] should be 1/3, got {P[1, 3]}"
    assert np.isclose(P[1, 4], 1/3), f"P[1, 4] should be 1/3, got {P[1, 4]}"
    
    # Leaf nodes (3, 4, 5, 6) should transition to their parent with probability 1.0
    for leaf, parent in [(3, 1), (4, 1), (5, 2), (6, 2)]:
        assert np.isclose(P[leaf, parent], 1.0), \
            f"P[{leaf}, {parent}] should be 1.0, got {P[leaf, parent]}"


def test_binary_tree_transitions_bidirectional_n_nodes():
    """Test creating a bidirectional binary tree with a specific number of nodes."""
    n_nodes = 5  # Not a complete binary tree
    P = create_binary_tree_transitions(n_nodes=n_nodes, bidirectional=True)
    assert P.shape == (n_nodes, n_nodes), f"Expected shape ({n_nodes}, {n_nodes}), got {P.shape}"
    
    # Root (0) should transition to children (1, 2) with probability 0.5 each
    assert np.isclose(P[0, 1], 0.5), f"P[0, 1] should be 0.5, got {P[0, 1]}"
    assert np.isclose(P[0, 2], 0.5), f"P[0, 2] should be 0.5, got {P[0, 2]}"
    
    # Node 1 should transition to parent (0) and children (3, 4 if it exists)
    # with equal probability (1/3 if both children exist, 1/2 if only one child exists)
    if 4 < n_nodes:  # Both children exist
        assert np.isclose(P[1, 0], 1/3), f"P[1, 0] should be 1/3, got {P[1, 0]}"
        assert np.isclose(P[1, 3], 1/3), f"P[1, 3] should be 1/3, got {P[1, 3]}"
        assert np.isclose(P[1, 4], 1/3), f"P[1, 4] should be 1/3, got {P[1, 4]}"
    else:  # Only one child exists
        assert np.isclose(P[1, 0], 0.5), f"P[1, 0] should be 0.5, got {P[1, 0]}"
        assert np.isclose(P[1, 3], 0.5), f"P[1, 3] should be 0.5, got {P[1, 3]}"
    
    # For node 2, we need to check if it has a child (node 4)
    # In a binary tree, node 4 is the right child of node 1, not the child of node 2
    # Node 2's children would be nodes 5 and 6, but with n_nodes=5, these don't exist
    # So node 2 should only transition to its parent (node 0) with probability 1.0
    assert np.isclose(P[2, 0], 1.0), f"P[2, 0] should be 1.0, got {P[2, 0]}"
    
    # Leaf nodes (3, 4 if it exists) should transition to their parent with probability 1.0
    assert np.isclose(P[3, 1], 1.0), f"P[3, 1] should be 1.0, got {P[3, 1]}"
    if 4 < n_nodes:
        assert np.isclose(P[4, 1], 1.0), f"P[4, 1] should be 1.0, got {P[4, 1]}"
