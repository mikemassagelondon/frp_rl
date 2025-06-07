"""
Transition matrix models for Linearly Solvable Markov Decision Processes (LS-MDPs).

This module provides functions to create transition probability matrices
for different state space structures, including finite lattices (grids)
and binary trees.
"""

import numpy as np


def create_lattice_transitions(n, allow_diagonal=False):
    """
    Create a transition probability matrix for states arranged in a lattice (grid).
    
    Parameters
    ----------
    n : int
        Number of states. Will be arranged in a square grid of size ceil(sqrt(n)) x ceil(sqrt(n)).
    allow_diagonal : bool, optional
        Whether to allow diagonal moves, by default False.
        
    Returns
    -------
    numpy.ndarray
        Transition probability matrix P of shape (n, n), where P[i,j] is the 
        probability of transitioning from state i to state j.
    """
    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(n)))
    
    # Initialize transition matrix
    P = np.zeros((n, n))
    
    # Define possible moves (row, col)
    if allow_diagonal:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    else:
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
    
    # Fill transition matrix
    for state in range(n):
        # Convert state index to grid position
        row, col = divmod(state, grid_size)
        
        # Find valid neighbors
        neighbors = []
        for dr, dc in moves:
            new_row, new_col = row + dr, col + dc
            if 0 <= new_row < grid_size and 0 <= new_col < grid_size:
                neighbor_state = new_row * grid_size + new_col
                if neighbor_state < n:  # Ensure we don't exceed n states
                    neighbors.append(neighbor_state)
        
        # Assign equal probability to each neighbor
        if neighbors:
            for neighbor in neighbors:
                P[state, neighbor] = 1.0 / len(neighbors)
        else:
            # If no neighbors (shouldn't happen in a grid), make state absorbing
            P[state, state] = 1.0
    
    return P


def create_binary_tree_transitions(depth=None, n_nodes=None, bidirectional=True):
    """
    Create a transition probability matrix for states arranged in a binary tree.
    
    Parameters
    ----------
    depth : int, optional
        Depth of the binary tree. If provided, n_nodes is calculated as 2^(depth+1) - 1.
    n_nodes : int, optional
        Number of nodes in the binary tree. If provided, the tree will have as many
        complete levels as possible.
    bidirectional : bool, optional
        If True, transitions can go both ways (to children and to parent), with
        probabilities based on the number of connected edges. Leaf nodes will have
        P[leaf, parent] = 1.0 and will not be absorbing.
        If False (default), transitions only go from parent to children, and leaf nodes
        are absorbing states.
        
    Returns
    -------
    numpy.ndarray
        Transition probability matrix P of shape (n, n), where P[i,j] is the 
        probability of transitioning from state i to state j.
        
    Notes
    -----
    Either depth or n_nodes must be provided. If both are provided, depth takes precedence.
    The tree is filled level by level, with the root at index 0, its left child at index 1,
    right child at index 2, and so on.
    """
    if depth is None and n_nodes is None:
        raise ValueError("Either depth or n_nodes must be provided")
    
    if depth is not None:
        # Calculate number of nodes in a complete binary tree of given depth
        n_nodes = 2**(depth + 1) - 1
    
    # Initialize transition matrix
    P = np.zeros((n_nodes, n_nodes))
    
    if not bidirectional:
        # Original implementation - transitions only from parent to children
        for state in range(n_nodes):
            left_child = 2 * state + 1
            right_child = 2 * state + 2
            
            if left_child < n_nodes and right_child < n_nodes:
                # Both children exist
                P[state, left_child] = 0.5
                P[state, right_child] = 0.5
            elif left_child < n_nodes:
                # Only left child exists
                P[state, left_child] = 1.0
            elif right_child < n_nodes:
                # Only right child exists (unlikely in a properly filled tree)
                P[state, right_child] = 1.0
            else:
                # No children (leaf node) - make it absorbing
                P[state, state] = 1.0
    else:
        # Bidirectional implementation - transitions based on number of connected edges
        for state in range(n_nodes):
            # Find connected nodes (children and parent)
            connected_nodes = []
            
            # Add children if they exist
            left_child = 2 * state + 1
            right_child = 2 * state + 2
            
            if left_child < n_nodes:
                connected_nodes.append(left_child)
            if right_child < n_nodes:
                connected_nodes.append(right_child)
            
            # Add parent if not root
            if state > 0:
                parent = (state - 1) // 2
                connected_nodes.append(parent)
            
            # Special case for leaf nodes
            is_leaf = left_child >= n_nodes and right_child >= n_nodes
            
            if is_leaf and state > 0:
                # Leaf node with 100% probability to go back to parent
                parent = (state - 1) // 2
                P[state, parent] = 1.0
            elif connected_nodes:
                # Distribute probability equally among connected nodes
                for node in connected_nodes:
                    P[state, node] = 1.0 / len(connected_nodes)
            else:
                # Should not happen in a properly structured tree
                P[state, state] = 1.0
    
    return P
