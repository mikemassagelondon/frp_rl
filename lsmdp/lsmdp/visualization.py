"""
Visualization utilities for Linearly Solvable Markov Decision Processes (LS-MDPs).

This module provides functions to visualize the results of LS-MDP solutions,
including the value function and optimal policy for different state space structures.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
import networkx as nx

# Check if pygraphviz is available
#try:
#    import pygraphviz
#    PYGRAPHVIZ_AVAILABLE = True
#except ImportError:
PYGRAPHVIZ_AVAILABLE = False


def visualize_grid_value(V, grid_shape, ax=None, cmap='viridis_r', title='Value Function'):
    """
    Visualize the value function on a grid as a heatmap.
    
    Parameters
    ----------
    V : numpy.ndarray
        Value function of shape (n,).
    grid_shape : tuple
        Shape of the grid (rows, cols).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    cmap : str, optional
        Colormap to use, by default 'viridis_r'.
    title : str, optional
        Title of the plot, by default 'Value Function'.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure
    
    # Reshape value function to grid
    V_grid = V[:grid_shape[0] * grid_shape[1]].reshape(grid_shape)
    
    # Create heatmap
    im = ax.imshow(V_grid, cmap=cmap)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Value')
    
    # Add grid lines
    ax.grid(which='major', color='w', linestyle='-', linewidth=1.5, alpha=0.7)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(grid_shape[1]))
    ax.set_yticks(np.arange(grid_shape[0]))
    ax.set_xticklabels(np.arange(grid_shape[1]))
    ax.set_yticklabels(np.arange(grid_shape[0]))
    
    # Add state indices as text
    for i in range(grid_shape[0]):
        for j in range(grid_shape[1]):
            state_idx = i * grid_shape[1] + j
            ax.text(j, i, f"{state_idx}\n{V_grid[i, j]:.2f}", ha="center", va="center", 
                    color="w" if V_grid[i, j] > np.median(V_grid) else "k", fontsize=9)
    
    ax.set_title(title)
    plt.tight_layout()
    
    return fig


def visualize_grid_policy(policy, grid_shape, ax=None, title='Optimal Policy'):
    """
    Visualize the optimal policy on a grid.
    
    Parameters
    ----------
    policy : dict
        Optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    grid_shape : tuple
        Shape of the grid (rows, cols).
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    title : str, optional
        Title of the plot, by default 'Optimal Policy'.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure
    
    rows, cols = grid_shape
    
    # Create a grid
    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    
    # Draw grid lines
    for i in range(rows + 1):
        ax.axhline(i - 0.5, color='k', linestyle='-', alpha=0.3)
    for j in range(cols + 1):
        ax.axvline(j - 0.5, color='k', linestyle='-', alpha=0.3)
    
    # Add state indices
    for i in range(rows):
        for j in range(cols):
            state_idx = i * cols + j
            ax.text(j, i, f"{state_idx}", ha="center", va="center", fontsize=10)
    
    # Draw policy arrows
    for state, next_states in policy.items():
        if state >= rows * cols:
            continue  # Skip states outside the grid
        
        i, j = divmod(state, cols)
        
        for next_state, prob in next_states.items():
            if next_state >= rows * cols:
                continue  # Skip transitions outside the grid
            
            ni, nj = divmod(next_state, cols)
            
            # Skip self-transitions with probability 1 (absorbing states)
            if state == next_state and prob == 1.0:
                # Draw a circle to indicate absorbing state
                circle = plt.Circle((j, i), 0.2, color='red', alpha=0.3)
                ax.add_patch(circle)
                continue
            
            # Calculate arrow properties based on probability
            width = 0.005 + 0.02 * prob
            alpha = 0.3 + 0.7 * prob
            
            # Draw arrow
            arrow = FancyArrowPatch(
                (j, i),
                (nj, ni),
                arrowstyle='-|>',
                mutation_scale=15,
                linewidth=width * 100,
                alpha=alpha,
                color='blue'
            )
            ax.add_patch(arrow)
            
            # Add probability text at midpoint
            mid_x = (j + nj) / 2
            mid_y = (i + ni) / 2
            ax.text(mid_x, mid_y, f"{prob:.2f}", ha="center", va="center", 
                    fontsize=8, bbox=dict(facecolor='white', alpha=0.7))
    
    ax.set_title(title)
    ax.set_aspect('equal')
    plt.tight_layout()
    
    return fig


def visualize_tree_value(V, depth=None, n_nodes=None, ax=None, title='Value Function on Tree'):
    """
    Visualize the value function on a binary tree.
    
    Parameters
    ----------
    V : numpy.ndarray
        Value function of shape (n,).
    depth : int, optional
        Depth of the binary tree. If provided, n_nodes is calculated as 2^(depth+1) - 1.
    n_nodes : int, optional
        Number of nodes in the binary tree. If provided, the tree will have as many
        complete levels as possible.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    title : str, optional
        Title of the plot, by default 'Value Function on Tree'.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    if depth is None and n_nodes is None:
        raise ValueError("Either depth or n_nodes must be provided")
    
    if depth is not None:
        # Calculate number of nodes in a complete binary tree of given depth
        n_nodes = 2**(depth + 1) - 1
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with value attributes
    for i in range(min(n_nodes, len(V))):
        G.add_node(i, value=float(V[i]))
    
    # Add edges
    for i in range(n_nodes):
        left = 2 * i + 1
        right = 2 * i + 2
        if left < n_nodes:
            G.add_edge(i, left)
        if right < n_nodes:
            G.add_edge(i, right)
    
    # Create a layout for the tree
    if PYGRAPHVIZ_AVAILABLE:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    else:
        # Fallback to a simpler layout algorithm
        pos = nx.spring_layout(G, iterations=100, seed=42)
    
    # Normalize values for coloring
    values = np.array([data['value'] for _, data in G.nodes(data=True)])
    vmin, vmax = np.min(values), np.max(values)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Draw nodes with colors based on values
    cmap = plt.cm.viridis_r
    node_colors = [cmap(norm(data['value'])) for _, data in G.nodes(data=True)]
    
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, ax=ax)
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=20, ax=ax)
    
    # Add labels with state index and value
    labels = {i: f"{i}\nV={V[i]:.2f}" for i in range(min(n_nodes, len(V)))}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Value')
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def visualize_tree_policy(policy, depth=None, n_nodes=None, ax=None, title='Optimal Policy on Tree'):
    """
    Visualize the optimal policy on a binary tree.
    
    Parameters
    ----------
    policy : dict
        Optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    depth : int, optional
        Depth of the binary tree. If provided, n_nodes is calculated as 2^(depth+1) - 1.
    n_nodes : int, optional
        Number of nodes in the binary tree. If provided, the tree will have as many
        complete levels as possible.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    title : str, optional
        Title of the plot, by default 'Optimal Policy on Tree'.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the plot.
    """
    if depth is None and n_nodes is None:
        raise ValueError("Either depth or n_nodes must be provided")
    
    if depth is not None:
        # Calculate number of nodes in a complete binary tree of given depth
        n_nodes = 2**(depth + 1) - 1
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = ax.figure
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes
    for i in range(n_nodes):
        G.add_node(i)
    
    # Add edges with weights based on policy
    for state, next_states in policy.items():
        if int(state) >= n_nodes:
            continue
        
        for next_state, prob in next_states.items():
            if int(next_state) >= n_nodes:
                continue
            
            G.add_edge(int(state), int(next_state), weight=prob)
    
    # Create a layout for the tree
    if PYGRAPHVIZ_AVAILABLE:
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    else:
        # Fallback to a simpler layout algorithm
        pos = nx.spring_layout(G, iterations=100, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=700, ax=ax)
    
    # Draw edges with varying width based on probability
    for u, v, data in G.edges(data=True):
        width = 1 + 5 * data['weight']
        nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                              arrows=True, arrowsize=20, ax=ax,
                              alpha=0.5 + 0.5 * data['weight'])
    
    # Add node labels
    nx.draw_networkx_labels(G, pos, labels={i: str(i) for i in range(n_nodes)}, 
                           font_size=10, ax=ax)
    
    # Add edge labels (probabilities)
    edge_labels = {(u, v): f"{data['weight']:.2f}" for u, v, data in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)
    
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    
    return fig


def visualize_results(z, V, policy, structure_type, **kwargs):
    """
    Visualize the results of an LS-MDP solution.
    
    Parameters
    ----------
    z : numpy.ndarray
        Desirability function of shape (n,).
    V : numpy.ndarray
        Value function of shape (n,).
    policy : dict
        Optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    structure_type : str
        Type of state space structure, either "lattice" or "tree".
    **kwargs : dict
        Additional parameters for the specific visualization functions.
        
    Returns
    -------
    tuple
        A tuple containing the figures for the value function and policy visualizations.
    """
    if structure_type.lower() == "lattice" or structure_type.lower() == "grid":
        if 'grid_shape' not in kwargs:
            # Try to infer grid shape
            n = len(V)
            grid_size = int(np.ceil(np.sqrt(n)))
            grid_shape = (grid_size, grid_size)
        else:
            grid_shape = kwargs['grid_shape']
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Visualize value function
        visualize_grid_value(V, grid_shape, ax=ax1, title='Value Function')
        
        # Visualize policy
        visualize_grid_policy(policy, grid_shape, ax=ax2, title='Optimal Policy')
        
        plt.tight_layout()
        return fig
    
    elif structure_type.lower() == "tree" or structure_type.lower() == "binary_tree":
        if 'depth' not in kwargs and 'n_nodes' not in kwargs:
            # Try to infer tree size
            n_nodes = len(V)
        else:
            depth = kwargs.get('depth')
            n_nodes = kwargs.get('n_nodes')
            if depth is not None:
                n_nodes = 2**(depth + 1) - 1
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Visualize value function
        visualize_tree_value(V, n_nodes=n_nodes, ax=ax1, title='Value Function on Tree')
        
        # Visualize policy
        visualize_tree_policy(policy, n_nodes=n_nodes, ax=ax2, title='Optimal Policy on Tree')
        
        plt.tight_layout()
        return fig
    
    else:
        raise ValueError(f"Unknown structure type: {structure_type}")
