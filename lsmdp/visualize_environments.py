#!/usr/bin/env python3
"""
Visualization script for lattice and tree environments in policy_aggregation_experiment.py.

This script creates visualizations for both lattice and tree environments with the same
state space size, showing nodes and edges colored according to the cost function (q).
The visualizations are combined into a single figure with a shared colorbar, following
paper-ready styling.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.colors as mcolors
import seaborn as sns

from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states


def set_paper_style(font_scale=1.5):
    """
    Set matplotlib style for paper-ready figures.
    
    Parameters
    ----------
    font_scale : float, optional
        Scale factor for font sizes, by default 1.5
    """
    sns.set_theme(style="whitegrid", font_scale=font_scale)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        'axes.labelsize': 14 * font_scale,
        'axes.titlesize': 16 * font_scale,
        'xtick.labelsize': 12 * font_scale,
        'ytick.labelsize': 12 * font_scale,
        'legend.fontsize': 12 * font_scale,
        'figure.figsize': (8, 6),
        'figure.dpi': 500,
        'savefig.dpi': 1000,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        'axes.grid': True,
        'grid.alpha': 0.5,
    })


def visualize_lattice_environment(q, state_space_size, goal_states=None, ax=None, norm=None, cmap=plt.cm.viridis_r):
    """
    Visualize a lattice environment with costs represented by node colors.
    
    Parameters
    ----------
    q : numpy.ndarray
        Cost function of shape (n,).
    state_space_size : int
        Size of the state space.
    goal_states : list, optional
        List of goal states, by default None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap, by default None.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use, by default plt.cm.viridis_r.
        
    Returns
    -------
    tuple
        A tuple containing:
        - matplotlib.figure.Figure: The figure containing the plot (None if ax is provided).
        - matplotlib.colors.Normalize: The normalization used for coloring.
    """
    # Calculate grid dimensions
    grid_size = int(np.sqrt(state_space_size))
    
    # Create transition matrix to get edge information
    P = create_lattice_transitions(state_space_size)
    
    # Create figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = None
    
    # Create a grid graph
    G = nx.grid_2d_graph(grid_size, grid_size)
    
    # Convert to directed graph
    G = nx.DiGraph(G)
    
    # Create a mapping from 1D state index to 2D grid position
    pos = {}
    node_labels = {}
    for i in range(grid_size):
        for j in range(grid_size):
            state_idx = i * grid_size + j
            pos[state_idx] = (j, grid_size - 1 - i)  # Flip y-axis for visualization
            node_labels[state_idx] = f"{state_idx}\n{q[state_idx]:.2f}"
    
    # Create a new graph with 1D state indices as nodes
    H = nx.DiGraph()
    
    # Add nodes
    for state_idx in range(state_space_size):
        H.add_node(state_idx, cost=q[state_idx])
    
    # Add edges based on transition matrix
    for i in range(state_space_size):
        for j in range(state_space_size):
            if P[i, j] > 0:
                H.add_edge(i, j, weight=P[i, j])
    
    # Create normalization if not provided
    if norm is None:
        norm = mcolors.Normalize(vmin=np.min(q), vmax=np.max(q))
    
    # Draw nodes with colors based on costs
    node_colors = [cmap(norm(q[i])) for i in range(state_space_size)]
    
    # Highlight goal states with a different color if provided
    if goal_states:
        for goal in goal_states:
            node_colors[goal] = 'red'
    
    # Draw nodes
    nx.draw_networkx_nodes(H, pos, node_color=node_colors, node_size=500, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(H, pos, arrows=True, arrowsize=15, ax=ax)
    
    # Draw node labels
    #nx.draw_networkx_labels(H, pos, labels=node_labels, font_size=10, ax=ax)
    
    ax.set_title('Lattice')
    ax.axis('off')
    
    return fig, norm


def visualize_tree_environment(q, depth, goal_states=None, ax=None, norm=None, cmap=plt.cm.viridis_r):
    """
    Visualize a binary tree environment with costs represented by node colors.
    
    Parameters
    ----------
    q : numpy.ndarray
        Cost function of shape (n,).
    depth : int
        Depth of the binary tree.
    goal_states : list, optional
        List of goal states, by default None.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on, by default None (creates a new figure).
    norm : matplotlib.colors.Normalize, optional
        Normalization for the colormap, by default None.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use, by default plt.cm.viridis_r.
        
    Returns
    -------
    tuple
        A tuple containing:
        - matplotlib.figure.Figure: The figure containing the plot (None if ax is provided).
        - matplotlib.colors.Normalize: The normalization used for coloring.
    """
    # Calculate number of nodes
    n_nodes = 2**(depth + 1) - 1
    
    # Create transition matrix to get edge information
    P = create_binary_tree_transitions(depth=depth)
    
    # Create figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))
    else:
        fig = None
    
    # Create a directed graph
    G = nx.DiGraph()
    
    # Add nodes with cost attributes
    for i in range(n_nodes):
        G.add_node(i, cost=float(q[i]))
    
    # Add edges based on transition matrix (only for connected vertices)
    for i in range(n_nodes):
        for j in range(n_nodes):
            if P[i, j] > 0:
                G.add_edge(i, j, weight=P[i, j])
    
    # Create a layout for the tree
    try:
        import pygraphviz
        pos = nx.drawing.nx_agraph.graphviz_layout(G, prog='dot')
    except ImportError:
        # Fallback to a simpler layout algorithm
        pos = nx.spring_layout(G, iterations=100, seed=42)
    
    # Create normalization if not provided
    if norm is None:
        norm = mcolors.Normalize(vmin=np.min(q), vmax=np.max(q))
    
    # Draw nodes with colors based on costs
    node_colors = [cmap(norm(data['cost'])) for _, data in G.nodes(data=True)]
    
    # Highlight goal states with a different color if provided
    if goal_states:
        for goal in goal_states:
            node_colors[goal] = 'red'
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=500, ax=ax)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, arrows=True, arrowsize=15, ax=ax)
    
    # Add labels with state index and cost
    labels = {i: f"{i}\nq={q[i]:.2f}" for i in range(n_nodes)}
    #import pdb; pdb.set_trace()
    #pos = [pos[key]+0.1*np.ones(2) for key in pos.keys() ]
    #nx.draw_networkx_labels(G, pos, labels=labels, font_size=10, ax=ax)
    
    ax.set_title('Tree')
    ax.axis('off')
    
    return fig, norm


def visualize_environments(lattice_q, tree_q, state_space_size, tree_depth, 
                          lattice_goal_states=None, tree_goal_states=None, 
                          save_path=None, cmap=plt.cm.viridis_r):
    """
    Visualize both lattice and tree environments in a single figure with a shared colorbar.
    
    Parameters
    ----------
    lattice_q : numpy.ndarray
        Cost function for lattice environment.
    tree_q : numpy.ndarray
        Cost function for tree environment.
    state_space_size : int
        Size of the state space for lattice.
    tree_depth : int
        Depth of the binary tree.
    lattice_goal_states : list, optional
        List of goal states for lattice, by default None.
    tree_goal_states : list, optional
        List of goal states for tree, by default None.
    save_path : str, optional
        Path to save the figure, by default None.
    cmap : matplotlib.colors.Colormap, optional
        Colormap to use, by default plt.cm.viridis_r.
        
    Returns
    -------
    matplotlib.figure.Figure
        The figure containing both plots.
    """
    # Set paper style
    set_paper_style()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)
    
    # Create a common normalization for both plots
    vmin = min(np.min(lattice_q), np.min(tree_q))
    vmax = max(np.max(lattice_q), np.max(tree_q))
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    
    # Visualize lattice environment
    _, _ = visualize_lattice_environment(
        lattice_q, 
        state_space_size, 
        goal_states=lattice_goal_states,
        ax=ax1,
        norm=norm,
        cmap=cmap
    )
    
    # Visualize tree environment
    _, _ = visualize_tree_environment(
        tree_q, 
        tree_depth, 
        goal_states=tree_goal_states,
        ax=ax2,
        norm=norm,
        cmap=cmap
    )
    
    # Add subplot labels
    #ax1.text(0.05, 0.95, '(a)', transform=ax1.transAxes, fontsize=16, fontweight='bold', va='top')
    #ax2.text(0.05, 0.95, '(b)', transform=ax2.transAxes, fontsize=16, fontweight='bold', va='top')
    
    # Add a single colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=[ax1, ax2], shrink=0.8, pad=0.02)
    cbar.set_label('Cost (q)')
    
    # Save figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        # Also save as PDF for paper
        if save_path.endswith('.png'):
            pdf_path = save_path.replace('.png', '.pdf')
        else:
            pdf_path = save_path + '.pdf'
        plt.savefig(pdf_path, bbox_inches='tight')
    
    return fig


def main():
    """
    Main function to create and visualize environments.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Set parameters
    state_space_size = 16  # Same size for both environments
    
    # For lattice
    grid_size = int(np.sqrt(state_space_size))
    lattice_goal_states = [state_space_size - 1]  # Bottom-right corner
    
    # For tree
    tree_depth = 3  # 2^(3+1) - 1 = 15 nodes (close to 16)
    tree_n_nodes = 2**(tree_depth + 1) - 1
    tree_goal_states = [tree_n_nodes - 1]  # A leaf node
    
    # Generate costs for lattice
    lattice_q = generate_costs(state_space_size, distribution="uniform", low=0.0, high=1.0)
    lattice_q = set_goal_states(lattice_q, lattice_goal_states)
    
    # Generate costs for tree
    tree_q = generate_costs(tree_n_nodes, distribution="uniform", low=0.0, high=1.0)
    tree_q = set_goal_states(tree_q, tree_goal_states)
    
    # Create output directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Visualize both environments in a single figure
    combined_fig = visualize_environments(
        lattice_q=lattice_q,
        tree_q=tree_q,
        state_space_size=state_space_size,
        tree_depth=tree_depth,
        lattice_goal_states=lattice_goal_states,
        tree_goal_states=tree_goal_states,
        save_path="figures/environments.png"
    )
    
    # Show the figure
    plt.show()


if __name__ == "__main__":
    main()
