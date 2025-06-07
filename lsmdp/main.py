"""
Main script demonstrating the functionality of the LS-MDP package.

This script creates both a lattice (grid) and a binary tree MDP, solves them
analytically using the desirability function approach, and visualizes the results.
"""

import numpy as np
import matplotlib.pyplot as plt
from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from lsmdp.visualization import visualize_results


def main():
    """
    Main function demonstrating the LS-MDP package functionality.
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Parameters
    alpha = 1.0  # Control cost parameter (temperature)
    gamma = 0.9  # Discount factor
    
    # ===== Lattice (Grid) MDP =====
    print("Solving Lattice (Grid) MDP...")
    
    # Create a 4x4 grid (16 states)
    n_lattice = 16
    grid_shape = (4, 4)
    
    # Create transition matrix
    P_lattice = create_lattice_transitions(n_lattice)
    
    # Generate costs using uniform distribution
    q_lattice = generate_costs(n_lattice, distribution="uniform", low=0.0, high=1.0, seed=42)
    
    # Set goal state (bottom-right corner)
    goal_lattice = [n_lattice - 1]
    q_lattice = set_goal_states(q_lattice, goal_lattice)
    
    # Solve the MDP
    z_lattice, V_lattice, policy_lattice = solve_lsmdp(P_lattice, q_lattice, alpha, gamma, goal_lattice)
    
    # Print results
    print("\nLattice MDP Results:")
    print(f"Grid Shape: {grid_shape}")
    print(f"Goal State: {goal_lattice[0]}")
    print("\nDesirability Function (z*):")
    for i, z_val in enumerate(z_lattice):
        print(f"  State {i}: {z_val:.4f}")
    
    print("\nValue Function (V*):")
    for i, v_val in enumerate(V_lattice):
        print(f"  State {i}: {v_val:.4f}")
    
    print("\nOptimal Policy (π*):")
    for state, next_states in policy_lattice.items():
        next_state_str = ", ".join([f"{s}: {p:.2f}" for s, p in next_states.items()])
        print(f"  State {state} -> {{{next_state_str}}}")
    
    # Visualize results
    fig_lattice = visualize_results(z_lattice, V_lattice, policy_lattice, "lattice", grid_shape=grid_shape)
    
    # ===== Binary Tree MDP =====
    print("\n\nSolving Binary Tree MDP...")
    
    # Create a binary tree with depth 2 (7 nodes)
    depth = 2
    n_tree = 2**(depth + 1) - 1  # 7 nodes
    
    # Create transition matrix
    P_tree = create_binary_tree_transitions(depth=depth)
    
    # Generate costs using exponential distribution
    q_tree = generate_costs(n_tree, distribution="exponential", scale=1.0, seed=42)
    
    # Set leaf nodes as goal states
    leaf_states = [i for i in range(n_tree) if 2*i+1 >= n_tree]  # States with no children
    q_tree = set_goal_states(q_tree, leaf_states)
    
    # Solve the MDP
    z_tree, V_tree, policy_tree = solve_lsmdp(P_tree, q_tree, alpha, gamma, leaf_states)
    
    # Print results
    print("\nBinary Tree MDP Results:")
    print(f"Depth: {depth}")
    print(f"Number of Nodes: {n_tree}")
    print(f"Leaf States (Goals): {leaf_states}")
    
    print("\nDesirability Function (z*):")
    for i, z_val in enumerate(z_tree):
        print(f"  State {i}: {z_val:.4f}")
    
    print("\nValue Function (V*):")
    for i, v_val in enumerate(V_tree):
        print(f"  State {i}: {v_val:.4f}")
    
    print("\nOptimal Policy (π*):")
    for state, next_states in policy_tree.items():
        next_state_str = ", ".join([f"{s}: {p:.2f}" for s, p in next_states.items()])
        print(f"  State {state} -> {{{next_state_str}}}")
    
    # Visualize results
    fig_tree = visualize_results(z_tree, V_tree, policy_tree, "tree", depth=depth)
    
    # Show plots
    plt.show()


if __name__ == "__main__":
    main()
