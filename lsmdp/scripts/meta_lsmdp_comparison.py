"""
Script for comparing the difference between original and permuted LSMDP solutions
for different combinations of (k, l) where k^l = 2^8.

This script evaluates the difference between (z, V, π) and (z', V', π') for:
- (k, l) = (2^8, 1) = (256, 1)
- (k, l) = (2^4, 2) = (16, 2)
- (k, l) = (2^2, 4) = (4, 4)
- (k, l) = (2, 8) = (2, 8)
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import argparse
from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from lsmdp.meta_lsmdp import evaluate_permutation_difference


def compare_kl_combinations(P, q, alpha, gamma, goal_states, measure_type="L2", seed=42):
    """
    Compare the difference between original and permuted LSMDP solutions
    for different combinations of (k, l) where k^l = 2^8.
    
    Parameters
    ----------
    P : numpy.ndarray
        Transition probability matrix.
    q : numpy.ndarray
        Cost function.
    alpha : float
        Control cost parameter.
    gamma : float
        Discount factor.
    goal_states : list
        List of goal states.
    measure_type : str, optional
        Type of difference measure to use, by default "L2".
    seed : int, optional
        Random seed for reproducibility, by default 42.
        
    Returns
    -------
    dict
        Dictionary of results for each (k, l) combination.
    """
    # Define the combinations of (k, l) where k^l = 2^8
    combinations = [
        (2**8, 1),  # (256, 1)
        (2**4, 2),  # (16, 2)
        (2**2, 4),  # (4, 4)
        (2, 8)      # (2, 8)
    ]
    
    results = {}
    
    for k, l in combinations:
        print(f"\nEvaluating (k, l) = ({k}, {l})...")
        
        # Evaluate permutation differences
        result = evaluate_permutation_difference(
            P, q, alpha, gamma, goal_states, 
            k=k, l=l, 
            measure_type=measure_type, 
            aggregation_type="average", 
            seed=seed
        )
        
        # Store the results
        results[(k, l)] = result
        
        # Print the results
        print(f"(k, l) = ({k}, {l}) Results:")
        print(f"  Average Desirability Difference: {result['average']['z']:.6f}")
        print(f"  Average Value Function Difference: {result['average']['V']:.6f}")
        print(f"  Average Policy Difference: {result['average']['policy']:.6f}")
    
    return results


def save_results_to_csv(results, mdp_type, measure_type):
    """
    Save the results to a CSV file.
    
    Parameters
    ----------
    results : dict
        Dictionary of results for each (k, l) combination.
    mdp_type : str
        Type of MDP (e.g., "Lattice" or "Binary Tree").
    measure_type : str
        Type of difference measure used.
        
    Returns
    -------
    str
        Path to the saved CSV file.
    """
    # Create a directory for results if it doesn't exist
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create a DataFrame from the results
    data = []
    for (k, l), result in results.items():
        row = {
            'k': k,
            'l': l,
            'k^l': k**l,
            'z_avg': result['average']['z'],
            'V_avg': result['average']['V'],
            'policy_avg': result['average']['policy'],
            'z_max': result['max']['z'],
            'V_max': result['max']['V'],
            'policy_max': result['max']['policy'],
            'z_std': result['std']['z'],
            'V_std': result['std']['V'],
            'policy_std': result['std']['policy']
        }
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # Sort by k and l
    df = df.sort_values(by=['k', 'l'])
    
    # Save to CSV
    filename = f"{results_dir}/{mdp_type.lower()}_comparison_{measure_type.lower()}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Results saved to {filename}")
    
    return filename


def plot_comparison_results(results, mdp_type, measure_type):
    """
    Plot the comparison results.
    
    Parameters
    ----------
    results : dict
        Dictionary of results for each (k, l) combination.
    mdp_type : str
        Type of MDP (e.g., "Lattice" or "Binary Tree").
    measure_type : str
        Type of difference measure used.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    # Extract the combinations and results
    combinations = list(results.keys())
    combinations.sort()  # Sort by k, then l
    
    labels = [f"({k}, {l})" for k, l in combinations]
    
    z_diffs = [results[(k, l)]['average']['z'] for k, l in combinations]
    V_diffs = [results[(k, l)]['average']['V'] for k, l in combinations]
    policy_diffs = [results[(k, l)]['average']['policy'] for k, l in combinations]
    
    # Create the figure
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot desirability function differences
    bar_width = 0.8
    bars = axes[0].bar(range(len(combinations)), z_diffs, width=bar_width)
    axes[0].set_title(f"{mdp_type} MDP: Desirability Function Differences ({measure_type} Norm)")
    axes[0].set_xlabel("(k, l) Combination")
    axes[0].set_ylabel("Average Difference")
    axes[0].set_xticks(range(len(combinations)))
    axes[0].set_xticklabels(labels)
    axes[0].axhline(y=np.mean(z_diffs), color='r', linestyle='--', label="Mean")
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    axes[0].legend()
    
    # Plot value function differences
    bars = axes[1].bar(range(len(combinations)), V_diffs, width=bar_width)
    axes[1].set_title(f"{mdp_type} MDP: Value Function Differences ({measure_type} Norm)")
    axes[1].set_xlabel("(k, l) Combination")
    axes[1].set_ylabel("Average Difference")
    axes[1].set_xticks(range(len(combinations)))
    axes[1].set_xticklabels(labels)
    axes[1].axhline(y=np.mean(V_diffs), color='r', linestyle='--', label="Mean")
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    axes[1].legend()
    
    # Plot policy differences
    bars = axes[2].bar(range(len(combinations)), policy_diffs, width=bar_width)
    axes[2].set_title(f"{mdp_type} MDP: Policy Differences ({measure_type} Norm)")
    axes[2].set_xlabel("(k, l) Combination")
    axes[2].set_ylabel("Average Difference")
    axes[2].set_xticks(range(len(combinations)))
    axes[2].set_xticklabels(labels)
    axes[2].axhline(y=np.mean(policy_diffs), color='r', linestyle='--', label="Mean")
    
    # Add value labels on top of the bars
    for bar in bars:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0)
    
    axes[2].legend()
    
    plt.tight_layout()
    
    # Create a directory for figures if it doesn't exist
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # Save the figure
    filename = f"{figures_dir}/{mdp_type.lower()}_comparison_{measure_type.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"Figure saved to {filename}")
    
    return fig


def plot_combined_comparison(results_lattice, results_tree, measure_type):
    """
    Plot a combined comparison of lattice and binary tree results.
    
    Parameters
    ----------
    results_lattice : dict
        Dictionary of results for each (k, l) combination for lattice MDP.
    results_tree : dict
        Dictionary of results for each (k, l) combination for binary tree MDP.
    measure_type : str
        Type of difference measure used.
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    # Extract the combinations and results
    combinations = list(results_lattice.keys())
    combinations.sort()  # Sort by k, then l
    
    labels = [f"({k}, {l})" for k, l in combinations]
    
    # Create the figure
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Set width of bars
    bar_width = 0.35
    index = np.arange(len(combinations))
    
    # Plot desirability function differences
    lattice_z = [results_lattice[(k, l)]['average']['z'] for k, l in combinations]
    tree_z = [results_tree[(k, l)]['average']['z'] for k, l in combinations]
    
    bars1 = axes[0].bar(index - bar_width/2, lattice_z, bar_width, label='Lattice')
    bars2 = axes[0].bar(index + bar_width/2, tree_z, bar_width, label='Binary Tree')
    
    axes[0].set_title(f"Desirability Function Differences ({measure_type} Norm)")
    axes[0].set_xlabel("(k, l) Combination")
    axes[0].set_ylabel("Average Difference")
    axes[0].set_xticks(index)
    axes[0].set_xticklabels(labels)
    axes[0].legend()
    
    # Add value labels on top of the bars
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Plot value function differences
    lattice_V = [results_lattice[(k, l)]['average']['V'] for k, l in combinations]
    tree_V = [results_tree[(k, l)]['average']['V'] for k, l in combinations]
    
    bars1 = axes[1].bar(index - bar_width/2, lattice_V, bar_width, label='Lattice')
    bars2 = axes[1].bar(index + bar_width/2, tree_V, bar_width, label='Binary Tree')
    
    axes[1].set_title(f"Value Function Differences ({measure_type} Norm)")
    axes[1].set_xlabel("(k, l) Combination")
    axes[1].set_ylabel("Average Difference")
    axes[1].set_xticks(index)
    axes[1].set_xticklabels(labels)
    axes[1].legend()
    
    # Add value labels on top of the bars
    for bar in bars1:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    # Plot policy differences
    lattice_policy = [results_lattice[(k, l)]['average']['policy'] for k, l in combinations]
    tree_policy = [results_tree[(k, l)]['average']['policy'] for k, l in combinations]
    
    bars1 = axes[2].bar(index - bar_width/2, lattice_policy, bar_width, label='Lattice')
    bars2 = axes[2].bar(index + bar_width/2, tree_policy, bar_width, label='Binary Tree')
    
    axes[2].set_title(f"Policy Differences ({measure_type} Norm)")
    axes[2].set_xlabel("(k, l) Combination")
    axes[2].set_ylabel("Average Difference")
    axes[2].set_xticks(index)
    axes[2].set_xticklabels(labels)
    axes[2].legend()
    
    # Add value labels on top of the bars
    for bar in bars1:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    for bar in bars2:
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.4f}', ha='center', va='bottom', rotation=0, fontsize=8)
    
    plt.tight_layout()
    
    # Create a directory for figures if it doesn't exist
    figures_dir = "figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    # Save the figure
    filename = f"{figures_dir}/combined_comparison_{measure_type.lower()}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    
    print(f"Combined figure saved to {filename}")
    
    return fig


def parse_args():
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Compare the difference between original and permuted LSMDP solutions "
                    "for different combinations of (k, l) where k^l = 2^8."
    )
    
    # MDP parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Control cost parameter (temperature) (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor (default: 0.9)")
    
    # Lattice MDP parameters
    parser.add_argument("--lattice-size", type=int, default=4,
                        help="Size of the lattice grid (n x n) (default: 4)")
    parser.add_argument("--lattice-cost-distribution", type=str, default="uniform",
                        choices=["uniform", "exponential", "normal"],
                        help="Cost distribution for lattice MDP (default: uniform)")
    
    # Binary tree MDP parameters
    parser.add_argument("--tree-depth", type=int, default=3,
                        help="Depth of the binary tree (default: 3)")
    parser.add_argument("--tree-cost-distribution", type=str, default="exponential",
                        choices=["uniform", "exponential", "normal"],
                        help="Cost distribution for binary tree MDP (default: exponential)")
    
    # Difference measure parameters
    parser.add_argument("--measure-type", type=str, default="L2",
                        choices=["L1", "L2", "KL", "Wasserstein"],
                        help="Type of difference measure to use (default: L2)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no-plots", action="store_true",
                        help="Do not show plots (default: False)")
    
    return parser.parse_args()


def main():
    """
    Main function for comparing different (k, l) combinations.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # ===== Lattice (Grid) MDP =====
    print("Evaluating Lattice (Grid) MDP...")
    
    # Create a lattice grid
    sqrtn = args.lattice_size
    n_lattice = sqrtn**2
    grid_shape = (sqrtn, sqrtn)
    
    # Create transition matrix
    P_lattice = create_lattice_transitions(n_lattice)
    
    # Generate costs using specified distribution
    if args.lattice_cost_distribution == "uniform":
        q_lattice = generate_costs(n_lattice, distribution="uniform", low=0.0, high=1.0, seed=args.seed)
    elif args.lattice_cost_distribution == "exponential":
        q_lattice = generate_costs(n_lattice, distribution="exponential", scale=1.0, seed=args.seed)
    elif args.lattice_cost_distribution == "normal":
        q_lattice = generate_costs(n_lattice, distribution="normal", loc=0.5, scale=0.1, seed=args.seed)
    
    # Set goal state (bottom-right corner)
    goal_lattice = [n_lattice - 1]
    q_lattice = set_goal_states(q_lattice, goal_lattice)
    
    # Compare different (k, l) combinations
    print("\nComparing different (k, l) combinations for Lattice MDP...")
    results_lattice = compare_kl_combinations(
        P_lattice, q_lattice, args.alpha, args.gamma, goal_lattice, 
        measure_type=args.measure_type, seed=args.seed
    )
    
    # Save results to CSV
    save_results_to_csv(results_lattice, "Lattice", args.measure_type)
    
    # Plot the comparison results
    fig_lattice = plot_comparison_results(results_lattice, "Lattice", args.measure_type)
    
    # ===== Binary Tree MDP =====
    print("\n\nEvaluating Binary Tree MDP...")
    
    # Create a binary tree
    depth = args.tree_depth
    n_tree = 2**(depth + 1) - 1
    
    # Create transition matrix
    P_tree = create_binary_tree_transitions(depth=depth)
    
    # Generate costs using specified distribution
    if args.tree_cost_distribution == "uniform":
        q_tree = generate_costs(n_tree, distribution="uniform", low=0.0, high=1.0, seed=args.seed)
    elif args.tree_cost_distribution == "exponential":
        q_tree = generate_costs(n_tree, distribution="exponential", scale=1.0, seed=args.seed)
    elif args.tree_cost_distribution == "normal":
        q_tree = generate_costs(n_tree, distribution="normal", loc=0.5, scale=0.1, seed=args.seed)
    
    # Set leaf nodes as goal states
    leaf_states = [i for i in range(n_tree) if 2*i+1 >= n_tree]  # States with no children
    q_tree = set_goal_states(q_tree, leaf_states)
    
    # Compare different (k, l) combinations
    print("\nComparing different (k, l) combinations for Binary Tree MDP...")
    results_tree = compare_kl_combinations(
        P_tree, q_tree, args.alpha, args.gamma, leaf_states, 
        measure_type=args.measure_type, seed=args.seed
    )
    
    # Save results to CSV
    save_results_to_csv(results_tree, "BinaryTree", args.measure_type)
    
    # Plot the comparison results
    fig_tree = plot_comparison_results(results_tree, "Binary Tree", args.measure_type)
    
    # Plot combined comparison
    fig_combined = plot_combined_comparison(results_lattice, results_tree, args.measure_type)
    
    # Show plots if not disabled
    if not args.no_plots:
        plt.show()


if __name__ == "__main__":
    main()
