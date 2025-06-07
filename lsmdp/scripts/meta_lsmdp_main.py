"""
Main script demonstrating the functionality of the Meta-LSMDP module.

This script creates an MDP (either lattice/grid or binary tree), solves it
analytically using the desirability function approach, and evaluates the
effect of random permutations on the solution.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import wandb
from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from visualization import visualize_results
from lsmdp.meta_lsmdp import (
    generate_permutations,
    apply_permutation_to_solution,
    compute_difference,
    aggregate_differences,
    evaluate_permutation_difference
)


def print_evaluation_results(results):
    """
    Print the evaluation results in a formatted way.
    
    Parameters
    ----------
    results : dict
        Dictionary of evaluation results.
    """
    print("\nMDP Permutation Evaluation Results:")
    
    print("\nAverage Differences:")
    print(f"  Desirability Function (z): {results['average']['z']:.6f}")
    print(f"  Value Function (V): {results['average']['V']:.6f}")
    print(f"  Policy (π): {results['average']['policy']:.6f}")
    
    print("\nMaximum Differences:")
    print(f"  Desirability Function (z): {results['max']['z']:.6f}")
    print(f"  Value Function (V): {results['max']['V']:.6f}")
    print(f"  Policy (π): {results['max']['policy']:.6f}")
    
    print("\nStandard Deviation of Differences:")
    print(f"  Desirability Function (z): {results['std']['z']:.6f}")
    print(f"  Value Function (V): {results['std']['V']:.6f}")
    print(f"  Policy (π): {results['std']['policy']:.6f}")


def plot_permutation_differences(permutations, differences, env_type):
    """
    Plot the differences for each permutation.
    
    Parameters
    ----------
    permutations : list
        List of permutations.
    differences : list
        List of difference dictionaries, one for each permutation.
    env_type : str
        Type of environment (e.g., "lattice" or "tree").
        
    Returns
    -------
    matplotlib.figure.Figure
        Figure object.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12))
    
    # Extract values for each component
    z_diffs = [d["z"] for d in differences]
    V_diffs = [d["V"] for d in differences]
    policy_diffs = [d["policy"] for d in differences]
    
    # Plot desirability function differences
    axes[0].bar(range(len(permutations)), z_diffs)
    axes[0].set_title(f"Desirability Function Differences")
    axes[0].set_xlabel("Permutation Index")
    axes[0].set_ylabel("Difference (L2 Norm)")
    axes[0].axhline(y=np.mean(z_diffs), color='r', linestyle='-', label="Mean")
    axes[0].legend()
    
    # Plot value function differences
    axes[1].bar(range(len(permutations)), V_diffs)
    axes[1].set_title(f"Value Function Differences")
    axes[1].set_xlabel("Permutation Index")
    axes[1].set_ylabel("Difference (L2 Norm)")
    axes[1].axhline(y=np.mean(V_diffs), color='r', linestyle='-', label="Mean")
    axes[1].legend()
    
    # Plot policy differences
    axes[2].bar(range(len(permutations)), policy_diffs)
    axes[2].set_title(f"Policy Differences")
    axes[2].set_xlabel("Permutation Index")
    axes[2].set_ylabel("Difference (L2 Norm)")
    axes[2].axhline(y=np.mean(policy_diffs), color='r', linestyle='-', label="Mean")
    axes[2].legend()
    
    plt.tight_layout()
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
        description="Demonstrate the functionality of the Meta-LSMDP module."
    )
    
    # Environment selection
    parser.add_argument("--env_type", type=str, choices=["lattice", "tree"], default="lattice",
                        help="Type of environment to evaluate (default: lattice)")
    
    # MDP parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Control cost parameter (temperature) (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor (default: 0.9)")
    
    # Permutation parameters
    parser.add_argument("--k", type=int, default=3,
                        help="Number of independent random permutations (default: 3)")
    parser.add_argument("--l", type=int, default=2,
                        help="Number of times the permutations will be combined (default: 2)")
    
    # Lattice MDP parameters
    parser.add_argument("--lattice_size", type=int, default=4,
                        help="Size n of the lattice grid (n x n) (default: 4)")
    parser.add_argument("--lattice_cost_distribution", type=str, default="uniform",
                        choices=["uniform", "exponential", "normal"],
                        help="Cost distribution for lattice MDP (default: uniform)")
    
    # Binary tree MDP parameters
    parser.add_argument("--tree_depth", type=int, default=2,
                        help="Depth of the binary tree (default: 2)")
    parser.add_argument("--tree_cost_distribution", type=str, default="uniform",
                        choices=["uniform", "exponential", "normal"],
                        help="Cost distribution for binary tree MDP (default: uniform)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Do not use wandb for logging (default: False)")
    parser.add_argument("--wandb-project", type=str, default="meta_lsmdp",
                        help="wandb project name (default: meta_lsmdp)")
    parser.add_argument("--wandb-entity", type=str, default="cmvl_nelf",
                        help="wandb entity name (default: None)")
    
    return parser.parse_args()


def generate_environment(env_type, args):
    """
    Generate an environment based on the specified type.
    
    Parameters
    ----------
    env_type : str
        Type of environment to generate ("lattice" or "tree").
    args : argparse.Namespace
        Command-line arguments.
        
    Returns
    -------
    tuple
        A tuple containing:
        - P (numpy.ndarray): Transition probability matrix.
        - q (numpy.ndarray): Cost function.
        - goal_states (list): List of goal states.
        - n (int): Number of states.
        - viz_params (dict): Parameters for visualization.
    """
    if env_type == "lattice":
        # Create a lattice grid
        n = args.lattice_size ** 2
        grid_shape = (args.lattice_size, args.lattice_size)
        
        # Create transition matrix
        P = create_lattice_transitions(n)
        
        # Generate costs using specified distribution
        if args.lattice_cost_distribution == "uniform":
            q = generate_costs(n, distribution="uniform", low=0.0, high=1.0, seed=args.seed)
        elif args.lattice_cost_distribution == "exponential":
            q = generate_costs(n, distribution="exponential", scale=1.0, seed=args.seed)
        elif args.lattice_cost_distribution == "normal":
            q = generate_costs(n, distribution="normal", loc=0.5, scale=0.1, seed=args.seed)
        
        # Set goal state (bottom-right corner)
        goal_states = [n - 1]
        q = set_goal_states(q, goal_states)
        
        # Visualization parameters
        viz_params = {"type": "lattice", "grid_shape": grid_shape}
        
    elif env_type == "tree":
        # Create a binary tree
        depth = args.tree_depth
        n = 2**(depth + 1) - 1
        
        # Create transition matrix
        P = create_binary_tree_transitions(depth=depth)
        
        # Generate costs using specified distribution
        if args.tree_cost_distribution == "uniform":
            q = generate_costs(n, distribution="uniform", low=0.0, high=1.0, seed=args.seed)
        elif args.tree_cost_distribution == "exponential":
            q = generate_costs(n, distribution="exponential", scale=1.0, seed=args.seed)
        elif args.tree_cost_distribution == "normal":
            q = generate_costs(n, distribution="normal", loc=0.5, scale=0.1, seed=args.seed)
        
        # Set leaf nodes as goal states
        goal_states = [i for i in range(n) if 2*i+1 >= n]  # States with no children
        q = set_goal_states(q, goal_states)
        
        # Visualization parameters
        viz_params = {"type": "tree", "depth": depth}
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return P, q, goal_states, n, viz_params


def main():
    """
    Main function demonstrating the Meta-LSMDP module functionality.
    """
    # Parse command-line arguments
    args = parse_args()
    args.k = 2**round(8/args.l)
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Initialize wandb if not disabled
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "env_type": args.env_type,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "k": args.k,
                "l": args.l,
                "lattice_size": args.lattice_size,
                "lattice_cost_distribution": args.lattice_cost_distribution,
                "tree_depth": args.tree_depth,
                "tree_cost_distribution": args.tree_cost_distribution,
                "seed": args.seed
            }
        )
    
    # Generate environment based on selected type
    print(f"Evaluating {args.env_type.capitalize()} MDP...")
    P, q, goal_states, n, viz_params = generate_environment(args.env_type, args)
    
    # Solve the MDP (only once)
    z, V, policy = solve_lsmdp(P, q, args.alpha, args.gamma, goal_states)
    
    # Visualize original solution
    fig_original = visualize_results(z, V, policy, viz_params["type"], 
                                    **{k: v for k, v in viz_params.items() if k != "type"})
    fig_original.suptitle(f"Original {args.env_type.capitalize()} MDP Solution")
    
    # Generate permutations (only once)
    permutations = generate_permutations(n, args.k, args.l, seed=args.seed)
    
    # Compute differences for each permutation (for visualization and default L2 measure)
    differences = []
    for perm in permutations:
        # Apply permutation to solution
        z_prime, V_prime, policy_prime = apply_permutation_to_solution(z, V, policy, perm)
        
        # Compute difference
        diff = compute_difference(z, V, policy, z_prime, V_prime, policy_prime, measure_type="L2")
        differences.append(diff)
        
        # Visualize permuted solution for the first permutation
        if len(differences) == 1:
            fig_perm = visualize_results(z_prime, V_prime, policy_prime, viz_params["type"], 
                                        **{k: v for k, v in viz_params.items() if k != "type"})
            fig_perm.suptitle(f"Permuted {args.env_type.capitalize()} MDP Solution (Permutation 0)")
    
    # Evaluate permutation differences with default measure type (L2) and aggregation type (average)
    results = evaluate_permutation_difference(
        P, q, args.alpha, args.gamma, goal_states, 
        k=args.k, l=args.l, 
        measure_type="L2", 
        aggregation_type="average", 
        seed=args.seed
    )
    
    # Print results
    print_evaluation_results(results)
    
    # Plot differences
    fig_diffs = plot_permutation_differences(permutations, differences, args.env_type)
    
    # Compare different measure types
    print("\n\nComparing Different Measure Types:")
    
    measure_types = ["L1", "L2", "KL", "Wasserstein"]
    
    for measure_type in measure_types:
        # Evaluate permutation differences with different measure types
        results = evaluate_permutation_difference(
            P, q, args.alpha, args.gamma, goal_states, 
            k=args.k, l=args.l, 
            measure_type=measure_type, 
            aggregation_type="average", 
            seed=args.seed
        )
        
        # Print results
        print(f"\n{measure_type} Measure:")
        print(f"  Average Desirability Difference: {results['average']['z']:.6f}")
        print(f"  Average Value Function Difference: {results['average']['V']:.6f}")
        print(f"  Average Policy Difference: {results['average']['policy']:.6f}")
        
        # Log to wandb if enabled
        if not args.no_wandb:
            wandb.log({
                f"{measure_type}_z": results['average']['z'],
                f"{measure_type}_V": results['average']['V'],
                f"{measure_type}_policy": results['average']['policy']
            })
    
    # Compare different aggregation types
    print("\n\nComparing Different Aggregation Types:")
    
    aggregation_types = ["average", "max", "std"]
    
    for agg_type in aggregation_types:
        # Evaluate permutation differences with different aggregation types
        results = evaluate_permutation_difference(
            P, q, args.alpha, args.gamma, goal_states, 
            k=args.k, l=args.l, 
            measure_type="L2", 
            aggregation_type=agg_type, 
            seed=args.seed
        )
        
        # Print results
        print(f"\n{agg_type.capitalize()} Aggregation:")
        print(f"  Desirability Difference: {results[agg_type]['z']:.6f}")
        print(f"  Value Function Difference: {results[agg_type]['V']:.6f}")
        print(f"  Policy Difference: {results[agg_type]['policy']:.6f}")
        
        # Log to wandb if enabled
        if not args.no_wandb:
            wandb.log({
                f"L2_{agg_type}_z": results[agg_type]['z'],
                f"L2_{agg_type}_V": results[agg_type]['V'],
                f"L2_{agg_type}_policy": results[agg_type]['policy']
            })
    
    # Finish wandb run if enabled
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
