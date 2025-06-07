"""
Policy aggregation experiment module for LS-MDP.

This module implements an experiment to evaluate different policy aggregation methods
for LSMDP solutions using random permutations.
"""

import numpy as np
import argparse
import math
import wandb
from scipy.special import kl_div
from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from lsmdp.meta_lsmdp import generate_permutations, apply_permutation_to_solution
from lsmdp.policy_aggregation import (
    aggregate_z_space,
    aggregate_probability_space,
    aggregate_majority_votes
)
from tqdm import tqdm


def parse_args():
    """
    Parse command-line arguments.
    
    Returns
    -------
    argparse.Namespace
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Evaluate policy aggregation methods for LSMDP solutions."
    )
    
    # Environment selection
    parser.add_argument("--env_type", type=str, choices=["lattice", "tree"], default="tree",
                        help="Type of environment to evaluate (default: %(default)s)")
    
    # State space size
    parser.add_argument("--state_space_size", type=int, default=64,
                        help="Size of the state space (default:64)")
    
    # Permutation parameters
    parser.add_argument("--word_lengthes", type=str, nargs="+",
                        default=[1, 2, 4, 8],
                        help="List of word length (default: %(default)s)")
    parser.add_argument("--max_length", type=int, default=8,
                        help="Maximal Length of words (default: %(default)s)")
    
    # MDP parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Control cost parameter (temperature) (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor (default: 0.9)")

    
    # Evaluation parameters
    parser.add_argument("--evaluation_methods", type=str, nargs="+", 
                        default=["L1", "L2", "KL", "CE"],
                        help="Evaluation methods to use (default: L1 L2 KL CE)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Do not use wandb for logging (default: False)")
    parser.add_argument("--wandb-project", type=str, default="lsmdp_policy_aggregation",
                        help="wandb project name (default: lsmdp_policy_aggregation)")
    parser.add_argument("--wandb-entity", type=str, default="cmvl_nelf",
                        help="wandb entity name (default: None)")
    
    return parser.parse_args()


def parse_env_args(args):
    """
    Parse environment arguments based on state space size.
    
    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
        
    Returns
    -------
    dict
        Environment parameters.
    """
    env_kwargs = {"type": args.env_type}
    
    if args.env_type == "lattice":
        # For lattice, determine the grid size
        grid_size = int(math.sqrt(args.state_space_size))
        env_kwargs["size"] = grid_size
    elif args.env_type == "tree":
        # For tree, determine the depth
        # For a complete binary tree with n nodes, depth = log2(n+1) - 1
        depth = int(math.log2(args.state_space_size + 1)) - 1
        env_kwargs["depth"] = depth
    
    return env_kwargs


def make_env(env_kwargs):
    """
    Create an environment based on the specified parameters.
    
    Parameters
    ----------
    env_kwargs : dict
        Environment parameters.
        
    Returns
    -------
    tuple
        A tuple containing:
        - P (numpy.ndarray): Transition probability matrix.
        - q (numpy.ndarray): Cost function.
        - goal_states (list): List of goal states.
    """
    env_type = env_kwargs["type"]

    if env_type == "lattice":
        # Create a lattice grid
        size = env_kwargs["size"]
        n = size ** 2
        
        # Create transition matrix
        P = create_lattice_transitions(n)
        
        # Generate costs using uniform distribution
        q = generate_costs(n, distribution="uniform", low=0.0, high=1.0)
        
        # Set goal state (bottom-right corner)
        goal_states = [n - 1]
        q = set_goal_states(q, goal_states)
        
    elif env_type == "tree":
        # Create a binary tree
        depth = env_kwargs["depth"]
        n = 2**(depth + 1) - 1
        
        # Create transition matrix
        P = create_binary_tree_transitions(depth=depth)
        
        # Generate costs using uniform distribution
        q = generate_costs(n, distribution="uniform", low=0.0, high=1.0)
        
        # Set leaf nodes as goal states
        #goal_states = [i for i in range(n) if 2*i+1 >= n]  # States with no children
        #goal_states = [0] #origin (only for bidirectional tree)
        goal_states = [n-1] # A leaf node
        q = set_goal_states(q, goal_states)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return P, q, goal_states


def compute_cross_entropy(p, q):
    """
    Compute the cross-entropy between two probability distributions.
    
    H(p,q) = -sum(p(x) * log(q(x)))
    
    Parameters
    ----------
    p : numpy.ndarray
        First probability distribution.
    q : numpy.ndarray
        Second probability distribution.
        
    Returns
    -------
    float
        Cross-entropy between p and q.
    """
    # Add small epsilon to avoid log(0)
    epsilon = 1e-10
    q_safe = np.maximum(q, epsilon)
    return -np.sum(p * np.log(q_safe))


def evaluate_difference(solution1, solution2, measure_type):
    """
    Evaluate the difference between two LSMDP solutions.
    
    Parameters
    ----------
    solution1 : tuple
        First solution as a tuple (z, v, pi).
    solution2 : tuple
        Second solution as a tuple (z, v, pi).
    measure_type : str
        Type of difference measure to use. Options include:
        - "L1": L1 norm (absolute difference)
        - "L2": L2 norm (Euclidean distance)
        - "KL": KL divergence (for policies)
        - "CE": Cross entropy (for policies)
        
    Returns
    -------
    dict
        Dictionary of differences for each component (z, v, pi).
        
    Raises
    ------
    ValueError
        If an unknown measure type is specified.
    """
    z1, v1, pi1 = solution1
    z2, v2, pi2 = solution2
    n = len(z2)  # Use z2 (optimal solution) as reference for size
    differences = {}
    
    # Handle case where z1 or v1 is None (for probability space and majority votes)
    if z1 is None or v1 is None:
        differences["z"] = float('nan')
        differences["v"] = float('nan')
    else:
        # Normalize desirability functions to have the same scale
        z1_norm = z1 / np.sum(z1)
        z2_norm = z2 / np.sum(z2)
        
        # Compute difference for desirability function
        if measure_type == "L1":
            differences["z"] = np.sum(np.abs(z1_norm - z2_norm))
        elif measure_type == "L2":
            differences["z"] = np.sqrt(np.sum((z1_norm - z2_norm) ** 2))
        elif measure_type == "KL":
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            z1_norm_safe = np.maximum(z1_norm, epsilon)
            z2_norm_safe = np.maximum(z2_norm, epsilon)
            differences["z"] = np.sum(kl_div(z1_norm_safe, z2_norm_safe))
        elif measure_type == "CE":
            # Use cross entropy for desirability function
            differences["z"] = compute_cross_entropy(z1_norm, z2_norm)
        else:
            raise ValueError(f"Unknown measure type: {measure_type}")
        
        # Compute difference for value function
        # Adjust for potential constant offset in value functions
        v_offset = np.mean(v1 - v2)
        v1_adjusted = v1 - v_offset
        
        if measure_type == "L1":
            differences["v"] = np.sum(np.abs(v1_adjusted - v2)) / n
        elif measure_type == "L2":
            differences["v"] = np.sqrt(np.sum((v1_adjusted - v2) ** 2)) / n
        elif measure_type in ["KL", "CE"]:
            # KL divergence and cross entropy don't make sense for value functions
            # Use L2 norm instead
            differences["v"] = np.sqrt(np.sum((v1_adjusted - v2) ** 2)) / n
    
    # Compute difference for policy
    policy_diff = 0.0
    count = 0
    
    for s in range(n):
        if s not in pi1 or s not in pi2:
            continue
        
        # Get all possible next states from both policies
        next_states = set(pi1[s].keys()) | set(pi2[s].keys())
        
        if not next_states:
            continue
        
        # Create probability distributions over next states
        p1 = np.zeros(len(next_states))
        p2 = np.zeros(len(next_states))
        
        for i, s_next in enumerate(next_states):
            p1[i] = pi1[s].get(s_next, 0.0)
            p2[i] = pi2[s].get(s_next, 0.0)
        
        # Normalize if needed
        if np.sum(p1) > 0:
            p1 = p1 / np.sum(p1)
        if np.sum(p2) > 0:
            p2 = p2 / np.sum(p2)
        
        # Compute difference based on measure type
        if measure_type == "L1":
            policy_diff += np.sum(np.abs(p1 - p2))
        elif measure_type == "L2":
            policy_diff += np.sqrt(np.sum((p1 - p2) ** 2))
        elif measure_type == "KL":
            # Add small epsilon to avoid division by zero
            epsilon = 1e-10
            p1_safe = np.maximum(p1, epsilon)
            p2_safe = np.maximum(p2, epsilon)
            policy_diff += np.sum(kl_div(p1_safe, p2_safe))
        elif measure_type == "CE":
            # Use cross entropy for policy
            policy_diff += compute_cross_entropy(p1, p2)
        
        count += 1
    
    # Average policy difference over all states
    if count > 0:
        differences["pi"] = policy_diff / count
    else:
        differences["pi"] = 0.0
    
    return differences


def main():
    """
    Main function for the policy aggregation experiment.
    """
    # Parse command-line arguments
    args = parse_args()
    
    # Set random seed for reproducibility
    np.random.seed(args.seed)
    
    # Initialize wandb if not disabled
    if not args.no_wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config={
                "env_type": args.env_type,
                "state_space_size": args.state_space_size,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "evaluation_methods": args.evaluation_methods,
                "seed": args.seed
            }
        )
    
    # Parse environment arguments
    env_kwargs = parse_env_args(args)
    
    # Create environment
    env = make_env(env_kwargs)
    P, q, goal_states = env
    
    # Solve LSMDP for the original environment (optimal solution)
    z_opt, v_opt, pi_opt = solve_lsmdp(P, q, args.alpha, args.gamma, goal_states)
    word_lengthes = [int(x) for x in args.word_lengthes]

    def _run_loop(ell):
        # Generate permutations
        n = len(z_opt)  # Get the actual state space size
        k = 2**round(args.max_length/ell)
        permutations = generate_permutations(n, k, ell, seed=args.seed)
        
        
        # Generate permuted solutions
        permuted_solutions = []
        for perm in permutations:
            # Apply permutation to solution
            z_perm, v_perm, pi_perm = apply_permutation_to_solution(z_opt, v_opt, pi_opt, perm)
            permuted_solutions.append((z_perm, v_perm, pi_perm))
        
        # Aggregate solutions using different methods
        #print("\nAggregating solutions...")
        
        # 1. Convex combination in z space
        z_agg, v_agg, pi_agg = aggregate_z_space(permuted_solutions, args.alpha, args.gamma)

        # 1.1 Convex combination in z space with access to P 
        z_agg_P, v_agg_P, pi_agg_P = aggregate_z_space(permuted_solutions, args.alpha, args.gamma, env_P=P)
        # 2. Convex combination in probability space
        _, _, pi_prob_agg = aggregate_probability_space(permuted_solutions)
        
        # 3. Majority votes
        _, _, pi_vote_agg = aggregate_majority_votes(permuted_solutions)
        
        # Evaluate differences between aggregated solutions and optimal solution
        #print("\nEvaluating differences...")
        
        # Initialize results dictionary
        results = {}
        
        # Evaluate each aggregation method for each evaluation measure
        for method in args.evaluation_methods:
            # Z-space aggregation
            z_space_diff = evaluate_difference((z_agg, v_agg, pi_agg), (z_opt, v_opt, pi_opt), method)
            # Z-space aggregation with access to P
            z_space_P_diff = evaluate_difference((z_agg_P, v_agg_P, pi_agg_P), (z_opt, v_opt, pi_opt), method)
          
            
            # Probability space aggregation
            prob_space_diff = evaluate_difference((None, None, pi_prob_agg), (z_opt, v_opt, pi_opt), method)
            
            # Majority votes aggregation
            majority_votes_diff = evaluate_difference((None, None, pi_vote_agg), (z_opt, v_opt, pi_opt), method)
            
            # Store results
            results[f"z_space_{method}_z"] = z_space_diff["z"]
            results[f"z_space_{method}_v"] = z_space_diff["v"]
            results[f"z_space_{method}_pi"] = z_space_diff["pi"]

            results[f"z_space_P_{method}_z"] = z_space_diff["z"]
            results[f"z_space_P_{method}_v"] = z_space_diff["v"]
            results[f"z_space_P_{method}_pi"] = z_space_diff["pi"]

            
            results[f"prob_space_{method}_pi"] = prob_space_diff["pi"]
            
            results[f"majority_votes_{method}_pi"] = majority_votes_diff["pi"]
        
        # Log results to wandb
        if not args.no_wandb:
            wandb.log(results)
        else:
            # Print results
            print("\nResults:")
            print("\nZ-Space Aggregation:")
            for method in args.evaluation_methods:
                print(f"  {method} - z: {results[f'z_space_{method}_z']:.6f}, "
                    f"v: {results[f'z_space_{method}_v']:.6f}, "
                    f"pi: {results[f'z_space_{method}_pi']:.6f}")
            
            print("\nProbability Space Aggregation:")
            for method in args.evaluation_methods:
                print(f"  {method} - pi: {results[f'prob_space_{method}_pi']:.6f}")
            
            print("\nMajority Votes Aggregation:")
            for method in args.evaluation_methods:
                print(f"  {method} - pi: {results[f'majority_votes_{method}_pi']:.6f}")
    
    for ell in tqdm(word_lengthes):
        _run_loop(ell)        
    # Finish wandb run if not disabled
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
