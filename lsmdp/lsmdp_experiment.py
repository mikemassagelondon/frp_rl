"""
LSMDP experiment module.

This module implements an experiment to evaluate the effect of permutations
on LSMDP solutions using different evaluation methods.
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
        description="Evaluate the effect of permutations on LSMDP solutions."
    )
    
    # Environment selection
    parser.add_argument("--env_type", type=str, choices=["lattice", "tree"], default="lattice",
                        help="Type of environment to evaluate (default: lattice)")
    
    # State space size
    parser.add_argument("--state_space_size", type=int, default=64,
                        help="Size of the state space (default:64)")
    
    # Permutation parameters
    parser.add_argument("--max_length", type=int, default=8,
                        help="Maximum length for permutation combinations (default: 8)")
    
    # MDP parameters
    parser.add_argument("--alpha", type=float, default=1.0,
                        help="Control cost parameter (temperature) (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=0.9,
                        help="Discount factor (default: 0.9)")
    
    # Other parameters
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--no_wandb", action="store_true",
                        help="Do not use wandb for logging (default: False)")
    parser.add_argument("--wandb-project", type=str, default="lsmdp_experiment",
                        help="wandb project name (default: lsmdp_experiment)")
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
        #goal_states = []
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
        goal_states = [i for i in range(n) if 2*i+1 >= n]  # States with no children
        #goal_states = [0] # origin
        #goal_states = []
        q = set_goal_states(q, goal_states)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    return P, q, goal_states


def lsmdp_solution(env, alpha, gamma):
    """
    Solve an LSMDP.
    
    Parameters
    ----------
    env : tuple
        A tuple containing:
        - P (numpy.ndarray): Transition probability matrix.
        - q (numpy.ndarray): Cost function.
        - goal_states (list): List of goal states.
    alpha : float
        Control cost parameter (temperature).
    gamma : float
        Discount factor.
        
    Returns
    -------
    tuple
        A tuple containing:
        - z (numpy.ndarray): Desirability function.
        - v (numpy.ndarray): Value function.
        - pi (dict): Optimal policy.
    """
    P, q, goal_states = env
    return solve_lsmdp(P, q, alpha, gamma, goal_states)


def new_solution(z, v, pi, permutation):
    """
    Apply a permutation to an LSMDP solution.
    
    Parameters
    ----------
    z : numpy.ndarray
        Desirability function of shape (n,).
    v : numpy.ndarray
        Value function of shape (n,).
    pi : dict
        Optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    permutation : numpy.ndarray
        Permutation of state indices.
        
    Returns
    -------
    tuple
        A tuple containing:
        - z_prime (numpy.ndarray): Permuted desirability function.
        - v_prime (numpy.ndarray): Permuted value function.
        - pi_prime (dict): Permuted optimal policy.
    """
    return apply_permutation_to_solution(z, v, pi, permutation)


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
    n = len(z1)
    differences = {}
    
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


def aggregate_each_method(results, aggregation_method="average"):
    """
    Aggregate results across permutations for each evaluation method.
    
    Parameters
    ----------
    results : dict
        Dictionary of results for each evaluation method.
        Each method has a list of difference dictionaries, one for each permutation.
    aggregation_method : str, optional
        Method to use for aggregation, by default "average".
        
    Returns
    -------
    dict
        Dictionary of aggregated results for each method and component.
    """
    aggregated = {}
    
    for method, method_results in results.items():
        # Extract values for each component
        z_diffs = [r["z"] for r in method_results]
        v_diffs = [r["v"] for r in method_results]
        pi_diffs = [r["pi"] for r in method_results]
        
        # Aggregate based on method
        if aggregation_method == "average":
            aggregated[f"{method}_z"] = np.mean(z_diffs)
            aggregated[f"{method}_v"] = np.mean(v_diffs)
            aggregated[f"{method}_pi"] = np.mean(pi_diffs)
        elif aggregation_method == "max":
            aggregated[f"{method}_z"] = np.max(z_diffs)
            aggregated[f"{method}_v"] = np.max(v_diffs)
            aggregated[f"{method}_pi"] = np.max(pi_diffs)
        elif aggregation_method == "std":
            aggregated[f"{method}_z"] = np.std(z_diffs)
            aggregated[f"{method}_v"] = np.std(v_diffs)
            aggregated[f"{method}_pi"] = np.std(pi_diffs)
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")
    
    return aggregated


def main():
    """
    Main function for the LSMDP experiment.
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
                "max_length": args.max_length,
                "alpha": args.alpha,
                "gamma": args.gamma,
                "seed": args.seed
            }
        )
    
    # Parse environment arguments
    env_kwargs = parse_env_args(args)
    
    # Create environment
    env = make_env(env_kwargs)
    
    # Solve LSMDP for the original environment
    z, v, pi = lsmdp_solution(env, args.alpha, args.gamma)
    
    # Define list of lengths and evaluation methods
    list_length = [1, 2, 4, 8]
    evaluation_methods = ["L1", "L2", "KL", "CE"]
    
    # Run experiment for each length
    for l in tqdm(list_length):
        n = len(z)  # Get the actual state space size
        k = 2**round(args.max_length/l)
        permutations = generate_permutations(n,k,l,seed=args.seed)
        
        # Initialize results dictionary
        results = {method: [] for method in evaluation_methods}
        
        # Evaluate each permutation
        for permutation in permutations:
            # Apply permutation to solution
            z1, v1, pi1 = new_solution(z, v, pi, permutation)
            
            # Evaluate difference for each method
            for method in evaluation_methods:
                result = evaluate_difference((z1, v1, pi1), (z, v, pi), method)
                results[method].append(result)
        
        # Aggregate results
        logging_results = aggregate_each_method(results, aggregation_method="average")
        
        # Add length information to logging results
        logging_results["length"] = l
        
        # Log results to wandb
        if not args.no_wandb:
            wandb.log(logging_results)
        
        # Print results
        if args.no_wandb:
            print(f"\nResults for length {l}:")
            for key, value in logging_results.items():
                if key != "length":
                    print(f"  {key}: {value:.6f}")
    # Finish wandb run if not disabled
    if not args.no_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
