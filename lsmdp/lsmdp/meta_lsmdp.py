"""
Meta-LSMDP module for evaluating permutation differences in LSMDP solutions.

This module extends the LSMDP package to handle permutations of state spaces
and evaluate differences between solutions. It provides functions for:
- Generating random permutations of state spaces
- Applying permutations to LSMDP solutions
- Computing differences between original and permuted solutions
- Aggregating differences across permutations
- Evaluating the overall effect of permutations on LSMDP solutions
"""

import numpy as np
import itertools
from scipy.special import kl_div
from lsmdp.solver import solve_lsmdp


def generate_permutations(n, k, l, seed=None):
    """
    Generate k^l permutations of n states by taking the product of k random permutations.
    
    Parameters
    ----------
    n : int
        Number of states.
    k : int
        Number of independent random permutations.
    l : int
        Number of times the permutations will be combined.
    seed : int, optional
        Random seed for reproducibility, by default None.
        
    Returns
    -------
    list
        List of permutations, where each permutation is a numpy array of indices.
        
    Raises
    ------
    ValueError
        If k or l is less than 1.
    """
    if k < 1 or l < 1:
        raise ValueError("k and l must be at least 1")
    
    if seed is not None:
        np.random.seed(seed)
    
    # Generate k base permutations
    base_permutations = []
    for _ in range(k):
        perm = np.arange(n)
        np.random.shuffle(perm)
        base_permutations.append(perm)
    
    # If k=1 and l=1, return the identity permutation
    if k == 1 and l == 1:
        return [np.arange(n)]
    
    # Generate all possible combinations of l permutations from the k base permutations
    # This gives k^l total permutations
    permutation_indices = list(itertools.product(range(k), repeat=l))
    
    # Compose the permutations
    permutations = []
    for indices in permutation_indices:
        # Start with the identity permutation
        composed_perm = np.arange(n)
        
        # Apply each permutation in sequence
        for idx in indices:
            composed_perm = base_permutations[idx][composed_perm]
        
        permutations.append(composed_perm)
    
    return permutations


def apply_permutation_to_solution(z, V, policy, permutation):
    """
    Apply a permutation to an LSMDP solution.
    
    Parameters
    ----------
    z : numpy.ndarray
        Desirability function of shape (n,).
    V : numpy.ndarray
        Value function of shape (n,).
    policy : dict
        Optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    permutation : numpy.ndarray
        Permutation of state indices.
        
    Returns
    -------
    tuple
        A tuple containing:
        - z_prime (numpy.ndarray): Permuted desirability function.
        - V_prime (numpy.ndarray): Permuted value function.
        - policy_prime (dict): Permuted optimal policy.
    """
    n = len(z)
    
    # Create inverse permutation for mapping states
    inverse_perm = np.zeros(n, dtype=int)
    for i in range(n):
        inverse_perm[permutation[i]] = i
    
    # Apply permutation to desirability and value functions
    z_prime = z[inverse_perm]
    V_prime = V[inverse_perm]
    
    # Apply permutation to policy
    policy_prime = {}
    for s in range(n):
        # Map the state through the permutation
        s_perm = permutation[s]
        
        # Get the original policy for the permuted state
        if s_perm in policy:
            # Map each next state and probability
            policy_prime[s] = {}
            for s_next, prob in policy[s_perm].items():
                # Map the next state through the inverse permutation
                s_next_perm = inverse_perm[s_next]
                policy_prime[s][s_next_perm] = prob
        else:
            # If the state has no policy (e.g., absorbing state), create an empty dict
            policy_prime[s] = {}
    
    return z_prime, V_prime, policy_prime


def compute_difference(z, V, policy, z_prime, V_prime, policy_prime, measure_type="L2"):
    """
    Compute the difference between the original LSMDP solution and the permuted solution.
    
    Parameters
    ----------
    z : numpy.ndarray
        Original desirability function of shape (n,).
    V : numpy.ndarray
        Original value function of shape (n,).
    policy : dict
        Original optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    z_prime : numpy.ndarray
        Permuted desirability function of shape (n,).
    V_prime : numpy.ndarray
        Permuted value function of shape (n,).
    policy_prime : dict
        Permuted optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    measure_type : str, optional
        Type of difference measure to use, by default "L2". Options include:
        - "L1": L1 norm (absolute difference)
        - "L2": L2 norm (Euclidean distance)
        - "KL": KL divergence (for policies)
        - "Wasserstein": Wasserstein distance (if applicable)
        
    Returns
    -------
    dict
        Dictionary of differences for each component (z, V, policy).
        
    Raises
    ------
    ValueError
        If an unknown measure type is specified.
    """
    n = len(z)
    differences = {}
    
    # Normalize desirability functions to have the same scale
    z_norm = z / np.sum(z)
    z_prime_norm = z_prime / np.sum(z_prime)
    
    # Compute difference for desirability function
    if measure_type == "L1":
        differences["z"] = np.sum(np.abs(z_norm - z_prime_norm))
    elif measure_type == "L2":
        differences["z"] = np.sqrt(np.sum((z_norm - z_prime_norm) ** 2))
    elif measure_type == "KL":
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        z_norm_safe = np.maximum(z_norm, epsilon)
        z_prime_norm_safe = np.maximum(z_prime_norm, epsilon)
        differences["z"] = np.sum(kl_div(z_norm_safe, z_prime_norm_safe))
    elif measure_type == "Wasserstein":
        # For simplicity, we'll use a 1D approximation of Wasserstein distance
        # This is only valid if the states have a natural ordering
        z_cumsum = np.cumsum(z_norm)
        z_prime_cumsum = np.cumsum(z_prime_norm)
        differences["z"] = np.sum(np.abs(z_cumsum - z_prime_cumsum))
    else:
        raise ValueError(f"Unknown measure type: {measure_type}")
    
    # Compute difference for value function
    # Adjust for potential constant offset in value functions
    V_offset = np.mean(V - V_prime)
    V_adjusted = V - V_offset
    
    if measure_type == "L1":
        differences["V"] = np.sum(np.abs(V_adjusted - V_prime)) / n
    elif measure_type == "L2":
        differences["V"] = np.sqrt(np.sum((V_adjusted - V_prime) ** 2)) / n
    elif measure_type == "KL":
        # KL divergence doesn't make sense for value functions
        # Use L2 norm instead
        differences["V"] = np.sqrt(np.sum((V_adjusted - V_prime) ** 2)) / n
    elif measure_type == "Wasserstein":
        # For simplicity, use L1 norm for value functions
        differences["V"] = np.sum(np.abs(V_adjusted - V_prime)) / n
    
    # Compute difference for policy
    policy_diff = 0.0
    count = 0
    
    for s in range(n):
        if s not in policy or s not in policy_prime:
            continue
        
        # Get all possible next states from both policies
        next_states = set(policy[s].keys()) | set(policy_prime[s].keys())
        
        if not next_states:
            continue
        
        # Create probability distributions over next states
        p1 = np.zeros(len(next_states))
        p2 = np.zeros(len(next_states))
        
        for i, s_next in enumerate(next_states):
            p1[i] = policy[s].get(s_next, 0.0)
            p2[i] = policy_prime[s].get(s_next, 0.0)
        
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
        elif measure_type == "Wasserstein":
            # For simplicity, use L1 norm for policies
            policy_diff += np.sum(np.abs(p1 - p2))
        
        count += 1
    
    # Average policy difference over all states
    if count > 0:
        differences["policy"] = policy_diff / count
    else:
        differences["policy"] = 0.0
    
    return differences


def aggregate_differences(differences, aggregation_type="average"):
    """
    Aggregate differences across permutations.
    
    Parameters
    ----------
    differences : list
        List of difference dictionaries, one for each permutation.
    aggregation_type : str, optional
        Type of aggregation to use, by default "average". Options include:
        - "average": Compute the mean difference
        - "max": Compute the maximum difference
        - "std": Compute the standard deviation of differences
        
    Returns
    -------
    dict
        Dictionary of aggregated differences for each component.
        
    Raises
    ------
    ValueError
        If an unknown aggregation type is specified.
    """
    if not differences:
        return {"z": 0.0, "V": 0.0, "policy": 0.0}
    
    # Extract values for each component
    z_diffs = [d["z"] for d in differences]
    V_diffs = [d["V"] for d in differences]
    policy_diffs = [d["policy"] for d in differences]
    
    # Aggregate based on type
    if aggregation_type == "average":
        return {
            "z": np.mean(z_diffs),
            "V": np.mean(V_diffs),
            "policy": np.mean(policy_diffs)
        }
    elif aggregation_type == "max":
        return {
            "z": np.max(z_diffs),
            "V": np.max(V_diffs),
            "policy": np.max(policy_diffs)
        }
    elif aggregation_type == "std":
        return {
            "z": np.std(z_diffs),
            "V": np.std(V_diffs),
            "policy": np.std(policy_diffs)
        }
    else:
        raise ValueError(f"Unknown aggregation type: {aggregation_type}")


def evaluate_permutation_difference(P, q, alpha, gamma, goal_states=None, 
                                   k=3, l=2, measure_type="L2", 
                                   aggregation_type="average", seed=None):
    """
    Evaluate the difference between LSMDP solutions under random permutations.
    
    Parameters
    ----------
    P : numpy.ndarray
        Transition probability matrix of shape (n, n), where P[i,j] is the 
        probability of transitioning from state i to state j.
    q : numpy.ndarray
        Cost function of shape (n,), where q[i] is the cost of being in state i.
    alpha : float
        Control cost parameter (temperature).
    gamma : float
        Discount factor.
    goal_states : list or numpy.ndarray, optional
        Indices of goal states, by default None. If provided, these states will
        have their desirability fixed to 1.0 and will be made absorbing in the
        transition matrix.
    k : int, optional
        Number of independent random permutations, by default 3.
    l : int, optional
        Number of times the permutations will be combined, by default 2.
    measure_type : str, optional
        Type of difference measure to use, by default "L2".
    aggregation_type : str, optional
        Type of aggregation to use, by default "average".
    seed : int, optional
        Random seed for reproducibility, by default None.
        
    Returns
    -------
    dict
        Dictionary of evaluation results, including:
        - "average": Average difference for each component
        - "max": Maximum difference for each component
        - "std": Standard deviation of differences for each component
    """
    n = P.shape[0]
    
    # Solve the original LSMDP
    z, V, policy = solve_lsmdp(P, q, alpha, gamma, goal_states)
    
    # Generate permutations
    permutations = generate_permutations(n, k, l, seed)
    
    # Compute differences for each permutation
    all_differences = []
    for perm in permutations:
        # Apply permutation to solution
        z_prime, V_prime, policy_prime = apply_permutation_to_solution(z, V, policy, perm)
        
        # Compute difference
        diff = compute_difference(z, V, policy, z_prime, V_prime, policy_prime, measure_type)
        all_differences.append(diff)
    
    # Aggregate differences
    results = {
        "average": aggregate_differences(all_differences, "average"),
        "max": aggregate_differences(all_differences, "max"),
        "std": aggregate_differences(all_differences, "std")
    }
    
    return results
