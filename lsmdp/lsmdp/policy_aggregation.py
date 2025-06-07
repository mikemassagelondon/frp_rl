"""
Policy aggregation methods for Linearly Solvable Markov Decision Processes (LS-MDPs).

This module provides functions to aggregate multiple LSMDP solutions into a single solution
using different aggregation methods:
1. Convex combination in the z's space (desirability function)
2. Convex combination in the probability space (policies)
3. Majority votes of policies
"""

import numpy as np
from lsmdp.solver import compute_value_function, compute_optimal_policy


def aggregate_z_space(solutions, alpha, gamma, env_P=None):
    """
    Aggregate solutions by taking the mean of desirability functions.
    
    Parameters
    ----------
    solutions : list
        List of tuples (z, V, pi) representing LSMDP solutions.
    alpha : float
        Control cost parameter (temperature).
    gamma : float
        Discount factor.
    env_P: np.ndarray
        (Optional)Transition Matrix of enviroment. 
        
    Returns
    -------
    tuple
        Aggregated solution (z', V', pi').
    """
    if not solutions:
        raise ValueError("Empty solutions list provided")
    
    # Extract z functions from all solutions
    z_list = [sol[0] for sol in solutions]
    
    # Compute mean z function
    z_prime = np.mean(z_list, axis=0)
    
    # Compute value function from aggregated z
    V_prime = compute_value_function(z_prime, alpha, gamma)
    
    if env_P is None:
        # Get transition matrix from the first solution's policy
        # We need this to compute the optimal policy from z_prime
        # This assumes all solutions use the same transition matrix
        P = np.zeros((len(z_prime), len(z_prime)))
        pi_first = solutions[0][2]
        
        for s in pi_first:
            for s_next, prob in pi_first[s].items():
                # Reconstruct P from the policy and z
                # This is an approximation since we don't have direct access to P
                z_s = solutions[0][0][s]
                z_s_next = solutions[0][0][s_next]
                if z_s > 0:
                    P[s, s_next] = prob * z_s / z_s_next
    else:
        P = env_P        
    # Compute optimal policy from aggregated z
    pi_prime = compute_optimal_policy(P, z_prime)
    
    return z_prime, V_prime, pi_prime


def aggregate_probability_space(solutions):
    """
    Aggregate solutions by taking the mean of policies.
    
    Parameters
    ----------
    solutions : list
        List of tuples (z, V, pi) representing LSMDP solutions.
        
    Returns
    -------
    tuple
        Aggregated solution (None, None, pi').
    """
    if not solutions:
        raise ValueError("Empty solutions list provided")
    
    # Extract policies from all solutions
    pi_list = [sol[2] for sol in solutions]
    
    # Initialize aggregated policy
    pi_prime = {}
    
    # Get all states from the first policy
    # Assuming all policies have the same state space
    states = set(pi_list[0].keys())
    
    # For each state
    for s in states:
        pi_prime[s] = {}
        
        # Get all possible next states from all policies
        next_states = set()
        for pi in pi_list:
            if s in pi:
                next_states.update(pi[s].keys())
        
        # For each possible next state
        for s_next in next_states:
            # Compute average probability
            prob_sum = 0.0
            count = 0
            
            for pi in pi_list:
                if s in pi and s_next in pi[s]:
                    prob_sum += pi[s][s_next]
                    count += 1
            
            if count > 0:
                pi_prime[s][s_next] = prob_sum / count
    
    # Normalize probabilities to ensure they sum to 1
    for s in pi_prime:
        prob_sum = sum(pi_prime[s].values())
        if prob_sum > 0:
            for s_next in pi_prime[s]:
                pi_prime[s][s_next] /= prob_sum
    
    # Return None for z and V since they are not computed
    return None, None, pi_prime


def aggregate_majority_votes(solutions):
    """
    Aggregate solutions by taking majority votes of policies.
    
    Parameters
    ----------
    solutions : list
        List of tuples (z, V, pi) representing LSMDP solutions.
        
    Returns
    -------
    tuple
        Aggregated solution (None, None, pi').
    """
    if not solutions:
        raise ValueError("Empty solutions list provided")
    
    # Extract policies from all solutions
    pi_list = [sol[2] for sol in solutions]
    m = len(pi_list)  # Number of policies
    
    # Initialize aggregated policy
    pi_prime = {}
    
    # Get all states from the first policy
    # Assuming all policies have the same state space
    states = set(pi_list[0].keys())
    
    # For each state
    for s in states:
        pi_prime[s] = {}
        
        # Get all possible next states from all policies
        next_states = set()
        for pi in pi_list:
            if s in pi:
                next_states.update(pi[s].keys())
        
        if not next_states:
            continue
        
        # Find the most likely next state for each policy
        T = {}  # T(i, k) = argmax_j pi_k(i, j)
        for k, pi in enumerate(pi_list):
            if s in pi and pi[s]:
                T[k] = max(pi[s].items(), key=lambda x: x[1])[0]
        
        # Count votes for each next state
        N = {}  # N(i, j) = # { k | T(i, k) = j, k=1,2,...,m }
        for j in next_states:
            N[j] = sum(1 for k in T if T[k] == j)
        
        # Compute probabilistic majority vote
        for j in N:
            pi_prime[s][j] = N[j] / m
    
    # Return None for z and V since they are not computed
    return None, None, pi_prime
