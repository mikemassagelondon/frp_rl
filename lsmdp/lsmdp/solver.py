"""
Solver functions for Linearly Solvable Markov Decision Processes (LS-MDPs).

This module provides functions to solve LS-MDPs analytically, computing
the desirability function, optimal value function, and optimal policy.
"""

import numpy as np


def solve_desirability(P, q, alpha, gamma, goal_states=None):
    """
    Solve for the desirability function z*(s) in a Linearly Solvable MDP.
    
    The desirability function satisfies the linear equation:
    z*(s) = exp(-gamma/alpha * q(s)) * sum_{s'} p(s'|s) * z*(s')
    
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
        
    Returns
    -------
    numpy.ndarray
        Desirability function z*(s) of shape (n,).
        
    Notes
    -----
    This function solves the linear system of equations for the desirability function.
    Goal states are treated as boundary conditions with z*(s) = 1.0.
    """
    n = P.shape[0]
    
    # Make a copy of P and q to avoid modifying the originals
    P_copy = P.copy()
    q_copy = q.copy()
    
    # Initialize desirability function
    z = np.zeros(n)
    
    # Handle goal states
    if goal_states is not None:
        goal_states = set(goal_states)
        for g in goal_states:
            z[g] = 1.0
            q_copy[g] = 0.0
            # Make goal absorbing in transition matrix
            P_copy[g, :] = 0.0
            P_copy[g, g] = 1.0
    else:
        goal_states = set()
    
    # Set up linear equations for non-goal states
    non_goals = [s for s in range(n) if s not in goal_states]
    
    if not non_goals:
        # All states are goals, return all ones
        return np.ones(n)
    
    # Create the linear system A * z_non = b
    A = np.eye(len(non_goals))
    b = np.zeros(len(non_goals))
    
    # Map from state index to index in the non_goals list
    idx_map = {s: idx for idx, s in enumerate(non_goals)}
    
    for s in non_goals:
        i = idx_map[s]
        # Calculate the weight for state s
        w = np.exp(-(gamma / alpha) * q_copy[s])
        
        # Sum over transitions to non-goal states (unknowns)
        for j in non_goals:
            A[i, idx_map[j]] -= w * P_copy[s, j]
        
        # Sum over transitions to goal states (known z=1)
        goal_prob_sum = 0.0
        for g in goal_states:
            goal_prob_sum += w * P_copy[s, g] * 1.0  # z(g)=1
        b[i] = goal_prob_sum
    
    # Solve the linear system
    try:
        z_non = np.linalg.solve(A, b)
        for s in non_goals:
            z[s] = z_non[idx_map[s]]
    except np.linalg.LinAlgError:
        # If the system is singular, try least squares solution
        z_non, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
        for s in non_goals:
            z[s] = z_non[idx_map[s]]
    
    return z


def compute_value_function(z, alpha, gamma):
    """
    Compute the optimal value function V*(s) from the desirability function z*(s).
    
    The optimal value function is given by:
    V*(s) = -alpha/gamma * ln(z*(s))
    
    Parameters
    ----------
    z : numpy.ndarray
        Desirability function of shape (n,).
    alpha : float
        Control cost parameter (temperature).
    gamma : float
        Discount factor.
        
    Returns
    -------
    numpy.ndarray
        Optimal value function V*(s) of shape (n,).
    """
    # Handle zero or negative desirability values
    z_safe = np.maximum(z, 1e-10)
    
    # Compute value function
    V = -(alpha / gamma) * np.log(z_safe)
    
    # Set value to infinity for states with zero desirability
    V[z <= 0] = np.inf
    
    return V


def compute_optimal_policy(P, z):
    """
    Compute the optimal policy π*(s'|s) from the desirability function z*(s).
    
    The optimal policy is given by:
    π*(s'|s) = p(s'|s) * z*(s') / sum_{s''} p(s''|s) * z*(s'')
    
    Parameters
    ----------
    P : numpy.ndarray
        Transition probability matrix of shape (n, n), where P[i,j] is the 
        probability of transitioning from state i to state j.
    z : numpy.ndarray
        Desirability function of shape (n,).
        
    Returns
    -------
    dict
        Optimal policy as a dictionary mapping each state to a dictionary of
        next states and their probabilities.
    """
    n = P.shape[0]
    policy = {}
    
    for s in range(n):
        # Find states with non-zero transition probability
        next_states = np.where(P[s] > 0)[0]
        
        if len(next_states) == 0:
            # No available transitions
            policy[s] = {}
            continue
        
        # Calculate denominator: sum_{s''} P[s,s''] * z[s'']
        denom = np.sum(P[s, next_states] * z[next_states])
        
        if denom <= 0:
            # Handle numerical issues
            policy[s] = {int(s): 1.0}  # Stay in the same state
            continue
        
        # Compute probability for each next state
        probs = {}
        for s_next in next_states:
            prob = (P[s, s_next] * z[s_next]) / denom
            if prob > 0:  # Only include non-zero probabilities
                probs[int(s_next)] = float(prob)
        
        policy[s] = probs
    
    return policy


def solve_lsmdp(P, q, alpha, gamma, goal_states=None):
    """
    Solve a Linearly Solvable MDP completely.
    
    This function computes the desirability function, optimal value function,
    and optimal policy for an LS-MDP.
    
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
        
    Returns
    -------
    tuple
        A tuple containing:
        - z (numpy.ndarray): Desirability function z*(s).
        - V (numpy.ndarray): Optimal value function V*(s).
        - policy (dict): Optimal policy π*(s'|s).
    """
    # Solve for desirability function
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    
    # Compute optimal value function
    V = compute_value_function(z, alpha, gamma)
    
    # Compute optimal policy
    policy = compute_optimal_policy(P, z)
    
    return z, V, policy
