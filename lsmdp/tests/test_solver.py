"""
Tests for the solver functions in the LS-MDP package.
"""

import numpy as np
import pytest
from lsmdp.solver import (
    solve_desirability,
    compute_value_function,
    compute_optimal_policy,
    solve_lsmdp
)


def create_simple_mdp():
    """Create a simple MDP for testing."""
    # 3x3 grid (9 states)
    # State 8 (bottom-right) is the goal
    n = 9
    
    # Create transition matrix (equal probability to adjacent states)
    P = np.zeros((n, n))
    
    # Define transitions for each state
    # State 0 (top-left)
    P[0, 1] = 0.5  # right
    P[0, 3] = 0.5  # down
    
    # State 1 (top-middle)
    P[1, 0] = 0.33  # left
    P[1, 2] = 0.33  # right
    P[1, 4] = 0.34  # down
    
    # State 2 (top-right)
    P[2, 1] = 0.5  # left
    P[2, 5] = 0.5  # down
    
    # State 3 (middle-left)
    P[3, 0] = 0.33  # up
    P[3, 4] = 0.33  # right
    P[3, 6] = 0.34  # down
    
    # State 4 (center)
    P[4, 1] = 0.25  # up
    P[4, 3] = 0.25  # left
    P[4, 5] = 0.25  # right
    P[4, 7] = 0.25  # down
    
    # State 5 (middle-right)
    P[5, 2] = 0.33  # up
    P[5, 4] = 0.33  # left
    P[5, 8] = 0.34  # down
    
    # State 6 (bottom-left)
    P[6, 3] = 0.5  # up
    P[6, 7] = 0.5  # right
    
    # State 7 (bottom-middle)
    P[7, 4] = 0.33  # up
    P[7, 6] = 0.33  # left
    P[7, 8] = 0.34  # right
    
    # State 8 (bottom-right, goal)
    P[8, 8] = 1.0  # absorbing
    
    # Create cost function (all 1.0 except goal state)
    q = np.ones(n)
    q[8] = 0.0  # Goal state has zero cost
    
    return P, q, [8]  # Return transition matrix, cost function, and goal states


def test_solve_desirability_shape():
    """Test that the desirability function has the correct shape."""
    P, q, goal_states = create_simple_mdp()
    n = P.shape[0]
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    assert z.shape == (n,), f"Expected shape ({n},), got {z.shape}"


def test_solve_desirability_goal_states():
    """Test that goal states have desirability 1.0."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    for g in goal_states:
        assert np.isclose(z[g], 1.0), f"Goal state {g} should have desirability 1.0, got {z[g]}"


def test_solve_desirability_non_negative():
    """Test that the desirability function is non-negative."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    assert np.all(z >= 0), "Desirability function should be non-negative"


def test_solve_desirability_equation():
    """Test that the desirability function satisfies the linear equation."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    
    # Check that z satisfies the equation: z(s) = exp(-gamma/alpha * q(s)) * sum_{s'} P[s,s'] * z(s')
    for s in range(P.shape[0]):
        if s in goal_states:
            continue  # Skip goal states
        
        # Calculate the right-hand side of the equation
        w = np.exp(-(gamma / alpha) * q[s])
        expected = w * np.sum(P[s] * z)
        
        assert np.isclose(z[s], expected, rtol=1e-5), \
            f"Desirability equation not satisfied for state {s}: {z[s]} != {expected}"


def test_compute_value_function_shape():
    """Test that the value function has the correct shape."""
    P, q, goal_states = create_simple_mdp()
    n = P.shape[0]
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    V = compute_value_function(z, alpha, gamma)
    
    assert V.shape == (n,), f"Expected shape ({n},), got {V.shape}"


def test_compute_value_function_goal_states():
    """Test that goal states have value 0.0."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    V = compute_value_function(z, alpha, gamma)
    
    for g in goal_states:
        assert np.isclose(V[g], 0.0), f"Goal state {g} should have value 0.0, got {V[g]}"


def test_compute_value_function_formula():
    """Test that the value function is correctly computed from the desirability function."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    V = compute_value_function(z, alpha, gamma)
    
    # Check that V(s) = -alpha/gamma * ln(z(s))
    for s in range(P.shape[0]):
        if z[s] <= 0:
            continue  # Skip states with zero desirability
        
        expected = -(alpha / gamma) * np.log(z[s])
        assert np.isclose(V[s], expected), \
            f"Value function formula not satisfied for state {s}: {V[s]} != {expected}"


def test_compute_optimal_policy_structure():
    """Test that the optimal policy has the correct structure."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    policy = compute_optimal_policy(P, z)
    
    # Check that the policy is a dictionary mapping states to dictionaries of next states and probabilities
    assert isinstance(policy, dict), "Policy should be a dictionary"
    
    for s in range(P.shape[0]):
        assert s in policy, f"State {s} should be in the policy"
        assert isinstance(policy[s], dict), f"Policy for state {s} should be a dictionary"
        
        # Check that probabilities sum to 1 (or 0 if no transitions)
        if policy[s]:
            prob_sum = sum(policy[s].values())
            assert np.isclose(prob_sum, 1.0), \
                f"Probabilities for state {s} should sum to 1.0, got {prob_sum}"


def test_compute_optimal_policy_formula():
    """Test that the optimal policy is correctly computed from the desirability function."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    z = solve_desirability(P, q, alpha, gamma, goal_states)
    policy = compute_optimal_policy(P, z)
    
    # Check that π*(s'|s) = P[s,s'] * z(s') / sum_{s''} P[s,s''] * z(s'')
    for s in range(P.shape[0]):
        next_states = np.where(P[s] > 0)[0]
        if len(next_states) == 0:
            continue  # Skip states with no transitions
        
        # Calculate denominator: sum_{s''} P[s,s''] * z(s'')
        denom = np.sum(P[s, next_states] * z[next_states])
        
        if denom <= 0:
            continue  # Skip states with zero denominator
        
        for s_next in next_states:
            if s_next in policy[s]:
                expected = (P[s, s_next] * z[s_next]) / denom
                assert np.isclose(policy[s][s_next], expected), \
                    f"Policy formula not satisfied for transition {s}->{s_next}: {policy[s][s_next]} != {expected}"


def test_solve_lsmdp():
    """Test that solve_lsmdp correctly computes the desirability, value function, and policy."""
    P, q, goal_states = create_simple_mdp()
    alpha, gamma = 1.0, 0.9
    
    # Solve the MDP
    z, V, policy = solve_lsmdp(P, q, alpha, gamma, goal_states)
    
    # Check that the results are consistent with the individual functions
    z_expected = solve_desirability(P, q, alpha, gamma, goal_states)
    V_expected = compute_value_function(z_expected, alpha, gamma)
    policy_expected = compute_optimal_policy(P, z_expected)
    
    assert np.allclose(z, z_expected), "Desirability function from solve_lsmdp should match solve_desirability"
    assert np.allclose(V, V_expected), "Value function from solve_lsmdp should match compute_value_function"
    
    # Check that the policies are the same
    for s in range(P.shape[0]):
        for s_next in policy[s]:
            assert s_next in policy_expected[s], f"Next state {s_next} missing from expected policy for state {s}"
            assert np.isclose(policy[s][s_next], policy_expected[s][s_next]), \
                f"Policy probability mismatch for {s}->{s_next}: {policy[s][s_next]} != {policy_expected[s][s_next]}"
