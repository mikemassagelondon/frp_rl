"""
Test module for policy aggregation methods.

This module tests the policy aggregation methods implemented in the
lsmdp.policy_aggregation module.
"""

import numpy as np
import unittest
from lsmdp.transition_models import create_lattice_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from lsmdp.meta_lsmdp import generate_permutations, apply_permutation_to_solution
from lsmdp.policy_aggregation import (
    aggregate_z_space,
    aggregate_probability_space,
    aggregate_majority_votes
)


class TestPolicyAggregation(unittest.TestCase):
    """Test case for policy aggregation methods."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a small grid world
        self.n = 16  # 4x4 grid
        self.P = create_lattice_transitions(self.n)
        self.q = generate_costs(self.n, distribution="uniform", low=0.0, high=1.0, seed=42)
        self.goal_states = [self.n - 1]  # Bottom-right corner
        self.q = set_goal_states(self.q, self.goal_states)
        
        # MDP parameters
        self.alpha = 1.0
        self.gamma = 0.9
        
        # Solve the original MDP
        self.z_opt, self.v_opt, self.pi_opt = solve_lsmdp(
            self.P, self.q, self.alpha, self.gamma, self.goal_states
        )
        
        # Generate permutations
        self.permutations = generate_permutations(self.n, k=2, l=1, seed=42)
        
        # Generate permuted solutions
        self.permuted_solutions = []
        for perm in self.permutations[:3]:  # Use only 3 permutations for testing
            z_perm, v_perm, pi_perm = apply_permutation_to_solution(
                self.z_opt, self.v_opt, self.pi_opt, perm
            )
            self.permuted_solutions.append((z_perm, v_perm, pi_perm))
    
    def test_aggregate_z_space(self):
        """Test z-space aggregation."""
        # Aggregate solutions
        z_agg, v_agg, pi_agg = aggregate_z_space(
            self.permuted_solutions, self.alpha, self.gamma
        )
        
        # Check that the aggregated z is the mean of the permuted z's
        z_mean = np.mean([sol[0] for sol in self.permuted_solutions], axis=0)
        np.testing.assert_allclose(z_agg, z_mean, rtol=1e-10)
        
        # Check that v_agg is computed from z_agg
        v_expected = -(self.alpha / self.gamma) * np.log(np.maximum(z_agg, 1e-10))
        np.testing.assert_allclose(v_agg, v_expected, rtol=1e-10)
        
        # Check that pi_agg is a valid policy
        for s in range(self.n):
            if s in pi_agg:
                # Check that probabilities sum to approximately 1
                prob_sum = sum(pi_agg[s].values())
                self.assertAlmostEqual(prob_sum, 1.0, places=6)
    
    def test_aggregate_probability_space(self):
        """Test probability space aggregation."""
        # Aggregate solutions
        _, _, pi_agg = aggregate_probability_space(self.permuted_solutions)
        
        # Check that pi_agg is a valid policy
        for s in range(self.n):
            if s in pi_agg:
                # Check that probabilities sum to approximately 1
                prob_sum = sum(pi_agg[s].values())
                self.assertAlmostEqual(prob_sum, 1.0, places=6)
        
        # Check that the aggregated policy is the mean of the permuted policies
        # for a sample state
        s = 0  # Check the first state
        if s in pi_agg:
            # Get all possible next states
            next_states = set()
            for sol in self.permuted_solutions:
                pi = sol[2]
                if s in pi:
                    next_states.update(pi[s].keys())
            
            # Compute expected probabilities
            expected_probs = {}
            for s_next in next_states:
                prob_sum = 0.0
                count = 0
                for sol in self.permuted_solutions:
                    pi = sol[2]
                    if s in pi and s_next in pi[s]:
                        prob_sum += pi[s][s_next]
                        count += 1
                if count > 0:
                    expected_probs[s_next] = prob_sum / count
            
            # Normalize expected probabilities
            prob_sum = sum(expected_probs.values())
            if prob_sum > 0:
                for s_next in expected_probs:
                    expected_probs[s_next] /= prob_sum
            
            # Compare with actual probabilities
            for s_next in expected_probs:
                if s_next in pi_agg[s]:
                    self.assertAlmostEqual(
                        pi_agg[s][s_next], expected_probs[s_next], places=6
                    )
    
    def test_aggregate_majority_votes(self):
        """Test majority votes aggregation."""
        # Aggregate solutions
        _, _, pi_agg = aggregate_majority_votes(self.permuted_solutions)
        
        # Check that pi_agg is a valid policy
        for s in range(self.n):
            if s in pi_agg:
                # Check that probabilities sum to approximately 1
                prob_sum = sum(pi_agg[s].values())
                self.assertAlmostEqual(prob_sum, 1.0, places=6)
        
        # Check that the aggregated policy follows the majority vote rule
        # for a sample state
        s = 0  # Check the first state
        if s in pi_agg:
            # Find the most likely next state for each policy
            T = {}
            for k, sol in enumerate(self.permuted_solutions):
                pi = sol[2]
                if s in pi and pi[s]:
                    T[k] = max(pi[s].items(), key=lambda x: x[1])[0]
            
            # Count votes for each next state
            N = {}
            next_states = set()
            for sol in self.permuted_solutions:
                pi = sol[2]
                if s in pi:
                    next_states.update(pi[s].keys())
            
            for j in next_states:
                N[j] = sum(1 for k in T if T[k] == j)
            
            # Compute expected probabilities
            m = len(self.permuted_solutions)
            expected_probs = {j: N[j] / m for j in N}
            
            # Compare with actual probabilities
            for s_next in expected_probs:
                if s_next in pi_agg[s]:
                    self.assertAlmostEqual(
                        pi_agg[s][s_next], expected_probs[s_next], places=6
                    )


if __name__ == "__main__":
    unittest.main()
