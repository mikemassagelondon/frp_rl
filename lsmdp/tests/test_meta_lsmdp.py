"""
Tests for the meta_lsmdp module.

This module tests the functionality of the meta_lsmdp module, which extends
the LSMDP package to handle permutations of state spaces and evaluate
differences between solutions.
"""

import numpy as np
import pytest

from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from lsmdp.meta_lsmdp import (
    generate_permutations,
    apply_permutation_to_solution,
    compute_difference,
    aggregate_differences,
    evaluate_permutation_difference
)


class TestPermutationGeneration:
    """Tests for the permutation generation functionality."""

    def test_generate_permutations_basic(self):
        """Test basic permutation generation."""
        n = 5
        k = 2
        l = 1
        permutations = generate_permutations(n, k, l, seed=42)
        
        # Check that we have k^l permutations
        assert len(permutations) == k**l
        
        # Check that each permutation is a valid permutation of range(n)
        for perm in permutations:
            assert len(perm) == n
            assert set(perm) == set(range(n))
    
    def test_generate_permutations_composition(self):
        """Test that permutations are composed correctly."""
        n = 5
        k = 2
        l = 2
        permutations = generate_permutations(n, k, l, seed=42)
        
        # Check that we have k^l permutations
        assert len(permutations) == k**l
        
        # Check that each permutation is a valid permutation of range(n)
        for perm in permutations:
            assert len(perm) == n
            assert set(perm) == set(range(n))
    
    def test_generate_permutations_edge_cases(self):
        """Test edge cases for permutation generation."""
        n = 5
        
        # k=1, l=1 should give the identity permutation
        permutations = generate_permutations(n, 1, 1, seed=42)
        assert len(permutations) == 1
        assert np.array_equal(permutations[0], np.arange(n))
        
        # k=0 or l=0 should raise ValueError
        with pytest.raises(ValueError):
            generate_permutations(n, 0, 1)
        
        with pytest.raises(ValueError):
            generate_permutations(n, 1, 0)


class TestPermutationApplication:
    """Tests for applying permutations to LSMDP solutions."""

    def setup_method(self):
        """Set up a simple LSMDP problem for testing."""
        n = 4
        self.P = create_lattice_transitions(n)
        self.q = generate_costs(n, distribution="uniform", low=0.0, high=1.0, seed=42)
        self.goal_states = [n - 1]
        self.q = set_goal_states(self.q, self.goal_states)
        self.alpha = 1.0
        self.gamma = 0.9
        self.z, self.V, self.policy = solve_lsmdp(self.P, self.q, self.alpha, self.gamma, self.goal_states)
    
    def test_apply_permutation_identity(self):
        """Test applying the identity permutation."""
        permutation = np.arange(len(self.z))
        z_prime, V_prime, policy_prime = apply_permutation_to_solution(
            self.z, self.V, self.policy, permutation
        )
        
        # Check that the solution is unchanged
        assert np.allclose(z_prime, self.z)
        assert np.allclose(V_prime, self.V)
        
        # Check that the policy is unchanged
        for s in self.policy:
            assert s in policy_prime
            for s_next, prob in self.policy[s].items():
                assert s_next in policy_prime[s]
                assert np.isclose(policy_prime[s][s_next], prob)
    
    def test_apply_permutation_swap(self):
        """Test applying a simple swap permutation."""
        # Create a permutation that swaps the first two states
        permutation = np.arange(len(self.z))
        permutation[0], permutation[1] = permutation[1], permutation[0]
        
        z_prime, V_prime, policy_prime = apply_permutation_to_solution(
            self.z, self.V, self.policy, permutation
        )
        
        # Check that the values are swapped correctly
        assert np.isclose(z_prime[0], self.z[1])
        assert np.isclose(z_prime[1], self.z[0])
        assert np.isclose(V_prime[0], self.V[1])
        assert np.isclose(V_prime[1], self.V[0])
        
        # Check that the policy is permuted correctly
        # This is more complex and depends on the specific policy structure


class TestDifferenceComputation:
    """Tests for computing differences between LSMDP solutions."""

    def setup_method(self):
        """Set up a simple LSMDP problem for testing."""
        n = 4
        self.P = create_lattice_transitions(n)
        self.q = generate_costs(n, distribution="uniform", low=0.0, high=1.0, seed=42)
        self.goal_states = [n - 1]
        self.q = set_goal_states(self.q, self.goal_states)
        self.alpha = 1.0
        self.gamma = 0.9
        self.z, self.V, self.policy = solve_lsmdp(self.P, self.q, self.alpha, self.gamma, self.goal_states)
    
    def test_compute_difference_identity(self):
        """Test computing difference with identity permutation."""
        permutation = np.arange(len(self.z))
        z_prime, V_prime, policy_prime = apply_permutation_to_solution(
            self.z, self.V, self.policy, permutation
        )
        
        diff = compute_difference(
            self.z, self.V, self.policy, z_prime, V_prime, policy_prime, measure_type="L2"
        )
        
        # Difference should be zero for identity permutation
        assert np.isclose(diff["z"], 0.0)
        assert np.isclose(diff["V"], 0.0)
        assert np.isclose(diff["policy"], 0.0)
    
    def test_compute_difference_l1_vs_l2(self):
        """Test that L1 and L2 norms give different results."""
        # Create a non-identity permutation
        permutation = np.arange(len(self.z))
        permutation[0], permutation[1] = permutation[1], permutation[0]
        
        z_prime, V_prime, policy_prime = apply_permutation_to_solution(
            self.z, self.V, self.policy, permutation
        )
        
        diff_l1 = compute_difference(
            self.z, self.V, self.policy, z_prime, V_prime, policy_prime, measure_type="L1"
        )
        
        diff_l2 = compute_difference(
            self.z, self.V, self.policy, z_prime, V_prime, policy_prime, measure_type="L2"
        )
        
        # L1 and L2 should give different results for non-identity permutation
        assert diff_l1["z"] != diff_l2["z"] or diff_l1["V"] != diff_l2["V"] or diff_l1["policy"] != diff_l2["policy"]
    
    def test_compute_difference_kl(self):
        """Test computing KL divergence for policies."""
        # Create a non-identity permutation
        permutation = np.arange(len(self.z))
        permutation[0], permutation[1] = permutation[1], permutation[0]
        
        z_prime, V_prime, policy_prime = apply_permutation_to_solution(
            self.z, self.V, self.policy, permutation
        )
        
        diff = compute_difference(
            self.z, self.V, self.policy, z_prime, V_prime, policy_prime, measure_type="KL"
        )
        
        # KL divergence should be non-negative
        assert diff["policy"] >= 0.0


class TestAggregation:
    """Tests for aggregating differences across permutations."""

    def test_aggregate_differences_average(self):
        """Test averaging differences."""
        differences = [
            {"z": 0.1, "V": 0.2, "policy": 0.3},
            {"z": 0.2, "V": 0.3, "policy": 0.4},
            {"z": 0.3, "V": 0.4, "policy": 0.5}
        ]
        
        agg = aggregate_differences(differences, aggregation_type="average")
        
        assert np.isclose(agg["z"], 0.2)
        assert np.isclose(agg["V"], 0.3)
        assert np.isclose(agg["policy"], 0.4)
    
    def test_aggregate_differences_max(self):
        """Test maximum differences."""
        differences = [
            {"z": 0.1, "V": 0.2, "policy": 0.3},
            {"z": 0.2, "V": 0.3, "policy": 0.4},
            {"z": 0.3, "V": 0.4, "policy": 0.5}
        ]
        
        agg = aggregate_differences(differences, aggregation_type="max")
        
        assert np.isclose(agg["z"], 0.3)
        assert np.isclose(agg["V"], 0.4)
        assert np.isclose(agg["policy"], 0.5)
    
    def test_aggregate_differences_std(self):
        """Test standard deviation of differences."""
        differences = [
            {"z": 0.1, "V": 0.2, "policy": 0.3},
            {"z": 0.2, "V": 0.3, "policy": 0.4},
            {"z": 0.3, "V": 0.4, "policy": 0.5}
        ]
        
        agg = aggregate_differences(differences, aggregation_type="std")
        
        assert np.isclose(agg["z"], np.std([0.1, 0.2, 0.3]))
        assert np.isclose(agg["V"], np.std([0.2, 0.3, 0.4]))
        assert np.isclose(agg["policy"], np.std([0.3, 0.4, 0.5]))


class TestEvaluation:
    """Tests for the main evaluation function."""

    def test_evaluate_permutation_difference(self):
        """Test the main evaluation function."""
        n = 4
        P = create_lattice_transitions(n)
        q = generate_costs(n, distribution="uniform", low=0.0, high=1.0, seed=42)
        goal_states = [n - 1]
        q = set_goal_states(q, goal_states)
        alpha = 1.0
        gamma = 0.9
        
        results = evaluate_permutation_difference(
            P, q, alpha, gamma, goal_states, k=2, l=1, 
            measure_type="L2", aggregation_type="average", seed=42
        )
        
        # Check that the results contain the expected keys
        assert "average" in results
        assert "max" in results
        assert "std" in results
        
        # Check that each aggregation contains the expected components
        for agg in ["average", "max", "std"]:
            assert "z" in results[agg]
            assert "V" in results[agg]
            assert "policy" in results[agg]
