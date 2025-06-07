"""
Tests for the LSMDP experiment module.

This module tests the functionality of the LSMDP experiment, which evaluates
the effect of permutations on LSMDP solutions using different evaluation methods.
"""

import numpy as np
import pytest
import argparse
from unittest.mock import patch, MagicMock

from lsmdp.transition_models import create_lattice_transitions, create_binary_tree_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from lsmdp.meta_lsmdp import generate_permutations, apply_permutation_to_solution
from lsmdp_experiment import (
    parse_args,
    parse_env_args,
    make_env,
    lsmdp_solution,
    new_solution,
    evaluate_difference,
    aggregate_each_method,
    compute_cross_entropy
)


class TestArgumentParsing:
    """Tests for the argument parsing functionality."""

    
    def test_parse_args_custom(self):
        """Test parsing arguments with custom values."""
        with patch('sys.argv', [
            'lsmdp_experiment.py',
            '--env_type', 'tree',
            '--state_space_size', '32',
            '--max_length', '4',
            '--alpha', '0.5',
            '--gamma', '0.8',
            '--seed', '123',
            '--no_wandb'
        ]):
            args = parse_args()
            assert args.env_type == "tree"
            assert args.state_space_size == 32
            assert args.max_length == 4
            assert args.alpha == 0.5
            assert args.gamma == 0.8
            assert args.seed == 123
            assert args.no_wandb is True


class TestEnvironmentCreation:
    """Tests for the environment creation functionality."""

    def test_parse_env_args_lattice(self):
        """Test parsing environment arguments for lattice."""
        args = argparse.Namespace(env_type="lattice", state_space_size=16)
        env_kwargs = parse_env_args(args)
        assert env_kwargs["type"] == "lattice"
        assert env_kwargs["size"] == 4  # sqrt(16)
    
    def test_parse_env_args_tree(self):
        """Test parsing environment arguments for tree."""
        args = argparse.Namespace(env_type="tree", state_space_size=15)
        env_kwargs = parse_env_args(args)
        assert env_kwargs["type"] == "tree"
        assert env_kwargs["depth"] == 3  # 2^(3+1) - 1 = 15
    
    def test_make_env_lattice(self):
        """Test creating a lattice environment."""
        env_kwargs = {"type": "lattice", "size": 4}
        P, q, goal_states = make_env(env_kwargs)
        
        assert P.shape == (16, 16)  # 4x4 grid
        assert len(q) == 16
        assert goal_states == [15]  # Bottom-right corner
    
    def test_make_env_tree(self):
        """Test creating a tree environment."""
        env_kwargs = {"type": "tree", "depth": 3}
        P, q, goal_states = make_env(env_kwargs)
        
        assert P.shape == (15, 15)  # 2^(3+1) - 1 = 15 nodes
        assert len(q) == 15
        # Leaf nodes are goal states
        assert set(goal_states) == {7, 8, 9, 10, 11, 12, 13, 14}


class TestLSMDPSolution:
    """Tests for the LSMDP solution functionality."""

    def setup_method(self):
        """Set up a simple LSMDP problem for testing."""
        self.P = create_lattice_transitions(4)  # 2x2 grid
        self.q = generate_costs(4, distribution="uniform", low=0.0, high=1.0, seed=42)
        self.goal_states = [3]  # Bottom-right corner
        self.q = set_goal_states(self.q, self.goal_states)
        self.alpha = 1.0
        self.gamma = 0.9
    
    def test_lsmdp_solution(self):
        """Test solving an LSMDP."""
        env = (self.P, self.q, self.goal_states)
        z, v, pi = lsmdp_solution(env, self.alpha, self.gamma)
        
        assert len(z) == 4
        assert len(v) == 4
        assert len(pi) == 4
        
        # Goal state should have highest desirability
        assert z[self.goal_states[0]] == max(z)
        
        # Goal state should have lowest value (cost)
        assert v[self.goal_states[0]] == min(v)


class TestPermutationAndEvaluation:
    """Tests for permutation and evaluation functionality."""

    def setup_method(self):
        """Set up a simple LSMDP solution for testing."""
        self.n = 4
        self.z = np.array([0.5, 0.7, 0.8, 1.0])
        self.v = np.array([0.6, 0.4, 0.2, 0.0])
        self.pi = {
            0: {1: 0.7, 2: 0.3},
            1: {0: 0.2, 3: 0.8},
            2: {0: 0.1, 3: 0.9},
            3: {3: 1.0}
        }
    
    def test_new_solution(self):
        """Test applying a permutation to a solution."""
        # Permutation that swaps states 0 and 1
        permutation = np.array([1, 0, 2, 3])
        
        z1, v1, pi1 = new_solution(self.z, self.v, self.pi, permutation)
        
        assert np.isclose(z1[0], self.z[1])
        assert np.isclose(z1[1], self.z[0])
        assert np.isclose(v1[0], self.v[1])
        assert np.isclose(v1[1], self.v[0])
        
        # Check that the policy is permuted correctly
        assert 0 in pi1[1]  # State 1 should transition to state 0
        assert 3 in pi1[0]  # State 0 should transition to state 3
    
    def test_evaluate_difference_l1(self):
        """Test evaluating L1 difference between solutions."""
        # Create a slightly different solution
        z1 = np.array([0.6, 0.7, 0.8, 1.0])
        v1 = np.array([0.5, 0.4, 0.2, 0.0])
        pi1 = {
            0: {1: 0.8, 2: 0.2},
            1: {0: 0.3, 3: 0.7},
            2: {0: 0.2, 3: 0.8},
            3: {3: 1.0}
        }
        
        result = evaluate_difference((z1, v1, pi1), (self.z, self.v, self.pi), "L1")
        
        assert "z" in result
        assert "v" in result
        assert "pi" in result
        assert result["z"] > 0
        assert result["v"] > 0
        assert result["pi"] > 0
    
    def test_evaluate_difference_l2(self):
        """Test evaluating L2 difference between solutions."""
        # Create a slightly different solution
        z1 = np.array([0.6, 0.7, 0.8, 1.0])
        v1 = np.array([0.5, 0.4, 0.2, 0.0])
        pi1 = {
            0: {1: 0.8, 2: 0.2},
            1: {0: 0.3, 3: 0.7},
            2: {0: 0.2, 3: 0.8},
            3: {3: 1.0}
        }
        
        result = evaluate_difference((z1, v1, pi1), (self.z, self.v, self.pi), "L2")
        
        assert "z" in result
        assert "v" in result
        assert "pi" in result
        assert result["z"] > 0
        assert result["v"] > 0
        assert result["pi"] > 0
    
    def test_evaluate_difference_kl(self):
        """Test evaluating KL divergence between solutions."""
        # Create a slightly different solution
        z1 = np.array([0.6, 0.7, 0.8, 1.0])
        v1 = np.array([0.5, 0.4, 0.2, 0.0])
        pi1 = {
            0: {1: 0.8, 2: 0.2},
            1: {0: 0.3, 3: 0.7},
            2: {0: 0.2, 3: 0.8},
            3: {3: 1.0}
        }
        
        result = evaluate_difference((z1, v1, pi1), (self.z, self.v, self.pi), "KL")
        
        assert "z" in result
        assert "v" in result
        assert "pi" in result
        assert result["z"] > 0
        assert result["pi"] > 0
    
    def test_evaluate_difference_ce(self):
        """Test evaluating Cross Entropy between solutions."""
        # Create a slightly different solution
        z1 = np.array([0.6, 0.7, 0.8, 1.0])
        v1 = np.array([0.5, 0.4, 0.2, 0.0])
        pi1 = {
            0: {1: 0.8, 2: 0.2},
            1: {0: 0.3, 3: 0.7},
            2: {0: 0.2, 3: 0.8},
            3: {3: 1.0}
        }
        
        result = evaluate_difference((z1, v1, pi1), (self.z, self.v, self.pi), "CE")
        
        assert "z" in result
        assert "v" in result
        assert "pi" in result
        assert result["z"] > 0
        assert result["pi"] > 0
    
    def test_compute_cross_entropy(self):
        """Test computing cross entropy between two probability distributions."""
        p = np.array([0.3, 0.7])
        q = np.array([0.4, 0.6])
        
        ce = compute_cross_entropy(p, q)
        
        assert ce > 0
        
        # Cross entropy should be higher than KL divergence
        from scipy.special import kl_div
        kl = np.sum(kl_div(p, q))
        assert ce > kl


class TestAggregation:
    """Tests for the aggregation functionality."""

    def test_aggregate_each_method(self):
        """Test aggregating results across permutations."""
        results = {
            "L1": [
                {"z": 0.1, "v": 0.2, "pi": 0.3},
                {"z": 0.2, "v": 0.3, "pi": 0.4}
            ],
            "L2": [
                {"z": 0.3, "v": 0.4, "pi": 0.5},
                {"z": 0.4, "v": 0.5, "pi": 0.6}
            ]
        }
        
        agg_results = aggregate_each_method(results, aggregation_method="average")
        
        assert "L1_z" in agg_results
        assert "L1_v" in agg_results
        assert "L1_pi" in agg_results
        assert "L2_z" in agg_results
        assert "L2_v" in agg_results
        assert "L2_pi" in agg_results
        
        assert np.isclose(agg_results["L1_z"], 0.15)
        assert np.isclose(agg_results["L1_v"], 0.25)
        assert np.isclose(agg_results["L1_pi"], 0.35)
        assert np.isclose(agg_results["L2_z"], 0.35)
        assert np.isclose(agg_results["L2_v"], 0.45)
        assert np.isclose(agg_results["L2_pi"], 0.55)


class TestIntegration:
    """Integration tests for the LSMDP experiment."""

    @patch('wandb.log')
    def test_experiment_workflow(self, mock_wandb_log):
        """Test the entire experiment workflow with a small example."""
        # Create a simple environment
        P = create_lattice_transitions(4)  # 2x2 grid
        q = generate_costs(4, distribution="uniform", low=0.0, high=1.0, seed=42)
        goal_states = [3]  # Bottom-right corner
        q = set_goal_states(q, goal_states)
        
        # Solve the LSMDP
        alpha = 1.0
        gamma = 0.9
        z, v, pi = solve_lsmdp(P, q, alpha, gamma, goal_states)
        
        # Use simple permutations for testing
        list_length = [1, 2]
        permutations = [
            np.array([0, 1, 2, 3]),  # Identity permutation
            np.array([1, 0, 3, 2])   # Swap first and second, third and fourth
        ]
        
        results = {}
        for method in ["L1", "L2", "KL", "CE"]:
            results[method] = []
        
        for permutation in permutations:
            z1, v1, pi1 = new_solution(z, v, pi, permutation)
            
            for method in ["L1", "L2", "KL", "CE"]:
                result = evaluate_difference((z1, v1, pi1), (z, v, pi), method)
                results[method].append(result)
        
        logging_results = aggregate_each_method(results, aggregation_method="average")
        
        # Check that the logging results have the expected structure
        for method in ["L1", "L2", "KL", "CE"]:
            for component in ["z", "v", "pi"]:
                assert f"{method}_{component}" in logging_results
