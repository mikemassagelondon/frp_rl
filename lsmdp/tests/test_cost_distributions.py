"""
Tests for the cost distribution functions in the LS-MDP package.
"""

import numpy as np
import pytest
from lsmdp.cost_distributions import (
    generate_uniform_costs,
    generate_gaussian_costs,
    generate_exponential_costs,
    generate_costs,
    set_goal_states
)


def test_generate_uniform_costs_shape():
    """Test that the uniform cost function has the correct shape."""
    n = 10
    costs = generate_uniform_costs(n)
    assert costs.shape == (n,), f"Expected shape ({n},), got {costs.shape}"


def test_generate_uniform_costs_range():
    """Test that the uniform costs are within the specified range."""
    n = 100
    low, high = 0.0, 1.0
    costs = generate_uniform_costs(n, low=low, high=high)
    assert np.all(costs >= low), f"All costs should be >= {low}"
    assert np.all(costs <= high), f"All costs should be <= {high}"


def test_generate_uniform_costs_seed():
    """Test that the uniform costs are reproducible with a fixed seed."""
    n = 10
    seed = 42
    costs1 = generate_uniform_costs(n, seed=seed)
    costs2 = generate_uniform_costs(n, seed=seed)
    assert np.allclose(costs1, costs2), "Costs should be the same with the same seed"


def test_generate_gaussian_costs_shape():
    """Test that the Gaussian cost function has the correct shape."""
    n = 10
    costs = generate_gaussian_costs(n)
    assert costs.shape == (n,), f"Expected shape ({n},), got {costs.shape}"


def test_generate_gaussian_costs_clipping():
    """Test that the Gaussian costs are clipped to the specified minimum."""
    n = 100
    clip_min = 0.0
    costs = generate_gaussian_costs(n, mean=0.0, std=1.0, clip_min=clip_min)
    assert np.all(costs >= clip_min), f"All costs should be >= {clip_min}"


def test_generate_gaussian_costs_seed():
    """Test that the Gaussian costs are reproducible with a fixed seed."""
    n = 10
    seed = 42
    costs1 = generate_gaussian_costs(n, seed=seed)
    costs2 = generate_gaussian_costs(n, seed=seed)
    assert np.allclose(costs1, costs2), "Costs should be the same with the same seed"


def test_generate_exponential_costs_shape():
    """Test that the exponential cost function has the correct shape."""
    n = 10
    costs = generate_exponential_costs(n)
    assert costs.shape == (n,), f"Expected shape ({n},), got {costs.shape}"


def test_generate_exponential_costs_non_negative():
    """Test that the exponential costs are non-negative."""
    n = 100
    costs = generate_exponential_costs(n)
    assert np.all(costs >= 0), "All exponential costs should be non-negative"


def test_generate_exponential_costs_seed():
    """Test that the exponential costs are reproducible with a fixed seed."""
    n = 10
    seed = 42
    costs1 = generate_exponential_costs(n, seed=seed)
    costs2 = generate_exponential_costs(n, seed=seed)
    assert np.allclose(costs1, costs2), "Costs should be the same with the same seed"


def test_generate_costs_uniform():
    """Test that generate_costs with 'uniform' distribution calls generate_uniform_costs."""
    n = 10
    seed = 42
    costs1 = generate_uniform_costs(n, seed=seed)
    costs2 = generate_costs(n, distribution="uniform", seed=seed)
    assert np.allclose(costs1, costs2), "generate_costs with 'uniform' should call generate_uniform_costs"


def test_generate_costs_gaussian():
    """Test that generate_costs with 'gaussian' distribution calls generate_gaussian_costs."""
    n = 10
    seed = 42
    costs1 = generate_gaussian_costs(n, seed=seed)
    costs2 = generate_costs(n, distribution="gaussian", seed=seed)
    assert np.allclose(costs1, costs2), "generate_costs with 'gaussian' should call generate_gaussian_costs"


def test_generate_costs_exponential():
    """Test that generate_costs with 'exponential' distribution calls generate_exponential_costs."""
    n = 10
    seed = 42
    costs1 = generate_exponential_costs(n, seed=seed)
    costs2 = generate_costs(n, distribution="exponential", seed=seed)
    assert np.allclose(costs1, costs2), "generate_costs with 'exponential' should call generate_exponential_costs"


def test_generate_costs_invalid_distribution():
    """Test that generate_costs raises an error for an invalid distribution."""
    n = 10
    with pytest.raises(ValueError):
        generate_costs(n, distribution="invalid")


def test_set_goal_states():
    """Test that set_goal_states sets the cost of goal states to zero."""
    n = 10
    costs = np.ones(n)
    goal_states = [0, 5, 9]
    modified_costs = set_goal_states(costs, goal_states)
    
    # Check that goal states have zero cost
    for g in goal_states:
        assert np.isclose(modified_costs[g], 0.0), f"Goal state {g} should have zero cost"
    
    # Check that non-goal states are unchanged
    non_goals = [i for i in range(n) if i not in goal_states]
    for s in non_goals:
        assert np.isclose(modified_costs[s], 1.0), f"Non-goal state {s} should have cost 1.0"


def test_set_goal_states_copy():
    """Test that set_goal_states does not modify the original costs."""
    n = 10
    costs = np.ones(n)
    goal_states = [0, 5, 9]
    modified_costs = set_goal_states(costs, goal_states)
    
    # Check that the original costs are unchanged
    assert np.all(costs == 1.0), "Original costs should be unchanged"
    
    # Check that the modified costs are different
    assert not np.array_equal(costs, modified_costs), "Modified costs should be different from original"
