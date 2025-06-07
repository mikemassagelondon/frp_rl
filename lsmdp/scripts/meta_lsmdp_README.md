# Meta-LSMDP Module

This module extends the LSMDP package to handle permutations of state spaces and evaluate differences between solutions. It provides tools for assessing how sensitive LSMDP solutions are to random permutations of the state space, which can help evaluate the generalizability and robustness of learned policies.

## Overview

The Meta-LSMDP module builds on the Linearly Solvable Markov Decision Process (LS-MDP) package to:

1. Generate random permutations of state spaces
2. Apply these permutations to LSMDP solutions
3. Compute differences between original and permuted solutions using various metrics
4. Aggregate these differences across permutations
5. Evaluate the overall effect of permutations on LSMDP solutions

## Installation

The Meta-LSMDP module is included as part of the LSMDP package. To install:

```bash
# Clone the repository
git clone <repository-url>
cd <repository-directory>

# Install the package
pip install -e .
```

## Dependencies

- NumPy
- SciPy (for KL divergence calculation)
- Matplotlib (for visualization)
- NetworkX (for tree visualization)
- pytest (for running tests)

## Usage

Here's a simple example of how to use the Meta-LSMDP module:

```python
import numpy as np
from lsmdp.transition_models import create_lattice_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.meta_lsmdp import evaluate_permutation_difference

# Create a 4x4 grid (16 states)
n = 16
P = create_lattice_transitions(n)

# Generate costs using uniform distribution
q = generate_costs(n, distribution="uniform", low=0.0, high=1.0, seed=42)

# Set goal state (bottom-right corner)
goal_states = [n - 1]
q = set_goal_states(q, goal_states)

# Parameters
alpha = 1.0  # Control cost parameter
gamma = 0.9  # Discount factor
k = 3  # Number of independent random permutations
l = 2  # Number of times the permutations will be combined

# Evaluate permutation differences
results = evaluate_permutation_difference(
    P, q, alpha, gamma, goal_states, 
    k=k, l=l, 
    measure_type="L2", 
    aggregation_type="average", 
    seed=42
)

# Print results
print("Average Differences:")
print(f"  Desirability Function (z): {results['average']['z']:.6f}")
print(f"  Value Function (V): {results['average']['V']:.6f}")
print(f"  Policy (π): {results['average']['policy']:.6f}")
```

For a more detailed example, see the `meta_lsmdp_main.py` script.

## Key Functions

### `generate_permutations(n, k, l, seed=None)`

Generates k^l permutations of n states by taking the product of k random permutations.

- `n`: Number of states
- `k`: Number of independent random permutations
- `l`: Number of times the permutations will be combined
- `seed`: Random seed for reproducibility

### `apply_permutation_to_solution(z, V, policy, permutation)`

Applies a permutation to an LSMDP solution.

- `z`: Desirability function
- `V`: Value function
- `policy`: Optimal policy
- `permutation`: Permutation of state indices

### `compute_difference(z, V, policy, z_prime, V_prime, policy_prime, measure_type="L2")`

Computes the difference between the original LSMDP solution and the permuted solution.

- `z, V, policy`: Original solution
- `z_prime, V_prime, policy_prime`: Permuted solution
- `measure_type`: Type of difference measure to use (options: "L1", "L2", "KL", "Wasserstein")

### `aggregate_differences(differences, aggregation_type="average")`

Aggregates differences across permutations.

- `differences`: List of difference dictionaries
- `aggregation_type`: Type of aggregation to use (options: "average", "max", "std")

### `evaluate_permutation_difference(P, q, alpha, gamma, goal_states=None, k=3, l=2, measure_type="L2", aggregation_type="average", seed=None)`

Evaluates the difference between LSMDP solutions under random permutations.

- `P, q, alpha, gamma, goal_states`: LSMDP parameters
- `k, l`: Permutation parameters
- `measure_type, aggregation_type`: Evaluation parameters
- `seed`: Random seed for reproducibility

## Difference Measures

The module supports several difference measures:

- **L1 Norm (Absolute Difference)**: Sum of absolute differences between corresponding values. Provides a straightforward aggregate of pointwise errors.
- **L2 Norm (Euclidean Distance)**: Square root of the sum of squared differences. Gives more weight to larger deviations due to squaring.
- **KL Divergence**: Measures how one probability distribution diverges from another. Suitable for comparing policies.
- **Wasserstein Distance**: Measures the "cost" of transforming one distribution into another. Useful for continuous distributions.

## Aggregation Methods

The module supports several aggregation methods:

- **Average**: Computes the mean difference across all permutations. Gives an overall sense of how far apart the solutions are on average.
- **Maximum**: Computes the worst-case (maximum) difference across all permutations. Highlights if there are any problematic cases.
- **Standard Deviation**: Computes the variability of the difference across permutations. Shows if the differences are consistent or highly variable.

## Running Tests

To run the tests:

```bash
pytest lsmdp/tests/test_meta_lsmdp.py
```

## License

[MIT License](LICENSE)
