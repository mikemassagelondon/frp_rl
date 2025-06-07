# Linearly Solvable MDP (LS-MDP)

Analytical solution framework for Linearly Solvable Markov Decision Processes with Free Random Projection integration. LS-MDPs allow computing optimal policies and value functions analytically through the "desirability function" z*(s).

## Key Features

- **Analytical Solutions**: Exact computation of optimal policies via desirability functions
- **Multiple State Spaces**: Support for lattice/grid and binary tree structures  
- **Cost Distributions**: Uniform, Gaussian, and exponential cost functions
- **Policy Aggregation**: Meta-learning experiments with policy composition
- **Visualization**: Built-in plotting for results analysis

## Main Experiments

```bash
python policy_aggregation_experiment.py  # Policy aggregation research
python main.py                          # Basic LS-MDP demonstration
python lsmdp_experiment.py             # Full LS-MDP experiments
```

## Usage

Here's a simple example of how to use the package:

```python
import numpy as np
import matplotlib.pyplot as plt
from lsmdp.transition_models import create_lattice_transitions
from lsmdp.cost_distributions import generate_costs, set_goal_states
from lsmdp.solver import solve_lsmdp
from lsmdp.visualization import visualize_results

# Create a 4x4 grid (16 states)
n = 16
grid_shape = (4, 4)

# Create transition matrix
P = create_lattice_transitions(n)

# Generate costs using uniform distribution
q = generate_costs(n, distribution="uniform", low=0.0, high=1.0)

# Set goal state (bottom-right corner)
goal_states = [n - 1]
q = set_goal_states(q, goal_states)

# Solve the MDP
alpha = 1.0  # Control cost parameter
gamma = 0.9  # Discount factor
z, V, policy = solve_lsmdp(P, q, alpha, gamma, goal_states)

# Visualize results
fig = visualize_results(z, V, policy, "lattice", grid_shape=grid_shape)
plt.show()
```

For more detailed examples, see the `main.py` script.

## Package Structure

- `transition_models.py`: Functions to create transition matrices for different state space structures
- `cost_distributions.py`: Functions to generate cost functions with different distributions
- `solver.py`: Core solver functions for computing the desirability function, value function, and policy
- `visualization.py`: Utilities for visualizing the results
- `tests/`: Unit tests for the package
- `main.py`: Example script demonstrating the package functionality

## Mathematical Background

In an LS-MDP, the desirability function z*(s) satisfies the linear equation:

```
z*(s) = exp(-gamma/alpha * q(s)) * sum_{s'} p(s'|s) * z*(s')
```

where:
- p(s'|s) is the passive (uncontrolled) transition probability
- q(s) is the state cost function
- alpha is the control cost parameter (temperature)
- gamma is the discount factor

The optimal value function is given by:

```
V*(s) = -alpha/gamma * ln(z*(s))
```

And the optimal policy is:

```
π*(s'|s) = p(s'|s) * z*(s') / sum_{s''} p(s''|s) * z*(s'')
```

## Running Tests

To run the tests:

```bash
pytest tests/
```
