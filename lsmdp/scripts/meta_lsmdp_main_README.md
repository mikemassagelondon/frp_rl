# Meta-LSMDP Main Script

This script demonstrates the functionality of the Meta-LSMDP module. It creates both a lattice (grid) and a binary tree MDP, solves them analytically using the desirability function approach, and evaluates the effect of random permutations on the solutions.

## Overview

The script performs the following operations:
1. Creates and solves a lattice (grid) MDP
2. Creates and solves a binary tree MDP
3. Generates permutations for both MDPs
4. Applies permutations to the solutions
5. Computes differences between original and permuted solutions
6. Evaluates permutation differences using various measure types and aggregation methods
7. Logs results to wandb (optional)

## Usage

To run the script with default parameters:

```bash
cd lsmdp
python meta_lsmdp_main.py
```

### Command-line Arguments

The script supports the following command-line arguments:

```
usage: meta_lsmdp_main.py [-h] [--alpha ALPHA] [--gamma GAMMA] [--k K] [--l L]
                         [--lattice-size LATTICE_SIZE]
                         [--lattice-cost-distribution {uniform,exponential,normal}]
                         [--tree-depth TREE_DEPTH]
                         [--tree-cost-distribution {uniform,exponential,normal}]
                         [--seed SEED] [--no-plots] [--no-wandb]
                         [--wandb-project WANDB_PROJECT]
                         [--wandb-entity WANDB_ENTITY]

Demonstrate the functionality of the Meta-LSMDP module.

optional arguments:
  -h, --help            show this help message and exit
  --alpha ALPHA         Control cost parameter (temperature) (default: 1.0)
  --gamma GAMMA         Discount factor (default: 0.9)
  --k K                 Number of independent random permutations (default: 3)
  --l L                 Number of times the permutations will be combined (default: 2)
  --lattice-size LATTICE_SIZE
                        Size of the lattice grid (n x n) (default: 4)
  --lattice-cost-distribution {uniform,exponential,normal}
                        Cost distribution for lattice MDP (default: uniform)
  --tree-depth TREE_DEPTH
                        Depth of the binary tree (default: 2)
  --tree-cost-distribution {uniform,exponential,normal}
                        Cost distribution for binary tree MDP (default: exponential)
  --seed SEED           Random seed for reproducibility (default: 42)
  --no-plots            Do not show plots (default: False)
  --no-wandb            Do not use wandb for logging (default: False)
  --wandb-project WANDB_PROJECT
                        wandb project name (default: meta-lsmdp)
  --wandb-entity WANDB_ENTITY
                        wandb entity name (default: None)
```

### Examples

1. Using different permutation parameters:

```bash
python meta_lsmdp_main.py --k 5 --l 3
```

2. Using a larger lattice grid and deeper binary tree:

```bash
python meta_lsmdp_main.py --lattice-size 6 --tree-depth 3
```

3. Using different cost distributions:

```bash
python meta_lsmdp_main.py --lattice-cost-distribution exponential --tree-cost-distribution uniform
```

4. Running with a different random seed:

```bash
python meta_lsmdp_main.py --seed 123
```

5. Disabling plots and wandb logging:

```bash
python meta_lsmdp_main.py --no-plots --no-wandb
```

## Weights & Biases (wandb) Integration

The script includes integration with [Weights & Biases](https://wandb.ai/) for experiment tracking and visualization. By default, the script logs the following metrics to wandb:

- Desirability function differences for each measure type and aggregation method
- Value function differences for each measure type and aggregation method
- Policy differences for each measure type and aggregation method

To use wandb logging, you need to have wandb installed and be logged in:

```bash
pip install wandb
wandb login
```

You can specify the wandb project and entity using the `--wandb-project` and `--wandb-entity` arguments:

```bash
python meta_lsmdp_main.py --wandb-project my-project --wandb-entity my-entity
```

To disable wandb logging, use the `--no-wandb` flag:

```bash
python meta_lsmdp_main.py --no-wandb
```

## Measure Types

The script compares different measure types for computing differences between original and permuted solutions:

- **L1**: L1 norm (Manhattan distance)
- **L2**: L2 norm (Euclidean distance)
- **KL**: Kullback-Leibler divergence
- **Wasserstein**: Wasserstein distance (Earth Mover's Distance)

## Aggregation Methods

The script also compares different aggregation methods for summarizing differences across permutations:

- **average**: Average difference across all permutations
- **max**: Maximum difference across all permutations
- **std**: Standard deviation of differences across all permutations

## Output

The script outputs:
1. Visualizations of original and permuted solutions
2. Plots of differences for each permutation
3. Printed evaluation results for both MDPs
4. Comparison of different measure types and aggregation methods
5. Logs to wandb (if enabled)
