# Meta-LSMDP Comparison

This script compares the difference between original and permuted LSMDP solutions for different combinations of (k, l) where k^l = 2^8. It evaluates how the choice of permutation parameters affects the differences in desirability function, value function, and policy.

## Overview

The script evaluates the difference between (z, V, π) and (z', V', π') for:
- (k, l) = (2^8, 1) = (256, 1)
- (k, l) = (2^4, 2) = (16, 2)
- (k, l) = (2^2, 4) = (4, 4)
- (k, l) = (2, 8) = (2, 8)

These combinations all have the same total number of permutations (k^l = 2^8 = 256), but they differ in how the permutations are generated. A higher k means more base permutations, while a higher l means more composition of permutations.

## Usage

To run the comparison with default parameters:

```bash
cd lsmdp
python meta_lsmdp_comparison.py
```

### Command-line Arguments

The script supports the following command-line arguments:

```
usage: meta_lsmdp_comparison.py [-h] [--alpha ALPHA] [--gamma GAMMA]
                               [--lattice-size LATTICE_SIZE]
                               [--lattice-cost-distribution {uniform,exponential,normal}]
                               [--tree-depth TREE_DEPTH]
                               [--tree-cost-distribution {uniform,exponential,normal}]
                               [--measure-type {L1,L2,KL,Wasserstein}]
                               [--seed SEED] [--no-plots]

Compare the difference between original and permuted LSMDP solutions for different
combinations of (k, l) where k^l = 2^8.

optional arguments:
  -h, --help            show this help message and exit
  --alpha ALPHA         Control cost parameter (temperature) (default: 1.0)
  --gamma GAMMA         Discount factor (default: 0.9)
  --lattice-size LATTICE_SIZE
                        Size of the lattice grid (n x n) (default: 4)
  --lattice-cost-distribution {uniform,exponential,normal}
                        Cost distribution for lattice MDP (default: uniform)
  --tree-depth TREE_DEPTH
                        Depth of the binary tree (default: 3)
  --tree-cost-distribution {uniform,exponential,normal}
                        Cost distribution for binary tree MDP (default: exponential)
  --measure-type {L1,L2,KL,Wasserstein}
                        Type of difference measure to use (default: L2)
  --seed SEED           Random seed for reproducibility (default: 42)
  --no-plots            Do not show plots (default: False)
```

### Examples

1. Using KL divergence as the difference measure:

```bash
python meta_lsmdp_comparison.py --measure-type KL
```

2. Using a larger lattice grid and deeper binary tree:

```bash
python meta_lsmdp_comparison.py --lattice-size 8 --tree-depth 5
```

3. Using different cost distributions:

```bash
python meta_lsmdp_comparison.py --lattice-cost-distribution exponential --tree-cost-distribution uniform
```

4. Running with a different random seed:

```bash
python meta_lsmdp_comparison.py --seed 42
```

5. Saving plots without displaying them:

```bash
python meta_lsmdp_comparison.py --no-plots
```

## Output

The script generates:

1. **CSV files** with detailed results for each (k, l) combination:
   - `results/lattice_comparison_l2.csv`
   - `results/binarytree_comparison_l2.csv`

2. **Figures** showing the differences for each (k, l) combination:
   - `figures/lattice_comparison_l2.png`
   - `figures/binary tree_comparison_l2.png`
   - `figures/combined_comparison_l2.png`

