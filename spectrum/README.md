# Spectrum Analysis

Spectral analysis tools for Free Random Projection matrices. Analyzes eigenvalue distributions, matrix moments, and effective dimensions of orthogonal matrix products.

## Key Tools

- **spectrum/eigs.py**: Core spectral analysis utilities
- **eigs_multi_average.py**: Eigenvalue distributions of averaged matrix products  
- **eigs_multi.py**: Spectral analysis with theoretical distribution overlays

## Usage

```bash
# Analyze averaged matrix eigenvalues 
python eigs_multi_average.py --d 64 --t 128 --c 1

# Analyze with partial transpose and theoretical overlay (For Experiments in Appendix)
python eigs_multi.py --d 64 --t 128 --trans 06243517 --matrix_type orthogonal

# Batch analysis for individual experiments
bash run_batch.sh
```

## Analysis Features

- **Eigenvalue Distributions**: Compare empirical vs theoretical (Wigner quarter-circle law)
- **Matrix Moments**: Statistical analysis of matrix product properties
- **Effective Dimension**: Dimension analysis of FRP transformations
- **Batch Processing**: Automated analysis across parameter ranges
