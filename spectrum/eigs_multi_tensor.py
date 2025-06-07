"""
Tensor product as coreprensetation (thus not commutative)
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from spectrum.eigs import sample_haar_matrices, calculate_moments
from tqdm import tqdm
from scipy.stats import ortho_group
from einops import rearrange, repeat,einsum

from spectrum.spectrum.style import set_paper_style

def generate_block_matrix(k, d, matrix_type='orthogonal'):
    """
    Generate a k×k block matrix where each block is a d×d Haar random matrix.
    
    Parameters:
        k (int): Number of blocks in each dimension
        d (int): Dimension of each block
        matrix_type (str): Type of random matrices to sample
        
    Returns:
        numpy.ndarray: Block tensor of shape (k, k, d, d)
    """
    # Initialize the block tensor
    block_tensor = np.zeros((k, k, d, d))
    
    # Generate k^2 independent d×d random matrices
    for i in range(k):
        for j in range(k):
            if matrix_type == 'orthogonal':
                # Generate Haar random orthogonal matrix
                block = ortho_group.rvs(d)
            elif matrix_type == 'permutation':
                # Generate uniform random permutation matrix
                perm = np.random.permutation(d)
                block = np.zeros((d, d))
                for idx, val in enumerate(perm):
                    block[idx, val] = 1
            else:
                raise ValueError(f"Unknown matrix type: {matrix_type}")
            
            # Place the block in the block tensor
            block_tensor[i, j] = block
    
    return block_tensor

def block_kronecker_product(A, B):
    """
    Compute the Kronecker product of two block tensors A and B,
    where each block is a d×d matrix, using einops for cleaner tensor manipulation.
    
    Parameters:
        A (numpy.ndarray): First block tensor of shape (m, n, d, d)
        B (numpy.ndarray): Second block tensor of shape (p, q, d, d)
        
    Returns:
        numpy.ndarray: Block Kronecker product of shape (m*p, n*q, d, d)
    """
    m, n, d1, d2 = A.shape
    p, q, d3, d4 = B.shape
    
    # Ensure the block dimensions match
    assert d1 == d3 and d2 == d4, "Block dimensions must match"
    
    result = einsum(A,B, "m n d1 d2, p q d2 d4 -> m p n q d1 d4")
    
    # Reshape to the final tensor shape (m*p, n*q, d, d)
    result = rearrange(result, 'm p n q d1 d2 -> (m p) (n q) d1 d2')
    
    return result

def generate_tensor_product_matrix(p, ell, d, matrix_type='orthogonal'):
    """
    Generate matrices according to the tensor product formulations in tensor_product.tex
    
    Parameters:
        p (int): Base dimension parameter
        ell (int): Case parameter (1, 2, 4, or 8)
        d (int): Dimension of each Haar random matrix
        matrix_type (str): Type of random matrices to sample
        
    Returns:
        numpy.ndarray: The generated matrix A according to the formulation
        All matrices have the same dimension (p^4, p^4, d, d)
    """
    if ell == 1:
        # Case ℓ=1: B_{p^4}
        # Generate a p^4 × p^4 block matrix where each block is a d×d Haar random matrix
        return generate_block_matrix(p**4, d, matrix_type)
    
    elif ell == 2:
        # Case ℓ=2: B_{p^2} ⊗ B_{p^2}
        # Generate two p^2 × p^2 block matrices and compute their block Kronecker product
        B = generate_block_matrix(p**2, d, matrix_type)
        # For ℓ=2, we need to ensure the result has dimension (p^4, p^4, d, d)
        # We'll use a custom block Kronecker product that preserves the d×d blocks
        #C = generate_block_matrix(p**2, d, matrix_type)
        B = block_kronecker_product(B, B)
        return B
    
    elif ell == 4:
        # Case ℓ=4: B_p ⊗ B_p ⊗ B_p ⊗ B_p
        # Generate four p × p block matrices and compute their block Kronecker product
        B = generate_block_matrix(p, d, matrix_type)
        # For ℓ=4, we need to ensure the result has dimension (p^4, p^4, d, d)
        # We'll use a custom block Kronecker product that preserves the d×d blocks
        B = block_kronecker_product(B, B)
        B = block_kronecker_product(B, B)
        return B
        
    elif ell == 8:
        # Case ℓ=8: P ⊗ P ⊗ P ⊗ P where P = v^T v
        # Generate a vector v of length p
        v = np.zeros((p,d,d))    
        for i in range(p):
                if matrix_type == 'orthogonal':
                    # Generate a Haar random orthogonal matrix and scale it by P_1d[i, j]
                    v[i] = ortho_group.rvs(d)
                elif matrix_type == 'permutation':
                    # Generate a permutation matrix and scale it by P_1d[i, j]
                    perm_matrix = np.zeros((d, d))
                    perm = np.random.permutation(d)
                    for idx, val in enumerate(perm):
                        perm_matrix[idx, val] = 1
                    v[i] =perm_matrix
        
        P = einsum(v,v, "i d1 d2, j d2 d4-> i j d1 d4")        
        # For ℓ=8, we need to ensure the result has dimension (p^4, p^4, d, d)
        # We'll use a custom block Kronecker product that preserves the d×d blocks
        P = block_kronecker_product(P, P)
        P = block_kronecker_product(P, P)
        return P
    
    else:
        raise ValueError(f"Invalid ell value: {ell}. Must be 1, 2, 4, or 8.")

def plot_tensor_distributions(args):
    # Set style
    set_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Cases to test
    cases = [
        (1, r"$\ell=1$"),
        (2, r"$\ell=2$"),
        (4, r"$\ell=4$"),
        (8, r"$\ell=8$")
    ]
    combinations = [
        (1, 8, r"$\ell=1, m=8$"),
        (2, 4, r"$\ell=2, m=4$"),
        (4, 2, r"$\ell=4, m=2$"),
        (8, 1, r"$\ell=8, m=1$")
    ]
    
    # Colors for different cases
    colors = sns.color_palette("husl", len(cases))
    
    # Verify dimensions for each case once before processing
    print("Verifying tensor dimensions for each case:")
    expected_dim = (args.p**4, args.p**4, args.d, args.d)
    for ell, label in cases:
        # Generate a test tensor
        test_tensor = generate_tensor_product_matrix(args.p, ell, args.d, args.matrix_type)
        if test_tensor.shape != expected_dim:
            print(f"WARNING: Tensor shape for ell={ell} is {test_tensor.shape}, expected {expected_dim}")
        else:
            print(f"Tensor shape for ell={ell} is correct: {test_tensor.shape}")

    all_moments_data = [[] for _ in range(len(cases))]
    i=0
    # Main progress bar for cases
    for (ell, label), color, ax in tqdm(zip(cases, colors, axes), 
                                      total=len(cases), 
                                      desc="Processing configurations"):
        # Initialize storage for singular values
        all_singular_values = []
        # Inner progress bar for trials
        for _ in tqdm(range(args.t), 
                     desc=f"Computing {label}", 
                     leave=False):
            # Generate tensor according to tensor product formulation
            tensor = generate_tensor_product_matrix(args.p, ell, args.d, args.matrix_type)
            
            # Reshape tensor to matrix for SVD
            coeff = tensor.shape[0]
            matrix = rearrange(tensor, 'a b c d -> (a c) (b d)')
            matrix /=  args.p**2
            # Compute singular values and normalize by p^4
            _, singular_values, _ = np.linalg.svd(matrix)
            singular_values = singular_values
            
            all_singular_values.extend(singular_values)
            moments = calculate_moments(matrix, max_k=4)
            all_moments_data[i].append(moments)
        # Convert to numpy array
        all_singular_values = np.array(all_singular_values)
        
        # Plot density histogram
        sns.histplot(data=all_singular_values, 
                    bins=100,
                    stat='density',
                    element='step',
                    fill=True,
                    linewidth=2,
                    color=color,
                    ax=ax)

        # Customize each subplot
        ax.set_xlabel('Singular Value')
        if ax.get_position().x0 < 0.1:  # Only leftmost plot
            ax.set_ylabel('Density')
        ax.set_title(label)
        
        # Add statistics text
        stats_text = f'μ = {np.mean(all_singular_values):.3f}\n'
        stats_text += f'σ = {np.std(all_singular_values):.3f}'
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
        i+=1


    # Adjust layout and save
    plt.tight_layout()
    suffix=f"p{args.p}_d{args.d}_t{args.t}"
    filename = f"figures/tensor_eigs_{suffix}"
    if args.matrix_type != "orthogonal":
        filename += f"_{args.matrix_type}"
    
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.close()

    from eigs_multi import plot_moments_comparison
    plot_moments_comparison(all_moments_data, combinations, args, suffix=suffix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot singular value distributions for tensor product matrices')
    parser.add_argument('--p', type=int, default=2, help='Base dimension parameter p')
    parser.add_argument('--d', type=int, default=64, help='Dimension of each Haar random matrix')
    parser.add_argument('--t', type=int, default=32, help='Number of trials for sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--matrix_type', type=str, choices=['orthogonal', 'permutation'], 
                        default='orthogonal', help='Type of random matrices to sample')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        
    plot_tensor_distributions(args)
