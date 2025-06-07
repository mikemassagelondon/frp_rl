import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import ortho_group
import seaborn as sns
from tqdm import tqdm

def calculate_moments(matrix, max_k=4):
    """
    Calculate k-th moments of A^T * A for k from 1 to max_k using an iterative approach.
    
    Parameters:
        matrix (numpy.ndarray): Input matrix A
        max_k (int): Maximum k value for moment calculation
        
    Returns:
        dict: Dictionary mapping k to the corresponding moment value
    """
    # Calculate A^T * A
    ata = matrix.T @ matrix
    d = matrix.shape[0]  # Dimension of the matrix
    
    # Initialize moments dictionary
    moments = {}
    
    # Calculate first moment (k=1)
    moments[1] = np.trace(ata) / d
    
    # For k > 1, use the previous result to calculate the next power
    current_power = ata.copy()
    for k in range(2, max_k + 1):
        # Multiply by ata to get the next power
        current_power = current_power @ ata
        # Calculate trace and normalize by dimension
        moments[k] = np.trace(current_power) / d
        
    return moments

def string_to_shape(shape_string):
    """
    Convert a string of digits into a tuple that can be used as a NumPy shape.
    
    Parameters:
        shape_string (str): String containing only digits (e.g., "64125307")
    
    Returns:
        tuple: A tuple that can be used as a NumPy shape
        
    Example:
        >>> string_to_shape("64125307")
        (6, 4, 1, 2, 5, 3, 0, 7)
    """
    # Convert each character to integer
    shape = tuple(int(x) for x in shape_string)
    
    # Validate input
    if not all(isinstance(x, int) for x in shape):
        raise ValueError("All characters must be digits")
        
    return shape

def sample_haar_matrices(d, b, matrix_type='orthogonal'):
    """Generate b independent d x d random matrices
    
    Parameters:
        d (int): Dimension of matrices
        b (int): Number of matrices to generate
        matrix_type (str): Type of matrices to generate ('orthogonal' or 'permutation')
        
    Returns:
        list: List of random matrices
    """
    if matrix_type == 'orthogonal':
        # Generate Haar random orthogonal matrices
        return [ortho_group.rvs(d) for _ in range(b)]
    elif matrix_type == 'permutation':
        # Generate uniform random permutation matrices
        matrices = []
        for _ in range(b):
            # Generate a random permutation
            perm = np.random.permutation(d)
            # Create permutation matrix
            perm_matrix = np.zeros((d, d))
            for i, j in enumerate(perm):
                perm_matrix[i, j] = 1
            matrices.append(perm_matrix)
        return matrices
    else:
        raise ValueError(f"Unknown matrix type: {matrix_type}")

def generate_matrix_products(base_matrices, ell):
    """Generate all possible products of length ell from base matrices"""
    if ell == 1:
        return base_matrices
    
    products = []
    prev_products = generate_matrix_products(base_matrices, ell-1)
    
    for prev in prev_products:
        for base in base_matrices:
            products.append(np.dot(prev, base))
            
    return products


def construct_linearization(matrices):
    """
    Constrcut matrices of V_{ij} = U_{ n^{1/2}i + j}
    for linearization
    """
    N = len(matrices)
    n = round(np.sqrt(N))
    assert n**2 == N
    d = matrices[0].shape[0]    
    
    V = np.array(matrices).reshape(n,n,d,d).transpose((0,2,1,3)).reshape(n*d, n*d)

    """
    #V = np.zeros((n*d, n*d))
    for i in range(n):
        for j in range(n):
            V[i*d:(i+1)*d, j*d:(j+1)*d] = matrices[n*i + j] 
    """
    return  V


def transform_matrices(matrices, transform=None, new_shape=None):
    import random


    if transform == "random":
        random.shuffle(matrices)
    elif transform in ["4th", "0213"] :
        n = len(matrices)
        base = np.power(n,1/4)
        base = round(base)
        assert base**4 == n
        new_index = np.arange(n).reshape(base,base,base,base).transpose((0,2,1,3)).flatten()
        matrices = [ matrices[new_index[i]] for i in np.arange(len(new_index))]

    elif transform == "None" or transform is None:
        pass

    elif len(transform) == 8:
        new_shape = string_to_shape(transform)
        n = len(matrices)
        base = np.power(n,1/8)
        base = round(base)
        assert base**8 == n
        new_index = np.arange(n).reshape(base,base,base,base, base, base, base, base).transpose(new_shape).flatten()
        matrices = [ matrices[new_index[i]] for i in np.arange(len(new_index))]


    else:
        raise ValueError()
    
    block_matrix = construct_linearization(matrices)
    return block_matrix

def compute_eigenvalues(matrices, normalize=True):
    """Compute eigenvalues of the block matrix"""
    
    X = construct_linearization(matrices)
    if normalize:
        # v1
        #eigenvalues = eigenvalues / np.sqrt(len(matrices))
        #arxiv v1
        #X /=np.sqrt(len(matrices))
        # v2
        X /= np.sqrt(np.sqrt(len(matrices)))
    

    #eigenvalues = np.linalg.eigvalsh(X)
    _, eigenvalues, _ = np.linalg.svd(X)


        
    
    return eigenvalues

def calculate_bin_width(data, method='scott'):
    """Calculate bin width using various methods"""
    n = len(data)
    
    if method == 'scott':
        # Scott's normal reference rule
        h = 3.49 * np.std(data) * (n ** (-1/3))
    elif method == 'fd':
        # Freedman-Diaconis rule
        iqr = np.percentile(data, 75) - np.percentile(data, 25)
        h = 2 * iqr * (n ** (-1/3))
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return h

def calculate_bins(data, method='scott'):
    """Calculate number of bins using various methods"""
    n = len(data)
    data_range = np.max(data) - np.min(data)
    
    if method == 'sturges':
        # Sturges' formula
        bins = int(np.ceil(1 + np.log2(n)))
    elif method == 'rice':
        # Rice Rule
        bins = int(np.ceil(2 * (n ** (1/3))))
    elif method in ['scott', 'fd']:
        # Convert bin width to number of bins
        h = calculate_bin_width(data, method)
        bins = int(np.ceil(data_range / h))
    elif method == "default":
        bins = 100
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return max(bins, 10)  # Ensure at least 10 bins


def plot_histogram(data, args, method='scott', plot_all=True, plot_filtered=True):
    """Plot histograms with given binning method, both for all eigenvalues and filtered ones"""
    n = len(data)
    filtered_data = data[data <= 1.0]
    n_filtered = len(filtered_data)
    
    for is_filtered, curr_data in [(False, data), (True, filtered_data)]:
        if (is_filtered and not plot_filtered) or (not is_filtered and not plot_all):
            continue
            
        bins = calculate_bins(curr_data, method)
        plt.figure(figsize=(10, 6))
        
        # Plot density-based histogram
        sns.histplot(data=curr_data, 
                    bins=bins, 
                    stat='density',
                    element='step',
                    fill=False,
                    linewidth=2)
        
        title = f'Eigenvalue Distribution (d={args.d}, m={args.m}, ell={args.ell})'
        if is_filtered:
            title += '\nFiltered (λ ≤ 1)'
        title += f'\nTransform: {args.trans}'
        plt.title(title)
        plt.xlabel('Singular Value')
        plt.ylabel('Density')
        
        # Add text with statistics
        curr_n = n_filtered if is_filtered else n
        stats_text = f'Bins: {bins}\n'
        stats_text += f'Total samples: {curr_n}\n'
        stats_text += f'Mean: {np.mean(curr_data):.4f}\n'
        stats_text += f'Std: {np.std(curr_data):.4f}'
        if is_filtered:
            stats_text += f'\nRatio: {(n_filtered/n)*100:.1f}%'
        
        plt.text(0.98, 0.98, 
                 stats_text,
                 transform=plt.gca().transAxes,
                 horizontalalignment='right',
                 verticalalignment='top',
                 bbox=dict(facecolor='white', alpha=0.8))
        
        # Save plots
        suffix = '_filtered' if is_filtered else ''
        suffix1 = "_tr" if args.trace == 1 else ""
        suffix2 = f"_{args.trans}" if args.trans is not None else ""
        suffix = f"_t{args.t}" + suffix + suffix1 + suffix2
        
        def _savefig(x, *args, **kwargs):
            plt.savefig(x)
            print(x)
        
        import os
        dirname = f"../linearization/{suffix}"
        os.makedirs(dirname, exist_ok=True)
        _savefig(f'{dirname}/eigs_d{args.d}_m{args.m}_l{args.ell}_{method}.png', 
                    dpi=300, bbox_inches='tight')
        _savefig(f'{dirname}/eigs_d{args.d}_m{args.m}_l{args.ell}_{method}.pdf', 
                    bbox_inches='tight')
        plt.close()

def main(args):
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
    
    # Calculate b = 2^m
    b = 2 ** args.m
    
    # Initialize storage for all eigenvalues
    all_eigenvalues = []
    # Repeat the process t times with progress bar
    for _ in tqdm(range(args.t), desc="Computing eigenvalues"):
        base_matrices = sample_haar_matrices(args.d, b, args.matrix_type)
        all_products = generate_matrix_products(base_matrices, args.ell)
        eigenvalues = compute_eigenvalues(all_products, transform=args.trans)
        all_eigenvalues.extend(eigenvalues)
    
    # Convert to numpy array
    all_eigenvalues = np.array(all_eigenvalues)
    
    # Plot histograms using different binning methods
    #methods = ['sturges', 'scott', 'fd', 'rice']
    methods = [args.bins]
    for method in methods:
        plot_histogram(all_eigenvalues, args, method)
    
    # Print summary statistics
    filtered_eigenvalues = all_eigenvalues[all_eigenvalues <= 1.0]
    
    print("\nSummary Statistics (All Eigenvalues):")
    print(f"Total samples: {len(all_eigenvalues)}")
    print(f"Mean: {np.mean(all_eigenvalues):.4f}")
    print(f"Std: {np.std(all_eigenvalues):.4f}")
    print(f"Min: {np.min(all_eigenvalues):.4f}")
    print(f"Max: {np.max(all_eigenvalues):.4f}")
    
    print("\nSummary Statistics (Filtered Eigenvalues λ ≤ 1):")
    print(f"Total samples: {len(filtered_eigenvalues)}")
    print(f"Ratio: {(len(filtered_eigenvalues)/len(all_eigenvalues))*100:.1f}%")
    print(f"Mean: {np.mean(filtered_eigenvalues):.4f}")
    print(f"Std: {np.std(filtered_eigenvalues):.4f}")
    
    print("\nNumber of bins by method:")
    for method in methods:
        print(f"{method}:")
        print(f"  All: {calculate_bins(all_eigenvalues, method)}")
        print(f"  Filtered: {calculate_bins(filtered_eigenvalues, method)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute eigenvalue distribution of matrix products')
    parser.add_argument('--d', type=int, required=True, help='Dimension of base matrices')
    parser.add_argument('--m', type=int, required=True, help='Power of 2 for number of base matrices (b=2^m)')
    parser.add_argument('--ell', type=int, required=True, help='Length of matrix products')
    parser.add_argument('--t', type=int, default=32, help='Number of trials for sampling (default: 100)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--bins', type=str, help='bin method', default="default")
    parser.add_argument('--trace', type=int, help='take normalized trace of entrices', default=0)
    parser.add_argument('--trans', type=str, help='trasnform of block matrices: None, random, 4th', default=0)
    parser.add_argument('--matrix_type', type=str, choices=['orthogonal', 'permutation'], 
                        default='orthogonal', help='Type of random matrices to sample (orthogonal or permutation)')
    
    args = parser.parse_args()
    main(args)
