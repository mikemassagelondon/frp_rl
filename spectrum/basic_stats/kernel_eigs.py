import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import ortho_group
import seaborn as sns
from tqdm import tqdm

def sample_haar_matrices(d, b):
    """Generate b independent d x d Haar random orthogonal matrices"""
    return [ortho_group.rvs(d) for _ in range(b)]

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

def construct_block_matrix(matrices, trace=True):
    """Construct block matrix X where X_ij = W_i^T W_j"""
    n = len(matrices)
    d = matrices[0].shape[0]
    # Initialize X as a block matrix with proper dimensions
    if trace:
        X = np.zeros((n,n))
    else:
        X = np.zeros((n * d, n * d))
    
    for i in range(n):
        for j in range(n):
            if trace:
                X[i,j] = np.trace(matrices[i].T @ matrices[j])/d
            else:
                # Fill the (i,j)-th block with W_i^T W_j
                X[i*d:(i+1)*d, j*d:(j+1)*d] = matrices[i].T @ matrices[j]
            
    return X



def compute_eigenvalues(matrices, normalize=False, trace=True):
    """Compute eigenvalues of the block matrix"""
    X = construct_block_matrix(matrices, trace=trace)

    _, eigenvalues, _ = np.linalg.svd(X)
    


        
    if normalize:
        eigenvalues = eigenvalues / len(matrices)
    
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
        title += f'\nBinning: {method}'
        plt.title(title)
        plt.xlabel('Eigenvalue')
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
        suffiix1 = "_tr" if args.trace == 1 else ""
        suffix = suffix + suffiix1
        plt.savefig(f'eigenvalue_dist_d{args.d}_m{args.m}_l{args.ell}_{method}{suffix}.png', 
                    dpi=300, bbox_inches='tight')
        plt.savefig(f'eigenvalue_dist_d{args.d}_m{args.m}_l{args.ell}_{method}{suffix}.pdf', 
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
        base_matrices = sample_haar_matrices(args.d, b)
        all_products = generate_matrix_products(base_matrices, args.ell)
        eigenvalues = compute_eigenvalues(all_products, trace=(args.trace==1))
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
    parser.add_argument('--t', type=int, default=100, help='Number of trials for sampling (default: 100)')
    parser.add_argument('--seed', type=int, help='Random seed (optional)')
    parser.add_argument('--bins', type=str, help='bin method', default="sturges")
    parser.add_argument('--trace', type=int, help='take normalized trace of entrices', default=1)
    
    args = parser.parse_args()
    main(args)