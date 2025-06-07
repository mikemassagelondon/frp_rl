import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import math
from spectrum.eigs import sample_haar_matrices, generate_matrix_products, compute_eigenvalues, calculate_moments
from spectrum.effective_dim import free_effective_dim_ratio, empirical_effective_dim_ratio
from tqdm import tqdm
from spectrum.style import set_paper_style



def sample_data_matrix(d, c=1):
    """Sample a random matrix with i.i.d. entries from N(0, 1/d).
    
    Args:
        d (int): Dimension parameter that determines the number of columns and scaling.
        c (float, optional): Parameter that affects the number of rows. Defaults to 1.
        
    Returns:
        numpy.ndarray: A matrix of size round(c*d) x d with entries sampled from N(0, 1/d).
    """
    # Calculate the number of rows
    num_rows = round(c * d)
    
    # Sample entries from N(0, 1/d)
    scale = 1.0 / np.sqrt(d)  # Standard deviation = 1/sqrt(d) to get variance = 1/d
    
    # Generate the random matrix
    return np.random.normal(0, scale, size=(num_rows, d))


def theoretical_effective_dim_ratio(r, l, m, with_mp=False, c=1):
    """
    Calculate the theoretical effective dimension using the solver from effective_dim.py.
    
    Parameters:
        r (float): Regularization parameter
        l (int): l parameter (ell)
        m (int): m parameter
        with_mp (bool, optional): Whether to use Marchenko-Pastur law. Defaults to False.
        c (float, optional): Parameter for MP law. Defaults to 1.
        
    Returns:
        float: Theoretical effective dimension
    """
    # Calculate beta based on m
    beta = math.pow(2, -m)
    
    # Use the solver from effective_dim.py
    return free_effective_dim_ratio(r, beta, l, with_mp=with_mp, c=c)

def plot_effective_dim_ratio(all_eff_dim_data, r_values, combinations, args, suffix=None, with_mp=False, c=1, title_prefix=""):
    """
    Plot normalized effective dimension for different combinations across r values.
    Also plot theoretical effective dimension for comparison.
    
    Parameters:
        all_eff_dim_data (list): List of normalized effective dimension data for each combination
        r_values (list): List of r values used for calculation
        combinations (list): List of (l, m, label) combinations
        args (argparse.Namespace): Command line arguments
        suffix (str, optional): Suffix for filename
        with_mp (bool, optional): Whether to use Marchenko-Pastur law for theoretical calculation. Defaults to False.
        c (float, optional): Parameter for MP law. Defaults to 1.
        title_prefix (str, optional): Prefix for the title. Defaults to "".
    """
    # Set style
    set_paper_style()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(6, 5))
    
    # Colors for different combinations
    colors = sns.color_palette("husl", len(combinations))
    markers = ['.', '.', '.', '.']
    
    # Plot for each combination
    for i, ((l, m, label), eff_dim_data, color) in enumerate(zip(combinations, all_eff_dim_data, colors)):
        # Calculate mean and std for each r value
        means = [np.mean(data) for data in eff_dim_data]
        stds = [np.std(data) for data in eff_dim_data]
        
        # Plot mean with error bars for std
        ax.errorbar(
            r_values, 
            means, 
            yerr=stds, 
            marker=markers[i], 
            linestyle="", 
            capsize=5, 
            markersize=8,
            color=color,
            label=f"Empirical: {label}"
        )
        
        # Calculate and plot theoretical effective dimension
        try:
            theo_eff_dims = []
            for r in r_values:
                try:
                    theo_eff_dim = theoretical_effective_dim_ratio(r, l, m, with_mp=with_mp, c=c)
                    theo_eff_dims.append(theo_eff_dim)
                except RuntimeError:
                    # Skip this r value if Newton's method doesn't converge
                    theo_eff_dims.append(None)
            
            # Filter out None values for plotting
            valid_r_values = [r for r, dim in zip(r_values, theo_eff_dims) if dim is not None]
            valid_theo_eff_dims = [dim for dim in theo_eff_dims if dim is not None]
            
            if valid_theo_eff_dims:  # Only plot if we have valid theoretical values
                ax.plot(
                    valid_r_values,
                    valid_theo_eff_dims,
                    marker='x',
                    linestyle='--',
                    color=color,
                    alpha=0.7,
                    label=f"Theoretical: {label}"
                )
        except Exception as e:
            print(f"Warning: Could not calculate theoretical effective dimension for {label}: {e}")
    
    #ax.set_xlim(0,max(r_values)+min(r_values))
    ax.set_xscale('log')
    ax.tick_params(axis='y', labelrotation=90) 
    # Set labels and title
    ax.set_xlabel(r'$\gamma$')
    ax.set_ylabel(r'$d_\text{eff}(\gamma)/p$')
    #if title_prefix:
    #    ax.set_title(title_prefix)
    
    # Add legend with smaller font size to accommodate more entries
    #ax.legend(loc='best', fontsize='small')
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    # Save figure
    mp_suffix = "_mp" if with_mp else ""
    filename = f"norm_eff_dim{mp_suffix}_d{args.d}{suffix}"
    plt.savefig(f"figures/{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/{filename}.pdf", bbox_inches='tight')
    plt.close()

def plot_moments_comparison(all_moments_data, combinations, args, suffix=None, title_prefix="", filename_prefix=""):
    """
    Plot comparison of moments for different combinations in a single figure.
    
    Parameters:
        all_moments_data (list): List of moment data for each combination
        combinations (list): List of (l, m, label) combinations
        args (argparse.Namespace): Command line arguments
        suffix (str, optional): Suffix for filename
        title_prefix (str, optional): Prefix for the title. Defaults to "".
        filename_prefix (str, optional): Prefix for the filename. Defaults to "".
    """
    # Set style
    set_paper_style()
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(6, 6))
    # k values for x-axis (1 to max_k)
    max_k = len(all_moments_data[0][0])  # Get the number of moments from the first combination's first trial
    k_values = list(range(1, max_k + 1))
    
    # Colors and markers for different combinations
    colors = sns.color_palette("husl", len(combinations))
    markers = ['.', '.', '.', '.']
    
    # For each combination and its corresponding data
    for i, ((l, m, label), moments_data) in enumerate(zip(combinations, all_moments_data)):
        # Extract means and stds for each k
        means = [np.mean([data[k] for data in moments_data]) for k in k_values]
        stds = [np.std([data[k] for data in moments_data]) for k in k_values]
        
        # Plot mean with error bars for std
        ax.errorbar(
            k_values, 
            means, 
            yerr=stds, 
            marker=markers[i], 
            linestyle='-', 
            capsize=5, 
            markersize=8,
            color=colors[i],
            label=label
        )
    
    # Set labels and title
    ax.set_xlabel(r'$k$')
    ax.set_ylabel(r'$m_k$')
    ax.set_xticks(k_values)
    #if title_prefix:
    #    ax.set_title(title_prefix)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    filename = f"{filename_prefix}moments_comp_avg_d{args.d}{suffix}"
    
    plt.savefig(f"figures/{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/{filename}.pdf", bbox_inches='tight')
    plt.close()

def plot_singular_value_distributions(all_eigenvalues_data, combinations, args, suffix=None, title_prefix="", filename_prefix=""):
    """
    Plot singular value distributions for different combinations.
    
    Parameters:
        all_eigenvalues_data (list): List of eigenvalues data for each combination
        combinations (list): List of (l, m, label) combinations
        args (argparse.Namespace): Command line arguments
        suffix (str, optional): Suffix for filename
        title_prefix (str, optional): Prefix for the title. Defaults to "".
        filename_prefix (str, optional): Prefix for the filename. Defaults to "".
    """
    # Set style
    set_paper_style()
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Colors for different combinations
    colors = sns.color_palette("husl", len(combinations))
    
    # Plot for each combination
    for i, ((l, m, label), eigenvalues, color, ax) in enumerate(zip(combinations, all_eigenvalues_data, colors, axes)):
        # Convert to numpy array
        eigenvalues_array = np.array(eigenvalues)
        
        # Plot density histogram
        sns.histplot(data=eigenvalues_array, 
                    bins=50,
                    stat='density',
                    element='step',
                    fill=True,
                    linewidth=0,
                    color=color,
                    ax=ax)

        # Customize each subplot
        ax.set_xlabel('Singular Value')
        if ax.get_position().x0 < 0.1:  # Only leftmost plot
            ax.set_ylabel('Density')
        ax.set_title(label)
        
        # Add statistics text
        stats_text = f'μ = {np.mean(eigenvalues_array):.3f}\n'
        stats_text += f'σ = {np.std(eigenvalues_array):.3f}'
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Add overall title if provided
    #if title_prefix:
    #    fig.suptitle(title_prefix, fontsize=16)
    #    fig.subplots_adjust(top=0.85)  # Adjust to make room for the title
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    filename = f"{filename_prefix}eigs_avg_comp_d{args.d}{suffix}"
    
    plt.savefig(f"figures/{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/{filename}.pdf", bbox_inches='tight')
    plt.close()

def plot_multiple_distributions(args):
    # Set style
    set_paper_style()
    
    # Combinations to test
    combinations = [
        (1, 8, r"$\ell=1, m=8$"),
        (2, 4, r"$\ell=2, m=4$"),
        (4, 2, r"$\ell=4, m=2$"),
        (8, 1, r"$\ell=8, m=1$")
    ]
    
    # Generate r values for normalized effective dimension 
    r_values = np.logspace(-4, -1, 10)
    # Colors for different combinations
    colors = sns.color_palette("husl", len(combinations))
    
    # Storage for average_matrix data
    all_eigenvalues_avg = [[] for _ in range(len(combinations))]
    all_moments_data_avg = [[] for _ in range(len(combinations))]
    all_eff_dim_data_avg = [[[] for _ in range(len(r_values))] for _ in range(len(combinations))]
    
    # Storage for Y = average_matrix @ X data
    all_eigenvalues_Y = [[] for _ in range(len(combinations))]
    all_moments_data_Y = [[] for _ in range(len(combinations))]
    all_eff_dim_data_Y = [[[] for _ in range(len(r_values))] for _ in range(len(combinations))]
    
    # Main progress bar for combinations
    for i, (l, m, label) in enumerate(tqdm(combinations, 
                                        total=len(combinations), 
                                        desc="Processing configurations")):
        # Calculate b = 2^m
        b = 2 ** m
        
        # Inner progress bar for trials
        for _ in tqdm(range(args.t), 
                     desc=f"Computing {label}", 
                     leave=False):
            # Sample data matrix X
            X = sample_data_matrix(args.d, c=args.c)
            
            # Generate base matrices and their products
            base_matrices = sample_haar_matrices(args.d, b, args.matrix_type)
            all_products = generate_matrix_products(base_matrices, l)
            
            # Calculate the average of all matrices in all_products
            avg_matrix = np.zeros_like(all_products[0])
            for matrix in all_products:
                avg_matrix += matrix
            
            # Normalize the average matrix
            if args.normalize == 1:
                avg_matrix /= np.sqrt(len(all_products))
            
            # 1. Process average_matrix
            # Compute singular values of the average matrix
            _, eigenvalues_avg, _ = np.linalg.svd(avg_matrix)
            all_eigenvalues_avg[i].extend(eigenvalues_avg)
            
            # Calculate moments of the average matrix
            moments_avg = calculate_moments(avg_matrix, max_k=6)
            all_moments_data_avg[i].append(moments_avg)
            
            # Calculate normalized effective dimension for each r value
            for r_idx, r in enumerate(r_values):
                eff_dim_avg = empirical_effective_dim_ratio(eigenvalues_avg, r)
                all_eff_dim_data_avg[i][r_idx].append(eff_dim_avg)
            
            # 2. Compute Y = average_matrix @ X and process it
            Y = avg_matrix @ X
            
            # Compute singular values of Y
            _, eigenvalues_Y, _ = np.linalg.svd(Y)
            all_eigenvalues_Y[i].extend(eigenvalues_Y)
            
            # Calculate moments of Y
            moments_Y = calculate_moments(Y, max_k=6)
            all_moments_data_Y[i].append(moments_Y)
            
            # Calculate normalized effective dimension for Y for each r value
            for r_idx, r in enumerate(r_values):
                eff_dim_Y = empirical_effective_dim_ratio(eigenvalues_Y, r)
                all_eff_dim_data_Y[i][r_idx].append(eff_dim_Y)
    
    # Generate suffix for filenames
    suffix = "_norm" if args.normalize == 1 else ""
    suffix3 = f"_t{args.t}"
    if args.matrix_type == "orthogonal":
        suffix4 = ""
    else:
        suffix4 = f"_{args.matrix_type}"
    suffix = suffix + suffix3 + suffix4
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    # 1. Plot and save results for average_matrix
    # Plot singular value distributions
    plot_singular_value_distributions(all_eigenvalues_avg, combinations, args, suffix=suffix, 
                                     title_prefix="Singular Value Distributions of Average Matrix")
    
    # Create and save the moments comparison plot
    plot_moments_comparison(all_moments_data_avg, combinations, args, suffix=suffix,
                           title_prefix="Moments of Average Matrix")
    
    # Create and save the normalized effective dimension plot
    plot_effective_dim_ratio(all_eff_dim_data_avg, r_values, combinations, args, suffix=suffix,
                            with_mp=False)
    
    # 2. Plot and save results for Y = average_matrix @ X
    # Plot singular value distributions for Y
    plot_singular_value_distributions(all_eigenvalues_Y, combinations, args, suffix=suffix, 
                                     title_prefix="Singular Value Distributions of Y = Average Matrix @ X",
                                     filename_prefix="Y_")
    
    # Create and save the moments comparison plot for Y
    plot_moments_comparison(all_moments_data_Y, combinations, args, suffix=suffix,
                           title_prefix="Moments of Y = Average Matrix @ X",
                           filename_prefix="Y_")
    
    # Create and save the normalized effective dimension plot for Y
    plot_effective_dim_ratio(all_eff_dim_data_Y, r_values, combinations, args, suffix=suffix,
                            with_mp=True, c=args.c, title_prefix="Effective Dimension of Y = Average Matrix @ X")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare eigenvalue distributions of average matrices for different l,m combinations')
    parser.add_argument('--d', type=int, required=True, help='Dimension of base matrices')
    parser.add_argument('--t', type=int, default=32, help='Number of trials for sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--normalize', type=int, help='apply normalization', default=1)
    parser.add_argument('--matrix_type', type=str, choices=['orthogonal', 'permutation'], 
                        default='orthogonal', help='Type of random matrices to sample (orthogonal or permutation)')
    parser.add_argument('--c', type=float,default=1, help='ratio for sample data matrix: round(cd) x d matrix')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        
    plot_multiple_distributions(args)
