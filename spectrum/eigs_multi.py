import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from spectrum.eigs import sample_haar_matrices, generate_matrix_products, compute_eigenvalues, calculate_moments
from spectrum.eigs import transform_matrices
from tqdm import tqdm
from spectrum.spectrum.style import set_paper_style

def plot_moments_comparison(all_moments_data, combinations, args, suffix=None):
    """
    Plot comparison of moments for different combinations in a single figure.
    
    Parameters:
        all_moments_data (list): List of moment data for each combination
        combinations (list): List of (l, m, label) combinations
        args (argparse.Namespace): Command line arguments
    """
    # Set style
    set_paper_style()
    
    # Create a single figure
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # k values for x-axis (1 to max_k)
    max_k = 2  # As specified in requirements
    k_values = list(range(1, max_k + 1))
    
    # Colors and markers for different combinations
    colors = sns.color_palette("husl", len(combinations))
    #markers = ['o', 's', 'D', '^']
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
    #ax.set_title('Comparison of Moments for Different Configurations')
    ax.set_xticks(k_values)
    
    # Add legend
    ax.legend(loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    filename = f"moments_comp_d{args.d}{suffix}"
    
    plt.savefig(f"figures/{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/{filename}.pdf", bbox_inches='tight')
    plt.close()

def plot_wigner_quarter_circle(ax, R=2.0, num_points=1000, **kwargs):
    """
    Plot the Wigner quarter-circular distribution density function.
    
    Parameters:
        ax (matplotlib.axes.Axes): The axes to plot on
        R (float): Radius parameter (default: 2.0)
        num_points (int): Number of points to use for plotting
        **kwargs: Additional keyword arguments to pass to plt.plot()
    """
    x = np.linspace(0, R, num_points)
    density = (4/(np.pi * R**2)) * np.sqrt(R**2 - x**2)
    
    # Plot the density function
    ax.plot(x, density, **kwargs)

def plot_multiple_distributions(args):
    # Set style
    set_paper_style()
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    # Combinations to test
    combinations = [
        (1, 8, r"$\ell=1, m=8$"),
        (2, 4, r"$\ell=2, m=4$"),
        (4, 2, r"$\ell=4, m=2$"),
        (8, 1, r"$\ell=8, m=1$")
    ]
    
    # Colors for different combinations
    colors = sns.color_palette("husl", len(combinations))
    
    # Storage for moment data
    all_moments_data = [[] for _ in range(len(combinations))]
    
    # Main progress bar for combinations
    for i, ((l, m, label), color, ax) in enumerate(tqdm(zip(combinations, colors, axes), 
                                        total=len(combinations), 
                                        desc="Processing configurations")):
        # Calculate b = 2^m
        b = 2 ** m
        
        # Initialize storage for eigenvalues
        all_eigenvalues = []
        
        # Inner progress bar for trials
        for _ in tqdm(range(args.t), 
                     desc=f"Computing {label}", 
                     leave=False):
            base_matrices = sample_haar_matrices(args.d, b, args.matrix_type)
            all_products = generate_matrix_products(base_matrices, l)
            X = transform_matrices(all_products, transform=args.trans)
            
            ### Normalize
            # v1
            #eigenvalues = eigenvalues / np.sqrt(len(matrices))
            #arxiv v1
            #X /=np.sqrt(len(matrices))
            # v2
            X /= np.sqrt(np.sqrt(len(all_products)))
            _, eigenvalues, _ = np.linalg.svd(X)
            all_eigenvalues.extend(eigenvalues)
            
            moments = calculate_moments(X, max_k=4)
            all_moments_data[i].append(moments)
            
        # Convert to numpy array
        all_eigenvalues = np.array(all_eigenvalues)
        
        # Plot density histogram
        sns.histplot(data=all_eigenvalues, 
                    bins=100,
                    stat='density',
                    element='step',
                    fill=True,
                    linewidth=2,
                    color=color,
                    ax=ax)

        # If ℓ=1 and plot_density is enabled, overlay the Wigner quarter-circular distribution
        if l == 1 and args.plot_density:
            # Plot Wigner quarter-circular distribution with the specified radius (or default R=2.0)
            R = args.density_radius if hasattr(args, 'density_radius') else 2.0
            plot_wigner_quarter_circle(
                ax, 
                R=R,
                color='red',
                linestyle='--',
                linewidth=2,
                label=None
            )

        # Customize each subplot
        ax.set_xlabel('Singular Value')
        if ax.get_position().x0 < 0.1:  # Only leftmost plot
            ax.set_ylabel('Density')
        ax.set_title(label)
        
        # Add statistics text
        #stats_text = f'n = {len(all_eigenvalues)}\n'
        stats_text = f'μ = {np.mean(all_eigenvalues):.3f}\n'
        stats_text += f'σ = {np.std(all_eigenvalues):.3f}'
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))

    # Adjust layout
    plt.tight_layout()
    
    # Save plots with different formats
    suffix = "" #"_tr" if args.trace == 1 else ""
    suffix2 = f"_{args.trans}" if args.trans is not None else ""
    suffix3 = f"_t{args.t}"
    if args.matrix_type=="orthogonoal":
        suffix4=""
    else:
        suffix4=f"_{args.matrix_type}"
    if args.plot_density is True:
        suffix5=f"_df"
    else:
        suffix5=""

    suffix = suffix + suffix2 + suffix3 +suffix4 + suffix5
    
    # Ensure figures directory exists
    os.makedirs("figures", exist_ok=True)
    
    filename=f"eigs_comp_d{args.d}{suffix}"
    
    plt.savefig(f"figures/{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"figures/{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    # Create and save the moments comparison plot
    plot_moments_comparison(all_moments_data, combinations, args, suffix=suffix)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compare eigenvalue distributions for different l,m combinations')
    parser.add_argument('--d', type=int, required=True, help='Dimension of base matrices')
    parser.add_argument('--t', type=int, default=32, help='Number of trials for sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    #parser.add_argument('--trace', type=int, help='take normalized trace of matrices', default=0)
    parser.add_argument('--trans', type=str, help='transform of block matrices', default="06243517")
    parser.add_argument('--matrix_type', type=str, choices=['orthogonal', 'permutation'], 
                        default='orthogonal', help='Type of random matrices to sample (orthogonal or permutation)')
    parser.add_argument('--plot_density', action='store_true', 
                        help='Plot Wigner quarter-circular density function (for ℓ=1)')
    parser.add_argument('--density_radius', type=float, default=2.0,
                        help='Radius parameter R for Wigner quarter-circular distribution (default: %(default)s)')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        
    plot_multiple_distributions(args)
