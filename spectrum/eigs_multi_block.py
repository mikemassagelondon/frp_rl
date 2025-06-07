import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import ortho_group
from tqdm import tqdm
from einops import rearrange
from spectrum.style import set_paper_style
from eigs_multi_tensor import generate_block_matrix


def plot_block_matrix_distributions(args):
    """
    Plot and compare singular values of generate_block_matrix(q, d, matrix_type) 
    for q = p^4, p^2, p
    
    Parameters:
        args: Command-line arguments
    """
    # Set style
    set_paper_style()
    
    # Create figure with 3 subplots (for p^4, p^2, p)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Configurations to test
    configs = [
        (args.p**4, rf"$q = p^4 = {{{args.p}}}^4$"),
        (args.p**2, rf"$q = p^2 = {{{args.p}}}^2$"),
        (args.p, rf"$q = p = {args.p}$")
    ]
    
    # Colors for different configurations
    colors = sns.color_palette("husl", len(configs))
    
    # Process each configuration
    for (q, label), color, ax in zip(configs, colors, axes):
        # Initialize storage for singular values
        all_singular_values = []
        
        # Inner progress bar for trials
        for _ in tqdm(range(args.t), 
                     desc=f"Computing {label}", 
                     leave=False):
            # Generate block matrix
            matrix = generate_block_matrix(q, args.d, args.matrix_type)
            
            # Reshape and compute singular values
            matrix_flat = rearrange(matrix, 'a b c d -> (a c) (b d)')
            
            # Apply normalization if specified
            if args.normalize:
                # Normalize by sqrt(q) to account for matrix size
                matrix_flat /= np.sqrt(q)
            
            # Compute singular values
            _, singular_values, _ = np.linalg.svd(matrix_flat)
            
            all_singular_values.extend(singular_values)
        
        # Convert to numpy array
        all_singular_values = np.array(all_singular_values)
        
        # Plot distribution
        sns.histplot(data=all_singular_values, 
                    bins=100,
                    stat='density',
                    element='step',
                    fill=True,
                    linewidth=2,
                    color=color,
                    ax=ax)
        
        # Customize plot
        ax.set_title(label)
        ax.set_xlabel('Singular Value')
        if ax.get_position().x0 < 0.1:  # Only leftmost plot
            ax.set_ylabel('Density')
        
        # Add statistics
        stats_text = f'μ = {np.mean(all_singular_values):.3f}\n'
        stats_text += f'σ = {np.std(all_singular_values):.3f}'
        
        ax.text(0.95, 0.95, stats_text,
                transform=ax.transAxes,
                horizontalalignment='right',
                verticalalignment='top',
                bbox=dict(facecolor='white', alpha=0.8))
    
    # Adjust layout and save
    plt.tight_layout()
    
    # Create filename with relevant parameters
    filename = f"figures/eigs_block_p{args.p}_d{args.d}_t{args.t}"
    if args.normalize:
        filename += "_norm"
    if args.matrix_type != "orthogonal":
        filename += f"_{args.matrix_type}"
    
    # Save as PNG and PDF
    plt.savefig(f"{filename}.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"{filename}.pdf", bbox_inches='tight')
    plt.close()
    
    print(f"Plots saved as {filename}.png and {filename}.pdf")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot singular value distributions for block matrices with different q values')
    parser.add_argument('--p', type=int, default=2, help='Base parameter p (will compare q=p^4, p^2, p)')
    parser.add_argument('--d', type=int, default=16, help='Dimension of each Haar random matrix')
    parser.add_argument('--t', type=int, default=32, help='Number of trials for sampling')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--matrix_type', type=str, choices=['orthogonal', 'permutation'], 
                        default='orthogonal', help='Type of random matrices to sample')
    parser.add_argument('--normalize', action='store_true', 
                        help='Normalize matrices by sqrt(q) before computing singular values')
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        
    plot_block_matrix_distributions(args)
