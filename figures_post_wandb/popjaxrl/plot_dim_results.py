#!/usr/bin/env python3
"""
Dimension Plotting Script

This script implements the dimension-based plotting requirements:
1. Load the preprocessed data from the preprocess_for_dim_plots.py script
2. Create plots with:
   - x-axis: dimension (dim)
   - y-axis: preprocessed eval_metric
   - Multiple lines in each plot for different depths (1, 2, 4, 8)
   - Separate plots for each combination of arch, env, and eval_method
3. Save plots as PNG and PDF
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def load_preprocessed_data(
    cache_dir: str = "wandb_cache",
    name_prefix: str = "preprocessed_dim_data",
    sweep_id: str = "",
) -> pd.DataFrame:
    """
    Load preprocessed data from a JSON file.
    
    Args:
        cache_dir: Directory where the data is stored
        name_prefix: Prefix of the filename
        sweep_id: ID of the sweep
        
    Returns:
        DataFrame containing the preprocessed data
    """
    # Create a descriptive filename
    filename = f"{name_prefix}_{sweep_id}.json"
    file_path = os.path.join(cache_dir, filename)
    
    # Check if file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Preprocessed data file not found: {file_path}")
    
    # Load data from JSON
    with open(file_path, 'r') as f:
        data_dict = json.load(f)
    
    # Convert to DataFrame
    df = pd.DataFrame(data_dict)
    
    return df


def set_paper_style(font_scale: float = 1.5):
    """
    Set matplotlib style for paper-ready figures.
    
    Args:
        font_scale: Scale factor for font sizes
    """
    sns.set_theme(style='whitegrid', font_scale=font_scale)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.labelsize': 14 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'xtick.labelsize': 12 * font_scale,
        'ytick.labelsize': 12 * font_scale,
        'legend.fontsize': 12 * font_scale,
        'figure.figsize': (8, 6),
        'figure.dpi': 400,
        'savefig.dpi': 1200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        'axes.grid': True,
        'grid.alpha': 0.5,
    })


def create_dim_plot(
    data: pd.DataFrame,
    env: str,
    eval_method: str,
    depths: List[int] = [1, 2, 4, 8],
    archs: List[str] = ["gru", "s5"],
    metric: str = "eval_metric",
    show_legend: bool = False,
) -> plt.Figure:
    """
    Create a plot with dimension on the x-axis and eval_metric on the y-axis.
    
    Args:
        data: DataFrame containing the preprocessed data
        env: Environment name
        eval_method: Evaluation method
        depths: List of depth values to plot
        archs: List of architecture types
        metric: Metric name to plot
        
    Returns:
        Matplotlib figure
    """
    set_paper_style()
    
    # Filter data for the specified env, eval_method, and metric
    filtered_data = data[
        (data["env"] == env) & 
        (data["eval_method"] == eval_method) &
        (data["metric"] == metric)
    ]
    
    # Create a figure with subplots for each architecture
    fig, axes = plt.subplots(1, len(archs), figsize=(8 * len(archs), 6), sharey=True)
    
    # If only one architecture, make axes a list for consistent indexing
    if len(archs) == 1:
        axes = [axes]
    
    # Use a color palette with enough colors for depths
    colors = sns.color_palette("husl", len(depths))
    
    # Create subplots for each architecture
    for i, arch in enumerate(archs):
        ax = axes[i]
        
        # Filter data for this architecture
        arch_data = filtered_data[filtered_data["arch"] == arch]
        
        # Plot each depth
        for j, depth in enumerate(depths):
            # Filter data for this depth
            depth_data = arch_data[arch_data["depth"] == depth]
            
            if len(depth_data) == 0:
                continue
            
            # Sort by dimension
            depth_data = depth_data.sort_values(by="dim")
            
            # Plot line with error bars
            ax.errorbar(
                depth_data["dim"],
                depth_data["mean"],
                yerr=depth_data["std"],
                fmt="o-",
                capsize=5,
                linewidth=2,
                markersize=8,
                color=colors[j],
                ecolor=colors[j],
                elinewidth=1.5,
                capthick=1.5,
                label=r"$\ell=$"+f"{depth}"
            )
        
        # Set axis labels
        ax.set_xlabel("Dimension")
        if i == 0:
            ax.set_ylabel("MMER")
        
        # Set subplot title
        ax.set_title(f"{arch.upper()}")
        
        # Add legend only if show_legend is True
        if show_legend:
            ax.legend(loc='best')
    
    # Add overall title
    plt.suptitle(f"{env.capitalize()} - {eval_method.capitalize()}", fontsize=16)
    
    plt.tight_layout()
    return fig


def create_standalone_legend(
    depths: List[int] = [1, 2, 4, 8],
    font_scale: float = 1.5,
) -> plt.Figure:
    """
    Create a standalone figure with only the legend.
    
    Args:
        depths: List of depth values to include in the legend
        font_scale: Scale factor for font sizes
        
    Returns:
        Matplotlib figure containing only the legend
    """
    set_paper_style(font_scale)
    
    # Create a figure with no axes
    fig = plt.figure(figsize=(8, 1))
    
    # Use a color palette with enough colors for depths
    colors = sns.color_palette("husl", len(depths))
    
    # Create dummy lines for the legend
    legend_handles = []
    legend_labels = []
    
    # Add entries for each depth
    for j, depth in enumerate(depths):
        # Create a dummy line for the legend
        line = plt.Line2D([0], [0], color=colors[j], linewidth=2, marker='o', markersize=8)
        legend_handles.append(line)
        legend_labels.append(r"$\ell=$"+f"{depth}")
    
    # Add the legend to the figure
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        ncol=len(depths),  # Display in one row
        frameon=False,
        fontsize=12 * font_scale,
        mode='expand',
        bbox_to_anchor=(0, 0.5, 1, 0),
    )
    
    plt.tight_layout()
    return fig


def save_standalone_legend(
    fig: plt.Figure,
    output_dir: str = "figures/wandb_plots",
    name_prefix: str = "legend_dim_results",
) -> None:
    """
    Save standalone legend as PDF.
    
    Args:
        fig: Matplotlib figure containing only the legend
        output_dir: Directory to save the legend
        name_prefix: Prefix for the filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a descriptive filename
    filename = f"{name_prefix}"
    
    # Save as PDF
    fig.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
    
    print(f"Saved legend to {os.path.join(output_dir, filename)}.pdf")


def save_plot(
    fig: plt.Figure,
    output_dir: str = "figures/wandb_plots",
    name_prefix: str = "dim_results",
    env: str = "",
    eval_method: str = "",
) -> None:
    """
    Save plot as PNG and PDF.
    
    Args:
        fig: Matplotlib figure to save
        output_dir: Directory to save the plot
        name_prefix: Prefix for the filename
        env: Environment name
        eval_method: Evaluation method
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a descriptive filename
    filename = f"{name_prefix}_{env}_{eval_method}"
    
    # Save as PNG
    fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    
    # Save as PDF
    fig.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
    
    print(f"Saved plot to {os.path.join(output_dir, filename)}.png")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Create dimension-based plots.")
    
    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the sweep to analyze")
    parser.add_argument("--envs", type=str, required=True, help="Comma-separated list of environments to plot")
    parser.add_argument("--eval_methods", type=str, default="tiling,padding", 
                        help="Comma-separated list of evaluation methods to plot")
    parser.add_argument("--depths", type=str, default="1,2,4,8", help="Comma-separated list of depths to plot")
    parser.add_argument("--archs", type=str, default="gru,s5", help="Comma-separated list of architectures to plot")
    parser.add_argument("--metric", type=str, default="eval_metric", help="Metric to plot")
    parser.add_argument("--output_dir", type=str, default="figures/wandb_plots", help="Directory to save plots")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache", help="Directory where preprocessed data is stored")
    parser.add_argument("--show_legend", action="store_true", help="Show legend in individual plots")
    parser.add_argument("--legend_only", action="store_true", help="Generate only the standalone legend file without creating plots")
    
    args = parser.parse_args()
    
    # Parse envs, eval_methods, depths, and archs
    envs = args.envs.split(',')
    eval_methods = args.eval_methods.split(',')
    depths = [int(depth) for depth in args.depths.split(',')]
    archs = args.archs.split(',')
    
    # Load preprocessed data
    processed_df = load_preprocessed_data(
        cache_dir=args.cache_dir,
        name_prefix="preprocessed_dim_data",
        sweep_id=args.sweep_id
    )
    
    # Create dimension plots for each env and eval_method (skip if legend_only is True)
    if not args.legend_only:
        for env in envs:
            for eval_method in eval_methods:
                # Create dimension plot
                fig = create_dim_plot(
                    data=processed_df,
                    env=env,
                    eval_method=eval_method,
                    depths=depths,
                    archs=archs,
                    metric=args.metric,
                    show_legend=args.show_legend
                )
                
                # Save plot
                save_plot(
                    fig=fig,
                    output_dir=args.output_dir,
                    name_prefix="dim_results",
                    env=env,
                    eval_method=eval_method
                )
                
                plt.close(fig)
    
    # Create and save standalone legend
    legend_fig = create_standalone_legend(
        depths=depths
    )
    
    # Save standalone legend
    save_standalone_legend(
        fig=legend_fig,
        output_dir=args.output_dir
    )
    
    plt.close(legend_fig)
    
    print("Plotting complete!")


if __name__ == "__main__":
    main()
