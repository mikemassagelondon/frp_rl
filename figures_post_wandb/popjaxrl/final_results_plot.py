#!/usr/bin/env python3
"""
Final Results Plotting Script

This script implements the final results plotting requirements from the plotting plan:
1. Use the preprocessed data from the preprocess_and_time_series.py script
2. Create final results plots:
   - For each env, create 2 subplots (2 arch types)
   - In each subplot, set depth as the x-axis and plot the values of the two metrics at the last time step
   - Include dim and eval_method in the filename
   - Label y-axis as MMER and x-axis as Length with a log2-scale
   - Save as PNG and PDF
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict


def load_preprocessed_data(
    cache_dir: str = "wandb_cache",
    name_prefix: str = "preprocessed_data",
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
    #plt.style.use('seaborn-v0_8-whitegrid')
    #sns.set_theme(style="whitegrid", font_scale=font_scale)
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


def create_final_results_plot(
    data: pd.DataFrame,
    env: str,
    dim: int,
    eval_method: str,
    depths: List[int] = [1, 2, 4, 8],
    archs: List[str] = ["gru", "s5"],
    metrics: List[str] = ["metric", "eval_metric"],
    show_legend: bool = False,
) -> plt.Figure:
    """
    Create a final results plot with 2 subplots (2 arch types).
    
    Args:
        data: DataFrame containing the preprocessed data
        env: Environment name
        dim: Dimension value
        eval_method: Evaluation method
        depths: List of depth values to plot
        archs: List of architecture types
        metrics: List of metric names
        
    Returns:
        Matplotlib figure
    """
    set_paper_style()
    # Filter data for the specified env, dim, and eval_method
    filtered_data = data[
        (data["env"] == env) & 
        (data["dim"] == dim) & 
        (data["eval_method"] == eval_method)
    ]
    
    # Create a 1x2 grid of subplots
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    # Use a color palette with enough colors for metrics
    #colors = sns.color_palette("deep", len(metrics))
    colors = sns.color_palette("husl", len(metrics))
    
    # Create subplots
    for i, arch in enumerate(archs):
        # Filter data for this arch
        arch_data = filtered_data[filtered_data["arch"] == arch]
        
        # Create a list to store the final values for each metric and depth
        final_values = []
        
        # Process each metric
        for j, metric in enumerate(metrics):
            # Filter data for this metric
            metric_data = arch_data[arch_data["metric"] == metric]
            
            # Extract the final value for each depth
            for depth in depths:
                # Filter data for this depth
                depth_data = metric_data[metric_data["depth"] == depth]
                
                if len(depth_data) == 0:
                    continue
                
                # Get the first row (should be only one row per depth and metric)
                row = depth_data.iloc[0]
                
                # Get the final value (last value in the mean array)
                final_value = row["mean"][-1]
                
                # Get the standard deviation at the final time step
                final_std = row["std"][-1]
                
                # Add to final values
                final_values.append({
                    "depth": depth,
                    "metric": metric,
                    "value": final_value,
                    "std": final_std
                })
        
        # Convert to DataFrame
        final_df = pd.DataFrame(final_values)
        
        # Plot each metric
        for j, metric in enumerate(metrics):
            # Filter data for this metric
            metric_df = final_df[final_df["metric"] == metric]
            
            if len(metric_df) == 0:
                continue
            
            # Sort by depth
            metric_df = metric_df.sort_values(by="depth")

            label=f"{arch}: ".upper()
            if metric =="metric":
                label+= "Train"
                fmt="o--"
                linestyle="--"
            elif metric == "eval_metric":
                label+= "Test" 
                fmt="o-"
                linestyle="-"
            color_idx=i
            """
            # Plot line with error bars
            ax.errorbar(
                metric_df["depth"],
                metric_df["value"],
                yerr=metric_df["std"],
                fmt=fmt,
                capsize=5,
                linewidth=2,
                markersize=8,
                color=colors[i],
                ecolor=colors[i],
                elinewidth=1.5,
                capthick=1.5,
                label=label
            )
            """

            # Plot mean line
            ax.plot(
                metric_df["depth"],
                metric_df["value"],
                label=label,
                color=colors[color_idx],
                linewidth=2,
                linestyle=linestyle
            )
            
            # Plot std as shaded area
            ax.fill_between(
                metric_df["depth"],
                metric_df["value"] - metric_df["std"],
                metric_df["value"] + metric_df["std"],
                alpha=0.2,
                color=colors[color_idx]
            )
        
        # Set axis labels
        ax.set_xlabel("Length")
        
        # Set log2 scale for x-axis
        ax.set_xscale('log', base=2)
        
        # Set x-ticks to the actual depth values
        ax.set_xticks(depths)
        ax.set_xticklabels(depths)
        
        # Set subplot title
        ax.set_title(f"{env}".capitalize())
        
        # Add legend only if show_legend is True
        if show_legend:
            ax.legend(loc='best')
        ax.set_ylabel("MMER")
    
    # Add overall title (only for debug)
    #plt.suptitle(f"Environment: {env}, dim={dim}, eval_method={eval_method}", fontsize=5) 
    
    plt.tight_layout()
    return fig


def create_latex_summary_tables(
    data: pd.DataFrame,
    envs: List[str],
    dim: int,
    eval_method: str,
    depths: List[int] = [1, 2, 4, 8],
    archs: List[str] = ["gru", "s5"],
    metrics: List[str] = ["metric", "eval_metric"],
    output_dir: str = "tables",
    name_prefix: str = "summary_tables",
) -> None:
    """
    Create LaTeX summary tables for a research paper.
    
    Creates two tables:
    1. Performance table with Architecture (GRU/S5) × Method (Baseline/FRA) as rows
       and Environment × Metric as columns
    2. Optimal depth table showing ℓ* for each environment and architecture
    
    Args:
        data: DataFrame containing the preprocessed data
        envs: List of environment names
        dim: Dimension value
        eval_method: Evaluation method
        depths: List of depth values
        archs: List of architecture types
        metrics: List of metric names
        output_dir: Directory to save the LaTeX tables
        name_prefix: Prefix for the filename
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store optimal depths and their performance
    optimal_depths = {}
    performance_data = {}
    
    # Process data for each environment and architecture
    for env in envs:
        optimal_depths[env] = {}
        performance_data[env] = {}
        
        # Filter data for the specified env, dim, and eval_method
        filtered_data = data[
            (data["env"] == env) & 
            (data["dim"] == dim) & 
            (data["eval_method"] == eval_method)
        ]
        
        for arch in archs:
            # Filter data for this architecture
            arch_data = filtered_data[filtered_data["arch"] == arch]
            
            # Dictionary to store performance for each depth and metric
            depth_performance = defaultdict(dict)
            
            # Process each metric
            for metric in metrics:
                # Filter data for this metric
                metric_data = arch_data[arch_data["metric"] == metric]
                
                # Extract performance for each depth
                for depth in depths:
                    # Filter data for this depth
                    depth_data = metric_data[metric_data["depth"] == depth]
                    
                    if len(depth_data) == 0:
                        continue
                    
                    # Get the first row (should be only one row per depth and metric)
                    row = depth_data.iloc[0]
                    
                    # Get the final value (last value in the mean array)
                    final_value = row["mean"][-1]
                    
                    # Get the standard deviation at the final time step
                    final_std = row["std"][-1]
                    
                    # Store performance
                    depth_performance[depth][metric] = (final_value, final_std)
            
            # Find optimal depth based on test metric (eval_metric)
            # Skip depth=1 (baseline) when finding optimal depth for FRA
            best_depth = 1  # Default to baseline
            best_performance = float('-inf')
            
            for depth in depths:
                if depth == 1:  # Skip baseline when finding optimal FRA depth
                    continue
                    
                if 'eval_metric' in depth_performance[depth]:
                    performance = depth_performance[depth]['eval_metric'][0]  # Get mean value
                    if performance > best_performance:
                        best_performance = performance
                        best_depth = depth
            
            # Store optimal depth
            optimal_depths[env][arch] = best_depth
            
            # Store performance data
            performance_data[env][arch] = {
                'baseline': {
                    metric: depth_performance[1][metric] if 1 in depth_performance and metric in depth_performance[1] else (None, None)
                    for metric in metrics
                },
                'fra': {
                    metric: depth_performance[best_depth][metric] if best_depth in depth_performance and metric in depth_performance[best_depth] else (None, None)
                    for metric in metrics
                }
            }
    
    # Create LaTeX table for performance
    performance_table = "\\begin{table}[htbp]\n"
    performance_table += "\\centering\n"
    performance_table += "\\caption{Performance comparison across environments}\n"
    performance_table += "\\label{tab:performance}\n"
    
    # Calculate column width based on number of environments
    col_spec = "l" + "cc" * len(envs)
    performance_table += f"\\begin{{tabular}}{{{col_spec}}}\n"
    performance_table += "\\toprule\n"
    
    # Create header row with environment names
    header_row = " & "
    for env in envs:
        # Format environment name: replace underscores with spaces and capitalize each word
        formatted_env = ' '.join(word.capitalize() for word in env.split('_'))
        header_row += f"\\multicolumn{{2}}{{c}}{{{formatted_env}}} & "
    header_row = header_row.rstrip(" & ") + " \\\\\n"
    performance_table += header_row
    
    # Create subheader row with metric names
    subheader_row = "Method & "
    for _ in envs:
        subheader_row += "Train & Test & "
    subheader_row = subheader_row.rstrip(" & ") + " \\\\\n"
    
    # Add cmidrule separators
    cmidrules = ""
    col_start = 2
    for _ in envs:
        cmidrules += f"\\cmidrule(lr){{{col_start}-{col_start+1}}} "
        col_start += 2
    
    performance_table += cmidrules + "\n"
    performance_table += subheader_row
    performance_table += "\\midrule\n"
    
    # Find best performance for each environment and metric to highlight
    best_performance = {}
    for env in envs:
        best_performance[env] = {}
        for metric in metrics:
            best_val = float('-inf')
            for arch in archs:
                for method in ['baseline', 'fra']:
                    val, _ = performance_data[env][arch][method][metric]
                    if val is not None and val > best_val:
                        best_val = val
            best_performance[env][metric] = best_val
    
    # Add rows for each architecture and method
    for arch in archs:
        for method, depth_label in [('baseline', '(\\ell=1)'), ('fra', '(\\ell=\\ell^*)')]:
            # Use shorter labels for architecture and method combinations
            if arch == "gru" and method == "baseline":
                row_label = f"GRU-B {depth_label}"
            elif arch == "gru" and method == "fra":
                row_label = f"GRU-F {depth_label}"
            elif arch == "s5" and method == "baseline":
                row_label = f"S5-B {depth_label}"
            elif arch == "s5" and method == "fra":
                row_label = f"S5-F {depth_label}"
            else:
                row_label = f"{arch.upper()}-{method[0].upper()} {depth_label}"
            
            row = f"{row_label} & "
            
            for env in envs:
                for metric in metrics:
                    val, std = performance_data[env][arch][method][metric]
                    
                    if val is None:
                        cell = "-- & "
                    else:
                        # Format with 2 decimal places and proper math mode
                        if abs(val - best_performance[env][metric]) < 1e-6:
                            # Bold if this is the best performance using mathbf
                            formatted = f"$\\mathbf{{{val:.2f}\\pm{std:.2f}}}$"
                        else:
                            formatted = f"${val:.2f}\\pm{std:.2f}$"
                        
                        cell = f"{formatted} & "
                    
                    row += cell
            
            row = row.rstrip(" & ") + " \\\\\n"
            performance_table += row
    
    performance_table += "\\bottomrule\n"
    performance_table += "\\end{tabular}\n"
    performance_table += "\\end{table}\n"
    
    # Create LaTeX table for optimal depths
    depth_table = "\\begin{table}[htbp]\n"
    depth_table += "\\centering\n"
    depth_table += "\\caption{Optimal depth (\\ell^*) for each environment and architecture}\n"
    depth_table += "\\label{tab:optimal_depth}\n"
    
    # Calculate column width based on number of environments
    col_spec = "l" + "c" * len(envs)
    depth_table += f"\\begin{{tabular}}{{{col_spec}}}\n"
    depth_table += "\\toprule\n"
    
    # Create header row with environment names
    header_row = "Arch & "
    for env in envs:
        # Format environment name: replace underscores with spaces and capitalize each word
        formatted_env = ' '.join(word.capitalize() for word in env.split('_'))
        header_row += f"{formatted_env} & "
    header_row = header_row.rstrip(" & ") + " \\\\\n"
    depth_table += header_row
    
    depth_table += "\\midrule\n"
    
    # Add rows for each architecture
    for arch in archs:
        row = f"{arch.upper()} & "
        
        for env in envs:
            row += f"{optimal_depths[env][arch]} & "
        
        row = row.rstrip(" & ") + " \\\\\n"
        depth_table += row
    
    depth_table += "\\bottomrule\n"
    depth_table += "\\end{tabular}\n"
    depth_table += "\\end{table}\n"
    
    # Create LaTeX table for test-only performance
    test_table = "\\begin{table}[htbp]\n"
    test_table += "\\centering\n"
    test_table += "\\caption{Test performance comparison across environments}\n"
    test_table += "\\label{tab:test_performance}\n"
    
    # Calculate column width based on number of environments
    col_spec = "l" + "c" * len(envs)
    test_table += f"\\begin{{tabular}}{{{col_spec}}}\n"
    test_table += "\\toprule\n"
    
    # Create header row with environment names
    header_row = "Method & "
    for env in envs:
        # Format environment name: replace underscores with spaces and capitalize each word
        formatted_env = ' '.join(word.capitalize() for word in env.split('_'))
        header_row += f"{formatted_env} & "
    header_row = header_row.rstrip(" & ") + " \\\\\n"
    test_table += header_row
    
    test_table += "\\midrule\n"
    
    # Find best test performance for each environment to highlight
    best_test_performance = {}
    for env in envs:
        best_val = float('-inf')
        for arch in archs:
            for method in ['baseline', 'fra']:
                val, _ = performance_data[env][arch][method]['eval_metric']
                if val is not None and val > best_val:
                    best_val = val
        best_test_performance[env] = best_val
    
    # Add rows for each architecture and method
    for method, depth_label in [('baseline', '(\\ell=1)'), ('fra', '(\\ell=\\ell^*)')]:
        for arch in archs:

            # Use shorter labels for architecture and method combinations
            if arch == "gru" and method == "baseline":
                row_label = f"B.(GRU)"
            elif arch == "gru" and method == "fra":
                row_label = f"F.(GRU)"
            elif arch == "s5" and method == "baseline":
                row_label = f"B.(S5)"
            elif arch == "s5" and method == "fra":
                row_label = f"F.(S5)"
            else:
                row_label = f"{arch.upper()}-{method[0].upper()} {depth_label}"
            
            row = f"{row_label} & "
            
            for env in envs:
                val, std = performance_data[env][arch][method]['eval_metric']
                
                if val is None:
                    cell = "-- & "
                else:
                    # Format with 2 decimal places and proper math mode
                    if abs(val - best_test_performance[env]) < 1e-6:
                        # Bold if this is the best performance using mathbf
                        formatted = f"$\\mathbf{{{val:.2f}\\pm{std:.2f}}}$"
                    else:
                        formatted = f"${val:.2f}\\pm{std:.2f}$"
                    
                    cell = f"{formatted} & "
                
                row += cell
            
            row = row.rstrip(" & ") + " \\\\\n"
            test_table += row
    
    test_table += "\\bottomrule\n"
    test_table += "\\end{tabular}\n"
    test_table += "\\end{table}\n"
    
    # Save tables to files
    tables_filename = f"{name_prefix}_{dim}_{eval_method}.tex"
    with open(os.path.join(output_dir, tables_filename), 'w') as f:
        f.write(performance_table)
        f.write("\n\n")
        f.write(test_table)
        f.write("\n\n")
        f.write(depth_table)
    
    print(f"Saved LaTeX tables to {os.path.join(output_dir, tables_filename)}")


def create_standalone_legend(
    archs: List[str] = ["gru", "s5"],
    metrics: List[str] = ["metric", "eval_metric"],
    font_scale: float = 2,
) -> plt.Figure:
    """
    Create a standalone figure with only the legend.
    
    Args:
        archs: List of architecture types
        metrics: List of metric names
        font_scale: Scale factor for font sizes
        
    Returns:
        Matplotlib figure containing only the legend
    """
    set_paper_style(font_scale)
    
    # Create a figure with no axes
    fig = plt.figure(figsize=(16, 1))
    
    # Use a color palette with enough colors for metrics
    colors = sns.color_palette("husl", len(metrics))
    
    # Create dummy lines for the legend
    legend_handles = []
    legend_labels = []
    
    # Add entries for each architecture and metric combination
    for i, arch in enumerate(archs):
        for j, metric in enumerate(metrics):
            if metric == "metric":
                label = "Train"
                linestyle = "--"
            elif metric == "eval_metric":
                label = "ICL-Test"
                linestyle = "-"
            label += f"({arch})".upper()
            
            # Create a dummy line for the legend
            line = plt.Line2D([0], [0], color=colors[i], linewidth=2, linestyle=linestyle)
            legend_handles.append(line)
            legend_labels.append(label)
    
    # Add the legend to the figure
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        ncol=len(archs) * len(metrics),  # Display in one row
        frameon=False,
        fontsize=12 * font_scale,
        mode='expand',
        bbox_to_anchor=(0, 0.5, 1, 0),
    )
    
    plt.tight_layout()
    return fig


def save_plot(
    fig: plt.Figure,
    output_dir: str = "figures/wandb_plots",
    name_prefix: str = "final_results",
    env: str = "",
    dim: int = 0,
    eval_method: str = "",
) -> None:
    """
    Save plot as PNG and PDF.
    
    Args:
        fig: Matplotlib figure to save
        output_dir: Directory to save the plot
        name_prefix: Prefix for the filename
        env: Environment name
        dim: Dimension value
        eval_method: Evaluation method
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a descriptive filename
    filename = f"{name_prefix}_{env}_dim{dim}_{eval_method}"
    
    # Save as PNG
    fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    
    # Save as PDF
    fig.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
    
    print(f"Saved plot to {os.path.join(output_dir, filename)}.png")


def save_standalone_legend(
    fig: plt.Figure,
    output_dir: str = "figures/wandb_plots",
    name_prefix: str = "legend_final_results",
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


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Create final results plots and LaTeX tables.")
    
    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the sweep to analyze")
    parser.add_argument("--envs", type=str, required=True, help="Comma-separated list of environments to plot")
    parser.add_argument("--dims", type=str, default="64", help="Comma-separated list of dimensions to plot")
    parser.add_argument("--eval_methods", type=str, default="in_context", 
                        help="Comma-separated list of evaluation methods to plot")
    parser.add_argument("--depths", type=str, default="1,2,4,8", help="Comma-separated list of depths to plot")
    parser.add_argument("--archs", type=str, default="gru,s5", help="Comma-separated list of architectures to plot")
    parser.add_argument("--metrics", type=str, default="metric,eval_metric", 
                        help="Comma-separated list of metrics to plot")
    parser.add_argument("--output_dir", type=str, default="figures/wandb_plots", help="Directory to save plots")
    parser.add_argument("--tables_dir", type=str, default="tables", help="Directory to save LaTeX tables")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache", help="Directory where preprocessed data is stored")
    parser.add_argument("--generate_tables", action="store_true", help="Generate LaTeX tables for the paper")
    parser.add_argument("--show_legend", action="store_true", help="Show legend in individual plots")
    parser.add_argument("--legend_only", action="store_true", help="Generate only the standalone legend file without creating plots")
    
    args = parser.parse_args()
    
    # Parse envs, dims, eval_methods, depths, archs, and metrics
    envs = args.envs.split(',')
    dims = [int(dim) for dim in args.dims.split(',')]
    eval_methods = args.eval_methods.split(',')
    depths = [int(depth) for depth in args.depths.split(',')]
    archs = args.archs.split(',')
    metrics = args.metrics.split(',')
    
    # Load preprocessed data
    processed_df = load_preprocessed_data(
        cache_dir=args.cache_dir,
        name_prefix="preprocessed_data",
        sweep_id=args.sweep_id
    )
    
    # Create final results plots for each env, dim, and eval_method (skip if legend_only is True)
    if not args.legend_only:
        for env in envs:
            for dim in dims:
                for eval_method in eval_methods:
                    # Create final results plot
                    fig = create_final_results_plot(
                        data=processed_df,
                        env=env,
                        dim=dim,
                        eval_method=eval_method,
                        depths=depths,
                        archs=archs,
                        metrics=metrics,
                        show_legend=args.show_legend
                    )
                    
                    # Save plot
                    save_plot(
                        fig=fig,
                        output_dir=args.output_dir,
                        name_prefix="final_results",
                        env=env,
                        dim=dim,
                        eval_method=eval_method
                    )
                    
                    plt.close(fig)
    
    # Create and save standalone legend
    legend_fig = create_standalone_legend(
        archs=archs,
        metrics=metrics
    )
    
    # Save standalone legend
    save_standalone_legend(
        fig=legend_fig,
        output_dir=args.output_dir
    )
    
    plt.close(legend_fig)
    
    # Generate LaTeX tables if requested
    if args.generate_tables:
        for dim in dims:
            for eval_method in eval_methods:
                create_latex_summary_tables(
                    data=processed_df,
                    envs=envs,
                    dim=dim,
                    eval_method=eval_method,
                    depths=depths,
                    archs=archs,
                    metrics=metrics,
                    output_dir=args.tables_dir,
                    name_prefix="summary_tables"
                )


if __name__ == "__main__":
    main()
