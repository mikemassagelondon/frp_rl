#!/usr/bin/env python3
"""
Preprocessing and Time Series Plotting Script

This script implements the preprocessing and time series plotting requirements from the plotting plan:
1. Download all runs from the specified sweep
2. Preprocess metrics:
   - Take the maximum of metrics along the time dimension
   - Group metrics by arch, dim, eval_method, and depth
   - Check that each group has results for 10 seeds
   - Calculate mean/std for each group
3. Create time series plots:
   - For each env, create 4 subplots (2 arch types × 2 metric types)
   - In each subplot, plot results for 4 depths (1, 2, 4, 8)
   - Include dim and eval_method in the filename
   - Label y-axis as MMER and x-axis as Time Steps
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
from tqdm import tqdm
import wandb


def fetch_wandb_data(
    project_name: str,
    entity: str,
    sweep_id: str,
    cache_dir: str = "wandb_cache",
    metrics: List[str] = None,
    filters: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Fetch data from wandb API for a specific sweep.
    
    Args:
        project_name: Name of the wandb project
        entity: Name of the wandb entity/username
        sweep_id: ID of the sweep to fetch data for
        cache_dir: Directory to store cached data
        metrics: List of metric names to fetch
        filters: Dictionary of parameter filters to apply
        
    Returns:
        DataFrame containing the run configurations and metrics
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"wandb_data_cache_{sweep_id}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading data for sweep {sweep_id} from cache...")
        with open(cache_file, 'r') as f:
            data = json.load(f)
    else:
        print(f"Fetching data for sweep {sweep_id} from WandB...")
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project_name}", {"sweep": sweep_id})
        
        data = []
        for run in tqdm(runs, desc=f"Fetching runs for sweep {sweep_id}"):
            # Extract run configuration
            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            
            # Extract history for the specified metrics
            run_data = {**config, "run_id": run.id}
            
            # Get history for all metrics
            history = run.history()
            
            # If metrics is None, get all metrics
            if metrics is None:
                metrics = [col for col in history.columns if not col.startswith('_')]
            
            # Extract each metric
            for metric in metrics:
                try:
                    metric_values = history[metric].tolist()
                    run_data[metric] = metric_values
                except KeyError:
                    print(f"Warning: Metric '{metric}' not found in run {run.id}")
                    run_data[metric] = []
            
            data.append(run_data)
        
        # Cache the data
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    df = pd.DataFrame(data)
    
    # Apply filters if provided
    if filters:
        for key, value in filters.items():
            if key in df.columns:
                if isinstance(value, list):
                    df = df[df[key].isin(value)]
                else:
                    df = df[df[key] == value]
    
    return df


def preprocess_metrics(
    df: pd.DataFrame,
    metrics: List[str],
    envs: List[str] = None,
    group_by: List[str] = ["arch", "dim", "eval_method", "depth"],
    expected_seeds: int = 10,
) -> pd.DataFrame:
    """
    Preprocess metrics as specified in the plotting plan:
    1. Take the maximum of metrics along the time dimension
    2. Group metrics by specified parameters
    3. Check that each group has results for the expected number of seeds
    4. Calculate mean/std for each group
    
    Args:
        df: DataFrame containing the wandb data
        metrics: List of metric names to preprocess
        envs: List of environments to include (filter out others)
        group_by: List of parameters to group by
        expected_seeds: Expected number of seeds in each group
        
    Returns:
        DataFrame with preprocessed metrics
    """
    # Create a copy of the DataFrame to avoid modifying the original
    processed_df = df.copy()
    
    # Filter out unnecessary environments if envs is provided
    if envs and "env" in processed_df.columns:
        print(f"\nFiltering data to include only specified environments: {envs}")
        processed_df = processed_df[processed_df["env"].isin(envs)]
        print(f"Filtered DataFrame contains {len(processed_df)} rows")
    
    # Step 1: Take the maximum of metrics along the time dimension
    for metric in metrics:
        processed_df[f"{metric}_max"] = processed_df.apply(
            lambda row: np.maximum.accumulate(row[metric]) if len(row[metric]) > 0 else [],
            axis=1
        )


    # Step 2: Group metrics by specified parameters
    # First, ensure all group_by columns are present
    for col in group_by:
        if col not in processed_df.columns:
            raise ValueError(f"Group by column '{col}' not found in data")
    
    # Create a new column for seed if it doesn't exist
    if "seed" not in processed_df.columns:
        raise ValueError("The column: seed is not found!")

    # Group by the specified parameters
    grouped = processed_df.groupby(group_by)
    
    # Step 3: Check that each group has results for the expected number of seeds
    group_counts = grouped.size()
    print("\nGroup counts:")
    print(group_counts)
    
    # Check if any group has fewer than expected_seeds
    groups_with_fewer_seeds = group_counts[group_counts < expected_seeds]
    if len(groups_with_fewer_seeds) > 0:
        print(f"\nWarning: The following groups have fewer than {expected_seeds} seeds:")
        print(groups_with_fewer_seeds)
    
    # Step 4: Calculate mean/std for each group
    # We'll do this for each time step in the metrics
    
    # First, find the maximum length of any metric time series
    max_length = 0
    for metric in metrics:
        max_metric_length = processed_df[f"{metric}_max"].apply(len).max()
        max_length = max(max_length, max_metric_length)
    
    # Initialize a list to store the processed data
    processed_data = []
    
    # Process each group
    for group_name, group_df in grouped:
        # Convert group_name to dict if it's a tuple
        if isinstance(group_name, tuple):
            group_dict = {col: val for col, val in zip(group_by, group_name)}
        else:
            group_dict = {group_by[0]: group_name}
        
        # Process each metric
        for metric in metrics:
            # Get all time series for this metric in this group
            all_series = []
            num = -1
            for _, row in group_df.iterrows():
                series = row[f"{metric}_max"]
                #if isinstance(series, np.ndarray) and len(series) > 0:
                if not np.any(np.isnan(series)):
                    if num == -1:
                        num = len(series)
                    else:
                        assert(num == len(series))    
                    all_series.append(series)
            
            # Skip if no valid series
            if not all_series:
                print(f"Warning: No valid series for {group_dict}")
                continue
            
            #Pad series to the same length
            #padded_series = [series + [series[-1]] * (max_length - len(series)) for series in all_series]

            
            # Convert to numpy array
            array_data = np.array(all_series)
            
            # Calculate mean and std for each time step
            mean = np.mean(array_data, axis=0)
            std = np.std(array_data, axis=0)
            
            # Add to processed data
            processed_data.append({
                **group_dict,
                "metric": metric,
                "mean": mean.tolist(),
                "std": std.tolist(),
                "count": len(all_series),
                "time_steps": list(range(len(mean)))
            })
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_data)
    assert set(processed_df.env) == set(envs)
    return processed_df


def set_paper_style(font_scale: float = 2.0):
    """
    Set matplotlib style for paper-ready figures.
    
    Args:
        font_scale: Scale factor for font sizes
    """
    sns.set_theme(style="whitegrid", font_scale=font_scale)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'axes.labelsize': 14 * font_scale,
        'axes.titlesize': 14 * font_scale,
        'xtick.labelsize': 12 * font_scale,
        'ytick.labelsize': 12 * font_scale,
        'legend.fontsize': 12 * font_scale,
        'figure.figsize': (16, 10),  # Wide figure for 4 subplots
        'figure.dpi': 400,
        'savefig.dpi': 1200,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        'axes.grid': True,
        'grid.alpha': 0.5,
    })


def create_time_series_plot(
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
    Create a time series plot with 4 subplots (2 arch types × 2 metric types).
    
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
    
    # Create a 4x1 grid of subplots
    fig, axes = plt.subplots(1,4, figsize=(8*4, 6),sharex=True, sharey=False)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Use a color palette with enough colors for depths
    #colors = sns.color_palette("deep", len(depths))
    colors = sns.color_palette("husl", len(depths))

    # Create subplots
    count = 0
    for i, arch in enumerate(archs):
        for j, metric in enumerate(metrics):
            ax = axes[count]
            # Filter data for this metric and arch
            subplot_data = filtered_data[
                (filtered_data["metric"] == metric) & 
                (filtered_data["arch"] == arch)
            ]
            
            # Plot each depth
            for k, depth in enumerate(depths):
                # Filter data for this depth
                depth_data = subplot_data[subplot_data["depth"] == depth]
                
                if len(depth_data) == 0:
                    continue
                
                # Get the first row (should be only one row per depth)
                row = depth_data.iloc[0]
                
                # Plot mean line
                ax.plot(
                    row["time_steps"],
                    row["mean"],
                    label=r"$\ell=$"+f"{depth}",
                    color=colors[k],
                    linewidth=2
                )
                
                # Plot std as shaded area
                ax.fill_between(
                    row["time_steps"],
                    np.array(row["mean"]) - np.array(row["std"]),
                    np.array(row["mean"]) + np.array(row["std"]),
                    alpha=0.2,
                    color=colors[k]
                )
            
            # Set axis labels
            ax.set_xlabel("Time Steps")
            
            # Set subplot title
            label=f"{arch}:".upper()
            if metric == "metric":
                label += " Train"
            elif metric == "eval_metric":
                label += " Test"
            #ax.set_title(label, pad=10)

            # Add subplot index (a, b, c, etc.) in the upper right corner
            subplot_index = chr(97 + count)  # 97 is ASCII for 'a'
            ax.text(0.48, 0.98, f'({subplot_index})', transform=ax.transAxes,  ha='left', va='top')
            
            # Add legend only if show_legend is True
            if count==0:
                ax.set_ylabel("MMER")
                if show_legend:
                    ax.legend(loc='best')

            count+=1

    # Add overall title only for debug
    #plt.suptitle(f"Environment: {env}, dim={dim}, eval_method={eval_method}", fontsize=5)    
    plt.tight_layout()
    return fig


def create_standalone_legend_time_series(
    depths: List[int] = [1, 2, 4, 8],
    font_scale: float = 2.0,
) -> plt.Figure:
    """
    Create a standalone figure with only the legend for time series plots.
    
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
        line = plt.Line2D([0], [0], color=colors[j], linewidth=2)
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


def create_standalone_legend_pickup(
    archs: List[str] = ["gru", "s5"],
    font_scale: float = 2,
) -> plt.Figure:
    """
    Create a standalone figure with only the legend for pickup plots.
    
    Args:
        archs: List of architecture types
        font_scale: Scale factor for font sizes
        
    Returns:
        Matplotlib figure containing only the legend
    """
    set_paper_style(font_scale)
    
    # Create a figure with no axes
    fig = plt.figure(figsize=(16, 1))
    
    # Use a color palette with enough colors
    colors = sns.color_palette("husl", len(archs)*2)
    
    # Create dummy lines for the legend
    legend_handles = []
    legend_labels = []
    
    # Add entries for each architecture and method combination
    label_count = 0
    for j, arch in enumerate(archs):
        for method in ["RP", "Free RP"]:
            label = f"{method}"+ f"({arch})".upper()
            
            # Create a dummy line for the legend
            line = plt.Line2D([0], [0], color=colors[label_count], linewidth=2)
            legend_handles.append(line)
            legend_labels.append(label)
            label_count += 1
    
    # Add the legend to the figure
    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='center',
        ncol=len(archs)*2,  # Display in one row
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
    name_prefix: str = "legend_time_series",
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


def create_time_series_pickup_plot(
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
    Create a time series plot with 4 subplots (2 arch types × 2 metric types).
    
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
    
    # Create a 2x1 grid of subplots
    fig, axes = plt.subplots(1,2, figsize=(8*2, 6),sharex=True, sharey=False)
    
    # Flatten axes for easier indexing
    axes = axes.flatten()
    
    # Use a color palette with enough colors for depths
    #colors = sns.color_palette("deep", len(archs)*2)
    colors = sns.color_palette("husl", len(archs)*2)

    # Create subplots

    for i, metric in enumerate(metrics):
        ax = axes[i]    
        label_count = 0    
        for j, arch in enumerate(archs):
            # Filter data for this metric and arch
            subplot_data = filtered_data[
                (filtered_data["metric"] == metric) & 
                (filtered_data["arch"] == arch)
            ]
            
            # Plot each depth
            # Select optimal depth for depth>1.
            optimal_val = -np.inf
            optimal_depth=0
            for k, depth in enumerate(depths):
                if depth==1:
                    continue
                # Filter data for this depth
                depth_data = subplot_data[subplot_data["depth"] == depth]
                if len(depth_data) == 0:
                    continue                
                # Get the first row (should be only one row per depth)
                row = depth_data.iloc[0]
                if row["mean"][-1] > optimal_val:
                    optimal_depth = depth
                    optimal_val =  row["mean"][-1]


            depth_list = [1,optimal_depth]
            for k, depth in enumerate(depth_list):
                depth_data = subplot_data[subplot_data["depth"] == depth]
                if len(depth_data) == 0:
                    continue                
                # Get the first row (should be only one row per depth)
                row = depth_data.iloc[0]
                            
            
                label = "Baseline" if depth==1 else "FRA"
                label+=f"({arch})"
                # Plot mean line
                ax.plot(
                    row["time_steps"],
                    row["mean"],
                    label=label,
                    color=colors[label_count],
                    linewidth=2
                )
                # Plot std as shaded area
                ax.fill_between(
                    row["time_steps"],
                    np.array(row["mean"]) - np.array(row["std"]),
                    np.array(row["mean"]) + np.array(row["std"]),
                    alpha=0.2,
                    color=colors[label_count]
                )
                label_count+=1
            
            # Set axis labels
            ax.set_xlabel("Time Steps")
            
            # Set subplot title
            label=f"{env}:".capitalize()
            if metric == "metric":
                label += " Train"
            elif metric == "eval_metric":
                label += " Test"
            ax.set_title(label)

            #Add subplot index (a, b, c, etc.) in the upper right corner
            #subplot_index = chr(97 + i)  # 97 is ASCII for 'a'
            #ax.text(0.48, 0.98, f'({subplot_index})', transform=ax.transAxes,  ha='left', va='top')
            
            # Add legend only if show_legend is True
            if i==0:
                ax.set_ylabel("MMER")
                if show_legend:
                    ax.legend(loc='best')


    # Add overall title only for debug
    #plt.suptitle(f"Environment: {env}, dim={dim}, eval_method={eval_method}", fontsize=5)    
    plt.tight_layout()
    return fig

def save_plot(
    fig: plt.Figure,
    output_dir: str = "figures/wandb_plots",
    name_prefix: str = "time_series",
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
    
    print(f"Saved plot (with pdf) to {os.path.join(output_dir, filename)}.png")


def parse_filter_arg(filter_str: str) -> Dict[str, Any]:
    """
    Parse filter argument string into a dictionary.
    
    Args:
        filter_str: Filter string in the format "key1=value1,key2=value2"
        
    Returns:
        Dictionary of filter parameters
    """
    if not filter_str:
        return {}
    
    filters = {}
    for item in filter_str.split(','):
        if '=' in item:
            key, value = item.split('=', 1)
            
            # Try to convert value to numeric if possible
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                # Keep as string if not numeric
                pass
            
            filters[key.strip()] = value
    
    return filters


def save_preprocessed_data(
    data: pd.DataFrame,
    output_dir: str = "wandb_cache",
    name_prefix: str = "preprocessed_data",
    sweep_id: str = "",
) -> None:
    """
    Save preprocessed data to a JSON file.
    
    Args:
        data: DataFrame containing the preprocessed data
        output_dir: Directory to save the data
        name_prefix: Prefix for the filename
        sweep_id: ID of the sweep
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a descriptive filename
    filename = f"{name_prefix}_{sweep_id}.json"
    
    # Convert DataFrame to dict
    data_dict = data.to_dict(orient="records")
    
    # Save as JSON
    with open(os.path.join(output_dir, filename), 'w') as f:
        json.dump(data_dict, f)
    
    print(f"Saved preprocessed data to {os.path.join(output_dir, filename)}")


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Preprocess metrics and create time series plots.")
    
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity/username")
    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the sweep to analyze")
    parser.add_argument("--metrics", type=str, default="metric,eval_metric,eval_metric_few_shot", 
                        help="Comma-separated list of metrics to plot")
    parser.add_argument("--envs", type=str, required=True, help="Comma-separated list of environments to plot")
    parser.add_argument("--dims", type=str, default="64", help="Comma-separated list of dimensions to plot")
    parser.add_argument("--eval_methods", type=str, default="tiling,padding", 
                        help="Comma-separated list of evaluation methods to plot")
    parser.add_argument("--depths", type=str, default="1,2,4,8", help="Comma-separated list of depths to plot")
    parser.add_argument("--archs", type=str, default="gru,s5", help="Comma-separated list of architectures to plot")
    parser.add_argument("--filter", type=str, help="Filter runs by parameters (e.g., 'state_space_size=64,env_type=tree')")
    parser.add_argument("--output_dir", type=str, default="figures/wandb_plots", help="Directory to save plots")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache", help="Directory to store cached data")
    parser.add_argument("--expected_seeds", type=int, default=10, help="Expected number of seeds in each group")
    parser.add_argument("--show_legend", action="store_true", help="Show legend in individual plots")
    parser.add_argument("--legend_only", action="store_true", help="Generate only the standalone legend files without creating plots")
    
    args = parser.parse_args()
    
    # Parse metrics, envs, dims, eval_methods, depths, and archs
    metrics = args.metrics.split(',')
    envs = args.envs.split(',')
    dims = [int(dim) for dim in args.dims.split(',')]
    eval_methods = args.eval_methods.split(',')
    depths = [int(depth) for depth in args.depths.split(',')]
    archs = args.archs.split(',')
    
    # Parse filter argument
    filters = parse_filter_arg(args.filter)
    
    # Fetch data from wandb
    df = fetch_wandb_data(
        project_name=args.project,
        entity=args.entity,
        sweep_id=args.sweep_id,
        cache_dir=args.cache_dir,
        metrics=metrics,
        filters=filters
    )
    
    # Preprocess metrics
    processed_df = preprocess_metrics(
        df=df,
        metrics=metrics,
        envs=envs,  # Pass the environments list to filter
        group_by=["env", "arch", "dim", "eval_method", "depth"],
        expected_seeds=args.expected_seeds
    )
    
    # Save preprocessed data
    save_preprocessed_data(
        data=processed_df,
        output_dir=args.cache_dir,
        name_prefix="preprocessed_data",
        sweep_id=args.sweep_id
    )
    
    # Create time series plots for each env, dim, and eval_method (skip if legend_only is True)
    if not args.legend_only:
        for env in envs:
            for dim in dims:
                for eval_method in eval_methods:
                    # Create time series plot
                    fig = create_time_series_plot(
                        data=processed_df,
                        env=env,
                        dim=dim,
                        eval_method=eval_method,
                        depths=depths,
                        archs=archs,
                        metrics=metrics[:2],  # Use only the first two metrics (metric, eval_metric)
                        show_legend=args.show_legend
                    )
                    
                    # Save plot
                    save_plot(
                        fig=fig,
                        output_dir=args.output_dir,
                        name_prefix="time_series",
                        env=env,
                        dim=dim,
                        eval_method=eval_method
                    )
                    
                    plt.close(fig)

                    # Create time series plot
                    fig = create_time_series_pickup_plot(
                        data=processed_df,
                        env=env,
                        dim=dim,
                        eval_method=eval_method,
                        depths=depths,
                        archs=archs,
                        metrics=metrics[:2],  # Use only the first two metrics (metric, eval_metric)
                        show_legend=args.show_legend
                    )
                    
                    # Save plot
                    save_plot(
                        fig=fig,
                        output_dir=args.output_dir,
                        name_prefix="time_series_pickup",
                        env=env,
                        dim=dim,
                        eval_method=eval_method
                    )
                    
                    plt.close(fig)
    
    # Create and save standalone legends
    # Legend for time series plots
    legend_fig = create_standalone_legend_time_series(
        depths=depths
    )
    
    # Save standalone legend
    save_standalone_legend(
        fig=legend_fig,
        output_dir=args.output_dir,
        name_prefix="legend_time_series"
    )
    
    plt.close(legend_fig)
    
    # Legend for pickup plots
    legend_fig = create_standalone_legend_pickup(
        archs=archs
    )
    
    # Save standalone legend
    save_standalone_legend(
        fig=legend_fig,
        output_dir=args.output_dir,
        name_prefix="legend_time_series_pickup"
    )
    
    plt.close(legend_fig)



if __name__ == "__main__":
    main()
