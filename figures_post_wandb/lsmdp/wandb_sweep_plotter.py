#!/usr/bin/env python3
"""
Wandb Sweep Data Plotter

This script loads and plots data from Weights & Biases (wandb) sweeps.
It creates paper-ready plots with error bars and saves them as PNG and PDF.

Features:
- Loads data from wandb API with caching
- Groups data by configurable parameters
- Creates plots with error bars (standard deviation)
- Saves plots as both PNG and PDF
- Configurable styling for paper-ready figures
- Supports filtering and comparing different parameter values
- Supports plotting multiple metrics in subplots
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Union, Any

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
    metric_names: List[str] = ["z_space_KL_pi"],
    filters: Dict[str, Any] = None,
) -> pd.DataFrame:
    """
    Fetch data from wandb API for a specific sweep.
    
    Args:
        project_name: Name of the wandb project
        entity: Name of the wandb entity/username
        sweep_id: ID of the sweep to fetch data for
        cache_dir: Directory to store cached data
        metric_names: List of metrics to fetch
        filters: Dictionary of parameter filters to apply (e.g., {"state_space_size": 64})
        
    Returns:
        DataFrame containing the run configurations and metrics
    """
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create a hash of the metric names to include in the cache file name
    metrics_hash = "_".join(sorted([m.split("_")[-1] for m in metric_names]))
    cache_file = os.path.join(cache_dir, f"wandb_data_cache_{sweep_id}_{metrics_hash}.json")
    
    if os.path.exists(cache_file):
        print(f"Loading data for sweep {sweep_id} with metrics {metrics_hash} from cache...")
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
            
            # Initialize run data with config
            run_data = {**config, "run_id": run.id}
            
            # Extract history for each specified metric
            history = run.history()
            metrics_found = False
            
            for metric_name in metric_names:
                try:
                    run_data[metric_name] = history[metric_name].tolist()
                    metrics_found = True
                except KeyError:
                    print(f"Warning: Metric '{metric_name}' not found in run {run.id}")
                    run_data[metric_name] = []
            
            # Only add the run if at least one metric was found
            if metrics_found:
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


def process_data(
    df: pd.DataFrame,
    metric_names: List[str] = ["z_space_KL_pi"],
    group_by: str = "state_space_size",
    compare_by: str = None,
    agg_method: str = "last",
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """
    Process the wandb data for plotting.
    
    Args:
        df: DataFrame containing the wandb data
        metric_names: List of metrics to process
        group_by: Parameter to group by
        compare_by: Parameter to compare different groups (e.g., "env_type")
        agg_method: Method to aggregate metric values ('last', 'min', 'max', 'mean')
        
    Returns:
        Dictionary containing processed DataFrames for each metric
    """
    # Ensure group_by column is present
    if group_by not in df.columns:
        raise ValueError(f"Group by column '{group_by}' not found in data")
    
    # Ensure compare_by column is present if specified
    if compare_by and compare_by not in df.columns:
        raise ValueError(f"Compare by column '{compare_by}' not found in data")
    
    results = {}
    
    for metric_name in metric_names:
        # Extract the metric values based on aggregation method
        def extract_metric(row):
            values = row[metric_name]
            if not isinstance(values, list):
                return values
            
            if len(values) == 0:
                return np.nan
                
            if agg_method == "last":
                return values[-1]
            elif agg_method == "min":
                return min(values)
            elif agg_method == "max":
                return max(values)
            elif agg_method == "mean":
                return np.mean(values)
            else:
                raise ValueError(f"Unknown aggregation method: {agg_method}")
        
        # Create a new column with the aggregated metric value
        df[f"{metric_name}_{agg_method}"] = df.apply(extract_metric, axis=1)
        
        # Group by the specified parameters
        if compare_by:
            grouped = df.groupby([group_by, compare_by])
        else:
            grouped = df.groupby(group_by)
        
        # Calculate statistics for each group
        stats_df = grouped.apply(lambda x: pd.Series({
            'mean': np.mean(x[f"{metric_name}_{agg_method}"]),
            'std': np.std(x[f"{metric_name}_{agg_method}"]),
            'min': np.min(x[f"{metric_name}_{agg_method}"]),
            'max': np.max(x[f"{metric_name}_{agg_method}"]),
            'count': len(x)
        })).reset_index()
        
        results[metric_name] = {
            'all_data': df,
            'stats': stats_df
        }
    
    return results


def set_paper_style(font_scale: float = 2):
    """
    Set matplotlib style for paper-ready figures.
    
    Args:
        font_scale: Scale factor for font sizes
    """
    sns.set_theme(style="whitegrid", font_scale=font_scale)
    plt.rcParams.update({
        'font.family': 'sans-serif',
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        'axes.labelsize': 14 * font_scale,
        'axes.titlesize': 16 * font_scale,
        'xtick.labelsize': 12 * font_scale,
        'ytick.labelsize': 12 * font_scale,
        'legend.fontsize': 12 * font_scale,
        'figure.figsize': (8, 6),
        'figure.dpi': 500,
        'savefig.dpi': 1000,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.01,
        'axes.grid': True,
        'grid.alpha': 0.5,
    })


def save_plot(
    fig: plt.Figure,
    output_dir: str = "figures",
    name_prefix: str = "wandb_plot",
    metric_names: List[str] = ["z_space_KL_pi"],
    x_param: str = "state_space_size",
    compare_param: str = None,
    plot_type: str = "line",
) -> None:
    """
    Save plot as PNG and PDF.
    
    Args:
        fig: Matplotlib figure to save
        output_dir: Directory to save the plot
        name_prefix: Prefix for the filename
        metric_names: List of metrics plotted
        x_param: Parameter used for x-axis
        compare_param: Parameter used for comparison
        plot_type: Type of plot ('line', 'bar', 'time_series', 'multi_metric')
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a descriptive filename
    metrics_str = "_".join([m.split("_")[-1] for m in metric_names])
    
    if compare_param:
        filename = f"{name_prefix}_{metrics_str}_{x_param}_by_{compare_param}_{plot_type}"
    else:
        filename = f"{name_prefix}_{metrics_str}_{x_param}_{plot_type}"
    
    # Save as PNG
    fig.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300, bbox_inches='tight')
    
    # Save as PDF
    fig.savefig(os.path.join(output_dir, f"{filename}.pdf"), bbox_inches='tight')
    
    print(f"Saved plot to {os.path.join(output_dir, filename)}.png")


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


def create_time_series_plot(
    data: pd.DataFrame,
    metric_name: str = "z_space_KL_pi",
    group_by: str = None,
    compare_by: str = None,
    ax: plt.Axes = None,
    y_label: str = None,
    show_legend: bool = False,
) -> None:
    """
    Create a time series plot with the metric values over time.
    
    Args:
        data: DataFrame containing the raw data
        metric_name: Name of the metric to plot
        group_by: Parameter to group runs by (e.g., "state_space_size")
        compare_by: Parameter to compare different groups (e.g., "env_type")
        ax: Matplotlib axes to plot on
        y_label: Custom y-axis label
        show_legend: Show legend 
        
    Returns:
        None (plots directly on the provided axes)
    """
    if ax is None:
        _, ax = plt.subplots()
    
    # Group runs by the specified parameters
    # Group by both parameters
    groups = data.groupby([group_by, compare_by])
    
    # Get unique values for the parameters
    group_values = sorted(data[group_by].unique())
    compare_values = sorted(data[compare_by].unique())
    
    # Use a color palette with enough colors
    colors = sns.color_palette("deep", len(group_values) * len(compare_values))
    
    # Plot each group with a different color
    color_idx = 0
    for (group_val, compare_val), group_data in groups:
        # Calculate mean and std for each time step
        #time_steps = range(len(group_data[metric_name].iloc[0]))
        time_steps = [1,2,4,8] # Hard Coding
        # Extract all time series for this group
        all_series = []
        for _, row in group_data.iterrows():
            series = row[metric_name]
            if isinstance(series, list):
                all_series.append(series)
        
        # Calculate mean and std for each time step
        if all_series:
            # Pad series to the same length
            max_length = max(len(series) for series in all_series)
            padded_series = [series + [np.nan] * (max_length - len(series)) for series in all_series]
            
            # Convert to numpy array
            array_data = np.array(padded_series)
            
            # Calculate mean and std
            mean = np.nanmean(array_data, axis=0)
            std = np.nanstd(array_data, axis=0)
            
            #label=f"{group_by}={group_val}, {compare_by}={compare_val}"
            label=None
            if  compare_val=="tree":
                label="Tree"
            elif compare_val=="lattice":
                label="Lattice"
            else:
                raise ValueError()
            
            # Plot mean line
            ax.plot(
                time_steps[:len(mean)],
                mean,
                label=label,
                color=colors[color_idx],
                linewidth=2
            )
            
            # Plot std as shaded area
            ax.fill_between(
                time_steps[:len(mean)],
                mean - std,
                mean + std,
                alpha=0.2,
                color=colors[color_idx]
            )
            
            color_idx += 1
    
    
    # Set axis labels
    ax.set_xlabel("Length")
    ax.set_xscale("log", base=2)

    # Set y-axis label
    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(metric_name.replace('_', ' ').title())

    # Set aspect ratio to 'equal' to make the plot box square
    #ax.set_aspect('equal')

    if show_legend:    
        # Add legend if there are multiple groups
        if (group_by and len(data[group_by].unique()) > 1) or (compare_by and len(data[compare_by].unique()) > 1):
            ax.legend(loc='lower left')


def create_multi_metric_plot(
    data: pd.DataFrame,
    metric_names: List[str],
    group_by: str = None,
    compare_by: str = None,
    y_labels: List[str] = None,
) -> plt.Figure:
    """
    Create a figure with multiple subplots, one for each metric.
    
    Args:
        data: DataFrame containing the raw data
        metric_names: List of metrics to plot
        group_by: Parameter to group runs by (e.g., "state_space_size")
        compare_by: Parameter to compare different groups (e.g., "env_type")
        y_labels: List of custom y-axis labels for each metric
        
    Returns:
        Matplotlib figure
    """
    set_paper_style(font_scale=2)
    
    # Create a figure with subplots
    fig, axes = plt.subplots(1, len(metric_names), figsize=(8 * len(metric_names), 6), sharey=False)
    
    # If only one metric, axes will not be an array
    if len(metric_names) == 1:
        axes = [axes]
    
    # Create custom y-labels if not provided
    if y_labels is None:
        y_labels = []
        for metric in metric_names:
            if metric == "z_space_KL_pi":
                y_labels.append(r"$D_\mathrm{KL}$")
            elif metric == "z_space_L1_pi":
                y_labels.append(r"$\ell_1$")
            elif metric == "z_space_L1_z":
                y_labels.append(r"$\ell_1$")                
            elif metric == "z_space_L2_z":
                y_labels.append(r"$\ell_2$")                
            else:
                y_labels.append(metric.replace('_', ' ').title())
    
    # Plot each metric in its own subplot
    for i, (metric, ax, y_label) in enumerate(zip(metric_names, axes, y_labels)):
        show_legend=False
        if i==0:
            show_legend=True
        create_time_series_plot(
            data=data,
            metric_name=metric,
            group_by=group_by,
            compare_by=compare_by,
            ax=ax,
            y_label=y_label,
            show_legend=show_legend
        )
        
        # Add subplot index (a, b, c, etc.) in the upper right corner
        subplot_index = chr(97 + i)  # 97 is ASCII for 'a'
        ax.text(0.95, 0.95, f'({subplot_index})', transform=ax.transAxes,  ha='right', va='top')
        
    # Use subplots_adjust instead of tight_layout to have more control over spacing
    fig.subplots_adjust(wspace=0.28)  # Reduce width spacing between subplots
    return fig


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Plot wandb sweep data with error bars.")
    
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity/username")
    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the sweep to analyze")
    parser.add_argument("--metrics", type=str, default="z_space_KL_pi,z_space_L1_pi", 
                        help="Comma-separated list of metrics to plot")
    parser.add_argument("--group_by", type=str, default="state_space_size", help="Parameter to group by")
    parser.add_argument("--compare_by", type=str, help="Parameter to compare different groups (e.g., 'env_type')")
    parser.add_argument("--filter", type=str, help="Filter runs by parameters (e.g., 'state_space_size=64,env_type=tree')")
    parser.add_argument("--agg_method", type=str, default="mean", choices=["last", "min", "max", "mean"], 
                        help="Method to aggregate metric values")
    parser.add_argument("--output_dir", type=str, default="figures/wandb_plots", help="Directory to save plots")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache", help="Directory to store cached data")
    parser.add_argument("--plot_type", type=str, default="multi_metric", 
                        choices=["line", "bar", "time_series", "multi_metric"], 
                        help="Type of plot to create")
    parser.add_argument("--log_scale", action="store_true", help="Use log scale for x-axis")
    
    args = parser.parse_args()
    
    # Parse metrics
    metric_names = [m.strip() for m in args.metrics.split(',')]
    
    # Parse filter argument
    filters = parse_filter_arg(args.filter)
    
    # Fetch data from wandb
    df = fetch_wandb_data(
        project_name=args.project,
        entity=args.entity,
        sweep_id=args.sweep_id,
        cache_dir=args.cache_dir,
        metric_names=metric_names,
        filters=filters
    )
    
    # Process data
    processed_data = process_data(
        df=df,
        metric_names=metric_names,
        group_by=args.group_by,
        compare_by=args.compare_by,
        agg_method=args.agg_method
    )
    
    # Create and save plots
    if args.plot_type == "multi_metric":
        # Create multi-metric plot
        multi_metric_fig = create_multi_metric_plot(
            data=df,
            metric_names=metric_names,
            group_by=args.group_by,
            compare_by=args.compare_by
        )
        save_plot(
            fig=multi_metric_fig,
            output_dir=args.output_dir,
            name_prefix=f"sweep_{args.sweep_id}",
            metric_names=metric_names,
            x_param="steps",
            compare_param=args.compare_by,
            plot_type="multi_metric"
        )
        plt.close(multi_metric_fig)
    elif args.plot_type == "time_series":
        # For backward compatibility, create individual time series plots
        for metric_name in metric_names:
            fig, ax = plt.subplots()
            create_time_series_plot(
                data=df,
                metric_name=metric_name,
                group_by=args.group_by,
                compare_by=args.compare_by,
                ax=ax
            )
            save_plot(
                fig=fig,
                output_dir=args.output_dir,
                name_prefix=f"sweep_{args.sweep_id}",
                metric_names=[metric_name],
                x_param="steps",
                compare_param=args.compare_by,
                plot_type="time_series"
            )
            plt.close(fig)
    else:
        raise ValueError(f"Unsupported plot type: {args.plot_type}")


if __name__ == "__main__":
    main()
