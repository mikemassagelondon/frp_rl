#!/usr/bin/env python3
"""
Preprocessing Script for Dimension Plots

This script implements the preprocessing requirements for dimension-based plots:
1. Download all runs from the specified sweep
2. Preprocess eval_metric:
   - Take the maximum of metrics along the time dimension
   - Group metrics by arch, dim, eval_method, depth, and env
   - Check that each group has results for 10 seeds
   - Calculate mean/std for each group
3. Save the preprocessed data to a cache file for use by the plotting script
"""

import os
import json
import argparse
from typing import Dict, List, Optional, Union, Any, Tuple

import numpy as np
import pandas as pd
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
    group_by: List[str] = ["arch", "dim", "eval_method", "depth", "env"],
    expected_seeds: int = 10,
) -> pd.DataFrame:
    """
    Preprocess metrics for dimension-based plots:
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
            lambda row: np.maximum.accumulate(row[metric])[-1] if len(row[metric]) > 0 else np.nan,
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
            # Get all values for this metric in this group
            values = group_df[f"{metric}_max"].dropna().tolist()
            
            # Skip if no valid values
            if not values:
                print(f"Warning: No valid values for {group_dict}, metric {metric}")
                continue
            
            # Calculate mean and std
            mean = np.mean(values)
            std = np.std(values)
            
            # Add to processed data
            processed_data.append({
                **group_dict,
                "metric": metric,
                "mean": mean,
                "std": std,
                "count": len(values)
            })
    
    # Convert to DataFrame
    processed_df = pd.DataFrame(processed_data)
    
    return processed_df


def save_preprocessed_data(
    data: pd.DataFrame,
    output_dir: str = "wandb_cache",
    name_prefix: str = "preprocessed_dim_data",
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


def main():
    """Main function to run the script."""
    parser = argparse.ArgumentParser(description="Preprocess metrics for dimension-based plots.")
    
    parser.add_argument("--project", type=str, required=True, help="WandB project name")
    parser.add_argument("--entity", type=str, required=True, help="WandB entity/username")
    parser.add_argument("--sweep_id", type=str, required=True, help="ID of the sweep to analyze")
    parser.add_argument("--metrics", type=str, default="metric,eval_metric", 
                        help="Comma-separated list of metrics to preprocess")
    parser.add_argument("--envs", type=str, required=True, help="Comma-separated list of environments to include")
    parser.add_argument("--dims", type=str, default="64,128", help="Comma-separated list of dimensions to include")
    parser.add_argument("--eval_methods", type=str, default="tiling,padding", 
                        help="Comma-separated list of evaluation methods to include")
    parser.add_argument("--depths", type=str, default="1,2,4,8", help="Comma-separated list of depths to include")
    parser.add_argument("--archs", type=str, default="gru,s5", help="Comma-separated list of architectures to include")
    parser.add_argument("--filter", type=str, help="Filter runs by parameters (e.g., 'state_space_size=64,env_type=tree')")
    parser.add_argument("--cache_dir", type=str, default="wandb_cache", help="Directory to store cached data")
    parser.add_argument("--expected_seeds", type=int, default=10, help="Expected number of seeds in each group")
    
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
    
    # Add filters for dims, eval_methods, depths, and archs
    if dims:
        filters["dim"] = dims
    if eval_methods:
        filters["eval_method"] = eval_methods
    if depths:
        filters["depth"] = depths
    if archs:
        filters["arch"] = archs
    
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
        envs=envs,
        group_by=["arch", "dim", "eval_method", "depth", "env"],
        expected_seeds=args.expected_seeds
    )
    
    # Save preprocessed data
    save_preprocessed_data(
        data=processed_df,
        output_dir=args.cache_dir,
        name_prefix="preprocessed_dim_data",
        sweep_id=args.sweep_id
    )
    
    print("Preprocessing complete!")


if __name__ == "__main__":
    main()
