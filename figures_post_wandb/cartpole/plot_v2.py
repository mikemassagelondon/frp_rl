import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import json
import seaborn as sns
from tqdm import tqdm
import argparse
import itertools

def fetch_wandb_data(project_name: str, entity: str, cache_dir: str, sweep_ids: List[str]) -> Dict[str, pd.DataFrame]:
    os.makedirs(cache_dir, exist_ok=True)
    all_data = {}

    for sweep_id in tqdm(sweep_ids, desc="Processing sweeps"):
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
            for run in tqdm(runs, desc=f"Fetching runs for sweep {sweep_id}", leave=False):
                run_data = {
                    'dim': run.config.get('dim'),
                    'depth': run.config.get('depth'), 
                    'max_depth': run.config.get('max_depth'),
                    'seed': run.config.get('seed'),
                    'with_adjoint': run.config.get('with_adjoint', 0),
                    'reset_patterns': run.config.get('reset_patterns', False),
                    'eval_method': run.config.get('eval_method', 'const'),
                    'metric': run.history()['metric'].tolist(),
                    'eval_metric': run.history()['eval_metric'].tolist()
                }
                data.append(run_data)
            
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        
        df = pd.DataFrame(data)
        all_data[sweep_id] = df

    return all_data

def process_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df['dim'] = df['dim'].astype(int)
    df['max_depth'] = df['max_depth'].astype(int) 
    df['with_adjoint'] = df['with_adjoint'].astype(int)
    grouped = df.groupby(['dim', 'depth', 'max_depth', 'with_adjoint', 'reset_patterns', 'eval_method'])
    processed_data = {
        'all_data': df,
        'max_eval_metric': grouped.apply(lambda x: pd.Series({
            'mean': np.max(x['eval_metric'].apply(lambda y: np.max(y) if isinstance(y, list) else y)),
            'std': np.std(x['eval_metric'].apply(lambda y: np.max(y) if isinstance(y, list) else y))
        })).reset_index()
    }
    
    return processed_data

def generate_filename_prefix(params: Dict) -> str:
    """Generate shortened but descriptive filename prefix"""
    # Convert params to short form
    components = []
    if 'dim' in params:
        components.append(f"d{params['dim']}")
    if 'depth' in params:
        components.append(f"l{params['depth']}")  # l for length/depth
    if 'max_depth' in params:
        components.append(f"m{params['max_depth']}")
    if 'with_adjoint' in params and params['with_adjoint']:
        components.append("adj")
    if 'reset_patterns' in params and params['reset_patterns']:
        components.append("rp")
    if 'eval_method' in params:
        components.append(f"e{params['eval_method'][:2]}")  # First 2 chars of eval method
    
    return '_'.join(components)

def save_figure(fig, base_name: str, params: Dict, dpi: int = 600):
    """Save figure with standardized naming convention"""
    prefix = generate_filename_prefix(params)
    os.makedirs(base_name, exist_ok=True)
    fig.savefig(f'{base_name}/{prefix}_{base_name}.png', dpi=dpi, bbox_inches='tight')
    fig.savefig(f'{base_name}/{prefix}_{base_name}.pdf', dpi=dpi, bbox_inches='tight')
    plt.close(fig)

def set_paper_style():
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.size': 20,
        'axes.labelsize': 24,
        'axes.titlesize': 28,
        'xtick.labelsize': 20,
        'ytick.labelsize': 20,
        'legend.fontsize': 20,
        'figure.figsize': (16, 16)
    })

def get_color_palette(n_colors):
    return sns.color_palette("husl", n_colors)


def plot_figure1(data: pd.DataFrame, dim: int, max_depth: int = None, with_adjoint: int = 0):
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    dim_data = data[(data['dim'] == dim) & (data['with_adjoint'] == with_adjoint)]
    if max_depth is not None:
        dim_data = dim_data[dim_data['max_depth'] == max_depth]
    
    depths = sorted(dim_data['depth'].unique())
    colors = get_color_palette(len(depths))
    
    max_std = 0
    for metric, ax in zip(['metric', 'eval_metric'], [ax1, ax2]):
        for depth, color in zip(depths, colors):
            depth_data = dim_data[dim_data['depth'] == depth]
            
            all_series = []
            for _, row in depth_data.iterrows():
                series = row[metric]
                if isinstance(series, list):
                    all_series.append(series)
                else:
                    all_series.append([series])
            
            max_length = max(len(series) for series in all_series)
            padded_series = [series + [np.nan] * (max_length - len(series)) for series in all_series]
            
            array_data = np.array(padded_series)
            
            mean = np.nanmean(array_data, axis=0)
            std = np.nanstd(array_data, axis=0)
            max_std = max(max_std, np.max(std))
            time_steps = range(len(mean))
            
            ax.plot(time_steps, mean, label=f'{depth}', color=color)

        if "eval" in metric:
            label="MER (Eval)"
        else:
            label="MER"

        ax.set_ylabel(label)
    
    ax1.legend(title='Depth', loc='lower right')
    ax2.set_xlabel('Time Steps')
    print("max_std:", max_std)    
    plt.tight_layout()
    return fig

def plot_figure2(data: pd.DataFrame, max_depth: int = None, with_adjoint: int = 0):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    data = data[data['with_adjoint'] == with_adjoint]
    if max_depth is not None:
        data = data[data['max_depth'] == max_depth]
    
    dims = sorted(data['dim'].unique())
    colors = get_color_palette(len(dims))
    
    for dim, color in zip(dims, colors):
        dim_data = data[data['dim'] == dim]
        ax.errorbar(dim_data['depth'], dim_data['mean'], yerr=dim_data['std'], 
                    label=f'{dim}', capsize=5, fmt='o-', color=color)
    
    ax.set_xlabel('Depth')
    ax.set_ylabel('MMER')
    ax.legend(title='Dimension', loc='best')
    
    plt.tight_layout()
    return fig

def plot_figure3(data: pd.DataFrame, max_depth: int = None, with_adjoint: int = 0):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    data = data[data['with_adjoint'] == with_adjoint]
    if max_depth is not None:
        data = data[data['max_depth'] == max_depth]
    
    depths = sorted(data['depth'].unique())
    colors = get_color_palette(len(depths))
    
    for depth, color in zip(depths, colors):
        depth_data = data[data['depth'] == depth]
        ax.errorbar(depth_data['dim'], depth_data['mean'], yerr=depth_data['std'], 
                    label=f'{depth}', capsize=5, fmt='o-', color=color)
    
    ax.set_xscale('log', base=2)
    ax.set_xlabel(r'Dimension ($\log_2$ scale)')
    ax.set_ylabel('MMER')
    
    dims = sorted(data['dim'].unique())
    ax.set_xticks(dims)
    ax.set_xticklabels(dims)
    
    ax.legend(title='Depth', loc='best')
    plt.grid(True, which="both", ls="-", alpha=0.2)
    
    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser(description="Process WandB data and create plots.")
    parser.add_argument("--project_name", type=str, required=True, help="WandB project name")
    parser.add_argument("--sweep_ids", nargs="+", required=True, help="List of sweep IDs")
    args = parser.parse_args()

    DPI = 600
    entity = "cmvl_nelf"
    cache_dir = "wandb_cache"
    
    all_sweep_data = fetch_wandb_data(args.project_name, entity, cache_dir, args.sweep_ids)
    
    for sweep_id, df in all_sweep_data.items():
        print(f"Processing data for sweep {sweep_id}...")
        processed_data = process_data(df)
        
        # Get all unique combinations
        param_combinations = []
        for params in [
            'max_depth',
            'with_adjoint',
            'reset_patterns',
            'eval_method'
        ]:
            if params in processed_data['all_data'].columns:
                param_combinations.append(sorted(processed_data['all_data'][params].unique()))
        
        # Iterate through all combinations
        for params in tqdm(list(itertools.product(*param_combinations)), desc=f"Processing parameter combinations"):
            param_dict = {
                'max_depth': params[0],
                'with_adjoint': params[1],
                'reset_patterns': params[2],
                'eval_method': params[3]
            }
            
            # Filter data for current parameter combination
            filtered_data = processed_data['all_data']
            for key, value in param_dict.items():
                filtered_data = filtered_data[filtered_data[key] == value]
            
            # Create and save plots
            all_dims = sorted(filtered_data['dim'].unique())
            for dim in tqdm(all_dims, desc="Processing dimensions", leave=False):
                param_dict['dim'] = dim
                fig1 = plot_figure1(filtered_data, dim=dim)
                save_figure(fig1, f'fig1_sweep{sweep_id}', param_dict, DPI)
            
            fig2 = plot_figure2(processed_data['max_eval_metric'].loc[
                (processed_data['max_eval_metric']['max_depth'] == param_dict['max_depth']) &
                (processed_data['max_eval_metric']['with_adjoint'] == param_dict['with_adjoint']) &
                (processed_data['max_eval_metric']['reset_patterns'] == param_dict['reset_patterns']) &
                (processed_data['max_eval_metric']['eval_method'] == param_dict['eval_method'])
            ])
            save_figure(fig2, f'fig2_sweep{sweep_id}', param_dict, DPI)
            
            fig3 = plot_figure3(processed_data['max_eval_metric'].loc[
                (processed_data['max_eval_metric']['max_depth'] == param_dict['max_depth']) &
                (processed_data['max_eval_metric']['with_adjoint'] == param_dict['with_adjoint']) &
                (processed_data['max_eval_metric']['reset_patterns'] == param_dict['reset_patterns']) &
                (processed_data['max_eval_metric']['eval_method'] == param_dict['eval_method'])
            ])
            save_figure(fig3, f'fig3_sweep{sweep_id}', param_dict, DPI)

if __name__ == "__main__":
    main()