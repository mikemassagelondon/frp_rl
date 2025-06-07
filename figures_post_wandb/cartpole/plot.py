import wandb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict
import os
import json
import seaborn as sns



def fetch_wandb_data(project_name: str, entity: str, cache_file: str) -> pd.DataFrame:
    if os.path.exists(cache_file):
        print("Loading data from cache...")
        with open(cache_file, 'r') as f:
            data = json.load(f)
    else:
        print("Fetching data from WandB...")
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project_name}")
        
        data = []
        for run in runs:
            run_data = {
                'dim': run.config.get('dim'),
                'depth': run.config.get('depth'),
                'seed': run.config.get('seed'),
                'metric': run.history()['metric'].tolist(),
                'eval_metric': run.history()['eval_metric'].tolist()
            }
            data.append(run_data)
        
        with open(cache_file, 'w') as f:
            json.dump(data, f)
    
    df = pd.DataFrame(data)
    # dim=4のケースを処理
    df.loc[df['dim'].isnull(), 'dim'] = 4
    return df

def process_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    df['dim'] = df['dim'].astype(int)
    grouped = df.groupby(['dim', 'depth'])
    processed_data = {
        'all_data': df,
        'max_eval_metric': grouped.apply(lambda x: pd.Series({
            'mean': np.max(x['eval_metric'].apply(lambda y: np.max(y) if isinstance(y, list) else y)),
            'std': np.std(x['eval_metric'].apply(lambda y: np.max(y) if isinstance(y, list) else y))
        })).reset_index()
    }
    
    return processed_data

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

def plot_figure1(data: pd.DataFrame, dim: int):
    set_paper_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    
    dim_data = data[data['dim'] == dim]
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
            #ax.fill_between(time_steps, mean - std, mean + std, alpha=0.2, color=color)


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

def plot_figure2(data: pd.DataFrame):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
    dims = sorted(data['dim'].unique())
    colors = get_color_palette(len(dims))
    
    for dim, color in zip(dims, colors):
        dim_data = data[data['dim'] == dim]
        ax.errorbar(dim_data['depth'], dim_data['mean'], yerr=dim_data['std'], 
                    label=f'{dim}', capsize=5, fmt='o-', color=color)
    
    ax.set_xlabel('Depth')
    ax.set_ylabel('Max Eval Metric')
    ax.legend(title='Dimension', loc='best')
    
    plt.tight_layout()
    return fig

def plot_figure3(data: pd.DataFrame):
    set_paper_style()
    fig, ax = plt.subplots(figsize=(8, 8))
    
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

    DPI = 600

    # WandBのプロジェクト名とエンティティを設定
    project_name = "popgym_eval_eye"
    entity = "cmvl_nelf"
    cache_file = "wandb_data_cache.json"
    
    # データの取得と処理
    df = fetch_wandb_data(project_name, entity, cache_file)
    processed_data = process_data(df)
    
    # グラフの作成
    # すべてのdimに対してfigure1を生成
    all_dims = sorted(processed_data['all_data']['dim'].unique())
    for dim in all_dims:
        fig1 = plot_figure1(processed_data['all_data'], dim=dim)
        fig1.savefig(f'figure1_dim{dim}.png', dpi=DPI, bbox_inches='tight')
        fig1.savefig(f'figure1_dim{dim}.pdf', dpi=DPI, bbox_inches='tight')
        plt.close(fig1)  # メモリ解放のため
    fig2 = plot_figure2(processed_data['max_eval_metric'])
    fig3 = plot_figure3(processed_data['max_eval_metric'])
    
    # グラフの保存
    #fig1.savefig('figure1.png', dpi=DPI, bbox_inches='tight')
    fig2.savefig('figure2.png', dpi=DPI, bbox_inches='tight')
    fig2.savefig('figure2.pdf', dpi=DPI, bbox_inches='tight')

    fig3.savefig('figure3.png', dpi=DPI, bbox_inches='tight')
    fig3.savefig('figure3.pdf', dpi=DPI, bbox_inches='tight')

    plt.show()

if __name__ == "__main__":
    main()