import wandb
import pandas as pd
from collections import defaultdict
import numpy as np
from typing import Dict, List, Tuple

def fetch_run_results(project_name: str, sweep_id: str = None) -> List[Dict]:
    """
    wandb APIから実験結果を取得
    
    Args:
        project_name: プロジェクト名
        sweep_id: 特定のsweepに限定する場合はそのID（オプション）
    
    Returns:
        実験結果のリスト
    """
    api = wandb.Api()
    
    # sweepが指定されている場合はそのrunのみを取得
    if sweep_id:
        sweep = api.sweep(f"{project_name}/{sweep_id}")
        runs = sweep.runs
    else:
        # すべてのrunを取得
        runs = api.runs(project_name)
    
    results = []
    for run in runs:
        if run.state != "finished":
            continue
            
        # 必要なデータを抽出
        try:
            result = {
                'perm_idx': run.config.get('perm_idx'),
                'permutation': run.summary.get('permutation'),
                'mean_diff': run.summary.get('mean_diff'),
                'std_diff': run.summary.get('std_diff'),
                'total_diff': run.summary.get('total_diff')
            }
            
            # 必要なデータが揃っているrunのみを保存
            if all(v is not None for v in result.values()):
                results.append(result)
                
        except Exception as e:
            print(f"Error processing run {run.id}: {e}")
            continue
    
    return results

def aggregate_results(results: List[Dict]) -> pd.DataFrame:
    """
    perm_idxごとに結果を集計
    
    Args:
        results: fetch_run_resultsで取得した結果のリスト
    
    Returns:
        集計結果のDataFrame
    """
    # perm_idxごとの集計用dictionary
    aggregated = defaultdict(lambda: {
        'permutation': [],
        'mean_diff_values': [],
        'std_diff_values': [],
        'total_diff_values': []
    })
    
    # 結果を集計
    for result in results:
        perm_idx = result['perm_idx']
        aggregated[perm_idx]['permutation'].append(result['permutation'])
        aggregated[perm_idx]['mean_diff_values'].append(result['mean_diff'])
        aggregated[perm_idx]['std_diff_values'].append(result['std_diff'])
        aggregated[perm_idx]['total_diff_values'].append(result['total_diff'])
    
    # DataFrameに変換しやすい形式に整理
    processed_results = []
    for perm_idx, data in aggregated.items():
        processed_results.append({
            'perm_idx': perm_idx,
            'permutation': data['permutation'][0],  # 同じperm_idxなら同じはず
            'mean_diff_avg': np.mean(data['mean_diff_values']),
            'mean_diff_max': np.max(data['mean_diff_values']),
            'std_diff_avg': np.mean(data['std_diff_values']),
            'std_diff_max': np.max(data['std_diff_values']),
            'total_diff_avg': np.mean(data['total_diff_values']),
            'total_diff_max': np.max(data['total_diff_values']),
            'num_runs': len(data['mean_diff_values'])
        })
    
    return pd.DataFrame(processed_results)

def sort_and_save_results(df: pd.DataFrame, output_path: str = 'results_analysis'):
    """
    結果をソートして保存
    
    Args:
        df: 集計結果のDataFrame
        output_path: 保存するベースファイル名
    """
    # mean_diffでソート
    mean_diff_sorted = df.sort_values('mean_diff_avg', ascending=False).copy()
    mean_diff_sorted.to_csv(f'{output_path}_by_mean_diff.csv', index=False)
    
    # total_diffでソート
    total_diff_sorted = df.sort_values('total_diff_avg', ascending=False).copy()
    total_diff_sorted.to_csv(f'{output_path}_by_total_diff.csv', index=False)
    
    # 上位の結果を表示
    topk=30
    
    print(f"\nTop {topk} results by mean_diff:")
    print(mean_diff_sorted[['perm_idx', 'permutation', 'mean_diff_avg', 'total_diff_avg']].head(topk))
    
    print(f"\nTop {topk} results by total_diff:")
    print(total_diff_sorted[['perm_idx', 'permutation', 'mean_diff_avg', 'total_diff_avg']].head(topk))
    
    return mean_diff_sorted, total_diff_sorted

def main():
    # wandbにログイン
    wandb.login()
    
    # 設定
    PROJECT_NAME = "nelf_eigs_linearization"  # あなたのプロジェクト名
    SWEEP_ID = None  # 特定のsweepに限定する場合は指定
    
    # 結果を取得
    print("Fetching results from wandb...")
    results = fetch_run_results(PROJECT_NAME, SWEEP_ID)
    
    # 結果がない場合は終了
    if not results:
        print("No results found!")
        return
    
    print(f"Found {len(results)} completed runs")
    
    # 結果を集計
    print("Aggregating results...")
    df = aggregate_results(results)
    
    # ソートして保存
    print("Sorting and saving results...")
    mean_sorted, total_sorted = sort_and_save_results(df)
    
    # 基本的な統計情報を表示
    print("\nSummary statistics:")
    print(f"Total unique perm_idx: {len(df)}")
    print("\nMean diff statistics:")
    print(df[['mean_diff_avg', 'mean_diff_max']].describe())
    print("\nTotal diff statistics:")
    print(df[['total_diff_avg', 'total_diff_max']].describe())

if __name__ == "__main__":
    main()