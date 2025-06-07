# Dimension-based Plotting Scripts

This directory contains scripts for creating plots with dimension on the x-axis and eval_metric on the y-axis, with different lines for each depth (1, 2, 4, 8).

## Scripts

1. `preprocess_for_dim_plots.py`: Preprocesses wandb sweep data for dimension-based plots
2. `plot_dim_results.py`: Creates plots with dimension on the x-axis
3. `plot_dim_results.sh`: Shell script to run both scripts in sequence

## Usage

### Running the Complete Pipeline

To run the complete pipeline (preprocessing and plotting), use the shell script:

```bash
cd figures/popjaxrl
./plot_dim_results.sh [PROJECT] [ENTITY] [SWEEP_ID] [ENVS] [DIMS] [EVAL_METHODS] [DEPTHS] [ARCHS] [METRICS] [OUTPUT_DIR] [CACHE_DIR]
```

All arguments are optional and have default values:

- `PROJECT`: WandB project name (default: "popgym_sweep_envs")
- `ENTITY`: WandB entity/username (default: "cmvl_nelf")
- `SWEEP_ID`: ID of the sweep to analyze (default: "8z7oojm2")
- `ENVS`: Comma-separated list of environments (default: "cartpole,higherlower,minesweeper,repeat_first,repeat_previous")
- `DIMS`: Comma-separated list of dimensions (default: "64,128")
- `EVAL_METHODS`: Comma-separated list of evaluation methods (default: "tiling,padding")
- `DEPTHS`: Comma-separated list of depths (default: "1,2,4,8")
- `ARCHS`: Comma-separated list of architectures (default: "gru,s5")
- `METRICS`: Comma-separated list of metrics (default: "metric,eval_metric")
- `OUTPUT_DIR`: Directory to save plots (default: "figures/wandb_plots")
- `CACHE_DIR`: Directory to store cached data (default: "wandb_cache")

### Running Individual Scripts

#### Preprocessing

```bash
cd figures/popjaxrl
./preprocess_for_dim_plots.py \
  --project "popgym_sweep_envs" \
  --entity "cmvl_nelf" \
  --sweep_id "8z7oojm2" \
  --metrics "metric,eval_metric" \
  --envs "cartpole,higherlower,minesweeper,repeat_first,repeat_previous" \
  --dims "64,128" \
  --eval_methods "tiling,padding" \
  --depths "1,2,4,8" \
  --archs "gru,s5" \
  --cache_dir "wandb_cache"
```

#### Plotting

```bash
cd figures/popjaxrl
./plot_dim_results.py \
  --sweep_id "8z7oojm2" \
  --envs "cartpole,higherlower,minesweeper,repeat_first,repeat_previous" \
  --eval_methods "tiling,padding" \
  --depths "1,2,4,8" \
  --archs "gru,s5" \
  --metric "eval_metric" \
  --output_dir "figures/wandb_plots" \
  --cache_dir "wandb_cache"
```

## Output

The scripts will create plots with the following characteristics:

- x-axis: dimension (dim)
- y-axis: preprocessed eval_metric
- Multiple lines in each plot for different depths (1, 2, 4, 8)
- Separate plots for each combination of arch, env, and eval_method

Plots are saved as both PNG and PDF files with the naming pattern:
`dim_results_<env>_<eval_method>.png/pdf`

## Example

To create plots for the default settings:

```bash
cd figures/popjaxrl
./plot_dim_results.sh
```

This will create plots for all combinations of environments and evaluation methods, with dimension on the x-axis and eval_metric on the y-axis, showing different lines for each depth.
