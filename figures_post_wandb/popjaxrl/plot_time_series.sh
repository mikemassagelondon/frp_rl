#!/bin/bash
# Script to run both preprocessing/time series and final results plotting scripts
# Replace YOUR_PROJECT_NAME, YOUR_ENTITY, and YOUR_SWEEP_ID with your actual values

# Create directories if they don't exist
mkdir -p figures/wandb_plots
mkdir -p wandb_cache

# Parse command line arguments
PROJECT=${1:-"popgym_sweep_envs"}
ENTITY=${2:-"cmvl_nelf"}
#SWEEP_ID=${3:-"3a5yximw"} for succeding hstate
SWEEP_ID=${3:-"8z7oojm2"}
#ENVS=${4:-"cartpole,minesweeper,higherlower,battleship,count_recall,repeat_first,repeat_previous"}
ENVS=${4:-"cartpole,higherlower,minesweeper,repeat_first,repeat_previous"}
DIMS=${5:-"64,128"}
EVAL_METHODS=${6:-"identity,tiling,padding"}
DEPTHS=${7:-"1,2,4,8"}
ARCHS=${8:-"gru,s5"}
METRICS=${9:-"metric,eval_metric"}
#OUTPUT_DIR=${10:-"figures/"}
OUTPUT_DIR=${10:-"figures_8z/"}
CACHE_DIR=${11:-"wandb_cache"}
LEGEND_ONLY=${12:-"true"}  # New parameter for legend_only flag

# Add legend_only flag if specified
LEGEND_ONLY_FLAG=""
if [ ! -z "$LEGEND_ONLY" ]; then
  LEGEND_ONLY_FLAG="--legend_only"
  echo "Running preprocessing and time series plotting (legend only)..."
else
  echo "Running preprocessing and time series plotting..."
fi

python preprocess_and_time_series.py \
  --project "$PROJECT" \
  --entity "$ENTITY" \
  --sweep_id "$SWEEP_ID" \
  --metrics "$METRICS" \
  --envs "$ENVS" \
  --dims "$DIMS" \
  --eval_methods "$EVAL_METHODS" \
  --depths "$DEPTHS" \
  --archs "$ARCHS" \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  $LEGEND_ONLY_FLAG

# Add legend_only flag if specified for final results plotting
if [ ! -z "$LEGEND_ONLY" ]; then
  echo "Running final results plotting (legend only)..."
else
  echo "Running final results plotting..."
fi

python final_results_plot.py \
  --sweep_id "$SWEEP_ID" \
  --envs "$ENVS" \
  --dims "$DIMS" \
  --eval_methods "$EVAL_METHODS" \
  --depths "$DEPTHS" \
  --archs "$ARCHS" \
  --metrics "${METRICS%%,*},${METRICS#*,}" \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  $LEGEND_ONLY_FLAG

echo "Plotting complete!"
echo "Plots have been saved to $OUTPUT_DIR/"
if [ ! -z "$LEGEND_ONLY" ]; then
  echo "Generated legend files:"
  echo "  - legend_time_series.pdf (standalone legend for time series plots)"
  echo "  - legend_time_series_pickup.pdf (standalone legend for pickup plots)"
  echo "  - legend_final_results.pdf (standalone legend for final results plots)"
else
  echo "Check the following files for each environment, dimension, and evaluation method:"
  echo "  - time_series_<env>_dim<dim>_<eval_method>.png/pdf"
  echo "  - time_series_pickup_<env>_dim<dim>_<eval_method>.png/pdf"
  echo "  - final_results_<env>_dim<dim>_<eval_method>.png/pdf"
  echo "  - legend_time_series.pdf (standalone legend for time series plots)"
  echo "  - legend_time_series_pickup.pdf (standalone legend for pickup plots)"
  echo "  - legend_final_results.pdf (standalone legend for final results plots)"
fi
