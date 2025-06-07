#!/bin/bash
# Script to run both preprocessing and dimension plotting scripts
# This script creates plots with dimension on the x-axis and eval_metric on the y-axis

# Create directories if they don't exist
mkdir -p figures/wandb_plots
mkdir -p wandb_cache

# Parse command line arguments
PROJECT=${1:-"popgym_sweep_dim"}
ENTITY=${2:-"cmvl_nelf"}
SWEEP_ID=${3:-"hogrka32"}
ENVS=${4:-"cartpole,minesweeper"}
DIMS=${5:-"32,64,96,128,256"}
EVAL_METHODS=${6:-"identity,tiling,padding"}
DEPTHS=${7:-"1,2,4,8"}
ARCHS=${8:-"gru,s5"}
METRICS=${9:-"metric,eval_metric"}
OUTPUT_DIR=${10:-"figures/wandb_plots"}
CACHE_DIR=${11:-"wandb_cache"}
LEGEND_ONLY=${12:-""}  # New parameter for legend_only flag

echo "Running preprocessing for dimension plots..."
python preprocess_for_dim_plots.py \
  --project "$PROJECT" \
  --entity "$ENTITY" \
  --sweep_id "$SWEEP_ID" \
  --metrics "$METRICS" \
  --envs "$ENVS" \
  --dims "$DIMS" \
  --eval_methods "$EVAL_METHODS" \
  --depths "$DEPTHS" \
  --archs "$ARCHS" \
  --cache_dir "$CACHE_DIR"

# Add legend_only flag if specified
LEGEND_ONLY_FLAG=""
if [ ! -z "$LEGEND_ONLY" ]; then
  LEGEND_ONLY_FLAG="--legend_only"
  echo "Running dimension plotting (legend only)..."
else
  echo "Running dimension plotting..."
fi

python plot_dim_results.py \
  --sweep_id "$SWEEP_ID" \
  --envs "$ENVS" \
  --eval_methods "$EVAL_METHODS" \
  --depths "$DEPTHS" \
  --archs "$ARCHS" \
  --metric "eval_metric" \
  --output_dir "$OUTPUT_DIR" \
  --cache_dir "$CACHE_DIR" \
  $LEGEND_ONLY_FLAG

echo "Plotting complete!"
echo "Plots have been saved to $OUTPUT_DIR/"
if [ ! -z "$LEGEND_ONLY" ]; then
  echo "Generated legend file:"
  echo "  - legend_dim_results.pdf (standalone legend)"
else
  echo "Check the following files for each environment and evaluation method:"
  echo "  - dim_results_<env>_<eval_method>.png/pdf"
  echo "  - legend_dim_results.pdf (standalone legend)"
fi
