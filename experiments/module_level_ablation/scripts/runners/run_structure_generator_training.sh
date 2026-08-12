#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
fi
conda activate gvae

mkdir -p logs
log_path="logs/structure_generator_$(date +%Y%m%d_%H%M%S).log"
RAW_STRUCTURE_DIR="${RAW_STRUCTURE_DIR:-datasets/structure_config}"
PROCESSED_DIR="${PROCESSED_DIR:-datasets/processed_dataset}"
SPLIT_DIR="${SPLIT_DIR:-${PROCESSED_DIR}/split_seed7_val20}"
CKPT_DIR="${CKPT_DIR:-ckpts}"

python -u scripts/train/train_structure_generator.py \
  --train-pairs "${SPLIT_DIR}/train_pairs.jsonl" \
  --val-pairs "${SPLIT_DIR}/val_pairs.jsonl" \
  --graph-imputation configs/graph_imputation.yaml \
  --angle-reference "${RAW_STRUCTURE_DIR}/4-bar/data_0000.json" \
  --output-dir "${CKPT_DIR}/structure_generator" \
  --max-pairs 0 \
  --epochs 40 \
  --batch-size 512 \
  --lr 0.001 \
  --weight-decay 0.0001 \
  --save-interval 5 \
  --log-interval 1 \
  --seed 7 \
  --w-nodes 1.0 \
  --w-edge-pose 1.0 \
  --w-leg-base 0.5 \
  --w-leg-angle 1.0 \
  --w-scale 1.0 \
  --w-scale-teacher 0.5 \
  --w-spacing 1.0 \
  --w-geometry 0.5 \
  --w-angle-equal 0.2 \
  --w-size 1.0 \
  --w-task 1.0 \
  --w-kl 0.0001 \
  2>&1 | tee "${log_path}"

echo "log_path=${log_path}"
