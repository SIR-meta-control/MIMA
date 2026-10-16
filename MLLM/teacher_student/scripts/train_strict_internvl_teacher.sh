#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/outputs/internvl_vr_data}"
export PYTHON="${PYTHON:-python}"
export META_PATH="${META_PATH:-${DATA_ROOT}/strict_teacher_train_meta.json}"
export OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/internvl_vr_strict_teacher_$(date +%Y%m%d_%H%M%S)}"

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to an authorized clean initialization checkpoint}"
export MODEL_NAME_OR_PATH
export FREEZE_LLM="${FREEZE_LLM:-True}"
export FREEZE_BACKBONE="${FREEZE_BACKBONE:-True}"
export FREEZE_PC_BACKBONE="${FREEZE_PC_BACKBONE:-False}"
export USE_LLM_LORA="${USE_LLM_LORA:-16}"
export LEARNING_RATE="${LEARNING_RATE:-5e-5}"
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-12}"
export SAVE_STEPS="${SAVE_STEPS:-50}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-20}"

bash "${PROJECT_ROOT}/scripts/train_internvl_vr_4gpu.sh"
