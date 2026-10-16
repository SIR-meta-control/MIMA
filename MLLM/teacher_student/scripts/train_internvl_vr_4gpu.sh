#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERNVL_ROOT="${PROJECT_ROOT}/vendor/internvl_chat"

GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-12320}"

PER_DEVICE_BATCH_SIZE="${PER_DEVICE_BATCH_SIZE:-1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-16}"
GRADIENT_ACC="$(( GLOBAL_BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS_PER_NODE ))"
if [ "${GRADIENT_ACC}" -lt 1 ]; then
  GRADIENT_ACC=1
fi

: "${MODEL_NAME_OR_PATH:?Set MODEL_NAME_OR_PATH to an authorized initialization checkpoint}"
: "${META_PATH:?Set META_PATH to the generated training dataset metadata JSON}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/internvl_vr_$(date +%Y%m%d_%H%M%S)}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-}"

EXTRA_TRAIN_ARGS=()
if [ -n "${DEEPSPEED_CONFIG}" ]; then
  EXTRA_TRAIN_ARGS+=(--deepspeed "${DEEPSPEED_CONFIG}")
fi

mkdir -p "${OUTPUT_DIR}"
cd "${INTERNVL_ROOT}"

export PYTHONPATH="${INTERNVL_ROOT}:${PROJECT_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export TOKENIZERS_PARALLELISM=true

LAUNCHER=("${PYTHON:-python}")
if [ "${GPUS_PER_NODE}" -gt 1 ]; then
  LAUNCHER+=(
    -m torch.distributed.run
    --nproc_per_node "${GPUS_PER_NODE}"
    --nnodes "${NNODES}"
    --node_rank "${NODE_RANK}"
    --master_addr "${MASTER_ADDR}"
    --master_port "${MASTER_PORT}"
  )
else
  export RANK="${RANK:-0}"
  export LOCAL_RANK="${LOCAL_RANK:-0}"
  export WORLD_SIZE="${WORLD_SIZE:-1}"
  export MASTER_ADDR
  export MASTER_PORT
fi

"${LAUNCHER[@]}" internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --conv_style "phi3-chat" \
  --output_dir "${OUTPUT_DIR}" \
  --meta_path "${META_PATH}" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch "${MAX_DYNAMIC_PATCH:-6}" \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm "${FREEZE_LLM:-True}" \
  --freeze_mlp1 "${FREEZE_MLP1:-False}" \
  --freeze_mlp2 "${FREEZE_MLP2:-False}" \
  --freeze_backbone "${FREEZE_BACKBONE:-True}" \
  --freeze_pc_backbone "${FREEZE_PC_BACKBONE:-True}" \
  --vision_select_layer -1 \
  --unfreeze_lm_head "${UNFREEZE_LM_HEAD:-False}" \
  --use_llm_lora "${USE_LLM_LORA:-0}" \
  --use_backbone_lora "${USE_BACKBONE_LORA:-0}" \
  --use_data_resampling False \
  --dataloader_num_workers 4 \
  --bf16 True \
  --num_train_epochs "${NUM_TRAIN_EPOCHS:-3}" \
  --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
  --gradient_accumulation_steps "${GRADIENT_ACC}" \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps "${SAVE_STEPS:-200}" \
  --save_total_limit "${SAVE_TOTAL_LIMIT:-3}" \
  --learning_rate "${LEARNING_RATE:-4e-5}" \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --optim "adamw_torch" \
  --logging_steps 1 \
  --max_seq_length 4096 \
  --do_train True \
  --grad_checkpoint "${GRAD_CHECKPOINT:-False}" \
  --group_by_length True \
  --ddp_find_unused_parameters "${DDP_FIND_UNUSED_PARAMETERS:-False}" \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version "v2" \
  "${EXTRA_TRAIN_ARGS[@]}" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
