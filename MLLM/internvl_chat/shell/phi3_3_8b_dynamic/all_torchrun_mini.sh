set -x

PARTITION=${PARTITION:-"INTERN2"}
GPUS=${GPUS:-128}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
QUOTA_TYPE=${QUOTA_TYPE:-"reserved"}
NODES=$((GPUS / GPUS_PER_NODE))
CPUS_PER_TASK=${CPUS_PER_TASK:-10}
SRUN_ARGS=${SRUN_ARGS:-""}
BATCH_SIZE=${BATCH_SIZE:-1024}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-4}
GRADIENT_ACC=$((BATCH_SIZE / PER_DEVICE_BATCH_SIZE / GPUS))

GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
MASTER_ADDR=127.0.0.2
MASTER_PORT=12321

DISTRIBUTED_ARGS="
--nproc_per_node $GPUS_PER_NODE
--nnodes $NNODES
--node_rank $NODE_RANK
--master_addr $MASTER_ADDR
--master_port $MASTER_PORT
"

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export MASTER_PORT=34229
export TF_CPP_MIN_LOG_LEVEL=3

OUTPUT_DIR=${OUTPUT_DIR:-"work_dirs/ours_mini_4b_finetune_f1"}

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir -p "$OUTPUT_DIR"
fi

# number of gpus: 128
# batch size per gpu: 4
# gradient accumulation steps: 2
# total batch size: 1024
# epoch: 1
torchrun $DISTRIBUTED_ARGS internvl/train/internvl_chat_finetune.py \
  --model_name_or_path "/fuxi_team14/users/haoyanghe/codes/jszn/InternVL/models/Ours_Mini-InternVL-4B/models--OpenGVLab--Mini-InternVL-Chat-4B-V1-5/snapshots/6f97087daec17e4b033d4d846c0b64c09c4268cd" \
  --conv_style "phi3-chat" \
  --output_dir ${OUTPUT_DIR} \
  --meta_path "/fuxi_team14/users/haoyanghe/codes/jszn/InternVL/internvl_chat/shell/data/finetune_all.json" \
  --overwrite_output_dir True \
  --force_image_size 448 \
  --max_dynamic_patch 12 \
  --down_sample_ratio 0.5 \
  --drop_path_rate 0.1 \
  --pad2square False \
  --freeze_llm True \
  --freeze_mlp1 True \
  --freeze_mlp2 False \
  --freeze_backbone True \
  --freeze_pc_backbone True \
  --vision_select_layer -1 \
  --use_data_resampling False \
  --dataloader_num_workers 0 \
  --bf16 True \
  --num_train_epochs 2 \
  --per_device_train_batch_size ${PER_DEVICE_BATCH_SIZE} \
  --gradient_accumulation_steps ${GRADIENT_ACC} \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 200 \
  --save_total_limit 3 \
  --learning_rate 4e-5 \
  --weight_decay 0.05 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "cosine" \
  --logging_steps 1 \
  --max_seq_length 8192 \
  --do_train True \
  --grad_checkpoint True \
  --group_by_length True \
  --dynamic_image_size True \
  --use_thumbnail True \
  --ps_version 'v2' \
  --deepspeed "zero_stage1_config.json" \
  --report_to "tensorboard" \
  2>&1 | tee -a "${OUTPUT_DIR}/training_log.txt"
