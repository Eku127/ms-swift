#!/bin/bash
# StreamVLN Single-Node Training Script - Qwen2.5-VL (ms-swift)
# 
# Usage:
#   bash examples/vln/streamvln/script/H100/train/train_streamvln_qwen2_5_vl_single_node.sh
#
# This script supports single-node training with configurable GPU count.

set -e  # Exit on error

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-train

# ============================================================================
# GPU Configuration
# ============================================================================
CUDA_DEVICES="0,1,2,3,4,5,6,7"    # GPUs to use (comma-separated)
MASTER_PORT=29500                  # Master port for distributed training

# Auto-detect GPU count from CUDA_DEVICES
GPUS_PER_NODE=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_TYPE="streamvln_qwen2_5_vl"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# Extract model size for experiment naming
MODEL_SIZE=$(echo "$MODEL_PATH" | grep -oE '[0-9]+B' | tr '[:upper:]' '[:lower:]')
MODEL_SIZE=${MODEL_SIZE:-"3b"}

# ============================================================================
# VLN Data Configuration
# ============================================================================
VLN_DATA_PATHS=(
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/R2R"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/RxR_new"
    "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/EnvDrop"
    # "/mnt/data3/jiangjiajun/dataset/streamvln_datasets/trajectory_data/ScaleVLN"
)
VLN_DATA_PATH=$(IFS=','; echo "${VLN_DATA_PATHS[*]}")

# VLN-Specific Parameters
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
USE_RANDOM=false
MAX_SAMPLES="0"  # 0 = use all samples

# ============================================================================
# Training Parameters
# ============================================================================
TRAIN_TYPE="full"
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=8
GRAD_ACCUM_STEPS=1
MAX_LENGTH=32768 # 32768 for 7b

# Model Freezing
FREEZE_VIT=false
FREEZE_LLM=false
FREEZE_ALIGNER=false

# Optimization
USE_DEEPSPEED=true
DEEPSPEED_CONFIG="zero2"
GRADIENT_CHECKPOINTING=true
TF32=true
TORCH_COMPILE=false

# Learning Rate Schedule
WARMUP_RATIO=0.075
WEIGHT_DECAY=0.
LR_SCHEDULER_TYPE="cosine_with_min_lr"
# LR_SCHEDULER_KWARGS='{"min_lr":3.7e-05}'
LR_SCHEDULER_KWARGS='{"min_lr":1.85e-05}'

# Attention Implementation
# ATTN_IMPL="sdpa"  # flash_attn, sdpa, or eager
ATTN_IMPL="flash_attn"  # flash_attn, sdpa, or eager

# ============================================================================
# Performance Acceleration (NEW)
# ============================================================================
# padding_free: Flatten data to avoid padding overhead (requires flash_attn)
PADDING_FREE=true

# use_liger_kernel: Liger kernel for faster training and less GPU memory
# Install: pip install liger-kernel
USE_LIGER_KERNEL=true  # Set to true after installing liger-kernel

# Dataloader optimization
DATALOADER_PREFETCH_FACTOR=10
DATALOADER_PERSISTENT_WORKERS=true

# Dataset preprocessing parallelism (keep low for multimodal)
DATASET_NUM_PROC=2

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * GPUS_PER_NODE))
EXP_NAME="streamvln-qwen2.5vl-${MODEL_SIZE}-full-${NUM_EPOCHS}ep-f${NUM_FRAMES}h${NUM_HISTORY}s${NUM_FUTURE_STEPS}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
OUTPUT_DIR="output/${EXP_NAME}"

# Checkpoint Management
# Note: Without a validation dataset, load_best_model_at_end cannot work properly
# For full training without validation, we just save checkpoints periodically
SAVE_STEPS=500
SAVE_TOTAL_LIMIT=3  # Keep last 3 checkpoints for safety
LOGGING_STEPS=10

# ============================================================================
# SwanLab Configuration
# ============================================================================
USE_SWANLAB=true
SWANLAB_PROJECT="StreamVLN"
SWANLAB_EXP_NAME="${EXP_NAME}"
SWANLAB_MODE="cloud"

# Enterprise WeChat (WXWork) Notification Configuration
# Reference: https://docs.swanlab.cn/plugin/notification-wxwork.html
USE_WXWORK_NOTIFICATION=true
SWANLAB_NOTIFICATION_METHOD="wxwork"
SWANLAB_WEBHOOK_URL="https://qyapi.weixin.qq.com/cgi-bin/webhook/send?key=d78d3128-7b16-4bf1-a6a7-403bf0915fe0"  # Replace with your actual webhook URL
SWANLAB_SECRET=""  # Optional: set if your webhook requires secret

# ============================================================================
# Environment Setup
# ============================================================================
export PYTORCH_ALLOC_CONF=expandable_segments:True
export NCCL_DEBUG=ERROR
export NCCL_TIMEOUT=1800
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=4
# ModelScope cache will use default: ~/.cache/modelscope/hub
# If you want to use a custom cache directory, uncomment and set:
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# ============================================================================
# Print Configuration
# ============================================================================
echo "=========================================="
echo "StreamVLN Single-Node Training"
echo "=========================================="
echo "Model: $MODEL_TYPE ($MODEL_PATH)"
echo "Data: $VLN_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "------------------------------------------"
echo "GPUs: $GPUS_PER_NODE ($CUDA_DEVICES)"
echo "Batch: ${BATCH_SIZE} x ${GRAD_ACCUM_STEPS} x ${GPUS_PER_NODE} = ${EFFECTIVE_BATCH_SIZE}"
echo "LR: $LEARNING_RATE | Epochs: $NUM_EPOCHS"
echo "Attention: $ATTN_IMPL"
echo "------------------------------------------"
echo "Freeze ViT: $FREEZE_VIT | LLM: $FREEZE_LLM | Aligner: $FREEZE_ALIGNER"
echo "DeepSpeed: $USE_DEEPSPEED ($DEEPSPEED_CONFIG)"
echo "------------------------------------------"
echo "Acceleration:"
echo "  padding_free: $PADDING_FREE"
echo "  use_liger_kernel: $USE_LIGER_KERNEL"
echo "  dataloader_prefetch: $DATALOADER_PREFETCH_FACTOR"
echo "------------------------------------------"
echo "SwanLab:"
echo "  Project: $SWANLAB_PROJECT"
echo "  Experiment: $SWANLAB_EXP_NAME"
if [ "$USE_WXWORK_NOTIFICATION" = true ]; then
    echo "  WXWork Notification: Enabled"
    echo "  Webhook URL: ${SWANLAB_WEBHOOK_URL:0:20}..."  # Show first 20 chars
fi
echo "=========================================="

# ============================================================================
# Build Arguments
# ============================================================================
# Calculate ms-swift root directory
# Script is at: examples/vln/streamvln/script/H100/train/train_streamvln_qwen2_5_vl_single_node.sh
# Need to go up 7 levels to reach ms-swift root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MS_SWIFT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"

# DeepSpeed argument
DEEPSPEED_ARG=""
[ "$USE_DEEPSPEED" = true ] && DEEPSPEED_ARG="--deepspeed $DEEPSPEED_CONFIG"

# SwanLab arguments
SWANLAB_ARGS=""
if [ "$USE_SWANLAB" = true ]; then
    SWANLAB_ARGS="--report_to swanlab --swanlab_project $SWANLAB_PROJECT --swanlab_exp_name $SWANLAB_EXP_NAME --swanlab_mode $SWANLAB_MODE"
    
    # Add WXWork notification if enabled
    if [ "$USE_WXWORK_NOTIFICATION" = true ]; then
        SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_notification_method $SWANLAB_NOTIFICATION_METHOD --swanlab_webhook_url $SWANLAB_WEBHOOK_URL"
        if [ -n "$SWANLAB_SECRET" ]; then
            SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_secret $SWANLAB_SECRET"
        fi
    fi
fi

# Attention implementation argument
ATTN_ARG=""
[ -n "$ATTN_IMPL" ] && ATTN_ARG="--attn_impl $ATTN_IMPL"

# ============================================================================
# Run Training
# ============================================================================
cd "$MS_SWIFT_ROOT"

torchrun \
    --nnodes=1 \
    --node_rank=0 \
    --nproc_per_node=$GPUS_PER_NODE \
    --master_addr=localhost \
    --master_port=$MASTER_PORT \
    examples/vln/streamvln/trainer.py \
    --custom_register_path examples/vln/streamvln \
    --model_type $MODEL_TYPE \
    --model $MODEL_PATH \
    --dataset $VLN_DATA_PATH \
    --train_type $TRAIN_TYPE \
    --torch_dtype bfloat16 \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS \
    --max_length $MAX_LENGTH \
    --output_dir $OUTPUT_DIR \
    --save_steps $SAVE_STEPS \
    --save_total_limit $SAVE_TOTAL_LIMIT \
    --logging_steps $LOGGING_STEPS \
    --save_strategy steps \
    --warmup_ratio $WARMUP_RATIO \
    --weight_decay $WEIGHT_DECAY \
    --lr_scheduler_type $LR_SCHEDULER_TYPE \
    --lr_scheduler_kwargs "$LR_SCHEDULER_KWARGS" \
    --gradient_checkpointing $GRADIENT_CHECKPOINTING \
    --freeze_vit $FREEZE_VIT \
    --freeze_llm $FREEZE_LLM \
    --freeze_aligner $FREEZE_ALIGNER \
    --dataloader_num_workers 8 \
    --dataloader_drop_last true \
    --dataloader_prefetch_factor $DATALOADER_PREFETCH_FACTOR \
    --dataloader_persistent_workers $DATALOADER_PERSISTENT_WORKERS \
    --dataset_num_proc $DATASET_NUM_PROC \
    --ddp_timeout 3600 \
    --num_frames $NUM_FRAMES \
    --num_history $NUM_HISTORY \
    --num_future_steps $NUM_FUTURE_STEPS \
    --use_random $USE_RANDOM \
    --vln_max_samples $MAX_SAMPLES \
    --tf32 $TF32 \
    --torch_compile $TORCH_COMPILE \
    --padding_free $PADDING_FREE \
    --use_liger_kernel $USE_LIGER_KERNEL \
    $ATTN_ARG \
    $DEEPSPEED_ARG \
    $SWANLAB_ARGS

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "=========================================="

