#!/bin/bash
# StreamVLN Training Script - Qwen2.5-VL
# 
# Usage: bash train_streamvln_qwen2_vl.sh
#
# This script trains a StreamVLN model for Visual Language Navigation
# using the Qwen2.5-VL architecture with custom VLN adaptations.
#
# Note: This script uses the refactored StreamVLN module at examples/vln/streamvln/
#
# Prerequisites:
# - Install SwanLab: pip install swanlab
# - Login to SwanLab: swanlab login (or set --swanlab_token)

# ============================================================================
# VLN-Specific Parameters
# ============================================================================
NUM_FRAMES=32              # Window size for trajectory segmentation
NUM_HISTORY=8              # Number of historical frames to sample
NUM_FUTURE_STEPS=4         # Number of actions to predict per round
USE_RANDOM=false           # Use random sampling for history (false=uniform)

# Dataset Size Limit (Optional)
# Set to limit the number of training samples (useful for quick testing)
# Example: MAX_SAMPLES=1000 will only use the first 1000 samples
# Set to empty or 0 to use all available samples
MAX_SAMPLES="2"             # Empty or 0 = use all samples, e.g., "1000" = use 1000 samples

# ============================================================================
# Model and Data Configuration
# ============================================================================
# MODEL_TYPE 必须与注册的模型类型名称匹配
# 当前仅支持 Qwen2.5-VL: streamvln_qwen2_5_vl
MODEL_TYPE="streamvln_qwen2_5_vl"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # Base model to extend

# VLN Data Path - MODIFY THIS TO YOUR DATA LOCATION
# Expected format: directory containing annotations.json and video folders
VLN_DATA_PATH="/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/R2R"

# ============================================================================
# Training Parameters
# ============================================================================
TRAIN_TYPE="full"          # Training type: full, lora, etc.
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=1               # Per-device batch size
GRAD_ACCUM_STEPS=1         # Gradient accumulation steps
MAX_LENGTH=16384            # Maximum sequence length

# ============================================================================
# Resource Configuration
# ============================================================================
NUM_GPUS=1
CUDA_DEVICES="0"
MASTER_PORT=29500

# ============================================================================
# Output Configuration
# ============================================================================
OUTPUT_DIR="output/streamvln-qwen2-vl-3b"
SAVE_STEPS=100
EVAL_STEPS=50
SAVE_TOTAL_LIMIT=3
LOGGING_STEPS=10

# ============================================================================
# Model Component Freezing Configuration
# ============================================================================
# Control which parts of the model are trainable
FREEZE_VIT=true            # Freeze vision encoder (ViT)
FREEZE_LLM=true           # Freeze language model (LLM)
FREEZE_ALIGNER=false       # Freeze MLP adapter/projector (aligner)
                            # Note: In original StreamVLN, mm_tunable_parts controls this
                            # Setting all to false means training all components

# ============================================================================
# Optimization Settings
# ============================================================================
USE_DEEPSPEED=false
DEEPSPEED_CONFIG="zero2"   # zero2 or zero3
GRADIENT_CHECKPOINTING=true
DDP_TIMEOUT=3600           # Distributed training timeout (seconds)

# ============================================================================
# Learning Rate Configuration
# ============================================================================
# Note: ms-swift doesn't support separate vision_tower_lr like original StreamVLN
# If you need different LR for vision, consider using freeze_vit=true or LoRA
WARMUP_RATIO=0.075         # Match original StreamVLN (0.075)
WEIGHT_DECAY=0.01          # ms-swift default (original uses 0.)
LR_SCHEDULER_TYPE="cosine_with_min_lr"  # Match original StreamVLN
# Note: JSON must not have spaces for shell compatibility
LR_SCHEDULER_KWARGS='{"min_lr":1.85e-05}'  # Match original StreamVLN min_lr

# ============================================================================
# SwanLab Configuration (Experiment Tracking)
# ============================================================================
USE_SWANLAB=true           # Enable SwanLab logging
SWANLAB_PROJECT="StreamVLN"
SWANLAB_EXP_NAME="streamvln-qwen2-vl-3b-$(date +%Y%m%d-%H%M%S)"
SWANLAB_MODE="cloud"       # cloud or local
# SWANLAB_TOKEN=""         # Optional: set if not logged in via CLI

# ============================================================================
# Environment Setup
# ============================================================================
# PyTorch CUDA memory allocation strategy
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# NCCL Configuration (for single-node multi-GPU training)
# Note: This script is for single-node 8-GPU training
# For multi-node training, add NNODES, NODE_RANK, MASTER_ADDR environment variables
export NCCL_TIMEOUT=1800           # 30 minutes timeout (for large model/data)
export NCCL_DEBUG=ERROR             # INFO/WARN/ERROR (WARN for production)
export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude docker and loopback interfaces

# Model Cache Configuration
# ms-swift 默认使用 ModelScope，可通过 MODELSCOPE_CACHE 设置缓存目录
# 如果使用 HuggingFace，设置 USE_HF=1 和 HF_HOME
export MODELSCOPE_CACHE=/shared_space/jiangjiajun/modelscope_cache  # ModelScope 缓存目录
# export USE_HF=1  # 取消注释以使用 HuggingFace
# export HF_HOME=/shared_space/jiangjiajun/hf_cache  # HuggingFace 缓存目录（当 USE_HF=1 时）

# Dataset size limit (passed to StreamVLNDataset via argument)
# if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "0" ] && [ "$MAX_SAMPLES" != "" ]; then
#     export VLN_MAX_SAMPLES=$MAX_SAMPLES
#     echo "Dataset size limit: $MAX_SAMPLES samples"
# else
#     unset VLN_MAX_SAMPLES
#     echo "Using all available samples"
# fi

# ============================================================================
# Training Command
# ============================================================================
echo "=========================================="
echo "Starting StreamVLN Training..."
echo "=========================================="
echo "Model: $MODEL_TYPE"
echo "Base Model: $MODEL_PATH"
echo "Data Path: $VLN_DATA_PATH"
echo "Output: $OUTPUT_DIR"
echo "----------------------------------------"
echo "Training Configuration:"
echo "  Epochs: $NUM_EPOCHS"
echo "  Batch Size: $BATCH_SIZE (per device)"
echo "  Gradient Accumulation: $GRAD_ACCUM_STEPS"
echo "  Effective Batch Size: $((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS))"
echo "  Learning Rate: $LEARNING_RATE"
if [ -n "$MAX_SAMPLES" ] && [ "$MAX_SAMPLES" != "0" ] && [ "$MAX_SAMPLES" != "" ]; then
    echo "  Max Samples: $MAX_SAMPLES (limited)"
fi
echo "----------------------------------------"
echo "Model Component Freezing:"
echo "  Vision Encoder (ViT): $([ "$FREEZE_VIT" = true ] && echo "FROZEN" || echo "TRAINABLE")"
echo "  Language Model (LLM): $([ "$FREEZE_LLM" = true ] && echo "FROZEN" || echo "TRAINABLE")"
echo "  MLP Adapter (Aligner): $([ "$FREEZE_ALIGNER" = true ] && echo "FROZEN" || echo "TRAINABLE")"
echo "----------------------------------------"
if [ "$USE_SWANLAB" = true ]; then
    echo "SwanLab Tracking: ENABLED"
    echo "  Project: $SWANLAB_PROJECT"
    echo "  Experiment: $SWANLAB_EXP_NAME"
fi
echo "=========================================="

# Build DeepSpeed argument
if [ "$USE_DEEPSPEED" = true ]; then
    DEEPSPEED_ARG="--deepspeed $DEEPSPEED_CONFIG"
else
    DEEPSPEED_ARG=""
fi

# Build SwanLab arguments
if [ "$USE_SWANLAB" = true ]; then
    SWANLAB_ARGS="--report_to swanlab --swanlab_project $SWANLAB_PROJECT --swanlab_exp_name $SWANLAB_EXP_NAME --swanlab_mode $SWANLAB_MODE"
    if [ -n "$SWANLAB_TOKEN" ]; then
        SWANLAB_ARGS="$SWANLAB_ARGS --swanlab_token $SWANLAB_TOKEN"
    fi
else
    SWANLAB_ARGS=""
fi

# Build training arguments
TRAIN_ARGS="
    --custom_register_path examples/vln/streamvln
    --model_type $MODEL_TYPE
    --model $MODEL_PATH
    --dataset $VLN_DATA_PATH
    --train_type $TRAIN_TYPE
    --torch_dtype bfloat16
    --num_train_epochs $NUM_EPOCHS
    --learning_rate $LEARNING_RATE
    --per_device_train_batch_size $BATCH_SIZE
    --per_device_eval_batch_size $BATCH_SIZE
    --gradient_accumulation_steps $GRAD_ACCUM_STEPS
    --max_length $MAX_LENGTH
    --output_dir $OUTPUT_DIR
    --save_steps $SAVE_STEPS
    --eval_steps $EVAL_STEPS
    --save_total_limit $SAVE_TOTAL_LIMIT
    --logging_steps $LOGGING_STEPS
    --warmup_ratio $WARMUP_RATIO
    --weight_decay $WEIGHT_DECAY
    --lr_scheduler_type $LR_SCHEDULER_TYPE
    --gradient_checkpointing $GRADIENT_CHECKPOINTING
    --freeze_vit $FREEZE_VIT
    --freeze_llm $FREEZE_LLM
    --freeze_aligner $FREEZE_ALIGNER
    --dataloader_num_workers 8
    --dataloader_drop_last true
    --ddp_timeout $DDP_TIMEOUT
    --num_frames $NUM_FRAMES
    --num_history $NUM_HISTORY
    --num_future_steps $NUM_FUTURE_STEPS
    --use_random $USE_RANDOM
    --vln_max_samples $MAX_SAMPLES
    $DEEPSPEED_ARG
    $SWANLAB_ARGS
"

# Run training
# Use torchrun for multi-GPU distributed training (DDP)
# Single GPU uses python directly, multi-GPU uses torchrun
# NOTE: DataParallel is NOT compatible with Qwen2.5-VL due to 3D position embeddings
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using torchrun for distributed training (DDP, NUM_GPUS=$NUM_GPUS)"
    torchrun \
        --nproc_per_node=$NUM_GPUS \
        --master_port=$MASTER_PORT \
        examples/vln/streamvln/trainer.py \
        $TRAIN_ARGS \
        --lr_scheduler_kwargs "$LR_SCHEDULER_KWARGS"
else
    echo "Using python for single GPU training"
    python examples/vln/streamvln/trainer.py \
        $TRAIN_ARGS \
        --lr_scheduler_kwargs "$LR_SCHEDULER_KWARGS"
fi

echo "=========================================="
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
if [ "$USE_SWANLAB" = true ]; then
    echo "View results at: https://swanlab.cn/$SWANLAB_PROJECT/$SWANLAB_EXP_NAME"
fi
echo "=========================================="
