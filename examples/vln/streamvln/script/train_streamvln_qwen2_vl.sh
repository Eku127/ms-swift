#!/bin/bash
# StreamVLN Training Script - Qwen2.5-VL
# 
# Usage: bash examples/vln/streamvln/script/train_streamvln_qwen2_vl.sh
#        or (from ms-swift root): bash examples/vln/streamvln/script/train_streamvln_qwen2_vl.sh
#
# This script trains a StreamVLN model for Visual Language Navigation
# using Qwen2.5-VL as the base model.
#
# Architecture:
#   - Model inherits directly from Qwen2.5-VL without modifications
#   - ms-swift framework handles all multimodal processing (image → pixel_values → fusion)
#   - Custom VLN dataset provides multi-turn dialogue with navigation actions
#
# Module location: examples/vln/streamvln/
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
MAX_SAMPLES="0"             # Empty or 0 = use all samples, e.g., "1000" = use 1000 samples

# ============================================================================
# Model and Data Configuration
# ============================================================================
# MODEL_TYPE 必须与注册的模型类型名称匹配
# 当前仅支持 Qwen2.5-VL: streamvln_qwen2_5_vl
MODEL_TYPE="streamvln_qwen2_5_vl"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"  # Base model to extend

# VLN Data Paths - MODIFY THESE TO YOUR DATA LOCATIONS
# Expected format: directory containing annotations.json and video folders
# Supports multiple paths (will be joined with comma)
VLN_DATA_PATHS=(
    "/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/R2R"
    # "/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/RxR_new"
    # "/shared_space/jiangjiajun/data/streamvln_datasets/trajectory_data/EnvDrop"
)
# Join paths with comma separator
VLN_DATA_PATH=$(IFS=','; echo "${VLN_DATA_PATHS[*]}")

# ============================================================================
# Training Parameters
# ============================================================================
# Configuration optimized for A100 40GB with full fine-tuning (ViT+Adapter+LLM)
# Memory calculation:
#   - Model weights (BF16): 3B × 2 bytes ≈ 6GB
#   - Optimizer states (AdamW): ~12GB (params + momentum)
#   - Gradients: ~6GB
#   - Activations (with gradient checkpointing): ~8-12GB (depends on batch size & seq len)
#   - Overhead: ~2-4GB
#   - Total: ~34-40GB (fits A100 40GB)
# 
# With 16 images per sample:
#   - Image tokens: 16 × 256 ≈ 4096 tokens
#   - Text tokens: ~2000-4000 tokens
#   - Total sequence length: ~8000-12000 tokens
TRAIN_TYPE="full"          # Training type: full, lora, etc.
NUM_EPOCHS=1               # Increased for better convergence
LEARNING_RATE=2e-5         # Lower LR for full fine-tuning (more stable)
BATCH_SIZE=1               # Per-device batch size (conservative for 16 images)
GRAD_ACCUM_STEPS=4         # Effective batch size = 1 × 8 = 8
MAX_LENGTH=16384           # Optimized for 16 images + text (~8000-12000 tokens)

# ============================================================================
# Resource Configuration
# ============================================================================
NUM_GPUS=8
CUDA_DEVICES="0,1,2,3,4,5,6,7"
MASTER_PORT=29500

# ============================================================================
# Output Configuration
# ============================================================================
# Build experiment name with key hyperparameters and timestamp
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
EFFECTIVE_BATCH_SIZE=$((BATCH_SIZE * GRAD_ACCUM_STEPS * NUM_GPUS))
EXP_NAME="streamvln-qwen2.5vl-3b-full-${NUM_EPOCHS}epoch-f${NUM_FRAMES}h${NUM_HISTORY}s${NUM_FUTURE_STEPS}-bs${EFFECTIVE_BATCH_SIZE}-lr${LEARNING_RATE}-${TIMESTAMP}"
OUTPUT_DIR="output/${EXP_NAME}"

SAVE_STEPS=500             # Save checkpoint every 500 steps (adjust based on dataset size)
EVAL_STEPS=250             # Evaluate every 250 steps
SAVE_TOTAL_LIMIT=3         # Keep only last 3 checkpoints to save disk space
LOGGING_STEPS=10           # Log every 10 steps

# ============================================================================
# Model Component Freezing Configuration
# ============================================================================
# Full fine-tuning: train all components (ViT + Adapter + LLM)
# Set all to false to enable full fine-tuning
FREEZE_VIT=false           # Train vision encoder (ViT) - requires more memory
FREEZE_LLM=false          # Train language model (LLM) - requires more memory
FREEZE_ALIGNER=false       # Train MLP adapter/projector (aligner)
                            # Note: Full fine-tuning requires ~34-40GB GPU memory
                            # If OOM occurs, consider freezing ViT (set FREEZE_VIT=true)

# ============================================================================
# Optimization Settings
# ============================================================================
# DeepSpeed ZeRO-2 recommended for full fine-tuning to reduce memory usage
# ZeRO-2 partitions optimizer states and gradients across GPUs
# Set USE_DEEPSPEED=true if using multiple GPUs, false for single GPU
USE_DEEPSPEED=true        # Set to true for multi-GPU training (recommended)
DEEPSPEED_CONFIG="zero2"   # zero2 or zero3 (zero2 is more memory efficient)
GRADIENT_CHECKPOINTING=true  # Critical for memory efficiency with 16 images
DDP_TIMEOUT=3600           # Distributed training timeout (seconds)

# Performance Optimizations (from original StreamVLN)
TF32=true                  # TensorFloat-32 for A100 (1.2-1.5x speedup, minimal precision loss)
TORCH_COMPILE=false        # PyTorch 2.0+ compilation (1.2-1.3x speedup, set true if stable)
                           # Note: Set to false initially for debugging, enable later for production
TORCH_COMPILE_BACKEND="inductor"  # Compilation backend (inductor is fastest for A100)

# ============================================================================
# Learning Rate Configuration
# ============================================================================
# Note: ms-swift doesn't support separate vision_tower_lr like original StreamVLN
# If you need different LR for vision, consider using freeze_vit=true or LoRA
WARMUP_RATIO=0.075         # Match original StreamVLN (0.075)
WEIGHT_DECAY=0.            # Match original StreamVLN (0. instead of 0.01)
LR_SCHEDULER_TYPE="cosine_with_min_lr"  # Match original StreamVLN
# Note: JSON must not have spaces for shell compatibility
LR_SCHEDULER_KWARGS='{"min_lr":1.85e-05}'  # Match original StreamVLN min_lr

# ============================================================================
# SwanLab Configuration (Experiment Tracking)
# ============================================================================
# Note: EXP_NAME is defined in Output Configuration section above
USE_SWANLAB=true           # Enable SwanLab logging
SWANLAB_PROJECT="StreamVLN"
SWANLAB_EXP_NAME="${EXP_NAME}"  # Use same name as output directory for consistency
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
export NCCL_DEBUG=ERROR            # INFO/WARN/ERROR (ERROR for production, INFO for debug)
export NCCL_SOCKET_IFNAME=^docker0,lo  # Exclude docker and loopback interfaces
# Performance optimizations for 8-GPU training (from original StreamVLN)
export NCCL_BUFFSIZE=2097152       # 2MB buffer for better throughput
export NCCL_MAX_NCHANNELS=4        # 4 communication channels

# Model Cache Configuration
# ms-swift 默认使用 ModelScope，可通过 MODELSCOPE_CACHE 设置缓存目录
# 如果使用 HuggingFace，设置 USE_HF=1 和 HF_HOME
export MODELSCOPE_CACHE=/shared_space/jiangjiajun/modelscope_cache  # ModelScope 缓存目录
# export USE_HF=1  # 取消注释以使用 HuggingFace
# export HF_HOME=/shared_space/jiangjiajun/hf_cache  # HuggingFace 缓存目录（当 USE_HF=1 时）

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
# Note: Script is in script/ subdirectory, so paths are relative to script/
# Get the absolute path to the streamvln module directory (parent of script/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STREAMVLN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
MS_SWIFT_ROOT="$(cd "$STREAMVLN_DIR/../../.." && pwd)"

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
    --tf32 $TF32
    --torch_compile $TORCH_COMPILE
    $DEEPSPEED_ARG
    $SWANLAB_ARGS
"

# Run training
# Use torchrun for multi-GPU distributed training (DDP)
# Single GPU uses python directly, multi-GPU uses torchrun
# NOTE: DataParallel is NOT compatible with Qwen2.5-VL due to 3D position embeddings
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICES

# Change to ms-swift root directory to run training
cd "$MS_SWIFT_ROOT"

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

