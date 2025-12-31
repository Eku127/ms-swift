#!/bin/bash
# StreamVLN Distributed Evaluation Script - Qwen2.5-VL (ms-swift)
# 
# Usage:
#   MODEL_PATH=/path/to/checkpoint bash examples/vln/streamvln/script/H100/eval/eval_streamvln_qwen2_5_vl_distributed.sh
#
# This script runs distributed VLN evaluation using torchrun.
# Each process handles a different subset of episodes (based on rank).

set -e  # Exit on error

# ============================================================================
# Conda Environment
# ============================================================================
source /mnt/data1/home/jiangjiajun/miniconda3/etc/profile.d/conda.sh
conda activate swift-vln-eval

# ============================================================================
# GPU Configuration
# ============================================================================
CUDA_DEVICES="${CUDA_DEVICES:-0,1,2,3,4,5,6,7}"
NUM_GPUS=$(echo "$CUDA_DEVICES" | tr ',' '\n' | wc -l)
MASTER_PORT="${MASTER_PORT:-29600}"

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_PATH="${MODEL_PATH:-/mnt/data1/home/jiangjiajun/workspace/ms-swift/output/streamvln-qwen2.5vl-3b-full-1ep-f32h8s4-bs64-lr4e-5-20251229-094943/v0-20251229-095006/checkpoint-7483}"

# ============================================================================
# Habitat Configuration
# ============================================================================
HABITAT_CONFIG_PATH="config/vln_r2r.yaml"
EVAL_SPLIT="${EVAL_SPLIT:-val_unseen}"

# ============================================================================
# VLN Parameters (should match training)
# ============================================================================
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4

# ============================================================================
# Output Configuration
# ============================================================================
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
# Extract model name from MODEL_PATH (the directory name after 'output/')
MODEL_NAME=$(echo "$MODEL_PATH" | sed -n 's|.*/output/\([^/]*\)/.*|\1|p')
# Fallback if parsing fails
MODEL_NAME="${MODEL_NAME:-unknown_model}"
OUTPUT_DIR="${OUTPUT_DIR:-./results/eval/${MODEL_NAME}/${EVAL_SPLIT}_${TIMESTAMP}}"

# ============================================================================
# Video Options
# ============================================================================
SAVE_VIDEO="${SAVE_VIDEO:-true}"
VIDEO_COMPRESSION="${VIDEO_COMPRESSION:-true}"

# ============================================================================
# Environment Setup
# ============================================================================
export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"
export NCCL_DEBUG=ERROR
export NCCL_TIMEOUT=7200
export NCCL_SOCKET_IFNAME=^docker0,lo
export NCCL_BUFFSIZE=2097152
export NCCL_MAX_NCHANNELS=4
# ModelScope cache
export MODELSCOPE_CACHE=/mnt/data1/home/jiangjiajun/.cache/modelscope

# ============================================================================
# Paths
# ============================================================================
# Calculate ms-swift root directory
# Script is at: examples/vln/streamvln/script/H100/eval/eval_streamvln_qwen2_5_vl_distributed.sh
# Need to go up 7 levels to reach ms-swift root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSWIFT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"
STREAMVLN_DIR="${MSSWIFT_ROOT}/examples/vln/streamvln"

# ============================================================================
# Print Configuration
# ============================================================================
echo "=============================================="
echo "StreamVLN Distributed Evaluation"
echo "=============================================="
echo "Model Path:      ${MODEL_PATH}"
echo "Eval Split:      ${EVAL_SPLIT}"
echo "Output Dir:      ${OUTPUT_DIR}"
echo "Num GPUs:        ${NUM_GPUS}"
echo "CUDA Devices:    ${CUDA_DEVICES}"
echo "Master Port:     ${MASTER_PORT}"
echo "Save Video:      ${SAVE_VIDEO}"
echo "Video Compression: ${VIDEO_COMPRESSION}"
echo "=============================================="

# ============================================================================
# Pre-check
# ============================================================================
if [ "$MODEL_PATH" == "/path/to/your/trained/checkpoint" ]; then
    echo "[ERROR] Please set MODEL_PATH!"
    echo "Usage: MODEL_PATH=/path/to/checkpoint EVAL_SPLIT=val_unseen bash $0"
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "[ERROR] Model path does not exist: $MODEL_PATH"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# ============================================================================
# Run Distributed Evaluation
# ============================================================================
cd "$MSSWIFT_ROOT"

echo "[INFO] Starting distributed evaluation on ${NUM_GPUS} GPUs..."

# Build video arguments
VIDEO_ARGS=""
[ "$SAVE_VIDEO" = "true" ] && VIDEO_ARGS="--save_video"
[ "$VIDEO_COMPRESSION" = "true" ] && VIDEO_ARGS="${VIDEO_ARGS} --video_compression"

torchrun \
    --nproc_per_node="${NUM_GPUS}" \
    --master_port="${MASTER_PORT}" \
    -m examples.vln.streamvln.eval \
    --model_path "${MODEL_PATH}" \
    --habitat_config_path "${STREAMVLN_DIR}/${HABITAT_CONFIG_PATH}" \
    --eval_split "${EVAL_SPLIT}" \
    --num_frames "${NUM_FRAMES}" \
    --num_history "${NUM_HISTORY}" \
    --num_future_steps "${NUM_FUTURE_STEPS}" \
    --output_dir "${OUTPUT_DIR}" \
    --distributed \
    ${VIDEO_ARGS}

echo "=============================================="
echo "Evaluation Complete!"
echo "Results saved to: ${OUTPUT_DIR}"
echo "=============================================="

