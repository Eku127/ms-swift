#!/bin/bash
# Download Qwen2.5-VL Models from ModelScope
# 
# Usage:
#   bash examples/vln/streamvln/script/download_qwen2_5_vl.sh [3b|7b|all]
#
# Examples:
#   Download 3B model:  bash download_qwen2_5_vl.sh 3b
#   Download 7B model:  bash download_qwen2_5_vl.sh 7b
#   Download both:      bash download_qwen2_5_vl.sh all
#   Download both:      bash download_qwen2_5_vl.sh  # default is 'all'
#
# The models will be downloaded to the default ModelScope cache directory:
#   ~/.cache/modelscope/hub/qwen/Qwen2.5-VL-{3B|7B}-Instruct
#
# You can also set a custom cache directory via MODEL_CACHE_DIR environment variable:
#   MODEL_CACHE_DIR=/path/to/models bash download_qwen2_5_vl.sh

set -e  # Exit on error

# ============================================================================
# Configuration
# ============================================================================

# Model IDs on ModelScope
MODEL_3B="qwen/Qwen2.5-VL-3B-Instruct"
MODEL_7B="qwen/Qwen2.5-VL-7B-Instruct"

# Default cache directory (ModelScope default)
# Can be overridden by MODEL_CACHE_DIR environment variable
DEFAULT_CACHE_DIR="${HOME}/.cache/modelscope/hub"

# Parse command line argument
DOWNLOAD_TARGET="${1:-all}"

# Convert to lowercase
DOWNLOAD_TARGET=$(echo "$DOWNLOAD_TARGET" | tr '[:upper:]' '[:lower:]')

# ============================================================================
# Helper Functions
# ============================================================================

print_info() {
    echo "[INFO] $1"
}

print_success() {
    echo "[SUCCESS] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

check_modelscope() {
    if ! python -c "import modelscope" 2>/dev/null; then
        print_error "ModelScope is not installed!"
        echo "Please install it with: pip install modelscope"
        exit 1
    fi
}

download_model() {
    local model_id=$1
    local model_name=$2
    
    print_info "Starting download: ${model_name} (${model_id})"
    
    # Use Python to download via ModelScope
    python << EOF
import os
from modelscope import snapshot_download

model_id = "${model_id}"
cache_dir = os.environ.get('MODEL_CACHE_DIR', '${DEFAULT_CACHE_DIR}')

print(f"Downloading model: {model_id}")
print(f"Cache directory: {cache_dir}")

try:
    model_dir = snapshot_download(
        model_id,
        cache_dir=cache_dir,
        revision='master'
    )
    print(f"\n✓ Model downloaded successfully!")
    print(f"  Model directory: {model_dir}")
except Exception as e:
    print(f"\n✗ Download failed: {e}")
    exit(1)
EOF

    if [ $? -eq 0 ]; then
        print_success "${model_name} downloaded successfully!"
    else
        print_error "Failed to download ${model_name}"
        return 1
    fi
}

# ============================================================================
# Main Script
# ============================================================================

print_info "Qwen2.5-VL Model Downloader from ModelScope"
print_info "=============================================="
echo ""

# Check if ModelScope is installed
check_modelscope

# Display cache directory
CACHE_DIR="${MODEL_CACHE_DIR:-${DEFAULT_CACHE_DIR}}"
print_info "Cache directory: ${CACHE_DIR}"
echo ""

# Download based on target
case "${DOWNLOAD_TARGET}" in
    3b)
        print_info "Downloading 3B model only..."
        download_model "${MODEL_3B}" "Qwen2.5-VL-3B-Instruct"
        ;;
    7b)
        print_info "Downloading 7B model only..."
        download_model "${MODEL_7B}" "Qwen2.5-VL-7B-Instruct"
        ;;
    all|*)
        print_info "Downloading both 3B and 7B models..."
        echo ""
        
        print_info "=== Downloading 3B Model ==="
        download_model "${MODEL_3B}" "Qwen2.5-VL-3B-Instruct"
        echo ""
        
        print_info "=== Downloading 7B Model ==="
        download_model "${MODEL_7B}" "Qwen2.5-VL-7B-Instruct"
        ;;
esac

echo ""
print_success "All downloads completed!"
print_info "Models are cached at: ${CACHE_DIR}"
print_info ""
print_info "To use the models in training, set MODEL_PATH to:"
if [ "${DOWNLOAD_TARGET}" = "3b" ] || [ "${DOWNLOAD_TARGET}" = "all" ]; then
    echo "  MODEL_PATH=\"${MODEL_3B}\"  # for 3B model"
fi
if [ "${DOWNLOAD_TARGET}" = "7b" ] || [ "${DOWNLOAD_TARGET}" = "all" ]; then
    echo "  MODEL_PATH=\"${MODEL_7B}\"  # for 7B model"
fi
print_info ""
print_info "Or use the local cache path directly:"
if [ "${DOWNLOAD_TARGET}" = "3b" ] || [ "${DOWNLOAD_TARGET}" = "all" ]; then
    echo "  MODEL_PATH=\"${CACHE_DIR}/qwen/Qwen2.5-VL-3B-Instruct\"  # for 3B model"
fi
if [ "${DOWNLOAD_TARGET}" = "7b" ] || [ "${DOWNLOAD_TARGET}" = "all" ]; then
    echo "  MODEL_PATH=\"${CACHE_DIR}/qwen/Qwen2.5-VL-7B-Instruct\"  # for 7B model"
fi

