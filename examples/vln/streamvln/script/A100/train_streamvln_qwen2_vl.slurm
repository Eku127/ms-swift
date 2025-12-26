#!/bin/bash
#SBATCH --job-name=streamvln-msswift
#SBATCH --nodes=2
#SBATCH --gres=gpu:8 
#SBATCH --time=96:00:00
#SBATCH --cpus-per-task=128
#SBATCH --mem-per-cpu=8G
#SBATCH --reservation=laser
#SBATCH --output=/shared_space/jiangjiajun/workspace/ms-swift/logs/streamvln-msswift-%j.out
#SBATCH --error=/shared_space/jiangjiajun/workspace/ms-swift/logs/streamvln-msswift-%j.err

# StreamVLN Training SLURM Script - ms-swift version
# 
# Usage: 
#   sbatch examples/vln/streamvln/script/A100/train_streamvln_qwen2_vl.slurm
#
# This script submits a multi-node training job using SLURM.
# The actual training script will detect SLURM environment and configure accordingly.

# Set project root directory (absolute path)
PROJECT_ROOT="/shared_space/jiangjiajun/workspace/ms-swift"

# Change to project root directory
cd "$PROJECT_ROOT" || exit 1

# Run training script on each node
srun bash examples/vln/streamvln/script/A100/train_streamvln_qwen2_vl.sh

