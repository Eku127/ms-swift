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

# StreamVLN Multi-Node Training - SLURM Submission Script
# 
# Usage: 
#   sbatch examples/vln/streamvln/script/A100/train_streamvln_qwen2_5_vl.slurm
#
# Configuration:
#   - Nodes: 2 (modify --nodes above)
#   - GPUs per node: 8 (modify --gres above)
#   - Total GPUs: 16

PROJECT_ROOT="/shared_space/jiangjiajun/workspace/ms-swift"
cd "$PROJECT_ROOT" || exit 1

# Run training script on each node
srun bash examples/vln/streamvln/script/A100/train_streamvln_qwen2_5_vl_slurm.sh

