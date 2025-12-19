#!/bin/bash
# Qwen2.5-VL-3B-Instruct 全量训练脚本（带 SwanLab 监控）
# 使用方法: bash train_qwen2.5_vl_3b_full.sh
# 
# 使用前请确保已安装 SwanLab: pip install swanlab
# 如果使用云端模式，需要先登录: swanlab login
# 或在脚本中设置 --swanlab_token <your-token>
#
# 训练配置：
# - 使用 8 张 GPU 进行全量训练
# - 数据集混合：
#   * Caption3o-LongCap-v4 (15000 条样本，多模态图像描述)
#   * alpaca-gpt4-data-zh (10000 条样本，中文对话)
# - 训练 3 个 epoch
# - 全参数训练，学习率 1e-5
# - 使用 DeepSpeed ZeRO-2 优化显存

# 设置环境变量，使用所有8张卡
# 设置 PyTorch CUDA 内存分配策略，减少碎片化
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL 环境变量设置（解决通信超时问题）
# 注意：如果遇到 NCCL 超时错误，可以尝试以下设置：
# 1. 增加超时时间（默认600秒可能不够，特别是大数据传输时）
export NCCL_TIMEOUT=1800  # 30分钟超时
# 2. InfiniBand 设置（如果使用 IB 网络）
export NCCL_IB_DISABLE=0  # 启用 InfiniBand（如果可用，设为1禁用）
export NCCL_IB_GID_INDEX=3  # InfiniBand GID 索引（根据实际网络配置调整）
# 3. 网络接口设置
export NCCL_SOCKET_IFNAME=^docker0,lo  # 排除 docker 和 loopback 接口
# 4. 调试信息（生产环境建议设为 WARN 或 ERROR）
export NCCL_DEBUG=WARN  # INFO/WARN/ERROR，INFO 会输出大量日志
# 5. 其他优化设置
export NCCL_P2P_DISABLE=0  # P2P 通信（如果遇到 P2P 问题可设为1）
export NCCL_SHM_DISABLE=0  # 共享内存（如果共享内存不足可设为1）
# 注意：如果端口 29500 被占用，可以修改 MASTER_PORT 为其他端口（如 29501）
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29500 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --dataset 'prithivMLmods/Caption3o-LongCap-v4#15000' 'AI-ModelScope/alpaca-gpt4-data-zh#10000' \
    --custom_register_path examples/my_scripts/custom_dataset_caption3o.py \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 4096 \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_steps 50 \
    --save_total_limit 1 \
    --output_dir output/qwen2.5-vl-3b-mixed-lang \
    --deepspeed zero2 \
    --ddp_timeout 3600 \
    --gradient_checkpointing true \
    --freeze_vit true \
    --dataloader_num_workers 4 \
    --warmup_ratio 0.05 \
    --report_to swanlab \
    --swanlab_project qwen2.5-vl-3b-full-training \
    --swanlab_exp_name qwen2.5-vl-3b-full-$(date +%Y%m%d-%H%M%S) \
    --swanlab_mode cloud

