#!/bin/bash
# Qwen2.5-VL-3B-Instruct 全量训练脚本（使用 LLaVA-Instruct-150K 数据集）
# 使用方法: bash train_qwen2.5_vl_3b_llava_full.sh
# 
# 使用前请确保已安装 SwanLab: pip install swanlab
# 如果使用云端模式，需要先登录: swanlab login
# 或在脚本中设置 --swanlab_token <your-token>
#
# 训练配置：
# - 使用 8 张 GPU 进行全量训练
# - 数据集：AI-ModelScope/LLaVA-Instruct-150K (15000 条样本，从 623302 条中采样)
# - 训练 3 个 epoch
# - 全参数训练，学习率 1e-5
# - 3B 模型可以使用更大的 batch size，无需 DeepSpeed
#
# 注意：
# - LLaVA-Instruct-150K 数据集包含图像资源，首次使用时会自动下载
# - 图像资源会下载到 ~/.cache/modelscope/media_resources/ 或 $MODELSCOPE_CACHE/media_resources/
# - 图像资源较大（可能数GB），首次下载需要一些时间

# 设置环境变量，使用所有8张卡
# 注意：如果端口 29500 被占用，可以修改 MASTER_PORT 为其他端口（如 29501）
NPROC_PER_NODE=8 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
MASTER_PORT=29501 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --train_type full \
    --dataset 'AI-ModelScope/LLaVA-Instruct-150K#15000' \
    --load_from_cache_file true \
    --split_dataset_ratio 0.01 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --max_length 4096 \
    --logging_steps 1 \
    --save_steps 100 \
    --eval_steps 100 \
    --save_total_limit 2 \
    --output_dir output/qwen2.5-vl-3b-llava-full \
    --gradient_checkpointing true \
    --freeze_vit true \
    --dataloader_num_workers 4 \
    --warmup_ratio 0.05 \
    --report_to swanlab \
    --swanlab_project qwen2.5-vl-3b-llava-training \
    --swanlab_exp_name qwen2.5-vl-3b-llava-full-$(date +%Y%m%d-%H%M%S) \
    --swanlab_mode cloud

