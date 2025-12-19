# Qwen2.5-VL-3B 学习指南

本指南将帮助你从零开始学习使用 ms-swift 进行 Qwen2.5-VL-3B 模型的推理和训练。

## 📋 目录
1. [环境准备](#环境准备)
2. [基础推理](#基础推理)
3. [图像理解推理](#图像理解推理)
4. [LoRA 微调训练](#lora-微调训练)
5. [使用训练后的模型](#使用训练后的模型)

---

## 环境准备

✅ **已完成**：
- Conda 环境：`swift-vln` 已创建并激活
- ms-swift 已安装（源码模式）
- Qwen2.5-VL 所需依赖已安装（qwen-vl-utils, decord）

### 验证安装

```bash
# 确保在 swift-vln 环境中
conda activate swift-vln

# 验证 swift 命令可用
swift sft --help
```

---

## 基础推理

### 1. 文本对话推理

首先测试模型的基本文本对话能力：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048
```

**使用说明**：
- 启动后会进入交互式命令行界面
- 直接输入问题，按回车即可
- 输入 `quit` 或 `exit` 退出
- 输入 `clear` 清除历史记录

**示例对话**：
```
<<< 你好，介绍一下你自己
<<< 什么是多模态大模型？
```

### 2. 图像理解推理

Qwen2.5-VL-3B 支持图像理解，可以分析图片内容：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048
```

**使用说明**：
- 在输入问题时，使用 `<image>` 标签表示图像位置
- 例如：`<image>这张图片里有什么？`
- 系统会提示你输入图像路径或 URL
- 支持本地路径和网络 URL

**示例**：
```
<<< <image>这张图片里有什么？
Input an image path or URL <<< http://modelscope-open.oss-cn-hangzhou.aliyuncs.com/images/cat.png
```

**多图像示例**：
```
<<< <image><image>这两张图有什么区别？
Input an image path or URL <<< /path/to/image1.jpg
Input an image path or URL <<< /path/to/image2.jpg
```

### 3. 视频理解推理（可选）

Qwen2.5-VL-3B 还支持视频理解：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
VIDEO_MAX_PIXELS=50176 \
FPS_MAX_FRAMES=12 \
swift infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --stream true \
    --infer_backend pt \
    --max_new_tokens 2048
```

**使用说明**：
- 使用 `<video>` 标签表示视频位置
- 例如：`<video>描述这段视频的内容`

---

## LoRA 微调训练

### 1. 准备数据集

ms-swift 支持多种数据集格式。对于多模态训练，推荐使用 JSONL 格式：

**数据集格式示例** (`train.jsonl`)：
```jsonl
{"messages": [{"role": "user", "content": "这张图片里有什么？"}, {"role": "assistant", "content": "图片中有一只可爱的小猫。"}], "images": ["/path/to/image1.jpg"]}
{"messages": [{"role": "user", "content": "<image>描述这张图片"}, {"role": "assistant", "content": "这是一张风景照，展示了美丽的山川和湖泊。"}], "images": ["/path/to/image2.jpg"]}
```

### 2. LoRA 微调训练

使用 LoRA 进行轻量级微调，显存占用较低：

```bash
# 单卡训练（推荐显存：22GB+）
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset train.jsonl \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output/qwen2.5-vl-3b-lora \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

**参数说明**：
- `--model`: 模型 ID 或路径
- `--dataset`: 数据集路径（支持本地文件或 ModelScope/HuggingFace 数据集 ID）
- `--train_type lora`: 使用 LoRA 训练方式
- `--freeze_vit true`: 冻结视觉编码器（ViT），只训练 LLM 部分，节省显存
- `--lora_rank 8`: LoRA 的秩，控制参数量
- `--lora_alpha 32`: LoRA 的缩放因子
- `--target_modules all-linear`: 对所有线性层应用 LoRA
- `--max_length 2048`: 最大序列长度
- `--output_dir`: 输出目录

### 3. 使用内置数据集（快速测试）

如果想快速测试，可以使用内置数据集：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset 'modelscope/coco_2014_caption:validation#1000' \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 16 \
    --eval_steps 50 \
    --save_steps 50 \
    --save_total_limit 2 \
    --logging_steps 5 \
    --max_length 2048 \
    --output_dir output/qwen2.5-vl-3b-lora \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 4
```

---

## 使用训练后的模型

### 1. 使用 LoRA 权重进行推理

训练完成后，checkpoint 会保存在 `output_dir` 下。使用训练后的模型：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
    --adapters output/qwen2.5-vl-3b-lora/vx-xxx/checkpoint-xxx \
    --stream true \
    --max_new_tokens 2048
```

**注意**：
- `--adapters` 路径需要替换为实际的 checkpoint 文件夹
- 由于 adapters 文件夹中包含 `args.json`，会自动读取模型配置，无需额外指定 `--model`

### 2. 合并 LoRA 权重（可选）

如果需要将 LoRA 权重合并到基础模型中：

```bash
CUDA_VISIBLE_DEVICES=0 \
swift export \
    --adapters output/qwen2.5-vl-3b-lora/vx-xxx/checkpoint-xxx \
    --merge_lora true \
    --output_dir output/qwen2.5-vl-3b-merged
```

合并后可以使用标准方式加载：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
    --model output/qwen2.5-vl-3b-merged \
    --stream true \
    --max_new_tokens 2048
```

---

## 进阶学习

### 1. 全参数微调

如果显存充足，可以进行全参数微调：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset train.jsonl \
    --train_type full \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-5 \
    --gradient_accumulation_steps 16 \
    --max_length 2048 \
    --output_dir output/qwen2.5-vl-3b-full
```

### 2. 多卡训练

如果有多张 GPU，可以使用 DeepSpeed 进行多卡训练：

```bash
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=0,1 \
MAX_PIXELS=1003520 \
swift sft \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset train.jsonl \
    --train_type lora \
    --deepspeed zero2 \
    --torch_dtype bfloat16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --learning_rate 1e-4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps $(expr 16 / 2) \
    --max_length 2048 \
    --output_dir output/qwen2.5-vl-3b-lora
```

### 3. 使用 vLLM 加速推理

对于推理加速，可以使用 vLLM：

```bash
CUDA_VISIBLE_DEVICES=0 \
MAX_PIXELS=1003520 \
swift infer \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --infer_backend vllm \
    --vllm_max_model_len 8192 \
    --stream true \
    --max_new_tokens 2048
```

---

## 常见问题

### 1. 显存不足

如果遇到显存不足，可以：
- 减小 `--per_device_train_batch_size`
- 增大 `--gradient_accumulation_steps`
- 使用 `--freeze_vit true` 冻结视觉编码器
- 减小 `MAX_PIXELS` 的值（降低图像分辨率）

### 2. 模型下载慢

默认使用 ModelScope 下载。如果想使用 HuggingFace，添加 `--use_hf true`：

```bash
swift sft --model Qwen/Qwen2.5-VL-3B-Instruct --use_hf true ...
```

### 3. 数据集格式问题

参考文档：[自定义数据集](https://swift.readthedocs.io/zh-cn/latest/Customization/Custom-dataset.html)

---

## 参考资源

- [ms-swift 官方文档](https://swift.readthedocs.io/zh-cn/latest/)
- [支持的模型和数据集](https://swift.readthedocs.io/zh-cn/latest/Instruction/Supported-models-and-datasets.html)
- [命令行参数说明](https://swift.readthedocs.io/zh-cn/latest/Instruction/Command-line-parameters.html)
- [Qwen2.5-VL 官方仓库](https://github.com/QwenLM/Qwen2.5-VL)

---

## 下一步

完成基础学习后，可以尝试：
1. 使用自己的数据集进行微调
2. 尝试不同的训练任务（DPO、KTO 等）
3. 学习使用 Megatron-SWIFT 进行大规模训练
4. 探索 GRPO 等强化学习算法

祝你学习愉快！🚀

