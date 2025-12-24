# 在 ms-swift 中复现 StreamVLN 架构实施方案

本文档详细说明如何在 ms-swift 框架中基于 Qwen2.5-VL 系列模型复现 StreamVLN 架构。

## 目录

1. [方案概述](#方案概述)
2. [文件结构](#文件结构)
3. [核心文件详解](#核心文件详解)
4. [实施路线图](#实施路线图)
5. [关键技术点](#关键技术点)
6. [与原始实现的差异](#与原始实现的差异)

---

## 方案概述

### 核心思路

**最大化利用 ms-swift 现有的 Qwen2VL 基础设施，只需添加 VLN 特定的数据处理和模型适配层。**

### 主要组件

在 ms-swift 中复现 StreamVLN 需要实现以下核心组件：

1. **自定义数据集类** - 处理 VLN 特定的数据加载和预处理
2. **自定义模型架构** - 扩展 Qwen2VL 以支持 StreamVLN 特性
3. **训练脚本配置** - 配置训练超参数和数据路径
4. **推理脚本** - 实现推理时的 KV cache 管理和历史帧处理

---

## 文件结构

### 完整目录结构

```
ms-swift/
├── swift/
│   ├── llm/
│   │   ├── model/
│   │   │   ├── model/
│   │   │   │   └── streamvln_qwen2_vl.py          # 新建: StreamVLN 模型定义
│   │   │   └── __init__.py                         # 修改: 注册模型
│   │   └── dataset/
│   │       ├── vln_action_dataset.py               # 新建: VLN 数据集
│   │       └── __init__.py                         # 修改: 注册数据集
│   └── trainers/
│       └── vln_trainer.py                          # 新建: VLN 专用 trainer (可选)
├── examples/
│   └── vln/
│       └── streamvln/
│           └── script/
│               ├── train_streamvln_qwen2_vl.sh    # 新建: 训练脚本
│               ├── infer_streamvln_qwen2_vl.sh    # 新建: 推理脚本
│               └── eval_streamvln_qwen2_vl.sh     # 新建: 评估脚本
└── vln/                                           # 现有的 vln 文件夹
    └── doc/                                        # 文档目录
        ├── streamvln_data_format.md
        ├── streamvln_data_and_training.md
        ├── streamvln_training_vs_inference.md
        └── streamvln_implementation_plan.md        # 本文档
```

### 文件说明

| 文件路径 | 类型 | 说明 |
|---------|------|------|
| `swift/llm/dataset/vln_action_dataset.py` | 新建 | VLN 数据集类，处理32帧窗口切分、历史帧采样、多轮对话构建 |
| `swift/llm/model/model/streamvln_qwen2_vl.py` | 新建 | StreamVLN 模型定义，包含图像特征池化、memory token 处理 |
| `swift/llm/model/__init__.py` | 修改 | 注册 StreamVLN 模型类型 |
| `swift/llm/dataset/__init__.py` | 修改 | 注册 VLN 数据集类型 |
| `swift/trainers/vln_trainer.py` | 新建 | VLN 专用训练器（可选，用于特殊需求） |
| `examples/vln/streamvln/script/train_streamvln_qwen2_vl.sh` | 新建 | 训练脚本，配置训练参数 |
| `examples/vln/streamvln/script/infer_streamvln_qwen2_vl.sh` | 新建 | 推理脚本，支持 KV cache |
| `examples/vln/streamvln/script/eval_streamvln_qwen2_vl.sh` | 新建 | 评估脚本 |

---

## 核心文件详解

### 1. 数据集文件

**位置**: `swift/llm/dataset/vln_action_dataset.py`

#### 主要功能

- **VLNActionDataset 类**
  - 处理 32 帧窗口切分
  - 历史帧采样（随机/均匀）
  - 多轮对话构建
  - 动作文本转换
  
- **collate_fn 函数**
  - 批处理数据对齐
  - Padding 处理
  - attention_mask 生成

#### 核心特性

```python
class VLNActionDataset(Dataset):
    """
    VLN 动作预测数据集
    
    关键参数:
    - num_frames: 32 (窗口大小)
    - num_history: 8 (历史帧数量)
    - num_future_steps: 4 (每轮预测的动作数)
    - use_random: 历史帧采样策略 (True=随机, False=均匀)
    """
    
    def __getitem__(self, i):
        """
        返回:
        - input_ids: (seq_len,) token ID 序列
        - labels: (seq_len,) 标签序列
        - images: (num_images, 3, H, W) 图像张量
        - time_ids: (num_frames,) 时间戳索引
        - task_type: int 任务类型标识
        """
        pass
```

#### 数据处理流程

1. **轨迹切分**: 将长轨迹按 32 帧切分成多个训练样本
2. **图像采样**:
   - 历史帧: 从 `[0, start_idx)` 采样 8 帧
   - 当前帧: 从 `[start_idx, start_idx+32)` 按 `num_future_steps=4` 间隔采样
3. **对话构建**: 组织成多轮对话格式
4. **特殊 token**: 插入 `<image>` 和 `<memory>` token

#### 对话模板

```python
# 系统消息
prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. 
         Devise an action sequence to follow the instruction using the four actions: 
         TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, 
         MOVE FORWARD (↑) by 25 centimeters, or STOP."

# 如果有历史帧
if start_idx != 0:
    prompt += " These are your historical observations: <memory>."

# 多轮对话 (每轮 4 个动作)
# User: "you can see <image>"
# Assistant: "↑←→↑"
# ...
```

### 2. 模型文件

**位置**: `swift/llm/model/model/streamvln_qwen2_vl.py`

#### 主要类

**StreamVLNQwen2VLModel**
- 继承 `Qwen2VLModel`
- 添加历史帧处理配置
- 添加空间池化配置

**StreamVLNQwen2VLForCausalLM**
- 继承 `Qwen2VLForCausalLM`
- 实现多模态输入处理
- 实现图像特征压缩

#### 关键方法

```python
class StreamVLNQwen2VLForCausalLM(Qwen2VLForCausalLM):
    
    def get_2dPool(self, image_feature, stride=2):
        """
        2D 空间池化
        - 输入: (num_frames, num_tokens, num_dim)
        - 输出: (num_frames, num_tokens//stride^2, num_dim)
        - 支持: average, max, bilinear
        """
        pass
    
    def encode_images(self, images):
        """
        图像编码
        - 使用 vision_tower 提取特征
        - 使用 mm_projector 投影
        """
        pass
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, labels, images, time_ids
    ):
        """
        准备多模态输入
        
        处理流程:
        1. 编码图像特征
        2. 分离历史帧和当前帧
        3. 压缩历史帧为 memory token
        4. 将 <image> 和 <memory> token 替换为实际特征
        5. 构建完整的 input_embeds
        """
        pass
```

#### 图像特征处理流程

```
原始图像 [B, V, 3, H, W]
    ↓
Vision Tower
    ↓
图像特征 [B, V, num_tokens, C]
    ↓
分离历史帧和当前帧
    ↓
├─ 历史帧 [B, N_history, num_tokens, C]
│     ↓
│  空间池化 (stride=2)
│     ↓
│  Memory Token [B, N_history * num_tokens/4, C]
│
└─ 当前帧 [B, N_current, num_tokens, C]
      ↓
   空间池化 (stride=2)
      ↓
   Image Token [B, N_current * num_tokens/4, C]
```

### 3. 模型注册

**位置**: `swift/llm/model/__init__.py`

#### 需要添加的内容

```python
# 在 ModelType 类中添加
class ModelType:
    # ... 现有模型类型 ...
    
    # StreamVLN 模型
    streamvln_qwen2_vl_3b = 'streamvln-qwen2-vl-3b'
    streamvln_qwen2_vl_7b = 'streamvln-qwen2-vl-7b'

# 在模型注册字典中添加
MODEL_MAPPING = {
    # ... 现有映射 ...
    
    ModelType.streamvln_qwen2_vl_3b: {
        'model_class': StreamVLNQwen2VLForCausalLM,
        'base_model': 'Qwen/Qwen2-VL-3B-Instruct',
        'requires_custom_dataset': True,
    },
    ModelType.streamvln_qwen2_vl_7b: {
        'model_class': StreamVLNQwen2VLForCausalLM,
        'base_model': 'Qwen/Qwen2-VL-7B-Instruct',
        'requires_custom_dataset': True,
    },
}
```

### 4. 数据集注册

**位置**: `swift/llm/dataset/__init__.py`

#### 需要添加的内容

```python
# 在 DatasetName 类中添加
class DatasetName:
    # ... 现有数据集 ...
    
    # VLN 数据集
    vln_action = 'vln-action'

# 在数据集注册字典中添加
DATASET_MAPPING = {
    # ... 现有映射 ...
    
    DatasetName.vln_action: {
        'dataset_class': VLNActionDataset,
        'collate_fn': collate_fn,
        'requires_images': True,
    },
}
```

### 5. 训练脚本

**位置**: `examples/vln/streamvln/script/train_streamvln_qwen2_vl.sh`

#### 配置示例

```bash
#!/bin/bash

# 模型配置
MODEL_TYPE="streamvln-qwen2-vl-3b"
MODEL_PATH="/path/to/Qwen2-VL-3B-Instruct"

# 数据配置
DATASET="vln-action"
DATA_PATH="/path/to/vln/data"

# VLN 特定参数
NUM_FRAMES=32
NUM_HISTORY=8
NUM_FUTURE_STEPS=4
HISTORY_STRIDE=2
CURRENT_STRIDE=2

# 训练参数
BATCH_SIZE=4
NUM_EPOCHS=10
LEARNING_RATE=2e-5
USE_CACHE=false  # 训练时禁用 KV cache

# 执行训练
swift sft \
    --model_type $MODEL_TYPE \
    --model_id_or_path $MODEL_PATH \
    --dataset $DATASET \
    --custom_train_dataset_path $DATA_PATH \
    --num_frames $NUM_FRAMES \
    --num_history $NUM_HISTORY \
    --num_future_steps $NUM_FUTURE_STEPS \
    --history_stride $HISTORY_STRIDE \
    --current_stride $CURRENT_STRIDE \
    --batch_size $BATCH_SIZE \
    --num_train_epochs $NUM_EPOCHS \
    --learning_rate $LEARNING_RATE \
    --use_cache $USE_CACHE \
    --output_dir ./output/streamvln
```

### 6. 推理脚本

**位置**: `examples/vln/infer_streamvln_qwen2_vl.sh`

#### 配置示例

```bash
#!/bin/bash

# 模型配置
MODEL_TYPE="streamvln-qwen2-vl-3b"
CHECKPOINT_PATH="./output/streamvln/checkpoint-best"

# 推理参数
USE_CACHE=true  # 推理时启用 KV cache
RESET_INTERVAL=32  # 每 32 帧重置 KV cache

# 执行推理
swift infer \
    --model_type $MODEL_TYPE \
    --ckpt_dir $CHECKPOINT_PATH \
    --use_cache $USE_CACHE \
    --reset_interval $RESET_INTERVAL
```

### 7. VLN 专用训练器 (可选)

**位置**: `swift/trainers/vln_trainer.py`

#### 功能

如果 ms-swift 的默认 Trainer 无法满足需求，可以实现自定义训练器：

```python
class VLNTrainer(Trainer):
    """
    VLN 专用训练器
    
    主要功能:
    - 自定义 collate_fn
    - 支持 time_ids 参数传递
    - 支持变长图像序列
    - 自定义损失计算（如果需要）
    """
    
    def get_train_dataloader(self):
        """使用自定义 collate_fn"""
        pass
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """自定义损失计算"""
        pass
```

---

## 实施路线图

### Phase 1: 基础架构搭建 (必须)

**目标**: 建立基本的训练框架

1. ✅ **实现数据集类**
   - 文件: `vln_action_dataset.py`
   - 功能: 32帧窗口切分、历史帧采样、多轮对话构建
   - 验证: 能够正确加载和处理 VLN 数据

2. ✅ **实现模型类**
   - 文件: `streamvln_qwen2_vl.py`
   - 功能: 图像特征池化、memory token 处理、多模态输入准备
   - 验证: 能够正确前向传播

3. ✅ **注册组件**
   - 修改: `swift/llm/model/__init__.py`
   - 修改: `swift/llm/dataset/__init__.py`
   - 验证: 能够通过 model_type 和 dataset_name 加载

4. ✅ **创建训练脚本**
   - 文件: `train_streamvln_qwen2_vl.sh`
   - 功能: 配置所有训练参数
   - 验证: 脚本能够正确执行

**预计时间**: 2-3 天

### Phase 2: 训练验证

**目标**: 确保训练流程正确

5. ✅ **验证数据加载**
   - 检查 batch 数据格式
   - 检查 `<image>` 和 `<memory>` token 是否正确插入
   - 检查图像数量和时间戳是否正确

6. ✅ **验证前向传播**
   - 检查 input_embeds 维度
   - 检查 labels 掩码是否正确
   - 检查梯度计算是否正常

7. ✅ **验证损失计算**
   - 确保只有 assistant 回答部分参与损失
   - 检查损失值是否合理
   - 监控训练指标

8. ✅ **完整训练流程**
   - 运行完整 epoch
   - 检查模型收敛情况
   - 保存 checkpoint

**预计时间**: 3-5 天

### Phase 3: 推理支持

**目标**: 实现高效推理

9. ✅ **实现推理脚本**
   - 文件: `infer_streamvln_qwen2_vl.sh`
   - 功能: 基本推理功能
   - 验证: 能够生成动作序列

10. ✅ **实现 KV cache 管理**
    - 推理时启用 `use_cache=True`
    - 保存和复用 past_key_values
    - 拼接历史输出

11. ✅ **实现 32 帧重置逻辑**
    - 每 32 帧重置 KV cache
    - 重新采样和压缩历史帧
    - 更新 memory token

12. ✅ **实现评估脚本**
    - 文件: `eval_streamvln_qwen2_vl.sh`
    - 功能: 在 VLN 任务上评估模型
    - 指标: SR (Success Rate), SPL, etc.

**预计时间**: 3-5 天

### 总预计时间: 8-13 天

---

## 关键技术点

### 1. 特殊 Token 处理

#### 需要添加的 Token

```python
# 在 tokenizer 中添加特殊 token
tokenizer.add_tokens(["<image>"], special_tokens=True)
tokenizer.add_tokens(["<memory>"], special_tokens=True)

# 定义 token index
IMAGE_TOKEN_INDEX = -200    # ms-swift 约定
MEMORY_TOKEN_INDEX = -201   # 自定义
```

#### Token 替换流程

```python
# 1. 在对话中插入特殊 token
conversation = "You are an assistant. <memory> You can see <image>"

# 2. Tokenization
input_ids = tokenizer(conversation)  # [..., <memory_token_id>, ..., <image_token_id>, ...]

# 3. 替换为特殊 index
input_ids[input_ids == memory_token_id] = MEMORY_TOKEN_INDEX
input_ids[input_ids == image_token_id] = IMAGE_TOKEN_INDEX

# 4. 在模型中替换为实际特征
# 找到 IMAGE_TOKEN_INDEX 的位置
image_positions = torch.where(input_ids == IMAGE_TOKEN_INDEX)[0]
# 替换为图像特征
input_embeds[image_positions] = image_features
```

### 2. 历史帧压缩

#### 压缩流程

```python
# 1. 提取历史帧特征
history_features = image_features[:num_history]  # [N, 729, C]

# 2. Reshape 为 2D 网格
height = width = 27  # sqrt(729)
history_features = history_features.view(N, height, width, C)
history_features = history_features.permute(0, 3, 1, 2)  # [N, C, H, W]

# 3. 2D 空间池化
pooled_features = F.avg_pool2d(history_features, kernel_size=2, stride=2)
# [N, C, H/2, W/2] = [N, C, 13, 13]

# 4. Flatten 为 token 序列
pooled_features = pooled_features.permute(0, 2, 3, 1)  # [N, 13, 13, C]
memory_tokens = pooled_features.flatten(0, 1)  # [N*169, C]
```

#### 为什么要压缩？

- **减少 token 数量**: 729 → 169 (减少 77%)
- **控制序列长度**: 避免超过模型最大长度
- **提高效率**: 减少计算量和内存占用
- **保留关键信息**: 空间池化保留全局结构

### 3. 多轮对话构建

#### 构建逻辑

```python
def prepare_conversation(instructions, actions, num_future_steps=4):
    """
    将动作序列组织成多轮对话
    
    输入:
    - instructions: "Go to the kitchen"
    - actions: [1, 1, 2, 1, ...]  # 32 个动作
    - num_future_steps: 4
    
    输出:
    - 8 轮对话 (32 / 4 = 8)
    """
    
    sources = []
    
    # 系统消息 + 用户指令
    system_prompt = f"Your task is to {instructions}. ..."
    if has_history:
        system_prompt += " These are your historical observations: <memory>."
    
    sources.append({"from": "human", "value": system_prompt})
    sources.append({"from": "gpt", "value": ""})  # 占位
    
    # 多轮对话
    for i in range(0, len(actions), num_future_steps):
        # User: 显示新图像
        sources.append({
            "from": "human", 
            "value": "you can see <image>"
        })
        
        # Assistant: 预测动作序列
        step_actions = actions[i:i+num_future_steps]
        action_text = actions2text(step_actions)  # "↑↑←→"
        sources.append({
            "from": "gpt", 
            "value": action_text
        })
    
    return sources
```

#### 对话结构示例

```
<|im_start|>system
You are an autonomous navigation assistant. Your task is to go to the kitchen. 
These are your historical observations: <memory>.
<|im_end|>
<|im_start|>user
you can see <image>
<|im_end|>
<|im_start|>assistant
↑↑←→
<|im_end|>
<|im_start|>user
in front of you is <image>
<|im_end|>
<|im_start|>assistant
↑←→↑
<|im_end|>
...
```

### 4. 训练时的数据流

```
原始数据
    ↓
轨迹切分 (每 32 帧一个 sample)
    ↓
图像采样 (历史帧 8 张 + 当前帧 8 张)
    ↓
对话构建 (多轮 user-assistant)
    ↓
Tokenization (input_ids + labels)
    ↓
Batch 组织 (collate_fn, padding)
    ↓
模型输入准备
    ├─ 图像编码
    ├─ 历史帧压缩为 memory token
    ├─ 当前帧压缩为 image token
    └─ 构建 input_embeds
    ↓
前向传播
    ↓
损失计算 (只对 assistant 部分)
    ↓
反向传播
```

### 5. 推理时的处理流程

```
环境初始化
    ↓
第 1 轮生成 (t=0-31)
    ├─ 输入: 指令 + 当前图像
    ├─ 输出: 动作序列 "↑←→↑"
    ├─ 保存: past_key_values, output_ids
    └─ 执行: 动作
    ↓
第 2 轮生成 (t=1-32)
    ├─ 输入: output_ids + 新图像
    ├─ 使用: past_key_values (复用 KV cache)
    ├─ 输出: 新动作序列
    └─ 更新: past_key_values, output_ids
    ↓
... (持续到 t=32)
    ↓
第 32 帧边界
    ├─ 重置: past_key_values = None, output_ids = None
    ├─ 采样: 历史帧 (t=0-31)
    ├─ 压缩: 历史帧 → memory token
    └─ 重新开始
    ↓
第 3 轮生成 (t=32-63)
    ├─ 输入: 指令 + memory token + 当前图像
    └─ ...
```

### 6. Padding 处理

#### 序列 Padding

```python
# input_ids 和 labels padding
input_ids_batch = pad_sequence(
    input_ids_batch, 
    batch_first=True, 
    padding_value=tokenizer.pad_token_id
)

labels_batch = pad_sequence(
    labels_batch, 
    batch_first=True, 
    padding_value=IGNORE_INDEX
)

# attention_mask 生成
attention_mask = input_ids_batch.ne(tokenizer.pad_token_id)
```

#### 图像 Padding

```python
def pad_tensors(tensors, max_len=None, pad=0):
    """
    Padding 图像到统一长度
    
    输入: [(N1, C, H, W), (N2, C, H, W), ...]
    输出: (B, max_N, C, H, W)
    """
    lens = [t.size(0) for t in tensors]
    max_len = max(lens)
    bs = len(tensors)
    
    output = torch.zeros(bs, max_len, *tensors[0].shape[1:])
    for i, (t, l) in enumerate(zip(tensors, lens)):
        output[i, :l] = t
    
    return output
```

### 7. 与 Qwen2VL 的集成

#### 利用现有组件

```python
# 1. 使用 Qwen2VL 的 vision tower
vision_tower = self.model.get_vision_tower()
image_features = vision_tower(images)  # 自动处理图像编码

# 2. 使用 Qwen2VL 的 mm_projector
projected_features = self.model.mm_projector(image_features)

# 3. 使用 Qwen2VL 的 conversation template
from transformers import Qwen2VLProcessor
processor = Qwen2VLProcessor.from_pretrained(model_path)
```

#### 需要扩展的部分

```python
# 1. 添加历史帧处理
self.num_history = config.num_history
self.history_stride = config.history_stride

# 2. 添加空间池化
def get_2dPool(self, image_feature, stride):
    # 自定义实现
    pass

# 3. 添加 memory token 处理
def prepare_inputs_labels_for_multimodal(...):
    # 扩展以支持 MEMORY_TOKEN_INDEX
    pass
```

---

## 与原始实现的差异

### 主要差异对比

| 方面 | 原始 StreamVLN | ms-swift 复现 |
|------|---------------|--------------|
| **基础模型** | 自定义 Qwen2 + LLaVA | Qwen2VL (ms-swift) |
| **视觉编码器** | SigLIP | Qwen2VL 自带 vision tower |
| **模型架构** | 手动组合 LLaVA components | 继承 Qwen2VL 类 |
| **数据加载** | 自定义 DataLoader | ms-swift Dataset + collate_fn |
| **训练器** | 自定义 Trainer | ms-swift Trainer |
| **特殊 token** | 手动实现和管理 | 框架注册机制 |
| **推理** | 自定义生成逻辑 | 框架 generate() 方法 |
| **配置管理** | 命令行参数 | ms-swift 配置系统 |
| **Checkpoint** | 手动保存加载 | 框架自动管理 |

### 保留的核心逻辑

✅ **完全保留**:
- 32 帧窗口切分
- 历史帧采样策略（随机/均匀）
- Memory token 压缩机制
- 2D 空间池化 (stride=2)
- 多轮对话构建
- 动作序列编码（↑←→）
- 训练-推理的一致性设计

### 简化建议

#### 初期可以简化的部分

1. **去掉深度信息**
   ```python
   # 原始: 支持 RGB + Depth + Pose + Intrinsics
   # 简化: 只用 RGB
   def encode_images(self, images):
       # 不需要 depths, poses, intrinsics 参数
       pass
   ```

2. **使用 Qwen2VL 的空间合并**
   ```python
   # 原始: 自定义 2D pooling
   # 可选: 使用 Qwen2VL 的 spatial_merge 配置
   config.vision_config.spatial_merge_size = 2
   ```

3. **简化历史帧采样**
   ```python
   # 初期只实现均匀采样，随机采样后续添加
   if time_ids[0] != 0:
       step = max(time_ids[0] // num_history, 1)
       history_step_ids = np.arange(0, time_ids[0], step)
   ```

#### 后期可以优化的部分

1. **动态窗口大小**
   - 当前: 固定 32 帧
   - 优化: 根据任务长度动态调整

2. **自适应历史帧数量**
   - 当前: 固定 8 帧
   - 优化: 根据可用历史动态调整

3. **多任务支持**
   - 添加 task_type 区分不同 VLN 任务
   - 支持 R2R, REVERIE, SOON 等多个数据集

### 优势与劣势

#### ms-swift 复现的优势

✅ **开发效率**
- 利用现有基础设施，减少重复开发
- 自动处理训练流程、checkpoint、日志等

✅ **维护性**
- 遵循 ms-swift 框架规范
- 容易集成新功能和更新

✅ **扩展性**
- 容易切换不同的 VLM 基座
- 容易添加新的数据集和任务

✅ **社区支持**
- ms-swift 的文档和社区资源
- 更容易获得帮助和反馈

#### 可能的劣势

⚠️ **灵活性**
- 需要适配框架约定，某些定制可能受限
- 可能需要在框架限制内找变通方案

⚠️ **依赖**
- 依赖 ms-swift 的版本更新
- 框架变化可能需要适配

⚠️ **调试难度**
- 多层抽象可能增加调试复杂度
- 需要理解框架内部机制

---

## 实施建议

### 开发顺序

1. **先实现基础功能** (Phase 1)
   - 数据集加载
   - 模型前向传播
   - 基本训练流程

2. **再验证正确性** (Phase 2)
   - 对比原始实现的输出
   - 检查数据格式
   - 验证损失计算

3. **最后优化性能** (Phase 3)
   - 推理加速
   - KV cache 优化
   - 内存优化

### 测试策略

1. **单元测试**
   ```python
   # 测试数据集
   def test_dataset():
       dataset = VLNActionDataset(...)
       sample = dataset[0]
       assert sample[0].shape  # input_ids
       assert sample[2].shape  # images
   
   # 测试模型
   def test_model():
       model = StreamVLNQwen2VLForCausalLM(...)
       output = model(**inputs)
       assert output.loss is not None
   ```

2. **集成测试**
   ```bash
   # 测试完整训练流程
   bash examples/vln/streamvln/script/train_streamvln_qwen2_vl.sh
   
   # 测试推理
   bash examples/vln/streamvln/script/infer_streamvln_qwen2_vl.sh
   ```

3. **对比测试**
   - 与原始 StreamVLN 实现对比输出
   - 检查损失曲线是否相似
   - 验证最终性能指标

### 调试技巧

1. **检查数据**
   ```python
   # 打印 batch 信息
   for batch in dataloader:
       print(f"input_ids: {batch['input_ids'].shape}")
       print(f"images: {batch['images'].shape}")
       print(f"time_ids: {batch['time_ids']}")
       break
   ```

2. **检查特征**
   ```python
   # 在模型中添加调试输出
   def prepare_inputs_labels_for_multimodal(...):
       print(f"Image features: {image_features.shape}")
       print(f"Memory features: {memory_features.shape}")
       print(f"Input embeds: {input_embeds.shape}")
   ```

3. **可视化**
   ```python
   # 可视化图像和动作序列
   import matplotlib.pyplot as plt
   
   for i, img in enumerate(images):
       plt.subplot(2, 4, i+1)
       plt.imshow(img.permute(1, 2, 0))
   plt.show()
   ```

---

## 常见问题

### Q1: 为什么选择 Qwen2VL 而不是其他 VLM？

**A**: 
- Qwen2VL 性能优秀，支持高分辨率图像
- ms-swift 对 Qwen2VL 有完善支持
- 与原始 StreamVLN 使用 Qwen2 一脉相承
- 容易迁移到其他 VLM（如果需要）

### Q2: 能否使用更小的模型（如 1B）？

**A**: 可以，只需：
1. 在模型注册中添加 `streamvln_qwen2_vl_1b`
2. 指向 1B 的 base model
3. 其他代码无需修改

### Q3: 如何处理内存不足？

**A**:
- 减小 batch_size
- 增大 num_future_steps（减少图像数量）
- 使用 gradient_checkpointing
- 使用更小的 base model

### Q4: 训练速度太慢怎么办？

**A**:
- 使用混合精度训练（fp16/bf16）
- 启用 flash attention
- 增大 num_future_steps
- 使用更大的 batch_size（如果内存允许）

### Q5: 推理时如何处理超长轨迹？

**A**:
- 实现 32 帧循环重置机制
- 使用 memory token 保留历史信息
- 监控 KV cache 大小，避免 OOM

---

## 参考资料

### 相关文档

- [StreamVLN 数据格式](./streamvln_data_format.md)
- [StreamVLN 数据与训练](./streamvln_data_and_training.md)
- [StreamVLN 训练与推理对比](./streamvln_training_vs_inference.md)

### 原始实现

- StreamVLN 数据集: `StreamVLN/streamvln/dataset/vln_action_dataset.py`
- StreamVLN 模型: `StreamVLN/streamvln/model/stream_video_vln.py`
- StreamVLN 训练: `StreamVLN/streamvln/streamvln_train.py`
- StreamVLN 评估: `StreamVLN/streamvln/streamvln_eval.py`

### ms-swift 资源

- [ms-swift 官方文档](https://github.com/modelscope/swift)
- [Qwen2VL 模型文档](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
- [ms-swift 自定义数据集教程](https://github.com/modelscope/swift/blob/main/docs/source/Multi-Modal/qwen2-vl最佳实践.md)

---

## 更新日志

- **2025-12-22**: 初始版本，完整实施方案
