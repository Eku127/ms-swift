# Qwen2.5-VL 图像 Tokenization 流程详解

本文档详细说明在 ms-swift 框架中使用 Qwen2.5-VL 模型时，图像从原始 PIL.Image 到最终 image embeddings 的完整处理流程。

## 目录

1. [概述](#概述)
2. [阶段 1: template.encode() - 数据预处理](#阶段-1-templateencode---数据预处理)
3. [阶段 2: 模型 forward() - 训练/推理](#阶段-2-模型-forward---训练推理)
4. [完整流程图](#完整流程图)
5. [关键点总结](#关键点总结)
6. [StreamVLN 与 Qwen2.5-VL 对比](#streamvln-与-qwen25-vl-对比)

---

## 概述

Qwen2.5-VL 的图像处理分为两个主要阶段：

1. **template.encode() 阶段**：将标准对话格式转换为模型输入格式
   - 入口函数：`swift/llm/template/template/qwen.py` → `Qwen2VLTemplate._encode()` (L343)
   - 语言：文本 → token (input_ids)
     - 处理位置：`swift/llm/template/base.py` → `Template._encode()` (通过 super() 调用)
   - 图像：PIL.Image → pixel_values（预处理后的图像数据）+ 占位符（image_token_id）
     - 处理位置：`swift/llm/template/template/qwen.py` → `Qwen2VLTemplate._encode()` (L351-380)

2. **模型 forward() 阶段**：将 token 转换为 embeddings（通过 forward pre-hook 触发）
   - 入口函数：`swift/llm/template/template/qwen.py` → `Qwen2VLTemplate._post_encode()` (L399-409)
   - 触发机制：`swift/llm/template/base.py` → `Template.pre_forward_hook()` (L1385-1401)
   - 语言：token → embedding
     - 处理位置：`swift/llm/template/template/qwen.py` → `_post_encode()` (L404-407)
     - 调用 `embed_tokens(input_ids)` 将所有 token（包括图像占位符）转换为 embedding
   - 图像：pixel_values → 视觉编码器 → patch embeddings → spatial merge → image embeddings
     - 处理位置：`swift/llm/template/base.py` → `_get_inputs_embeds_hf()` (L1994-2043)
     - 视觉编码 + Merge：`visual(pixel_values_mixed, grid_thw=grid_thw)` (L2019)
     - 替换占位符：`inputs_embeds.masked_scatter(image_mask, image_embeds)` (L2036)

---

## 阶段 1: template.encode() - 数据预处理

### 输入格式

```python
{
    'messages': [
        {'role': 'user', 'content': 'Look at <image>.'}
    ],
    'images': [PIL.Image(448×448)]  # 原始 PIL 图像
}
```

### 处理流程

#### 步骤 1: 语言 Tokenization

```python
# 文本 tokenize
文本: "Look at <image>."
  ↓ tokenize
input_ids = [..., 151655, ...]  # <image> 被替换为 image_token_id (151655)
```

#### 步骤 2: 图像预处理

```python
# 调用 processor.image_processor()
media_inputs = processor.image_processor(
    images=[PIL.Image(448×448)],
    return_tensors='pt',
    do_resize=False
)

# 返回结果
{
    'pixel_values': tensor([1024, 3, 14, 14]),  # 1024 个 patches，每个 14×14×3
    'image_grid_thw': tensor([[1, 32, 32]])     # 32×32 网格 = 1024 patches
}
```

**处理过程**：
- Resize 图像（根据 MAX_PIXELS 限制）
- 分割成 14×14 的 patches
- 归一化（ImageNet 均值和标准差）
- 输出：`pixel_values` 形状为 `[num_patches, 3, 14, 14]`

#### 步骤 3: 计算占位符数量

```python
# 关键：提前计算 spatial merge 后的 token 数量
merge_size = processor.image_processor.merge_size  # = 2
merge_length = merge_size**2  # = 4

# 计算占位符数量
media_grid_thw = [1, 32, 32]  # 原始网格：32×32 = 1024 patches
token_len = (1 × 32 × 32) // 4 = 1024 // 4 = 256  # merge 后的 token 数量
```

**为什么提前计算？**
- 占位符数量必须与实际 image embeddings 数量完全匹配
- 如果插入 1024 个占位符，但模型只产生 256 个 embeddings，会导致位置不匹配
- 所以必须在数据预处理阶段就确定最终的 token 数量

#### 步骤 4: 插入占位符

```python
# 在 input_ids 中找到 <image> token 的位置
idx_list = findall(input_ids, image_token_id)  # 找到所有 image_token_id 的位置

# 用 256 个占位符替换单个 <image> token
def _get_new_tokens(i):
    token_len = (media_grid_thw[i].prod() // merge_length)  # = 256
    return [image_token_id] * token_len  # 返回 256 个 image_token_id

# 扩展 input_ids
input_ids = [..., 151655, 151655, ..., 151655, ...]
            #            └────── 256 个占位符 ──────┘
```

### 输出格式

```python
{
    'input_ids': [..., 151655×256, ...],  # 256 个占位符
    'pixel_values': tensor([1024, 3, 14, 14]),  # 原始 patches
    'image_grid_thw': tensor([[1, 32, 32]]),    # 网格信息
    'labels': [...],  # 标签（用于计算 loss）
}
```

---

## 阶段 2: 模型 forward() - 训练/推理

### 输入格式

```python
{
    'input_ids': [..., 151655×256, ...],
    'pixel_values': tensor([1024, 3, 14, 14]),
    'image_grid_thw': tensor([[1, 32, 32]])
}
```

### 处理流程

#### 步骤 1: 语言 Token → Embedding

```python
# 文本部分通过 word embedding 层
text_embeds = model.embed_tokens(input_ids)
# 文本部分变成 embedding，占位符位置暂时是 image_token_id 的 embedding
```

#### 步骤 2: 图像处理 - 视觉编码器

```python
# pixel_values 通过视觉编码器（ViT）
pixel_values: tensor([1024, 3, 14, 14])  # 1024 个 patches
  ↓ visual_encoder (ViT)
patch_embeddings: tensor([1024, hidden_size])  # 1024 个独立的 patch embeddings
```

**处理过程**：
- 每个 14×14 的 patch 通过 ViT 编码
- 输出 1024 个独立的 patch embeddings
- 每个 embedding 维度为 `hidden_size`（例如 3584）

#### 步骤 3: Spatial Merge

```python
# Spatial Merge: 每 2×2 个 patch 合并成 1 个 token
patch_embeddings: tensor([1024, hidden_size])
  ↓ spatial_merge (merge_size=2)
image_embeddings: tensor([256, hidden_size])  # 256 个最终的 image embeddings
```

**合并过程**：
- 将 32×32 的网格重新组织
- 每 2×2 个相邻的 patch embeddings 合并（平均或拼接）
- 32×32 → 16×16 = 256 个 tokens
- 压缩比：4:1（1024 → 256）

#### 步骤 4: 替换占位符

```python
# 找到 input_ids 中 image_token_id 的位置
image_token_positions = torch.where(input_ids == image_token_id)[0]

# 用 image_embeddings 替换对应的 text_embeds
inputs_embeds = text_embeds.clone()
inputs_embeds[image_token_positions] = image_embeddings
```

#### 步骤 5: 最终 inputs_embeds

```python
inputs_embeds = [
    text_embeds[0], text_embeds[1], ...,        # 文本部分
    image_embeds[0], image_embeds[1], ...,     # 图像部分（256 个）
    image_embeds[255],
    text_embeds[...], ...                       # 后续文本
]
# 形状: [seq_len, hidden_size]
```

---

## 完整流程图

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段 1: template.encode() - 数据预处理                                        │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入:                                                                      │
│    messages: [{'role': 'user', 'content': 'Look at <image>.'}]            │
│    images: [PIL.Image(448×448)]                                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 语言处理                                                             │   │
│  │  文本: "Look at <image>"                                            │   │
│  │    ↓ tokenize                                                       │   │
│  │  input_ids = [..., 151655, ...]  # <image> → image_token_id        │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 图像处理                                                             │   │
│  │  PIL.Image(448×448)                                                 │   │
│  │    ↓ processor.image_processor()                                    │   │
│  │  pixel_values: [1024, 3, 14, 14]  # 1024 个 patches                 │   │
│  │  image_grid_thw: [1, 32, 32]      # 32×32 网格                      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 计算占位符数量（考虑 merge_size）                                     │   │
│  │  merge_size = 2                                                     │   │
│  │  merge_length = 2² = 4                                              │   │
│  │  token_len = (1 × 32 × 32) // 4 = 256                               │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 插入占位符                                                           │   │
│  │  input_ids = [..., 151655, 151655, ..., 151655, ...]               │   │
│  │                    └────────── 256 个占位符 ──────────┘            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  输出:                                                                      │
│    input_ids: [..., 151655×256, ...]                                       │
│    pixel_values: [1024, 3, 14, 14]                                         │
│    image_grid_thw: [1, 32, 32]                                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ↓
┌─────────────────────────────────────────────────────────────────────────────┐
│ 阶段 2: 模型 forward() - 训练/推理                                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  输入:                                                                      │
│    input_ids: [..., 151655×256, ...]                                       │
│    pixel_values: [1024, 3, 14, 14]                                         │
│    image_grid_thw: [1, 32, 32]                                             │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 语言 Token → Embedding                                               │   │
│  │  text_embeds = model.embed_tokens(input_ids)                       │   │
│  │  # 文本部分变成 embedding，占位符位置暂时是 image_token_id 的 embedding│   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 图像处理 - 视觉编码器                                                 │   │
│  │                                                                     │   │
│  │  pixel_values [1024, 3, 14, 14]                                    │   │
│  │    ↓ ViT (视觉编码器)                                               │   │
│  │  patch_embeddings [1024, hidden_size]  ← 1024 个独立的 embeddings   │   │
│  │    ↓ Spatial Merge (2×2 合并)                                       │   │
│  │  image_embeddings [256, hidden_size]  ← 256 个最终的 embeddings     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │ 替换占位符                                                           │   │
│  │  找到 input_ids 中 image_token_id 的位置                            │   │
│  │  用 image_embeddings 替换对应的 text_embeds                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  输出:                                                                      │
│    inputs_embeds: [text_embeds + image_embeddings]                         │
│    # 形状: [seq_len, hidden_size]                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 关键点总结

### 1. template.encode() 的作用

- ✅ **语言**：文本 → token (input_ids)
- ✅ **图像**：PIL.Image → pixel_values（预处理后的图像数据）+ 占位符（image_token_id）

### 2. 模型 forward() 的作用

- ✅ **语言 token** → 通过 `embed_tokens` 变成语言 embedding
- ✅ **图像**：pixel_values → 视觉编码器 → 1024 个 patch embeddings → spatial merge → 256 个 image embeddings

### 3. Spatial Merge 的位置

在视觉编码器内部完成，具体是：

```
pixel_values [1024, 3, 14, 14]
  ↓ ViT (视觉编码器)
patch_embeddings [1024, hidden_size]  ← 1024 个独立的 patch embeddings
  ↓ Spatial Merge (每 2×2 合并)
image_embeddings [256, hidden_size]  ← 256 个最终的 image embeddings
```

### 4. 为什么提前计算占位符数量？

- **数量必须匹配**：占位符数量（256）必须与实际 image embeddings 数量（256）完全匹配
- **位置一一对应**：每个占位符位置对应一个 image embedding
- **序列长度正确**：确保最终的 inputs_embeds 序列长度正确

### 5. 数据流示例

假设一张 448×448 的图像：

```
原始图像: PIL.Image(448×448)
  ↓ processor.image_processor()
pixel_values: [1024, 3, 14, 14]  # 1024 个 patches
image_grid_thw: [1, 32, 32]      # 32×32 网格
  ↓ 计算占位符
占位符数量: 1024 // 4 = 256
  ↓ 插入占位符
input_ids: [..., 151655×256, ...]
  ↓ 模型 forward
patch_embeddings: [1024, hidden_size]
  ↓ spatial merge
image_embeddings: [256, hidden_size]  ✅ 匹配！
```

---

## StreamVLN 与 Qwen2.5-VL 对比

### 图像 Token 压缩方式对比

| 方面 | StreamVLN | Qwen2.5-VL |
|------|-----------|------------|
| **压缩位置** | 模型 forward 中显式调用 | 视觉编码器内部隐式完成 |
| **压缩方法** | 2D Pooling (avg_pool2d) | Spatial Merge (merge_size) |
| **压缩比例** | 729 → 196 tokens (约 3.7:1) | 1024 → 256 tokens (4:1) |
| **可配置性** | stride 可调整，支持多种模式 | merge_size 固定（通常=2） |
| **实现位置** | `get_2dPool()` 方法 | 视觉编码器内部 |

### StreamVLN 的 2D Pooling

```python
# 在模型的 forward 中显式调用
def get_2dPool(self, image_feature, stride=2):
    # [N, 729, 1152] → [N, 27, 27, 1152] → [N, 1152, 27, 27]
    image_feature = image_feature.view(num_frames, height, width, -1)
    image_feature = image_feature.permute(0, 3, 1, 2).contiguous()
    
    # 2D 池化: [N, 1152, 27, 27] → [N, 1152, 14, 14]
    image_feature = nn.functional.avg_pool2d(image_feature, stride)
    
    # [N, 1152, 14, 14] → [N, 14, 14, 1152] → [N, 196, 1152]
    image_feature = image_feature.permute(0, 2, 3, 1)
    image_feature = image_feature.view(num_frames, -1, num_dim)
    return image_feature
```

**特点**：
- 显式操作：在视觉编码器输出后进行
- 可配置：stride 可调整，支持 average/max/bilinear
- 用途：减少 VLN 任务中大量图像的 token 数量

### Qwen2.5-VL 的 Spatial Merge

```python
# 在视觉编码器内部，通过 merge_size 参数控制
merge_length = processor.image_processor.merge_size**2  # = 4
token_len = (media_grid_thw[i].prod() // merge_length)
# 32×32=1024 patches → 1024÷4 = 256 tokens
```

**特点**：
- 内置操作：在视觉编码器内部自动完成
- 固定方式：通过 merge_size 参数控制（通常=2）
- 原理：每 2×2 个 patch 特征合并为 1 个 token

### 对应关系

**StreamVLN 的 `get_2dPool()` 对应 Qwen2.5-VL 中视觉编码器内部的 spatial merge 操作。**

两者都是为了减少 image token 数量（约 4 倍压缩），只是实现方式不同：
- StreamVLN：在 vision tower 输出后，**显式** 进行 2D pooling
- Qwen2.5-VL：在视觉编码器内部，**隐式** 通过 merge_size 实现

---

## 总结

1. **template.encode()** 阶段：
   - 语言变成 token，图像变成占位符 + pixel_values
   - 占位符数量提前计算（考虑 merge_size）

2. **模型 forward()** 阶段：
   - 语言 token 变成 embedding
   - 图像先变成 1024 个 patch embeddings，再 merge 成 256 个 image embeddings
   - 用 image embeddings 替换占位符

3. **设计优势**：
   - 占位符数量与实际 embeddings 数量完全匹配
   - 位置一一对应，不会出错
   - Spatial merge 在视觉编码器内部完成，对用户透明

---

## 参考

- Qwen2.5-VL Template: `ms-swift/swift/llm/template/template/qwen.py`
- StreamVLN Model: `StreamVLN/streamvln/model/stream_video_vln.py`
- StreamVLN Dataset: `StreamVLN/streamvln/dataset/vln_action_dataset.py`

