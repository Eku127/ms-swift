# StreamVLN 数据特点与训练方案

本文档全面介绍 VLN（视觉语言导航）数据的特点、StreamVLN 的数据处理方式以及训练方案设计。

## 目录

1. [VLN 数据特点](#vln-数据特点)
2. [StreamVLN 数据处理方案](#streamvln-数据处理方案)
3. [训练数据组织](#训练数据组织)
4. [训练范式设计](#训练范式设计)
5. [设计合理性分析](#设计合理性分析)

---

## VLN 数据特点

### 1. 数据基本结构

VLN 数据包含以下核心要素：

- **导航指令（Instructions）**：自然语言描述的目标位置，如 "Go to the kitchen"
- **图像序列（Image Sequence）**：导航过程中采集的 RGB 图像序列
- **动作序列（Action Sequence）**：对应的动作序列，包括前进、转向、停止等

### 2. 数据格式

#### 原始数据格式

```json
{
  "id": 1803,
  "video": "images/scene_xxx_r2r_001803",
  "instructions": ["Go to the kitchen"],
  "actions": [-1, 1, 1, 2, 1, 0]
}
```

#### 动作编码

- **-1**：初始占位符，表示起始状态
- **0**：STOP（停止/等待）
- **1**：MOVE_FORWARD（前进 25 厘米）
- **2**：TURN_LEFT（左转 15 度）
- **3**：TURN_RIGHT（右转 15 度）

### 3. 关键特点

#### 特点1：图像-动作对应关系

**核心规则**：当前图像用于预测下一个动作

- 图像 `{i:03d}.jpg` 是执行 `actions[i-1]` 后的观察结果
- 图像 `{i:03d}.jpg` 用于预测 `actions[i]`（下一个动作）
- 训练时需要将动作序列向后偏移一位

#### 特点2：长序列特性

- 单个轨迹可能包含几十到上百帧图像
- 需要处理长序列的上下文依赖
- 全局指令贯穿整个导航过程

#### 特点3：时序连续性

- 图像序列具有强时序依赖
- 历史观察对当前决策很重要
- 需要保留历史信息

#### 特点4：多模态融合

- 需要同时理解视觉信息（图像）和语言信息（指令）
- 图像和文本需要对齐和融合

---

## StreamVLN 数据处理方案

### 1. 数据切分策略

#### 固定窗口切分

- **窗口大小**：`num_frames = 32`（固定）
- **切分方式**：将长轨迹按 32 帧切分成多个训练样本
- **切分公式**：`num_rounds = (actions_len - valid_idx) // num_frames`

**示例**：
- 轨迹长度：96 帧
- 切分结果：3 个样本
  - Sample 0: 帧 [0-31]
  - Sample 1: 帧 [32-63]
  - Sample 2: 帧 [64-95]

#### 切分的合理性

**优点**：
- 序列长度可控，避免超长序列
- 训练效率高，批处理友好
- 与模型最大长度限制匹配

**缺点**：
- 固定窗口可能不适合所有任务
- 32 帧边界可能切断关键信息
- 前 32 帧的 sample 与全局 instruction 关联较弱

### 2. 历史帧处理

#### 历史帧采样

当 `time_ids[0] != 0`（不是第一个片段）时，会采样历史帧：

**采样策略1：随机采样**
```python
# 从 [0, start_idx) 范围随机采样 num_history 帧
history_step_ids = np.random.choice(
    available_history_indices, 
    size=num_history, 
    replace=False
)
```

**采样策略2：均匀采样**
```python
# 等间隔采样
step = max(time_ids[0] // num_history, 1)
history_step_ids = np.arange(0, time_ids[0], step)
```

#### 历史帧压缩

- 历史帧通过空间池化压缩（`history_stride = 2`）
- 压缩后的历史帧编码为 memory token (`<memory>`)
- 减少 token 数量，控制计算量

### 3. 当前帧采样

#### 采样间隔

- 根据 `num_future_steps` 参数间隔采样
- 公式：`sample_step_ids = np.arange(start_idx, end_idx, interval)`
- 如果 `num_future_steps = 4`，则每 4 帧采样一次

**示例**：
- 32 帧片段，`num_future_steps = 4`
- 采样结果：8 张图像（帧 0, 4, 8, 12, 16, 20, 24, 28）

### 4. 对话构建

#### 多轮对话结构

32 帧被组织成多轮对话：

- **轮数**：`32 帧 ÷ num_future_steps = 8 轮`（假设 `num_future_steps = 4`）
- **每轮结构**：
  - User: `"you can see <image>"`（随机选择连接词）
  - Assistant: `"↑←→↑"`（对应的动作序列）

#### 对话模板

```python
prompt = "You are an autonomous navigation assistant. Your task is to <instruction>. 
         Devise an action sequence to follow the instruction using the four actions: 
         TURN LEFT (←) or TURN RIGHT (→) by 15 degrees, 
         MOVE FORWARD (↑) by 25 centimeters, or STOP."

# 如果有历史帧
if start_idx != 0:
    prompt += " These are your historical observations: <memory>."
```

#### Memory Token 插入

- 历史帧压缩后编码为 `<memory>` token
- 插入到对话的开头部分
- 模型通过 memory token 访问历史信息

---

## 训练数据组织

### 1. 单个 Sample 的结构

每个训练样本（`__getitem__`）返回 5 个元素：

1. **`input_ids`** (torch.Tensor, shape: `[seq_len]`)
   - 完整对话序列的 token ID
   - 包含：系统消息 + 用户指令 + 图像 token (`<image>`) + 历史记忆 token (`<memory>`) + 多轮对话

2. **`labels`** (torch.Tensor, shape: `[seq_len]`)
   - 标签序列，只有 assistant 回答部分需要预测
   - 系统消息和用户输入部分标记为 `IGNORE_INDEX`

3. **`images`** (torch.Tensor, shape: `[num_images, 3, H, W]`)
   - 历史帧 + 当前帧的图像张量
   - 历史帧：`num_history` 帧（如 8 帧），压缩采样
   - 当前帧：根据 `num_future_steps` 间隔采样（如 8 帧）

4. **`time_ids`** (torch.Tensor, shape: `[num_frames]`)
   - 当前片段的时间戳索引
   - 范围：`[start_idx, start_idx + num_frames)`

5. **`task`** (int)
   - 任务类型标识（通常为 0）

### 2. Sample 组织流程

#### 步骤1：确定时间片段
- 从导航轨迹中切出一个片段（长度为 `num_frames = 32`）
- 计算 `time_ids = [start_idx, start_idx+1, ..., start_idx+31]`

#### 步骤2：采样图像
- **历史帧**（如果 `time_ids[0] != 0`）：
  - 从 `[0, start_idx)` 采样 `num_history` 帧（如 8 帧）
  - 采样策略：随机或均匀采样
- **当前帧**：
  - 从 `[start_idx, start_idx+num_frames)` 按 `num_future_steps` 间隔采样
  - 例如 `num_future_steps=4`：每 4 帧采样一次，得到约 8 张图像
- **最终**：`images = [历史帧（8张） + 当前帧（8张）] = 16 张图像`

#### 步骤3：构建对话
- **基础模板**：系统消息 + 用户指令（替换 `<instruction>` 占位符）
- **添加历史记忆**（如果有历史帧）：`"These are your historical observations: <memory>."`
- **构建多轮对话**：
  - 将动作序列按 `num_future_steps` 分组（如每 4 个动作一组）
  - 每组生成一轮对话：
    - `user`: `"you can see <image>"`（随机选择连接词）
    - `assistant`: `"↑←→↑"`（对应的动作序列）
  - 例如 32 个动作 ÷ 4 = 8 轮对话

#### 步骤4：Tokenization
- 使用 `preprocess_qwen()` 处理对话
- **`input_ids`**：完整对话的 token ID
- **`labels`**：只有 assistant 回答部分是真实 token，其他都是 `IGNORE_INDEX`

### 3. Batch 组织方式

#### Collate 函数处理

`collate_fn` 函数将多个样本组织成一个 batch：

1. **Padding 序列数据**：
   - `input_ids` 和 `labels` 统一 padding 到 batch 中最长序列的长度
   - 使用 `pad_sequence` 进行 padding

2. **生成 attention_mask**：
   - 标记哪些是真实 token（True），哪些是 padding token（False）

3. **Padding 图像数据**：
   - 统一 padding 到 batch 中最多图像的数量
   - 使用 `pad_tensors` 进行 padding

4. **Padding time_ids**：
   - 统一 padding，使用 `-1` 作为 padding 值

#### Batch 特点

- **序列长度不一致**：不同样本的对话长度可能不同
- **图像数量不一致**：不同样本的图像数量可能不同
- **时间戳对齐**：`time_ids` 用于标识每个样本的时间片段

### 4. 示例说明

假设一个样本的参数：
- `num_frames = 32`（片段长度）
- `num_history = 8`（历史帧数量）
- `num_future_steps = 4`（动作分组大小）
- 动作序列：32 个动作

**组织结果**：

```python
# 1. images: [8 历史帧 + 8 当前帧] = 16 张图像
images.shape = [16, 3, 384, 384]

# 2. time_ids: [0, 1, 2, ..., 31]（32 个时间戳）
time_ids.shape = [32]

# 3. 对话结构（简化版）：
# <|im_start|>system
# You are an autonomous navigation assistant. Your task is to "go to the kitchen". 
# These are your historical observations: <memory>.
# <|im_end|>
# <|im_start|>user
# you can see <image>
# <|im_end|>
# <|im_start|>assistant
# ↑↑←→
# <|im_end|>
# <|im_start|>user
# in front of you is <image>
# <|im_end|>
# <|im_start|>assistant
# ↑↑←→
# <|im_end|>
# ...（共 8 轮对话，因为 32 个动作 / 4 = 8 组）

# 4. input_ids: 上述对话的 token ID 序列
input_ids.shape = [seq_len]  # 例如 500 tokens

# 5. labels: 与 input_ids 同长，只有 assistant 回答部分是真实 token，其他都是 IGNORE_INDEX
labels.shape = [seq_len]
```

---

## 训练范式设计

### 1. 核心设计理念

#### 32 帧窗口 + 多轮对话

- **窗口大小**：固定 32 帧
- **对话结构**：32 帧组织成多轮对话（每轮 4 个动作）
- **历史信息**：通过 memory token 编码压缩后的历史帧

#### 训练-推理一致性

- **训练时**：32 帧 sample + memory token（压缩历史帧）
- **推理时**：32 帧内使用 KV cache，32 帧边界重置 + memory token（压缩历史帧）
- **一致性**：两者都使用 memory token 编码历史，边界处理一致

### 2. 训练时学到的能力

#### 能力1：增量视觉-动作映射

- 看到新图像 → 立即做出动作决策
- 每帧图像对应一组动作（4 个动作）
- 学习视觉观察与动作的实时对应关系

#### 能力2：动作序列规划

- 每轮预测 4 个动作（`num_future_steps=4`）
- 学习短期规划：预测未来几步的动作
- 学习动作连贯性：4 个动作需要连贯合理

#### 能力3：上下文记忆与利用

- 利用前面的对话历史
- 理解动作序列的连续性
- 避免重复或矛盾的动作

#### 能力4：指令理解与执行

- 理解全局导航指令（如 "go to the kitchen"）
- 将指令转化为具体动作序列
- 在指令指导下做出每一步决策

#### 能力5：多模态融合

- 理解视觉信息（图像）
- 将视觉信息与文本指令结合
- 整合历史视觉信息与当前观察

### 3. 训练配置

#### 关键参数

- **`num_frames`**：32（窗口大小）
- **`num_history`**：8（历史帧数量）
- **`num_future_steps`**：4（每轮预测的动作数）
- **`history_stride`**：2（历史帧空间池化步长）
- **`current_stride`**：2（当前帧空间池化步长）

#### 训练设置

- **KV Cache**：训练时禁用（`use_cache=False`）
- **批处理**：使用 `collate_fn` 处理不同长度的样本
- **损失计算**：只有 assistant 回答部分参与损失计算

---

## 设计合理性分析

### 1. 合理之处

#### ✅ 训练-推理一致性较好

- 训练时：32 帧 sample + memory token（压缩历史帧）
- 推理时：32 帧内使用 KV cache，32 帧边界重置 + memory token（压缩历史帧）
- 一致性：两者都使用 memory token 编码历史，边界处理一致

#### ✅ 计算效率可控

- 32 帧窗口限制序列长度，避免超长序列
- Memory token 压缩历史帧，减少 token 数量
- 32 帧边界重置 KV cache，避免无限增长
- 空间池化进一步压缩

#### ✅ 设计简洁

- 固定窗口大小，实现简单
- Memory token 机制直观
- 边界处理清晰

#### ✅ 符合实际需求

- VLN 任务中，32 帧窗口通常足够
- 历史信息通过 memory token 保留
- 与语言模型的对话格式兼容

### 2. 可改进之处

#### ⚠️ 训练-推理不完全一致

- **训练时**：一次性看到完整的 32 帧 + memory token
- **推理时**：逐步生成，KV cache 累积，32 帧边界重置
- **问题**：训练时没有模拟逐步生成的过程

#### ⚠️ Memory Token 与 KV Cache 的冗余

- **32 帧边界时**：清理 KV cache + 重新编码历史帧为 memory token
- **问题**：KV cache 已包含历史信息，清理后再用 memory token 重新编码，存在冗余

#### ⚠️ 历史信息可能丢失

- **32 帧边界重置 KV cache**：丢失已计算的注意力信息
- **Memory token 压缩**：可能丢失细节信息

#### ⚠️ 固定窗口大小的限制

- **32 帧固定**：无法适应不同长度的任务
- **短任务**：可能浪费
- **长任务**：可能不够

### 3. 设计权衡

StreamVLN 在以下方面做了权衡：

| 方面 | 选择 | 原因 | 代价 |
|------|------|------|------|
| **窗口大小** | 固定 32 帧 | 平衡效率和上下文 | 灵活性不足 |
| **历史信息** | Memory token | 保留历史，控制计算 | 可能丢失细节 |
| **KV cache** | 32 帧重置 | 避免无限增长 | 丢失已计算信息 |
| **训练方式** | 一次性处理 | 训练效率高 | 与推理不完全一致 |

### 4. 总体评价

**整体合理性**：⭐⭐⭐⭐（4/5）

**优点**：
- 训练-推理一致性较好
- 计算效率可控
- 实现相对简单
- 符合实际需求

**缺点**：
- 训练-推理不完全一致（一次性 vs 逐步）
- Memory token 与 KV cache 存在冗余
- 固定窗口限制灵活性
- 历史信息可能丢失

**结论**：StreamVLN 的设计在当前技术约束下是合理的，在效率、效果、实现复杂度之间取得了平衡。虽然不是最优，但在实际应用中是一个合理的选择。

---

## 总结

### 核心设计

1. **32 帧窗口**：固定窗口大小，平衡效率和上下文
2. **多轮对话**：32 帧组织成多轮对话，每轮预测 4 个动作
3. **Memory Token**：压缩历史帧，保留历史信息
4. **训练-推理一致**：两者都使用 memory token，边界处理一致

### 关键特点

- **数据切分**：固定 32 帧窗口切分长轨迹
- **历史处理**：采样 + 压缩历史帧为 memory token
- **对话构建**：多轮对话结构，模拟推理过程
- **训练目标**：学习增量决策、序列规划、上下文利用等能力

### 适用场景

- 中等长度的 VLN 任务（30-100 步）
- 需要保留历史信息的导航任务
- 计算资源有限的实际应用场景

---

## 参考

- StreamVLN 数据格式文档：`streamvln_data_format.md`
- StreamVLN 训练与推理对比：`streamvln_training_vs_inference.md`
- StreamVLN 数据加载类：`streamvln/dataset/vln_action_dataset.py`
- StreamVLN 模型实现：`streamvln/model/stream_video_vln.py`
