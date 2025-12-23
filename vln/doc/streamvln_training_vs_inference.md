# StreamVLN 训练与推理机制对比分析

本文档详细分析 StreamVLN 在训练和推理阶段对 KV cache 和历史输出（上一时刻的 output）的使用差异。

## 核心结论

**KV cache 和历史输出的使用仅在推理阶段进行，训练阶段不使用这些机制。**

## 训练阶段（Training）

### 1. KV Cache 设置

**代码位置**：`streamvln/streamvln_train.py`

```python
# 第1619行：训练开始时禁用 KV cache
model.config.use_cache = False

# ... 训练过程 ...

# 第1892行：训练结束后才启用（用于保存模型）
model.config.use_cache = True
```

**关键点**：
- 训练时 `use_cache = False`，模型不会缓存注意力计算的 Key-Value 对
- 每个训练样本都是独立处理的，不保留历史状态
- 训练结束后才启用 cache，仅用于模型保存

### 2. 训练数据格式

**代码位置**：`streamvln/dataset/vln_action_dataset.py`

训练时，每个样本包含：
- **完整的对话序列**：包含多轮图像和动作的完整对话
- **历史帧 + 当前帧**：通过数据预处理阶段采样得到
- **动作序列**：一次性提供完整的动作序列（如 "↑←→↑"）

**示例训练样本结构**：
```python
# 一个训练样本可能包含：
# - 历史帧：8 帧（通过采样得到）
# - 当前帧：32 帧（按 stride 采样）
# - 对话：多轮 user-assistant 对话
# - 动作序列：预测未来 N 步动作（如 num_future_steps=4）
```

### 3. Batch 中每个 Sample 的组织方式

**代码位置**：`streamvln/dataset/vln_action_dataset.py`

#### 3.1 单个 Sample 的组织方式

**代码位置**：`streamvln/dataset/vln_action_dataset.py` 第737-859行

每个训练样本（`__getitem__`）返回 5 个元素：`input_ids`、`labels`、`images`、`time_ids`、`task`。

**数据组织流程**：

1. **确定时间片段**
   - 从导航轨迹中切出一个片段（长度为 `num_frames`，如 32 帧）
   - 计算 `time_ids = [start_idx, start_idx+1, ..., start_idx+num_frames-1]`

2. **采样图像**
   - **历史帧**（如果 `time_ids[0] != 0`，即不是第一个片段）：
     - 从 `[0, start_idx)` 范围采样 `num_history` 帧（如 8 帧）
     - 采样策略：随机采样或均匀采样（由 `use_random` 参数控制）
   - **当前帧**：
     - 从 `[start_idx, start_idx+num_frames)` 按 `num_future_steps` 间隔采样
     - 例如 `num_future_steps=4`：每 4 帧采样一次，得到约 8 张图像
   - **最终**：`images = [历史帧（8张） + 当前帧（8张）]` = 16 张图像，shape: `[16, 3, H, W]`

3. **构建对话**
   - **基础模板**：系统消息 + 用户指令（替换 `<instruction>` 占位符）
   - **添加历史记忆**（如果有历史帧）：`"These are your historical observations: <memory>."`
   - **构建多轮对话**（`prepare_conversation` 函数）：
     - 将动作序列按 `num_future_steps` 分组（如每 4 个动作一组）
     - 每组生成一轮对话：
       - `user`: `"you can see <image>"`（随机选择一个连接词）
       - `assistant`: `"↑←→↑"`（对应的动作序列）
     - 例如 32 个动作 ÷ 4 = 8 轮对话

4. **Tokenization**
   - 使用 `preprocess_qwen()` 处理对话：
     - **`input_ids`**：完整对话的 token ID（包含系统消息、用户输入、图像 token、记忆 token、助手回答）
     - **`labels`**：与 `input_ids` 同长
       - 系统消息和用户输入部分：`IGNORE_INDEX`（不参与损失）
       - 助手回答部分：真实 token ID（参与损失计算）

**返回的 5 个字段**：

- **`input_ids`** (torch.Tensor, shape: `[seq_len]`)：完整对话序列的 token ID
- **`labels`** (torch.Tensor, shape: `[seq_len]`)：标签序列，只有 assistant 回答部分需要预测
- **`images`** (torch.Tensor, shape: `[num_images, 3, H, W]`)：历史帧 + 当前帧的图像张量
- **`time_ids`** (torch.Tensor, shape: `[num_frames]`)：当前片段的时间戳索引
- **`task`** (int)：任务类型标识（通常为 0）

#### 3.2 Batch 的组织方式（`collate_fn` 函数）

**代码位置**：`streamvln/dataset/vln_action_dataset.py` 第879-902行

`collate_fn` 函数将多个样本组织成一个 batch，主要处理不同样本长度不一致的问题：

1. **解包**：从 batch 中提取所有样本的各个字段（`input_ids`、`labels`、`images`、`time_ids`、`task`）

2. **Padding 序列数据**：
   - `input_ids` 和 `labels` 使用 `pad_sequence` 统一 padding 到 batch 中最长序列的长度
   - `input_ids` 使用 `tokenizer.pad_token_id` 作为 padding 值
   - `labels` 使用 `IGNORE_INDEX` 作为 padding 值
   - 截断到模型最大长度（`tokenizer.model_max_length`）

3. **生成 attention_mask**：
   - 标记哪些是真实 token（True），哪些是 padding token（False）
   - shape: `[batch_size, max_seq_len]`

4. **Padding 图像数据**：
   - 使用 `pad_tensors` 统一 padding 到 batch 中最多图像的数量
   - shape: `[batch_size, max_num_images, 3, H, W]`

5. **Padding time_ids**：
   - 使用 `pad_sequence` 统一 padding，使用 `-1` 作为 padding 值
   - shape: `[batch_size, max_num_frames]`

**返回的 batch 字典**：
- `input_ids`: `[batch_size, max_seq_len]`
- `labels`: `[batch_size, max_seq_len]`
- `images`: `[batch_size, max_num_images, 3, H, W]`
- `time_ids`: `[batch_size, max_num_frames]`
- `attention_mask`: `[batch_size, max_seq_len]`
- `task_type`: tuple of ints

#### 3.3 Batch 中 Sample 的组织特点

1. **序列长度不一致**：
   - 不同样本的对话长度可能不同（取决于历史帧数量、动作序列长度等）
   - 通过 `pad_sequence` 统一 padding 到 batch 中最长序列的长度
   - 使用 `attention_mask` 标记真实 token 和 padding token

2. **图像数量不一致**：
   - 不同样本的图像数量可能不同（取决于是否有历史帧、采样间隔等）
   - 通过 `pad_tensors` 统一 padding 到 batch 中最多图像的数量
   - 第一个样本如果没有历史帧，图像数量会少于有历史帧的样本

3. **时间戳对齐**：
   - `time_ids` 用于标识每个样本中哪些帧属于同一个时间片段
   - 通过 padding value `-1` 标记无效的时间戳

4. **对话结构**：
   - 每个样本的对话通过 `prepare_conversation()` 函数构建
   - 对话格式：多轮 user-assistant 对话，每轮对应一组动作
   - 每个样本是独立的，包含完整的对话历史

#### 3.4 示例说明

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

**关键点**：
- 每个样本是独立的，不依赖其他样本，包含完整对话历史
- 图像组织：历史帧 + 当前帧，按 `num_future_steps` 间隔采样
- 对话结构：多轮 user-assistant 对话，每轮对应一组动作
- 标签掩码：只有 assistant 的回答参与损失计算

### 4. 训练时的前向传播

**特点**：
- **一次性处理**：模型一次性处理整个对话序列（包含多轮图像和动作）
- **无状态**：每个 batch 的样本都是独立的，不保留前一个 batch 的状态
- **完整序列**：输入包含完整的对话历史（通过数据预处理构建）

**数据流**：
```
训练样本 → 预处理（构建完整对话） → Batch 组织（collate_fn） → 模型前向传播 → 计算损失
```

**Batch 处理流程**：
1. DataLoader 从 Dataset 中采样 `batch_size` 个样本
2. 每个样本通过 `__getitem__` 返回 5 个字段（input_ids, labels, images, time_ids, task）
3. `collate_fn` 将所有样本的字段分别进行 padding 和对齐
4. 返回统一的 batch 字典，包含所有字段的 batch 版本
5. 模型接收 batch，进行前向传播和损失计算

### 5. 为什么训练时不使用 KV Cache？

**原因**：
1. **梯度计算需求**：训练时需要计算梯度，KV cache 会干扰反向传播
2. **批处理效率**：训练时使用批处理，每个样本长度可能不同，KV cache 管理复杂
3. **数据完整性**：训练数据已经包含了完整的历史信息（通过历史帧采样），不需要缓存
4. **内存效率**：不使用 cache 可以更灵活地处理不同长度的序列

## 推理阶段（Inference）

### 1. KV Cache 设置

**代码位置**：`streamvln/streamvln_eval.py`

```python
# 第340行：初始化
past_key_values = None
output_ids = None

# 第472行：推理时启用 KV cache
outputs = self.model.generate(
    **input_dict,
    use_cache=True,                    # 启用 KV cache
    past_key_values=past_key_values,   # 传入上一轮的 KV cache
    ...
)

# 第480行：保存 KV cache 用于下一轮
past_key_values = outputs.past_key_values
```

**关键点**：
- 推理时 `use_cache = True`，启用 KV cache 加速生成
- `past_key_values` 保存上一轮计算的注意力 Key-Value 对
- 下一轮生成时复用这些缓存，避免重复计算

### 2. 历史输出的使用

**代码位置**：`streamvln/streamvln_eval.py` 第438-439行

```python
# 如果存在上一时刻的输出，将其拼接到当前输入
if output_ids is not None:
    input_ids = torch.cat([output_ids, input_ids.to(output_ids.device)], dim=1)
```

**关键点**：
- `output_ids` 保存上一轮模型生成的 token 序列
- 下一轮生成时，将上一轮的输出拼接到新的输入前面
- 这样模型可以看到完整的对话历史（包括自己之前的输出）

### 3. 推理时的增量生成流程

**完整流程**：

```
第1轮生成：
  输入：指令 + 当前图像 + <memory>（如果有历史）
  输出：动作序列 "↑←→↑"
  保存：past_key_values, output_ids

第2轮生成（action_seq 为空时）：
  输入：output_ids（上一轮输出） + 新的 prompt + 新的图像
  使用：past_key_values（复用上一轮的注意力计算）
  输出：新的动作序列
  更新：past_key_values, output_ids

第3轮生成：
  ... 重复上述过程
```

**关键代码逻辑**（第408-480行）：

```python
while not env.episode_over:
    if len(action_seq) == 0:  # 动作序列执行完毕，需要生成新的
        if output_ids is None:
            # 第一次生成：构建完整对话
            sources = copy.deepcopy(self.conversation)
            # ... 添加指令和历史信息 ...
        else:
            # 后续生成：只构建新的 prompt
            sources = [{"from": "human", "value": ""}, {"from": "gpt", "value": ""}]
        
        # 预处理对话
        input_ids, conversations = self.preprocess_qwen([sources], ...)
        
        # 拼接上一轮输出
        if output_ids is not None:
            input_ids = torch.cat([output_ids, input_ids], dim=1)
        
        # 生成（使用 KV cache）
        outputs = self.model.generate(
            ...,
            use_cache=True,
            past_key_values=past_key_values
        )
        
        # 保存状态用于下一轮
        output_ids = outputs.sequences
        past_key_values = outputs.past_key_values
        
        # 解析动作序列
        action_seq = self.parse_actions(llm_outputs)
    
    # 执行动作
    action = action_seq.pop(0)
    observations = env.step(action)
    
    # 每32步重置一次（清理 KV cache）
    if step_id % self.num_frames == 0:
        self.model.reset_for_env(idx)
        output_ids = None
        past_key_values = None
```

### 4. 为什么推理时使用 KV Cache 和历史输出？

**原因**：
1. **加速生成**：KV cache 避免重复计算已处理 token 的注意力，大幅提升生成速度
2. **内存效率**：只缓存必要的 Key-Value 对，而不是重新计算整个序列
3. **对话连续性**：通过拼接历史输出，模型可以看到完整的对话历史
4. **实际应用场景**：推理时是逐步生成的过程，需要保持状态

### 5. 定期重置机制

**代码位置**：`streamvln/streamvln_eval.py` 第535-539行

```python
# 每32步（num_frames）重置一次
if step_id % self.num_frames == 0:
    self.model.reset_for_env(idx)
    output_ids = None
    past_key_values = None
    time_ids = []
```

**原因**：
- **防止序列过长**：避免 KV cache 无限增长导致内存溢出
- **重新开始**：每32步重新开始一个片段，保持模型性能
- **历史帧采样**：重置时重新采样历史帧，提供新的上下文

## 关键差异总结

| 特性 | 训练阶段 | 推理阶段 |
|------|---------|---------|
| **KV Cache** | ❌ 禁用（`use_cache=False`） | ✅ 启用（`use_cache=True`） |
| **历史输出** | ❌ 不使用 | ✅ 拼接上一轮输出到输入 |
| **数据格式** | 完整对话序列（多轮） | 增量生成（单轮） |
| **状态保持** | ❌ 无状态（每个样本独立） | ✅ 有状态（保持 past_key_values） |
| **处理方式** | 一次性处理完整序列 | 逐步生成，复用缓存 |
| **历史信息** | 通过数据预处理提供 | 通过 KV cache + 输出拼接提供 |

## 设计原理

### 训练阶段的设计

1. **数据完整性**：训练数据已经包含了完整的历史信息（通过历史帧采样和多轮对话构建）
2. **批处理友好**：不使用 cache 可以更灵活地处理不同长度的序列
3. **梯度计算**：禁用 cache 避免干扰反向传播

### 推理阶段的设计

1. **效率优化**：KV cache 大幅提升生成速度，避免重复计算
2. **对话连续性**：通过输出拼接和 cache，模型可以看到完整的对话历史
3. **实际应用**：推理时是逐步生成的过程，需要保持状态

## 对 ms-swift 实现的启示

在 ms-swift 中实现 VLN 训练时：

1. **训练阶段**：
   - 不需要实现 KV cache 机制
   - 数据预处理时构建完整的对话序列（包含历史帧和当前帧）
   - 每个训练样本是独立的，不保留状态

2. **推理阶段**（如果需要）：
   - 可以实现 KV cache 加速生成
   - 需要处理历史输出的拼接
   - 需要实现定期重置机制

3. **数据格式**：
   - 训练数据：完整的对话序列（多轮图像+动作）
   - 推理数据：单轮输入，逐步生成

## 参考代码位置

- **训练代码**：`streamvln/streamvln_train.py`（第1619行，第1892行）
- **推理代码**：`streamvln/streamvln_eval.py`（第340行，第438-439行，第472行，第480行，第535-539行）
- **模型定义**：`streamvln/model/stream_video_vln.py`（`prepare_inputs_for_generation` 方法）
