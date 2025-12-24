# StreamVLN 在 ms-swift 中的训练流程详解

## 概述

本文档详细描述了 StreamVLN 模型在 ms-swift 框架中的完整训练流程，从脚本启动到模型保存的每一个步骤。该流程基于 ms-swift 的统一训练框架，通过 `swift sft` 命令实现端到端的训练过程。

---

## 一、训练启动阶段

### 1.1 脚本初始化

当执行 `bash examples/vln/streamvln/script/train_streamvln_qwen2_vl.sh` 时，训练流程开始：

**环境准备**：
- 脚本首先设置各种环境变量，包括 PyTorch CUDA 内存分配策略、NCCL 通信配置等
- **模型缓存目录配置**：
  - ms-swift **默认使用 ModelScope** 下载和缓存模型
  - 可通过 `MODELSCOPE_CACHE` 环境变量设置 ModelScope 缓存目录（默认：`~/.cache/modelscope`）
  - 如果设置 `USE_HF=1`，则使用 HuggingFace（通过 `HF_HOME` 设置缓存目录）
  - 示例：`export MODELSCOPE_CACHE=/shared_space/jiangjiajun/modelscope_cache`
- 如果启用了样本数量限制（MAX_SAMPLES），会设置环境变量 `VLN_MAX_SAMPLES` 传递给数据集

**参数解析**：
- 脚本读取所有配置参数，包括 VLN 特定参数（num_frames、num_history 等）、模型配置、训练超参数等
- 构建 DeepSpeed 和 SwanLab 相关的命令行参数
- 打印训练配置摘要，包括模型信息、数据路径、批次大小、组件冻结状态等

### 1.2 swift sft 命令调用

脚本最终调用 `swift sft` 命令，传入所有配置参数。ms-swift 框架的 CLI 层会：

**参数验证**：
- 验证模型类型 `streamvln-qwen2-vl-3b` 是否已注册
- 检查数据路径是否存在
- 验证所有必需参数是否提供

**分布式检测**：
- 检测环境变量 `NPROC_PER_NODE`，如果设置则自动启用分布式训练
- 对于单节点多卡（如 8 卡），框架会自动使用 `torch.distributed.run` 启动多进程训练
- 设置主进程和从进程的通信地址和端口

---

## 二、框架初始化阶段

### 2.1 SwiftSft 类初始化

ms-swift 框架创建 `SwiftSft` 训练器实例，这是整个训练流程的核心控制器：

**组件准备顺序**：
1. **参数对象创建**：将命令行参数解析为 `TrainArguments` 对象，包含所有训练配置
2. **模型和分词器准备**：调用 `_prepare_model_tokenizer()` 方法
3. **模板准备**：调用 `_prepare_template()` 设置对话模板
4. **回调函数准备**：设置训练过程中的回调函数（如 SwanLab 日志记录）
5. **Flash Checkpoint 准备**：如果启用，准备检查点相关功能

### 2.2 模型加载

**模型注册查找**：
- 框架根据 `model_type="streamvln-qwen2-vl-3b"` 查找注册的模型加载函数
- 找到 `get_model_tokenizer_streamvln_qwen2_vl()` 函数

**基础模型加载**：
- **模型来源选择**：
  - ms-swift **默认使用 ModelScope** 下载模型
  - 如果设置 `USE_HF=1` 环境变量，则使用 HuggingFace
  - 模型会从 ModelScope 或 HuggingFace 下载到缓存目录
- 从 ModelScope/HuggingFace 或本地路径加载基础模型 `Qwen/Qwen2.5-VL-3B-Instruct`
- 加载对应的 tokenizer，支持中文和多模态输入
- 加载 Qwen2VLProcessor，用于图像预处理
- 模型缓存位置：
  - ModelScope：`$MODELSCOPE_CACHE/hub`（默认：`~/.cache/modelscope/hub`）
  - HuggingFace：`$HF_HOME/hub`（默认：`~/.cache/huggingface/hub`）

**StreamVLN 模型包装**：
- 将基础 Qwen2VL 模型包装为 `StreamVLNQwen2VLForConditionalGeneration` 实例
- 设置 StreamVLN 特定参数：num_history、current_stride、history_stride、spatial_pool_mode
- 这些参数控制历史帧采样、空间池化等 VLN 特定行为

**模型组件冻结**：
- 根据配置参数 `freeze_vit`、`freeze_llm`、`freeze_aligner` 决定哪些组件可训练
- 冻结的组件参数设置为 `requires_grad=False`，不参与梯度更新
- 例如，如果 `freeze_vit=true`，视觉编码器的所有参数都会被冻结

### 2.3 模板初始化

**模板选择**：
- 根据模型类型选择对应的对话模板，StreamVLN 使用 `TemplateType.qwen2_vl`
- 模板定义了系统消息格式、用户-助手对话格式、特殊 token 处理等

**特殊 Token 处理**：
- 模板确保 `<image>` 和 `<memory>` token 被正确处理
- 这些 token 在后续的多模态融合阶段会被替换为实际的视觉特征

---

## 三、数据集准备阶段

### 3.1 数据集实例化

**数据集类创建**：
- 框架创建 `StreamVLNDataset` 实例
- 传入数据路径、tokenizer、processor 以及所有 VLN 特定参数

**数据加载过程**：

**步骤 1：加载导航数据**
- 解析数据路径（支持逗号分隔的多个路径）
- 遍历每个路径，查找 `annotations.json` 文件
- 读取 JSON 文件，获取所有导航轨迹数据
- 每个轨迹包含：视频路径、指令列表、动作序列

**步骤 2：构建数据索引**
- 遍历所有导航轨迹（episode）
- 对于每个轨迹，检查动作序列长度（至少需要 4 个动作）
- 对于每个指令（一个轨迹可能有多个指令），计算可用的时间窗口
- 按照 `num_frames=32` 的窗口大小切分轨迹
- 生成数据索引列表：`(episode_id, instruction_id, start_frame)`
- 这个索引列表定义了所有可用的训练样本

**步骤 3：样本数量限制**
- 如果设置了 `max_samples` 或环境变量 `VLN_MAX_SAMPLES`，截取前 N 个样本
- 打印限制信息，告知用户实际使用的样本数量

**步骤 4：初始化辅助组件**
- 设置动作到文本的映射（0→STOP, 1→↑, 2→←, 3→→）
- 准备提示词模板（conjunctions）用于构建多轮对话
- 设置基础对话模板，包含系统提示和占位符

### 3.2 数据加载器创建

**DataLoader 配置**：
- 框架创建 PyTorch DataLoader，使用 `StreamVLNDataset` 作为数据源
- 设置 `num_workers`（数据加载并行度）
- 设置 `collate_fn` 函数，用于批处理多个样本

**批处理函数（collate_fn）**：
- 接收一个 batch 的样本（每个样本包含 input_ids、labels、images、time_ids、task_type）
- 对 input_ids 和 labels 进行 padding，使所有序列长度一致
- 对 images 进行 padding，使所有样本的图像数量一致
- 对 time_ids 进行 padding
- 生成 attention_mask，标记哪些位置是真实数据，哪些是 padding
- 返回批处理后的字典，包含所有必要的数据

---

## 四、训练循环阶段

### 4.1 训练器初始化

**Trainer 创建**：
- ms-swift 框架创建 HuggingFace Trainer 实例
- 配置优化器（AdamW）、学习率调度器（Cosine）
- 如果启用 DeepSpeed，初始化 DeepSpeed 引擎
- 设置梯度检查点（gradient checkpointing）以节省显存

**分布式初始化**：
- 如果使用多 GPU，初始化进程组（ProcessGroup）
- 设置每个进程的 rank 和 world_size
- 配置模型并行和数据并行策略

### 4.2 每个训练步骤的详细流程

对于每个训练步骤（step），执行以下流程：

#### 步骤 1：数据获取

**数据采样**：
- DataLoader 从数据集中采样一个 batch 的样本
- 每个样本通过 `StreamVLNDataset.__getitem__()` 方法获取

**单个样本处理**（在 `__getitem__` 中）：

**1.1 获取轨迹片段**：
- 根据数据索引 `(ep_id, ins_id, start_idx)` 定位到具体的轨迹片段
- 从导航数据中获取对应的轨迹信息（视频路径、指令、动作序列）
- `ep_id`：轨迹编号，`ins_id`：指令编号，`start_idx`：窗口起始帧

**1.2 动作序列处理**：
- 获取动作序列，并向后移动一位（因为要预测下一个动作）
- 在末尾添加 STOP 动作（0）作为结束标记
- 根据时间窗口（time_ids）提取对应的动作子序列

**1.3 图像帧采样**：

**当前帧采样**：
- 根据时间窗口和 `num_future_steps` 计算需要采样的帧索引
- 使用均匀间隔采样（stride = num_future_steps）
- 从视频的 rgb 文件夹中加载对应的图像文件

**历史帧采样**（如果存在）：
- 如果当前片段不是轨迹的第一个片段（start_idx > 0），需要采样历史帧
- 根据 `use_random` 参数选择采样策略：
  - 随机采样：从可用历史帧中随机选择 `num_history` 个
  - 均匀采样：等间隔采样 `num_history` 个历史帧
- 历史帧用于构建 memory token

**1.4 图像预处理**：
- 使用 Qwen2VLProcessor 的 image_processor 处理每张图像
- 调整大小、归一化、转换为张量格式
- 所有图像堆叠成张量：`(num_images, 3, H, W)`

**1.5 对话构建**：

**基础模板准备**：
- 复制基础对话模板
- 如果存在历史帧，在系统提示中添加 memory token 占位符
- 将指令文本替换到模板的 `<instruction>` 占位符

**多轮对话生成**：
- 调用 `prepare_conversation()` 方法
- 将动作序列按照 `num_future_steps` 分组（每组 4 个动作）
- 为每组动作创建一轮用户-助手对话：
  - 用户消息：随机选择一个 conjunction + `<image>` token
  - 助手消息：动作序列的文本表示（如 "↑←→↑"）
- 所有轮次连接成完整的对话序列

**1.6 Tokenization**：
- 调用 `preprocess_qwen_vln()` 函数处理对话
- 添加特殊 token（`<image>` 和 `<memory>`）到 tokenizer
- 使用 tokenizer 的 chat_template 格式化对话
- 将特殊 token 替换为对应的索引：
  - `<image>` → `IMAGE_TOKEN_INDEX (-200)`
  - `<memory>` → `MEMORY_TOKEN_INDEX (-201)`
- 生成 input_ids 和 labels：
  - input_ids：完整的对话 token 序列
  - labels：只有 assistant 部分的 token 保留，其他部分设为 `IGNORE_INDEX (-100)`

**1.7 返回样本**：
- 返回元组：`(input_ids, labels, images, time_ids, task_type)`
- **关于 task_type**：
  - 数据集返回 `task_type`（默认值为 `task_id=0`）
  - `collate_fn` 收集 `task_type_batch` 并包含在返回字典中
  - 但模型的前向传播方法**没有接收或使用** `task_type` 参数
  - 这是为了兼容原始 StreamVLN 的接口而保留的，但实际功能未实现
  - 如果未来需要支持多任务训练（如不同数据集类型），可以扩展使用 `task_type`

#### 步骤 2：批处理

**Collate 函数处理**：
- 接收一个 batch 的样本
- 对所有序列进行 padding，使长度一致
- 对所有图像进行 padding，使数量一致
- 生成 attention_mask
- 收集 `task_type`（虽然当前未被模型使用）
- 返回批处理字典，包含：`input_ids`, `labels`, `images`, `time_ids`, `attention_mask`, `task_type`

#### 步骤 3：前向传播

**模型前向传播**（在 `StreamVLNQwen2VLForConditionalGeneration.forward()` 中）：

**3.1 多模态输入准备**：
- 调用 `prepare_inputs_labels_for_multimodal()` 方法
- **注意**：该方法**不接收** `task_type` 参数，只使用 `images` 和 `time_ids`
- 与原始 StreamVLN 不同，ms-swift 版本简化了接口，移除了未使用的参数（depths、poses、intrinsics、task_ids）

**3.2 图像编码**（在 `encode_images_with_history()` 中）：

**视觉编码**：
- 将所有图像（历史帧 + 当前帧）展平为 `(batch_size * num_images, 3, H, W)`
- 通过 Qwen2VL 的视觉编码器（vision tower）处理
- 输出视觉特征：`(batch_size * num_images, num_tokens, hidden_dim)`
- 其中 num_tokens 通常是 729（27×27 的 patch 网格）

**特征分离**：
- 根据 `time_ids` 判断每个样本是否有历史帧
- 将特征分离为历史帧和当前帧两部分

**历史帧压缩**（Memory Token 生成）：
- 对于有历史帧的样本：
  - 提取前 `num_history` 帧的特征
  - 通过 MLP projector 投影到语言模型空间
  - 应用 2D 空间池化（`get_2d_pool`），stride = `history_stride`
  - 将多帧特征展平，生成 memory token 特征
- 对于没有历史帧的样本，memory_features 设为 None

**当前帧处理**：
- 提取当前帧的特征（剩余的帧）
- 通过 MLP projector 投影
- 应用 2D 空间池化，stride = `current_stride`
- 每帧生成压缩后的特征

**3.3 Token 替换**：
- 在 input_ids 中找到所有 `IMAGE_TOKEN_INDEX` 和 `MEMORY_TOKEN_INDEX` 的位置
- 用实际的视觉特征替换这些特殊 token：
  - `<image>` token → 当前帧的视觉特征（多 token）
  - `<memory>` token → 历史帧压缩后的特征（多 token）
- 构建新的 input_embeds，将文本 embedding 和视觉特征拼接

**3.4 标签对齐**：
- 调整 labels 的长度，使其与新的 input_embeds 对齐
- 视觉特征对应的位置在 labels 中设为 `IGNORE_INDEX`（不计算损失）

**3.5 语言模型前向传播**：
- 将处理后的 input_embeds 输入到 Qwen2VL 的语言模型部分
- 通过 Transformer 层进行前向传播
- 输出 logits：`(batch_size, seq_len, vocab_size)`

#### 步骤 4：损失计算

**交叉熵损失**：
- 使用 logits 和 labels 计算交叉熵损失
- 只对 labels 中非 `IGNORE_INDEX` 的位置计算损失（即 assistant 的回答部分）
- 如果使用梯度累积，损失会被累积

**损失缩放**（如果使用混合精度）：
- 如果启用 bfloat16，损失会被缩放以防止下溢

#### 步骤 5：反向传播

**梯度计算**：
- 调用 `loss.backward()` 计算梯度
- 只对未冻结的参数计算梯度
- 如果启用梯度检查点，会重新计算部分激活值以节省显存

**梯度累积**：
- 如果 `gradient_accumulation_steps > 1`，梯度会被累积
- 只有在累积步数达到设定值时才更新参数

#### 步骤 6：参数更新

**优化器步骤**：
- 如果达到累积步数，调用优化器的 `step()` 方法
- 应用梯度裁剪（如果配置）
- 更新模型参数

**学习率更新**：
- 学习率调度器更新学习率
- 根据配置的调度策略（如 cosine）调整学习率

**梯度清零**：
- 清零梯度，为下一步做准备

#### 步骤 7：日志记录

**指标记录**：
- 记录损失值、学习率等指标
- 如果启用 SwanLab，将指标发送到 SwanLab 服务器
- 定期打印训练进度

**检查点保存**：
- 根据 `save_steps` 配置，定期保存模型检查点
- 保存模型权重、优化器状态、训练状态等
- 如果启用 DeepSpeed，使用 DeepSpeed 的检查点格式

---

## 五、训练完成阶段

### 5.1 最终保存

**模型保存**：
- 训练完成后，保存最终的模型检查点
- 保存配置文件和 tokenizer
- 如果启用了 LoRA，保存 LoRA 权重

**清理工作**：
- 清理临时文件
- 关闭数据加载器
- 释放 GPU 内存

### 5.2 日志总结

**训练摘要**：
- 打印总训练步数、总训练时间
- 打印最终损失值
- 如果使用 SwanLab，提供实验链接

---

## 六、关键技术细节

### 6.1 32 帧窗口机制

**窗口切分**：
- 每个训练样本对应一个 32 帧的时间窗口
- 长轨迹会被切分为多个 32 帧的样本
- 窗口之间可能有重叠，取决于轨迹长度和切分策略

**时间对齐**：
- 每个样本的 `time_ids` 记录了该样本在原始轨迹中的帧索引
- 用于正确关联图像和动作序列

### 6.2 历史帧压缩

**Memory Token 生成**：
- 历史帧通过视觉编码器编码后，经过 MLP projector 投影
- 应用 2D 空间池化（stride=2），将 27×27 的特征图压缩为 13×13
- 多帧特征展平后形成 memory token 序列
- 这些 token 在对话中通过 `<memory>` 占位符插入

**压缩效果**：
- 原始历史帧：8 帧 × 729 tokens = 5832 tokens
- 压缩后：8 帧 × 169 tokens = 1352 tokens
- 压缩比约为 4.3:1

### 6.3 多轮对话构建

**动作分组**：
- 动作序列按照 `num_future_steps=4` 分组
- 每组对应一轮用户-助手对话
- 用户消息包含图像 token，助手消息包含动作序列

**对话示例**：
```
System: You are an autonomous navigation assistant. Your task is to [instruction]. 
        These are your historical observations: <memory>.

User: you can see <image>.
Assistant: ↑←→↑

User: in front of you is <image>.
Assistant: ←→↑↑

...
```

### 6.4 空间池化

**2D 池化操作**：
- 将图像特征从 `(num_frames, 729, hidden_dim)` reshape 为 `(num_frames, 27, 27, hidden_dim)`
- 转换为卷积格式：`(num_frames, hidden_dim, 27, 27)`
- 应用池化（average/max/bilinear），stride=2
- 输出：`(num_frames, hidden_dim, 13, 13)`
- 重新展平：`(num_frames, 169, hidden_dim)`

**池化模式**：
- `average`：平均池化，保留整体特征
- `max`：最大池化，保留显著特征
- `bilinear`：双线性插值，平滑降采样

### 6.5 梯度累积和分布式训练

**梯度累积**：
- 在显存有限时，使用小批次大小
- 通过梯度累积模拟大批次训练
- 例如：batch_size=4, grad_accum=2, 8 GPUs → 有效批次 = 64

**分布式训练**：
- 使用数据并行（Data Parallelism）
- 每个 GPU 处理不同的数据批次
- 梯度通过 NCCL 进行 AllReduce 同步
- DeepSpeed ZeRO-2 进一步优化显存使用

---

## 七、数据流图

```
原始数据 (annotations.json)
    ↓
导航轨迹加载
    ↓
32帧窗口切分
    ↓
历史帧采样 (8帧) + 当前帧采样
    ↓
图像加载和预处理
    ↓
动作序列分组 (每组4个动作)
    ↓
多轮对话构建
    ↓
Tokenization (添加 <image> 和 <memory> token)
    ↓
批处理 (Padding, Attention Mask)
    ↓
模型前向传播
    ├─ 视觉编码 (Vision Tower)
    ├─ 历史帧压缩 (Memory Token)
    ├─ 当前帧处理
    ├─ Token 替换 (文本 + 视觉特征)
    └─ 语言模型推理
    ↓
损失计算 (只计算 Assistant 部分)
    ↓
反向传播 (只更新未冻结参数)
    ↓
参数更新 (优化器 + 学习率调度)
    ↓
检查点保存
```

---

## 八、训练监控

### 8.1 SwanLab 集成

**实验跟踪**：
- 自动记录所有训练指标（损失、学习率等）
- 记录超参数配置
- 记录系统资源使用情况（GPU、内存）

**可视化**：
- 实时查看训练曲线
- 对比不同实验
- 导出训练报告

### 8.2 日志输出

**控制台输出**：
- 训练进度条
- 每步的损失值
- 定期打印训练状态

**文件日志**：
- 保存到输出目录
- 包含详细的训练信息
- 可用于调试和分析

---

## 九、性能优化

### 9.1 显存优化

**梯度检查点**：
- 在前向传播时不保存所有中间激活值
- 在反向传播时重新计算需要的激活值
- 以计算时间换取显存空间

**DeepSpeed ZeRO**：
- ZeRO-2：优化器状态分片
- ZeRO-3：模型参数分片
- 显著减少单卡显存占用

**混合精度训练**：
- 使用 bfloat16 进行前向和反向传播
- 保持 fp32 的优化器状态
- 加速训练并节省显存

### 9.2 计算优化

**数据加载并行**：
- 使用多个 worker 进程并行加载数据
- 减少数据加载的等待时间

**图像预处理优化**：
- 使用 Qwen2VLProcessor 的高效图像处理
- 批量处理图像

---

## 十、设计说明

### 10.1 task_type 的状态

**当前实现**：
- `task_type` 在数据集中被定义和返回（默认值为 0）
- `collate_fn` 收集 `task_type` 并包含在批处理字典中
- **但模型的前向传播方法不接收或使用 `task_type`**

**设计分析**：
- 原始 StreamVLN 中 `task_ids` 参数被传递到 `encode_rgbd` 和 `prepare_inputs_labels_for_multimodal`，但这些函数内部**从未使用**该参数
- ms-swift 版本优化了接口，移除了模型中未使用的参数（`task_ids`、`depths`、`poses`、`intrinsics`）
- 但在数据集层面保留了 `task_type` 的返回和收集，以便未来扩展

**保留原因**：
1. **扩展接口**：为未来多任务学习预留钩子
2. **无性能损失**：作为整数传递，开销可忽略
3. **兼容性**：便于实验不同的数据集组合

**未来扩展场景**：
- 支持多个 VLN 数据集的混合训练（如 R2R + REVERIE + SOON）
- 为不同数据集设置不同的 `task_id`
- 在模型中根据 `task_type` 选择不同的处理策略或任务特定的适配器
- 实现多任务联合训练以提升模型的泛化能力

**详细分析**：
- 完整的冗余分析见 `vln/doc/code_redundancy_analysis.md`

### 10.2 与原始 StreamVLN 的差异

**简化的参数**：
- ✅ 移除了 `depths`、`poses`、`intrinsics` 参数
  - 原因：原始版本加载但从不使用，造成不必要的 I/O 和内存开销
- ✅ 移除了模型接口中的 `task_ids` 参数
  - 原因：原始版本中接收但函数内部从未使用
- ✅ 保留了核心功能：`images`、`time_ids` 用于历史帧处理
- ✅ 移除了 `valid_idx` 机制
  - 原因：`clean_initial_rotations` 功能未实现，`valid_idx` 恒为 0

**保留的兼容性**：
- 数据集返回格式与原始版本一致
- `task_type` 在数据集层面保留，作为扩展接口
- 数据索引简化为三元组：`(ep_id, ins_id, start_idx)`

**优化总结**：
- 代码更简洁，移除了所有实质性冗余
- 接口更清晰，只保留实际使用的参数
- 保留必要的扩展点，便于未来增强
- 详细分析见 `vln/doc/code_redundancy_analysis.md`

## 十一、总结

StreamVLN 在 ms-swift 中的训练流程是一个高度集成和自动化的过程。框架负责处理分布式训练、优化器配置、检查点管理等复杂任务，用户只需关注数据准备和模型配置。整个流程从数据加载到模型保存都是端到端的，大大简化了训练过程，同时保持了灵活性和可扩展性。

关键优势：
1. **统一接口**：通过 `swift sft` 命令统一所有训练任务
2. **自动优化**：框架自动处理分布式、混合精度、梯度累积等
3. **易于扩展**：通过注册机制轻松添加新模型和数据集
4. **完整监控**：集成 SwanLab 等工具进行实验跟踪
5. **代码优化**：
   - ✅ 移除了未使用的参数（`depths`、`poses`、`intrinsics`、模型中的 `task_ids`）
   - ✅ 清理了冗余逻辑（`valid_idx`、重复检查）
   - ✅ 简化了数据索引结构（三元组）
   - ✅ 保留了必要的扩展点（数据集中的 `task_type`）
   - 详细分析见 `vln/doc/code_redundancy_analysis.md`

这使得研究人员可以专注于模型架构和数据，而不需要处理底层训练细节。
