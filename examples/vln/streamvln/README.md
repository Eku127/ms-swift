# StreamVLN 训练架构与流程详解

本文档详细介绍 StreamVLN 模块在 ms-swift 框架中的完整架构设计和训练流程。

---

## 一、模块架构概览

```
examples/vln/streamvln/
├── __init__.py        # 模块入口：模型注册、数据集注册、公共 API
├── arguments.py       # VLN 专属训练参数定义
├── dataset.py         # StreamVLN 数据集实现
├── model.py           # StreamVLN 模型（基于 Qwen2.5-VL）
└── trainer.py         # 自定义训练器入口
```

### 1.1 各文件职责

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `__init__.py` | 模块初始化，向 ms-swift 注册模型和数据集 | `register_model()`, `register_dataset()` |
| `arguments.py` | 定义 VLN 特有参数（如 `num_frames`, `num_history`） | `StreamVLNTrainArguments` |
| `dataset.py` | 加载 VLN 数据，构建多轮对话格式 | `StreamVLNDataset` |
| `model.py` | 扩展 Qwen2.5-VL，添加空间池化和历史编码 | `StreamVLNQwen25VLForConditionalGeneration` |
| `trainer.py` | 自定义 SFT 训练流程，适配 VLN 数据集 | `StreamVLNSft`, `train_main()` |

---

## 二、启动训练时发生了什么

当执行 `bash examples/vln/train_streamvln_qwen2_vl.sh` 时，整个流程如下：

### 阶段 1：脚本启动

```
┌─────────────────────────────────────────────────────────────────┐
│  train_streamvln_qwen2_vl.sh                                    │
│  ├── 设置环境变量（CUDA、NCCL 等）                               │
│  ├── 解析配置参数（VLN 参数、模型路径、训练超参等）              │
│  ├── 根据 NUM_GPUS 决定启动方式：                               │
│  │   ├── NUM_GPUS=1: python trainer.py ...                      │
│  │   └── NUM_GPUS>1: torchrun --nproc_per_node=N trainer.py ... │
│  └── 执行训练命令                                                │
└─────────────────────────────────────────────────────────────────┘
```

### 阶段 2：模块加载与注册

当 `trainer.py` 被执行时，首先触发 `__init__.py` 的导入：

```
┌─────────────────────────────────────────────────────────────────┐
│  __init__.py 加载过程                                           │
│                                                                  │
│  1. 导入 ms-swift 注册 API                                      │
│     from swift.llm import register_model, register_dataset      │
│                                                                  │
│  2. 注册自定义模型                                               │
│     register_model(ModelMeta(                                   │
│         model_type='streamvln_qwen2_5_vl',                      │
│         template=TemplateType.qwen2_5_vl,  # 复用 Qwen2.5-VL    │
│         get_function=get_model_tokenizer_streamvln_qwen2_5_vl,  │
│         is_multimodal=True,                                     │
│     ))                                                           │
│     ✓ 输出: [StreamVLN] Custom model registered successfully!  │
│                                                                  │
│  3. 注册自定义数据集加载器                                       │
│     register_dataset(DatasetMeta(...))                          │
│     ✓ 输出: [StreamVLN] Custom dataset registration loaded!    │
└─────────────────────────────────────────────────────────────────┘
```

### 阶段 3：StreamVLNSft 初始化

```
┌─────────────────────────────────────────────────────────────────┐
│  StreamVLNSft.__init__(args)                                    │
│                                                                  │
│  继承链: StreamVLNSft → SwiftSft → SwiftPipeline               │
│                                                                  │
│  1. 解析命令行参数 → StreamVLNTrainArguments                    │
│     - 标准训练参数（learning_rate, batch_size, ...）           │
│     - VLN 专属参数（num_frames=32, num_history=8, ...）        │
│                                                                  │
│  2. 设置随机种子                                                 │
│                                                                  │
│  3. 初始化 ProcessorMixin                                       │
│     - 加载 Tokenizer 和 Processor                               │
└─────────────────────────────────────────────────────────────────┘
```

### 阶段 4：main() 调用 run()

```python
# trainer.py
def train_main(args):
    return StreamVLNSft(args).main()  # 调用 SwiftPipeline.main()

# SwiftPipeline.main() 内部：
def main(self):
    result = self.run()  # 调用子类重写的 run()
    return result
```

---

## 三、核心训练流程 (run 方法)

`SwiftSft.run()` 是训练的核心，StreamVLNSft 通过重写关键方法来定制流程：

```
┌─────────────────────────────────────────────────────────────────┐
│  SwiftSft.run() 主流程                                          │
│                                                                  │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 1: _prepare_dataset()                                │  │
│  │   ├── _get_dataset()         ← StreamVLNSft 重写          │  │
│  │   ├── _encode_dataset()      ← StreamVLNSft 重写          │  │
│  │   ├── _post_process_datasets() ← StreamVLNSft 重写        │  │
│  │   └── _show_dataset()        ← StreamVLNSft 重写          │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 2: prepare_model()                                   │  │
│  │   ├── 模型加载（_prepare_model_tokenizer()）              │  │
│  │   │   - args.get_model_processor() 调用                 │  │
│  │   │   - 查询 MODEL_MAPPING → 找到注册的 get_function    │  │
│  │   │   - 调用 get_model_tokenizer_streamvln_qwen2_5_vl() │  │
│  │   │   - 返回 StreamVLNQwen25VLForConditionalGeneration    │  │
│  │   ├── 应用 PEFT（如 LoRA）                                │  │
│  │   └── 配置冻结组件（freeze_vit, freeze_llm）              │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 3: 创建 Trainer                                      │  │
│  │   trainer = Trainer(                                      │  │
│  │       model=self.model,                                   │  │
│  │       train_dataset=train_dataset,                        │  │
│  │       data_collator=data_collator,                        │  │
│  │       template=self.template,  # Qwen2.5-VL Template      │  │
│  │   )                                                        │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │ Step 4: self.train(trainer)                               │  │
│  │   └── trainer.train()  # HuggingFace Trainer              │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 四、数据集处理流程详解

这是 StreamVLNSft 最核心的定制部分：

### 4.1 _get_dataset()：创建数据集

```python
def _get_dataset(self):
    # 检测数据路径是否包含 annotations.json
    if os.path.exists(os.path.join(data_path, 'annotations.json')):
        # 创建 StreamVLNDataset（PyTorch Dataset）
        return StreamVLNDataset(
            data_path=data_path,
            num_frames=self.args.num_frames,      # 32
            num_history=self.args.num_history,    # 8
            num_future_steps=self.args.num_future_steps,  # 4
        ), None
    else:
        # 非 VLN 数据，使用父类逻辑
        return super()._get_dataset()
```

### 4.2 StreamVLNDataset.__getitem__()：构建样本

每个样本的构建过程：

```
┌─────────────────────────────────────────────────────────────────┐
│  输入: (episode_id, instruction_id, start_frame)               │
│                                                                  │
│  1. 加载导航数据                                                 │
│     - 读取 annotations.json                                     │
│     - 获取指令文本和动作序列                                     │
│                                                                  │
│  2. 采样帧                                                       │
│     ┌─────────────────────────────────────────────────────────┐ │
│     │ 历史帧（如果不是第一个窗口）                              │ │
│     │   - 从 [0, start_frame) 采样 num_history=8 帧            │ │
│     │   - 采样方式: uniform 或 random                          │ │
│     ├─────────────────────────────────────────────────────────┤ │
│     │ 当前帧                                                    │ │
│     │   - 从 [start_frame, start_frame+32) 每隔 4 帧采样       │ │
│     │   - 最多 8 帧                                            │ │
│     └─────────────────────────────────────────────────────────┘ │
│                                                                  │
│  3. 加载 PIL 图像                                                │
│     images = [历史帧...] + [当前帧...]                          │
│                                                                  │
│  4. 构建多轮对话（ms-swift 标准格式）                           │
│     messages = [                                                 │
│       {'role': 'system', 'content': '你是导航助手... <image>x8'},│
│       {'role': 'user', 'content': '你看到 <image>.'},           │
│       {'role': 'assistant', 'content': '↑↑←→'},                 │
│       {'role': 'user', 'content': '前方是 <image>.'},           │
│       {'role': 'assistant', 'content': '↑↑↑→'},                 │
│       ... (重复多轮)                                            │
│     ]                                                            │
│                                                                  │
│  输出: {'messages': [...], 'images': [PIL.Image, ...]}          │
└─────────────────────────────────────────────────────────────────┘
```

### 4.3 _encode_dataset()：跳过 HuggingFace 预处理

```python
def _encode_dataset(self, train_dataset, val_dataset, pre_process=True):
    if isinstance(train_dataset, StreamVLNDataset):
        # StreamVLNDataset 是 PyTorch Dataset，不需要 HF 预处理
        # 直接返回，不做任何转换
        return train_dataset, val_dataset
    return super()._encode_dataset(...)
```

**为什么跳过？** 
- 父类 `_encode_dataset` 针对 HuggingFace Dataset 做列映射、添加长度等操作
- `StreamVLNDataset` 是 PyTorch Dataset，已经返回正确格式，无需额外处理

### 4.4 _post_process_datasets()：包装为 LazyLLMDataset

这是最关键的一步，确保训练时能正确编码数据：

```python
def _post_process_datasets(self, datasets):
    for i, dataset in enumerate(datasets):
        if isinstance(dataset, StreamVLNDataset):
            # 用 LazyLLMDataset 包装
            datasets[i] = LazyLLMDataset(
                dataset,           # 原始 PyTorch Dataset
                template.encode,   # Qwen2.5-VL Template 的编码函数
                strict=args.strict,
                random_state=args.data_seed
            )
    return datasets
```

**LazyLLMDataset 的作用：**

```
┌─────────────────────────────────────────────────────────────────┐
│  LazyLLMDataset 工作原理                                        │
│                                                                  │
│  普通访问:                                                       │
│    dataset[i] → StreamVLNDataset.__getitem__(i)                │
│               → {'messages': [...], 'images': [...]}            │
│                                                                  │
│  训练时访问（通过 LazyLLMDataset）:                             │
│    lazy_dataset[i]                                               │
│      1. raw_data = dataset[i]  # 获取原始数据                   │
│      2. encoded = template.encode(raw_data)  # 编码为模型输入   │
│         ├── 文本 tokenize → input_ids                           │
│         ├── 图像处理 → pixel_values                             │
│         └── 生成 attention_mask, labels                         │
│      3. return encoded                                           │
│                                                                  │
│  优势: 延迟编码，节省内存，训练时按需处理                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 五、Template 编码流程

`template.encode()` 是 ms-swift 的核心，负责将对话+图像转换为模型输入：

```
┌─────────────────────────────────────────────────────────────────┐
│  template.encode(sample) 流程                                   │
│                                                                  │
│  输入:                                                           │
│    {                                                             │
│      'messages': [{'role': 'user', 'content': '...'},...],     │
│      'images': [PIL.Image, PIL.Image, ...]                      │
│    }                                                             │
│                                                                  │
│  处理步骤:                                                       │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 1. 图像预处理 (qwen_vl_utils.smart_resize)                  ││
│  │    - 调整图像尺寸到合适范围                                  ││
│  │    - 转换为 tensor: pixel_values                            ││
│  │    - 计算 image_grid_thw（时空网格信息）                    ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 2. 文本处理                                                  ││
│  │    - 应用聊天模板（Qwen2.5-VL 格式）                        ││
│  │    - Tokenize 文本 → input_ids                              ││
│  │    - <image> 替换为特殊 token (151655)                      ││
│  └─────────────────────────────────────────────────────────────┘│
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ 3. 生成 labels                                               ││
│  │    - 用户消息部分: IGNORE_INDEX (-100)                      ││
│  │    - 助手消息部分: 保留 token ID（计算 loss）               ││
│  └─────────────────────────────────────────────────────────────┘│
│                                                                  │
│  输出:                                                           │
│    {                                                             │
│      'input_ids': tensor([...]),                                │
│      'attention_mask': tensor([...]),                           │
│      'labels': tensor([...]),  # -100 表示不计算 loss           │
│      'pixel_values': tensor([...]),                             │
│      'image_grid_thw': tensor([...]),  # Qwen2.5-VL 特有       │
│    }                                                             │
└─────────────────────────────────────────────────────────────────┘
```

---

## 六、训练循环

最终，HuggingFace Trainer 执行训练循环：

```
┌─────────────────────────────────────────────────────────────────┐
│  Trainer.train() 训练循环                                       │
│                                                                  │
│  for epoch in range(num_epochs):                                │
│    for batch in DataLoader(train_dataset):                      │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 1. 从 LazyLLMDataset 获取已编码的 batch               │  │
│      │    - input_ids, attention_mask, labels                │  │
│      │    - pixel_values, image_grid_thw                     │  │
│      └───────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 2. 模型前向传播                                        │  │
│      │    outputs = model(                                    │  │
│      │        input_ids=input_ids,                           │  │
│      │        pixel_values=pixel_values,                     │  │
│      │        labels=labels,                                 │  │
│      │    )                                                   │  │
│      │    ├── Vision Encoder 处理图像                        │  │
│      │    ├── 图像特征与文本 embedding 融合                  │  │
│      │    └── LLM 生成 logits                                │  │
│      └───────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 3. 计算 Loss                                           │  │
│      │    - 只对 labels != -100 的位置计算 CrossEntropy      │  │
│      │    - 即只对助手回复部分（动作序列）计算损失           │  │
│      └───────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 4. 反向传播 & 更新                                     │  │
│      │    loss.backward()                                     │  │
│      │    optimizer.step()                                    │  │
│      └───────────────────────────────────────────────────────┘  │
│                          ↓                                       │
│      ┌───────────────────────────────────────────────────────┐  │
│      │ 5. 日志记录 & 检查点保存                               │  │
│      │    - 每 logging_steps 步记录 loss                     │  │
│      │    - 每 save_steps 步保存模型                         │  │
│      └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 七、模型加载机制详解

### 7.1 `args.get_model_processor()` 如何定位到自定义模型？

这是 ms-swift 注册机制的核心，完整流程如下：

```
┌─────────────────────────────────────────────────────────────────┐
│  步骤 1: 模型注册（__init__.py 模块加载时）                    │
│                                                                  │
│  register_model(ModelMeta(                                     │
│      model_type='streamvln_qwen2_5_vl',                         │
│      model_groups=[                                             │
│          ModelGroup([                                           │
│              Model('streamvln-qwen2.5-vl-3b',                  │
│                    'Qwen/Qwen2.5-VL-3B-Instruct'),             │
│          ])                                                     │
│      ],                                                         │
│      get_function=get_model_tokenizer_streamvln_qwen2_5_vl,    │
│  ))                                                             │
│       ↓                                                          │
│  MODEL_MAPPING['streamvln_qwen2_5_vl'] = ModelMeta(...)        │
│  ✓ 将 get_function 存储在 MODEL_MAPPING 字典中                  │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  步骤 2: 调用 get_model_processor()                            │
│                                                                  │
│  # BaseArguments.get_model_processor()                        │
│  def get_model_processor(self, ...):                           │
│      return get_model_tokenizer(                               │
│          model_id_or_path=self.model,  # 'Qwen/Qwen2.5-VL-3B' │
│          model_type=self.model_type,  # 'streamvln_qwen2_5_vl'│
│          ...                                                    │
│      )                                                          │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  步骤 3: get_model_tokenizer() 查找 ModelMeta                  │
│                                                                  │
│  def get_model_tokenizer(model_id_or_path, model_type, ...):   │
│      # 3.1 获取 ModelMeta                                      │
│      model_info, model_meta = get_model_info_meta(...)        │
│          ↓                                                      │
│      # 3.2 get_model_info_meta() 内部逻辑：                    │
│      if model_type is not None:                                │
│          # 方式 A: 通过 model_type 直接查找                   │
│          model_meta = MODEL_MAPPING[model_type]                │
│          # 找到: MODEL_MAPPING['streamvln_qwen2_5_vl']         │
│      else:                                                      │
│          # 方式 B: 通过 model_id 匹配                         │
│          model_meta = get_matched_model_meta(model_id)        │
│          # 遍历 MODEL_MAPPING，匹配 ModelGroup 中的 Model     │
│          # 如果 model_id 是 'Qwen/Qwen2.5-VL-3B-Instruct'     │
│          # 会匹配到 'streamvln-qwen2.5-vl-3b' → ModelGroup   │
│                                                                  │
│      # 3.3 从 ModelMeta 中获取 get_function                   │
│      get_function = model_meta.get_function                    │
│      # get_function = get_model_tokenizer_streamvln_qwen2_5_vl │
└─────────────────────────────────────────────────────────────────┘
                          ↓
┌─────────────────────────────────────────────────────────────────┐
│  步骤 4: 调用 get_function 加载模型                            │
│                                                                  │
│  model, processor = get_function(                              │
│      model_dir,      # 下载后的模型目录                        │
│      model_info,     # 模型信息（torch_dtype 等）             │
│      model_kwargs,   # 模型参数（device_map 等）               │
│      load_model,     # True                                    │
│      **kwargs                                                   │
│  )                                                              │
│       ↓                                                          │
│  # get_model_tokenizer_streamvln_qwen2_5_vl() 执行：          │
│  processor = AutoProcessor.from_pretrained(model_dir)          │
│  model = StreamVLNQwen25VLForConditionalGeneration.from_pretrained(...)│
│       ↓                                                          │
│  返回 (StreamVLNQwen25VLForConditionalGeneration, processor)   │
└─────────────────────────────────────────────────────────────────┘
```

**关键点**：

1. **注册时机**: `register_model()` 在 `__init__.py` 模块加载时执行（通过 `--custom_register_path` 触发）

2. **查找方式**:
   - **方式 A（推荐）**: 命令行指定 `--model_type streamvln_qwen2_5_vl`，直接通过 `MODEL_MAPPING[model_type]` 查找
   - **方式 B**: 不指定 `model_type`，通过 `model_id` 匹配 `ModelGroup` 中的 `Model` 列表

3. **ModelGroup 的作用**: 
   - 将多个模型 ID（如 `'Qwen/Qwen2.5-VL-3B-Instruct'`）映射到同一个 `model_type`
   - 支持通过模型 ID 自动匹配到对应的 `model_type`

4. **get_function 的作用**:
   - 存储在 `ModelMeta` 中，是实际加载模型的函数
   - 可以自定义加载逻辑（如 StreamVLN 返回自定义模型类）

---

## 八、关键设计决策

### 7.1 为什么重写 `_prepare_dataset()` 但不需要重写 `prepare_model()`？

这是 StreamVLN 架构的核心设计决策，原因如下：

#### **为什么 `_prepare_dataset()` 需要重写？**

**问题**: ms-swift 默认使用 HuggingFace Dataset 加载数据，但 VLN 数据格式特殊：
- 数据存储在 `annotations.json` + 视频帧文件夹中
- 需要自定义采样逻辑（历史帧、当前帧、动作序列）
- 需要构建多轮对话格式

**解决**: 重写 `_prepare_dataset()` 的四个子方法：
1. `_get_dataset()`: 检测 VLN 数据路径，创建 `StreamVLNDataset`
2. `_encode_dataset()`: 跳过 HuggingFace 预处理（PyTorch Dataset 不需要）
3. `_post_process_datasets()`: 包装为 `LazyLLMDataset`（确保能调用 `template.encode()`）
4. `_show_dataset()`: 自定义展示逻辑

#### **为什么 `prepare_model()` 不需要重写？**

**关键原因**: 模型加载已经通过**注册机制**完成，`prepare_model()` 只做**通用操作**。

**详细流程**:

```
1. 模型注册（__init__.py）
   register_model(ModelMeta(
       model_type='streamvln_qwen2_5_vl',
       get_function=get_model_tokenizer_streamvln_qwen2_5_vl,  ← 注册加载函数
   ))

2. 模型加载（SwiftSft._prepare_model_tokenizer()）
   args.get_model_processor()
       ↓
   get_model_tokenizer(model_id, model_type='streamvln_qwen2_5_vl')
       ↓
   查询 MODEL_MAPPING → 找到注册的 get_function
       ↓
   调用 get_model_tokenizer_streamvln_qwen2_5_vl()
       ↓
   返回 (StreamVLNQwen25VLForConditionalGeneration, processor)
       ↓
   self.model 已经是自定义模型实例！

3. prepare_model() 执行（TunerMixin.prepare_model()）
   - 接收的 model 参数已经是 StreamVLNQwen25VLForConditionalGeneration
   - 只做通用操作：
     * 应用 PEFT（LoRA 等）
     * 冻结参数（freeze_vit, freeze_llm）
     * 设置训练模式
   - 不涉及模型特定逻辑，所以不需要重写
```

**总结**:
- **数据集**: 格式特殊，需要自定义加载逻辑 → **必须重写**
- **模型**: 通过注册机制加载，`prepare_model()` 只做通用操作 → **无需重写**

### 7.2 为什么重写 `_get_dataset()`？

**问题**: ms-swift 默认使用 HuggingFace Dataset 加载数据，但 VLN 数据格式特殊（包含视频帧、动作序列等）。

**解决**: 检测 `annotations.json` 存在时，直接创建 `StreamVLNDataset`（PyTorch Dataset）。

### 7.3 为什么重写 `_encode_dataset()`？

**问题**: 父类 `_encode_dataset()` 对 HuggingFace Dataset 做列映射等预处理。

**解决**: `StreamVLNDataset` 已经返回正确格式，跳过 HF 预处理避免报错。

### 7.4 为什么重写 `_post_process_datasets()`？

**问题**: 父类有复杂的判断逻辑（streaming、packing 等），可能跳过 `LazyLLMDataset` 包装。

**解决**: 显式为 `StreamVLNDataset` 创建 `LazyLLMDataset` 包装，确保训练时能调用 `template.encode()`。

### 7.5 为什么使用 LazyLLMDataset？

**优势**:
1. **延迟编码**: 训练时按需编码，不需要预先处理所有数据
2. **内存效率**: 不需要将所有编码结果存储在内存中
3. **灵活性**: 每次访问可以应用不同的数据增强

### 7.6 为什么复用 Qwen2.5-VL Template？

**优势**:
1. 无需重写复杂的图像处理逻辑（smart_resize、patch embedding 等）
2. 无需重写文本 tokenization 和聊天模板
3. 只需要数据集返回 ms-swift 标准格式，框架自动处理其余部分

---

## 八、数据流总结

```
annotations.json
       ↓
StreamVLNDataset.__getitem__(i)
       ↓
{'messages': [...], 'images': [PIL.Image, ...]}  ← ms-swift 标准格式
       ↓
LazyLLMDataset 访问时调用 template.encode()
       ↓
{'input_ids': tensor, 'pixel_values': tensor, 'labels': tensor, ...}
       ↓
DataLoader 批处理
       ↓
Model.forward(input_ids, pixel_values, labels)
       ↓
Loss 计算 & 反向传播
       ↓
模型更新
```

---

## 九、分布式训练

### 单 GPU
```bash
NUM_GPUS=1
python trainer.py --args...
```

### 多 GPU（DDP）
```bash
NUM_GPUS=4
torchrun --nproc_per_node=4 trainer.py --args...
```

**注意**: 必须使用 `torchrun` 而非 `python` 启动多 GPU 训练，否则会使用 `DataParallel`（与 Qwen2.5-VL 的 3D position embeddings 不兼容）。

---

## 十、常见问题

### Q1: 为什么不直接修改 ms-swift 源码？

**A**: 非侵入式设计。通过 `register_model()` 和 `register_dataset()` API 注册自定义组件，保持 ms-swift 核心代码不变，便于升级和维护。

### Q2: Template 是如何知道有多少图像的？

**A**: Template 扫描 `messages` 中的 `<image>` token 数量，与 `images` 列表一一对应。

### Q3: 如何添加新的 VLN 参数？

**A**: 在 `StreamVLNTrainArguments` 中添加 `dataclass` 字段，然后在 `StreamVLNDataset` 或 `StreamVLNSft` 中使用。

---

## 十一、扩展建议

1. **自定义 Loss**: 重写 `Trainer.compute_loss()` 添加 VLN 特定损失（如轨迹平滑度）
2. **自定义评估**: 重写 `Trainer.evaluate()` 添加 VLN 评估指标（如 SR、SPL）
3. **自定义模型**: 修改 `model.py` 中的 `StreamVLNQwen25VLForConditionalGeneration`

