# StreamVLN 调试指南 - 理解训练流程

本指南帮助初学者通过断点调试理解 StreamVLN 的完整训练流程。

---

## 一、训练流程总览

```
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 1: 启动                                                     │
│   trainer.py: train_main() → StreamVLNSft().main()              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 2: 数据集创建                                               │
│   trainer.py: _get_dataset() → StreamVLNDataset()               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 3: 数据集包装                                               │
│   trainer.py: _post_process_datasets() → LazyLLMDataset()       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 阶段 4: 训练循环（每个 batch）                                    │
│   4.1 数据获取: dataset.__getitem__() → {'messages', 'images'}  │
│   4.2 Template编码: template.encode() → pixel_values, input_ids │
│   4.3 模型前向: model.forward(inputs_embeds) → loss             │
│   4.4 反向传播: loss.backward() → 参数更新                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 二、推荐断点位置

### 断点 1: 训练入口 ⭐⭐⭐

**文件**: `examples/vln/streamvln/trainer.py`
**行号**: 165
**位置**: `train_main()` 函数

```python
def train_main(args: Optional[Union[List[str], StreamVLNTrainArguments]] = None):
    """Main entry point for StreamVLN training."""
    return StreamVLNSft(args).main()  # ← 在这里设断点
```

**观察什么**:
- `args` 参数内容（所有训练配置）
- 进入 `StreamVLNSft` 初始化过程

---

### 断点 2: 数据集创建 ⭐⭐⭐

**文件**: `examples/vln/streamvln/trainer.py`
**行号**: 78
**位置**: `_get_dataset()` 方法中创建 `StreamVLNDataset`

```python
train_dataset = StreamVLNDataset(  # ← 在这里设断点
    data_path=data_path,
    num_frames=self.args.num_frames,
    ...
)
```

**观察什么**:
- `data_path` - 数据路径
- `self.args.num_frames` 等 VLN 参数
- 步进进入 `StreamVLNDataset.__init__()` 看初始化过程

---

### 断点 3: 数据集包装（关键！）⭐⭐⭐⭐⭐

**文件**: `examples/vln/streamvln/trainer.py`
**行号**: 120
**位置**: `_post_process_datasets()` 中创建 `LazyLLMDataset`

```python
datasets[i] = LazyLLMDataset(
    dataset,           # ← StreamVLNDataset
    template.encode,   # ← 这是关键！Template 的编码函数
    strict=args.strict,
    random_state=args.data_seed
)
```

**观察什么**:
- `template` - 这是 Qwen2.5-VL 的 Template 对象
- `template.encode` - 这个函数负责将 messages+images 转换为模型输入

---

### 断点 4: 数据样本获取 ⭐⭐⭐⭐⭐

**文件**: `examples/vln/streamvln/dataset.py`
**行号**: 148
**位置**: `StreamVLNDataset.__getitem__()`

```python
def __getitem__(self, i) -> Dict[str, Any]:  # ← 在这里设断点
    """Get a training sample in ms-swift standard format."""
    ep_id, ins_id, start_idx = self.data_list[i]
    ...
```

**观察什么**:
- `i` - 样本索引
- `ep_id, ins_id, start_idx` - 轨迹分段信息
- 最后返回的 `{'messages': [...], 'images': [...]}`

**重点观察返回值**:
```python
return {
    'messages': messages,  # 查看对话结构
    'images': images,      # 查看 PIL 图像列表
}
```

---

### 断点 5: Template 编码（最重要！）⭐⭐⭐⭐⭐

这个断点需要在 ms-swift 框架内部设置：

**文件**: `swift/llm/template/base.py` (大约第 400-500 行)
**搜索**: `def encode(` 方法

或者更简单的方法，在 `LazyLLMDataset` 调用 encode 的地方：

**文件**: `swift/llm/dataset/dataset.py`
**搜索**: `class LazyLLMDataset` 的 `__getitem__` 方法

**观察什么**:
- 输入: `{'messages': [...], 'images': [PIL.Image, ...]}`
- 输出: `{'input_ids': tensor, 'pixel_values': tensor, 'labels': tensor, ...}`

---

## 三、设置断点的方法

### 方法 1: 在 Cursor/VS Code 中点击

1. 打开对应文件
2. 在行号左侧点击，出现红点
3. 按 F5 启动调试

### 方法 2: 在代码中添加 `breakpoint()`

```python
def __getitem__(self, i):
    breakpoint()  # 添加这行，程序会在这里暂停
    ep_id, ins_id, start_idx = self.data_list[i]
    ...
```

---

## 四、调试操作指南

### 常用快捷键

| 操作 | 快捷键 | 说明 |
|------|--------|------|
| 继续执行 | F5 | 运行到下一个断点 |
| 单步跳过 | F10 | 执行当前行，不进入函数内部 |
| 单步进入 | F11 | 进入函数内部 |
| 跳出函数 | Shift+F11 | 从当前函数返回 |
| 重启调试 | Ctrl+Shift+F5 | 重新开始 |

### 在调试控制台查看变量

在 "DEBUG CONSOLE" 中可以输入 Python 表达式：

```python
# 查看 messages 结构
sample['messages']

# 查看图像数量
len(sample['images'])

# 查看第一张图像尺寸
sample['images'][0].size

# 查看 input_ids 形状
encoded['input_ids'].shape
```

---

## 五、建议的调试流程

### 第一次调试：理解数据流

1. **断点 4** (`dataset.py:148`) - 看原始数据如何变成 messages + images
2. **观察返回的 dict**，理解 ms-swift 期望的格式

### 第二次调试：理解 Template 编码

1. **断点 3** (`trainer.py:120`) - 看 LazyLLMDataset 如何包装
2. 使用 F11 进入 `LazyLLMDataset.__getitem__()` 
3. 观察 `template.encode()` 如何将 messages+images 转换为 tensor

### 第三次调试：理解完整流程

1. **断点 1** (`trainer.py:165`) - 从头开始
2. 逐步执行，观察整个训练初始化过程

---

## 六、重点变量说明

### StreamVLNDataset.__getitem__() 返回值

```python
{
    'messages': [
        {'role': 'system', 'content': 'You are an autonomous navigation assistant...'},
        {'role': 'user', 'content': 'you can see <image>.'},
        {'role': 'assistant', 'content': '↑↑←→'},
        {'role': 'user', 'content': 'in front of you is <image>.'},
        {'role': 'assistant', 'content': '↑↑↑→'},
        ...
    ],
    'images': [PIL.Image, PIL.Image, ...]  # 每个 <image> 对应一张图
}
```

### Template.encode() 输出

```python
{
    'input_ids': tensor([...]),        # 文本 token IDs
    'attention_mask': tensor([...]),   # 注意力掩码
    'labels': tensor([...]),           # 训练标签 (-100 表示不计算 loss)
    'pixel_values': tensor([...]),     # 图像像素值
    'image_grid_thw': tensor([...]),   # 图像网格信息 (Qwen2.5-VL 特有)
}
```

---

## 七、常见问题

### Q: 为什么我的断点没有被触发？

A: 检查 launch.json 中的 `dataloader_num_workers` 是否为 0。多进程数据加载会导致断点在子进程中无法工作。

### Q: 如何只调试一个训练步？

A: 在 launch.json 中设置 `--max_steps 1` 和 `--vln_max_samples 1`

### Q: 如何查看模型实际收到了什么？

A: 可以在 `Qwen2_5_VLForConditionalGeneration.forward()` 设断点，但这需要修改 transformers 源码或使用 monkey patch。更简单的方法是在 Template.encode 输出后观察。

