# StreamVLN 训练模块

本模块提供基于 Qwen2.5-VL 的视觉语言导航（VLN）训练支持，完全利用 ms-swift 框架的原生多模态处理能力。

---

## 一、模块架构

```
examples/vln/streamvln/
├── __init__.py        # 模块入口：模型注册
├── arguments.py       # VLN 专属训练参数（用于数据集构建）
├── dataset.py         # StreamVLN 数据集实现
├── model.py           # 模型定义（继承 Qwen2.5-VL）
├── trainer.py         # 自定义训练器入口
└── script/            # 训练脚本目录
    └── train_streamvln_qwen2_vl.sh  # 训练启动脚本
```

### 各文件职责

| 文件 | 职责 | 关键类/函数 |
|------|------|-------------|
| `__init__.py` | 向 ms-swift 注册模型 | `register_model()` |
| `arguments.py` | 定义 VLN 数据集参数 | `StreamVLNTrainArguments` |
| `dataset.py` | 加载 VLN 数据，构建多轮对话 | `StreamVLNDataset` |
| `model.py` | 模型定义（直接继承 Qwen2.5-VL） | `StreamVLNQwen25VLForConditionalGeneration` |
| `trainer.py` | 自定义 SFT 训练流程 | `StreamVLNSft`, `train_main()` |

---

## 二、核心设计：使用 Qwen2.5-VL 原生处理

### 架构说明

StreamVLN 模型**直接继承** `Qwen2_5_VLForConditionalGeneration`，不重写任何方法：

```python
class StreamVLNQwen25VLForConditionalGeneration(Qwen2_5_VLForConditionalGeneration):
    """直接使用 Qwen2.5-VL 的原生多模态处理"""
    config_class = StreamVLNQwen25VLConfig
    # 不重写 forward() 等方法
```

### 为什么这样设计？

ms-swift 框架使用 Qwen2.5-VL 的原生 Template 处理多模态输入：

1. **Template 预处理**：将 PIL 图像转换为 `pixel_values` 和 `image_grid_thw`
2. **视觉编码**：Qwen2.5-VL 内部处理图像特征提取
3. **多模态融合**：框架在调用 `forward()` 前完成文本和图像的融合
4. **模型接收**：`forward()` 收到的是已融合的 `inputs_embeds`

### 数据流

```
StreamVLNDataset.__getitem__()
       ↓
{'messages': [...], 'images': [PIL.Image, ...]}  ← ms-swift 标准格式
       ↓
Template.encode() (Qwen2.5-VL Template)
       ↓
pixel_values + input_ids → 融合成 inputs_embeds
       ↓
model.forward(inputs_embeds=已融合结果)
       ↓
Loss 计算 & 反向传播
```

---

## 三、数据集格式

### 数据目录结构

```
/path/to/vln_data/
├── annotations.json          # 导航标注文件
└── episode_001/
    └── rgb/
        ├── 000000.jpg
        ├── 000001.jpg
        └── ...
```

### annotations.json 格式

```json
[
  {
    "video": "episode_001",
    "instructions": ["Go to the kitchen and find the refrigerator"],
    "actions": [1, 1, 2, 1, 1, 3, 1, 0]
  }
]
```

动作定义：
- `0`: STOP
- `1`: MOVE_FORWARD (↑)
- `2`: TURN_LEFT (←)
- `3`: TURN_RIGHT (→)

### StreamVLNDataset 返回格式

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
    'images': [PIL.Image, PIL.Image, ...]
}
```

---

## 四、训练参数

### VLN 专属参数（用于数据集构建）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_frames` | 32 | 轨迹窗口大小 |
| `num_history` | 8 | 历史帧采样数量 |
| `num_future_steps` | 4 | 每轮预测的动作数 |
| `use_random` | False | 历史帧采样方式（False=均匀采样） |
| `vln_max_samples` | None | 限制训练样本数量 |

### 模型冻结参数

| 参数 | 说明 |
|------|------|
| `freeze_vit` | 冻结视觉编码器 |
| `freeze_llm` | 冻结语言模型 |
| `freeze_aligner` | 冻结多模态对齐层 |

---

## 五、使用方法

### 训练命令

```bash
bash examples/vln/streamvln/script/train_streamvln_qwen2_vl.sh
```

### 主要配置项

编辑 `script/train_streamvln_qwen2_vl.sh` 修改：

```bash
# 数据路径
VLN_DATA_PATH="/path/to/your/vln_data"

# 模型配置
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"

# 训练参数
NUM_EPOCHS=1
LEARNING_RATE=2e-5
BATCH_SIZE=1

# 模型冻结
FREEZE_VIT=true    # 冻结视觉编码器
FREEZE_LLM=true    # 冻结语言模型
FREEZE_ALIGNER=false  # 训练对齐层
```

### 调试模式

```bash
# 快速测试（1个样本，1步）
python examples/vln/streamvln/trainer.py \
    --custom_register_path examples/vln/streamvln \
    --model_type streamvln_qwen2_5_vl \
    --model Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset /path/to/vln_data \
    --vln_max_samples 1 \
    --max_steps 1
```

---

## 六、训练流程详解

### 启动时

```
script/train_streamvln_qwen2_vl.sh
       ↓
python trainer.py --custom_register_path examples/vln/streamvln ...
       ↓
__init__.py 加载 → register_model() 注册模型
       ↓
StreamVLNSft 初始化
       ↓
_get_dataset() → 创建 StreamVLNDataset
       ↓
_post_process_datasets() → LazyLLMDataset 包装
       ↓
Trainer.train() 开始训练
```

### 每个训练步

```
LazyLLMDataset[i]
       ↓
StreamVLNDataset.__getitem__(i) → {'messages': [...], 'images': [...]}
       ↓
template.encode() → pixel_values, input_ids, labels
       ↓
DataLoader 批处理
       ↓
model.forward(inputs_embeds=..., labels=...)  # 框架已完成融合
       ↓
Loss 计算 → 反向传播 → 参数更新
```

---

## 七、扩展指南

### 添加新的 VLN 参数

在 `arguments.py` 中添加：

```python
@dataclass
class StreamVLNTrainArguments(TrainArguments):
    my_new_param: int = field(
        default=10,
        metadata={"help": "Description of parameter"}
    )
```

### 修改数据集格式

在 `dataset.py` 的 `__getitem__()` 中修改返回的 `messages` 格式。

### 自定义模型逻辑

如果需要自定义视觉处理（如空间池化），需要：
1. 重写 `forward()` 方法
2. 创建自定义 Template 来绕过框架预处理
3. 或者在框架处理流程中找到合适的 hook 点

---

## 八、常见问题

### Q1: 为什么模型直接继承不重写任何方法？

A: ms-swift 框架在调用模型 `forward()` 之前，已经通过 Qwen2.5-VL Template 完成了所有多模态处理（图像预处理、视觉编码、文本-图像融合）。模型收到的是已融合的 `inputs_embeds`，不需要再处理原始图像。

### Q2: 如何添加空间池化等自定义视觉处理？

A: 需要深入框架找到视觉处理的 hook 点，或创建自定义 Template。这是一个较复杂的改动，建议先使用原生处理验证基本功能。

### Q3: Template 是如何处理图像的？

A: Qwen2.5-VL Template 执行：
1. `smart_resize()` 调整图像尺寸
2. 转换为 `pixel_values` tensor
3. 计算 `image_grid_thw`（时空网格信息）
4. 通过视觉编码器提取特征
5. 与文本 embedding 融合

### Q4: 数据集为什么返回 PIL.Image 而不是 tensor？

A: ms-swift 标准格式要求返回 PIL.Image，Template 会自动处理预处理。这保持了与框架其他多模态数据集的一致性。

---

## 九、参考资源

- [ms-swift 官方文档](https://github.com/modelscope/ms-swift)
- [Qwen2.5-VL 模型](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct)
- [原始 StreamVLN 论文/代码](https://github.com/your-repo/StreamVLN)
