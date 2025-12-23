# StreamVLN 数据格式说明

本文档说明 StreamVLN 项目中使用的视觉语言导航（VLN）数据集的格式和组织方式。

## 目录结构

```
trajectory_data/
├── R2R/
│   ├── annotations.json          # 所有轨迹的元数据
│   └── images/                   # 图像序列目录
│       └── {scene_id}_{dataset}_{episode_id:06d}/
│           └── rgb/
│               ├── 001.jpg
│               ├── 002.jpg
│               └── ...
├── RxR_new/
│   └── ...
└── EnvDrop/
    └── ...
```

## 数据生成流程

### 1. 数据来源

数据通过 Habitat 仿真环境生成，使用以下组件：
- **Habitat 环境**：3D 室内场景仿真器
- **ShortestPathFollower**：最短路径跟随器，用于生成专家轨迹
- **参考路径（reference_path）**：来自原始 VLN 数据集的标注路径点序列

### 2. 生成过程

#### 初始化阶段
- 加载 Habitat 环境和配置
- 遍历所有 episode（每个 episode 包含一个导航任务）
- 为每个 episode 创建 ShortestPathFollower 代理

#### 轨迹生成循环
对每个 episode 执行以下步骤：

1. **获取任务信息**
   - 导航指令（instructions）：自然语言描述的目标位置
   - 参考路径（reference_path）：3D 坐标点序列，表示从起点到终点的最优路径

2. **重置环境**
   - 将智能体放置在起始位置
   - 获取初始观察（RGB 图像）

3. **逐步导航**
   对于每个时间步：
   - **保存当前图像**：将 RGB 观察保存为 JPG 文件
   - **计算下一个动作**：使用 ShortestPathFollower 计算到下一个路径点的动作
   - **处理停止动作**：如果动作是 STOP(0)，移动到下一个路径点
   - **执行动作**：在环境中执行动作
   - **记录动作**：将动作添加到动作序列中

4. **数据保存**
   - 图像序列保存在 `images/{scene_id}_{dataset}_{episode_id}/rgb/` 目录
   - 元数据（ID、路径、指令、动作序列）保存在 `annotations.json`

## 数据格式

### annotations.json 格式

每个样本包含以下字段：

```json
{
  "id": 1803,                    // episode ID
  "video": "images/scene_xxx_r2r_001803",  // 图像目录的相对路径
  "instructions": ["Go to the kitchen"],   // 导航指令（列表格式，支持多语言）
  "actions": [-1, 1, 1, 2, 1, 0]           // 动作序列
}
```

### 动作编码

- **-1**：初始占位符，表示起始状态
- **0**：STOP（停止/等待）
- **1**：MOVE_FORWARD（前进 25 厘米）
- **2**：TURN_LEFT（左转 15 度）
- **3**：TURN_RIGHT（右转 15 度）

### 图像-动作对应关系

#### 代码执行时间线

让我们通过代码执行的时间线来理解图像和动作的对应关系：

**初始化阶段：**
```
actions = [-1]           # 初始动作占位符
rgb_list = []           # 图像列表为空
observation = env.reset()  # 获取初始观察（此时还没有保存图像）
```

**第1次循环：**
```
步骤1: rgb = observation["rgb"]              # 获取初始观察
步骤2: rgb_list.append(rgb)                 # rgb_list长度 = 1
步骤3: 保存为 001.jpg                       # 文件名基于len(rgb_list)=1
步骤4: next_action = agent.get_next_action() # 计算要执行的动作（假设是1）
步骤5: observation = env.step(next_action)   # 执行动作1，得到新观察
步骤6: actions.append(next_action)           # actions = [-1, 1]
```
**此时状态：**
- `001.jpg` 已保存（初始观察）
- `actions = [-1, 1]`
- `observation` 是执行动作1后的新观察

**第2次循环：**
```
步骤1: rgb = observation["rgb"]              # 获取执行动作1后的观察
步骤2: rgb_list.append(rgb)                 # rgb_list长度 = 2
步骤3: 保存为 002.jpg                       # 文件名基于len(rgb_list)=2
步骤4: next_action = agent.get_next_action() # 计算下一个动作（假设是2）
步骤5: observation = env.step(next_action)   # 执行动作2，得到新观察
步骤6: actions.append(next_action)           # actions = [-1, 1, 2]
```
**此时状态：**
- `002.jpg` 已保存（执行动作1后的观察）
- `actions = [-1, 1, 2]`
- `observation` 是执行动作2后的新观察

**第3次循环：**
```
步骤1: rgb = observation["rgb"]              # 获取执行动作2后的观察
步骤2: rgb_list.append(rgb)                 # rgb_list长度 = 3
步骤3: 保存为 003.jpg                       # 文件名基于len(rgb_list)=3
步骤4: next_action = agent.get_next_action() # 计算下一个动作
步骤5: observation = env.step(next_action)   # 执行动作
步骤6: actions.append(next_action)           # actions = [-1, 1, 2, ...]
```

#### 关键理解

**核心结论：一张图片对应的标签就是它下一步应该执行的action**

**问题1：每一张图像和action是怎么对应的？**

答案：**当前图像用于预测下一个动作**（即：图像是输入，下一步动作是标签）

| 图像 | 图像含义 | 用于预测的动作 | 动作索引 |
|------|---------|--------------|---------|
| `001.jpg` | 初始观察（执行动作前） | `actions[1]` | 索引1 |
| `002.jpg` | 执行 `actions[1]` 后的观察 | `actions[2]` | 索引2 |
| `003.jpg` | 执行 `actions[2]` 后的观察 | `actions[3]` | 索引3 |
| `{i:03d}.jpg` | 执行 `actions[i-1]` 后的观察 | `actions[i]` | 索引i |

**问题2：图像和action之间的索引是不是不一样？**

答案：**是的，索引不一样，存在偏移**

- **图像索引**：从 1 开始（`001.jpg`, `002.jpg`, `003.jpg`, ...）
- **动作索引**：从 0 开始（`actions[0]`, `actions[1]`, `actions[2]`, ...）
- **对应关系**：
  - 图像 `{i:03d}.jpg`（i≥1）用于预测 `actions[i]`
  - 图像 `{i:03d}.jpg`（i≥1）是执行 `actions[i-1]` 后的观察结果

#### 示例说明

假设最终生成的数据：
- 图像：`001.jpg`, `002.jpg`, `003.jpg`
- 动作：`actions = [-1, 1, 2, 0]`

对应关系（训练时的输入-标签对）：
- **输入**：`001.jpg`（初始观察）→ **标签**：`actions[1] = 1`（下一步动作）
- **输入**：`002.jpg`（执行动作1后的观察）→ **标签**：`actions[2] = 2`（下一步动作）
- **输入**：`003.jpg`（执行动作2后的观察）→ **标签**：`actions[3] = 0`（下一步动作）

**总结**：每张图像对应的标签就是它下一步应该执行的action。

注意：`actions[0] = -1` 是初始占位符，不对应任何图像，因为它在保存第一张图像之前就已经存在了。

### 关键设计点

1. **图像先于动作保存**：在计算和执行动作之前保存当前观察，确保图像反映执行动作前的状态
2. **动作偏移**：训练时，模型根据当前图像预测下一个动作，因此需要将动作序列向后偏移一位
3. **路径点跟踪**：使用 `next_waypoint_id` 跟踪当前目标路径点，当到达路径点时移动到下一个
4. **动态目标半径**：接近终点时，将目标半径从 0.5 米减小到 0.25 米，提高导航精度

## 数据集统计

### R2R 数据集
- 样本数量：约 10,819 个轨迹
- 图像命名：从 `001.jpg` 开始
- 平均轨迹长度：约 30-50 步

### RxR_new 数据集
- 样本数量：约 19,990 个轨迹
- 图像命名：从 `001.jpg` 开始
- 平均轨迹长度：约 50-100 步

### EnvDrop 数据集
- 样本数量：约 146,304 个轨迹
- 图像命名：从 `000.jpg` 开始
- 平均轨迹长度：约 20-50 步

## 训练时的数据处理

### 数据切分
- 将长轨迹按 `num_frames`（如 32 帧）切分成多个训练样本
- 每个训练样本包含一段连续的图像序列和对应的动作序列

### 历史帧采样
- 支持从历史轨迹中采样 `num_history` 帧作为上下文
- 采样策略：随机采样或均匀采样

### 未来动作预测
- 根据 `num_future_steps` 参数，预测未来 N 步动作
- 动作序列转换为文本格式（如 "↑←→"）作为模型输出

### 对话格式构建
- 使用类似 LLaVA 的对话格式
- 包含图像 token (`<image>`) 和记忆 token (`<memory>`)
- 提示词模板：`"You are an autonomous navigation assistant. Your task is to <instruction>..."`

## 与 ms-swift 的适配

在 ms-swift 框架中使用这些数据时，需要：

1. **数据格式转换**：将 StreamVLN 格式转换为 ms-swift 支持的 JSONL 格式
2. **自定义数据集类**：实现类似 `VLNActionDataset` 的数据加载逻辑
3. **多图像处理**：支持在一个样本中处理多帧图像（历史帧 + 当前帧）
4. **动作序列生成**：模型输出动作文本序列，需要转换为可执行的动作

## 注意事项

1. **图像数量 = 动作数量**：每个动作都对应一帧图像（包括初始的 -1）
2. **动作偏移**：训练时，当前图像预测下一个动作，需要正确处理索引偏移
3. **路径点处理**：当动作是 STOP(0) 时，需要移动到下一个路径点
4. **文件命名差异**：不同数据集的图像文件命名起始索引可能不同（000 vs 001）

## 参考

- StreamVLN 数据生成脚本：`streamvln/streamvln_trajectory_generation.py`
- StreamVLN 数据加载类：`streamvln/dataset/vln_action_dataset.py`
- Habitat 文档：https://aihabitat.org/
