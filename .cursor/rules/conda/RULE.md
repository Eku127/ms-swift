---
alwaysApply: true
---

# Conda 环境规则

**重要提示：** 所有与 ms-swift 相关的安装、运行和开发操作都必须在 `swift-vln` conda 环境中执行。

## 使用指南

1. **激活环境**：在执行任何命令前，确保已激活 conda 环境
   ```bash
   conda activate swift-vln
   ```

2. **适用范围**：
   - 安装 Python 包（pip install）
   - 运行 swift 命令（swift sft, swift infer 等）
   - 运行 Python 脚本
   - 执行测试和开发相关操作

3. **检查环境**：在执行命令前，可以通过以下方式确认当前环境
   ```bash
   conda info --envs  # 查看所有环境
   echo $CONDA_DEFAULT_ENV  # 查看当前激活的环境
   ```

## 注意事项

- 如果命令执行失败，首先检查是否在正确的 conda 环境中
- 安装新包时，确保 `swift-vln` 环境已激活
- 避免在系统 Python 或其他 conda 环境中执行 ms-swift 相关操作
- **除非用户明确要求，否则不要自动生成 markdown 文档文件**。优先使用简洁的文本回复或直接在终端中展示结果
