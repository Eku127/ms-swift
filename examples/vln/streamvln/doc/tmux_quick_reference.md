# Tmux 快速参考卡片

## 🚀 快速开始

```bash
tmux new -s training          # 创建名为 training 的会话
tmux attach -t training      # 连接到 training 会话
tmux ls                       # 列出所有会话
Ctrl-a d                      # 断开当前会话（会话继续运行）
```

## ⌨️ 快捷键速查

### 前缀键
- **前缀键**: `Ctrl-a` (所有命令都需要先按这个)

### 会话操作
| 快捷键 | 功能 |
|--------|------|
| `Ctrl-a d` | 断开会话（后台继续运行） |
| `Ctrl-a s` | 列出并切换会话 |
| `Ctrl-a $` | 重命名当前会话 |

### 窗口操作
| 快捷键 | 功能 |
|--------|------|
| `Ctrl-a c` | 创建新窗口 |
| `Ctrl-a n` | 下一个窗口 |
| `Ctrl-a p` | 上一个窗口 |
| `Ctrl-a 0-9` | 切换到指定窗口 |
| `Ctrl-a ,` | 重命名当前窗口 |
| `Ctrl-a &` | 关闭当前窗口 |

### 窗格操作
| 快捷键 | 功能 |
|--------|------|
| `Ctrl-a v` | 垂直分割（左右） |
| `Ctrl-a h` | 水平分割（上下） |
| `Ctrl-a x` | 关闭当前窗格 |
| `Ctrl-a z` | 最大化/恢复窗格 |
| `Alt-←→↑↓` | **直接切换窗格**（无需前缀键） |
| `Ctrl-a H/J/K/L` | 调整窗格大小 |
| `Ctrl-a h/j/k/l` | 切换到指定方向窗格 |
| `Ctrl-a q` | 显示窗格编号 |

### 复制粘贴
| 快捷键 | 功能 |
|--------|------|
| `Ctrl-a [` | 进入复制模式 |
| `v` (复制模式) | 开始选择 |
| `y` (复制模式) | 复制并退出 |
| `Ctrl-a ]` | 粘贴 |
| `Ctrl-a y` | 复制整个命令输出 |

### 插件快捷键
| 快捷键 | 功能 |
|--------|------|
| `Ctrl-a Ctrl-s` | 保存会话状态 |
| `Ctrl-a Ctrl-r` | 恢复会话状态 |
| `Ctrl-a o` | 打开文件/URL |
| `Ctrl-a /` | 搜索 |
| `Ctrl-a b` | 打开/关闭侧边栏 |
| `Ctrl-a I` | 安装插件 |
| `Ctrl-a U` | 更新插件 |

### 其他
| 快捷键 | 功能 |
|--------|------|
| `Ctrl-a r` | 重新加载配置 |
| `Ctrl-a ?` | 查看所有快捷键 |
| `Ctrl-a :` | 进入命令模式 |

## 📋 训练任务工作流

### 1. 启动训练会话
```bash
tmux new -s training
cd /mnt/data1/home/jiangjiajun/workspace/ms-swift
sbatch examples/vln/streamvln/script/A100/train_streamvln_qwen2_5_vl.slurm
```

### 2. 监控训练（多窗格）
```bash
# 窗格1: 任务状态
watch -n 5 squeue -u $USER

# 窗格2: 训练日志
tail -f /shared_space/jiangjiajun/workspace/ms-swift/logs/streamvln-msswift-*.out

# 窗格3: GPU监控
watch -n 1 nvidia-smi
```

### 3. 断开连接
```
Ctrl-a d
```

### 4. 重新连接
```bash
tmux attach -t training
```

## 💡 实用技巧

1. **快速切换**: `Alt-方向键` 直接切换窗格，无需按前缀键
2. **同步输入**: `Ctrl-a :setw synchronize-panes` 同步输入到所有窗格
3. **自动保存**: 会话每15分钟自动保存，重启后自动恢复
4. **搜索功能**: `Ctrl-a /` 快速搜索终端内容
5. **文件浏览器**: `Ctrl-a b` 打开侧边栏浏览文件

## 🔧 常用命令

```bash
# 会话管理
tmux ls                          # 列出所有会话
tmux kill-session -t <name>      # 杀死指定会话
tmux rename-session -t old new   # 重命名会话

# 在 tmux 内部（命令模式）
:new-window                      # 创建新窗口
:kill-window                     # 关闭当前窗口
:split-window -h                 # 垂直分割
:split-window -v                 # 水平分割
:select-pane -L                  # 选择左侧窗格
```

## ⚠️ 注意事项

- **前缀键已改为 `Ctrl-a`**（不再是默认的 `Ctrl-b`）
- **VPN 断开不影响**: tmux 会话和 SLURM 任务都会继续运行
- **自动恢复**: 使用 `tmux-continuum` 插件，会话会自动保存和恢复
- **复制模式**: 使用 vi 风格的复制模式，按 `v` 开始选择，`y` 复制

