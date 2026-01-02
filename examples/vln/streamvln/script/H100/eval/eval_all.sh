#!/bin/bash
# 自动评测所有已训练完成的模型
# 
# 功能：
# 1. 遍历 results/eval 和 output 文件夹
# 2. 如果某个模型名称已经eval过则跳过
# 3. 如果没有eval过，检查是否是训练完毕的模型（通过logging.jsonl）
# 4. 如果训练完毕则加入待选list
# 5. 对于待eval list的model，选择最新的checkpoints作为评测的model
# 6. 依次运行评测脚本
# 7. 运行完毕后给出总结

# 不使用 set -e，因为即使某个模型评测失败，也应该继续评测其他模型
set +e

# ============================================================================
# 配置
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MSSWIFT_ROOT="$(cd "$SCRIPT_DIR/../../../../../../" && pwd)"
OUTPUT_DIR="${MSSWIFT_ROOT}/output"
RESULTS_EVAL_DIR="${MSSWIFT_ROOT}/results/eval"
EVAL_SCRIPT="${SCRIPT_DIR}/eval_streamvln_qwen2_5_vl_distributed.sh"

# ============================================================================
# 辅助函数
# ============================================================================

# 检查训练是否完成
# 参数: logging.jsonl文件路径
# 返回: 0表示完成，1表示未完成
check_training_completed() {
    local logging_file="$1"
    
    if [ ! -f "$logging_file" ]; then
        return 1
    fi
    
    # 读取最后一行
    local last_line=$(tail -n 1 "$logging_file" 2>/dev/null)
    
    if [ -z "$last_line" ]; then
        return 1
    fi
    
    # 检查是否包含 train_runtime（训练完成的标志）
    # train_runtime字段只在训练完成时才会写入
    if echo "$last_line" | grep -q '"train_runtime"'; then
        return 0
    fi
    
    return 1
}

# 获取模型名称（从output目录路径中提取）
# 参数: output目录下的模型路径
# 返回: 模型名称
get_model_name() {
    local model_path="$1"
    # 从路径中提取模型名称，例如：
    # /path/to/output/streamvln-qwen2.5vl-3b-full-1ep-f32h8s4-bs64-lr4e-5-20251229-094943
    # 提取: streamvln-qwen2.5vl-3b-full-1ep-f32h8s4-bs64-lr4e-5-20251229-094943
    basename "$model_path"
}

# 检查模型是否已经评测过
# 参数: 模型名称
# 返回: 0表示已评测，1表示未评测
is_model_evaluated() {
    local model_name="$1"
    local eval_dir="${RESULTS_EVAL_DIR}/${model_name}"
    
    if [ -d "$eval_dir" ] && [ "$(ls -A "$eval_dir" 2>/dev/null)" ]; then
        return 0
    fi
    
    return 1
}

# 找到最新的checkpoint
# 参数: 版本目录路径（例如: output/model_name/v0-xxx）
# 返回: 最新checkpoint的完整路径
find_latest_checkpoint() {
    local version_dir="$1"
    local latest_checkpoint=""
    local max_step=0
    
    # 查找所有checkpoint-*目录
    for checkpoint_dir in "${version_dir}"/checkpoint-*; do
        if [ -d "$checkpoint_dir" ]; then
            # 提取checkpoint编号
            local step=$(basename "$checkpoint_dir" | sed 's/checkpoint-//')
            
            # 检查是否是数字
            if [[ "$step" =~ ^[0-9]+$ ]]; then
                if [ "$step" -gt "$max_step" ]; then
                    max_step=$step
                    latest_checkpoint="$checkpoint_dir"
                fi
            fi
        fi
    done
    
    echo "$latest_checkpoint"
}

# ============================================================================
# 主逻辑
# ============================================================================

echo "=============================================="
echo "开始扫描模型..."
echo "=============================================="

# 存储待评测的模型列表 (格式: model_name|checkpoint_path)
declare -a models_to_eval=()
declare -a already_evaluated=()
declare -a not_completed=()

# 遍历output目录
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "[WARNING] Output目录不存在: $OUTPUT_DIR"
    echo "[INFO] 请检查路径是否正确"
    exit 1
fi

# 检查评测脚本是否存在
if [ ! -f "$EVAL_SCRIPT" ]; then
    echo "[ERROR] 评测脚本不存在: $EVAL_SCRIPT"
    exit 1
fi

for model_dir in "${OUTPUT_DIR}"/*; do
    if [ ! -d "$model_dir" ]; then
        continue
    fi
    
    model_name=$(get_model_name "$model_dir")
    echo "[INFO] 检查模型: $model_name"
    
    # 检查是否已经评测过
    if is_model_evaluated "$model_name"; then
        echo "  -> 已评测过，跳过"
        already_evaluated+=("$model_name")
        continue
    fi
    
    # 查找版本目录（v*-*格式），选择最新的训练完成的版本
    found_trained_model=false
    latest_checkpoint=""
    latest_version_time=0
    
    # 先收集所有版本目录并按时间排序（最新的在前）
    while IFS= read -r version_dir; do
        if [ ! -d "$version_dir" ]; then
            continue
        fi
        
        # 检查logging.jsonl
        logging_file="${version_dir}/logging.jsonl"
        
        if check_training_completed "$logging_file"; then
            # 找到最新的checkpoint
            checkpoint=$(find_latest_checkpoint "$version_dir")
            
            if [ -n "$checkpoint" ] && [ -d "$checkpoint" ]; then
                # 获取版本目录的修改时间，选择最新的
                version_time=$(stat -c %Y "$version_dir" 2>/dev/null || echo 0)
                
                if [ "$version_time" -gt "$latest_version_time" ]; then
                    latest_version_time=$version_time
                    latest_checkpoint="$checkpoint"
                    found_trained_model=true
                    echo "  -> 找到训练完成的版本: $(basename "$version_dir")"
                    echo "  -> Checkpoint: $checkpoint"
                fi
            fi
        fi
    done < <(find "$model_dir" -maxdepth 1 -type d -name "v*-*" | sort -r)
    
    if [ "$found_trained_model" = true ] && [ -n "$latest_checkpoint" ]; then
        echo "  -> 加入待评测列表"
        models_to_eval+=("${model_name}|${latest_checkpoint}")
    else
        echo "  -> 不满足评测条件"
        not_completed+=("$model_name")
    fi
done

# ============================================================================
# 执行评测
# ============================================================================

echo ""
echo "=============================================="
echo "开始评测..."
echo "=============================================="

eval_count=0
eval_success=0
eval_failed=()

if [ ${#models_to_eval[@]} -eq 0 ]; then
    echo "[INFO] 没有需要评测的模型"
else
    for model_info in "${models_to_eval[@]}"; do
        IFS='|' read -r model_name checkpoint_path <<< "$model_info"
        
        echo ""
        echo "[INFO] 评测模型 ($((eval_count + 1))/${#models_to_eval[@]}): $model_name"
        echo "[INFO] Checkpoint: $checkpoint_path"
        
        eval_count=$((eval_count + 1))
        
        # 运行评测脚本
        if MODEL_PATH="$checkpoint_path" bash "$EVAL_SCRIPT"; then
            echo "[SUCCESS] 模型 $model_name 评测成功"
            eval_success=$((eval_success + 1))
        else
            echo "[ERROR] 模型 $model_name 评测失败"
            eval_failed+=("$model_name")
        fi
    done
fi

# ============================================================================
# 总结
# ============================================================================

echo ""
echo "=============================================="
echo "评测总结"
echo "=============================================="
echo ""

echo "本次运行评测的模型 (${eval_count}个):"
if [ ${#models_to_eval[@]} -eq 0 ]; then
    echo "  无"
else
    for model_info in "${models_to_eval[@]}"; do
        IFS='|' read -r model_name checkpoint_path <<< "$model_info"
        echo "  - $model_name"
    done
fi

echo ""
echo "之前已评测过的模型 (${#already_evaluated[@]}个):"
if [ ${#already_evaluated[@]} -eq 0 ]; then
    echo "  无"
else
    for model_name in "${already_evaluated[@]}"; do
        echo "  - $model_name"
    done
fi

echo ""
echo "不满足评测要求的模型 (${#not_completed[@]}个):"
if [ ${#not_completed[@]} -eq 0 ]; then
    echo "  无"
else
    for model_name in "${not_completed[@]}"; do
        echo "  - $model_name"
    done
fi

echo ""
if [ $eval_count -gt 0 ]; then
    echo "评测结果: 成功 ${eval_success}/${eval_count}"
    if [ ${#eval_failed[@]} -gt 0 ]; then
        echo "失败的模型:"
        for model_name in "${eval_failed[@]}"; do
            echo "  - $model_name"
        done
    fi
fi

echo ""
echo "=============================================="
echo "完成"
echo "=============================================="

