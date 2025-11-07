#!/bin/bash

# 使脚本在命令失败时立即退出
set -e
# 如果管道中的任何命令失败，则将整个管道的退出状态视为失败
set -o pipefail

# --- 0. 参数解析与校验 ---
if [ "$#" -ne 4 ]; then
    echo "错误: 需要提供三个参数。" >&2
    echo "用法: $0 <libero_suite> <ckpt> <total_chunks> <start_chunk_idx>" >&2
    exit 1
fi

suite=$1
ckpt=$2
total_chunks=$3
start_chunk_idx=$4

# 验证输入是否为有效的非负整数
if ! [[ "$total_chunks" =~ ^[0-9]+$ ]] || ! [[ "$start_chunk_idx" =~ ^[0-9]+$ ]]; then
    echo "错误: <total_chunks> 和 <start_chunk_idx> 必须是有效的非负整数。" >&2
    exit 1
fi

if [ "$start_chunk_idx" -ge "$total_chunks" ]; then
    echo "错误: <start_chunk_idx> ($start_chunk_idx) 必须小于 <total_chunks> ($total_chunks)。" >&2
    exit 1
fi

echo "Starting evaluation script for checkpoint: $ckpt"
echo "Total chunks to process: $total_chunks"
echo "This instance will start processing from chunk index: $start_chunk_idx"


# 检查并安装必要的工具 (jq, bc)
echo "---------------------------------------------"
echo "检查依赖项 (jq, bc)..."
if ! command -v jq &> /dev/null || ! command -v bc &> /dev/null; then
    echo "jq 或 bc 未找到。尝试使用 apt 安装..."
    # 仅在需要时运行 apt update
    if [ -z "$(find /var/lib/apt/lists -maxdepth 1 -mmin -60)" ]; then
        apt update -y
    fi
    apt install -y jq bc
fi

# --- 配置 ---
TARGET_DIR="./ckpt/MMaDA-VLA/$ckpt"
# 动态构建结果目录，以匹配 Python 脚本的保存逻辑
# Python 逻辑: 
# 1. 如果 ckpt 不含'/' (e.g., '0621f9...'), 则它会读取模型目录下的 num_train_epochs 来构建最终路径。
# 2. 如果 ckpt 包含'/' (e.g., '4eabd.../checkpoint_10'), 它会直接使用 ckpt 作为子路径。
RESULT_SUB_DIR=""
# 检查 $ckpt 变量是否包含路径分隔符 '/'
if [[ "$ckpt" != */* ]]; then
    # --- 情况 1: ckpt 是一个简单的目录名 ---
    # 需要从 $TARGET_DIR/args.json 中读取 'num_train_epochs'
    ARGS_JSON_PATH="$TARGET_DIR/args.json"
    # 在等待目录之后，还需确保配置文件存在
    echo "正在等待模型配置文件: '$ARGS_JSON_PATH'..."
    while [ ! -f "$ARGS_JSON_PATH" ]; do
      echo "文件不存在。将在 10 秒后重试..."
      sleep 10
    done
    echo "找到模型配置文件。"
    # 使用 jq 安全地提取 num_train_epochs
    num_epochs=$(jq -r '.num_train_epochs' "$ARGS_JSON_PATH") || {
        echo "错误: 无法使用 jq 从 '$ARGS_JSON_PATH' 读取 'num_train_epochs'。" >&2
        exit 1
    }
    # 验证提取的值是否为数字
    if ! [[ "$num_epochs" =~ ^[0-9]+$ ]]; then
        echo "错误: 从 '$ARGS_JSON_PATH' 中提取的 'num_train_epochs' 值无效: '$num_epochs'。" >&2
        exit 1
    fi
    echo "从配置文件中读取到 num_train_epochs = $num_epochs"
    RESULT_SUB_DIR="$ckpt/checkpoint_$num_epochs"
else
    # --- 情况 2: ckpt 本身就是一个包含 checkpoint 的完整路径 ---
    # Python 脚本会直接使用这个路径
    RESULT_SUB_DIR="$ckpt"
fi
# 最终构建出与 Python 脚本完全一致的结果目录
RESULT_DIR="./rollouts/$RESULT_SUB_DIR/result"
# 增加一行日志，用于调试和确认路径是否正确
echo "已将预期结果目录设置为: $RESULT_DIR/result"
# 假设Python脚本生成的结果文件名为 "results_chunk_idx_N.json"
# 如果你的文件名格式不同, 请修改下面的 check_and_aggregate_results 函数
RESULT_FILE_PATTERN="task_id=" 
SLEEP_INTERVAL=600 # 检查周期可以适当缩短


# --- 1. 检查和准备环境 ---

echo "Checking for available GPUs..."
# 使用新的命令替换语法，并添加错误处理
available_gpus=$(python -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null) || {
    echo "错误: 运行 python 或导入 torch 失败。PyTorch 是否已正确安装？" >&2
    exit 1
}

# 验证 available_gpus 是一个正整数
if ! [[ "$available_gpus" =~ ^[1-9][0-9]*$ ]]; then
    echo "错误: 未找到可用的 GPU 或 GPU 数量无效: '$available_gpus'。退出。" >&2
    exit 1
fi
echo "Available GPUs: $available_gpus"

# # --- 2. 等待模型检查点 ---

echo "Waiting for directory '$TARGET_DIR' to appear..."
while [ ! -d "$TARGET_DIR" ]; do
  echo "目录不存在。将在 $SLEEP_INTERVAL 秒后重试..."
  sleep $SLEEP_INTERVAL
done
echo "Directory has been found."

# --- 3. 并行执行分配的评估区块 ---

pids=() # 声明一个数组来存储 PIDs
tasks_to_run=0
for ((i=0; i<$available_gpus; i++)); do
    current_chunk_idx=$((start_chunk_idx + i))
    
    # 确保当前区块索引没有超过总区块数
    if [ "$current_chunk_idx" -ge "$total_chunks" ]; then
        echo "已达到总区块数上限 ($total_chunks)，不再启动新的评估任务。"
        break
    fi

    echo "启动评估: 全局区块 $current_chunk_idx / $total_chunks (使用 GPU $i)..."
    python mmadavla/eval/eval_libero.py \
        --mmadavla_path $TARGET_DIR \
        --task_suite $suite \
        --num_chunks $total_chunks \
        --chunk_idx $current_chunk_idx \
        --device_idx $i &
    pids[$i]=$!
    tasks_to_run=$((tasks_to_run + 1))
    sleep 2 # 短暂休眠以错开启动，避免瞬间资源竞争
done

# 如果没有启动任何任务，则直接跳到聚合阶段
if [ ${#pids[@]} -eq 0 ]; then
    echo "没有需要在此实例上运行的评估任务。"
else
    # 等待所有后台进程完成，并检查它们的退出状态
    echo "等待此实例分配的 $tasks_to_run 个评估进程完成..."
    all_success=true
    for pid in "${pids[@]}"; do
        if wait "$pid"; then
            echo "进程 $pid 成功完成。"
        else
            echo "警告: 进程 $pid 失败，退出码 $?。" >&2
            all_success=false
            # 不在此处退出，允许脚本继续尝试聚合
        fi
    done
    
    if [ "$all_success" = true ]; then
        echo "此实例分配的所有评估进程均已成功完成。"
    else
        echo "警告: 此实例中存在失败的评估进程。"
    fi
fi


# --- 4. 等待所有区块的结果文件生成 ---

echo "============================================="
echo "等待所有 $total_chunks 个区块的结果文件全部生成..."
echo "结果目录: $RESULT_DIR"

while true; do
  found_files=0
  missing_chunks=""
  for ((j=0; j<$total_chunks; j++)); do
    expected_file="$RESULT_DIR/${RESULT_FILE_PATTERN}${j}.json"
    if [ -f "$expected_file" ]; then
        found_files=$((found_files + 1))
    else
        missing_chunks+="$j "
    fi
  done

  if [ "$found_files" -eq "$total_chunks" ]; then
    echo "全部 $total_chunks 个结果文件均已找到。准备聚合结果。"
    break
  else
    echo "已找到 $found_files / $total_chunks 个结果文件。缺失区块: $missing_chunks"
    echo "将在 $SLEEP_INTERVAL 秒后再次检查..."
    sleep $SLEEP_INTERVAL
  fi
done

# --- 5. 聚合结果 ---

echo "============================================="
echo "聚合来自 '$RESULT_DIR' 的所有结果..."

total_episodes=0
total_successes=0
json_files_found=0

for json_file in "$RESULT_DIR"/*.json; do
    # 这个检查确保在没有json文件时循环不会出错
    [ -e "$json_file" ] || continue
    
    # 提取区块索引以提供更详细的日志
    filename=$(basename "$json_file")
    echo "处理文件: $filename"
    
    # 添加更健壮的jq解析，如果key不存在则返回0
    episodes=$(jq '.task_episodes // 0' "$json_file")
    successes=$(jq '.task_successes // 0' "$json_file")
    
    if [[ "$episodes" =~ ^[0-9]+$ ]] && [[ "$successes" =~ ^[0-9]+$ ]]; then
        total_episodes=$((total_episodes + episodes))
        total_successes=$((total_successes + successes))
        json_files_found=$((json_files_found + 1))
    else
        echo "警告: 无法从 '$json_file' 解析有效的 episodes 或 successes。原始值: episodes='$episodes', successes='$successes'。"
    fi
done

echo "---"
echo "处理的 JSON 文件总数: $json_files_found"
if [ "$json_files_found" -ne "$total_chunks" ]; then
    echo "警告: 找到的JSON文件数 ($json_files_found) 与预期的总区块数 ($total_chunks) 不符！"
fi
echo "聚合的总 episodes: $total_episodes"
echo "聚合的总 successes: $total_successes"

if [ $total_episodes -gt 0 ]; then
    # 使用 bc 计算成功率，保留两位小数
    overall_success_rate_percent=$(echo "scale=2; $total_successes * 100 / $total_episodes" | bc)
    echo "---------------------------------------------"
    echo "总体成功率: $overall_success_rate_percent%"
    echo "---------------------------------------------"
else
    echo "总 episodes 为 0，无法计算成功率。"
fi
echo "聚合完成。"

# --- 5. 存储结果 ---
echo # 输出一个空行以作分隔
# 确保结果目录存在
mkdir -p "$RESULT_DIR"
echo "正在将结构化结果保存到 $RESULT_DIR/overall_success_rate.json..."
{
    printf '{\n'
    printf '  "timestamp": "%s",\n' "$(date -u +"%Y-%m-%dT%H:%M:%SZ")"
    printf '  "summary": {\n'
    printf '    "json_files_found": %d,\n' "$json_files_found"
    printf '    "expected_chunks": %d,\n' "$total_chunks"
    printf '    "total_episodes": %d,\n' "$total_episodes"
    printf '    "total_successes": %d,\n' "$total_successes"
    printf '    "overall_success_rate_percent": "%s"\n' "$overall_success_rate_percent"
    printf '  }'
    printf '\n}\n'
} > "$RESULT_DIR/overall_success_rate.json"
echo "JSON 结果已成功保存。"
echo "--- 存储完成 ---"