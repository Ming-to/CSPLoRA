#!/bin/bash

########################################
# Commonsense Reasoning 任务评测脚本 - 多GPU并行版本
# 评测 BoolQ, PIQA, SIQA, HellaSwag
########################################

# 可用的GPU列表（根据你的机器修改）
GPUS=(0 1 4 5)
NUM_GPUS=${#GPUS[@]}

# 使用vLLM版本（设置为1）还是原版（设置为0）
USE_VLLM=0

# 模型配置
MODEL_NAMES=(
  "llama2-7b-hf"
  "Llama-3.1-8B"
)
MODEL_ROOT="/data/share_weight"

# 实验输出根目录
OUTPUT_ROOT="experiments_llama/commonsense_reasoning"

# 评测结果保存目录
EVAL_RESULTS_DIR="eval_results/commonsense"
mkdir -p ${EVAL_RESULTS_DIR}

# LoRA 方法和配置
# METHODS=("full" "lora" "pissa" "adalora" "dora" "lora_plus" "gora")
# METHODS=("lora" "pissa" "dora" "lora_plus")
METHODS=("pissa")
RANK=8

# CSpLoRA 开关
USE_CSPLORA=1

# 评测任务
EVAL_TASKS=("boolq" "hellaswag" "piqa" "siqa" "winogrande" "arc_easy" "arc_challenge" "obqa")

# Seeds
SEEDS=(21074 2235 13767)

########################################
# 信号处理：Ctrl+C时终止所有子进程
########################################

cleanup() {
  echo ""
  echo "Caught interrupt signal. Killing all child processes..."
  for pid in "${PIDS[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM "$pid" 2>/dev/null
    fi
  done
  pkill -P $$ 2>/dev/null
  echo "All processes terminated."
  exit 1
}

trap cleanup SIGINT SIGTERM

########################################
# 收集所有待评测任务
########################################

declare -a TASKS_TO_RUN

echo "=========================================="
echo "Collecting evaluation tasks..."
echo "=========================================="

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  BASE_MODEL="${MODEL_ROOT}/${MODEL_NAME}"

  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      if [ "${USE_CSPLORA}" = "1" ]; then
        SUFFIX="_csplora_ada"
      else
        SUFFIX=""
      fi

      CKPT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${method}_r${RANK}_seed${seed}${SUFFIX}/final_model"

      if [ ! -d "${CKPT_DIR}" ]; then
        continue
      fi

      for eval_task in "${EVAL_TASKS[@]}"; do
        OUTPUT_FILE="${EVAL_RESULTS_DIR}/${MODEL_NAME}_${method}_r${RANK}_seed${seed}${SUFFIX}_${eval_task}.jsonl"

        if [ -f "${OUTPUT_FILE}" ]; then
          echo "[SKIP] Already evaluated: ${OUTPUT_FILE}"
          continue
        fi

        TASKS_TO_RUN+=("${BASE_MODEL}|${CKPT_DIR}|${eval_task}|${OUTPUT_FILE}")
      done
    done
  done
done

TOTAL_TASKS=${#TASKS_TO_RUN[@]}
echo ""
echo "Total tasks to run: ${TOTAL_TASKS}"
echo "Using ${NUM_GPUS} GPUs: ${GPUS[*]}"
echo ""

if [ ${TOTAL_TASKS} -eq 0 ]; then
  echo "No tasks to run. Exiting."
  exit 0
fi

########################################
# 并行执行评测
########################################

echo "=========================================="
echo "Starting parallel evaluation..."
echo "=========================================="

# 选择评测脚本
if [ "${USE_VLLM}" = "1" ]; then
  EVAL_SCRIPT="instruction_tuning_eval/eval_commonsense_vllm.py"
  echo "Using vLLM version: ${EVAL_SCRIPT}"
else
  EVAL_SCRIPT="instruction_tuning_eval/eval_commonsense.py"
  echo "Using original version: ${EVAL_SCRIPT}"
fi

# 创建临时目录存放日志
LOG_DIR="${EVAL_RESULTS_DIR}/logs"
mkdir -p ${LOG_DIR}

# 并行执行
PIDS=()
GPU_IDX=0

for i in "${!TASKS_TO_RUN[@]}"; do
  TASK="${TASKS_TO_RUN[$i]}"
  IFS='|' read -r BASE_MODEL CKPT_DIR EVAL_TASK OUTPUT_FILE <<< "${TASK}"

  GPU_ID=${GPUS[$GPU_IDX]}
  LOG_FILE="${LOG_DIR}/task_${i}.log"

  echo "[Task $((i+1))/${TOTAL_TASKS}] GPU ${GPU_ID}: ${CKPT_DIR} -> ${EVAL_TASK}"

  CUDA_VISIBLE_DEVICES=${GPU_ID} python3 ${EVAL_SCRIPT} \
    --base_model "${BASE_MODEL}" \
    --ckpt_dir "${CKPT_DIR}" \
    --eval_task "${EVAL_TASK}" \
    --output_file "${OUTPUT_FILE}" \
    > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
  GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))

  # 如果所有GPU都在使用，等待任意一个完成
  if [ ${#PIDS[@]} -ge ${NUM_GPUS} ]; then
    wait -n
    NEW_PIDS=()
    for pid in "${PIDS[@]}"; do
      if kill -0 "$pid" 2>/dev/null; then
        NEW_PIDS+=("$pid")
      fi
    done
    PIDS=("${NEW_PIDS[@]}")
  fi
done

# 等待所有剩余任务完成
echo ""
echo "Waiting for remaining tasks to complete..."
wait

########################################
# 汇总结果
########################################

echo ""
echo "=========================================="
echo "Evaluation Complete! Summarizing results..."
echo "=========================================="

# 解析结果文件，兼容两种格式：
# - 旧格式 JSONL: 每行一条记录 {"question":..., "correct":...}
# - 新格式 JSON: 单个对象 {"metadata":..., "accuracy":..., "num_correct":..., "num_total":..., "results":[...]}
parse_result_file() {
  local file="$1"

  # 检测格式：新格式前几行包含 "metadata"（因为 JSON 可能带缩进）
  if head -5 "$file" | grep -q '"metadata"'; then
    # 新格式：用 python 提取字段
    python3 -c "
import json
with open('$file', 'r') as f:
    data = json.load(f)
print(data.get('num_correct', 0))
print(data.get('num_total', 0))
print(f\"{data.get('accuracy', 0):.4f}\")
"
  else
    # 旧格式 JSONL：用 wc 和 grep
    local total correct acc
    total=$(wc -l < "$file")
    correct=$(grep -c '"correct": true' "$file" 2>/dev/null || echo 0)
    if [ "${total}" -gt 0 ]; then
      acc=$(echo "scale=4; $correct / $total" | bc)
    else
      acc="0"
    fi
    echo "$correct"
    echo "$total"
    echo "$acc"
  fi
}

SUMMARY_FILE="${EVAL_RESULTS_DIR}/summary.txt"
echo "Commonsense Reasoning Evaluation Summary" > ${SUMMARY_FILE}
echo "Generated at: $(date)" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

for result_file in ${EVAL_RESULTS_DIR}/*.jsonl; do
  if [ -f "${result_file}" ]; then
    filename=$(basename "${result_file}" .jsonl)
    # 使用兼容函数解析
    read -r correct total acc <<< "$(parse_result_file "${result_file}" | tr '\n' ' ')"
    if [ "${total}" -gt 0 ]; then
      echo "${filename}: ${correct}/${total} = ${acc}" >> ${SUMMARY_FILE}
      echo "${filename}: ${correct}/${total} = ${acc}"
    fi
  fi
done

echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo "Logs saved to: ${LOG_DIR}"
echo "All commonsense evaluations completed!"
