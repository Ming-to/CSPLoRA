#!/bin/bash

# Arithmetic Evaluation - Multi-GPU Parallel

GPUS=(0 1)
NUM_GPUS=${#GPUS[@]}

USE_BATCH=1
USE_VLLM=0
BATCH_SIZE=8

MODEL_NAMES=(
  "llama2-7b-hf"
  "Llama-3.1-8B"
)
MODEL_ROOT="<your_model_path>"

OUTPUT_ROOT="experiments_llama/instruction_tuning"

EVAL_RESULTS_DIR="eval_results/arithmetic"
mkdir -p ${EVAL_RESULTS_DIR}

METHODS=("lora" "pissa" "dora" "lora_plus")
RANK=8

USE_CSPLORA=1

EVAL_TASKS=("gsm8k" "math")

SEEDS=(42 123 456)

MAX_NEW_TOKENS=512
TEMPERATURE=0.0
TOP_P=1.0
PROMPT_STYLE="instruction"

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

declare -a TASKS_TO_RUN

echo "=========================================="
echo "Collecting evaluation tasks..."
echo "=========================================="

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  BASE_MODEL="${MODEL_ROOT}/${MODEL_NAME}"

  for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do

      if [ "${USE_CSPLORA}" = "1" ]; then
        SUFFIX="_csplora"
      else
        SUFFIX=""
      fi

      CKPT_DIR="${OUTPUT_ROOT}/${MODEL_NAME}/${method}_r${RANK}_seed${seed}${SUFFIX}/final_model"

      if [ ! -d "${CKPT_DIR}" ]; then
        continue
      fi

      for eval_task in "${EVAL_TASKS[@]}"; do
        OUTPUT_FILE="${EVAL_RESULTS_DIR}/${MODEL_NAME}_${method}_r${RANK}_seed${seed}${SUFFIX}_${eval_task}.json"

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

echo "=========================================="
echo "Starting parallel evaluation..."
echo "=========================================="

if [ "${USE_BATCH}" = "1" ]; then
  EVAL_SCRIPT="instruction_tuning_eval/eval_arithmetic_batch.py"
  EXTRA_ARGS="--batch_size ${BATCH_SIZE}"
elif [ "${USE_VLLM}" = "1" ]; then
  EVAL_SCRIPT="instruction_tuning_eval/eval_arithmetic_vllm.py"
  EXTRA_ARGS=""
else
  EVAL_SCRIPT="instruction_tuning_eval/eval_arithmetic.py"
  EXTRA_ARGS=""
fi

LOG_DIR="${EVAL_RESULTS_DIR}/logs"
mkdir -p ${LOG_DIR}

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
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --temperature ${TEMPERATURE} \
    --top_p ${TOP_P} \
    --prompt_style ${PROMPT_STYLE} \
    --output_file "${OUTPUT_FILE}" \
    ${EXTRA_ARGS} \
    > "${LOG_FILE}" 2>&1 &

  PIDS+=($!)
  GPU_IDX=$(( (GPU_IDX + 1) % NUM_GPUS ))

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

echo ""
echo "Waiting for remaining tasks to complete..."
wait

echo ""
echo "=========================================="
echo "Evaluation Complete! Summarizing results..."
echo "=========================================="

parse_result_file() {
  local file="$1"

  if head -5 "$file" | grep -q '"metadata"'; then
    python3 -c "
import json
with open('$file', 'r') as f:
    data = json.load(f)
print(data.get('num_correct', 0))
print(data.get('num_total', 0))
print(f\"{data.get('accuracy', 0):.4f}\")
"
  else
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
echo "Arithmetic Evaluation Summary" > ${SUMMARY_FILE}
echo "Generated at: $(date)" >> ${SUMMARY_FILE}
echo "" >> ${SUMMARY_FILE}

for result_file in ${EVAL_RESULTS_DIR}/*.json ${EVAL_RESULTS_DIR}/*.jsonl; do
  if [ -f "${result_file}" ]; then
    filename=$(basename "${result_file}")
    filename="${filename%.*}"
    read -r correct total acc <<< "$(parse_result_file "${result_file}" | tr '\n' ' ')"
    if [ "${total}" -gt 0 ]; then
      echo "${filename}: ${correct}/${total} = ${acc}" >> ${SUMMARY_FILE}
      echo "${filename}: ${correct}/${total} = ${acc}"
    fi
  fi
done

echo ""
echo "Summary saved to: ${SUMMARY_FILE}"
echo "All arithmetic evaluations completed!"
