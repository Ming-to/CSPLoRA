#!/bin/bash

########################################
# 基本环境设置
########################################

GPU_IDS="0,7"
NUM_GPUS=$(echo "${GPU_IDS}" | awk -F, '{print NF}')

# 避免多次运行时端口冲突（可通过外部设置 MASTER_PORT 覆盖）
if [ -z "${MASTER_PORT}" ]; then
  MASTER_PORT=$((12000 + (RANDOM % 20000)))
  export MASTER_PORT
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"

LAUNCHER=(torchrun --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")

# 本地模型目录名（对应 /data/share_weight 下的文件夹）
MODEL_NAMES=(
  # "llama2-7b-hf"
  "Llama-3.1-8B"
)

# 本地权重根目录（和 GLUE 一致）
MODEL_ROOT="/data/share_weight"

# HF / 数据缓存目录（和 GLUE 一致）
export HF_DISK_ROOT="/data/dhming/Second/lora-sb/data"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 避免 CUDA allocator 碎片化导致“跑一会儿才 OOM”
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# wandb 项目
export WANDB_PROJECT="llama_arithmetic_adacsp"

########################################
# LoRA / 训练超参 —— 对齐 GoRA (Table 11)
########################################

RANK=8          # LoRA rank (r or r_ref)
LORA_ALPHA=16   # LoRA alpha
LORA_DROPOUT=0.0  # GoRA 实验中 dropout=0

MAX_SEQ_LEN=512
EPOCHS=1

# ========== Llama3.1-8B-Base 参数 (GoRA + LoRA-SB 折中) ==========
# 参考 GoRA 的 WD/LR decay，并对齐 LoRA-SB 的 max_seq_len=512、warmup=0.02、global batch=32
LLAMA3_BASE_LR=1e-4
LLAMA3_ADALORA_LR=5e-4   # AdaLoRA 专用
LLAMA3_BATCH_SIZE=1
LLAMA3_GRAD_ACC=32       # 有效 batch = 1 * 32 = 32
LLAMA3_WARMUP=0.02
LLAMA3_WEIGHT_DECAY=5e-4
LLAMA3_LR_DECAY=0.1

# ========== Llama2-7B-Base 参数 (GoRA + LoRA-SB 折中) ==========
# 对齐 LoRA-SB 的 global batch=32 与 warmup=0.02
LLAMA2_BASE_LR=5e-5
LLAMA2_ADALORA_LR=5e-4   # AdaLoRA 专用
LLAMA2_BATCH_SIZE=2
LLAMA2_GRAD_ACC=16       # 有效 batch = 2 * 16 = 32（给显存留余量）
LLAMA2_WARMUP=0.02
LLAMA2_WEIGHT_DECAY=0
LLAMA2_LR_DECAY=0.0

# 固定 3 个 seeds，与 GoRA 对齐（报告 mean ± std）
SEEDS=(21074 2235 13767)
echo "Using fixed seeds (aligned with GoRA): ${SEEDS[@]}"

# GoRA 超参数（和 GLUE 侧保持一致；只有 method=gora 时才会生效）
GORA_R_MIN=1
GORA_R_MAX=64
GORA_GAMMA=1.0
GORA_NUM_STEPS=50

# 方法集合（默认不跑 full：Llama3 在单卡上极易 OOM）
RUN_FULL=0
# METHODS=("lora" "pissa" "adalora" "dora" "lora_plus" "gora")
# METHODS=("lora" "pissa" "lora_plus" "dora")
METHODS=("lora_plus")
if [ "${RUN_FULL}" = "1" ]; then
  METHODS=("full" "${METHODS[@]}")
fi

########################################
# CSpLoRA 开关（设置为 1 时启用，默认使用 ada 模式）
########################################
USE_CSPLORA=1
CSPLORA_RHO=1
CSPLORA_GAMMA=2.0
CSPLORA_R_MIN=2


########################################
# 数据集设置（MetaMathQA-395K，取 50K 子集）
########################################

AR_DATA_PATH="${HF_DISK_ROOT}/arithmetic/MetaMathQA-395K.json"
DATASET_SPLIT="train[:50000]"
DATASET_FIELD_ARGS=(--dataset_field query response)


########################################
# 主循环
########################################

for MODEL_NAME in "${MODEL_NAMES[@]}"; do

  MODEL_PATH="${MODEL_ROOT}/${MODEL_NAME}"

  echo "======== MODEL: ${MODEL_PATH} ========"

  # 根据模型选择对应的超参数
  if [[ "${MODEL_NAME}" == *"Llama-3"* ]] || [[ "${MODEL_NAME}" == *"llama-3"* ]] || [[ "${MODEL_NAME}" == *"Llama3"* ]]; then
    BASE_LR=${LLAMA3_BASE_LR}
    ADALORA_LR=${LLAMA3_ADALORA_LR}
    BATCH_SIZE=${LLAMA3_BATCH_SIZE}
    BASE_GRAD_ACC=${LLAMA3_GRAD_ACC}
    WARMUP_RATIO=${LLAMA3_WARMUP}
    WEIGHT_DECAY=${LLAMA3_WEIGHT_DECAY}
    LR_DECAY=${LLAMA3_LR_DECAY}
    if (( BASE_GRAD_ACC % NUM_GPUS != 0 )); then
      echo "[ERROR] LLAMA3_GRAD_ACC (${BASE_GRAD_ACC}) not divisible by NUM_GPUS (${NUM_GPUS})."
      exit 1
    fi
    GRAD_ACC=$((BASE_GRAD_ACC / NUM_GPUS))
    echo "  -> Using Llama3 config: LR=${BASE_LR}, BS=${BATCH_SIZE}x${GRAD_ACC}x${NUM_GPUS}=$((BATCH_SIZE*GRAD_ACC*NUM_GPUS)), WD=${WEIGHT_DECAY}, LR_DECAY=${LR_DECAY}"
  else
    # Llama2 或其他模型
    BASE_LR=${LLAMA2_BASE_LR}
    ADALORA_LR=${LLAMA2_ADALORA_LR}
    BATCH_SIZE=${LLAMA2_BATCH_SIZE}
    BASE_GRAD_ACC=${LLAMA2_GRAD_ACC}
    WARMUP_RATIO=${LLAMA2_WARMUP}
    WEIGHT_DECAY=${LLAMA2_WEIGHT_DECAY}
    LR_DECAY=${LLAMA2_LR_DECAY}
    if (( BASE_GRAD_ACC % NUM_GPUS != 0 )); then
      echo "[ERROR] LLAMA2_GRAD_ACC (${BASE_GRAD_ACC}) not divisible by NUM_GPUS (${NUM_GPUS})."
      exit 1
    fi
    GRAD_ACC=$((BASE_GRAD_ACC / NUM_GPUS))
    echo "  -> Using Llama2 config: LR=${BASE_LR}, BS=${BATCH_SIZE}x${GRAD_ACC}x${NUM_GPUS}=$((BATCH_SIZE*GRAD_ACC*NUM_GPUS)), WD=${WEIGHT_DECAY}, LR_DECAY=${LR_DECAY}"
  fi

  for method in "${METHODS[@]}"; do

    # 按方法选学习率
    if [ "${method}" = "adalora" ]; then
      LR=${ADALORA_LR}
    else
      # full / lora / pissa / dora / lora_plus 都用 BASE_LR
      LR=${BASE_LR}
    fi

    for seed in "${SEEDS[@]}"; do

      # 构建输出目录路径，检查是否已训练完成
      if [ "${USE_CSPLORA}" = "1" ]; then
        CSPLORA_SUFFIX="_csplora_ada"
      else
        CSPLORA_SUFFIX=""
      fi
      CKPT_DIR="experiments_llama/instruction_tuning/${MODEL_NAME}/${method}_r${RANK}_seed${seed}${CSPLORA_SUFFIX}/final_model"

      # 检查模型文件是否已存在（LoRA用adapter_model，full用model.safetensors）
      if [ -f "${CKPT_DIR}/adapter_model.safetensors" ] || [ -f "${CKPT_DIR}/adapter_model.bin" ] || [ -f "${CKPT_DIR}/model.safetensors" ]; then
        echo "[SKIP] Already trained: ${CKPT_DIR}"
        continue
      fi

      echo "===================================================="
      echo "[ARITH] model=${MODEL_NAME}, method=${method}${RUN_SUFFIX}, lr=${LR}, seed=${seed}"
      echo "        LoRA r=${RANK}, alpha=${LORA_ALPHA}, max_len=${MAX_SEQ_LEN}, epochs=${EPOCHS}"
      echo "        CSpLoRA=${USE_CSPLORA}"
      echo "===================================================="

      # 构建 CSpLoRA 参数（默认使用 ada 模式，rho 固定为 CSPLORA_RHO）
      CSPLORA_ARGS=""
      if [ "${USE_CSPLORA}" = "1" ]; then
        CSPLORA_ARGS="--csplora_ada --csplora_rho ${CSPLORA_RHO} --csplora_gamma ${CSPLORA_GAMMA} --csplora_r_min ${CSPLORA_R_MIN} --ada_no_adaptive_rho"
      fi

      # GoRA 参数：只有 method=gora 时才传
      GORA_ARGS=""
      if [ "${method}" = "gora" ]; then
        GORA_ARGS="--gora_r_min ${GORA_R_MIN} --gora_r_max ${GORA_R_MAX} --gora_gamma ${GORA_GAMMA} --gora_num_steps ${GORA_NUM_STEPS}"
      fi

      CUDA_VISIBLE_DEVICES=${GPU_IDS} "${LAUNCHER[@]}" train_arithmetic.py \
        --model "${MODEL_PATH}" \
        --method "${method}" \
        --lora_r ${RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --attn_implementation "sdpa" \
        --gradient_checkpointing \
        --batch_size ${BATCH_SIZE} \
        --grad_acc_steps ${GRAD_ACC} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --weight_decay ${WEIGHT_DECAY} \
        --lr_decay_ratio ${LR_DECAY} \
        --scheduler "cosine" \
        --warmup_ratio ${WARMUP_RATIO} \
        --max_seq_length ${MAX_SEQ_LEN} \
        --seed ${seed} \
        --output_dir "experiments_llama" \
        --project_name "${WANDB_PROJECT}" \
        --data_path "${AR_DATA_PATH}" \
        --dataset_split "${DATASET_SPLIT}" \
        "${DATASET_FIELD_ARGS[@]}" \
        ${GORA_ARGS} \
        ${CSPLORA_ARGS} 

      echo
    done
  done
done

echo "All arithmetic experiments completed!"
