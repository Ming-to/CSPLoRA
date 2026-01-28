#!/bin/bash

# Arithmetic Reasoning CSPLoRA experiments

GPU_IDS="0,1"
NUM_GPUS=$(echo "${GPU_IDS}" | awk -F, '{print NF}')
if [ -z "${MASTER_PORT}" ]; then
    MASTER_PORT=$((12000 + (RANDOM % 20000)))
    export MASTER_PORT
fi
export MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
LAUNCHER=(torchrun --nproc_per_node="${NUM_GPUS}" --master_addr="${MASTER_ADDR}" --master_port="${MASTER_PORT}")

MODEL_NAMES=("Llama-3.1-8B")
MODEL_ROOT="<your_model_path>"

export HF_DISK_ROOT="<your_data_path>"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="csplora_arithmetic"

RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.0
MAX_SEQ_LEN=512
EPOCHS=1

LLAMA3_BASE_LR=1e-4
LLAMA3_BATCH_SIZE=1
LLAMA3_GRAD_ACC=32
LLAMA3_WARMUP=0.02
LLAMA3_WEIGHT_DECAY=5e-4
LLAMA3_LR_DECAY=0.1

SEEDS=(42 123 456)
METHODS=("lora" "pissa" "dora" "lora_plus")

USE_CSPLORA=1
CSPLORA_RHO=1
CSPLORA_GAMMA=2.0
CSPLORA_R_MIN=2

DATA_PATH="${HF_DISK_ROOT}/arithmetic/math_50k.json"
DATASET_SPLIT="train[:50000]"
DATASET_FIELD="instruction output"

for model_name in "${MODEL_NAMES[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${model_name}"

    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[Run] model=${model_name}, method=${method}, seed=${seed}"

            CSPLORA_FLAGS=""
            if [ "${USE_CSPLORA}" = "1" ]; then
                CSPLORA_FLAGS="--csplora --csplora_rho ${CSPLORA_RHO} --csplora_gamma ${CSPLORA_GAMMA} --csplora_r_min ${CSPLORA_R_MIN}"
            fi

            CUDA_VISIBLE_DEVICES=${GPU_IDS} ${LAUNCHER[@]} train_arithmetic.py \
                --model "${MODEL_PATH}" \
                --method "${method}" \
                --lora_r ${RANK} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_dropout ${LORA_DROPOUT} \
                --batch_size ${LLAMA3_BATCH_SIZE} \
                --grad_acc_steps ${LLAMA3_GRAD_ACC} \
                --epochs ${EPOCHS} \
                --max_seq_length ${MAX_SEQ_LEN} \
                --lr ${LLAMA3_BASE_LR} \
                --warmup_ratio ${LLAMA3_WARMUP} \
                --weight_decay ${LLAMA3_WEIGHT_DECAY} \
                --lr_decay_ratio ${LLAMA3_LR_DECAY} \
                --seed ${seed} \
                --data_path "${DATA_PATH}" \
                --dataset_split "${DATASET_SPLIT}" \
                --dataset_field ${DATASET_FIELD} \
                ${CSPLORA_FLAGS} \
                --output_dir "./output"
        done
    done
done

echo "All experiments completed."
