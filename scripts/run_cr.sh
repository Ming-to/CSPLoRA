#!/bin/bash

# Commonsense Reasoning CSPLoRA experiments

GPU_IDS="0,1"
NUM_GPUS=$(echo "${GPU_IDS}" | awk -F, '{print NF}')
MASTER_PORT=$((29500 + $RANDOM % 10000))
LAUNCHER=(torchrun --nproc_per_node="${NUM_GPUS}" --master_port="${MASTER_PORT}")

MODEL_NAMES=("Llama-3.1-8B")
MODEL_ROOT="<your_model_path>"

export HF_DISK_ROOT="<your_data_path>"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export WANDB_PROJECT="csplora_cr"

RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.05
MAX_SEQ_LEN=256
EPOCHS=1

LLAMA3_BASE_LR=1e-4
LLAMA3_BATCH_SIZE=1
LLAMA3_GRAD_ACC=32
LLAMA3_WARMUP=0.02
LLAMA3_WEIGHT_DECAY=5e-4

SEEDS=(42 123 456)
METHODS=("lora" "pissa" "dora" "lora_plus")

USE_CSPLORA=1
CSPLORA_RHO=1
CSPLORA_GAMMA=2.0
CSPLORA_GAMMA_STRATEGY=${CSPLORA_GAMMA_STRATEGY:-fixed}
CSPLORA_GAMMA_SCALE=${CSPLORA_GAMMA_SCALE:-1.0}
CSPLORA_R_MIN=2

CR_DATA_PATH="${HF_DISK_ROOT}/commonsense/commonsense_170k.json"
TRAIN_ON_INPUTS="--train_on_inputs"

for model_name in "${MODEL_NAMES[@]}"; do
    MODEL_PATH="${MODEL_ROOT}/${model_name}"

    for method in "${METHODS[@]}"; do
        for seed in "${SEEDS[@]}"; do
            echo "[Run] model=${model_name}, method=${method}, seed=${seed}"

            CSPLORA_FLAGS=""
            if [ "${USE_CSPLORA}" = "1" ]; then
                CSPLORA_FLAGS="--csplora --csplora_rho ${CSPLORA_RHO} --csplora_gamma ${CSPLORA_GAMMA} --csplora_gamma_strategy ${CSPLORA_GAMMA_STRATEGY} --csplora_gamma_scale ${CSPLORA_GAMMA_SCALE} --csplora_r_min ${CSPLORA_R_MIN}"
            fi

            CUDA_VISIBLE_DEVICES=${GPU_IDS} ${LAUNCHER[@]} train_cr.py \
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
                --seed ${seed} \
                --data_path "${CR_DATA_PATH}" \
                ${TRAIN_ON_INPUTS} \
                ${CSPLORA_FLAGS} \
                --output_dir "./output"
        done
    done
done

echo "All experiments completed."
