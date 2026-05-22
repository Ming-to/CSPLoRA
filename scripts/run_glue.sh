#!/bin/bash

# GLUE CSPLoRA experiments

GPU_ID=0
MODEL_PATH="<your_model_path>/roberta-large"
RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1
SEEDS=(42 123 456)

SMALL_TASKS=("cola" "mrpc" "rte" "stsb")
BIG_TASKS=("mnli" "qqp" "qnli" "sst2")
ALL_METHODS=("lora" "pissa" "dora" "lora_plus")

RUN_ALL=false
RUN_ALL_METHODS=false
SELECTED_METHOD="lora"

while [[ $# -gt 0 ]]; do
    case $1 in
        --all) RUN_ALL=true; shift ;;
        --method) SELECTED_METHOD="$2"; shift 2 ;;
        --all-methods) RUN_ALL_METHODS=true; shift ;;
        *) echo "Usage: $0 [--all] [--method lora|pissa|dora|lora_plus] [--all-methods]"; exit 1 ;;
    esac
done

if [ "$RUN_ALL" = true ]; then
    TASKS=("${SMALL_TASKS[@]}" "${BIG_TASKS[@]}")
else
    TASKS=("${SMALL_TASKS[@]}")
fi

if [ "$RUN_ALL_METHODS" = true ]; then
    METHODS=("${ALL_METHODS[@]}")
else
    METHODS=("${SELECTED_METHOD}")
fi

LORA_LR=5e-5
LORA_PLUS_LR_RATIO=10.0

CSPLORA_R_MIN=2
CSPLORA_R_MAX_FACTOR=4.0
CSPLORA_GAMMA=2.0
CSPLORA_GAMMA_STRATEGY=${CSPLORA_GAMMA_STRATEGY:-fixed}
CSPLORA_GAMMA_SCALE=${CSPLORA_GAMMA_SCALE:-1.0}
CSPLORA_TAU=1.0

ADA_MIN_STEPS=50
ADA_MAX_STEPS=500
ADA_CONVERGENCE_THRESHOLD=0.02
ADA_CHECK_INTERVAL=10
ADA_PATIENCE=3
ADA_TOP_K=20
ADA_RANK_TOLERANCE=5
ADA_RHO_METHOD="coverage"
ADA_RHO_MIN=0.6
ADA_RHO_MAX=1.0

export HF_DISK_ROOT="<your_data_path>"
export WANDB_PROJECT="csplora_glue"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

is_big_task() {
    local t="$1"
    for bt in "${BIG_TASKS[@]}"; do
        if [ "$bt" == "$t" ]; then return 0; fi
    done
    return 1
}

get_method_args() {
    local method=$1
    case "${method}" in
        "lora") echo "--method lora" ;;
        "pissa") echo "--method pissa" ;;
        "dora") echo "--method dora" ;;
        "lora_plus"|"lora+") echo "--method lora_plus --lora_plus_lr_ratio ${LORA_PLUS_LR_RATIO}" ;;
        *) echo "Unknown method: ${method}"; exit 1 ;;
    esac
}

run_experiment() {
    local task=$1
    local seed=$2
    local epochs=$3
    local method=$4
    local method_args=$(get_method_args "${method}")

    echo "[Run] method=${method}, task=${task}, seed=${seed}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_glue.py \
        --task "${task}" \
        --model "${MODEL_PATH}" \
        --lora_r ${RANK} \
        --lora_alpha ${LORA_ALPHA} \
        --lora_dropout ${LORA_DROPOUT} \
        --lr ${LORA_LR} \
        --seed ${seed} \
        ${method_args} \
        --epochs ${epochs} \
        --csplora \
        --csplora_r_min ${CSPLORA_R_MIN} \
        --csplora_r_max_factor ${CSPLORA_R_MAX_FACTOR} \
        --csplora_gamma ${CSPLORA_GAMMA} \
        --csplora_gamma_strategy ${CSPLORA_GAMMA_STRATEGY} \
        --csplora_gamma_scale ${CSPLORA_GAMMA_SCALE} \
        --csplora_tau_scale ${CSPLORA_TAU} \
        --ada_min_steps ${ADA_MIN_STEPS} \
        --ada_max_steps ${ADA_MAX_STEPS} \
        --ada_convergence_threshold ${ADA_CONVERGENCE_THRESHOLD} \
        --ada_check_interval ${ADA_CHECK_INTERVAL} \
        --ada_patience ${ADA_PATIENCE} \
        --ada_top_k ${ADA_TOP_K} \
        --ada_rank_tolerance ${ADA_RANK_TOLERANCE} \
        --ada_rho_method ${ADA_RHO_METHOD} \
        --ada_rho_min ${ADA_RHO_MIN} \
        --ada_rho_max ${ADA_RHO_MAX} \
        --csplora_cache_dir "./csplora_cache"
}

for method in "${METHODS[@]}"; do
    for task in "${TASKS[@]}"; do
        if is_big_task "${task}"; then
            EPOCHS=3
        else
            EPOCHS=10
        fi
        for seed in "${SEEDS[@]}"; do
            run_experiment "${task}" "${seed}" "${EPOCHS}" "${method}"
        done
    done
done

echo "All experiments completed."
