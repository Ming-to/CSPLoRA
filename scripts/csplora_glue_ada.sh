#!/bin/bash

# ============================================================
# GLUE 自适应 C-SPLoRA 对比实验脚本 (支持多种 PEFT 方法)
#
# 支持的 PEFT 方法：
# - lora: 标准 LoRA
# - pissa: PiSSA 初始化
# - dora: DoRA
# - lora_plus: LoRA+ (不同学习率)
#
# 对比配置：
# 1. 原始 C-SPLoRA (固定 step=200, 固定 rho=0.9)
# 2. 自适应 C-SPLoRA (自适应 step + 自适应 rho)
# 3. 消融实验：只自适应 step / 只自适应 rho
#
# 用法：
#   bash csplora_glue_ada.sh                    # 小任务 + lora
#   bash csplora_glue_ada.sh --all              # 全部任务 + lora
#   bash csplora_glue_ada.sh --method pissa     # 小任务 + pissa
#   bash csplora_glue_ada.sh --all --method dora # 全部任务 + dora
#   bash csplora_glue_ada.sh --all-methods      # 小任务 + 所有方法
# ============================================================

# 使用的 GPU
GPU_ID=4

# 底座模型
MODEL_NAME="roberta-large"
MODEL_PATH="/data/share_weight/${MODEL_NAME}"

# LoRA 配置
RANK=8
LORA_ALPHA=16
LORA_DROPOUT=0.1

# 固定随机种子
SEEDS=(42 123 456)
echo "Using FIXED seeds: ${SEEDS[@]}"

# GLUE 任务（先跑小任务快速验证）
# 完整任务列表：TASKS=("mnli" "qqp" "qnli" "sst2" "cola" "mrpc" "rte" "stsb")
SMALL_TASKS=("cola" "mrpc" "rte" "stsb")
BIG_TASKS=("mnli" "qqp" "qnli" "sst2")

# 支持的 PEFT 方法
ALL_METHODS=("lora" "pissa" "dora" "lora_plus")

# 解析命令行参数
RUN_ALL=false
RUN_ALL_METHODS=false
SELECTED_METHOD="lora"

while [[ $# -gt 0 ]]; do
    case $1 in
        --all)
            RUN_ALL=true
            shift
            ;;
        --method)
            SELECTED_METHOD="$2"
            shift 2
            ;;
        --all-methods)
            RUN_ALL_METHODS=true
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--all] [--method lora|pissa|dora|lora_plus] [--all-methods]"
            exit 1
            ;;
    esac
done

# 设置任务列表
if [ "$RUN_ALL" = true ]; then
    TASKS=("${SMALL_TASKS[@]}" "${BIG_TASKS[@]}")
    echo "Running ALL tasks: ${TASKS[@]}"
else
    TASKS=("${SMALL_TASKS[@]}")
    echo "Running SMALL tasks only: ${TASKS[@]}"
    echo "(Use --all to run all tasks)"
fi

# 设置方法列表
if [ "$RUN_ALL_METHODS" = true ]; then
    METHODS=("${ALL_METHODS[@]}")
    echo "Running ALL methods: ${METHODS[@]}"
else
    METHODS=("${SELECTED_METHOD}")
    echo "Running method: ${METHODS[@]}"
    echo "(Use --all-methods to run all methods, or --method <name> to select)"
fi

# 学习率
LORA_LR=5e-5
LORA_PLUS_LR_RATIO=10.0  # LoRA+ 的 A/B 学习率比例

# C-SPLoRA 共同参数
CSPLORA_R_MIN=2
CSPLORA_R_MAX_FACTOR=4.0
CSPLORA_GAMMA=2.0
CSPLORA_TAU=1.0

# 原始 C-SPLoRA 固定参数
ORIG_RHO=0.9
ORIG_STEPS=200

# 自适应 C-SPLoRA 参数
ADA_MIN_STEPS=50
ADA_MAX_STEPS=500
ADA_CONVERGENCE_THRESHOLD=0.02
ADA_CHECK_INTERVAL=10
ADA_PATIENCE=3
ADA_TOP_K=20
ADA_RANK_TOLERANCE=5
ADA_RHO_METHOD="coverage"
ADA_RHO_COVERAGE_TARGET=0.95
ADA_RHO_MIN=0.6
ADA_RHO_MAX=1.0

# 数据 / HF 设置
export HF_DISK_ROOT="/data/dhming/Second/lora-sb/data"
export WANDB_PROJECT="csplora_ada_comparison"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1

# 切换到项目目录
cd /data/dhming/Second/lora-sb

is_big_task() {
  local t="$1"
  for bt in "${BIG_TASKS[@]}"; do
    if [ "$bt" == "$t" ]; then
      return 0
    fi
  done
  return 1
}

# ============================================================
# 实验配置
# ============================================================

# 构建方法特定参数
get_method_args() {
    local method=$1
    local args=""

    case "${method}" in
        "lora")
            args="--method lora"
            ;;
        "pissa")
            args="--method pissa"
            ;;
        "dora")
            args="--method dora"
            ;;
        "lora_plus"|"lora+")
            args="--method lora_plus --lora_plus_lr_ratio ${LORA_PLUS_LR_RATIO}"
            ;;
        *)
            echo "Unknown method: ${method}"
            exit 1
            ;;
    esac

    echo "${args}"
}

run_experiment() {
    local task=$1
    local seed=$2
    local exp_type=$3  # "original", "adaptive", "ada_step_only", "ada_rho_only"
    local epochs=$4
    local method=$5

    # 获取方法特定参数
    local method_args=$(get_method_args "${method}")

    echo "===================================================="
    echo "[Experiment] method=${method}, type=${exp_type}, task=${task}, seed=${seed}"
    echo "===================================================="

    case "${exp_type}" in
        "original")
            # 原始 C-SPLoRA：固定 step=200, 固定 rho=0.9
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
                --csplora_rho ${ORIG_RHO} \
                --csplora_r_min ${CSPLORA_R_MIN} \
                --csplora_r_max_factor ${CSPLORA_R_MAX_FACTOR} \
                --csplora_gamma ${CSPLORA_GAMMA} \
                --csplora_tau_scale ${CSPLORA_TAU} \
                --probe_steps ${ORIG_STEPS} \
                --csplora_cache_dir "./csplora_cache_fixed_step" \
                --csplora_task_suffix "_orig"
            ;;

        "adaptive")
            # 自适应 C-SPLoRA：自适应 step + 自适应 rho
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_glue_ada.py \
                --task "${task}" \
                --model "${MODEL_PATH}" \
                --lora_r ${RANK} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_dropout ${LORA_DROPOUT} \
                --lr ${LORA_LR} \
                --seed ${seed} \
                ${method_args} \
                --epochs ${epochs} \
                --csplora_ada \
                --csplora_r_min ${CSPLORA_R_MIN} \
                --csplora_r_max_factor ${CSPLORA_R_MAX_FACTOR} \
                --csplora_gamma ${CSPLORA_GAMMA} \
                --csplora_tau_scale ${CSPLORA_TAU} \
                --ada_min_steps ${ADA_MIN_STEPS} \
                --ada_max_steps ${ADA_MAX_STEPS} \
                --ada_convergence_threshold ${ADA_CONVERGENCE_THRESHOLD} \
                --ada_check_interval ${ADA_CHECK_INTERVAL} \
                --ada_patience ${ADA_PATIENCE} \
                --ada_top_k ${ADA_TOP_K} \
                --ada_rank_tolerance ${ADA_RANK_TOLERANCE} \
                --ada_rho_method ${ADA_RHO_METHOD} \
                --ada_rho_coverage_target ${ADA_RHO_COVERAGE_TARGET} \
                --ada_rho_min ${ADA_RHO_MIN} \
                --ada_rho_max ${ADA_RHO_MAX} \
                --csplora_cache_dir "./csplora_cache_adaptive_step"
            ;;

        "ada_step_only")
            # 消融：只自适应 step，固定 rho
            # 与 adaptive 共享 cache (都是自适应 step)
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_glue_ada.py \
                --task "${task}" \
                --model "${MODEL_PATH}" \
                --lora_r ${RANK} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_dropout ${LORA_DROPOUT} \
                --lr ${LORA_LR} \
                --seed ${seed} \
                ${method_args} \
                --epochs ${epochs} \
                --csplora_ada \
                --csplora_r_min ${CSPLORA_R_MIN} \
                --csplora_r_max_factor ${CSPLORA_R_MAX_FACTOR} \
                --csplora_gamma ${CSPLORA_GAMMA} \
                --csplora_tau_scale ${CSPLORA_TAU} \
                --ada_min_steps ${ADA_MIN_STEPS} \
                --ada_max_steps ${ADA_MAX_STEPS} \
                --ada_convergence_threshold ${ADA_CONVERGENCE_THRESHOLD} \
                --ada_check_interval ${ADA_CHECK_INTERVAL} \
                --ada_patience ${ADA_PATIENCE} \
                --ada_top_k ${ADA_TOP_K} \
                --ada_rank_tolerance ${ADA_RANK_TOLERANCE} \
                --ada_no_adaptive_rho \
                --csplora_rho ${ORIG_RHO} \
                --csplora_cache_dir "./csplora_cache_adaptive_step"
            ;;

        "ada_rho_only")
            # 消融：固定 step，只自适应 rho
            # 与 original 共享 cache (都是固定 step=200)
            CUDA_VISIBLE_DEVICES=${GPU_ID} python3 train_glue_ada.py \
                --task "${task}" \
                --model "${MODEL_PATH}" \
                --lora_r ${RANK} \
                --lora_alpha ${LORA_ALPHA} \
                --lora_dropout ${LORA_DROPOUT} \
                --lr ${LORA_LR} \
                --seed ${seed} \
                ${method_args} \
                --epochs ${epochs} \
                --csplora_ada \
                --csplora_r_min ${CSPLORA_R_MIN} \
                --csplora_r_max_factor ${CSPLORA_R_MAX_FACTOR} \
                --csplora_gamma ${CSPLORA_GAMMA} \
                --csplora_tau_scale ${CSPLORA_TAU} \
                --ada_no_adaptive_steps \
                --probe_steps ${ORIG_STEPS} \
                --ada_rho_method ${ADA_RHO_METHOD} \
                --ada_rho_coverage_target ${ADA_RHO_COVERAGE_TARGET} \
                --ada_rho_min ${ADA_RHO_MIN} \
                --ada_rho_max ${ADA_RHO_MAX} \
                --csplora_cache_dir "./csplora_cache_fixed_step"
            ;;
    esac

    echo "[Experiment] Finished: method=${method}, type=${exp_type}, task=${task}, seed=${seed}"
    echo ""
}

# ============================================================
# 主实验循环
# ============================================================

echo ""
echo "############################################################"
echo "# C-SPLoRA Adaptive vs Original Comparison Experiments"
echo "############################################################"
echo ""

# 实验类型列表
# 完整对比：EXP_TYPES=("original" "adaptive" "ada_step_only" "ada_rho_only")
# 简单对比：EXP_TYPES=("original" "adaptive")
EXP_TYPES=("adaptive")

for method in "${METHODS[@]}"; do
    echo ""
    echo "============================================================"
    echo "Running experiments for method: ${method}"
    echo "============================================================"
    echo ""

    for task in "${TASKS[@]}"; do
        # 按任务设置 epoch 数
        if is_big_task "${task}"; then
            EPOCHS=3
        else
            EPOCHS=10
        fi

        for exp_type in "${EXP_TYPES[@]}"; do
            for seed in "${SEEDS[@]}"; do
                run_experiment "${task}" "${seed}" "${exp_type}" "${EPOCHS}" "${method}"
            done
        done
    done
done

echo ""
echo "############################################################"
echo "# ALL EXPERIMENTS COMPLETED!"
echo "############################################################"
echo ""
echo "Summary:"
echo "  - Seeds: ${SEEDS[@]}"
echo "  - Tasks: ${TASKS[@]}"
echo "  - Methods: ${METHODS[@]}"
echo "  - Experiment types: ${EXP_TYPES[@]}"
echo "  - Total runs: $((${#TASKS[@]} * ${#EXP_TYPES[@]} * ${#SEEDS[@]} * ${#METHODS[@]}))"
echo ""
echo "Results logged to wandb project: ${WANDB_PROJECT}"
echo "============================================================"
