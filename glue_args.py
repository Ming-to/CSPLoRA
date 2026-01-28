# glue_args.py
import argparse

def get_glue_args():
    parser = argparse.ArgumentParser()

    # ========= 基本训练参数 =========
    parser.add_argument(
        "--task",
        type=str,
        default="cola",
        help="GLUE task to fine-tune on",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="roberta-large",  # RoBERTa-large 作为默认 backbone
        help="Model name or local path (default: roberta-large)",
    )

    # LoRA 相关（RoBERTa-large + GLUE 的“标准基线”）
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank r (baseline: 8 for RoBERTa-large on GLUE)",
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=16,
        help="LoRA alpha (baseline: 16 = 2 * r)",
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (baseline: 0.1)",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,  # RoBERTa-large 常用 32
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (实际按任务在 .sh 里覆盖：大任务3，小任务10)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum number of training steps (overrides epochs if set)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.06,
        help="Warmup ratio for linear schedule (baseline: 0.06)",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=2e-4,  # LoRA 系列 GLUE 上常用量级
        help="Learning rate",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (具体会在 .sh 里用不同 seed 多次调用)",
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device (cuda / cpu)',
    )

    parser.add_argument(
        "--method",
        type=str,
        default="lora",   # 默认 LoRA baseline
        choices=["full", "lora", "lora-sb", "adalora", "dora", "pissa", "lora_plus", "lora+", "gora"],
        help="fine-tuning method: full / lora / lora-sb / adalora / dora / pissa / gora",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10,
        help="Log to wandb every N steps instead of every step."
    )
    parser.add_argument(
        "--amp_dtype",
        type=str,
        default="none",
        choices=["none", "fp16", "bf16"],
        help="Use AMP mixed precision: none / fp16 / bf16"
    )
    # LoRA+ 专用：η = lr_A / lr_B
    parser.add_argument(
        "--lora_plus_lr_ratio",
        type=float,
        default=10.0,
        help="LoRA+ 中 LoRA A 矩阵相对于 B 的学习率放大倍率 η = lr_A / lr_B",
    )

    # ========= GoRA 相关参数 =========
    parser.add_argument(
        "--gora_r_min",
        type=int,
        default=1,
        help="GoRA minimum rank per layer (default: 1)",
    )
    parser.add_argument(
        "--gora_r_max",
        type=int,
        default=64,
        help="GoRA maximum rank per layer (default: 64)",
    )
    parser.add_argument(
        "--gora_gamma",
        type=float,
        default=1.0,
        help="GoRA scaling factor for B initialization (default: 1.0)",
    )
    parser.add_argument(
        "--gora_num_steps",
        type=int,
        default=50,
        help="GoRA number of gradient accumulation steps (default: 50)",
    )
    parser.add_argument(
        "--gora_use_rslora",
        action="store_true",
        default=True,
        help="GoRA use rsLoRA scaling (alpha/sqrt(r) instead of alpha/r)",
    )
    parser.add_argument(
        "--gora_no_rslora",
        action="store_true",
        help="GoRA disable rsLoRA scaling",
    )
    parser.add_argument(
        "--gora_use_pinv_init",
        action="store_true",
        default=True,
        help="GoRA use pseudo-inverse initialization for B matrix",
    )
    parser.add_argument(
        "--gora_no_pinv_init",
        action="store_true",
        help="GoRA disable pseudo-inverse initialization",
    )

    # ========= CSpLoRA 相关参数 =========
    parser.add_argument(
        "--csplora",
        action="store_true",
        help="Enable CSpLoRA rank planning for LoRA-family methods.",
    )
    parser.add_argument(
        "--csplora_cache_dir",
        type=str,
        default="./csplora_cache",
        help="Directory to cache CSpLoRA probe scores.",
    )
    parser.add_argument(
        "--csplora_task_suffix",
        type=str,
        default="",
        help="Suffix to append to task name for CSpLoRA cache (e.g., '_v2' to use separate cache).",
    )

    # 预算控制：默认为 -1 (Auto)，如果大于0则为手动指定
    parser.add_argument(
        "--csplora_R_tot",
        type=int,
        default=-1,
        help="Total rank budget. Set to -1 to auto-calculate (Layers * lora_r).",
    )
    
    # 核心分配超参 (按讨论调整后的默认值)
    parser.add_argument(
        "--csplora_rho",
        type=float,
        default=1.0,
        help="Coverage ratio rho for layer selection (0-1). "
             "1.0 = use all layers; 0.9 = select layers covering 90%% cumulative gradient energy.",
    )
    parser.add_argument(
        "--csplora_gamma",
        type=float,
        default=1.0,
        help="Gamma for confidence-based sample weighting (default: 2.0).",
    )
    parser.add_argument(
        "--csplora_r_min",
        type=int,
        default=2,
        help="Minimal rank per selected site (default: 2).",
    )
    parser.add_argument(
        "--csplora_r_max_factor",
        type=float,
        default=4.0,
        help="Max rank factor relative to base rank (default: 4.0).",
    )
    parser.add_argument(
        "--csplora_tau_scale",
        type=float,
        default=1.0,
        help="Tau scale for water-filling saturation (default: 1.0).",
    )
    parser.add_argument(
        "--csplora_skip_smooth",
        action="store_true",
        help="Skip log1p smoothing after normalization (ablation experiment).",
    )
    parser.add_argument(
        "--csplora_skip_normalize",
        action="store_true",
        help="Skip per-module-type normalization (ablation experiment).",
    )

    # Probe 阶段超参
    parser.add_argument(
        "--probe_rank",
        type=int,
        default=2,
        help="Probe LoRA rank r_probe.",
    )
    parser.add_argument(
        "--probe_lr",
        type=float,
        default=5e-4,
        help="Learning rate during probe phase.",
    )
    parser.add_argument(
        "--probe_steps",
        type=int,
        default=200,
        help="Max steps for probing.",
    )

    # ========= 输出和额外参数 =========
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory for saving model checkpoints and results.",
    )
    parser.add_argument(
        "--csplora_rank_pattern",
        type=str,
        default=None,
        help="Path to a JSON file containing pre-computed rank pattern (skips probe phase).",
    )

    # ========= Adaptive C-SPLoRA 参数 =========
    parser.add_argument(
        "--csplora_ada",
        action="store_true",
        help="Use adaptive C-SPLoRA (auto step and rho selection).",
    )
    parser.add_argument(
        "--ada_min_steps",
        type=int,
        default=50,
        help="Minimum probe steps for adaptive C-SPLoRA.",
    )
    parser.add_argument(
        "--ada_max_steps",
        type=int,
        default=500,
        help="Maximum probe steps for adaptive C-SPLoRA.",
    )
    parser.add_argument(
        "--ada_convergence_threshold",
        type=float,
        default=0.01,
        help="GoRA-style convergence threshold (1%% relative energy change).",
    )
    parser.add_argument(
        "--ada_check_interval",
        type=int,
        default=10,
        help="Check convergence every N steps.",
    )
    parser.add_argument(
        "--ada_patience",
        type=int,
        default=3,
        help="Require N consecutive stable checks before convergence (GoRA-style).",
    )
    parser.add_argument(
        "--ada_top_k",
        type=int,
        default=20,
        help="Monitor top-K layers for ranking stability in convergence detection.",
    )
    parser.add_argument(
        "--ada_rank_tolerance",
        type=int,
        default=5,
        help="Allow rank changes up to this amount within top-K layers.",
    )
    parser.add_argument(
        "--ada_rho_method",
        type=str,
        default="coverage",
        choices=["coverage", "entropy", "gini", "elbow"],
        help="Method for adaptive rho selection: coverage (recommended) / entropy / gini / elbow.",
    )
    parser.add_argument(
        "--ada_rho_coverage_target",
        type=float,
        default=0.95,
        help="Target energy coverage for 'coverage' rho method (default: 95%%).",
    )
    parser.add_argument(
        "--ada_rho_min",
        type=float,
        default=0.6,
        help="Minimum rho for adaptive selection.",
    )
    parser.add_argument(
        "--ada_rho_max",
        type=float,
        default=1.0,
        help="Maximum rho for adaptive selection.",
    )
    parser.add_argument(
        "--ada_no_adaptive_steps",
        action="store_true",
        help="Disable adaptive steps (use fixed probe_steps).",
    )
    parser.add_argument(
        "--ada_no_adaptive_rho",
        action="store_true",
        help="Disable adaptive rho (use fixed csplora_rho).",
    )
    parser.add_argument(
        "--count_params_only",
        action="store_true",
        help="只统计参数量后退出，不进行训练（用于快速获取参数量）",
    )

    args = parser.parse_args()
    return args