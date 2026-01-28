# llama_args.py
import argparse

def add_llama_common_args(parser: argparse.ArgumentParser):
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["full", "lora", "adalora", "pissa", "dora", "lora_plus", "lora+", "gora"],
    )
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # CSpLoRA 开关 + 超参
    parser.add_argument("--csplora", action="store_true",
                        help="Enable CSpLoRA rank planning.")
    parser.add_argument("--csplora_rho", type=float, default=0.9)
    parser.add_argument("--csplora_gamma", type=float, default=2.0)
    parser.add_argument("--csplora_eps", type=float, default=1e-5)
    parser.add_argument("--csplora_r_min", type=int, default=2)
    parser.add_argument("--csplora_tau_scale", type=float, default=1.0)
    parser.add_argument("--csplora_skip_smooth", action="store_true",
                        help="Skip log1p smoothing after normalization.")

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


def get_arithmetic_args():
    parser = argparse.ArgumentParser()

    # ========= 基本模型与方法 =========
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="预训练 LLaMA 模型本地路径，例如 /data/share_weight/Llama-2-7b-hf",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["full", "lora", "adalora", "pissa", "dora", "lora_plus", "lora+", "gora"],
        help="微调方式：full / LoRA / AdaLoRA / PiSSA / DoRA / LoRA+",
    )

    # ========= LoRA / PEFT 参数 =========
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # ========= CSpLoRA 相关开关（先占位，不一定在大模型上用） =========
    parser.add_argument(
        "--csplora",
        action="store_true",
        help="打开时启用 CSpLoRA（目前 arithmetic 脚本里只是占位，不执行 probe）",
    )
    parser.add_argument("--csplora_rho", type=float, default=0.9)
    parser.add_argument("--csplora_gamma", type=float, default=2.0)
    parser.add_argument("--csplora_eps", type=float, default=1e-5)
    parser.add_argument("--csplora_r_min", type=int, default=2)
    parser.add_argument(
        "--csplora_tau_scale",
        type=float,
        default=1.0,
        help="理论里 tau 的缩放因子，占位参数",
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

    # ========= Adaptive CSpLoRA 参数 =========
    parser.add_argument(
        "--csplora_ada",
        action="store_true",
        help="启用 Adaptive CSpLoRA（自适应 probe steps + 自适应 rho）",
    )
    parser.add_argument("--ada_min_steps", type=int, default=50,
                        help="Adaptive probe: minimum probe steps")
    parser.add_argument("--ada_max_steps", type=int, default=500,
                        help="Adaptive probe: maximum probe steps")
    parser.add_argument("--ada_convergence_threshold", type=float, default=0.02,
                        help="Adaptive probe: energy relative change threshold (default: 2%)")
    parser.add_argument("--ada_check_interval", type=int, default=10,
                        help="Adaptive probe: check convergence every N steps")
    parser.add_argument("--ada_patience", type=int, default=3,
                        help="Adaptive probe: require N consecutive stable checks")
    parser.add_argument("--ada_top_k", type=int, default=20,
                        help="Adaptive probe: monitor top-K layers for ranking stability")
    parser.add_argument("--ada_rank_tolerance", type=int, default=5,
                        help="Adaptive probe: allowed rank changes in top-K")
    parser.add_argument("--ada_no_adaptive_steps", action="store_true",
                        help="Disable adaptive step selection (use fixed probe_steps)")
    parser.add_argument("--ada_no_adaptive_rho", action="store_true",
                        help="Disable adaptive rho selection (use fixed csplora_rho)")
    parser.add_argument("--ada_rho_method", type=str, default="coverage",
                        choices=["coverage", "entropy", "gini", "elbow"],
                        help="Adaptive rho selection method")
    parser.add_argument("--ada_rho_min", type=float, default=0.6,
                        help="Adaptive rho: minimum rho value")
    parser.add_argument("--ada_rho_max", type=float, default=1.0,
                        help="Adaptive rho: maximum rho value")
    parser.add_argument("--csplora_importance_metric", type=str, default="taylor",
                        choices=["fisher", "taylor", "gora"],
                        help="Importance metric for probe: taylor (|g*W|, default), fisher (||g||^2), gora (mean|g*W|)")

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

    # ========= 训练超参 =========
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of training steps (overrides epochs if set)")
    parser.add_argument("--max_seq_length", type=int, default=512)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_decay_ratio",
        type=float,
        default=0.0,
        help=(
            "Learning-rate decay ratio (GoRA: Llama3=0.1, Llama2=0.0). "
            "When using cosine decay, this is interpreted as the minimum LR rate "
            "(min_lr = lr * lr_decay_ratio)."
        ),
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer (GoRA: Llama3=5e-4, Llama2=0)")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory (slower).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention backend. `sdpa` uses torch scaled_dot_product_attention; "
            "`flash_attention_2` requires flash-attn installed."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments_llama",
        help="训练输出目录（run_arithmetic.sh 会传进来）",
    )

    # ========= 训练策略（.sh 里用到的几个） =========
    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=32,
        help="梯度累积步数，对应 run_arithmetic.sh 里的 --grad_acc_steps",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine"],
        help="学习率调度器类型",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="llama_arithmetic_baselines",
        help="wandb 项目名",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="wandb run 名称（不指定就用时间戳自动生成）",
    )

    # ========= 数据相关（.sh 传进来的） =========
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="训练数据的本地路径，比如 data/arithmetic/math_10k.json",
    )

    parser.add_argument(
        "--dataset_split",
        type=str,
        default="train[:100000]",
        help='使用 HF 切片语法的子集选择，例如 "train[:100000]"',
    )

    parser.add_argument(
        "--dataset_field",
        type=str,
        nargs=2,
        default=["instruction", "output"],
        help="数据集中作为 (input, output) 的字段名，run_arithmetic.sh 里传的是: --dataset_field instruction output",
    )

    parser.add_argument(
        "--count_params_only",
        action="store_true",
        help="只统计参数量后退出，不进行训练（用于快速获取参数量）",
    )

    args = parser.parse_args()
    return args


def get_cr_args():
    """Commonsense Reasoning 任务的参数解析器"""
    parser = argparse.ArgumentParser()

    # ========= 基本模型与方法 =========
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="预训练 LLaMA 模型本地路径，例如 /data/share_weight/llama2-7b-hf",
    )

    parser.add_argument(
        "--method",
        type=str,
        default="lora",
        choices=["full", "lora", "adalora", "pissa", "dora", "lora_plus", "lora+", "gora"],
        help="微调方式：full / LoRA / AdaLoRA / PiSSA / DoRA / LoRA+",
    )

    # ========= LoRA / PEFT 参数 =========
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # ========= CSpLoRA 相关开关 =========
    parser.add_argument(
        "--csplora",
        action="store_true",
        help="打开时启用 CSpLoRA（目前 CR 脚本里只是占位，不执行 probe）",
    )
    parser.add_argument("--csplora_rho", type=float, default=0.9)
    parser.add_argument("--csplora_gamma", type=float, default=2.0)
    parser.add_argument("--csplora_eps", type=float, default=1e-5)
    parser.add_argument("--csplora_r_min", type=int, default=2)
    parser.add_argument("--csplora_tau_scale", type=float, default=1.0)
    parser.add_argument("--csplora_skip_smooth", action="store_true",
                        help="Skip log1p smoothing after normalization.")
    parser.add_argument("--csplora_skip_normalize", action="store_true",
                        help="Skip per-module-type normalization (ablation experiment).")
    parser.add_argument("--csplora_layer_normalize", action="store_true",
                        help="Enable layer-wise normalization (÷cap). Disabled by default for Taylor metric.")
    parser.add_argument("--csplora_task_suffix", type=str, default="",
                        help="Suffix for CSpLoRA cache task name (for ablation experiments).")
    parser.add_argument("--csplora_cache_dir", type=str, default=None,
                        help="Custom cache directory for CSpLoRA (overrides default).")

    # ========= Adaptive CSpLoRA 参数 =========
    parser.add_argument(
        "--csplora_ada",
        action="store_true",
        help="启用 Adaptive CSpLoRA（自适应 probe steps + 自适应 rho）",
    )
    parser.add_argument("--ada_min_steps", type=int, default=50,
                        help="Adaptive probe: minimum probe steps")
    parser.add_argument("--ada_max_steps", type=int, default=500,
                        help="Adaptive probe: maximum probe steps")
    parser.add_argument("--ada_convergence_threshold", type=float, default=0.02,
                        help="Adaptive probe: energy relative change threshold (default: 2%)")
    parser.add_argument("--ada_check_interval", type=int, default=10,
                        help="Adaptive probe: check convergence every N steps")
    parser.add_argument("--ada_patience", type=int, default=3,
                        help="Adaptive probe: require N consecutive stable checks")
    parser.add_argument("--ada_top_k", type=int, default=20,
                        help="Adaptive probe: monitor top-K layers for ranking stability")
    parser.add_argument("--ada_rank_tolerance", type=int, default=5,
                        help="Adaptive probe: allowed rank changes in top-K")
    parser.add_argument("--ada_no_adaptive_steps", action="store_true",
                        help="Disable adaptive step selection (use fixed probe_steps)")
    parser.add_argument("--ada_no_adaptive_rho", action="store_true",
                        help="Disable adaptive rho selection (use fixed csplora_rho)")
    parser.add_argument("--ada_rho_method", type=str, default="coverage",
                        choices=["coverage", "entropy", "gini", "elbow"],
                        help="Adaptive rho selection method")
    parser.add_argument("--ada_rho_min", type=float, default=0.6,
                        help="Adaptive rho: minimum rho value")
    parser.add_argument("--ada_rho_max", type=float, default=1.0,
                        help="Adaptive rho: maximum rho value")
    parser.add_argument("--csplora_importance_metric", type=str, default="taylor",
                        choices=["fisher", "taylor", "gora"],
                        help="Importance metric for probe: taylor (|g*W|, default), fisher (||g||^2), gora (mean|g*W|)")

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

    # ========= 训练超参 =========
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum number of training steps (overrides epochs if set)")
    parser.add_argument("--max_seq_length", type=int, default=256)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--lr_decay_ratio",
        type=float,
        default=0.0,
        help=(
            "Learning-rate decay ratio (GoRA: Llama3=0.1, Llama2=0.0). "
            "When using cosine decay, this is interpreted as the minimum LR rate "
            "(min_lr = lr * lr_decay_ratio)."
        ),
    )
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="Weight decay for AdamW optimizer (GoRA: Llama3=5e-4, Llama2=0)")
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Enable gradient checkpointing to reduce activation memory (slower).",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default=None,
        choices=["eager", "sdpa", "flash_attention_2"],
        help=(
            "Attention backend. `sdpa` uses torch scaled_dot_product_attention; "
            "`flash_attention_2` requires flash-attn installed."
        ),
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./experiments_llama",
        help="训练输出目录",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="自定义实验名称，用于 ablation 实验。指定后输出路径为 {output_dir}/{exp_name}/",
    )

    parser.add_argument(
        "--grad_acc_steps",
        type=int,
        default=32,
        help="梯度累积步数",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="linear",
        choices=["linear", "cosine"],
        help="学习率调度器类型",
    )

    parser.add_argument(
        "--project_name",
        type=str,
        default="llama_cr_baselines",
        help="wandb 项目名",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="wandb run 名称（不指定就用时间戳自动生成）",
    )

    # ========= 数据相关 =========
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="训练数据的本地路径，比如 data/commonsense/commonsense_170k.json",
    )

    parser.add_argument(
        "--train_on_inputs",
        action="store_true",
        help="是否把输入也算进 loss",
    )

    parser.add_argument(
        "--count_params_only",
        action="store_true",
        help="只统计参数量后退出，不进行训练（用于快速获取参数量）",
    )

    args = parser.parse_args()
    return args


def get_eval_args():
    parser = argparse.ArgumentParser()

    # 选择 base 模型（一般就是你训练时的 args.model）
    parser.add_argument("--base_model", type=str, required=True)

    # 选择 finetune 后的 checkpoint 路径（就是上面说的 final_model 目录）
    parser.add_argument("--ckpt_dir", type=str, required=True)

    # 评测哪个任务：gsm8k / math / humaneval / mtbench / boolq / piqa / siqa / hellaswag
    parser.add_argument("--eval_task", type=str, required=True)

    # 通用推理参数
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)

    # 加速优化选项
    parser.add_argument("--use_flash_attn", action="store_true",
                        help="Use Flash Attention 2 (requires flash-attn package)")
    parser.add_argument("--use_compile", action="store_true",
                        help="Use torch.compile for model optimization (PyTorch 2.0+)")

    # 输出结果路径
    parser.add_argument("--output_file", type=str, default="eval_result.jsonl")

    # Prompt 风格（评测时可选）
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="assistant",
        choices=["assistant", "instruction"],
        help="Eval prompt style: 'assistant' uses task-specific prompt; 'instruction' wraps with training-style template.",
    )

    # ========= Eval 数据路径（可选） =========
    # 默认会尝试：args.eval_data_dir -> $HF_DISK_ROOT -> <repo>/data
    # 其中 commonsense eval 期望在：<eval_data_dir>/commonsense_eval/
    parser.add_argument(
        "--eval_data_dir",
        type=str,
        default=None,
        help="Local eval data root directory (optional).",
    )
    parser.add_argument(
        "--eval_data_file",
        type=str,
        default=None,
        help="Optional explicit eval data file path (overrides eval_data_dir).",
    )
    parser.add_argument(
        "--max_eval_samples",
        type=int,
        default=None,
        help="Optional cap on number of eval samples.",
    )

    # ========= MT-Bench 专用参数 =========
    parser.add_argument(
        "--mtbench_data",
        type=str,
        default=None,
        help="MT-Bench 问题数据路径（可选，不指定则使用内置精简版）",
    )
    parser.add_argument(
        "--judge_model",
        type=str,
        default=None,
        help="用于评分的 judge 模型路径（MT-Bench 评测时使用）",
    )
    parser.add_argument(
        "--count_params_only",
        action="store_true",
        help="只统计参数量后退出，不进行训练（用于快速获取参数量）",
    )

    return parser.parse_args()
