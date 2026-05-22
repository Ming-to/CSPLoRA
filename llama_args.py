import argparse


def get_arithmetic_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="lora",
                        choices=["full", "lora", "adalora", "pissa", "dora", "lora_plus", "lora+", "gora"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.0)

    # LoRA+ 专用参数
    parser.add_argument("--lora_plus_lr_ratio", type=float, default=16.0,
                        help="LoRA+ 中 LoRA A 矩阵相对于 B 的学习率放大倍率 η = lr_A / lr_B")

    # CSPLoRA
    parser.add_argument("--csplora", action="store_true")
    parser.add_argument("--csplora_rho", type=float, default=0.9)
    parser.add_argument("--csplora_gamma", type=float, default=2.0)
    parser.add_argument("--csplora_gamma_strategy", type=str, default="fixed",
                        choices=["none", "uniform", "fixed", "adaptive_std"])
    parser.add_argument("--csplora_gamma_scale", type=float, default=1.0)
    parser.add_argument("--csplora_gamma_max", type=float, default=10.0)
    parser.add_argument("--csplora_gamma_min_std", type=float, default=1e-4)
    parser.add_argument("--csplora_no_causal_weighting", action="store_true")
    parser.add_argument("--csplora_eps", type=float, default=1e-5)
    parser.add_argument("--csplora_r_min", type=int, default=2)
    parser.add_argument("--csplora_tau_scale", type=float, default=1.0)
    parser.add_argument("--csplora_skip_smooth", action="store_true")
    parser.add_argument("--csplora_skip_normalize", action="store_true")

    # Adaptive
    parser.add_argument("--ada_min_steps", type=int, default=50)
    parser.add_argument("--ada_max_steps", type=int, default=500)
    parser.add_argument("--ada_convergence_threshold", type=float, default=0.02)
    parser.add_argument("--ada_check_interval", type=int, default=10)
    parser.add_argument("--ada_patience", type=int, default=3)
    parser.add_argument("--ada_top_k", type=int, default=20)
    parser.add_argument("--ada_rank_tolerance", type=int, default=5)
    parser.add_argument("--ada_no_adaptive_steps", action="store_true")
    parser.add_argument("--ada_no_adaptive_rho", action="store_true")
    parser.add_argument("--ada_rho_method", type=str, default="coverage",
                        choices=["coverage", "entropy", "gini", "elbow"])
    parser.add_argument("--ada_rho_min", type=float, default=0.6)
    parser.add_argument("--ada_rho_max", type=float, default=1.0)
    parser.add_argument("--csplora_importance_metric", type=str, default="taylor",
                        choices=["fisher", "taylor", "gora"])

    # GoRA
    parser.add_argument("--gora_r_min", type=int, default=1)
    parser.add_argument("--gora_r_max", type=int, default=64)
    parser.add_argument("--gora_gamma", type=float, default=1.0)
    parser.add_argument("--gora_num_steps", type=int, default=50)
    parser.add_argument("--gora_use_rslora", action="store_true", default=True)
    parser.add_argument("--gora_no_rslora", action="store_true")
    parser.add_argument("--gora_use_pinv_init", action="store_true", default=True)
    parser.add_argument("--gora_no_pinv_init", action="store_true")

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./experiments_llama")
    parser.add_argument("--grad_acc_steps", type=int, default=32)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine"])
    parser.add_argument("--project_name", type=str, default="llama_arithmetic_baselines")
    parser.add_argument("--run_name", type=str, default=None)

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--dataset_split", type=str, default="train[:100000]")
    parser.add_argument("--dataset_field", type=str, nargs=2, default=["instruction", "output"])
    parser.add_argument("--count_params_only", action="store_true")

    args = parser.parse_args()
    return args


def get_cr_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--method", type=str, default="lora",
                        choices=["full", "lora", "adalora", "pissa", "dora", "lora_plus", "lora+", "gora"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # LoRA+ 专用参数
    parser.add_argument("--lora_plus_lr_ratio", type=float, default=16.0,
                        help="LoRA+ 中 LoRA A 矩阵相对于 B 的学习率放大倍率 η = lr_A / lr_B")

    # CSPLoRA
    parser.add_argument("--csplora", action="store_true")
    parser.add_argument("--csplora_rho", type=float, default=0.9)
    parser.add_argument("--csplora_gamma", type=float, default=2.0)
    parser.add_argument("--csplora_gamma_strategy", type=str, default="fixed",
                        choices=["none", "uniform", "fixed", "adaptive_std"])
    parser.add_argument("--csplora_gamma_scale", type=float, default=1.0)
    parser.add_argument("--csplora_gamma_max", type=float, default=10.0)
    parser.add_argument("--csplora_gamma_min_std", type=float, default=1e-4)
    parser.add_argument("--csplora_no_causal_weighting", action="store_true")
    parser.add_argument("--csplora_eps", type=float, default=1e-5)
    parser.add_argument("--csplora_r_min", type=int, default=2)
    parser.add_argument("--csplora_tau_scale", type=float, default=1.0)
    parser.add_argument("--csplora_skip_smooth", action="store_true")
    parser.add_argument("--csplora_skip_normalize", action="store_true")
    parser.add_argument("--csplora_layer_normalize", action="store_true")
    parser.add_argument("--csplora_task_suffix", type=str, default="")
    parser.add_argument("--csplora_cache_dir", type=str, default=None)

    # Adaptive
    parser.add_argument("--ada_min_steps", type=int, default=50)
    parser.add_argument("--ada_max_steps", type=int, default=500)
    parser.add_argument("--ada_convergence_threshold", type=float, default=0.02)
    parser.add_argument("--ada_check_interval", type=int, default=10)
    parser.add_argument("--ada_patience", type=int, default=3)
    parser.add_argument("--ada_top_k", type=int, default=20)
    parser.add_argument("--ada_rank_tolerance", type=int, default=5)
    parser.add_argument("--ada_no_adaptive_steps", action="store_true")
    parser.add_argument("--ada_no_adaptive_rho", action="store_true")
    parser.add_argument("--ada_rho_method", type=str, default="coverage",
                        choices=["coverage", "entropy", "gini", "elbow"])
    parser.add_argument("--ada_rho_min", type=float, default=0.6)
    parser.add_argument("--ada_rho_max", type=float, default=1.0)
    parser.add_argument("--csplora_importance_metric", type=str, default="taylor",
                        choices=["fisher", "taylor", "gora"])

    # GoRA
    parser.add_argument("--gora_r_min", type=int, default=1)
    parser.add_argument("--gora_r_max", type=int, default=64)
    parser.add_argument("--gora_gamma", type=float, default=1.0)
    parser.add_argument("--gora_num_steps", type=int, default=50)
    parser.add_argument("--gora_use_rslora", action="store_true", default=True)
    parser.add_argument("--gora_no_rslora", action="store_true")
    parser.add_argument("--gora_use_pinv_init", action="store_true", default=True)
    parser.add_argument("--gora_no_pinv_init", action="store_true")

    # Training
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--max_seq_length", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr_decay_ratio", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action="store_true")
    parser.add_argument("--attn_implementation", type=str, default=None,
                        choices=["eager", "sdpa", "flash_attention_2"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./experiments_llama")
    parser.add_argument("--exp_name", type=str, default=None)
    parser.add_argument("--grad_acc_steps", type=int, default=32)
    parser.add_argument("--scheduler", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--project_name", type=str, default="llama_cr_baselines")
    parser.add_argument("--run_name", type=str, default=None)

    # Data
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--train_on_inputs", action="store_true")
    parser.add_argument("--count_params_only", action="store_true")

    args = parser.parse_args()
    return args


def get_eval_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--base_model", type=str, required=True)
    parser.add_argument("--ckpt_dir", type=str, required=True)
    parser.add_argument("--eval_task", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--use_flash_attn", action="store_true")
    parser.add_argument("--use_compile", action="store_true")
    parser.add_argument("--output_file", type=str, default="eval_result.jsonl")
    parser.add_argument("--prompt_style", type=str, default="assistant", choices=["assistant", "instruction"])
    parser.add_argument("--eval_data_dir", type=str, default=None)
    parser.add_argument("--eval_data_file", type=str, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--mtbench_data", type=str, default=None)
    parser.add_argument("--judge_model", type=str, default=None)
    parser.add_argument("--count_params_only", action="store_true")

    return parser.parse_args()
