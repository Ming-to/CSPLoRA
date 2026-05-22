import argparse


def get_glue_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--task", type=str, default="cola")
    parser.add_argument("--model", type=str, default="roberta-large")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.06)
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--method", type=str, default="lora",
                        choices=["full", "lora", "lora-sb", "adalora", "dora", "pissa", "lora_plus", "lora+", "gora"])
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--amp_dtype", type=str, default="none", choices=["none", "fp16", "bf16"])
    parser.add_argument("--lora_plus_lr_ratio", type=float, default=10.0)

    # GoRA
    parser.add_argument("--gora_r_min", type=int, default=1)
    parser.add_argument("--gora_r_max", type=int, default=64)
    parser.add_argument("--gora_gamma", type=float, default=1.0)
    parser.add_argument("--gora_num_steps", type=int, default=50)
    parser.add_argument("--gora_use_rslora", action="store_true", default=True)
    parser.add_argument("--gora_no_rslora", action="store_true")
    parser.add_argument("--gora_use_pinv_init", action="store_true", default=True)
    parser.add_argument("--gora_no_pinv_init", action="store_true")

    # CSPLoRA
    parser.add_argument("--csplora", action="store_true")
    parser.add_argument("--csplora_cache_dir", type=str, default="./csplora_cache")
    parser.add_argument("--csplora_task_suffix", type=str, default="")
    parser.add_argument("--csplora_R_tot", type=int, default=-1)
    parser.add_argument("--csplora_rho", type=float, default=1.0)
    parser.add_argument("--csplora_gamma", type=float, default=1.0)
    parser.add_argument("--csplora_gamma_strategy", type=str, default="fixed",
                        choices=["none", "uniform", "fixed", "adaptive_std"])
    parser.add_argument("--csplora_gamma_scale", type=float, default=1.0)
    parser.add_argument("--csplora_gamma_max", type=float, default=10.0)
    parser.add_argument("--csplora_gamma_min_std", type=float, default=1e-4)
    parser.add_argument("--csplora_no_causal_weighting", action="store_true")
    parser.add_argument("--csplora_r_min", type=int, default=2)
    parser.add_argument("--csplora_r_max_factor", type=float, default=4.0)
    parser.add_argument("--csplora_tau_scale", type=float, default=1.0)
    parser.add_argument("--csplora_skip_smooth", action="store_true")
    parser.add_argument("--csplora_skip_normalize", action="store_true")

    # Probe
    parser.add_argument("--probe_rank", type=int, default=2)
    parser.add_argument("--probe_lr", type=float, default=5e-4)
    parser.add_argument("--probe_steps", type=int, default=200)

    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--csplora_rank_pattern", type=str, default=None)

    # Adaptive
    parser.add_argument("--ada_min_steps", type=int, default=50)
    parser.add_argument("--ada_max_steps", type=int, default=500)
    parser.add_argument("--ada_convergence_threshold", type=float, default=0.01)
    parser.add_argument("--ada_check_interval", type=int, default=10)
    parser.add_argument("--ada_patience", type=int, default=3)
    parser.add_argument("--ada_top_k", type=int, default=20)
    parser.add_argument("--ada_rank_tolerance", type=int, default=5)
    parser.add_argument("--ada_rho_method", type=str, default="coverage",
                        choices=["coverage", "entropy", "gini", "elbow"])
    parser.add_argument("--ada_rho_coverage_target", type=float, default=0.95)
    parser.add_argument("--ada_rho_min", type=float, default=0.6)
    parser.add_argument("--ada_rho_max", type=float, default=1.0)
    parser.add_argument("--ada_no_adaptive_steps", action="store_true")
    parser.add_argument("--ada_no_adaptive_rho", action="store_true")
    parser.add_argument("--count_params_only", action="store_true")

    args = parser.parse_args()
    return args
