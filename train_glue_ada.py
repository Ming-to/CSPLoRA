"""
Training script for GLUE with Adaptive C-SPLoRA.

This script uses AdaptiveCSpLoRAPlanner which features:
1. Adaptive probe steps (convergence-based early stopping)
2. Adaptive rho/top-p (entropy-based layer coverage selection)

Usage:
    python train_glue_ada.py --task mrpc --method lora --csplora_ada
"""

import torch
from torch.utils.data import DataLoader
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from datasets import load_dataset
from tqdm.auto import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import argparse
import warnings
import os
from datetime import datetime
import numpy as np
import wandb
from train_eval import *
import json
import yaml
import atexit

from utils.data_utils import *
from models import *
from utils.initialization_utils import *
from utils.gradient_utils import *
from utils.misc import *

from glue_args import get_glue_args
# Import both original and adaptive planners
from csplora_improved import CSpLoRAConfig, CSpLoRAPlanner
from csplora_ada import AdaptiveCSpLoRAConfig, AdaptiveCSpLoRAPlanner
from torch.amp import autocast

args = get_glue_args()

print("Using base model:", args.model)

# -------- Configure wandb --------
dataset_name = args.task
model_name = os.path.basename(str(args.model)).replace("/", "-")
method_name = args.method

# Add "ada" suffix if using adaptive
if getattr(args, "csplora_ada", False):
    run_name = f"{dataset_name}_{model_name}_{method_name}_ada"
else:
    run_name = f"{dataset_name}_{model_name}_{method_name}"

wandb.init(
    group=dataset_name,
    name=run_name,
    config=vars(args),
)

def cleanup_wandb():
    if wandb.run is not None:
        wandb.finish()

atexit.register(cleanup_wandb)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def finetune(task):

    GLUE_TASK_NUM_LABELS = {
        'cola': 2,
        'mnli': 3,
        'mrpc': 2,
        'qnli': 2,
        'qqp': 2,
        'rte': 2,
        'sst2': 2,
        'stsb': 1,
        'wnli': 2
    }

    num_labels = GLUE_TASK_NUM_LABELS[task]

    # Create model
    model, tokenizer = create_model_tokenizer(num_labels, args)

    # Data handling
    train_data, val_data, _ = load_and_preprocess_data(task=task, tokenizer=tokenizer, args=args)

    train_loader, val_loader = create_dataloader(train_data, args, shuffle=True), create_dataloader(val_data, args, shuffle=False)
    probe_loader = create_dataloader(train_data, args, shuffle=True)

    max_metric_1 = 0
    max_metric_2 = 0

    method = getattr(args, "method", "lora").lower()
    layer_ranks = None

    if method == "full":
        lora_config = None
        named_grads = None

    elif method in {"lora", "pissa", "dora", "adalora", "lora_plus", "lora+"}:
        named_grads = None
        layer_ranks = None

        # Check if we should use adaptive C-SPLoRA
        use_csplora = getattr(args, "csplora", False) or getattr(args, "csplora_ada", False)
        use_adaptive = getattr(args, "csplora_ada", False)

        if use_csplora:
            # Determine target modules
            target_modules = None
            if "roberta" in args.model or "bert" in args.model:
                target_modules = ["query", "value", "output.dense"]
            elif "llama" in args.model or "mistral" in args.model:
                target_modules = ["q_proj", "v_proj", "down_proj", "up_proj"]
            elif "t5" in args.model:
                target_modules = ["q", "k", "v", "o"]

            if target_modules is None:
                target_modules = ["query", "value"]
                print("[CSpLoRA] Warning: Could not auto-detect target modules, using default.")

            r_tot_val = args.csplora_R_tot if args.csplora_R_tot > 0 else None
            csplora_task_name = task + getattr(args, "csplora_task_suffix", "")

            if use_adaptive:
                # ========== Adaptive C-SPLoRA ==========
                print(f"[AdaCSpLoRA] Enabling Adaptive Rank Planning for task: {task}")

                ada_cfg = AdaptiveCSpLoRAConfig(
                    model_id=args.model,
                    task_name=csplora_task_name,
                    target_modules=target_modules,
                    task_type=TaskType.SEQ_CLS,
                    cache_dir=getattr(args, "csplora_cache_dir", "./csplora_cache_ada"),
                    device=args.device,

                    # Core allocation parameters
                    r_base=args.lora_r,
                    r_min=getattr(args, "csplora_r_min", 2),
                    r_max_factor=getattr(args, "csplora_r_max_factor", 4.0),
                    gamma=getattr(args, "csplora_gamma", 2.0),
                    tau_scale=getattr(args, "csplora_tau_scale", 1.0),
                    skip_smooth=getattr(args, "csplora_skip_smooth", False),
                    skip_normalize=getattr(args, "csplora_skip_normalize", False),
                    R_tot=r_tot_val,

                    # Probe parameters
                    probe_r=getattr(args, "probe_rank", 2),
                    probe_lr=getattr(args, "probe_lr", 5e-4),

                    # Adaptive step settings (hybrid convergence)
                    adaptive_steps=not getattr(args, "ada_no_adaptive_steps", False),
                    min_probe_steps=getattr(args, "ada_min_steps", 50),
                    max_probe_steps=getattr(args, "ada_max_steps", 500),
                    step_convergence_threshold=getattr(args, "ada_convergence_threshold", 0.02),
                    step_check_interval=getattr(args, "ada_check_interval", 10),
                    step_patience=getattr(args, "ada_patience", 3),
                    step_top_k=getattr(args, "ada_top_k", 20),
                    step_rank_tolerance=getattr(args, "ada_rank_tolerance", 5),

                    # Adaptive rho settings (coverage-based)
                    adaptive_rho=not getattr(args, "ada_no_adaptive_rho", False),
                    rho_min=getattr(args, "ada_rho_min", 0.6),
                    rho_max=getattr(args, "ada_rho_max", 1.0),
                    rho_method=getattr(args, "ada_rho_method", "coverage"),
                    rho_coverage_target=getattr(args, "ada_rho_coverage_target", 0.95),
                    rho_fixed=getattr(args, "csplora_rho", 0.9),
                )

                planner = AdaptiveCSpLoRAPlanner(ada_cfg)
                layer_ranks, scores = planner.plan(model, probe_loader)

                # Log adaptive stats to wandb
                if hasattr(planner, 'adaptive_stats') and planner.adaptive_stats:
                    stats = planner.adaptive_stats
                    wandb.log({
                        "ada_actual_probe_steps": stats.get('actual_probe_steps', -1),
                        "ada_converged": stats.get('converged', False),
                        "ada_final_rho": stats.get('final_rho', -1),
                        "ada_rho_method": stats.get('rho_method', 'unknown'),
                        # Effective-layers based stats (coverage method)
                        "ada_entropy": stats.get('entropy', -1),
                        "ada_norm_entropy": stats.get('norm_entropy', -1),
                        "ada_effective_layers": stats.get('effective_layers', -1),
                        "ada_effective_ratio": stats.get('effective_ratio', -1),
                        "ada_total_layers": stats.get('total_layers', -1),
                        # Gini stats (if gini method used)
                        "ada_gini": stats.get('gini', -1),
                    })

            else:
                # ========== Original C-SPLoRA ==========
                print(f"[CSpLoRA] Enabling Rank Planning for task: {task}")

                cs_cfg = CSpLoRAConfig(
                    model_id=args.model,
                    task_name=csplora_task_name,
                    target_modules=target_modules,
                    task_type=TaskType.SEQ_CLS,
                    cache_dir=getattr(args, "csplora_cache_dir", "./csplora_cache"),
                    device=args.device,

                    r_base=args.lora_r,
                    r_min=getattr(args, "csplora_r_min", 2),
                    r_max_factor=getattr(args, "csplora_r_max_factor", 4.0),
                    rho=getattr(args, "csplora_rho", 0.9),
                    gamma=getattr(args, "csplora_gamma", 2.0),
                    tau_scale=getattr(args, "csplora_tau_scale", 1.0),
                    skip_smooth=getattr(args, "csplora_skip_smooth", False),
                    skip_normalize=getattr(args, "csplora_skip_normalize", False),
                    R_tot=r_tot_val,

                    probe_r=getattr(args, "probe_rank", 2),
                    probe_lr=getattr(args, "probe_lr", 5e-4),
                    max_probe_steps=getattr(args, "probe_steps", 200),
                )

                planner = CSpLoRAPlanner(cs_cfg)
                layer_ranks, scores = planner.plan(model, probe_loader)

            # Log distribution
            if layer_ranks:
                print(f"[CSpLoRA] Planned ranks for {len(layer_ranks)} layers.")
                print(f"[CSpLoRA] Max rank: {max(layer_ranks.values())}, Min rank: {min(layer_ranks.values())}")
                wandb.log({
                    "csplora_num_layers": len(layer_ranks),
                    "csplora_max_rank": max(layer_ranks.values()),
                    "csplora_min_rank": min(layer_ranks.values()),
                    "csplora_avg_rank": sum(layer_ranks.values()) / len(layer_ranks),
                    "csplora_total_rank": sum(layer_ranks.values()),
                })

        # Create PEFT model
        model, lora_config = create_peft_model(model, args, rank_pattern=layer_ranks)
        model.to(args.device)

    else:
        raise ValueError(f"Unknown method: {method}")

    # Ensure contiguous parameters
    for param in model.parameters():
        param.data = param.data.contiguous()

    param_counts = count_parameters(model, verbose=False)

    total_params = param_counts['total_trainable_params']
    classifier_params = param_counts['classifier_params']
    non_classifier_params = param_counts['non_classifier_params']

    wandb.log({"total_params": total_params, "classifier_params": classifier_params, "non_classifier_params": non_classifier_params})

    # Setting up optimizer
    if method in ["lora_plus", "lora+"]:
        lora_A_params, lora_B_params, other_params = [], [], []

        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if "lora_A" in n:
                lora_A_params.append(p)
            elif "lora_B" in n:
                lora_B_params.append(p)
            else:
                other_params.append(p)

        eta = getattr(args, "lora_plus_lr_ratio", 10.0)
        lr_B = args.lr
        lr_A = lr_B * eta

        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr_B})
        if lora_A_params:
            param_groups.append({"params": lora_A_params, "lr": lr_A})
        if lora_B_params:
            param_groups.append({"params": lora_B_params, "lr": lr_B})

        optimizer = torch.optim.AdamW(param_groups)

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(train_loader) * args.epochs
    num_warmup_steps = int(total_steps * args.warmup_ratio)

    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps
    )

    for epoch in tqdm(range(args.epochs), desc='Epochs'):
        model.train()

        total_steps = len(train_loader)
        running_loss = 0

        progress_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', leave=False, total=total_steps)

        for step, data in progress_bar:
            data = {k: v.to(args.device, non_blocking=True) for k, v in data.items()}

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**data)
                loss = outputs.loss

            running_loss = loss.item()
            progress_bar.set_postfix({'loss': f'{running_loss:.4f}'})

            wandb.log({
                "train_loss": loss.detach().cpu().float().numpy(),
                "epoch": epoch,
                "step": step
            })

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        max_metric_1, max_metric_2 = evaluate_glue(
            model, val_loader, args, max_metric_1, max_metric_2
        )

    # Save eval results
    if getattr(args, "output_dir", None):
        os.makedirs(args.output_dir, exist_ok=True)
        eval_results = {
            "task": task,
            "max_metric1": float(max_metric_1),
            "max_metric2": float(max_metric_2) if max_metric_2 is not None else None,
            "adaptive_csplora": getattr(args, "csplora_ada", False),
        }
        results_file = os.path.join(args.output_dir, "eval_results.json")
        with open(results_file, 'w') as f:
            json.dump(eval_results, f, indent=2)
        print(f"[INFO] Saved eval results to {results_file}")


# Main execution
if __name__ == "__main__":
    task = args.task
    model = finetune(task)
