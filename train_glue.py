import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm.auto import tqdm
import numpy as np
from peft import get_peft_model, LoraConfig, TaskType
import os
import wandb
from train_eval import *
import json
import atexit

from utils.data_utils import *
from models import *
from utils.misc import *
from glue_args import get_glue_args
from csplora import CSpLoRAConfig, CSpLoRAPlanner
from torch.amp import autocast

args = get_glue_args()

dataset_name = args.task
model_name = os.path.basename(str(args.model)).replace("/", "-")
method_name = args.method

if getattr(args, "csplora", False):
    run_name = f"{dataset_name}_{model_name}_{method_name}_csplora"
else:
    run_name = f"{dataset_name}_{model_name}_{method_name}"

wandb.init(group=dataset_name, name=run_name, config=vars(args))

def cleanup_wandb():
    if wandb.run is not None:
        wandb.finish()

atexit.register(cleanup_wandb)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)


def finetune(task):
    GLUE_TASK_NUM_LABELS = {
        'cola': 2, 'mnli': 3, 'mrpc': 2, 'qnli': 2,
        'qqp': 2, 'rte': 2, 'sst2': 2, 'stsb': 1, 'wnli': 2
    }

    num_labels = GLUE_TASK_NUM_LABELS[task]
    model, tokenizer = create_model_tokenizer(num_labels, args)
    train_data, val_data, _ = load_and_preprocess_data(task=task, tokenizer=tokenizer, args=args)
    train_loader = create_dataloader(train_data, args, shuffle=True)
    val_loader = create_dataloader(val_data, args, shuffle=False)
    probe_loader = create_dataloader(train_data, args, shuffle=True)

    max_metric_1, max_metric_2 = 0, 0
    method = getattr(args, "method", "lora").lower()
    layer_ranks = None

    if method == "full":
        lora_config = None

    elif method in {"lora", "pissa", "dora", "adalora", "lora_plus", "lora+"}:
        use_csplora = getattr(args, "csplora", False)

        if use_csplora:
            if "roberta" in args.model or "bert" in args.model:
                target_modules = ["query", "value", "output.dense"]
            elif "llama" in args.model or "mistral" in args.model:
                target_modules = ["q_proj", "v_proj", "down_proj", "up_proj"]
            elif "t5" in args.model:
                target_modules = ["q", "k", "v", "o"]
            else:
                target_modules = ["query", "value"]

            r_tot_val = args.csplora_R_tot if args.csplora_R_tot > 0 else None
            csplora_task_name = task + getattr(args, "csplora_task_suffix", "")

            ada_cfg = CSpLoRAConfig(
                model_id=args.model,
                task_name=csplora_task_name,
                target_modules=target_modules,
                task_type=TaskType.SEQ_CLS,
                cache_dir=getattr(args, "csplora_cache_dir", "./csplora_cache"),
                device=args.device,
                r_base=args.lora_r,
                r_min=getattr(args, "csplora_r_min", 2),
                r_max_factor=getattr(args, "csplora_r_max_factor", 4.0),
                gamma=getattr(args, "csplora_gamma", 2.0),
                gamma_strategy=getattr(args, "csplora_gamma_strategy", "fixed"),
                gamma_scale=getattr(args, "csplora_gamma_scale", 1.0),
                gamma_max=getattr(args, "csplora_gamma_max", 10.0),
                gamma_min_std=getattr(args, "csplora_gamma_min_std", 1e-4),
                causal_confidence_weighting=not getattr(args, "csplora_no_causal_weighting", False),
                tau_scale=getattr(args, "csplora_tau_scale", 1.0),
                skip_smooth=getattr(args, "csplora_skip_smooth", False),
                skip_normalize=getattr(args, "csplora_skip_normalize", False),
                R_tot=r_tot_val,
                probe_r=getattr(args, "probe_rank", 2),
                probe_lr=getattr(args, "probe_lr", 5e-4),
                adaptive_steps=not getattr(args, "ada_no_adaptive_steps", False),
                min_probe_steps=getattr(args, "ada_min_steps", 50),
                max_probe_steps=getattr(args, "ada_max_steps", 500),
                step_convergence_threshold=getattr(args, "ada_convergence_threshold", 0.02),
                step_check_interval=getattr(args, "ada_check_interval", 10),
                step_patience=getattr(args, "ada_patience", 3),
                step_top_k=getattr(args, "ada_top_k", 20),
                step_rank_tolerance=getattr(args, "ada_rank_tolerance", 5),
                adaptive_rho=not getattr(args, "ada_no_adaptive_rho", False),
                rho_min=getattr(args, "ada_rho_min", 0.6),
                rho_max=getattr(args, "ada_rho_max", 1.0),
                rho_method=getattr(args, "ada_rho_method", "coverage"),
                rho_coverage_target=getattr(args, "ada_rho_coverage_target", 0.95),
                rho_fixed=getattr(args, "csplora_rho", 0.9),
            )

            planner = CSpLoRAPlanner(ada_cfg)
            layer_ranks, scores = planner.plan(model, probe_loader)

            if hasattr(planner, 'adaptive_stats') and planner.adaptive_stats:
                stats = planner.adaptive_stats
                wandb.log({
                    "ada_actual_probe_steps": stats.get('actual_probe_steps', -1),
                    "ada_converged": stats.get('converged', False),
                    "ada_final_rho": stats.get('final_rho', -1),
                    "csplora_gamma_eff_mean": stats.get('gamma_eff_mean', 0),
                    "csplora_weighted_batch_count": stats.get('weighted_batch_count', 0),
                })

            if layer_ranks:
                wandb.log({
                    "csplora_num_layers": len(layer_ranks),
                    "csplora_max_rank": max(layer_ranks.values()),
                    "csplora_min_rank": min(layer_ranks.values()),
                    "csplora_avg_rank": sum(layer_ranks.values()) / len(layer_ranks),
                })

        model, lora_config = create_peft_model(model, args, rank_pattern=layer_ranks)
        model.to(args.device)

    else:
        raise ValueError(f"Unknown method: {method}")

    for param in model.parameters():
        param.data = param.data.contiguous()

    param_counts = count_parameters(model, verbose=False)
    wandb.log({
        "total_params": param_counts['total_trainable_params'],
        "classifier_params": param_counts['classifier_params'],
        "non_classifier_params": param_counts['non_classifier_params']
    })

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
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=total_steps)

    for epoch in tqdm(range(args.epochs), desc='Epochs'):
        model.train()
        progress_bar = tqdm(enumerate(train_loader), desc=f'Epoch {epoch}', leave=False, total=len(train_loader))

        for step, data in progress_bar:
            data = {k: v.to(args.device, non_blocking=True) for k, v in data.items()}

            with autocast("cuda", dtype=torch.bfloat16):
                outputs = model(**data)
                loss = outputs.loss

            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
            wandb.log({"train_loss": loss.item(), "epoch": epoch, "step": step})

            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        max_metric_1, max_metric_2 = evaluate_glue(model, val_loader, args, max_metric_1, max_metric_2)

    if getattr(args, "output_dir", None):
        os.makedirs(args.output_dir, exist_ok=True)
        eval_results = {
            "task": task,
            "max_metric1": float(max_metric_1),
            "max_metric2": float(max_metric_2) if max_metric_2 is not None else None,
        }
        with open(os.path.join(args.output_dir, "eval_results.json"), 'w') as f:
            json.dump(eval_results, f, indent=2)


if __name__ == "__main__":
    finetune(args.task)
