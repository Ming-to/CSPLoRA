import os
import json
from datetime import datetime

import torch
from torch.utils.data import DataLoader
import numpy as np
import transformers
from transformers import TrainingArguments, Trainer
from peft import TaskType

import wandb

from utils.data_utils import load_and_preprocess_it, DataCollatorForSupervisedDataset
from models import create_model_tokenizer_it, create_peft_model_it, _get_llama_target_modules
from utils.misc import count_parameters  # 你原来的工具函数
from utils.gora_utils import create_gora_peft_model, GoRAConfig

from llama_args import get_arithmetic_args
from csplora_planner import CSpLoRAPlanner, CSpLoRAConfig
from csplora_ada import AdaptiveCSpLoRAPlanner, AdaptiveCSpLoRAConfig


# 避免 HF accelerate 在单机上抱怨没设置 master
os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_run_dir_path(args):
    """Get the run directory path without creating it.

    保存路径格式:
    {output_dir}/instruction_tuning/{model_name}/{method}_r{rank}_seed{seed}/
    例如: experiments_llama/instruction_tuning/Llama-3.1-8B/lora_r8_seed42/
    """
    base_dir = os.path.join(args.output_dir, "instruction_tuning")
    model_name = args.model.split("/")[-1]

    # 简洁的命名：method_r{rank}_seed{seed}，如果启用 csplora 则加后缀
    run_name = f"{args.method}_r{args.lora_r}_seed{args.seed}"
    if getattr(args, "csplora_ada", False):
        run_name += "_csplora_ada"
    elif getattr(args, "csplora", False):
        run_name += "_csplora"

    return os.path.join(base_dir, model_name, run_name)


def check_already_trained(run_dir):
    """Check if training has already been completed for this run.

    Returns True if final_model exists with valid checkpoint files.
    """
    final_model_dir = os.path.join(run_dir, "final_model")
    if not os.path.exists(final_model_dir):
        return False

    # Check for LoRA adapter or full model checkpoint
    has_adapter = os.path.exists(os.path.join(final_model_dir, "adapter_model.safetensors")) or \
                  os.path.exists(os.path.join(final_model_dir, "adapter_model.bin"))
    has_full_model = os.path.exists(os.path.join(final_model_dir, "model.safetensors"))

    return has_adapter or has_full_model


def create_run_directory(args, run_dir):
    """Create directory structure and save config for the current training run."""
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    # Save config
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return run_dir


def finetune(args):
    is_main_process = int(os.environ.get("RANK", "0")) == 0

    # Check if already trained BEFORE creating directories or initializing wandb
    # 但如果是 count_params_only 模式，不跳过（因为只是统计参数）
    run_dir = get_run_dir_path(args)
    if not getattr(args, "count_params_only", False) and check_already_trained(run_dir):
        if is_main_process:
            print(f"[SKIP] Already trained: {run_dir}/final_model")
        return run_dir

    # Now create directories and save config (only if we're actually training)
    create_run_directory(args, run_dir)

    # ---- wandb ----
    if not is_main_process:
        os.environ["WANDB_MODE"] = "disabled"
    wandb_run_name = args.run_name or os.path.basename(run_dir)
    wandb_run = wandb.init(
        project=args.project_name,
        name=wandb_run_name,
        config=vars(args),
        dir=os.path.join(run_dir, "logs"),
    )
    if is_main_process:
        with open(os.path.join(run_dir, "wandb_run_id.txt"), "w") as f:
            f.write(wandb_run.id)

    # ---- model & tokenizer ----
    model, tokenizer = create_model_tokenizer_it(args)

    # ---- memory / training behavior ----
    # For training and (especially) gradient checkpointing, cache must be disabled.
    model.config.use_cache = False
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    # ---- dataset & dataloader ----
    train_dataset = load_and_preprocess_it(tokenizer=tokenizer, args=args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    # ---- LoRA / PEFT ----
    method = args.method.lower()

    # Get local rank for distributed training
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if method == "full":
        # 直接全参数微调，不插 LoRA
        peft_config = None
        rank_pattern = None
        print("[INFO] method=full: 使用全参数微调，不插入 LoRA。")
    elif method == "gora":
        if getattr(args, "csplora", False):
            raise ValueError("[GoRA] method='gora' 与 --csplora 冲突：请二选一。")

        print("[GoRA] Initializing GoRA for task: arithmetic")

        # 处理 rsLoRA / pinv_init 的开关（对齐 train_glue.py 的逻辑）
        use_rslora = getattr(args, "gora_use_rslora", True)
        if getattr(args, "gora_no_rslora", False):
            use_rslora = False

        use_pinv_init = getattr(args, "gora_use_pinv_init", True)
        if getattr(args, "gora_no_pinv_init", False):
            use_pinv_init = False

        gora_config = GoRAConfig(
            r_ref=args.lora_r,
            r_min=getattr(args, "gora_r_min", 1),
            r_max=getattr(args, "gora_r_max", 64),
            gamma=getattr(args, "gora_gamma", 1.0),
            num_grad_steps=getattr(args, "gora_num_steps", 50),
            use_rslora_scaling=use_rslora,
            use_pseudoinverse_init=use_pinv_init,
        )

        # GoRA 需要一个 dataloader 来做梯度累积/重要性估计
        probe_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=data_collator,
        )

        model, peft_config, rank_allocation = create_gora_peft_model(
            model=model,
            dataloader=probe_dataloader,
            args=args,
            gora_config=gora_config,
            task_type=TaskType.CAUSAL_LM,
            target_modules=_get_llama_target_modules(),
            modules_to_save=None,
        )

        # 保存 GoRA 的 rank allocation（方便复现实验/对齐论文）
        if is_main_process:
            with open(os.path.join(run_dir, "gora_rank_allocation.json"), "w") as f:
                json.dump(rank_allocation, f, indent=2)

        # 记录 GoRA rank 分布到 wandb
        if is_main_process and rank_allocation:
            ranks = list(rank_allocation.values())
            wandb.log(
                {
                    "gora_rank_min": min(ranks),
                    "gora_rank_max": max(ranks),
                    "gora_rank_avg": sum(ranks) / len(ranks),
                    "gora_rank_total": sum(ranks),
                    "gora_num_layers": len(ranks),
                }
            )
    else:
        # CSpLoRA 动态 rank 分配
        rank_pattern = None
        if args.csplora or getattr(args, "csplora_ada", False):
            use_ada = getattr(args, "csplora_ada", False)
            mode_name = "Adaptive CSpLoRA" if use_ada else "CSpLoRA"

            # Only run probe on main process to avoid duplicate model loading
            if is_main_process:
                print(f"[{mode_name}] 启用 {mode_name} 动态 rank 分配...")

                # 创建 probe 用的 dataloader
                probe_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=data_collator,
                )

                if use_ada:
                    # Adaptive CSpLoRA with adaptive step and rho
                    ada_cfg = AdaptiveCSpLoRAConfig(
                        model_id=args.model,
                        task_name="arithmetic",
                        target_modules=_get_llama_target_modules(),
                        task_type=TaskType.CAUSAL_LM,
                        cache_dir=os.path.join(args.output_dir, "csplora_cache"),
                        gamma=args.csplora_gamma,
                        r_base=args.lora_r,
                        r_min=args.csplora_r_min,
                        tau_scale=args.csplora_tau_scale,
                        skip_smooth=getattr(args, "csplora_skip_smooth", False),
                        skip_normalize=getattr(args, "csplora_skip_normalize", False),
                        skip_layer_normalize=not getattr(args, "csplora_layer_normalize", False),
                        device=f"cuda:{local_rank}",
                        # Importance metric
                        importance_metric=getattr(args, "csplora_importance_metric", "taylor"),
                        # Adaptive step settings
                        adaptive_steps=not getattr(args, "ada_no_adaptive_steps", False),
                        min_probe_steps=getattr(args, "ada_min_steps", 50),
                        max_probe_steps=getattr(args, "ada_max_steps", 500),
                        step_convergence_threshold=getattr(args, "ada_convergence_threshold", 0.02),
                        step_check_interval=getattr(args, "ada_check_interval", 10),
                        step_patience=getattr(args, "ada_patience", 3),
                        step_top_k=getattr(args, "ada_top_k", 20),
                        step_rank_tolerance=getattr(args, "ada_rank_tolerance", 5),
                        # Adaptive rho settings
                        adaptive_rho=not getattr(args, "ada_no_adaptive_rho", False),
                        rho_min=getattr(args, "ada_rho_min", 0.6),
                        rho_max=getattr(args, "ada_rho_max", 1.0),
                        rho_method=getattr(args, "ada_rho_method", "coverage"),
                        rho_fixed=args.csplora_rho,
                    )

                    planner = AdaptiveCSpLoRAPlanner(ada_cfg)
                    rank_pattern, scores = planner.plan(model, probe_dataloader)

                    # Log adaptive stats to wandb
                    ada_stats = planner.adaptive_stats
                    wandb.log({
                        "ada_actual_probe_steps": ada_stats.get("actual_probe_steps", 0),
                        "ada_converged": ada_stats.get("converged", False),
                        "ada_final_rho": ada_stats.get("final_rho", args.csplora_rho),
                        "ada_rho_method": getattr(args, "ada_rho_method", "coverage"),
                        "ada_entropy": ada_stats.get("entropy", 0),
                        "ada_norm_entropy": ada_stats.get("norm_entropy", 0),
                        "ada_effective_layers": ada_stats.get("effective_layers", 0),
                        "ada_effective_ratio": ada_stats.get("effective_ratio", 0),
                    })
                else:
                    # Original CSpLoRA with fixed probe steps
                    csplora_cfg = CSpLoRAConfig(
                        model_id=args.model,
                        task_name="arithmetic",
                        target_modules=_get_llama_target_modules(),
                        task_type=TaskType.CAUSAL_LM,
                        cache_dir=os.path.join(args.output_dir, "csplora_cache"),
                        rho=args.csplora_rho,
                        gamma=args.csplora_gamma,
                        r_base=args.lora_r,
                        r_min=args.csplora_r_min,
                        tau_scale=args.csplora_tau_scale,
                        skip_smooth=getattr(args, "csplora_skip_smooth", False),
                        device=f"cuda:{local_rank}",
                    )

                    planner = CSpLoRAPlanner(csplora_cfg)
                    rank_pattern, scores = planner.plan(model, probe_dataloader)

                # 保存 rank_pattern 到 run_dir
                with open(os.path.join(run_dir, "csplora_rank_pattern.json"), "w") as f:
                    json.dump(rank_pattern, f, indent=2)

                print(f"[{mode_name}] Rank pattern 已保存到 {run_dir}/csplora_rank_pattern.json")

            # Synchronize rank_pattern across all processes in distributed training
            if world_size > 1:
                import torch.distributed as dist
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")

                # Broadcast rank_pattern from main process to all others
                if is_main_process:
                    # Serialize rank_pattern to JSON string, then to tensor
                    rank_pattern_json = json.dumps(rank_pattern)
                    rank_pattern_bytes = rank_pattern_json.encode('utf-8')
                    rank_pattern_tensor = torch.ByteTensor(list(rank_pattern_bytes)).cuda(local_rank)
                    size_tensor = torch.LongTensor([len(rank_pattern_bytes)]).cuda(local_rank)
                else:
                    size_tensor = torch.LongTensor([0]).cuda(local_rank)

                # Broadcast size first
                dist.broadcast(size_tensor, src=0)
                size = size_tensor.item()

                if not is_main_process:
                    rank_pattern_tensor = torch.ByteTensor(size).cuda(local_rank)

                # Broadcast the actual data
                dist.broadcast(rank_pattern_tensor, src=0)

                if not is_main_process:
                    rank_pattern_bytes = bytes(rank_pattern_tensor.cpu().tolist())
                    rank_pattern_json = rank_pattern_bytes.decode('utf-8')
                    rank_pattern = json.loads(rank_pattern_json)
                    print(f"[Rank {local_rank}] Received rank_pattern with {len(rank_pattern)} layers from main process")

        model, peft_config = create_peft_model_it(
            model=model,
            args=args,
            rank_pattern=rank_pattern,
        )

    # ---- 参数数量统计 ----
    param_counts = count_parameters(model, verbose=False)
    total_params = param_counts["total_trainable_params"]
    classifier_params = param_counts.get("classifier_params", 0)
    non_classifier_params = param_counts.get("non_classifier_params", 0)

    if is_main_process:
        wandb.log(
            {
                "total_params": total_params,
                "classifier_params": classifier_params,
                "non_classifier_params": non_classifier_params,
            }
        )

    # 如果只需要统计参数量，打印后退出
    if getattr(args, "count_params_only", False):
        if is_main_process:
            print("\n" + "=" * 60)
            print(f"[count_params_only] 参数统计完成，退出训练")
            print(f"  Method: {method}")
            print(f"  Model: {args.model}")
            print(f"  Total trainable params: {total_params}")
            print("=" * 60 + "\n")
        wandb.finish()
        return run_dir

    # ---- training args ----
    # GoRA uses cosine LR decay with a non-zero minimum LR ratio for Llama3 (decay ratio=0.1).
    # In transformers>=4.46, this maps to `cosine_with_min_lr` + `lr_scheduler_kwargs={"min_lr_rate": ...}`.
    lr_scheduler_type = args.scheduler
    lr_scheduler_kwargs = {}
    if args.scheduler == "cosine" and getattr(args, "lr_decay_ratio", 0.0) > 0:
        lr_scheduler_type = "cosine_with_min_lr"
        lr_scheduler_kwargs = {"min_lr_rate": args.lr_decay_ratio}

    training_args = TrainingArguments(
        output_dir=os.path.join(run_dir, "checkpoints"),
        run_name=wandb_run_name,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        lr_scheduler_kwargs=lr_scheduler_kwargs,
        optim="adamw_bnb_8bit",  # 8-bit Adam 节省优化器内存
        adam_beta1=0.9,
        adam_beta2=0.999,
        seed=args.seed,
        report_to=["wandb"],
        gradient_accumulation_steps=args.grad_acc_steps,
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="no",
        bf16=True,
        tf32=False,
        fp16=False,
        logging_steps=1,
        logging_first_step=True,
        logging_dir=os.path.join(run_dir, "logs"),
    )

    with open(os.path.join(run_dir, "training_args.json"), "w") as f:
        json.dump(training_args.to_dict(), f, indent=4)

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=training_args,
        **data_module,
    )

    if is_main_process:
        tokenizer.save_pretrained(os.path.join(run_dir, "tokenizer"))
    trainer.train()

    final_model_path = os.path.join(run_dir, "final_model")
    if is_main_process:
        trainer.save_state()
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)

    return run_dir


if __name__ == "__main__":
    args = get_arithmetic_args()

    set_seed(args.seed)

    run_dir = finetune(args)
    print(f"[DONE] run_dir = {run_dir}")
