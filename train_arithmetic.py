import os
import json

import torch
from torch.utils.data import DataLoader
import numpy as np
import transformers
from transformers import TrainingArguments, Trainer
from peft import TaskType

import bitsandbytes as bnb
import wandb


class LoRAPlusTrainer(Trainer):
    """Custom Trainer that supports LoRA+ learning rate differentiation.

    LoRA+ uses different learning rates for lora_A and lora_B matrices:
    - lora_A: lr * lora_plus_lr_ratio (default 16x)
    - lora_B: lr (base learning rate)
    """

    def __init__(self, *args, lora_plus_lr_ratio: float = 16.0, **kwargs):
        self.lora_plus_lr_ratio = lora_plus_lr_ratio
        super().__init__(*args, **kwargs)

    def create_optimizer(self):
        """Create optimizer with different learning rates for LoRA A/B matrices."""
        if self.optimizer is not None:
            return self.optimizer

        # Separate parameters into groups
        lora_A_params = []
        lora_B_params = []
        other_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "lora_A" in name:
                lora_A_params.append(param)
            elif "lora_B" in name:
                lora_B_params.append(param)
            else:
                other_params.append(param)

        lr_B = self.args.learning_rate
        lr_A = lr_B * self.lora_plus_lr_ratio

        # Build parameter groups
        param_groups = []
        if other_params:
            param_groups.append({"params": other_params, "lr": lr_B})
        if lora_A_params:
            param_groups.append({"params": lora_A_params, "lr": lr_A})
        if lora_B_params:
            param_groups.append({"params": lora_B_params, "lr": lr_B})

        # Use 8-bit AdamW to match the default behavior
        self.optimizer = bnb.optim.AdamW8bit(
            param_groups,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            eps=self.args.adam_epsilon,
            weight_decay=self.args.weight_decay,
        )

        return self.optimizer

from utils.data_utils import load_and_preprocess_it, DataCollatorForSupervisedDataset
from models import create_model_tokenizer_it, create_peft_model_it, _get_llama_target_modules
from utils.misc import count_parameters
from utils.gora_utils import create_gora_peft_model, GoRAConfig

from llama_args import get_arithmetic_args
from csplora import CSpLoRAPlanner, CSpLoRAConfig


os.environ.setdefault("MASTER_ADDR", "localhost")
os.environ.setdefault("MASTER_PORT", "12355")


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_run_dir_path(args):
    base_dir = os.path.join(args.output_dir, "instruction_tuning")
    model_name = args.model.split("/")[-1]

    run_name = f"{args.method}_r{args.lora_r}_seed{args.seed}"
    if getattr(args, "csplora", False):
        run_name += "_csplora"

    return os.path.join(base_dir, model_name, run_name)


def check_already_trained(run_dir):
    final_model_dir = os.path.join(run_dir, "final_model")
    if not os.path.exists(final_model_dir):
        return False

    has_adapter = os.path.exists(os.path.join(final_model_dir, "adapter_model.safetensors")) or \
                  os.path.exists(os.path.join(final_model_dir, "adapter_model.bin"))
    has_full_model = os.path.exists(os.path.join(final_model_dir, "model.safetensors"))

    return has_adapter or has_full_model


def create_run_directory(args, run_dir):
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, "checkpoints"), exist_ok=True)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)

    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    return run_dir


def finetune(args):
    is_main_process = int(os.environ.get("RANK", "0")) == 0

    run_dir = get_run_dir_path(args)
    if not getattr(args, "count_params_only", False) and check_already_trained(run_dir):
        if is_main_process:
            print(f"[SKIP] Already trained: {run_dir}/final_model")
        return run_dir

    create_run_directory(args, run_dir)

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

    model, tokenizer = create_model_tokenizer_it(args)

    model.config.use_cache = False
    if getattr(args, "gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    train_dataset = load_and_preprocess_it(tokenizer=tokenizer, args=args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    data_module = dict(train_dataset=train_dataset, data_collator=data_collator)

    method = args.method.lower()

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    if method == "full":
        peft_config = None
        rank_pattern = None
    elif method == "gora":
        if getattr(args, "csplora", False):
            raise ValueError("method='gora' conflicts with --csplora")

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

        if is_main_process:
            with open(os.path.join(run_dir, "gora_rank_allocation.json"), "w") as f:
                json.dump(rank_allocation, f, indent=2)

        if is_main_process and rank_allocation:
            ranks = list(rank_allocation.values())
            wandb.log({
                "gora_rank_min": min(ranks),
                "gora_rank_max": max(ranks),
                "gora_rank_avg": sum(ranks) / len(ranks),
                "gora_rank_total": sum(ranks),
                "gora_num_layers": len(ranks),
            })
    else:
        rank_pattern = None
        if args.csplora:
            if is_main_process:
                probe_dataloader = DataLoader(
                    train_dataset,
                    batch_size=args.batch_size,
                    shuffle=True,
                    collate_fn=data_collator,
                )

                csplora_cfg = CSpLoRAConfig(
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
                    importance_metric=getattr(args, "csplora_importance_metric", "taylor"),
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
                    rho_fixed=args.csplora_rho,
                )

                planner = CSpLoRAPlanner(csplora_cfg)
                rank_pattern, scores = planner.plan(model, probe_dataloader)

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

                with open(os.path.join(run_dir, "csplora_rank_pattern.json"), "w") as f:
                    json.dump(rank_pattern, f, indent=2)

            if world_size > 1:
                import torch.distributed as dist
                if not dist.is_initialized():
                    dist.init_process_group(backend="nccl")

                if is_main_process:
                    rank_pattern_json = json.dumps(rank_pattern)
                    rank_pattern_bytes = rank_pattern_json.encode('utf-8')
                    rank_pattern_tensor = torch.ByteTensor(list(rank_pattern_bytes)).cuda(local_rank)
                    size_tensor = torch.LongTensor([len(rank_pattern_bytes)]).cuda(local_rank)
                else:
                    size_tensor = torch.LongTensor([0]).cuda(local_rank)

                dist.broadcast(size_tensor, src=0)
                size = size_tensor.item()

                if not is_main_process:
                    rank_pattern_tensor = torch.ByteTensor(size).cuda(local_rank)

                dist.broadcast(rank_pattern_tensor, src=0)

                if not is_main_process:
                    rank_pattern_bytes = bytes(rank_pattern_tensor.cpu().tolist())
                    rank_pattern_json = rank_pattern_bytes.decode('utf-8')
                    rank_pattern = json.loads(rank_pattern_json)

        model, peft_config = create_peft_model_it(
            model=model,
            args=args,
            rank_pattern=rank_pattern,
        )

    param_counts = count_parameters(model, verbose=False)
    total_params = param_counts["total_trainable_params"]
    classifier_params = param_counts.get("classifier_params", 0)
    non_classifier_params = param_counts.get("non_classifier_params", 0)

    if is_main_process:
        wandb.log({
            "total_params": total_params,
            "classifier_params": classifier_params,
            "non_classifier_params": non_classifier_params,
        })

    if getattr(args, "count_params_only", False):
        if is_main_process:
            print(f"[count_params_only] Total trainable params: {total_params}")
        wandb.finish()
        return run_dir

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
        optim="adamw_bnb_8bit",
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

    # Use LoRAPlusTrainer for lora_plus method, standard Trainer otherwise
    if method in ["lora_plus", "lora+"]:
        lora_plus_lr_ratio = getattr(args, "lora_plus_lr_ratio", 16.0)
        if is_main_process:
            print(f"[LoRA+] Using LoRAPlusTrainer with lr_ratio={lora_plus_lr_ratio}")
            print(f"[LoRA+] lora_A lr={args.lr * lora_plus_lr_ratio}, lora_B lr={args.lr}")
        trainer = LoRAPlusTrainer(
            model=model,
            args=training_args,
            lora_plus_lr_ratio=lora_plus_lr_ratio,
            **data_module,
        )
    else:
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
