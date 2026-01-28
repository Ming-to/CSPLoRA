"""
GoRA (Gradient-driven Adaptive Low Rank Adaptation) Implementation.
Based on: "GoRA: Gradient-driven Adaptive Low Rank Adaptation"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import math
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as LoraLinear


def _is_dispatched_model(model: nn.Module) -> bool:
    device_map = getattr(model, "hf_device_map", None)
    return isinstance(device_map, dict) and len(device_map) > 0


def _infer_input_device(model: nn.Module, fallback: str) -> torch.device:
    try:
        get_emb = getattr(model, "get_input_embeddings", None)
        if callable(get_emb):
            emb = get_emb()
            if emb is not None and getattr(emb, "weight", None) is not None:
                return emb.weight.device
    except Exception:
        pass
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device(fallback)


def get_target_module_names(model_name: str) -> List[str]:
    model_name = model_name.lower()
    if "roberta" in model_name or "bert" in model_name:
        return ["query", "value", "attention.output.dense", "output.dense"]
    elif "llama" in model_name or "mistral" in model_name:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    elif "t5" in model_name:
        return ["q", "k", "v", "o", "wi", "wo"]
    else:
        return ["query", "value"]


def accumulate_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
    num_steps: int = 50,
    target_modules: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    model.train()
    if not _is_dispatched_model(model):
        model.to(device)

    input_device = _infer_input_device(model, fallback=device)

    target_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                target_layers[name] = module

    accumulated_grads = {name: torch.zeros_like(module.weight.data)
                         for name, module in target_layers.items()}

    data_iter = iter(dataloader)
    steps_done = 0

    pbar = tqdm(total=num_steps, desc="[GoRA] Accumulating gradients")

    while steps_done < num_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        batch = {k: v.to(input_device, non_blocking=True) for k, v in batch.items()}

        model.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        for name, module in target_layers.items():
            if module.weight.grad is not None:
                accumulated_grads[name] += module.weight.grad.data.clone()

        steps_done += 1
        pbar.update(1)

    pbar.close()

    for name in accumulated_grads:
        accumulated_grads[name] /= num_steps

    return accumulated_grads


def calculate_importance_scores(
    model: nn.Module,
    accumulated_grads: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """I(W) = avg(|W * G|)"""
    importance_scores = {}

    for name, module in model.named_modules():
        if name in accumulated_grads and isinstance(module, nn.Linear):
            W = module.weight.data
            G = accumulated_grads[name]
            importance = torch.abs(W * G).mean().item()
            importance_scores[name] = importance

    return importance_scores


def allocate_ranks(
    importance_scores: Dict[str, float],
    layer_shapes: Dict[str, Tuple[int, int]],
    r_ref: int = 8,
    r_min: int = 1,
    r_max: int = 64,
) -> Dict[str, int]:
    if not importance_scores:
        return {}

    sum_sqrt_dims = sum(math.sqrt(m + n) for m, n in layer_shapes.values())
    sum_importance = sum(importance_scores.values())

    if sum_importance < 1e-10:
        return {name: r_ref for name in importance_scores}

    b = r_ref * sum_sqrt_dims / sum_importance

    rank_allocation = {}
    for name in importance_scores:
        a_i = importance_scores[name]
        m, n = layer_shapes[name]
        r_i = round(b * a_i / math.sqrt(m + n))
        r_i = max(r_min, min(r_max, r_i))
        rank_allocation[name] = r_i

    return rank_allocation


def initialize_lora_b_with_pseudoinverse(
    peft_model: nn.Module,
    accumulated_grads: Dict[str, torch.Tensor],
    gamma: float = 1.0,
    lora_alpha: int = 16,
) -> None:
    """B_0 = gamma * sqrt(m) / alpha * -(A^T A)^{-1} A^T G"""
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLinear):
            base_name = name.replace("base_model.model.", "").replace(".base_layer", "")

            grad_key = None
            for key in accumulated_grads:
                if key in name or name.endswith(key) or key.endswith(base_name):
                    grad_key = key
                    break

            if grad_key is None:
                continue

            G = accumulated_grads[grad_key]

            if hasattr(module, 'lora_A') and 'default' in module.lora_A:
                A = module.lora_A['default'].weight.data
                B = module.lora_B['default'].weight.data

                m = B.shape[0]
                r = A.shape[0]
                alpha = module.scaling.get('default', lora_alpha / r) * r

                compute_dtype = torch.float32

                A_compute = A.to(dtype=compute_dtype)
                G_compute = G.to(device=A.device, dtype=compute_dtype)

                ATA = A_compute @ A_compute.T
                ATA_inv = torch.linalg.pinv(
                    ATA + 1e-6 * torch.eye(r, device=ATA.device, dtype=compute_dtype)
                )

                B_new = -G_compute @ A_compute.T @ ATA_inv

                scale = gamma * math.sqrt(m) / alpha
                B_new = (scale * B_new).to(dtype=B.dtype)

                module.lora_B['default'].weight.data.copy_(B_new)


def apply_rslora_scaling(peft_model: nn.Module) -> None:
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLinear):
            for adapter_name in module.scaling:
                r = module.r.get(adapter_name, 8)
                alpha = module.lora_alpha.get(adapter_name, 16)
                module.scaling[adapter_name] = alpha / math.sqrt(r)


class GoRAConfig:
    def __init__(
        self,
        r_ref: int = 8,
        r_min: int = 1,
        r_max: int = 64,
        gamma: float = 1.0,
        num_grad_steps: int = 50,
        use_rslora_scaling: bool = True,
        use_pseudoinverse_init: bool = True,
    ):
        self.r_ref = r_ref
        self.r_min = r_min
        self.r_max = r_max
        self.gamma = gamma
        self.num_grad_steps = num_grad_steps
        self.use_rslora_scaling = use_rslora_scaling
        self.use_pseudoinverse_init = use_pseudoinverse_init


def create_gora_peft_model(
    model: nn.Module,
    dataloader: DataLoader,
    args,
    gora_config: Optional[GoRAConfig] = None,
    task_type=None,
    target_modules: Optional[List[str]] = None,
    modules_to_save: Optional[List[str]] = None,
) -> Tuple[nn.Module, LoraConfig, Dict[str, int]]:
    from peft import TaskType

    if gora_config is None:
        gora_config = GoRAConfig(
            r_ref=args.lora_r,
            r_min=getattr(args, 'gora_r_min', 1),
            r_max=getattr(args, 'gora_r_max', 64),
            gamma=getattr(args, 'gora_gamma', 1.0),
            num_grad_steps=getattr(args, 'gora_num_steps', 50),
        )

    if target_modules is None:
        target_modules = get_target_module_names(args.model)

    if task_type is None:
        if "roberta" in args.model.lower() or "bert" in args.model.lower():
            task_type = TaskType.SEQ_CLS
        else:
            task_type = TaskType.CAUSAL_LM

    device = args.device

    print(f"[GoRA] Accumulating gradients for {gora_config.num_grad_steps} steps...")
    accumulated_grads = accumulate_gradients(
        model=model, dataloader=dataloader, device=device,
        num_steps=gora_config.num_grad_steps, target_modules=target_modules,
    )

    print("[GoRA] Calculating layer importance scores...")
    importance_scores = calculate_importance_scores(model, accumulated_grads)

    layer_shapes = {}
    for name, module in model.named_modules():
        if name in importance_scores and isinstance(module, nn.Linear):
            layer_shapes[name] = (module.out_features, module.in_features)

    print("[GoRA] Allocating per-layer ranks...")
    rank_allocation = allocate_ranks(
        importance_scores=importance_scores, layer_shapes=layer_shapes,
        r_ref=gora_config.r_ref, r_min=gora_config.r_min, r_max=gora_config.r_max,
    )

    if rank_allocation:
        ranks = list(rank_allocation.values())
        print(f"[GoRA] Rank allocation: min={min(ranks)}, max={max(ranks)}, "
              f"avg={sum(ranks)/len(ranks):.1f}, total={sum(ranks)}")

    rank_pattern = {name: rank for name, rank in rank_allocation.items()}

    lora_kwargs = dict(
        task_type=task_type,
        r=gora_config.r_ref,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    if modules_to_save is not None:
        lora_kwargs["modules_to_save"] = modules_to_save

    if rank_pattern:
        lora_kwargs["rank_pattern"] = rank_pattern

    if gora_config.use_rslora_scaling:
        lora_kwargs["use_rslora"] = True

    lora_config = LoraConfig(**lora_kwargs)
    peft_model = get_peft_model(model, lora_config)
    if not _is_dispatched_model(model):
        peft_model.to(device)

    if gora_config.use_pseudoinverse_init:
        print("[GoRA] Initializing B matrices with pseudo-inverse...")
        remapped_grads = {}
        for key, grad in accumulated_grads.items():
            new_key = f"base_model.model.{key}"
            remapped_grads[new_key] = grad
            remapped_grads[key] = grad

        initialize_lora_b_with_pseudoinverse(
            peft_model=peft_model,
            accumulated_grads=remapped_grads,
            gamma=gora_config.gamma,
            lora_alpha=args.lora_alpha,
        )

    print("[GoRA] Model creation complete!")

    return peft_model, lora_config, rank_allocation
