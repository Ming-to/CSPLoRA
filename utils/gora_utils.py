"""
GoRA (Gradient-driven Adaptive Low Rank Adaptation) Implementation

Based on the paper: "GoRA: Gradient-driven Adaptive Low Rank Adaptation"

Key components:
1. Gradient accumulation for importance estimation
2. Layer importance calculation: I(W) = avg(|W ⊙ G|)
3. Adaptive rank allocation based on importance
4. B matrix initialization using pseudo-inverse
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import math
from collections import defaultdict
from copy import deepcopy
from peft import LoraConfig, get_peft_model
from peft.tuners.lora import Linear as LoraLinear


def _is_dispatched_model(model: nn.Module) -> bool:
    """
    Detect whether a model is loaded with `device_map` / accelerate dispatch hooks.

    In that case, calling `.to(device)` is either a no-op or can raise an error like:
    "You can't move a model that has been dispatched using accelerate hooks."
    """
    device_map = getattr(model, "hf_device_map", None)
    return isinstance(device_map, dict) and len(device_map) > 0


def _infer_input_device(model: nn.Module, fallback: str) -> torch.device:
    """
    Infer the device that input tensors should be moved to before a forward pass.

    For dispatched models (device_map), inputs should be placed on the same device as
    the input embedding layer (typically the first GPU).
    """
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
    """Get target module names based on model type."""
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
    """
    Accumulate gradients over N training steps.

    GoRA Algorithm 1, Step 2-6:
    For each training step, accumulate gradients for target modules.

    Args:
        model: The base model (not PEFT wrapped)
        dataloader: Training data loader
        device: Device to use
        num_steps: Number of gradient accumulation steps (N in paper)
        target_modules: List of module name patterns to target

    Returns:
        Dictionary of {module_name: accumulated_gradient}
    """
    model.train()
    if not _is_dispatched_model(model):
        model.to(device)

    input_device = _infer_input_device(model, fallback=device)

    # Identify target linear layers
    target_layers = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if target_modules is None or any(t in name for t in target_modules):
                target_layers[name] = module

    # Initialize gradient accumulators
    accumulated_grads = {name: torch.zeros_like(module.weight.data)
                         for name, module in target_layers.items()}

    # Accumulate gradients
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

        # Accumulate gradients
        for name, module in target_layers.items():
            if module.weight.grad is not None:
                accumulated_grads[name] += module.weight.grad.data.clone()

        steps_done += 1
        pbar.update(1)

    pbar.close()

    # Average the gradients
    for name in accumulated_grads:
        accumulated_grads[name] /= num_steps

    return accumulated_grads


def calculate_importance_scores(
    model: nn.Module,
    accumulated_grads: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Calculate layer importance scores.

    GoRA Algorithm 1, Step 7:
    I(W) = avg(|W ⊙ G|) = sum(|W * G|) / (m * n)

    Args:
        model: The base model
        accumulated_grads: Dictionary of accumulated gradients

    Returns:
        Dictionary of {module_name: importance_score}
    """
    importance_scores = {}

    for name, module in model.named_modules():
        if name in accumulated_grads and isinstance(module, nn.Linear):
            W = module.weight.data
            G = accumulated_grads[name]

            # I(W) = avg(|W ⊙ G|)
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
    """
    Allocate ranks based on importance scores.

    GoRA Algorithm 1, Step 8-9:
    a^i = I(W^i)
    b = r_ref * sum(sqrt(m_i + n_i)) / sum(a^i)
    r^i = clip(round(b * a^i / sqrt(m_i + n_i)), r_min, r_max)

    Args:
        importance_scores: Dictionary of importance scores
        layer_shapes: Dictionary of {module_name: (out_features, in_features)}
        r_ref: Reference rank (average rank budget)
        r_min: Minimum rank
        r_max: Maximum rank

    Returns:
        Dictionary of {module_name: allocated_rank}
    """
    if not importance_scores:
        return {}

    # Calculate scaling factor b
    sum_sqrt_dims = sum(math.sqrt(m + n) for m, n in layer_shapes.values())
    sum_importance = sum(importance_scores.values())

    if sum_importance < 1e-10:
        # If all importance scores are near zero, use uniform allocation
        return {name: r_ref for name in importance_scores}

    b = r_ref * sum_sqrt_dims / sum_importance

    # Allocate ranks
    rank_allocation = {}
    for name in importance_scores:
        a_i = importance_scores[name]
        m, n = layer_shapes[name]

        # r^i = round(b * a^i / sqrt(m + n))
        r_i = round(b * a_i / math.sqrt(m + n))

        # Clip to [r_min, r_max]
        r_i = max(r_min, min(r_max, r_i))

        rank_allocation[name] = r_i

    return rank_allocation


def initialize_lora_b_with_pseudoinverse(
    peft_model: nn.Module,
    accumulated_grads: Dict[str, torch.Tensor],
    gamma: float = 1.0,
    lora_alpha: int = 16,
) -> None:
    """
    Initialize LoRA B matrix using pseudo-inverse of gradient.

    GoRA Algorithm 1, Step 13-14:
    B_0 = -(A_0^T A_0)^{-1} A_0^T G
    B_0 = gamma * sqrt(m) / alpha * B_0

    This ensures the initial LoRA update approximates the gradient direction.

    Args:
        peft_model: PEFT model with LoRA adapters
        accumulated_grads: Dictionary of accumulated gradients
        gamma: Scaling factor (default 1.0)
        lora_alpha: LoRA alpha parameter
    """
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLinear):
            # Extract the base layer name (without adapter suffixes)
            base_name = name.replace("base_model.model.", "").replace(".base_layer", "")

            # Try to find matching gradient
            grad_key = None
            for key in accumulated_grads:
                if key in name or name.endswith(key) or key.endswith(base_name):
                    grad_key = key
                    break

            if grad_key is None:
                continue

            G = accumulated_grads[grad_key]

            # Get A and B matrices for default adapter
            if hasattr(module, 'lora_A') and 'default' in module.lora_A:
                A = module.lora_A['default'].weight.data  # Shape: (r, in_features)
                B = module.lora_B['default'].weight.data  # Shape: (out_features, r)

                m = B.shape[0]  # out_features
                r = A.shape[0]  # rank
                alpha = module.scaling.get('default', lora_alpha / r) * r

                # NOTE:
                # Base model weights/gradients may be in bf16/fp16, while LoRA A/B
                # parameters are commonly kept in fp32 for stability. Mixed-dtype
                # matmul is not allowed, so we explicitly do the pseudo-inverse
                # initialization math in fp32 and then cast back to B's dtype.
                compute_dtype = torch.float32

                # B_0 = -(A_0^T A_0)^{-1} A_0^T G
                # A: (r, n), G: (m, n)
                # A^T: (n, r), A^T @ A: (r, r)
                # (A^T @ A)^{-1}: (r, r)
                # G @ A^T: (m, r)
                # Result: (m, r)

                A_compute = A.to(dtype=compute_dtype)
                G_compute = G.to(device=A.device, dtype=compute_dtype)

                ATA = A_compute @ A_compute.T  # (r, r)

                # Add small regularization for numerical stability
                ATA_inv = torch.linalg.pinv(
                    ATA + 1e-6 * torch.eye(r, device=ATA.device, dtype=compute_dtype)
                )

                # B_0 = -G @ A^T @ (A @ A^T)^{-1}
                B_new = -G_compute @ A_compute.T @ ATA_inv  # (m, r)

                # Scale: B_0 = gamma * sqrt(m) / alpha * B_0
                scale = gamma * math.sqrt(m) / alpha
                B_new = (scale * B_new).to(dtype=B.dtype)

                # Update B matrix
                module.lora_B['default'].weight.data.copy_(B_new)


def apply_rslora_scaling(peft_model: nn.Module) -> None:
    """
    Apply rsLoRA-style scaling: alpha/sqrt(r) instead of alpha/r.

    GoRA uses this scaling to better preserve gradient magnitude.

    Args:
        peft_model: PEFT model with LoRA adapters
    """
    for name, module in peft_model.named_modules():
        if isinstance(module, LoraLinear):
            # Get current scaling and rank
            for adapter_name in module.scaling:
                r = module.r.get(adapter_name, 8)
                alpha = module.lora_alpha.get(adapter_name, 16)

                # rsLoRA scaling: alpha / sqrt(r)
                new_scaling = alpha / math.sqrt(r)
                module.scaling[adapter_name] = new_scaling


class GoRAConfig:
    """Configuration for GoRA."""

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
        """
        Args:
            r_ref: Reference rank (average rank)
            r_min: Minimum rank per layer
            r_max: Maximum rank per layer
            gamma: Scaling factor for B initialization
            num_grad_steps: Number of gradient accumulation steps
            use_rslora_scaling: Whether to use rsLoRA scaling (alpha/sqrt(r))
            use_pseudoinverse_init: Whether to initialize B using pseudo-inverse
        """
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
    """
    Create a PEFT model with GoRA-style adaptive rank allocation and initialization.

    This function implements the full GoRA pipeline:
    1. Accumulate gradients on training data
    2. Calculate layer importance scores
    3. Allocate ranks based on importance
    4. Create PEFT model with per-layer ranks
    5. Initialize B matrices using pseudo-inverse

    Args:
        model: Base model (not PEFT wrapped)
        dataloader: Training data loader for gradient accumulation
        args: Arguments containing device, lora_alpha, lora_dropout, etc.
        gora_config: GoRA configuration (uses defaults if None)
        task_type: PEFT task type
        target_modules: Target module patterns
        modules_to_save: Modules to save (e.g., classifier)

    Returns:
        Tuple of (peft_model, lora_config, rank_allocation)
    """
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

    # Step 1: Accumulate gradients
    print(f"[GoRA] Starting gradient accumulation for {gora_config.num_grad_steps} steps...")
    accumulated_grads = accumulate_gradients(
        model=model,
        dataloader=dataloader,
        device=device,
        num_steps=gora_config.num_grad_steps,
        target_modules=target_modules,
    )

    # Step 2: Calculate importance scores
    print("[GoRA] Calculating layer importance scores...")
    importance_scores = calculate_importance_scores(model, accumulated_grads)

    # Get layer shapes for rank allocation
    layer_shapes = {}
    for name, module in model.named_modules():
        if name in importance_scores and isinstance(module, nn.Linear):
            layer_shapes[name] = (module.out_features, module.in_features)

    # Step 3: Allocate ranks
    print("[GoRA] Allocating per-layer ranks...")
    rank_allocation = allocate_ranks(
        importance_scores=importance_scores,
        layer_shapes=layer_shapes,
        r_ref=gora_config.r_ref,
        r_min=gora_config.r_min,
        r_max=gora_config.r_max,
    )

    # Log rank statistics
    if rank_allocation:
        ranks = list(rank_allocation.values())
        print(f"[GoRA] Rank allocation: min={min(ranks)}, max={max(ranks)}, "
              f"avg={sum(ranks)/len(ranks):.1f}, total={sum(ranks)}")

    # Convert rank_allocation keys to match PEFT's expected format
    # PEFT expects patterns like "model.encoder.layer.0.attention.self.query"
    rank_pattern = {}
    for name, rank in rank_allocation.items():
        # Keep the full module name path
        rank_pattern[name] = rank

    # Step 4: Create PEFT model with rank pattern
    lora_kwargs = dict(
        task_type=task_type,
        r=gora_config.r_ref,  # Default rank (will be overridden by rank_pattern)
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    if modules_to_save is not None:
        lora_kwargs["modules_to_save"] = modules_to_save

    if rank_pattern:
        lora_kwargs["rank_pattern"] = rank_pattern

    # Use rsLoRA scaling if enabled
    if gora_config.use_rslora_scaling:
        lora_kwargs["use_rslora"] = True

    lora_config = LoraConfig(**lora_kwargs)
    peft_model = get_peft_model(model, lora_config)
    if not _is_dispatched_model(model):
        peft_model.to(device)

    # Step 5: Initialize B matrices using pseudo-inverse
    if gora_config.use_pseudoinverse_init:
        print("[GoRA] Initializing B matrices with pseudo-inverse...")
        # Remap gradient keys to match PEFT model structure
        remapped_grads = {}
        for key, grad in accumulated_grads.items():
            new_key = f"base_model.model.{key}"
            remapped_grads[new_key] = grad
            remapped_grads[key] = grad  # Keep original key too

        initialize_lora_b_with_pseudoinverse(
            peft_model=peft_model,
            accumulated_grads=remapped_grads,
            gamma=gora_config.gamma,
            lora_alpha=args.lora_alpha,
        )

    print("[GoRA] Model creation complete!")

    return peft_model, lora_config, rank_allocation
