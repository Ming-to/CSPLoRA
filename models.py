import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)
from peft import (
    get_peft_model,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
)
from typing import Dict, Optional


def _get_llama_target_modules():
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def create_model_tokenizer(num_labels, args):
    if 'roberta' in args.model:
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
    model.to(args.device)
    return model, tokenizer


def create_peft_model(model, args, rank_pattern: Optional[Dict[str, int]] = None):
    method = getattr(args, "method", "lora").lower()

    if method == "gora":
        raise ValueError("GoRA should be handled separately via create_gora_peft_model().")

    modules_to_save = None

    if "roberta" in args.model:
        task_type = TaskType.SEQ_CLS
        target_modules = ["query", "value", "attention.output.dense", "output.dense"]
        modules_to_save = ["classifier"]
    elif "t5" in args.model:
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q", "k", "v", "o", "wi", "wo"]

    if method in ["lora", "lora-sb", "pissa", "dora", "lora_plus", "lora+"]:
        lora_kwargs = dict(
            task_type=task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            modules_to_save=modules_to_save,
        )

        if rank_pattern is not None and len(rank_pattern) > 0:
            lora_kwargs["rank_pattern"] = rank_pattern

        if method == "pissa":
            lora_kwargs["init_lora_weights"] = "pissa"

        if method == "dora":
            lora_kwargs["use_dora"] = True

        peft_config = LoraConfig(**lora_kwargs)

    elif method == "adalora":
        adalora_kwargs = dict(
            task_type=task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )
        if rank_pattern is not None and len(rank_pattern) > 0:
            adalora_kwargs["rank_pattern"] = rank_pattern

        peft_config = AdaLoraConfig(**adalora_kwargs)
    else:
        raise ValueError(f"Unsupported method: {args.method}")

    model = get_peft_model(model, peft_config)
    model.to(args.device)

    return model, peft_config


def create_model_tokenizer_it(args):
    local_rank = os.environ.get("LOCAL_RANK")
    device_map = {"": int(local_rank)} if local_rank is not None else "auto"
    from_pretrained_kwargs = dict(
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    if getattr(args, "attn_implementation", None):
        from_pretrained_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **from_pretrained_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        model_max_length=args.max_seq_length,
        padding="max_length",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    return model, tokenizer


def create_model_tokenizer_cr(args):
    local_rank = os.environ.get("LOCAL_RANK")
    device_map = {"": int(local_rank)} if local_rank is not None else "auto"
    from_pretrained_kwargs = dict(
        device_map=device_map,
        torch_dtype=torch.bfloat16,
    )
    if getattr(args, "attn_implementation", None):
        from_pretrained_kwargs["attn_implementation"] = args.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(args.model, **from_pretrained_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        model_max_length=args.max_seq_length,
        padding="max_length",
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    return model, tokenizer


def create_peft_model_it(model, args, rank_pattern: Optional[Dict[str, int]] = None):
    method = getattr(args, "method", "lora").lower()

    if method == "full":
        return model, None

    target_modules = _get_llama_target_modules()

    common_lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if rank_pattern is not None and len(rank_pattern) > 0:
        common_lora_kwargs["rank_pattern"] = rank_pattern

    if method in ["lora", "pissa", "dora", "lora_plus", "lora+"]:
        lora_kwargs = dict(common_lora_kwargs)

        if method == "pissa":
            lora_kwargs["init_lora_weights"] = "pissa"

        if method == "dora":
            lora_kwargs["use_dora"] = True

        peft_config = LoraConfig(**lora_kwargs)

    elif method == "adalora":
        adalora_kwargs = dict(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if rank_pattern is not None and len(rank_pattern) > 0:
            adalora_kwargs["rank_pattern"] = rank_pattern

        peft_config = AdaLoraConfig(**adalora_kwargs)

    else:
        raise ValueError(f"Unsupported method: {args.method}")

    model = get_peft_model(model, peft_config)
    return model, peft_config


def create_peft_model_cr(model, args, rank_pattern: Optional[Dict[str, int]] = None):
    method = getattr(args, "method", "lora").lower()

    if method == "full":
        return model, None

    target_modules = _get_llama_target_modules()

    common_lora_kwargs = dict(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    if rank_pattern is not None and len(rank_pattern) > 0:
        common_lora_kwargs["rank_pattern"] = rank_pattern

    if method in ["lora", "pissa", "dora", "lora_plus", "lora+"]:
        lora_kwargs = dict(common_lora_kwargs)

        if method == "pissa":
            lora_kwargs["init_lora_weights"] = "pissa"

        if method == "dora":
            lora_kwargs["use_dora"] = True

        peft_config = LoraConfig(**lora_kwargs)

    elif method == "adalora":
        adalora_kwargs = dict(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        if rank_pattern is not None and len(rank_pattern) > 0:
            adalora_kwargs["rank_pattern"] = rank_pattern

        peft_config = AdaLoraConfig(**adalora_kwargs)

    else:
        raise ValueError(f"Unsupported method: {args.method}")

    model = get_peft_model(model, peft_config)
    return model, peft_config
