import os
import torch
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaForSequenceClassification
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    AutoModelForSequenceClassification,
    AutoModelForSeq2SeqLM,
)
from torch.optim import AdamW
from datasets import load_dataset
import numpy as np
from peft import (
    get_peft_model,
    AdaLoraModel,
    AdaLoraConfig,
    TaskType,
    LoraConfig,
    prepare_model_for_kbit_training,
)
from utils.data_utils import *
import argparse
from copy import deepcopy
from tqdm import tqdm

from peft.utils import _get_submodules
from typing import Dict, Optional

def _get_llama_target_modules():
    """
    统一管理 LLaMA 系列的 LoRA 作用位置。
    """
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def create_model_tokenizer(num_labels, args): 
    if 'roberta' in args.model: 
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels) 
        tokenizer = AutoTokenizer.from_pretrained(args.model) 
    model.to(args.device) 
    return model, tokenizer

def create_peft_model(
    model,
    args,
    rank_pattern: Optional[Dict[str, int]] = None,
):

    """
    根据 args.method 选择不同的 PEFT 配置:
      - full      : 不在这里处理（train_glue 里直接用全参数）
      - lora      : 标准 LoRA（当前 C-SPLoRA 只接在这里）
      - lora-sb   : 和 lora 一样的 LoraConfig，只是初始化不同（在 finetune 里处理）
      - adalora   : AdaLoRA (AdaLoraConfig)
      - dora      : LoRA + use_dora=True（如果当前 peft 版本支持）
      - gora      : GoRA - 自适应 rank 分配（在 train_glue 里单独处理）

    rank_pattern:
      - 来自 C-SPLoRA 的 {layer_name: rank}，只在 method == "lora" 时使用；
      - 其他 method 暂时忽略（后面想扩再说）。
    """

    method = getattr(args, "method", "lora").lower()

    # GoRA 在 train_glue.py 里单独处理，这里只是兼容性检查
    if method == "gora":
        raise ValueError(
            "GoRA should be handled separately via create_gora_peft_model(). "
            "Do not call create_peft_model() with method='gora'."
        )
    # 1) 确定 task_type 和 target_modules
    modules_to_save = None  # <--- 新增变量
    
    if "roberta" in args.model:
        task_type = TaskType.SEQ_CLS
        target_modules = ["query", "value", "attention.output.dense", "output.dense"]
        # === 关键修改：指定需要全参数训练的层 ===
        # 对于分类任务，classifier 必须被训练
        modules_to_save = ["classifier"] 
        
    elif "t5" in args.model:
        task_type = TaskType.SEQ_2_SEQ_LM
        target_modules = ["q", "k", "v", "o", "wi", "wo"]
    # ...

    # 2) 配置 LoraConfig
    if method in ["lora", "lora-sb", "pissa", "dora", "lora_plus", "lora+"]:
        lora_kwargs = dict(
            task_type=task_type,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=target_modules,
            bias="none",
            # === 关键修改：传入 modules_to_save ===
            # 这会告诉 PEFT：除了 LoRA，这些层也给我解冻，参与训练！
            modules_to_save=modules_to_save, 
        )

        # CSpLoRA 输出的 per-site rank_pattern：只有在 csplora 打开时才非空
        if rank_pattern is not None and len(rank_pattern) > 0:
            lora_kwargs["rank_pattern"] = rank_pattern

        # PiSSA 只是初始化方式不同
        if method == "pissa":
            lora_kwargs["init_lora_weights"] = "pissa"

        # DoRA：在 LoraConfig 上加 use_dora=True
        if method == "dora":
            try:
                lora_kwargs["use_dora"] = True
            except TypeError as e:
                raise RuntimeError(
                    "当前 peft 版本不支持 DoRA（LoraConfig(..., use_dora=True)）。"
                ) from e

        peft_config = LoraConfig(**lora_kwargs)

    elif method == "adalora":
        # AdaLoRA：暂时不接 rank_pattern，后面要支持可以再改
        if rank_pattern is not None:
            print("[C-SPLoRA] rank_pattern provided but method == 'adalora'; 当前版本暂时忽略。")
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
        raise ValueError(f"create_peft_model got unsupported method={args.method}")

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

    #tokenizer.pad_token_id = tokenizer.eos_token_id
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    #model.to(args.device)

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
    
    # 统一使用 AutoTokenizer，支持本地路径加载，无需额外依赖
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        use_fast=True,
        model_max_length=args.max_seq_length,
        padding="max_length",
    )

    #tokenizer.pad_token_id = (0)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"


    return model, tokenizer


def create_peft_model_it(
    model,
    args,
    rank_pattern: Optional[Dict[str, int]] = None,
):
    """
    arithmetic / instruction-tuning 任务用的 PEFT 构造。

    支持的方法：
      - full      : 不在这里处理（调用处自己用 full 模型）
      - lora      : 标准 LoRA
      - adalora   : AdaLoRA
      - pissa     : PiSSA（通过 init_lora_weights='pissa'）
      - dora      : DoRA（use_dora=True）
      - lora_plus : LoRA+（暂时 config 与 LoRA 一致，优化器里再区分 lr）

    rank_pattern:
      - 来自 CSpLoRA 的 {模块名: rank}，用于非均匀 rank；关闭 csplora 时传 None。
    """
    method = getattr(args, "method", "lora").lower()

    if method == "full":
        # 全参微调，直接返回原模型
        return model, None

    target_modules = _get_llama_target_modules()

    # CSpLoRA 输出的 per-site rank_pattern
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

        # PiSSA：通过 init_lora_weights='pissa'（你的 peft 版本已经在 roberta 那边这样用了）
        if method == "pissa":
            lora_kwargs["init_lora_weights"] = "pissa"

        # DoRA：use_dora=True
        if method == "dora":
            try:
                lora_kwargs["use_dora"] = True
            except TypeError as e:
                raise RuntimeError(
                    "当前 peft 版本不支持 DoRA（LoraConfig(..., use_dora=True)）。"
                ) from e

        # LoRA+：config 和 LoRA 一样，只是后面 optimizer 里会对 A/B 层给更大学习率
        # 这里先不在 config 里做区分，方便兼容 peft 版本
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
        raise ValueError(f"[IT] Unsupported method: {args.method}")

    model = get_peft_model(model, peft_config)
    return model, peft_config


def create_peft_model_cr(
    model,
    args,
    rank_pattern: Optional[Dict[str, int]] = None,
):
    """
    commonsense reasoning 任务用的 PEFT 构造。

    本质和 create_peft_model_it 一样，只是默认 dropout 用 args.lora_dropout。
    """
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
            try:
                lora_kwargs["use_dora"] = True
            except TypeError as e:
                raise RuntimeError(
                    "当前 peft 版本不支持 DoRA（LoraConfig(..., use_dora=True)）。"
                ) from e

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
        raise ValueError(f"[CR] Unsupported method: {args.method}")

    model = get_peft_model(model, peft_config)
    return model, peft_config
