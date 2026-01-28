# eval_commonsense.py

import os
import json
import re
import glob
from datetime import datetime
from typing import List, Tuple

import torch
from torch.nn import functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm.auto import tqdm
import sys
# 获取当前脚本的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 获取父目录 (即 lora-sb 文件夹)
parent_dir = os.path.dirname(current_dir)
# 将父目录加入到系统查找路径中
sys.path.append(parent_dir)

from llama_args import get_eval_args


def set_seed(seed: int = 42):
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_json_any(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)

        data: List[dict] = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
        return data


def _get_default_data_root(args) -> str:
    # 1) CLI override
    eval_data_dir = getattr(args, "eval_data_dir", None)
    if eval_data_dir:
        return eval_data_dir

    # 2) env var override (align with existing training scripts)
    env_root = os.environ.get("HF_DISK_ROOT")
    if env_root:
        return env_root

    # 3) repo-local default: <repo>/data
    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def _get_local_eval_file_candidates(args, task: str) -> List[str]:
    """
    Local-file-first workflow:
      - preferred: <data_root>/commonsense_eval/*.jsonl
      - fallback : <data_root>/commonsense/*.jsonl (folder already exists in this repo)

    File format expectations (JSONL):
      - BoolQ: {"passage": str, "question": str, "answer": bool}
      - HellaSwag: {"ctx_a": str, "ctx_b": str, "endings": [str,str,str,str], "label": int}
    """
    task = task.lower()
    # Handle task name aliases
    task_aliases = {
        "arc-e": "arc_easy",
        "arc_e": "arc_easy",
        "arce": "arc_easy",
        "arc-c": "arc_challenge",
        "arc_c": "arc_challenge",
        "arcc": "arc_challenge",
        "openbookqa": "obqa",
    }
    task = task_aliases.get(task, task)

    filename_map = {
        "boolq": "boolq_validation.jsonl",
        "hellaswag": "hellaswag_validation.jsonl",
        "piqa": "piqa_validation.jsonl",
        "siqa": "siqa_validation.jsonl",
        "winogrande": "winogrande_validation.jsonl",
        "arc_easy": "arc_easy_validation.jsonl",
        "arc_challenge": "arc_challenge_validation.jsonl",
        "obqa": "obqa_validation.jsonl",
    }
    # Map canonical task name to actual directory name in data/cr_eval/
    dir_name_map = {
        "boolq": "boolq",
        "hellaswag": "hellaswag",
        "piqa": "piqa",
        "siqa": "social_i_qa",
        "winogrande": "winogrande",
        "arc_easy": "ARC-Easy",
        "arc_challenge": "ARC-Challenge",
        "obqa": "openbookqa",
    }
    dir_name = dir_name_map.get(task, task)

    data_root = _get_default_data_root(args)
    candidates = [
        os.path.join(data_root, "cr_eval", dir_name, "test.json"),
        os.path.join(data_root, "cr_eval", dir_name, "validation.json"),
        os.path.join(data_root, "cr_eval", task, "test.json"),  # fallback to canonical name
        os.path.join(data_root, "cr_eval", task, "validation.json"),
    ]
    fname = filename_map.get(task)
    if fname:
        candidates.extend(
            [
                os.path.join(data_root, "commonsense_eval", fname),
                os.path.join(data_root, "commonsense", fname),
            ]
        )
    return candidates


# ========== 模型加载（与 arithmetic 版本类似） ==========

def load_model_and_tokenizer(base_model: str, ckpt_dir: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=True)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    try:
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto" if torch.cuda.is_available() else None,
        )
        model = PeftModel.from_pretrained(base, ckpt_dir)
        print(f"[INFO] Loaded PEFT model: base={base_model}, adapter={ckpt_dir}")
    except Exception as e:
        print(f"[WARN] Loading as PEFT model failed ({e}), fallback to full model from ckpt_dir.")
        model = AutoModelForCausalLM.from_pretrained(
            ckpt_dir,
            torch_dtype=torch.bfloat16,
        )

    model.to(device)
    model.eval()
    model.config.use_cache = True

    return model, tokenizer, device


def get_model_metadata(ckpt_dir: str) -> dict:
    """Collect metadata about the model checkpoint for traceability.

    Returns a dict containing:
    - ckpt_dir: the checkpoint directory path
    - model_file: the main model file (adapter or full)
    - model_mtime: modification time of the model file
    - training_wandb_run_id: the wandb run ID that produced this model (if available)
    - adapter_config: adapter configuration (if it's a LoRA model)
    """
    metadata = {
        "ckpt_dir": os.path.abspath(ckpt_dir),
        "eval_time": datetime.now().isoformat(),
    }

    # Find the main model file and get its mtime
    adapter_file = None
    for fname in ["adapter_model.safetensors", "adapter_model.bin"]:
        fpath = os.path.join(ckpt_dir, fname)
        if os.path.exists(fpath):
            adapter_file = fpath
            break

    full_model_file = os.path.join(ckpt_dir, "model.safetensors")

    if adapter_file:
        metadata["model_type"] = "lora_adapter"
        metadata["model_file"] = os.path.basename(adapter_file)
        metadata["model_mtime"] = datetime.fromtimestamp(
            os.path.getmtime(adapter_file)
        ).isoformat()

        # Read adapter config
        adapter_config_path = os.path.join(ckpt_dir, "adapter_config.json")
        if os.path.exists(adapter_config_path):
            with open(adapter_config_path, "r") as f:
                metadata["adapter_config"] = json.load(f)
    elif os.path.exists(full_model_file):
        metadata["model_type"] = "full_model"
        metadata["model_file"] = "model.safetensors"
        metadata["model_mtime"] = datetime.fromtimestamp(
            os.path.getmtime(full_model_file)
        ).isoformat()
    else:
        metadata["model_type"] = "unknown"
        metadata["model_file"] = None
        metadata["model_mtime"] = None

    # Try to find the training wandb run ID
    # The run directory is typically the parent of final_model
    run_dir = os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "final_model" else ckpt_dir

    # Method 1: Look for wandb run directories that match the model mtime
    wandb_logs_dir = os.path.join(run_dir, "logs", "wandb")
    if os.path.exists(wandb_logs_dir) and metadata.get("model_mtime"):
        model_mtime = datetime.fromisoformat(metadata["model_mtime"])
        run_dirs = glob.glob(os.path.join(wandb_logs_dir, "run-*"))

        best_match = None
        best_time_diff = float("inf")

        for rd in run_dirs:
            # Extract timestamp from directory name: run-YYYYMMDD_HHMMSS-runid
            dirname = os.path.basename(rd)
            parts = dirname.split("-")
            if len(parts) >= 3:
                try:
                    run_time_str = parts[1]  # YYYYMMDD_HHMMSS
                    run_time = datetime.strptime(run_time_str, "%Y%m%d_%H%M%S")
                    run_id = parts[2]

                    # The training run should start BEFORE the model was saved
                    if run_time < model_mtime:
                        time_diff = (model_mtime - run_time).total_seconds()
                        # Pick the run that started closest to (but before) model save time
                        if time_diff < best_time_diff:
                            best_time_diff = time_diff
                            best_match = run_id
                except (ValueError, IndexError):
                    continue

        if best_match:
            metadata["training_wandb_run_id"] = best_match
            metadata["training_wandb_run_id_source"] = "inferred_from_logs_mtime"

    # Method 2: If we couldn't infer, try reading wandb_run_id.txt (but mark it as potentially stale)
    if "training_wandb_run_id" not in metadata:
        wandb_id_file = os.path.join(run_dir, "wandb_run_id.txt")
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, "r") as f:
                run_id = f.read().strip()
            # Check if this file was modified after the model (indicating it might be stale)
            id_file_mtime = os.path.getmtime(wandb_id_file)
            if metadata.get("model_mtime"):
                model_mtime_ts = datetime.fromisoformat(metadata["model_mtime"]).timestamp()
                if id_file_mtime > model_mtime_ts + 60:  # More than 1 minute after model save
                    metadata["training_wandb_run_id"] = run_id
                    metadata["training_wandb_run_id_source"] = "wandb_run_id.txt (WARNING: may be stale)"
                else:
                    metadata["training_wandb_run_id"] = run_id
                    metadata["training_wandb_run_id_source"] = "wandb_run_id.txt"
            else:
                metadata["training_wandb_run_id"] = run_id
                metadata["training_wandb_run_id_source"] = "wandb_run_id.txt"

    return metadata


def save_eval_results(results: list, metadata: dict, accuracy: float, output_file: str):
    """Save evaluation results with model metadata to a JSON file."""
    output_data = {
        "metadata": metadata,
        "accuracy": accuracy,
        "num_correct": sum(1 for r in results if r.get("correct")),
        "num_total": len(results),
        "results": results,
    }

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Results saved to: {output_file}")
    print(f"[INFO] Model checkpoint: {metadata.get('ckpt_dir')}")
    print(f"[INFO] Model saved at: {metadata.get('model_mtime')}")
    if "training_wandb_run_id" in metadata:
        print(f"[INFO] Training wandb run: {metadata.get('training_wandb_run_id')} ({metadata.get('training_wandb_run_id_source')})")


# ========== 通用：计算选项 logprob ==========

def option_logprob(
    model,
    tokenizer,
    device,
    prompt: str,
    option: str,
    max_length: int = 512,
) -> float:
    """
    计算 log P(option | prompt) 的近似：
      - 拼接为 prompt + option；
      - 对 option 对应的 token 位置求 log-prob 总和。
    """

    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    option_ids = tokenizer(option, add_special_tokens=False)["input_ids"]

    input_ids = prompt_ids + option_ids
    if len(input_ids) > max_length:
        input_ids = input_ids[-max_length:]

    input_ids = torch.tensor([input_ids], dtype=torch.long, device=device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids)
        logits = outputs.logits  # [1, seq_len, vocab]
        log_probs = F.log_softmax(logits, dim=-1)

    # option 的 token 对应的是最后 len(option_ids) 个位置（前面是 prompt）
    seq_len = input_ids.shape[1]
    L = min(len(option_ids), seq_len - 1)
    if L <= 0:
        return -1e9

    target_ids = input_ids[:, -L:]
    log_probs_option = log_probs[:, -L - 1 : -1, :]  # 对齐：第 t+1 个位置预测第 t+1 个token
    gathered = log_probs_option.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
    return float(gathered.sum().item())


# ========== 各数据集的 prompt & option 构造 ==========

def _build_instruction_prompt(instruction: str, input_text: str | None) -> str:
    if input_text:
        return (
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "            ### Instruction:\n"
            f"            {instruction}\n\n"
            "            ### Input:\n"
            f"            {input_text}\n\n"
            "            ### Response:\n"
        )
    return (
        "Below is an instruction that describes a task. Write a response that appropriately completes the request.\n\n"
        "            ### Instruction:\n"
        f"            {instruction}\n\n"
        "            ### Response:\n"
    )


def _parse_bool_answer(raw) -> bool | None:
    if isinstance(raw, bool):
        return raw
    if raw is None:
        return None
    s = str(raw).strip().lower()
    if s in ("true", "yes", "1"):
        return True
    if s in ("false", "no", "0"):
        return False
    if "true" in s and "false" not in s:
        return True
    if "false" in s and "true" not in s:
        return False
    return None


def _parse_ending_answer(raw) -> int | None:
    if raw is None:
        return None
    s = str(raw).strip().lower()
    m = re.search(r"ending\\s*([1-4])", s) or re.search(r"ending([1-4])", s)
    if m:
        return int(m.group(1)) - 1
    if s in ("1", "2", "3", "4"):
        return int(s) - 1
    return None


def _parse_answer_index(raw, prefix="answer", max_idx=4) -> int | None:
    """Parse answer like 'answer1', 'answer2', 'solution1', 'option1' etc."""
    if raw is None:
        return None
    s = str(raw).strip().lower()
    # Try to find pattern like "answer1", "answer2", etc.
    m = re.search(rf"{prefix}\s*(\d+)", s)
    if m:
        return int(m.group(1)) - 1
    # Try just number
    m = re.search(r"(\d+)", s)
    if m:
        idx = int(m.group(1)) - 1
        if 0 <= idx < max_idx:
            return idx
    return None


def make_boolq_example(ex):
    if "instruction" in ex:
        prompt = _build_instruction_prompt(ex["instruction"], ex.get("input", ""))
        options = [" the correct answer is true", " the correct answer is false"]
        gold_raw = ex.get("answer", ex.get("output", ""))
        gold_bool = _parse_bool_answer(gold_raw)
        if gold_bool is None:
            raise ValueError(f"Unrecognized BoolQ answer: {gold_raw}")
        gold_index = 0 if gold_bool else 1
        return prompt, options, gold_index

    # passage + question -> 选项 yes / no
    passage = ex["passage"]
    question = ex["question"]
    gold = ex["answer"]  # bool / str
    if isinstance(gold, str):
        gold = gold.strip().lower() in ("true", "yes", "1")

    prompt = (
        "Read the following passage and answer the question with 'yes' or 'no'.\n\n"
        f"Passage: {passage}\n\n"
        f"Question: {question}\nAnswer:"
    )
    options = [" yes", " no"]
    gold_index = 0 if gold else 1
    return prompt, options, gold_index


def make_piqa_example(ex):
    # Instruction-tuning format
    if "instruction" in ex:
        prompt = _build_instruction_prompt(ex["instruction"], ex.get("input", ""))
        options = [" the correct answer is solution1", " the correct answer is solution2"]
        gold_raw = ex.get("answer", ex.get("output", "")).strip().lower()
        if "solution1" in gold_raw:
            gold_index = 0
        elif "solution2" in gold_raw:
            gold_index = 1
        else:
            raise ValueError(f"Unrecognized PIQA answer: {gold_raw}")
        return prompt, options, gold_index

    # Raw HuggingFace format
    goal = ex["goal"]
    sol1 = ex["sol1"]
    sol2 = ex["sol2"]
    prompt = (
        "Choose the more plausible solution to complete the goal.\n\n"
        f"Goal: {goal}\n"
        "Solutions:\n"
        "A) " + sol1 + "\n"
        "B) " + sol2 + "\n\n"
        "Answer with 'A' or 'B'.\nAnswer:"
    )
    options = [" A", " B"]
    # label: 0 or 1
    gold_index = int(ex["label"])
    return prompt, options, gold_index


def make_siqa_example(ex):
    # Instruction-tuning format
    if "instruction" in ex:
        prompt = _build_instruction_prompt(ex["instruction"], ex.get("input", ""))
        options = [" the correct answer is answer1", " the correct answer is answer2", " the correct answer is answer3"]
        gold_raw = ex.get("answer", ex.get("output", "")).strip().lower()
        gold_index = _parse_answer_index(gold_raw, prefix="answer", max_idx=3)
        if gold_index is None:
            raise ValueError(f"Unrecognized SIQA answer: {gold_raw}")
        return prompt, options, gold_index

    # Raw HuggingFace format
    context = ex["context"]
    question = ex["question"]
    a1 = ex["answerA"]
    a2 = ex["answerB"]
    a3 = ex["answerC"]

    prompt = (
        "Read the situation and answer the question.\n\n"
        f"Context: {context}\n"
        f"Question: {question}\n"
        "Options:\n"
        "A) " + a1 + "\n"
        "B) " + a2 + "\n"
        "C) " + a3 + "\n\n"
        "Answer with 'A', 'B' or 'C'.\nAnswer:"
    )
    options = [" A", " B", " C"]
    gold_index = int(ex["label"]) - 1  # 原始 label 是 "1"/"2"/"3"
    return prompt, options, gold_index


def make_hellaswag_example(ex):
    if "instruction" in ex:
        prompt = _build_instruction_prompt(ex["instruction"], ex.get("input", ""))
        options = [
            " the correct answer is ending1",
            " the correct answer is ending2",
            " the correct answer is ending3",
            " the correct answer is ending4",
        ]
        gold_raw = ex.get("answer", ex.get("output", ""))
        gold_index = _parse_ending_answer(gold_raw)
        if gold_index is None:
            raise ValueError(f"Unrecognized HellaSwag answer: {gold_raw}")
        return prompt, options, gold_index

    ctx_a = ex["ctx_a"]
    ctx_b = ex["ctx_b"]
    ctx = ctx_a + " " + ctx_b
    endings = ex["endings"]
    gold_index = int(ex["label"])

    prompt = (
        "Complete the following description in the most plausible way.\n\n"
        f"Context: {ctx}\n"
        "Options:\n"
    )
    for i, option in enumerate(endings):
        prompt += f"{i}) {option}\n"
    prompt += "\nAnswer with the index (0-3). Answer:"

    options = [f" {i}" for i in range(len(endings))]
    return prompt, options, gold_index


def make_winogrande_example(ex):
    """WinoGrande: Commonsense coreference resolution."""
    # Instruction-tuning format
    if "instruction" in ex:
        prompt = _build_instruction_prompt(ex["instruction"], ex.get("input", ""))
        options = [" the correct answer is option1", " the correct answer is option2"]
        gold_raw = ex.get("answer", ex.get("output", "")).strip().lower()
        if "option1" in gold_raw:
            gold_index = 0
        elif "option2" in gold_raw:
            gold_index = 1
        else:
            raise ValueError(f"Unrecognized WinoGrande answer: {gold_raw}")
        return prompt, options, gold_index

    # Raw HuggingFace format
    sentence = ex["sentence"]
    option1 = ex["option1"]
    option2 = ex["option2"]
    answer = ex["answer"]  # "1" or "2"

    prompt = (
        "Fill in the blank with the correct option.\n\n"
        f"Sentence: {sentence}\n"
        f"Option 1: {option1}\n"
        f"Option 2: {option2}\n\n"
        "Answer with '1' or '2'.\nAnswer:"
    )
    options = [" 1", " 2"]
    gold_index = int(answer) - 1
    return prompt, options, gold_index


def make_arc_example(ex):
    """ARC (Easy/Challenge): AI2 Reasoning Challenge."""
    # Instruction-tuning format
    if "instruction" in ex:
        prompt = _build_instruction_prompt(ex["instruction"], ex.get("input", ""))
        # Determine number of options (usually 4, sometimes 5)
        gold_raw = ex.get("answer", ex.get("output", "")).strip().lower()
        gold_index = _parse_answer_index(gold_raw, prefix="answer")
        if gold_index is None:
            raise ValueError(f"Unrecognized ARC answer: {gold_raw}")
        # Generate options based on detected count
        num_opts = 4
        if "answer5" in ex["instruction"].lower():
            num_opts = 5
        options = [f" the correct answer is answer{i+1}" for i in range(num_opts)]
        return prompt, options, gold_index

    # Raw HuggingFace format
    question = ex["question"]
    choices = ex["choices"]
    answer_key = ex["answerKey"]

    labels = choices["label"]
    texts = choices["text"]

    prompt = (
        "Answer the following question by choosing the correct option.\n\n"
        f"Question: {question}\n"
        "Options:\n"
    )
    for label, text in zip(labels, texts):
        prompt += f"{label}) {text}\n"
    prompt += f"\nAnswer with '{'/'.join(labels)}'.\nAnswer:"

    options = [f" {label}" for label in labels]
    gold_index = labels.index(answer_key)
    return prompt, options, gold_index


def make_obqa_example(ex):
    """OBQA: OpenBookQA."""
    # Instruction-tuning format
    if "instruction" in ex:
        prompt = _build_instruction_prompt(ex["instruction"], ex.get("input", ""))
        gold_raw = ex.get("answer", ex.get("output", "")).strip().lower()
        gold_index = _parse_answer_index(gold_raw, prefix="answer")
        if gold_index is None:
            raise ValueError(f"Unrecognized OBQA answer: {gold_raw}")
        options = [f" the correct answer is answer{i+1}" for i in range(4)]
        return prompt, options, gold_index

    # Raw HuggingFace format
    question_stem = ex.get("question_stem", ex.get("question", ""))
    choices = ex["choices"]
    answer_key = ex["answerKey"]

    labels = choices["label"]
    texts = choices["text"]

    prompt = (
        "Answer the following question using common knowledge.\n\n"
        f"Question: {question_stem}\n"
        "Options:\n"
    )
    for label, text in zip(labels, texts):
        prompt += f"{label}) {text}\n"
    prompt += f"\nAnswer with '{'/'.join(labels)}'.\nAnswer:"

    options = [f" {label}" for label in labels]
    gold_index = labels.index(answer_key)
    return prompt, options, gold_index


# ========== 通用评测循环 ==========

def eval_commonsense_single_task(args):
    task = args.eval_task.lower()
    # Handle task name aliases
    task_aliases = {
        "arc-e": "arc_easy",
        "arc_e": "arc_easy",
        "arce": "arc_easy",
        "arc-c": "arc_challenge",
        "arc_c": "arc_challenge",
        "arcc": "arc_challenge",
        "openbookqa": "obqa",
    }
    task = task_aliases.get(task, task)
    print(f"[INFO] Evaluating commonsense task = {task}")

    # 评测数据：优先走本地文件（类似 ./data/math_eval）
    local_candidates = _get_local_eval_file_candidates(args, task)
    explicit_path = getattr(args, "eval_data_file", None)
    local_path = None
    if explicit_path and os.path.exists(explicit_path):
        local_path = explicit_path
    else:
        local_path = next((p for p in local_candidates if os.path.exists(p)), None)
    ds = None
    if local_path and os.path.exists(local_path):
        print(f"[INFO] Using local file: {local_path}")
        ds = load_json_any(local_path)

    # 兼容旧工作流：本地文件不存在时，尝试从 HuggingFace datasets 加载。
    # 默认启用离线模式，避免误触发网络；如果你想让它自动下载，请在命令行前设置：
    #   export HF_DATASETS_OFFLINE=0
    if ds is None and "HF_DATASETS_OFFLINE" not in os.environ:
        os.environ["HF_DATASETS_OFFLINE"] = "1"

    # Task to make_example function mapping
    task_makers = {
        "boolq": make_boolq_example,
        "piqa": make_piqa_example,
        "siqa": make_siqa_example,
        "hellaswag": make_hellaswag_example,
        "winogrande": make_winogrande_example,
        "arc_easy": make_arc_example,
        "arc_challenge": make_arc_example,
        "obqa": make_obqa_example,
    }

    # HuggingFace dataset names for fallback loading
    hf_dataset_names = {
        "boolq": ("boolq", None),
        "piqa": ("piqa", None),
        "siqa": ("social_i_qa", None),
        "hellaswag": ("hellaswag", None),
        "winogrande": ("winogrande", "winogrande_xl"),
        "arc_easy": ("ai2_arc", "ARC-Easy"),
        "arc_challenge": ("ai2_arc", "ARC-Challenge"),
        "obqa": ("openbookqa", "main"),
    }

    if task not in task_makers:
        raise ValueError(f"Unsupported commonsense eval_task: {args.eval_task}. Supported: {list(task_makers.keys())}")

    make_example = task_makers[task]

    try:
        if ds is None:
            hf_name, hf_config = hf_dataset_names[task]
            if hf_config:
                ds = load_dataset(hf_name, hf_config, split="validation")
            else:
                ds = load_dataset(hf_name, split="validation")
    except Exception as e:
        print(f"[ERROR] Failed to load dataset '{task}': {e}")
        if local_candidates:
            print("[HINT] Local-file workflow (recommended):")
            for p in local_candidates:
                print(f"  - Put a JSON/JSONL file at: {p}")
            print("  - Then rerun this eval command.")
        raise

    # Collect model metadata for traceability
    metadata = get_model_metadata(args.ckpt_dir)
    metadata["eval_task"] = task
    metadata["base_model"] = args.base_model

    model, tokenizer, device = load_model_and_tokenizer(args.base_model, args.ckpt_dir)

    total = 0
    correct = 0
    results = []

    for ex in tqdm(ds, desc=task):
        prompt, options, gold_index = make_example(ex)

        # 遍历每个选项计算 logprob
        logps = []
        for opt in options:
            lp = option_logprob(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                option=opt,
                max_length=512,
            )
            logps.append(lp)

        pred_index = int(torch.tensor(logps).argmax().item())
        is_correct = (pred_index == gold_index)

        total += 1
        correct += int(is_correct)

        results.append(
            {
                "prompt": prompt,
                "options": options,
                "gold_index": gold_index,
                "pred_index": pred_index,
                "correct": bool(is_correct),
            }
        )

    acc = correct / max(total, 1)
    print(f"[RESULT] {task} accuracy = {acc:.4f}  ({correct}/{total})")

    save_eval_results(results, metadata, acc, args.output_file)


if __name__ == "__main__":
    args = get_eval_args()
    set_seed(42)
    eval_commonsense_single_task(args)
