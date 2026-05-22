import sys
import os
import json
import re
import glob
from datetime import datetime
from typing import List
from tqdm.auto import tqdm

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from llama_args import get_eval_args

PROMPT_TEMPLATE = (
    "Below is an instruction that describes a task. "
    "Write a response that appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n### Response:"
)


def set_seed(seed: int = 42):
    import numpy as np
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _get_default_data_root(args) -> str:
    eval_data_dir = getattr(args, "eval_data_dir", None)
    if eval_data_dir:
        return eval_data_dir

    env_root = os.environ.get("HF_DISK_ROOT")
    if env_root:
        return env_root

    return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")


def load_json_any(file_path: str):
    with open(file_path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == "[":
            return json.load(f)
        return [json.loads(line) for line in f if line.strip()]


def _maybe_limit_samples(ds, max_eval_samples: int | None):
    if not max_eval_samples:
        return ds
    if hasattr(ds, "select"):
        return ds.select(range(min(max_eval_samples, len(ds))))
    return ds[:max_eval_samples]


def _get_first_existing_path(candidates: list[str]) -> str | None:
    return next((p for p in candidates if p and os.path.exists(p)), None)


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
    metadata = {
        "ckpt_dir": os.path.abspath(ckpt_dir),
        "eval_time": datetime.now().isoformat(),
    }

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

    run_dir = os.path.dirname(ckpt_dir) if os.path.basename(ckpt_dir) == "final_model" else ckpt_dir

    wandb_logs_dir = os.path.join(run_dir, "logs", "wandb")
    if os.path.exists(wandb_logs_dir) and metadata.get("model_mtime"):
        model_mtime = datetime.fromisoformat(metadata["model_mtime"])
        run_dirs = glob.glob(os.path.join(wandb_logs_dir, "run-*"))

        best_match = None
        best_time_diff = float("inf")

        for rd in run_dirs:
            dirname = os.path.basename(rd)
            parts = dirname.split("-")
            if len(parts) >= 3:
                try:
                    run_time_str = parts[1]
                    run_time = datetime.strptime(run_time_str, "%Y%m%d_%H%M%S")
                    run_id = parts[2]

                    if run_time < model_mtime:
                        time_diff = (model_mtime - run_time).total_seconds()
                        if time_diff < best_time_diff:
                            best_time_diff = time_diff
                            best_match = run_id
                except (ValueError, IndexError):
                    continue

        if best_match:
            metadata["training_wandb_run_id"] = best_match
            metadata["training_wandb_run_id_source"] = "inferred_from_logs_mtime"

    if "training_wandb_run_id" not in metadata:
        wandb_id_file = os.path.join(run_dir, "wandb_run_id.txt")
        if os.path.exists(wandb_id_file):
            with open(wandb_id_file, "r") as f:
                run_id = f.read().strip()
            id_file_mtime = os.path.getmtime(wandb_id_file)
            if metadata.get("model_mtime"):
                model_mtime_ts = datetime.fromisoformat(metadata["model_mtime"]).timestamp()
                if id_file_mtime > model_mtime_ts + 60:
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


def batch_generate(
    model,
    tokenizer,
    device,
    prompts: List[str],
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    all_generated = []

    for i in tqdm(range(0, len(prompts), batch_size), desc="Batch generation"):
        batch_prompts = prompts[i:i + batch_size]

        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]

        for j, output_id in enumerate(output_ids):
            gen_ids = output_id[input_len:]
            gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
            all_generated.append(gen_text)

    return all_generated


def build_gsm8k_prompt(question: str, prompt_style: str) -> str:
    if prompt_style == "instruction":
        instruction = (
            "Solve the following problem step by step. "
            "At the end, output ONLY the final answer in the format '#### answer'.\n\n"
            f"Question: {question}"
        )
        return PROMPT_TEMPLATE.format(instruction=instruction)

    return (
        "You are a helpful and precise math assistant.\n"
        "Solve the following problem step by step. "
        "At the end, output ONLY the final answer in the format '#### answer'.\n\n"
        f"Question: {question}\n\nAnswer:"
    )


def extract_gsm8k_answer(text: str) -> str:
    if "####" in text:
        tail = text.split("####")[-1]
    else:
        tail = text
    nums = re.findall(r"-?\d+\.?\d*", tail)
    if not nums:
        return tail.strip()
    return nums[-1].lstrip("0") or "0"


def evaluate_gsm8k(args):
    print("[INFO] Loading GSM8K test split...")
    data_root = _get_default_data_root(args)
    local_candidates = [
        getattr(args, "eval_data_file", None),
        os.path.join(data_root, "math_eval", "gsm8k_test.jsonl"),
        os.path.join(data_root, "math_eval", "gsm8k_test.json"),
    ]
    local_path = _get_first_existing_path(local_candidates)
    if local_path:
        print(f"[INFO] Using local file: {local_path}")
        ds = load_json_any(local_path)
    else:
        print("[INFO] Local file not found, downloading from HuggingFace...")
        ds = load_dataset("gsm8k", "main", split="test")
    ds = _maybe_limit_samples(ds, getattr(args, "max_eval_samples", None))

    metadata = get_model_metadata(args.ckpt_dir)
    metadata["eval_task"] = "gsm8k"
    metadata["base_model"] = args.base_model
    metadata["batch_size"] = args.batch_size

    model, tokenizer, device = load_model_and_tokenizer(
        args.base_model, args.ckpt_dir
    )

    prompt_style = getattr(args, "prompt_style", "assistant")
    prompts = []
    golds = []
    questions = []

    print("[INFO] Building prompts...")
    for ex in ds:
        question = ex["question"]
        gold = extract_gsm8k_answer(ex["answer"])
        prompt = build_gsm8k_prompt(question, prompt_style)
        prompts.append(prompt)
        golds.append(gold)
        questions.append(question)

    print(f"[INFO] Batch generating with batch_size={args.batch_size}...")
    generated_texts = batch_generate(
        model, tokenizer, device, prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results = []
    correct = 0
    for i, gen_text in enumerate(generated_texts):
        pred = extract_gsm8k_answer(gen_text)
        is_correct = (pred == golds[i])
        correct += int(is_correct)

        results.append({
            "question": questions[i],
            "gold": golds[i],
            "generated": gen_text,
            "pred": pred,
            "correct": bool(is_correct),
        })

    acc = correct / max(len(results), 1)
    print(f"[RESULT] GSM8K accuracy = {acc:.4f}  ({correct}/{len(results)})")

    save_eval_results(results, metadata, acc, args.output_file)


def build_math_prompt(problem: str, prompt_style: str) -> str:
    if prompt_style == "instruction":
        instruction = (
            "Solve the following competition-level math problem. "
            "Show your reasoning briefly, and at the end output the final answer "
            "in LaTeX format using \\boxed{answer}.\n\n"
            f"Problem: {problem}"
        )
        return PROMPT_TEMPLATE.format(instruction=instruction)

    return (
        "You are a helpful math assistant.\n"
        "Solve the following competition-level math problem. "
        "Show your reasoning briefly, and at the end output the final answer "
        "in LaTeX format using \\boxed{answer}.\n\n"
        f"Problem: {problem}\n\nSolution:"
    )


def extract_math_answer(text: str) -> str:
    m = re.findall(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m[-1].strip()
    lines = text.strip().splitlines()
    if lines:
        return lines[-1].strip()
    return text.strip()


def normalize_latex(s: str) -> str:
    s = s.strip()
    s = s.replace(" ", "")
    s = s.replace("\\,", "")
    return s


def evaluate_math(args):
    print("[INFO] Loading MATH test split...")
    data_root = _get_default_data_root(args)
    local_candidates = [
        getattr(args, "eval_data_file", None),
        os.path.join(data_root, "math_eval", "MATH_test.jsonl"),
        os.path.join(data_root, "math_eval", "MATH_test.json"),
    ]
    local_path = _get_first_existing_path(local_candidates)
    if local_path:
        print(f"[INFO] Using local file: {local_path}")
        ds = load_json_any(local_path)
    else:
        print("[INFO] Local file not found, downloading from HuggingFace...")
        ds = load_dataset("hendrycks/math", split="test")
    ds = _maybe_limit_samples(ds, getattr(args, "max_eval_samples", None))

    metadata = get_model_metadata(args.ckpt_dir)
    metadata["eval_task"] = "math"
    metadata["base_model"] = args.base_model
    metadata["batch_size"] = args.batch_size

    model, tokenizer, device = load_model_and_tokenizer(
        args.base_model, args.ckpt_dir
    )

    prompt_style = getattr(args, "prompt_style", "assistant")
    prompts = []
    golds = []
    gold_raws = []
    problems = []

    print("[INFO] Building prompts...")
    for ex in ds:
        problem = ex.get("instruction") or ex.get("problem")
        gold_raw = ex.get("output") or ex.get("solution")
        gold = extract_math_answer(gold_raw)
        prompt = build_math_prompt(problem, prompt_style)
        prompts.append(prompt)
        golds.append(gold)
        gold_raws.append(gold_raw)
        problems.append(problem)

    print(f"[INFO] Batch generating with batch_size={args.batch_size}...")
    generated_texts = batch_generate(
        model, tokenizer, device, prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results = []
    correct = 0
    for i, gen_text in enumerate(generated_texts):
        pred = extract_math_answer(gen_text)
        pred_norm = normalize_latex(pred)
        gold_norm = normalize_latex(golds[i])
        is_correct = (pred_norm == gold_norm)
        correct += int(is_correct)

        results.append({
            "problem": problems[i],
            "gold_solution": gold_raws[i],
            "gold": golds[i],
            "generated": gen_text,
            "pred": pred,
            "correct": bool(is_correct),
        })

    acc = correct / max(len(results), 1)
    print(f"[RESULT] MATH accuracy (strict string match) = {acc:.4f}  ({correct}/{len(results)})")

    save_eval_results(results, metadata, acc, args.output_file)


def build_metamathqa_prompt(
    instruction: str,
    input_text: str | None,
    prompt_style: str,
) -> str:
    question = instruction if not input_text else f"{instruction}\n{input_text}"
    if prompt_style == "instruction":
        instruction_text = (
            "Solve the following problem. At the end, output ONLY the final answer in Arabic numerals.\n\n"
            f"Problem: {question}"
        )
        return PROMPT_TEMPLATE.format(instruction=instruction_text)

    return (
        "You are a helpful and precise math assistant.\n"
        "Solve the following problem. At the end, output ONLY the final answer in Arabic numerals.\n\n"
        f"Problem: {question}\n\nAnswer:"
    )


def _extract_last_number(text: str) -> str:
    nums = re.findall(r"-?\d+\.?\d*", text)
    if not nums:
        return ""
    return nums[-1]


def _to_float_maybe(text: str):
    num = _extract_last_number(text)
    if not num:
        return None
    try:
        return float(num)
    except ValueError:
        return None


def _numeric_match(pred_text: str, gold_text: str) -> bool:
    pred_num = _to_float_maybe(pred_text)
    gold_num = _to_float_maybe(gold_text)
    if pred_num is not None and gold_num is not None:
        return abs(pred_num - gold_num) < 1e-4
    return _extract_last_number(pred_text) == _extract_last_number(gold_text)


def evaluate_metamathqa(args):
    print("[INFO] Loading MetaMathQA eval split...")
    data_root = _get_default_data_root(args)
    local_candidates = [
        getattr(args, "eval_data_file", None),
        os.path.join(data_root, "arithmetic", "MetaMathQA-395K.json"),
        os.path.join(data_root, "arithmetic", "MetaMathQA-100K.json"),
        os.path.join(data_root, "arithmetic", "math_10k.json"),
    ]
    local_path = _get_first_existing_path(local_candidates)
    if not local_path:
        raise FileNotFoundError(
            "[ERROR] MetaMathQA file not found. "
            "Pass --eval_data_file or place it under data/arithmetic/."
        )

    print(f"[INFO] Using local file: {local_path}")
    ds = load_json_any(local_path)
    ds = _maybe_limit_samples(ds, getattr(args, "max_eval_samples", None))

    metadata = get_model_metadata(args.ckpt_dir)
    metadata["eval_task"] = "metamathqa"
    metadata["base_model"] = args.base_model
    metadata["batch_size"] = args.batch_size

    model, tokenizer, device = load_model_and_tokenizer(
        args.base_model, args.ckpt_dir
    )

    prompt_style = getattr(args, "prompt_style", "assistant")
    prompts = []
    gold_raws = []
    instructions = []
    input_texts = []

    print("[INFO] Building prompts...")
    for ex in ds:
        instruction = ex.get("instruction") or ex.get("question") or ""
        input_text = ex.get("input") or ""
        gold_raw = ex.get("answer") or ex.get("output") or ""
        prompt = build_metamathqa_prompt(instruction, input_text, prompt_style)
        prompts.append(prompt)
        gold_raws.append(gold_raw)
        instructions.append(instruction)
        input_texts.append(input_text)

    print(f"[INFO] Batch generating with batch_size={args.batch_size}...")
    generated_texts = batch_generate(
        model, tokenizer, device, prompts,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    results = []
    correct = 0
    for i, gen_text in enumerate(generated_texts):
        is_correct = _numeric_match(gen_text, gold_raws[i])
        correct += int(is_correct)

        results.append({
            "instruction": instructions[i],
            "input": input_texts[i],
            "gold": gold_raws[i],
            "generated": gen_text,
            "correct": bool(is_correct),
        })

    acc = correct / max(len(results), 1)
    print(f"[RESULT] MetaMathQA accuracy = {acc:.4f}  ({correct}/{len(results)})")

    save_eval_results(results, metadata, acc, args.output_file)


if __name__ == "__main__":
    args = get_eval_args()

    if not hasattr(args, 'batch_size'):
        args.batch_size = 16

    set_seed(42)

    if args.eval_task.lower() == "gsm8k":
        evaluate_gsm8k(args)
    elif args.eval_task.lower() == "math":
        evaluate_math(args)
    elif args.eval_task.lower() in ("metamathqa", "metamath"):
        evaluate_metamathqa(args)
    else:
        raise ValueError(f"Unsupported eval_task for arithmetic: {args.eval_task}")
