import argparse
import json
import re
import jsonlines
from fraction import Fraction
from vllm import LLM, SamplingParams
import sys
import torch
import gc
from grader import math_equal
import wandb
from tqdm.auto import tqdm
import os

MAX_INT = sys.maxsize


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def extract_answer_number(completion):
    text = completion.split('The answer is: ')
    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)
        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]
                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        return round(float(frac.numerator / frac.denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def batch_data(data_list, batch_size=1):
    n = len(data_list) // batch_size
    batch_data = []
    for i in range(n-1):
        start = i * batch_size
        end = (i+1)*batch_size
        batch_data.append(data_list[start:end])

    last_start = (n-1) * batch_size
    last_end = MAX_INT
    batch_data.append(data_list[last_start:last_end])
    return batch_data


def gsm8k_test(model, data_path, start=0, end=MAX_INT, batch_size=1, tensor_parallel_size=1):
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    gc.collect()

    INVALID_ANS = "[invalid]"
    gsm8k_ins = []
    gsm8k_answers = []
    problem_prompt = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: Let's think step by step."
    )
    with open(data_path, "r+", encoding="utf8") as f:
        for idx, item in enumerate(jsonlines.Reader(f)):
            temp_instr = problem_prompt.format(instruction=item["question"])
            gsm8k_ins.append(temp_instr)
            temp_ans = item['answer'].split('#### ')[1]
            temp_ans = int(temp_ans.replace(',', ''))
            gsm8k_answers.append(temp_ans)

    gsm8k_ins = gsm8k_ins[start:end]
    gsm8k_answers = gsm8k_answers[start:end]
    print(f'[INFO] GSM8K samples: {len(gsm8k_ins)}')
    batch_gsm8k_ins = batch_data(gsm8k_ins, batch_size=batch_size)

    stop_tokens = ["Instruction:", "Instruction", "Response:", "Response"]
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=256, stop=stop_tokens)
    llm = LLM(model=model, tensor_parallel_size=tensor_parallel_size)
    res_completions = []
    result = []

    print("\nGenerating responses...")
    for idx, (prompt, prompt_answer) in enumerate(
        tqdm(zip(batch_gsm8k_ins, gsm8k_answers),
            total=len(batch_gsm8k_ins),
            desc="Generating responses")
    ):
        if isinstance(prompt, list):
            pass
        else:
            prompt = [prompt]

        completions = llm.generate(prompt, sampling_params)
        for output in completions:
            generated_text = output.outputs[0].text
            res_completions.append(generated_text)

    print("\nEvaluating responses...")
    invalid_outputs = []
    for idx, (prompt, completion, prompt_answer) in enumerate(
        tqdm(
            zip(gsm8k_ins, res_completions, gsm8k_answers),
            total=len(gsm8k_ins),
            desc="Evaluating answers")
    ):
        y_pred = extract_answer_number(completion)
        if y_pred is not None:
            result.append(float(y_pred) == float(prompt_answer) or math_equal(y_pred, prompt_answer))
        else:
            result.append(False)
            temp = {'question': prompt, 'output': completion, 'answer': prompt_answer}
            invalid_outputs.append(temp)

    acc = sum(result) / len(result)

    if not args.no_wandb:
        wandb.log({"eval/gsm8k_acc": acc})

    print(f'[INFO] Invalid outputs: {len(invalid_outputs)}')
    print(f'[RESULT] GSM8K length={len(result)}, accuracy={acc:.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str)
    parser.add_argument("--data_file", type=str, default='data/math_eval/gsm8k_test.jsonl')
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--end", type=int, default=MAX_INT)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
    parser.add_argument("--run_dir", type=str)
    parser.add_argument("--no_wandb", action="store_true")

    args = parser.parse_args()

    if args.run_dir and not args.no_wandb:
        try:
            with open(os.path.join(args.run_dir, "wandb_run_id.txt"), "r") as f:
                wandb_run_id = f.read().strip()
            wandb.init(id=wandb_run_id, project="csplora_eval", resume="must")
        except FileNotFoundError:
            print("WandB run ID file not found, starting new run")
            wandb.init(project="csplora_eval")

    return args


if __name__ == "__main__":
    args = parse_args()
    gsm8k_test(model=args.model, data_path=args.data_file, start=args.start, end=args.end,
               batch_size=args.batch_size, tensor_parallel_size=args.tensor_parallel_size)
