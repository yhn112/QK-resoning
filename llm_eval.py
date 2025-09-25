import argparse
from collections import defaultdict
from typing import Dict, Any, List, Optional
from utils import (
    load_jsonl, save_jsonl, save_metrics,
    extract_boxed, majority_vote,
    DIRECT_PROMPT, COT_PROMPT, THINKING_PROMPT
)
from models_local import LocalModel
from tqdm import tqdm
import os
import json
from datasets import load_dataset



def solve_direct(model: LocalModel, problem: str):
    prompt = DIRECT_PROMPT.format(problem=problem)
    outs, gen_counts = model.generate(prompt, n=1, temperature=0.0)
    finals = [extract_boxed(o) for o in outs]
    return {"samples": outs, "finals": finals, "chosen": finals[0], "gen_counts": gen_counts}

def solve_cot(model: LocalModel, problem: str):
    prompt = COT_PROMPT.format(problem=problem)
    outs, gen_counts = model.generate(prompt, n=1, temperature=0.0, repetition_penalty = 1.2)
    finals = [extract_boxed(o) for o in outs]
    return {"samples": outs, "finals": finals, "chosen": finals[0], "gen_counts": gen_counts}

def solve_cotsc(model: LocalModel, problem: str, k: int, temperature: float):
    #prompt = COT_PROMPT.format(problem=problem)
    prompt = THINKING_PROMPT.format(problem=problem)
    outs, gen_counts = model.generate(prompt, n=k, temperature=temperature, repetition_penalty = 1.2)
    finals = [extract_boxed(o) for o in outs]
    chosen = majority_vote(finals)
    return {"samples": outs, "finals": finals, "chosen": chosen, "gen_counts": gen_counts}

def load_hle():
    dataset = load_dataset("krammnic/hle-multichoice", split = 'train')
    rows = []
    for example in dataset:
        if example['answer_type'] == 'exactMatch':
            line = {
                'unique_id': example['id'],
                'problem': example['question'],
                'answer': example['answer'],
                'subject': example['raw_subject']
                }
            rows.append(line)
    return rows

def run_eval(
    mode: str,
    model_name_or_path: str,
    k: int = 20,
    temperature: float = 0.7,
    out_path: Optional[str] = "outputs/run.jsonl",
):
    ds = load_hle()

    model = LocalModel(model_name_or_path=model_name_or_path)

    total_gen_tokens = 0

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        out_f = open(out_path, "w", encoding="utf-8")
    else:
        out_f = None


    for ex in tqdm(ds, total=len(ds)):
        pid = ex['unique_id']
        problem = ex["problem"]
        gold = ex["answer"]
        subject = ex["subject"]

        if mode == "direct":
            res = solve_direct(model, problem)
        elif mode == "cot":
            res = solve_cot(model, problem)
        elif mode == "cotsc":
            res = solve_cotsc(model, problem, k=k, temperature=temperature)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        pred = res["chosen"]

        total_gen_tokens += sum(res["gen_counts"])

        row = {
            "id": pid,
            "subject": subject,
            "gold": gold,
            "chosen": pred,
            "finals": res["finals"],
            "gen_token_counts": res["gen_counts"],
            "generation": res["samples"]
        }

        if out_f:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()

    if out_f:
        out_f.close()



def main():
    ap = argparse.ArgumentParser(description="LLM math evaluator")
    ap.add_argument("--mode", type=str, choices=["direct", "cot", "cotsc"], default="cotsc")
    ap.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    ap.add_argument("--k", type=int, default=8, help="samples for cotsc")
    ap.add_argument("--temperature", type=float, default=0.7, help="temperature for cotsc")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    if args.out is None:
        model_short = args.model.split("/")[-1]  
        args.out = f"outputs/{args.mode}_{model_short}.jsonl"

    run_eval(
        mode=args.mode,
        model_name_or_path=args.model,
        k=args.k,
        temperature=args.temperature,
        out_path=args.out,
    )

if __name__ == "__main__":
    main()
