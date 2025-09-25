import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
import numpy as np
from typing import List, Dict, Any
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv("with_llm_decision_hle_qwen3_8B_classic.csv")

_tokenizer = None
_model = None


def load_model_and_tokenizer():
    global _tokenizer, _model
    if _tokenizer is None or _model is None:
        print("Loading Qwen3-8B model and tokenizer...")
        _tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-8B",
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        _model = _model.cuda()
        _model.eval()
        print("Model and tokenizer loaded successfully!")
    return _tokenizer, _model


def extract_question_marks_embeddings(text: str, tokenizer, model, layer_idx: int = -1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]

    token_ids = inputs['input_ids'][0]
    question_mark_positions = []

    for i, token_id in enumerate(token_ids):
        token = tokenizer.decode([token_id])
        if '?' in token:
            question_mark_positions.append(i)

    if question_mark_positions:
        Q_embeddings = hidden_states[0, question_mark_positions, :]
        return Q_embeddings, question_mark_positions
    else:
        return None, []


def extract_sentence_endings_embeddings(text: str, tokenizer, model, layer_idx: int = -1):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    device = next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[layer_idx]

    token_ids = inputs['input_ids'][0]
    sentence_ending_positions = []

    for i, token_id in enumerate(token_ids):
        token = tokenizer.decode([token_id])
        if any(punct in token for punct in ['.', '!', '?']):
            sentence_ending_positions.append(i)

    if sentence_ending_positions:
        K_embeddings = hidden_states[0, sentence_ending_positions, :]
        return K_embeddings, sentence_ending_positions
    else:
        return None, []


def calculate_qk_norm(Q_embeddings, K_embeddings):
    if Q_embeddings is None or K_embeddings is None:
        return 0.0

    QK = torch.matmul(Q_embeddings, K_embeddings.transpose(0, 1))
    qk_norm = torch.norm(QK, p='fro').item()

    return qk_norm


def process_solution_qk(solution_text: str, layer_idx: int = -1):
    tokenizer, model = load_model_and_tokenizer()

    Q_embeddings, q_positions = extract_question_marks_embeddings(solution_text, tokenizer, model, layer_idx)
    K_embeddings, k_positions = extract_sentence_endings_embeddings(solution_text, tokenizer, model, layer_idx)
    qk_norm = calculate_qk_norm(Q_embeddings, K_embeddings)

    return {
        'qk_norm': qk_norm,
        'num_question_marks': len(q_positions),
        'num_sentence_endings': len(k_positions),
        'q_positions': q_positions,
        'k_positions': k_positions
    }

solve1 = df.iloc[0]["llm_solution"]
solve2 = df.iloc[10]["llm_solution"]

i = 0
def for_apply(solution):
    global i
    i += 1
    val = process_solution_qk(solution, 2)["qk_norm"] <= 2000
    print("working ", i, " ", val)
    return val

df2 = df.copy()
df2["qk_correct"] = df["llm_solution"].apply(for_apply)
df2.to_csv("with_qk_full_hle_qwen_8B_classic.csv")
