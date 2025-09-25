import json, math, os, re
from collections import Counter, defaultdict
from typing import Iterable, Dict, Any, List, Tuple, Optional
# from math_evaluations.evaluations import is_equiv


def extract_boxed(text: str) -> Optional[str]:
    start_positions = [m.start() for m in re.finditer(r"\\boxed\{", text)]
    if not start_positions:
        return None

    start = start_positions[-1] + len(r"\boxed{")

    depth = 1
    i = start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    if depth != 0:  # unbalanced braces
        return None

    return text[start : i - 1].strip()

def normalize(ans: Optional[str]) -> Optional[str]:
    if ans is None: return None
    s = ans.strip()
    try:
        if "/" in s and all(tok.strip("-+").isdigit() for tok in s.split("/", 1)):
            num, den = s.split("/", 1)
            return f"{int(num)}/{int(den)}"
        x = float(s)
        if math.isfinite(x):
            if abs(x - round(x)) < 1e-9:
                return str(int(round(x)))
            return f"{x:.12g}"
    except Exception:
        pass
    return s.replace(" ", "")

def equal(a: Optional[str], b: Optional[str]) -> bool:
    if a is None or b is None: return False
    a, b = normalize(a), normalize(b)
    if a == b: return True
    try:
        return abs(float(a) - float(b)) <= 1e-9
    except Exception:
        return False

def majority_vote(finals: List[Optional[str]]) -> Optional[str]:
    vals = [f for f in finals if f is not None]
    if not vals: return None
    return Counter(vals).most_common(1)[0][0]


# def majority_vote(
#     finals: List[Optional[str]],
# ) -> Optional[str]:
#     """Majority vote that merges mathematically equivalent answers.

#     - Counts None as missing.
#     - First aggregates exact string matches (fast).
#     - Then merges those buckets into equivalence classes via `is_equiv`.
#     - Returns the most frequent literal within the winning class.
#     """

#     vals = [f for f in finals if f is not None]
#     exact_counts = Counter(vals)

#     first_seen_index = {}
#     for i, v in enumerate(vals):
#         first_seen_index.setdefault(v, i)

#     # each class: {"rep": str, "members": Counter, "total": int, "order": int}
#     classes = []

#     def find_class(ans: str):
#         for cls in classes:
#             if is_equiv(cls["rep"], ans):
#                 return cls
#         return None

#     for ans, cnt in exact_counts.items():
#         cls = find_class(ans)
#         if cls is None:
#             cls = {
#                 "rep": ans,
#                 "members": Counter(),
#                 "total": 0,
#                 "order": first_seen_index[ans],
#             }
#             classes.append(cls)
#         cls["members"][ans] += cnt
#         cls["total"] += cnt

#     # choose the winning class: max total; tie-break by earliest order
#     winning = max(classes, key=lambda c: (c["total"], -c["order"]))

#     # within the winning class, choose the most frequent literal;
#     # tie-break by earliest first appearance
#     best_literal = max(
#         winning["members"].items(),
#         key=lambda kv: (kv[1], -first_seen_index[kv[0]])
#     )[0]

#     return best_literal


def load_jsonl(path: str, limit: Optional[int] = None) -> List[Dict[str, Any]]:
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                items.append(json.loads(line))
                if limit and len(items) >= limit: break
    return items

def save_jsonl(rows: Iterable[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def save_metrics(d: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=2)



DIRECT_PROMPT = """Give only the final answer to the math problem. Put it in \\boxed{{...}}.
Problem:
{problem}
"""

COT_PROMPT = """Solve the problem step by step. Then give only the final answer in \\boxed{{...}}.
Problem:
{problem}
"""

THINKING_PROMPT = """Solve the math problem. Think concisely. Do NOT show steps. Then give only the final answer in \\boxed{{...}}.
Problem:
{problem}
"""
