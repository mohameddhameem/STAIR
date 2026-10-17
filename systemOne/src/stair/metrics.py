"""Answer EM/F1 with the normalisation and yes/no rule of the official hotpot_evaluate_v1.py."""

import re
import string
from collections import Counter

_NO_PARTIAL_CREDIT = {"yes", "no", "noanswer"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def f1_score(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in _NO_PARTIAL_CREDIT or ref in _NO_PARTIAL_CREDIT) and pred != ref:
        return 0.0
    pred_tokens, ref_tokens = pred.split(), ref.split()
    num_same = sum((Counter(pred_tokens) & Counter(ref_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))
