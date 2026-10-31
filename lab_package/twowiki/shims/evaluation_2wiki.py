"""
evaluation_2wiki.py -- IMPORT SHIM for AgentTTS's code_others/archon_main.py (and archon_hotpotqa.py).

WHY THIS FILE EXISTS. Their script does `from evaluation_2wiki import evaluate, metric_evaluation`; the file was never
committed to github.com/fairyfali/agenttts (checked 31 Oct 2026). The author's description of the missing scorer
(Fali Wang, 11 Aug 2026, reconfirmed 21 Oct): not strict exact match -- a lenient check that the gold answer appears
in the prediction. For 2Wiki the script passes the whole `answers` list (gold + Wikidata aliases).

WHAT THIS SHIM DOES. `metric_evaluation(response, gold_answers)` RECORDS the response and its gold list to a sidecar
JSONL -- the generated answers are otherwise never written to disk by their script -- and returns PROVISIONAL
(em, f1, precision, recall) in the same order their code unpacks. The authoritative scores (lenient / strict EM /
QA F1, certified `stair_common.score`) are computed afterwards from the sidecar. Nothing about prompts, sampling or
model calls is touched.

Provisional rules here (for the console only): em = any normalised gold is a substring of the normalised response;
f1/precision/recall = best token-overlap over the gold list (SQuAD-style normalisation).
"""
import json
import os
import re
import string
import time
from collections import Counter

_SIDECAR = os.environ.get("STAIR_SIDECAR")


def _norm(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _prf(pred, gold):
    p, g = _norm(pred).split(), _norm(gold).split()
    same = sum((Counter(p) & Counter(g)).values())
    if same == 0:
        return 0.0, 0.0, 0.0
    precision, recall = same / len(p), same / len(g)
    return 2 * precision * recall / (precision + recall), precision, recall


def metric_evaluation(response, gold_answers):
    golds = [gold_answers] if isinstance(gold_answers, str) else list(gold_answers)
    em = float(any(_norm(g) and _norm(g) in _norm(response) for g in golds))
    f1, precision, recall = max((_prf(response, g) for g in golds), key=lambda t: t[0])
    if _SIDECAR:
        with open(_SIDECAR, "a", encoding="utf-8") as f:
            f.write(json.dumps({"kind": "answer", "response": response, "gold_answers": golds,
                                "prov_em": em, "prov_f1": f1, "t": time.time()}, ensure_ascii=False) + "\n")
    return em, f1, precision, recall


def evaluate(examples, _unused=""):
    """Referenced by their script only in a commented-out line; provided so the import succeeds."""
    rows = [metric_evaluation(e.get("output", ""), e.get("answers", [])) for e in examples]
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows], [r[3] for r in rows]
