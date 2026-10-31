"""
utils.py -- IMPORT SHIM for AgentTTS's code_others/archon_main.py (and archon_hotpotqa.py).

WHY THIS FILE EXISTS. Their script does `from utils import precision_recall_f1, accuracy`, but no such file was ever
committed to github.com/fairyfali/agenttts (checked 31 Oct 2026: single commit 3f0c2df, no history of the file).
Without it the script cannot import. This shim supplies the two functions from their definitions -- set
precision / recall / F1 between the retrieval stage's selected passage ids and the gold passage ids -- and RECORDS
its inputs to a sidecar JSONL so the retrieval selections are kept (their script never writes them to disk).

It changes nothing about prompts, sampling or model calls. The numbers their script prints from these functions are
provisional; the reported Ret F1 is recomputed afterwards from the sidecar with the team's scorer.

Sidecar path: $STAIR_SIDECAR (set by twowiki_repro_run.sh). One JSON line per call.
"""
import json
import os
import time

_SIDECAR = os.environ.get("STAIR_SIDECAR")


def _record(kind, payload):
    if not _SIDECAR:
        return
    payload = dict(payload)
    payload["kind"] = kind
    payload["t"] = time.time()
    with open(_SIDECAR, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def precision_recall_f1(gt_ids, pred_ids):
    """Set precision/recall/F1 of predicted passage ids against gold passage ids (0-based ints as their script passes)."""
    gt, pred = set(gt_ids), set(pred_ids)
    inter = len(gt & pred)
    precision = inter / len(pred) if pred else 0.0
    recall = inter / len(gt) if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    _record("retrieval", {"gt_ids": sorted(gt), "pred_ids": list(pred_ids),
                          "precision": precision, "recall": recall, "f1": f1})
    return precision, recall, f1


def accuracy(gt_ids, pred_ids):
    """1.0 if every gold passage id was selected, else 0.0. (Their definition is unpublished; this quantity is NOT reported.)"""
    return 1.0 if set(gt_ids) <= set(pred_ids) else 0.0
