#!/usr/bin/env python3
"""
twowiki_cells_score.py -- score OUR four cells (WW / WS / SW / SS) on AgentTTS's 2Wiki 100 from the Config B grid
runner's stores, with the certified rulers. No model, no GPU. DRAFT 31 Oct 2026 for Jonathan's line-by-line review.

INPUT (<run dir> = --out of twowiki_cells.sbatch)
  tts_test_2wiki_adapted.json      the 100 (adapter output; `answers` = gold + aliases, `ctxs_gt` = gold passages)
  retr_weak.jsonl / retr_strong.jsonl      {"k": "<idx>",        "v": <retrieval text>}      (run_labels_agenttts.Store)
  qa_weak.jsonl   / qa_strong.jsonl        {"k": "<idx>|<retr>", "v": <answer text>}   retr in {weak, strong}
  meta_*.jsonl                             {"k": ..., "n_in": .., "n_out": .., "secs": .., "truncated": ..}
CELLS   first letter = retrieval choice (W = Qwen2.5-7B, S = Qwen2.5-72B), second = answering model (W = Llama-3B, S = Llama-70B)
  WW = retr_weak   + qa_weak["i|weak"]      WS = retr_weak   + qa_strong["i|weak"]
  SW = retr_strong + qa_weak["i|strong"]    SS = retr_strong + qa_strong["i|strong"]
SCORERS (registration §6; certified `stair_common.score`, never re-implemented)
  lenient (standard ruler over the alias list) · em (--em-gold aliases|single) · f1 = QA F1 · Ret F1 = set F1 of the
  retrieval's selected reference numbers vs the gold passages' positions in the pool (parse_selection = the runner's).
COST    stair_common.cost.units(params_b, n_in, n_out) per generation, params retr 7.0 / 72.0, QA 3.0 / 70.0; cell cost =
  retrieval call + answering call; reported in units and as % of SS.
OUTPUT  scores_cells_per_instance.csv (+ .xlsx), scores_cells_summary.json: per cell overall and by question type
  (types joined from --types <json: question -> type> when given; tts_test.json carries none), oracle over the four
  cells, the Row A numbers alongside when --rowA <scores_summary.json of twowiki_score_sidecar.py> is given.
USAGE   python3 twowiki_cells_score.py --run <run dir> [--types their100_types.json] [--rowA ...] [--em-gold aliases]
"""
import argparse, collections, csv, json, os, re, sys

for p in ("/mnt/hdd/jonathan/STAIR/common", os.path.expanduser("~/slr/STAIR/common")):
    sys.path.insert(0, p)
try:  # stair_common needs tomllib (Python >= 3.11); on the cluster's 3.10 venv map it to the `tomli` backport (pip install tomli)
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli as _tomli; sys.modules["tomllib"] = _tomli
from stair_common.score import lenient, em, f1  # noqa: E402
from stair_common.cost import units  # noqa: E402

PARAMS = {"retr": {"weak": 7.0, "strong": 72.0}, "qa": {"weak": 3.0, "strong": 70.0}}
CELLS = {"WW": ("weak", "weak"), "WS": ("weak", "strong"), "SW": ("strong", "weak"), "SS": ("strong", "strong")}


def parse_selection(txt, n_refs):                    # identical to run_labels_agenttts.parse_selection
    part = txt.split('Selected references')[-1] if 'Selected references' in txt else txt
    nums = [int(x) for x in re.findall(r'\[(\d+)\]', part)]
    return sorted(set(n for n in nums if 1 <= n <= n_refs))


def load_store(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def set_f1(pred, gold):
    pred, gold = set(pred), set(gold)
    inter = len(pred & gold)
    if not inter:
        return 0.0
    p, r = inter / len(pred), inter / len(gold)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--types", default=None); ap.add_argument("--rowA", default=None)
    ap.add_argument("--em-gold", choices=["aliases", "single"], default="aliases")
    a = ap.parse_args()
    data = json.load(open(os.path.join(a.run, "tts_test_2wiki_adapted.json"), encoding="utf-8"))
    types = json.load(open(a.types, encoding="utf-8")) if a.types else {}
    types = types.get("question_to_type", types)
    retr = {w: load_store(os.path.join(a.run, f"retr_{w}.jsonl")) for w in ("weak", "strong")}
    qa = {w: load_store(os.path.join(a.run, f"qa_{w}.jsonl")) for w in ("weak", "strong")}
    meta = {f"{k}_{w}": load_store(os.path.join(a.run, f"meta_{k}_{w}.jsonl")) for k in ("retr", "qa") for w in ("weak", "strong")}
    rows = []
    for i, inst in enumerate(data):
        golds = list(inst["answers"]); pool = inst["ctxs_candidate"]
        gold_ids = sorted({pool.index(g) + 1 for g in inst["ctxs_gt"]})
        base = {"idx": i, "_id": inst["_id"], "type": types.get(inst["question"]), "question": inst["question"], "gold": golds[0]}
        for cell, (rw, qw) in CELLS.items():
            sel = parse_selection(retr[rw][str(i)], len(pool))
            ans = qa[qw][f"{i}|{rw}"]
            mr, mq = meta[f"retr_{rw}"][str(i)], meta[f"qa_{qw}"][f"{i}|{rw}"]
            cost = units(PARAMS["retr"][rw], mr["n_in"], mr["n_out"]) + units(PARAMS["qa"][qw], mq["n_in"], mq["n_out"])
            rows.append({**base, "cell": cell, "ret_choice": rw, "qa_choice": qw, "n_selected": len(sel), "gold_ids": json.dumps(gold_ids),
                         "ret_f1": round(set_f1(sel, gold_ids), 4), "answer": ans, "lenient": lenient(ans, golds),
                         "em": em(ans, golds if a.em_gold == "aliases" else golds[0]), "qa_f1": round(f1(ans, golds), 4),
                         "units": round(cost, 1), "truncated": int(bool(mr.get("truncated")) or bool(mq.get("truncated")))})
    def summ(rs):
        n = len(rs)
        return {"n": n, **{k: round(100 * sum(r[k] for r in rs) / n, 2) for k in ("lenient", "em", "qa_f1", "ret_f1")},
                "units": round(sum(r["units"] for r in rs) / n, 1), "n_selected": round(sum(r["n_selected"] for r in rs) / n, 2)}
    summary = {"em_gold": a.em_gold, "n": len(data), "cells": {}}
    ss_units = sum(r["units"] for r in rows if r["cell"] == "SS")
    for cell in CELLS:
        rs = [r for r in rows if r["cell"] == cell]
        s = summ(rs); s["cost_pct_of_SS"] = round(100 * sum(r["units"] for r in rs) / ss_units, 1)
        if types:
            s["by_type"] = {t: summ([r for r in rs if r["type"] == t]) for t in sorted({r["type"] for r in rs if r["type"]})}
        summary["cells"][cell] = s
    # oracle: best cell per instance on lenient (ties -> cheapest)
    oracle = []
    for i in range(len(data)):
        rs = [r for r in rows if r["idx"] == i]
        best = max(rs, key=lambda r: (r["lenient"], -r["units"]))
        oracle.append(best)
    summary["oracle_lenient"] = {**summ(oracle), "cost_pct_of_SS": round(100 * sum(r["units"] for r in oracle) / ss_units, 1),
                                 "cell_shares": dict(collections.Counter(r["cell"] for r in oracle))}
    if a.rowA and os.path.exists(a.rowA):
        summary["rowA_agenttts_reproduced"] = json.load(open(a.rowA))["overall"]
    keys = list(rows[0].keys())
    with open(os.path.join(a.run, "scores_cells_per_instance.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    json.dump(summary, open(os.path.join(a.run, "scores_cells_summary.json"), "w"), indent=1)
    try:
        import pandas as pd
        with pd.ExcelWriter(os.path.join(a.run, "scores_cells_per_instance.xlsx")) as xw:
            pd.DataFrame(rows).to_excel(xw, index=False, sheet_name="per_instance")
            pd.DataFrame([{"cell": c, **{k: v for k, v in s.items() if k != "by_type"}} for c, s in summary["cells"].items()] +
                         [{"cell": "oracle", **{k: v for k, v in summary["oracle_lenient"].items() if k != "cell_shares"}}]).to_excel(xw, index=False, sheet_name="summary")
    except Exception as e:
        print("xlsx not written:", e)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
