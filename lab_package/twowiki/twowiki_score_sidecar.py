#!/usr/bin/env python3
"""
twowiki_score_sidecar.py -- authoritative scores for Row A ("AgentTTS (reproduced), their 100") from the sidecar that
the two import shims wrote during their archon_main.py run. No model, no GPU, no network. DRAFT 31 Oct 2026 for
Jonathan's line-by-line review.

INPUT   <run dir>/sidecar.jsonl  -- one "answer" record (response + gold list) and one "retrieval" record (gold ids +
        selected ids) per instance, in processing order; their script walks tts_test.json in file order, so record i
        belongs to instance i of the data file (the script asserts that the counts match and that the gold list of
        record i equals instance i's `answers`, so a misalignment cannot pass silently).
        <data>  -- their tts_test.json (the 100), for question / type / answers.

SCORERS (certified `stair_common.score`, imported -- never re-implemented here; see common/stair_common/score.py):
  lenient  = standard ruler over the gold list (gold or any alias appears in the output)   -> AgentTTS-comparable
  em       = normalised exact match; --em-gold aliases (default; certified behaviour: best over the gold list)
             or --em-gold single (against the first gold string only)                       -> Jonathan's [OPEN] item, §6
  f1       = token-overlap F1, best over the gold list                                      -> their "Gen F1" / QA F1
  ret_f1   = set F1 of selected vs gold passage ids (from the retrieval record)              -> their "Ret F1"
The shims' provisional console numbers are ignored here.

OUTPUT  <run dir>/scores_per_instance.csv (+ .xlsx when openpyxl is present) and scores_summary.json:
        overall and per question type (compositional / comparison / bridge_comparison / inference), n per cell.

USAGE (lab, slr-env has pandas):  python3 twowiki_score_sidecar.py --run /mnt/hdd/jonathan/runs/twowiki/repro_their100 \
        --data /mnt/hdd/jonathan/agenttts_verify/data/2wikimultihopqa/tts_test.json [--em-gold aliases|single]
"""
import argparse, collections, hashlib, json, os, sys

STAIR_COMMON = "/mnt/hdd/jonathan/STAIR/common"
sys.path.insert(0, STAIR_COMMON)
from stair_common.score import lenient, em, f1  # noqa: E402  (certified rulers; RULER_SHA256 check lives in that module)


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ret_f1(gt_ids, pred_ids):
    gt, pred = set(gt_ids), set(pred_ids)
    inter = len(gt & pred)
    if not inter:
        return 0.0
    p, r = inter / len(pred), inter / len(gt)
    return 2 * p * r / (p + r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--em-gold", choices=["aliases", "single"], default="aliases")
    a = ap.parse_args()

    data = json.load(open(a.data, encoding="utf-8"))
    rows = [json.loads(l) for l in open(os.path.join(a.run, "sidecar.jsonl"), encoding="utf-8")]
    answers = [r for r in rows if r["kind"] == "answer"]
    retrievals = [r for r in rows if r["kind"] == "retrieval"]
    n = len(answers)
    assert n == len(retrievals), f"{n} answer records vs {len(retrievals)} retrieval records"
    assert n <= len(data), f"{n} records but only {len(data)} instances in the data file"
    if n < len(data):
        print(f"WARNING: {n} of {len(data)} instances scored (smoke or partial run) -- not a reportable row")

    per = []
    for i, (ans, ret) in enumerate(zip(answers, retrievals)):
        ex = data[i]
        golds = [ex["answers"]] if isinstance(ex["answers"], str) else list(ex["answers"])
        assert ans["gold_answers"] == golds, f"instance {i}: sidecar gold list differs from the data file -- misaligned"
        out = ans["response"]
        per.append({
            "idx": i, "type": ex.get("type"), "question": ex["question"], "gold": golds[0], "n_aliases": len(golds) - 1,
            "output": out, "output_chars": len(out),
            "lenient": lenient(out, golds),
            "em": em(out, golds if a.em_gold == "aliases" else golds[0]),
            "qa_f1": round(f1(out, golds), 4),
            "ret_f1": round(ret_f1(ret["gt_ids"], ret["pred_ids"]), 4),
            "ret_gt_ids": json.dumps(ret["gt_ids"]), "ret_pred_ids": json.dumps(ret["pred_ids"]),
        })

    def summ(group):
        m = len(group)
        return {"n": m, **{k: round(100 * sum(g[k] for g in group) / m, 2) for k in ("lenient", "em", "qa_f1", "ret_f1")},
                "median_output_chars": sorted(g["output_chars"] for g in group)[m // 2]}

    by_type = collections.defaultdict(list)
    for g in per:
        by_type[g["type"]].append(g)
    summary = {"row": "AgentTTS (reproduced), their 100", "n_scored": n, "em_gold": a.em_gold,
               "sidecar_sha256": sha256(os.path.join(a.run, "sidecar.jsonl")), "data_sha256": sha256(a.data),
               "overall": summ(per), "by_type": {t: summ(v) for t, v in sorted(by_type.items())}}

    import csv
    keys = list(per[0].keys())
    with open(os.path.join(a.run, "scores_per_instance.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per)
    try:
        import pandas as pd
        with pd.ExcelWriter(os.path.join(a.run, "scores_per_instance.xlsx")) as xw:
            pd.DataFrame(per).to_excel(xw, index=False, sheet_name="per_instance")
            pd.DataFrame([{"group": "overall", **summary["overall"]}] +
                         [{"group": t, **s} for t, s in summary["by_type"].items()]).to_excel(xw, index=False, sheet_name="summary")
    except Exception as e:  # xlsx is a convenience for review; the CSV is the record
        print("xlsx not written:", e)
    json.dump(summary, open(os.path.join(a.run, "scores_summary.json"), "w"), indent=1)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
