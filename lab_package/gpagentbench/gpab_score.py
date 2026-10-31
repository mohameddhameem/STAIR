#!/usr/bin/env python3
"""
gpab_score.py -- score GPAgentBench-2K episodes: deterministic metrics (Appendix C.2, ours byte-identical), FLOPs cost,
Table 4 dollars, and -- optionally -- the Dx / Tx judge with the paper's C.2 prompts on a LOCAL model.
DRAFT v1.0 2026-10-31 for Jonathan's line-by-line review.

INPUT   <run dir>/episodes_weak.jsonl, episodes_strong.jsonl (gpab_run.py), the data dir (ground truth), the case list.
JUDGE   --judge-url/--judge-model: Qwen2.5-72B-Instruct-AWQ via vLLM, temperature 0, max_tokens 16, prompts
        prompts/judge_dx_C2.txt and judge_tx_C2.txt VERBATIM. Dx in {0,1}; Tx in {0,0.5,1}; unparsable -> 0 and flagged.
        Only episodes that ended with `treat` have a Tx prediction; Dx is judged when a diagnosis exists (diagnose or treat).
        Timed-out / referred episodes: Dx = Tx = 0 (D2) -- reported with and without them.
COST    stair_common.cost.units(params_b, prompt_tokens, completion_tokens) summed over every agent call (incl. re-prompts),
        params 7.0 (weak) / 72.0 (strong); the simulator's calls are reported separately and NOT charged to the agent.
OUTPUT  <run dir>/scores_per_case.csv (+ .xlsx), scores_summary.json -- overall and by label / urgency, per agent, plus
        the oracle (best of two per case on mgmt_correct) and the paper's Qwen rows for the consistency check.
"""
import argparse, collections, csv, json, os, sys, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, "/mnt/hdd/jonathan/STAIR/common"); sys.path.insert(0, os.path.expanduser("~/slr/STAIR/common"))
import gpab_env as E  # noqa: E402
from stair_common.cost import units  # noqa: E402  (certified; never re-implemented)

PARAMS = {"weak": 7.0, "strong": 72.0}
PAPER = {"weak": {"mgmt": 47.4, "miss_refer": 89.6, "miss_gp": 21.1, "turns": 6.27, "dx": 40.4, "tx": 18.4, "usd": 445},
         "strong": {"mgmt": 46.4, "miss_refer": 54.7, "miss_gp": 52.8, "turns": 6.86, "dx": 54.1, "tx": 25.4, "usd": 473}}
DX_PROMPT = open(os.path.join(HERE, "prompts", "judge_dx_C2.txt"), encoding="utf-8").read()
TX_PROMPT = open(os.path.join(HERE, "prompts", "judge_tx_C2.txt"), encoding="utf-8").read()


def chat(url, model, prompt, max_tokens=16):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": max_tokens}).encode()
    req = urllib.request.Request(url.rstrip("/") + "/chat/completions", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["choices"][0]["message"]["content"].strip()


def parse_score(text, allowed):
    t = text.strip().split()[0].rstrip(".,;:") if text.strip() else ""
    try:
        v = float(t)
    except ValueError:
        return None
    return v if v in allowed else None


def judge(g, a, cache):
    """Fill dx_judge / tx_judge for one graded row; cache keyed on the exact prompt so repeats cost nothing."""
    g["dx_judge"] = g["tx_judge"] = 0.0; g["dx_judge_raw"] = g["tx_judge_raw"] = None
    if g["dx_pred"]:
        p = DX_PROMPT.format(predicted=g["dx_pred"], ground_truth=g["dx_gt"])
        raw = cache.get(p) or chat(a.judge_url, a.judge_model, p); cache[p] = raw
        v = parse_score(raw, {0.0, 1.0}); g["dx_judge_raw"] = raw; g["dx_judge"] = v if v is not None else 0.0; g["dx_unparsed"] = int(v is None)
    if g["tx_pred"]:
        p = TX_PROMPT.format(predicted=g["tx_pred"], ground_truth=g["tx_gt"])
        raw = cache.get(p) or chat(a.judge_url, a.judge_model, p); cache[p] = raw
        v = parse_score(raw, {0.0, 0.5, 1.0}); g["tx_judge_raw"] = raw; g["tx_judge"] = v if v is not None else 0.0; g["tx_unparsed"] = int(v is None)


def load_cases(data_dir, ids):
    import glob
    rows = {}
    for p in sorted(glob.glob(os.path.join(data_dir, "**", "*.jsonl"), recursive=True)):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                if r["case_id"] in ids:
                    for k, v in list(r.items()):
                        if isinstance(v, str) and v[:1] in "{[":
                            try: r[k] = json.loads(v)
                            except Exception: pass
                    rows[r["case_id"]] = r
    return rows


def summarize(rows):
    n = len(rows)
    if not n:
        return {"n": 0}
    m = lambda k: round(100 * sum(float(r[k]) for r in rows) / n, 2)
    refer_rows = [r for r in rows if r["label"] == "abstain_and_refer"]; manage_rows = [r for r in rows if r["label"] == "diagnose_and_manage"]
    out = {"n": n, "mgmt": m("mgmt_correct"), "timeout_rate": m("timeout"),
           "miss_refer": round(100 * sum(r["miss_refer"] for r in refer_rows) / len(refer_rows), 2) if refer_rows else None,
           "miss_gp": round(100 * sum(r["miss_gp"] for r in manage_rows) / len(manage_rows), 2) if manage_rows else None,
           "turns": round(sum(r["rounds"] for r in rows) / n, 2), "usd": round(sum(r["dollars"] for r in rows) / n, 1),
           "usd_performed": round(sum(r["dollars_performed"] or 0 for r in rows) / n, 1),
           "invalid_action_end_rate": m("invalid_action_end"),
           "flops_units": round(sum(r["agent_units"] for r in rows) / n, 1),
           "label_plan_tension": m("label_plan_tension")}
    if "dx_judge" in rows[0]:
        out["dx"] = m("dx_judge"); out["tx"] = m("tx_judge")
        treated = [r for r in rows if r["terminal"] == "treat"]
        out["dx_on_treated"] = round(100 * sum(r["dx_judge"] for r in treated) / len(treated), 2) if treated else None
        out["tx_on_treated"] = round(100 * sum(r["tx_judge"] for r in treated) / len(treated), 2) if treated else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--data", required=True); ap.add_argument("--cases", required=True)
    ap.add_argument("--judge-url", default=None); ap.add_argument("--judge-model", default=None)
    a = ap.parse_args()
    ids = [l.strip() for l in open(a.cases) if l.strip()]
    cases = load_cases(a.data, set(ids))
    per = {}; cache_path = os.path.join(a.run, "judge_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for tag in ("weak", "strong"):
        p = os.path.join(a.run, f"episodes_{tag}.jsonl")
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            rec = json.loads(line); case = cases[rec["case_id"]]
            g = E.grade(rec, case); g["agent"] = tag
            agent_calls = [c for c in rec["calls"] if c["who"].startswith("agent")]
            sim_calls = [c for c in rec["calls"] if c["who"] == "simulator"]
            g["agent_units"] = round(sum(units(PARAMS[tag], c["prompt_tokens"] or 0, c["completion_tokens"] or 0) for c in agent_calls), 1)
            g["sim_units"] = round(sum(units(7.0, c["prompt_tokens"] or 0, c["completion_tokens"] or 0) for c in sim_calls), 1)
            g["n_reprompts"] = sum(c["who"] == "agent_reprompt" for c in rec["calls"])
            if a.judge_url:
                judge(g, a, cache)
            per[(tag, rec["case_id"])] = g
    if a.judge_url:
        json.dump(cache, open(cache_path, "w"))
    rows = list(per.values())
    keys = list(rows[0].keys())
    with open(os.path.join(a.run, "scores_per_case.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)
    summary = {}
    for tag in ("weak", "strong"):
        rs = [r for r in rows if r["agent"] == tag]
        if not rs:
            continue
        summary[tag] = {"overall": summarize(rs), "paper_qwen_row": PAPER[tag],
                        "by_label": {lab: summarize([r for r in rs if r["label"] == lab]) for lab in sorted({r["label"] for r in rs})},
                        "by_urgency": {u: summarize([r for r in rs if r["gt_urgency"] == u]) for u in sorted({str(r["gt_urgency"]) for r in rs})}}
    # oracle and the four-way split on mgmt_correct (the routing headroom)
    both = [(per[("weak", c)], per[("strong", c)]) for c in ids if ("weak", c) in per and ("strong", c) in per]
    if both:
        summary["oracle_vs_pairs"] = {"n_pairs": len(both),
            "oracle_mgmt": round(100 * sum(max(w["mgmt_correct"], s["mgmt_correct"]) for w, s in both) / len(both), 2),
            "both_correct": sum(w["mgmt_correct"] and s["mgmt_correct"] for w, s in both),
            "weak_only": sum(w["mgmt_correct"] and not s["mgmt_correct"] for w, s in both),
            "strong_only": sum(s["mgmt_correct"] and not w["mgmt_correct"] for w, s in both),
            "both_fail": sum(not w["mgmt_correct"] and not s["mgmt_correct"] for w, s in both),
            "weak_missed_referral_on_refer_cases": sum(w["miss_refer"] for w, s in both)}
    json.dump(summary, open(os.path.join(a.run, "scores_summary.json"), "w"), indent=1)
    try:
        import pandas as pd
        with pd.ExcelWriter(os.path.join(a.run, "scores_per_case.xlsx")) as xw:
            pd.DataFrame(rows).to_excel(xw, index=False, sheet_name="per_case")
            flat = [{"agent": t, "group": "overall", **summary[t]["overall"]} for t in summary if t in PARAMS] + \
                   [{"agent": t, "group": "paper (Table 1)", **summary[t]["paper_qwen_row"]} for t in summary if t in PARAMS]
            pd.DataFrame(flat).to_excel(xw, index=False, sheet_name="summary")
    except Exception as e:
        print("xlsx not written:", e)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
