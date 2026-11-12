#!/usr/bin/env python3
"""
twowiki_price_sidecar.py -- PRICE the AgentTTS pipeline from the token counts logged by shims/sitecustomize.py (12 Nov 2026,
DRAFT for Jonathan's review). CPU, seconds, no model, no network. Reads <run>/sidecar.jsonl ("gen" records interleaved with the
"retrieval"/"answer" records of the other two shims) and groups the gen calls per question (everything up to each "answer").
Per question: retrieval = gen calls on the 72B (params 72); generation = gen calls on the 3B except the last one (params 3);
fuser = the last 3B call before the answer. Units = stair_common.cost.units(params_b, prompt_tokens, generated_tokens), the same
function and parameter counts as our four cells (72 / 3). AgentTTS's convention counts the prompt ONCE per call even when
several samples are drawn from it; because sitecustomize logs the summed batch prompt (n_in) and the batch size (n_seq_in),
both conventions are reported: "prompt once" = n_in / n_seq_in (their convention; headline) and "prompt per sequence" = n_in.
Denominator: our standalone always-strong on the same questions (their 100 -> C_SS 585.7 units/question from the test600 cells,
positions 0-99; override with --c-ss). Writes <run>/cost_summary.json. Usage:
  python3 -I twowiki_price_sidecar.py --run /common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/repro_their100_cost \
      --cost-py ~/slr/STAIR/common/stair_common/cost.py [--c-ss 585.7]
"""
import argparse, importlib.util, json, os, statistics
ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--cost-py", required=True); ap.add_argument("--c-ss", type=float, default=585.7)
a = ap.parse_args()
spec = importlib.util.spec_from_file_location("stair_cost", os.path.expanduser(a.cost_py)); cost = importlib.util.module_from_spec(spec); spec.loader.exec_module(cost)
units = cost.units
PARAMS = {"72B": 72.0, "3B": 3.0}


def model_class(name):
    n = name.lower()
    if "72b" in n: return "72B"
    if "3b" in n: return "3B"
    raise SystemExit(f"unexpected model in gen record: {name}")


rows = [json.loads(l) for l in open(os.path.join(a.run, "sidecar.jsonl"), encoding="utf-8") if l.strip()]
questions, cur = [], []
for r in rows:
    if r["kind"] == "gen":
        cur.append(r)
    elif r["kind"] == "answer":
        questions.append(cur); cur = []
assert questions and all(questions), "a question has no gen records -- sitecustomize did not instrument this run"
per = []
for q in questions:
    ret = [g for g in q if model_class(g["model"]) == "72B"]
    gen3 = [g for g in q if model_class(g["model"]) == "3B"]
    assert ret and len(gen3) >= 2, f"unexpected call pattern: {len(ret)} x 72B, {len(gen3)} x 3B"
    fuser, gens = gen3[-1], gen3[:-1]
    def price(calls, p, once):
        return sum(units(p, (g["n_in"] / max(g["n_seq_in"], 1)) if once else g["n_in"], g["n_out"]) for g in calls)
    rec = {}
    for once, tag in ((True, "once"), (False, "perseq")):
        rec[f"ret_{tag}"] = price(ret, 72.0, once); rec[f"gen_{tag}"] = price(gens, 3.0, once); rec[f"fuser_{tag}"] = price([fuser], 3.0, once)
        rec[f"total_{tag}"] = rec[f"ret_{tag}"] + rec[f"gen_{tag}"] + rec[f"fuser_{tag}"]
    rec["n_gen_calls"] = len(gens); rec["n_samples"] = sum(g["n_seq_out"] for g in gens); rec["gen_tokens_out"] = sum(g["n_out"] for g in gens)
    rec["ret_prompt_tokens"] = sum(g["n_in"] / max(g["n_seq_in"], 1) for g in ret)
    per.append(rec)
n = len(per)
def mean(k): return sum(p[k] for p in per) / n
summary = {"version": "twowiki_price_sidecar v1.0 2026-11-12", "run": a.run, "n_questions": n, "c_ss_units": a.c_ss,
           "mean_units": {k: round(mean(k), 1) for k in ("ret_once", "gen_once", "fuser_once", "total_once", "ret_perseq", "gen_perseq", "fuser_perseq", "total_perseq")},
           "pct_of_SS": {"prompt_once (AgentTTS convention, headline)": round(100 * mean("total_once") / a.c_ss, 1),
                         "prompt_per_sequence": round(100 * mean("total_perseq") / a.c_ss, 1)},
           "mean_samples_per_question": round(mean("n_samples"), 1), "mean_gen_calls_per_question": round(mean("n_gen_calls"), 1),
           "mean_ret_prompt_tokens": round(mean("ret_prompt_tokens"), 0), "mean_gen_tokens_out": round(mean("gen_tokens_out"), 0),
           "median_total_once": round(statistics.median(p["total_once"] for p in per), 1)}
json.dump(summary, open(os.path.join(a.run, "cost_summary.json"), "w"), indent=1)
m = summary["mean_units"]
print(f"n={n} | per question (prompt once): retrieval {m['ret_once']:.0f} + {summary['mean_samples_per_question']:.0f} samples {m['gen_once']:.0f} + fuser {m['fuser_once']:.0f} = {m['total_once']:.0f} units")
print(f"AgentTTS pipeline cost = {summary['pct_of_SS']['prompt_once (AgentTTS convention, headline)']:.1f} % of our always-strong (C_SS {a.c_ss}) | prompt-per-sequence convention: {summary['pct_of_SS']['prompt_per_sequence']:.1f} %")
print(f"retrieval prompt ≈ {summary['mean_ret_prompt_tokens']:.0f} tokens; generated tokens per question ≈ {summary['mean_gen_tokens_out']:.0f}")
print("wrote cost_summary.json")
