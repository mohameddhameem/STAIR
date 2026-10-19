# 13 token_costs.py -- measure the ruler. CPU-only, read-only.
#
# The grid campaign logged every call's real token counts at generation time:
#   meta_retr_weak/strong.jsonl, meta_qa_weak/strong.jsonl
#   (FLAT lines: {"k": "0", "n_in": 12007, "truncated": false, "n_out": 768,
#    "secs": 13.409}; QA keys are "idx|weak" / "idx|strong", naming which
#    retrieval produced the evidence. Verified against the real files 8 Oct.)
#
# This script turns those measured counts into the TOKEN-AWARE cost family
# (the AgentTTS convention): cost per call = params (B) x (prompt + generated
# tokens), normalized so the cheapest call type = 1.
#
# SIX prices, not four (decided 8 Oct after the double-measurement check):
# QA prompts carry the selected passages, and the strong retriever selects far
# fewer tokens than the weak one (means ~331 vs ~855), so each QA price is
# conditional on the retrieval choice. GIVEN_W / GIVEN_S = evidence came from
# the weak / strong retrieval.
import os, json
import numpy as np

GRID_DIR = "/mnt/hdd/jonathan/runs/full_run_p100"
GRID_FILE = os.path.join(GRID_DIR, "router_data_lenient.jsonl")
OUT = "/mnt/hdd/jonathan/runs/token_costs"
PARAMS_WEAK, PARAMS_STRONG = 1.5, 7.0   # Config A: Qwen2.5-1.5B / 7B, both stages

def load_meta(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line)
            d[r["k"]] = {kk: vv for kk, vv in r.items() if kk != "k"}
        except Exception: pass
    return d

def summarize(meta, graded, suffix=None):
    """Mean n_in+n_out over graded instances; suffix filters QA keys by the
    retrieval choice that produced the evidence ('|weak' or '|strong')."""
    toks, oom, trunc = [], 0, 0
    for k, v in meta.items():
        if suffix is not None and not k.endswith(suffix): continue
        if int(k.split("|")[0]) not in graded: continue
        if v.get("oom"): oom += 1; continue
        if v.get("truncated"): trunc += 1
        toks.append(v["n_in"] + v["n_out"])
    return {"n_calls": len(toks), "mean_tokens": float(np.mean(toks)),
            "median_tokens": float(np.median(toks)),
            "oom_skipped": oom, "truncated": trunc}

def main():
    os.makedirs(OUT, exist_ok=True)
    graded = {json.loads(l)["idx"] for l in open(GRID_FILE)}
    metas = {}
    for f in ("retr_weak", "retr_strong", "qa_weak", "qa_strong"):
        path = os.path.join(GRID_DIR, f"meta_{f}.jsonl")
        if not os.path.exists(path):
            raise SystemExit(f"MISSING: {path} -- report this before going further.")
        metas[f] = load_meta(path)

    # six call types: (name, meta file, QA evidence filter, params)
    kinds = [("retr_weak",         "retr_weak",   None,      PARAMS_WEAK),
             ("retr_strong",       "retr_strong", None,      PARAMS_STRONG),
             ("qa_weak_given_W",   "qa_weak",     "|weak",   PARAMS_WEAK),
             ("qa_weak_given_S",   "qa_weak",     "|strong", PARAMS_WEAK),
             ("qa_strong_given_W", "qa_strong",   "|weak",   PARAMS_STRONG),
             ("qa_strong_given_S", "qa_strong",   "|strong", PARAMS_STRONG)]
    report = {"params_B": {"weak": PARAMS_WEAK, "strong": PARAMS_STRONG},
              "convention": "AgentTTS token-aware: params x (n_in + n_out), "
                            "measured at generation time; QA price conditional "
                            "on the retrieval choice (GIVEN_W / GIVEN_S)"}
    raw, abs_cost = {}, {}
    for name, f, suffix, params in kinds:
        raw[name] = summarize(metas[f], graded, suffix)
        abs_cost[name] = params * raw[name]["mean_tokens"]
    report["measured"] = raw

    unit = min(abs_cost.values())
    norm = {c: round(abs_cost[c] / unit, 3) for c in abs_cost}
    report["normalized_costs"] = norm
    report["gate_eval_constants"] = {       # read by gate_eval.py at import
        "COST_RET_W": norm["retr_weak"],
        "COST_RET_S": norm["retr_strong"],
        "COST_QA_W_GIVEN_W": norm["qa_weak_given_W"],
        "COST_QA_W_GIVEN_S": norm["qa_weak_given_S"],
        "COST_QA_S_GIVEN_W": norm["qa_strong_given_W"],
        "COST_QA_S_GIVEN_S": norm["qa_strong_given_S"]}
    report["flat_family_for_comparison"] = {
        "COST_RET_W": 1.0, "COST_RET_S": 4.7, "COST_QA_W": 1.0, "COST_QA_S": 4.7,
        "note": "params ratio only; kept for the Ruler slide's simple story"}
    out_path = os.path.join(OUT, "token_costs.json")
    json.dump(report, open(out_path, "w"), indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nwrote {out_path}")

if __name__ == "__main__":
    main()
