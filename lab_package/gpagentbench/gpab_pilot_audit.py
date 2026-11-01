#!/usr/bin/env python3
"""
gpab_pilot_audit.py -- READ-ONLY audit of the pilot episodes (job 347293), CPU, seconds. DRAFT v1.0 2026-11-01 for
Jonathan's review. Writes one Excel file; touches nothing else.

Why: the 7B's management accuracy on manage cases is 46.7 % vs the paper's 78.9 %, and 11 of its 30 manage episodes
ended as `invalid_action` (D1: three malformed / not-allowed actions in one turn). This shows WHAT the 7B wrote.
Sheets
  invalid_7B   every weak episode that ended invalid_action: case, label, round, actions so far, workflow state at the
               failure (diagnosed or still collecting), and the failed reply classified by the SAME parser the harness
               uses (gpab_env._ACTION_RE / ALLOWED / MASKED): no <action> block · several blocks · unknown type ·
               valid type but not allowed in this state (e.g. ask/refer after diagnose) -- plus the reply text (the
               harness saved the last of the three attempts, first 500 characters).
  timeouts_72B every strong episode that hit the 10-turn cap: the sequence of action types and the dollars ordered.
  repeats      the 10 repeated cases (rep2) vs their first run, per agent: same terminal action? same rounds? (decoding
               = the model file's sampling defaults, so some churn is expected).
  summary      counts per failure class and per state.
Usage (cluster login node, vllm-env for pandas/openpyxl):
  ~/slr/vllm-env/bin/python gpab_pilot_audit.py --run /common/scratch/users/d/dohyung.ahn.2025/runs/gpagentbench/pilot60
"""
import argparse, collections, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gpab_env as E


def load(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()] if os.path.exists(p) else []


def classify(reply, state):
    blocks = E._ACTION_RE.findall(reply or "")
    if not blocks:
        return "no <action> block", ""
    if len(blocks) > 1:
        return f"{len(blocks)} action blocks", ",".join(b[0] for b in blocks)
    raw = blocks[0][0].strip().lower()
    if raw not in set(E.ALLOWED[state]) | set(E.MASKED[state]):
        return "unknown action type", raw
    if raw in E.MASKED[state]:
        return f"not allowed while {state}", raw
    return "parsed and allowed (?)", raw


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); a = ap.parse_args()
    import pandas as pd
    out = os.path.join(a.run, "pilot_audit.xlsx")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    weak = load(os.path.join(a.run, "episodes_weak.jsonl")); strong = load(os.path.join(a.run, "episodes_strong.jsonl"))
    import csv                                         # labels from the rescored per-case table (episode records carry none)
    label = {r["case_id"]: r["label"] for r in csv.DictReader(open(os.path.join(a.run, "scores_per_case.csv"), encoding="utf-8"))}
    inv = []
    for r in weak:
        if r["terminal"] != "invalid_action":
            continue
        acts = r.get("actions", [])
        prior = [x["type"] for x in acts if x["type"] != "invalid"]
        state = r.get("state") or ("diagnosed" if "diagnose" in prior else "collecting_evidence")   # state when it ended
        reply = next((x.get("body", "") for x in acts if x["type"] == "invalid"), "")
        cls, found = classify(reply, state)
        inv.append({"case_id": r["case_id"], "label": label.get(r["case_id"]), "failed_at_round": len(prior) + 1,
                    "actions_before": " > ".join(prior), "state_at_failure": state, "failure_class": cls,
                    "action_type_found": found, "reprompts_total": sum(c["who"] == "agent_reprompt" for c in r.get("calls", [])),
                    "failed_reply_first500": reply})
    to = []
    for r in strong:
        if r["terminal"] != "timeout":
            continue
        acts = [x["type"] for x in r.get("actions", [])]
        to.append({"case_id": r["case_id"], "label": label.get(r["case_id"]), "actions": " > ".join(acts),
                   "n_ask": acts.count("ask"), "n_exam": acts.count("body_exam"), "n_test": acts.count("test"),
                   "diagnosed": "diagnose" in acts, "dollars": r.get("dollars")})
    rep = []
    for tag, first in (("weak", weak), ("strong", strong)):
        by = {r["case_id"]: r for r in first}
        for r in load(os.path.join(a.run, f"episodes_{tag}_rep2.jsonl")):
            f = by.get(r["case_id"])
            if f:
                rep.append({"agent": tag, "case_id": r["case_id"], "label": label.get(r["case_id"]),
                            "terminal_run1": f["terminal"], "terminal_run2": r["terminal"], "same_terminal": f["terminal"] == r["terminal"],
                            "rounds_run1": f.get("rounds"), "rounds_run2": r.get("rounds"), "dollars_run1": f.get("dollars"), "dollars_run2": r.get("dollars")})
    summ = [{"item": "7B invalid-action episodes", "n": len(inv)}] + \
           [{"item": f"  class: {k}", "n": v} for k, v in collections.Counter(x["failure_class"] for x in inv).most_common()] + \
           [{"item": f"  state: {k}", "n": v} for k, v in collections.Counter(x["state_at_failure"] for x in inv).most_common()] + \
           [{"item": f"  label: {k}", "n": v} for k, v in collections.Counter(x["label"] for x in inv).most_common()] + \
           [{"item": "72B timeouts", "n": len(to)}] + \
           [{"item": f"repeats {t}: same terminal", "n": f"{sum(x['same_terminal'] for x in rep if x['agent']==t)} / {sum(x['agent']==t for x in rep)}"} for t in ("weak", "strong")]
    with pd.ExcelWriter(out) as xw:
        pd.DataFrame(summ).to_excel(xw, index=False, sheet_name="summary")
        pd.DataFrame(inv).to_excel(xw, index=False, sheet_name="invalid_7B")
        pd.DataFrame(to).to_excel(xw, index=False, sheet_name="timeouts_72B")
        pd.DataFrame(rep).to_excel(xw, index=False, sheet_name="repeats")
    for s in summ:
        print(f"{s['item']:45s} {s['n']}")
    print("wrote", out)


if __name__ == "__main__":
    main()
