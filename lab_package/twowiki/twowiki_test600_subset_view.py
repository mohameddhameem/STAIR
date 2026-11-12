#!/usr/bin/env python3
"""
twowiki_test600_subset_view.py -- SUBSET VIEW of the two test600 reads already made (GRPO fresh, 10 Nov; SFT (label), 11 Nov)
on the two halves of test600: AgentTTS's released 100 (dev[0:100], Row A) and dev[500:1000] (n=500, Row C) -- so our routers
can be set next to AgentTTS (reproduced) 70.00 / 70.40 on the SAME instances. Read-only, CPU, seconds. NO new test exposure:
the per-instance router scores were stored by the two declared reads; the points are the LOCKED ones already read (plus greedy,
as in the reads); nothing is chosen here. Same precedent as the HotpotQA random-500 view (11 Nov). v1.1 (12 Nov): all five declared reads (the three Nov-11/12 pairs added;
the warm-SFT pair at its eval-mode lock); writes test600_subset_view_v11.json. Nothing else changed.
Verdicts, weak-first prices and the routing rule are copied from twowiki_fresh_test_readout.py (reviewed, run).
MEMBERSHIP CHECK: test600 positions 0-99 must equal AgentTTS's tts_test.json questions in order (Row A); positions 100-599
are then dev[500:1000] (Row C) by the registered split (1 Nov 08:25). The script stops if the first check fails.
Usage (lab):  python3 -I twowiki_test600_subset_view.py
"""
import csv, json, math, os, importlib.util

_spec = importlib.util.spec_from_file_location("stair_cost", "/mnt/hdd/jonathan/STAIR/common/stair_common/cost.py")
_cost = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cost)
units = _cost.units

L   = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/test600"
TTS = "/mnt/hdd/jonathan/agenttts_verify/data/2wikimultihopqa/tts_test.json"          # AgentTTS's released 100
S5 = "/mnt/hdd/jonathan/runs/twowiki"
READS = {  # name: (store folder with ret/qa_scores.jsonl, LOCK file)   -- v1.1 12 Nov: all five declared reads
    "GRPO(Reward) fresh, locked":        (f"{S5}/grpo_2w_fresh_test600", f"{S5}/grpo_2w_a1_fresh_ep6_objective/LOCKED_thresholds.json"),
    "SFT (label), locked":               (f"{S5}/sft_2w_pair/test600", f"{S5}/sft_2w_pair/LOCKED_thresholds.json"),
    "SFT (label) + Reward, locked":      (f"{S5}/reward_2w_pair/test600", f"{S5}/reward_2w_pair/LOCKED_thresholds.json"),
    "SFT (label) + GRPO (Reward), locked": (f"{S5}/grpo_2w_warm_sft_ep3_standard/test600", f"{S5}/grpo_2w_warm_sft_ep3_standard/LOCKED_thresholds_eval.json"),
    "HotpotQA-warmed GRPO (Reward), locked": (f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective/test600", f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective/LOCKED_thresholds.json"),
}
OUT = "/mnt/hdd/jonathan/runs/twowiki/test600_subset_view_v11.json"
N = 600
P = {"retr": {"W": 7.0, "S": 72.0}, "qa": {"W": 3.0, "S": 70.0}}
CELLS = ["WW", "WS", "SW", "SS"]
if os.path.exists(OUT):
    raise SystemExit(f"REFUSING: {OUT} exists")


def store(name):
    d = {}
    for line in open(os.path.join(L, name + ".jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def u(kind, w, m):
    return units(P[kind][w], m["n_in"], m["n_out"])


# ---- membership check (Row A = positions 0-99) ----
data = json.load(open(os.path.join(L, "tts_test_2wiki_adapted.json"), encoding="utf-8"))
tts = json.load(open(TTS, encoding="utf-8"))
assert len(data) == N and len(tts) == 100
same = sum(" ".join(data[i]["question"].split()) == " ".join(tts[i]["question"].split()) for i in range(100))
print(f"MEMBERSHIP: test600[0:100] == AgentTTS released 100 (by question, in order): {same}/100")
if same != 100:
    raise SystemExit("STOP: Row A positions not confirmed")
SUBSETS = {"their 100 (Row A, dev[0:100])": range(0, 100), "dev[500:1000] (Row C, n=500)": range(100, 600), "test600": range(0, 600)}

# ---- verdicts and weak-first prices (as the reviewed readout) ----
ver = {}
for r in csv.DictReader(open(os.path.join(L, "scores_cells_per_instance.csv"))):
    ver.setdefault(int(r["idx"]), {})[r["cell"]] = int(float(r["lenient"]))
mr = {w: store(f"meta_retr_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
mq = {w: store(f"meta_qa_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
rows = []
for i in range(N):
    rW, rS = u("retr", "W", mr["W"][str(i)]), u("retr", "S", mr["S"][str(i)])
    q = {(a, b): u("qa", a, mq[a][f"{i}|{'weak' if b == 'W' else 'strong'}"]) for a in "WS" for b in "WS"}
    rows.append({"idx": i, "em": [ver[i][c] for c in CELLS],
                 "price": [rW + q["W", "W"], rW + q["W", "W"] + q["S", "W"], rW + rS + q["W", "S"], rW + rS + q["W", "S"] + q["S", "S"]],
                 "alone": [rW + q["W", "W"], rW + q["S", "W"], rS + q["W", "S"], rS + q["S", "S"]]})
assert all(len(ver[i]) == 4 for i in range(N))

out = {"version": "twowiki_test600_subset_view v1.1 2026-11-12", "membership_rowA": same}
for sname, idxs in SUBSETS.items():
    R = [rows[i] for i in idxs]; n = len(R)
    C_SS = sum(g["alone"][3] for g in R) / n
    ss = [g["em"][3] for g in R]
    orc = [max(range(4), key=lambda c: (g["em"][c], -g["alone"][c])) for g in R]
    print(f"\n{sname}: n={n} | always-SS {100*sum(ss)/n:.2f} at 100% (C_SS {C_SS:.1f} units) | "
          f"oracle {100*sum(g['em'][c] for g, c in zip(R, orc))/n:.2f} at {100*sum(g['alone'][c] for g, c in zip(R, orc))/n/C_SS:.1f}%")
    res = {"n": n, "always_SS": round(100 * sum(ss) / n, 2), "C_SS_units": round(C_SS, 2)}
    for rname, (sto, lockfile) in READS.items():
        ret = {r["idx"]: r["pW"] for r in map(json.loads, open(os.path.join(sto, "ret_scores.jsonl")))}
        qa = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(os.path.join(sto, "qa_scores.jsonl")))}
        lock = json.load(open(lockfile))                                                   # v1.1: lock file per read
        for pname, (tr, tw, ts) in (("locked", (lock["t_ret"], lock["t_qaW"], lock["t_qaS"])), ("greedy", (0.5, 0.5, 0.5))):
            got, cost = [], 0.0
            for g in R:
                i = g["idx"]
                if ret[i] >= tr: c = 0 if qa[(i, "W")] >= tw else 1
                else:            c = 2 if qa[(i, "S")] >= ts else 3
                got.append(g["em"][c]); cost += g["price"][c]
            d = [a - b for a, b in zip(got, ss)]
            md = sum(d) / n; se = math.sqrt(sum((x - md) ** 2 for x in d) / (n - 1) / n)
            acc, cp = 100 * sum(got) / n, 100 * cost / n / C_SS
            print(f"  {rname:40s} {pname:6s} t={tr:.2f}/{tw:.2f}/{ts:.2f}: {acc:.2f} at {cp:.1f}% | vs SS {100*md:+.2f} (SE {100*se:.2f})")
            res[f"{rname} | {pname}"] = {"acc": round(acc, 2), "cost_pct": round(cp, 1), "vs_SS": round(100 * md, 2), "se": round(100 * se, 2)}
    out[sname] = res
json.dump(out, open(OUT, "w"), indent=1)
print("\nwrote", OUT, "-- subset view of stored reads; no new exposure")
