#!/usr/bin/env python3
"""
twowiki_pair_test_readout.py -- ONE readout for the three pending reads (12 Nov 2026; DRAFT for review) = twowiki_sft_test_readout.py
(reviewed, run 11 Nov) with --pair selecting the lock/test600 folder and a THIRD declared point per pair (the ties-SS point
recorded in the registration before the read; reported, labelled, never used for the claim). Lines marked # PAIR differ.
Claim is tested at LOCKED only: non-inferior to always-SS if the lower 95 % bound of the paired difference (n = 600) > -3.
"""

import csv, json, math, os, importlib.util

_spec = importlib.util.spec_from_file_location("stair_cost", "/mnt/hdd/jonathan/STAIR/common/stair_common/cost.py")
_cost = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cost)
units = _cost.units

L    = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/test600"
import argparse                                                           # PAIR
_ap = argparse.ArgumentParser(); _ap.add_argument("--pair", choices=["reward", "warm-sft", "warm-hotpotqa"], required=True)
_ap.add_argument("--eval-scores", action="store_true")
_args = _ap.parse_args(); PAIRCHOICE, EVALS = _args.pair, _args.eval_scores
S5 = "/mnt/hdd/jonathan/runs/twowiki"
RUN = {"reward": f"{S5}/reward_2w_pair", "warm-sft": f"{S5}/grpo_2w_warm_sft_ep3_standard",
       "warm-hotpotqa": f"{S5}/grpo_2w_warm_hotpotqa_ep3_objective"}[PAIRCHOICE]           # PAIR: lock + test600 live here
# PAIR: the labelled SECONDARY point per pair, declared in the registration BEFORE the read (11 Nov 18:26 / 19:55 / 12 Nov 09:38):
SECONDARY = {"reward": (0.90, 1.00, 0.60), "warm-sft": (0.50, 0.90, 0.75), "warm-hotpotqa": (0.25, 0.10, 0.50)}[PAIRCHOICE]
if EVALS and PAIRCHOICE == "warm-sft":                                     # 12 Nov 09:58: ties-SS of the EVAL-mode sweep (registered before the read)
    SECONDARY = (0.55, 0.95, 0.50)
STO  = os.path.join(RUN, "test600")                                      # SFT: twowiki_sft_test_store.py output
OUT  = os.path.join(STO, "readout.json")
N    = 600
P    = {"retr": {"W": 7.0, "S": 72.0}, "qa": {"W": 3.0, "S": 70.0}}
CELLS = ["WW", "WS", "SW", "SS"]

if os.path.exists(OUT):
    raise SystemExit(f"REFUSING: {OUT} exists -- the test600 readout has already been read")


def store(name):
    d = {}
    for line in open(os.path.join(L, name + ".jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def u(kind, w, m):
    return units(P[kind][w], m["n_in"], m["n_out"])


ver = {}
for r in csv.DictReader(open(os.path.join(L, "scores_cells_per_instance.csv"))):
    ver.setdefault(int(r["idx"]), {})[r["cell"]] = int(float(r["lenient"]))
mr = {w: store(f"meta_retr_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
mq = {w: store(f"meta_qa_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
rows, ss_units, alone = [], 0.0, []
for i in range(N):
    rW, rS = u("retr", "W", mr["W"][str(i)]), u("retr", "S", mr["S"][str(i)])
    q = {(a, b): u("qa", a, mq[a][f"{i}|{'weak' if b == 'W' else 'strong'}"]) for a in "WS" for b in "WS"}
    price = [rW + q["W", "W"], rW + q["W", "W"] + q["S", "W"], rW + rS + q["W", "S"], rW + rS + q["W", "S"] + q["S", "S"]]
    standalone = [rW + q["W", "W"], rW + q["S", "W"], rS + q["W", "S"], rS + q["S", "S"]]   # for the oracle reference
    rows.append({"idx": i, "em": [ver[i][c] for c in CELLS], "price": price, "alone": standalone})
    ss_units += standalone[3]
assert len(rows) == N and all(len(ver[i]) == 4 for i in range(N))
C_SS = ss_units / N

ret = {r["idx"]: r["pW"] for r in map(json.loads, open(os.path.join(STO, "ret_scores.jsonl")))}
qa = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(os.path.join(STO, "qa_scores.jsonl")))}
assert len(ret) == N and len(qa) == 2 * N
lock = json.load(open(os.path.join(RUN, "LOCKED_thresholds_eval.json" if EVALS else "LOCKED_thresholds.json")))   # 12 Nov
POINTS = [("LOCKED (Rule B on val400)", lock["t_ret"], lock["t_qaW"], lock["t_qaS"]), ("greedy", 0.50, 0.50, 0.50),
          ("SECONDARY ties-SS on val (labelled; claim NOT tested here)", *SECONDARY)]                       # PAIR

ss = [g["em"][3] for g in rows]
out = {"version": "twowiki_pair_test_readout v1.0 2026-11-12", "pair": PAIRCHOICE, "n": N, "C_SS_units": round(C_SS, 2),
       "always_SS": round(100 * sum(ss) / N, 2), "always_WW": round(100 * sum(g["em"][0] for g in rows) / N, 2),
       "always_WW_cost_pct": round(100 * sum(g["alone"][0] for g in rows) / N / C_SS, 1)}
orc = [max(range(4), key=lambda c: (g["em"][c], -g["alone"][c])) for g in rows]
out["oracle"] = round(100 * sum(g["em"][c] for g, c in zip(rows, orc)) / N, 2)
out["oracle_cost_pct_standalone"] = round(100 * sum(g["alone"][c] for g, c in zip(rows, orc)) / N / C_SS, 1)
print(f"test600 n={N} | always-SS {out['always_SS']:.2f} at 100% | always-WW {out['always_WW']:.2f} at "
      f"{out['always_WW_cost_pct']}% | oracle {out['oracle']:.2f} at {out['oracle_cost_pct_standalone']}% (standalone cells)")
out["points"] = {}
for name, tr, tw, ts in POINTS:
    got, cost, cells = [], 0.0, [0, 0, 0, 0]
    for g in rows:
        i = g["idx"]
        if ret[i] >= tr: c = 0 if qa[(i, "W")] >= tw else 1
        else:            c = 2 if qa[(i, "S")] >= ts else 3
        got.append(g["em"][c]); cost += g["price"][c]; cells[c] += 1
    d = [a - b for a, b in zip(got, ss)]
    md = sum(d) / N; se = math.sqrt(sum((x - md) ** 2 for x in d) / (N - 1) / N)
    p = {"t": [tr, tw, ts], "lenient": round(100 * sum(got) / N, 2), "cost_pct_of_SS_weak_first": round(100 * cost / N / C_SS, 1),
         "diff_vs_SS_pts": round(100 * md, 2), "diff_se_pts": round(100 * se, 2),
         "noninferior_3pt_margin": (100 * (md - 1.96 * se)) > -3.0, "cell_counts": dict(zip(CELLS, cells))}
    out["points"][name] = p
    print(f"  {name:28s} t={tr:.2f}/{tw:.2f}/{ts:.2f}: {p['lenient']:.2f} at {p['cost_pct_of_SS_weak_first']}% | "
          f"vs SS {p['diff_vs_SS_pts']:+.2f} (SE {p['diff_se_pts']:.2f}) | non-inferior (3 pts, 95%): "
          f"{p['noninferior_3pt_margin']} | cells {p['cell_counts']}")
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT, "-- one declared read")
