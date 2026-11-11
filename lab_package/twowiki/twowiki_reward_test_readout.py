#!/usr/bin/env python3
"""
REWARD COPY (11 Nov 2026) of twowiki_sft_test_readout.py: only RUN/version differ (# RW). Lock = Rule B on val400 for the
SFT (label) + Reward pair (to be locked by `twowiki_pair_sweep.py reward`). Same two declared points: LOCKED and greedy. Same claim test (within 3 pts).
twowiki_fresh_test_readout.py -- the ONE held-out test600 readout of the 2Wiki fresh-adapter GRPO(Reward) router
(read-only, CPU, seconds). DRAFT v1.0 2026-11-10 for Jonathan's line-by-line review.

Thresholds are read from LOCKED_thresholds.json (Rule B on val400, locked 10 Nov 11:48) -- no tuning here. Reports two
points declared before the read: the LOCKED point and the router's own greedy point (0.50/0.50/0.50). Routing rule,
verdicts (reviewed scorer's per-instance lenient, standard ruler) and weak-first per-instance prices: identical to
twowiki_pair_sweep.py (the code below is that file's, re-pointed at test600). References from the same file: always-SS,
always-WW, oracle. Paired difference vs always-SS with its standard error (n = 600). One declared read: writes
readout.json and refuses to run again.
Usage (lab):  python3 twowiki_fresh_test_readout.py
"""
import csv, json, math, os, importlib.util

_spec = importlib.util.spec_from_file_location("stair_cost", "/mnt/hdd/jonathan/STAIR/common/stair_common/cost.py")
_cost = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cost)
units = _cost.units

L    = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/test600"
RUN  = "/mnt/hdd/jonathan/runs/twowiki/reward_2w_pair"                   # RW: lock lives in the pair folder
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
lock = json.load(open(os.path.join(RUN, "LOCKED_thresholds.json")))
POINTS = [("LOCKED (Rule B on val400)", lock["t_ret"], lock["t_qaW"], lock["t_qaS"]), ("greedy", 0.50, 0.50, 0.50)]

ss = [g["em"][3] for g in rows]
out = {"version": "twowiki_reward_test_readout v1.0 2026-11-11", "n": N, "C_SS_units": round(C_SS, 2),
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
