# twowiki_pair_sweep.py -- threshold sweep for the 2Wiki fresh-adapter GRPO routers (read-only, CPU, seconds).
# COPY of the champion's hotpotqa/grpo_pair_sweep.py (sha 6de4b33c...) -- same 21^3 grid, same routing rule, same
# weak-first cost accounting, same two printed points (best accuracy / cheapest >= SS). Lines that differ are marked # 2W.
# DRAFT v1.0 2026-11-10 for Jonathan's line-by-line review. Never reads test600.
#
# What changes for 2Wiki, and why (# 2W):
#  * verdicts: the reviewed cells scorer's per-instance file for val400 (lenient = standard ruler, the same ruler as
#    HotpotQA's em_standard), instead of HotpotQA's router_data_standard.jsonl
#  * router scores: the 2Wiki run's ret/qa_val_scores.jsonl; the staged idx is 5000 + val400 position
#  * PRICE per instance, not four constants: HotpotQA's PRICE [80, 195, 894, 1009] was built from fixed per-call
#    costs; 2Wiki uses the measured tokens of every call (meta stores) in AgentTTS units, same weak-first rule
#    ("count everything that ran": the router's inputs need the weak retrieval and the weak answer on that branch):
#      WW = rW + qW|W      WS = rW + qW|W + qS|W      SW = rW + rS + qW|S      SS = rW + rS + qW|S + qS|S
#  * C_SS = mean standalone SS cost (rS + qS|S), the budget without the router -- as HotpotQA's 929
#  * added: the registered Rule B point (highest accuracy; within 1 SE of it = tied; cheapest tie wins) and the
#    router's own greedy point; the full grid saved to Excel; the Rule B point LOCKED to a file (refuses to overwrite)
import csv, json, math, os, sys, importlib.util
# 2W: load stair_common/cost.py directly -- the package __init__ imports tomllib, absent on the lab's Python 3.10
_spec = importlib.util.spec_from_file_location("stair_cost", "/mnt/hdd/jonathan/STAIR/common/stair_common/cost.py")
_cost = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cost)
units = _cost.units                                                                       # 2W

L    = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/val400"                          # 2W
RUN  = "/mnt/hdd/jonathan/runs/twowiki/grpo_2w_a1_fresh_ep6_objective"                    # 2W
OFF  = 5000                                                                               # 2W: staged idx of val400 position 0
P    = {"retr": {"W": 7.0, "S": 72.0}, "qa": {"W": 3.0, "S": 70.0}}                       # 2W: as twowiki_cells_score.py
CELLS = ["WW", "WS", "SW", "SS"]                                                          # c = 0, 1, 2, 3 as in PRICE


def store(name):                                                                          # 2W: = twowiki_cells_score.load_store
    d = {}                                                                                # (meta records carry n_in/n_out at top level, no "v")
    for line in open(os.path.join(L, name + ".jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def u(kind, w, m):                                                                        # 2W
    return units(P[kind][w], m["n_in"], m["n_out"])


# ---- per-instance verdicts (reviewed scorer output) and weak-first prices -------------------------------- # 2W
ver = {}
for r in csv.DictReader(open(os.path.join(L, "scores_cells_per_instance.csv"))):
    ver.setdefault(int(r["idx"]), {})[r["cell"]] = int(float(r["lenient"]))
mr = {w: store(f"meta_retr_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
mq = {w: store(f"meta_qa_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
val, ss_units = [], 0.0
for i in range(400):
    rW, rS = u("retr", "W", mr["W"][str(i)]), u("retr", "S", mr["S"][str(i)])
    q = {(a, b): u("qa", a, mq[a][f"{i}|{'weak' if b == 'W' else 'strong'}"]) for a in "WS" for b in "WS"}  # (answerer, branch)
    price = [rW + q["W", "W"], rW + q["W", "W"] + q["S", "W"], rW + rS + q["W", "S"], rW + rS + q["W", "S"] + q["S", "S"]]
    em = [ver[i][c] for c in CELLS]
    val.append({"idx": OFF + i, "em": em, "price": price})
    ss_units += rS + q["S", "S"]
assert len(val) == 400 and all(len(ver[i]) == 4 for i in range(400))
C_SS = ss_units / len(val)                                                                # 2W: per-question budget
ss_acc = sum(g["em"][3] for g in val) / len(val)
T = [i/20 for i in range(21)]
print(f"validation n={len(val)} | always-SS {100*ss_acc:.2f} at 100%  (C_SS = {C_SS:.1f} units/question)")

PAIRS = {"GRPO(Reward) fresh adapter, 2Wiki": (f"{RUN}/ret_val_scores.jsonl", f"{RUN}/qa_val_scores.jsonl")}  # 2W
for name, (rf, qf) in PAIRS.items():
    ret = {r["idx"]: r["pW"] for r in map(json.loads, open(rf))}
    qa  = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(qf))}
    rows = [(ret[g["idx"]], qa[(g["idx"],"W")], qa[(g["idx"],"S")], g["em"], g["price"]) for g in val]  # 2W: price per row
    n = len(rows); res = []
    for tr in T:
        for tw in T:
            for ts in T:
                acc = cost = 0
                for pr, pqw, pqs, em, price in rows:
                    if pr >= tr:  c = 0 if pqw >= tw else 1
                    else:         c = 2 if pqs >= ts else 3
                    acc += em[c]; cost += price[c]                                         # 2W: price[c] of this row
                res.append((acc/n, cost/n/C_SS, tr, tw, ts))
    best = max(res)
    beats = [r for r in res if r[0] >= ss_acc]
    print(f"\n{name}:")
    print(f"  best accuracy : {100*best[0]:.2f} at {100*best[1]:.1f}%  (t={best[2]:.2f}/{best[3]:.2f}/{best[4]:.2f})")
    if beats:
        cheap = min(beats, key=lambda r: r[1])
        print(f"  cheapest >= SS: {100*cheap[0]:.2f} at {100*cheap[1]:.1f}%  (t={cheap[2]:.2f}/{cheap[3]:.2f}/{cheap[4]:.2f})")
    else:
        print("  no swept point reaches always-SS accuracy")
    # ---- added for 2Wiki (# 2W): greedy point, Rule B, Excel, lock ---------------------------------------------------
    greedy = next(r for r in res if r[2:] == (0.50, 0.50, 0.50))
    print(f"  greedy 0.50/0.50/0.50: {100*greedy[0]:.2f} at {100*greedy[1]:.1f}%")
    se = math.sqrt(best[0] * (1 - best[0]) / n)
    tied = [r for r in res if r[0] >= best[0] - se]
    ruleB = min(tied, key=lambda r: (r[1], -r[0], r[2:]))
    print(f"  RULE B (ties = within 1 SE = {100*se:.2f} pts of best; cheapest tie): {100*ruleB[0]:.2f} at {100*ruleB[1]:.1f}%"
          f"  (t={ruleB[2]:.2f}/{ruleB[3]:.2f}/{ruleB[4]:.2f})  [{len(tied)} tied points]")
    try:
        import pandas as pd
        pd.DataFrame([{"t_ret": r[2], "t_qaW": r[3], "t_qaS": r[4], "val_lenient_pct": round(100*r[0], 2),
                       "cost_pct_of_SS": round(100*r[1], 1), "tied_with_best": r[0] >= best[0] - se} for r in res]
                     ).to_excel(os.path.join(RUN, "sweep_val400.xlsx"), index=False)
        print(f"  grid saved: {RUN}/sweep_val400.xlsx")
    except Exception as e:                                                                # no pandas -> CSV (opens in Excel)
        with open(os.path.join(RUN, "sweep_val400.csv"), "w", newline="") as f:
            w = csv.writer(f); w.writerow(["t_ret", "t_qaW", "t_qaS", "val_lenient_pct", "cost_pct_of_SS", "tied_with_best"])
            for r in res:
                w.writerow([r[2], r[3], r[4], round(100*r[0], 2), round(100*r[1], 1), r[0] >= best[0] - se])
        print(f"  (Excel not written: {e}) -> grid saved as CSV: {RUN}/sweep_val400.csv")
    lock = os.path.join(RUN, "LOCKED_thresholds.json")
    if os.path.exists(lock):
        print(f"  LOCK exists, not overwritten: {lock}")
    else:
        json.dump({"version": "twowiki_pair_sweep v1.0 2026-11-10", "rule": "Rule B (1 SE ties, cheapest)",
                   "t_ret": ruleB[2], "t_qaW": ruleB[3], "t_qaS": ruleB[4],
                   "val_lenient": round(ruleB[0], 4), "val_cost_pct_of_SS": round(100*ruleB[1], 1),
                   "greedy": {"val_lenient": round(greedy[0], 4), "val_cost_pct_of_SS": round(100*greedy[1], 1)},
                   "always_SS_val_lenient": round(ss_acc, 4), "C_SS_units": round(C_SS, 2), "se": round(se, 4)},
                  open(lock, "w"), indent=1)
        print(f"  LOCKED: {lock}")
