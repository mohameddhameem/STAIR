# bootstrap_frontiers.py -- are the morning's frontier differences real?
# Paired bootstrap (B=1000) over the 1,110 validation instances, on FIXED
# operating points taken from each pair's already-swept frontier:
#   parity = cheapest row with accuracy >= strong-strong
#   margin = cheapest row with accuracy >= strong-strong + 0.45
#   peak   = the max-accuracy row
# CPU-only, read-only. Verifies each policy reproduces its frontier row first.
import json, csv
import numpy as np

B, SEED = 1000, 20261012
GRID = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
COSTS = "/mnt/hdd/jonathan/runs/token_costs/token_costs.json"
R = "/mnt/hdd/jonathan/runs"
PAIRS = {  # name -> (ret scores, qa scores, frontier csv)
 "baseline":      (f"{R}/sft_router5_ret/val_scores.jsonl", f"{R}/sft_router5_qa/val_scores.jsonl",  f"{R}/gate_eval/frontier.csv"),
 "reward answering":(f"{R}/sft_router5_ret/val_scores.jsonl", f"{R}/reward_train_qa4/val_scores.jsonl", f"{R}/gate_eval_rw_qa4/frontier.csv"),
 "reward retrieval":(f"{R}/reward_train_ret/val_scores.jsonl", f"{R}/sft_router5_qa/val_scores.jsonl",  f"{R}/gate_eval_rw_ret/frontier.csv"),
 "both reward":    (f"{R}/reward_train_ret/val_scores.jsonl", f"{R}/reward_train_qa4/val_scores.jsonl", f"{R}/gate_eval_rw_both/frontier.csv"),
 "both reward v2":  (f"{R}/reward_train_ret2/val_scores.jsonl", f"{R}/reward_train_qa4/val_scores.jsonl", f"{R}/gate_eval_rw_both2/frontier.csv")}

tc = json.load(open(COSTS))["gate_eval_constants"]
RW, RS = tc["COST_RET_W"], tc["COST_RET_S"]
QWW, QWS = tc["COST_QA_W_GIVEN_W"], tc["COST_QA_W_GIVEN_S"]
QSW, QSS = tc["COST_QA_S_GIVEN_W"], tc["COST_QA_S_GIVEN_S"]
C_SS = RW + RS + QWS + QSS

grid = {g["idx"]: g for g in map(json.loads, open(GRID)) if g["split"] == "val"}
idxs = sorted(grid)
em = np.array([grid[i]["em_lenient"] for i in idxs])    # WW, WS, SW, SS
n = len(idxs); ss_acc = em[:, 3].mean()

def policy_vectors(retf, qaf, tR, tQ):
    ret = {r["idx"]: r["pW"] for r in map(json.loads, open(retf))}
    qa = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(qaf))}
    pR = np.array([ret[i] for i in idxs])
    pQW = np.array([qa[(i, "W")] for i in idxs]); pQS = np.array([qa[(i, "S")] for i in idxs])
    escR = pR < tR
    qp = np.where(escR, pQS, pQW)
    dW = qp >= tQ                                        # deliver the weak answer
    acc = np.where(escR, np.where(dW, em[:, 2], em[:, 3]),
                         np.where(dW, em[:, 0], em[:, 1]))
    cost = (RW + escR*RS + np.where(escR, QWS, QWW)
            + (~dW) * np.where(escR, QSS, QSW))
    return acc.astype(float), cost

def pick_rows(fcsv):
    rows = []
    for r in csv.DictReader(open(fcsv)):
        a = float(r["val_accuracy"]); a = a if a < 1.5 else a/100
        rows.append((float(r["cost"]), a, float(r["theta_ret"]), float(r["theta_qa"])))
    par = next(r for r in rows if r[1] >= ss_acc - 1e-9)
    mar = next((r for r in rows if r[1] >= ss_acc + 0.0045 - 1e-9), None)
    peak = max(rows, key=lambda r: (r[1], -r[0]))
    return {"parity": par, "margin": mar, "peak": peak}

rng = np.random.default_rng(SEED)
IDX = rng.integers(0, n, (B, n))
SSA = em[:, 3][IDX].mean(1)                              # SS accuracy per resample

res = {}
print(f"val n={n}  SS={100*ss_acc:.2f}  B={B}  seed={SEED}")
print(f"{'pair / point':34s} {'full acc':>8s} {'full %SS':>9s} {'acc 95% CI':>15s} {'%SS 95% CI':>14s} {'P(acc>=SS)':>10s}")
for name, (retf, qaf, fcsv) in PAIRS.items():
    for pt, row in pick_rows(fcsv).items():
        if row is None: continue
        cost0, acc0, tR, tQ = row
        a, c = policy_vectors(retf, qaf, tR, tQ)
        ok = abs(a.mean() - acc0) < 5e-4 and abs(c.mean() - cost0) < 0.05
        A = a[IDX].mean(1); C = 100 * c[IDX].mean(1) / C_SS
        res[(name, pt)] = (A, C)
        lo, hi = np.percentile(A, [2.5, 97.5]); cl, ch = np.percentile(C, [2.5, 97.5])
        print(f"{name+' / '+pt:34s} {100*a.mean():8.2f} {100*c.mean()/C_SS:8.1f}% "
              f"[{100*lo:5.2f},{100*hi:5.2f}] [{cl:5.1f},{ch:5.1f}] {np.mean(A >= SSA):10.3f}"
              + ("" if ok else "  REPRO-MISMATCH"))

print("\nHEAD-TO-HEAD (share of resamples won):")
def duel(tag, k1, k2, what):
    A1, C1 = res[k1]; A2, C2 = res[k2]
    x, y = (A1, A2) if what == "acc" else (C2, C1)       # cost: lower wins
    print(f"  {tag:62s} win {np.mean(x > y):.3f}  tie {np.mean(x == y):.3f}")
duel("peak accuracy: reward answering vs baseline", ("reward answering","peak"), ("baseline","peak"), "acc")
duel("peak cost: reward answering cheaper than baseline", ("reward answering","peak"), ("baseline","peak"), "cost")
duel("parity cost: both reward cheaper than baseline", ("both reward","parity"), ("baseline","parity"), "cost")
duel("margin cost: both reward cheaper than baseline", ("both reward","margin"), ("baseline","margin"), "cost")
duel("margin cost: reward retrieval cheaper than baseline", ("reward retrieval","margin"), ("baseline","margin"), "cost")
duel("parity cost: stabilized both-reward cheaper than baseline", ("both reward v2","parity"), ("baseline","parity"), "cost")
duel("parity cost: stabilized cheaper than yesterday's both-reward", ("both reward v2","parity"), ("both reward","parity"), "cost")
duel("margin cost: stabilized both-reward cheaper than baseline", ("both reward v2","margin"), ("baseline","margin"), "cost")
