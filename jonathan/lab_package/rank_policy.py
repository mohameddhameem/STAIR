#!/usr/bin/env python3
# =============================================================================
# rank_policy.py — VERSION: rank_policy v1.0 2026-09-07
#
# CALIBRATION SIDESTEP for any scores file (zero-shot or trained).
# The 6 Sep sweep showed the zero-shot scores RANK well (AUC 0.620) but no
# fixed probability cutoff works (miscalibrated thermometer). This policy sets
# the cutoff from the scores' own distribution instead of their face value:
#
#   "deviate to the cheap choice on the X% of instances the router is MOST
#    confident about (by rank); SS for the rest"
#
# Implementation: for a deviation budget X, τ is the (1−X) quantile of the
# VALIDATION scores — a data-driven τ. Test instances are judged against that
# frozen number. Sweep X on val, freeze the chosen (X_ret, X_qa) under the
# same three rules as Run 1c, report test. Pure CPU, seconds.
#
# USAGE (lab):
#   python3 rank_policy.py --data .../router_data.jsonl --splits splits.json \
#       --scores /mnt/hdd/jonathan/runs/zs_evidence/router_scores.jsonl \
#       --out /mnt/hdd/jonathan/runs/zs_evidence_rank
# =============================================================================
import os, json, argparse

VERSION = "rank_policy v1.0 2026-09-07"
from threshold_policy import eval_policy, baselines   # same protocol, imported


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--scores", required=True, help="router_scores.jsonl (any source)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--budgets", default="0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="deviation budgets X to sweep (fraction of instances allowed to go cheap)")
    return ap.parse_args()


def quantile(sorted_vals, q):
    """value at quantile q of a pre-sorted list (nearest-rank, q in [0,1])."""
    if not sorted_vals:
        return 1.01
    k = min(len(sorted_vals) - 1, max(0, int(q * len(sorted_vals))))
    return sorted_vals[k]


def main():
    a = build_args()
    os.makedirs(a.out, exist_ok=True)
    print(VERSION, flush=True)
    rows = {}
    for l in open(a.data):
        r = json.loads(l); rows[r["idx"]] = r
    sp = json.load(open(a.splits))["splits"]
    va = [i for i in sp["val"] if i in rows]
    te = [i for i in sp["test"] if i in rows]
    scores = {}
    for l in open(a.scores):
        r = json.loads(l); scores[r["idx"]] = r
    assert all(i in scores for i in va + te), "scores file must cover val+test"

    base_va, base_te = baselines(rows, va), baselines(rows, te)
    ss_val = base_va["always_SS"]["lenient"]
    budgets = [float(x) for x in a.budgets.split(",")]

    # τ from VAL quantiles: budget X -> τ = value below which (1-X) of val scores lie,
    # i.e. only the top-X most weak-confident instances clear the bar.
    ret_sorted = sorted(scores[i]["p_ret_W"] for i in va)
    qa_sorted = sorted([scores[i]["p_qaW_W"] for i in va] + [scores[i]["p_qaS_W"] for i in va])

    def tau_of(budget, vals):
        if budget <= 0:  return 1.01           # no deviation ever
        if budget >= 1:  return 0.0            # always cheap
        return quantile(vals, 1.0 - budget)

    grid = {}
    for bx in budgets:
        for by in budgets:
            t = (tau_of(bx, ret_sorted), tau_of(by, qa_sorted))
            grid[(bx, by)] = (t, eval_policy(rows, scores, va, *t))
    parity = max((g for g in grid if grid[g][1][0] >= ss_val),
                 key=lambda g: -grid[g][1][2], default=None)
    best_acc = max(grid, key=lambda g: (grid[g][1][0], -grid[g][1][2]))
    knee = max(grid, key=lambda g: grid[g][1][0] - 0.1 * grid[g][1][2])

    result = {"version": VERSION, "scores": a.scores, "baselines_test": base_te, "chosen": {}}
    for name, g in (("parity", parity), ("best_acc", best_acc), ("knee", knee)):
        if g is None:
            result["chosen"][name] = None
            print(f"[{name}] no budget reaches SS accuracy on val", flush=True); continue
        t, (lv, sv, cv) = grid[g]
        lt, st, ct = eval_policy(rows, scores, te, *t)
        result["chosen"][name] = {"budget_ret": g[0], "budget_qa": g[1],
                                  "tau_ret": round(t[0], 4), "tau_qa": round(t[1], 4),
                                  "val": {"lenient": lv, "cost_pct": cv},
                                  "test": {"lenient": lt, "strict": st, "cost_pct": ct}}
        print(f"[{name}] budgets ({g[0]}, {g[1]}) -> τ ({t[0]:.3f}, {t[1]:.3f}) | "
              f"val {lv} @ {cv}% | TEST {lt} lenient @ {ct}% of SS cost", flush=True)
    print(f"[baselines TEST] SS {base_te['always_SS']['lenient']} @ 100% | "
          f"WW {base_te['always_WW']['lenient']} @ {base_te['always_WW']['cost_pct_of_SS']}%", flush=True)
    json.dump(result, open(os.path.join(a.out, "rank_policy_result.json"), "w"), indent=2)
    print(f"saved -> {a.out}/rank_policy_result.json", flush=True)


if __name__ == "__main__":
    main()
