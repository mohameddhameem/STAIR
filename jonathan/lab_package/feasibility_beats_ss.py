# feasibility_beats_ss.py -- step 0 of the beat-SS campaign (1.5B/7B pair).
# CPU-only, read-only. Measures, from the existing grid cache, the raw
# material any router has for BEATING always-strong (SS) on accuracy at
# lower cost:
#
#   PRIZE  = instances where SS FAILS but some cheaper cell (WW/WS/SW)
#            succeeds -- the only source of accuracy above SS.
#   DANGER = instances where SS succeeds and every cheaper cell fails --
#            what a router loses when it wrongly stays cheap.
#
#   A router beats SS by delta only if (captured prize - leaked danger)
#   >= delta. Oracle = SS + full prize (perfect capture, zero leak).
#
# Costs: the measured token-aware family (13), read from token_costs.json.
import json, os
import numpy as np

GRID = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
TOKEN_COSTS = "/mnt/hdd/jonathan/runs/token_costs/token_costs.json"

tc = json.load(open(TOKEN_COSTS))["gate_eval_constants"]
# per-cell cost: retrieval (weak always; strong adds) + answering priced by branch
CELL_COST = {
    "WW": tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"],
    "WS": tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"] + tc["COST_QA_S_GIVEN_W"],
    "SW": tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"],
    "SS": tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"] + tc["COST_QA_S_GIVEN_S"],
}
CELLS = ["WW", "WS", "SW", "SS"]          # = em_lenient column order
CHEAPER = ["WW", "WS", "SW"]              # all cost less than SS

rows = [json.loads(l) for l in open(GRID)]
print(f"grid: {len(rows)} instances | cell costs: " +
      " ".join(f"{c}={CELL_COST[c]:.1f}" for c in CELLS))

for split in ("train", "val", "test"):
    sub = [r for r in rows if r["split"] == split]
    em = np.array([r["em_lenient"] for r in sub])          # [n, 4]
    n = len(sub)
    ss = em[:, 3].astype(bool)
    cheap_any = em[:, :3].max(axis=1).astype(bool)
    prize = (~ss) & cheap_any
    danger = ss & ~cheap_any
    both = ss & cheap_any
    neither = ~ss & ~cheap_any
    acc_ss = 100 * ss.mean()
    oracle = 100 * (ss | cheap_any).mean()
    print(f"\n==== split={split} (n={n}) ====")
    print(f"SS accuracy {acc_ss:.2f} | oracle {oracle:.2f} | "
          f"headroom above SS = prize = {100*prize.mean():.2f} points")
    print(f"PRIZE  (SS wrong, cheaper cell right): {prize.sum():4d} ({100*prize.mean():5.2f}%)")
    print(f"DANGER (SS right, all cheaper wrong):  {danger.sum():4d} ({100*danger.mean():5.2f}%)")
    print(f"both succeed (free savings, no risk):  {both.sum():4d} ({100*both.mean():5.2f}%)")
    print(f"neither (unwinnable):                  {neither.sum():4d} ({100*neither.mean():5.2f}%)")
    # which cheaper cell carries the prize (an instance may have several)
    for ci, c in enumerate(CHEAPER):
        share = 100 * (prize & em[:, ci].astype(bool)).mean()
        print(f"  prize reachable via {c}: {share:5.2f}% of instances")
    # breakeven: to beat SS by delta points, the router's captured prize
    # minus its leaked danger must be >= delta.
    pz = 100 * prize.mean()
    print("  margin room by capture rate (max leakable danger points):")
    print(f"    {'capture':>8s} {'delta=0.5':>10s} {'delta=1.0':>10s}")
    for cap in (0.3, 0.4, 0.5, 0.6, 0.8, 1.0):
        print(f"    {int(cap*100):7d}% {cap*pz-0.5:10.2f} {cap*pz-1.0:10.2f}")
