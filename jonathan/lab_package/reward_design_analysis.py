# reward_design_analysis.py -- choose the reward function BY MEASUREMENT,
# before training anything against it (the coverage-analysis playbook).
# CPU-only, read-only. Runs fine while a training job holds the GPU.
#
# E1  R-oracle sweep: for each candidate shape of R, what would PERFECTLY
#     following argmax-R per instance achieve (accuracy, cost, mean R)?
#     Shapes: hinge (the registered R), hinge+linear, pure linear.
# E2  Stage-stakes decomposition: per decision and stage, the size of
#     |delta R| -- where does each R shape actually have leverage?
# E3/E4  Reward-frontier alignment: mean R for every row of the existing
#     swept frontier (Wednesday's, from the class-weighted routers) and the
#     row argmax-R would SELECT -- compared with the max-accuracy row.
#     Exact for linear shapes (R is linear in cost); for hinge shapes the
#     frontier aggregate only bounds mean R, so E3/E4 reports linear shapes.
import json, os, csv

GRID = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
COSTS = "/mnt/hdd/jonathan/runs/token_costs/token_costs.json"
FRONTIER = "/mnt/hdd/jonathan/runs/gate_eval/frontier.csv"   # Wednesday's sweep

tc = json.load(open(COSTS))["gate_eval_constants"]
C = {  # path cost per cell, measured family (same as feasibility script)
  "WW": tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"],
  "WS": tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"] + tc["COST_QA_S_GIVEN_W"],
  "SW": tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"],
  "SS": tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"] + tc["COST_QA_S_GIVEN_S"]}
CELLS = ["WW", "WS", "SW", "SS"]
C_SS = C["SS"]

# candidate shapes: (name, alpha or None, lambda)  R = em - hinge(alpha) - lam*C/C_SS
SHAPES = ([("hinge a=%.2f" % a, a, 0.0) for a in (0.6, 0.7, 0.74, 0.8, 0.9, 1.0)]
        + [("hinge a=0.74 +lin l=%.2f" % l, 0.74, l) for l in (0.1, 0.25, 0.5)]
        + [("linear l=%.2f" % l, None, l) for l in (0.1, 0.25, 0.5, 1.0)])

def R(em, c, a, lam):
    pen = max(0.0, c / (a * C_SS) - 1.0) if a is not None else 0.0
    return em - pen - lam * c / C_SS

rows = [json.loads(l) for l in open(GRID)]
val = [r for r in rows if r["split"] == "val"]
print(f"val instances: {len(val)} | path costs: " +
      " ".join(f"{k}={v:.1f}" for k, v in C.items()) + f"\n")

# ---------------- E1: R-oracle per shape ----------------
print("E1  R-ORACLE (follow argmax-R perfectly, validation split)")
print(f"{'shape':26s} {'acc':>6s} {'cost':>7s} {'%ofSS':>6s} {'meanR':>7s}  cell usage")
for name, a, lam in SHAPES:
    acc = cost = mr = 0.0
    use = {c: 0 for c in CELLS}
    for r in val:
        em = dict(zip(CELLS, r["em_lenient"]))
        best = max(CELLS, key=lambda c: (R(em[c], C[c], a, lam), -C[c]))
        acc += em[best]; cost += C[best]; mr += R(em[best], C[best], a, lam)
        use[best] += 1
    n = len(val)
    print(f"{name:26s} {100*acc/n:6.2f} {cost/n:7.1f} {100*(cost/n)/C_SS:5.1f}% "
          f"{mr/n:7.3f}  " + " ".join(f"{c}:{use[c]}" for c in CELLS))
print("  reference: SS accuracy 52.79 at 250.1 (100%) | accuracy-oracle 60.63 at 27.7% (prior measurement)\n")

# ---------------- E2: stage stakes per shape ----------------
print("E2  STAGE STAKES  mean |delta R| per decision (validation)")
print(f"{'shape':26s} {'RET stage':>10s} {'QA|retW':>9s} {'QA|retS':>9s}")
for name, a, lam in SHAPES:
    sret = sqw = sqs = 0.0
    for r in val:
        em = dict(zip(CELLS, r["em_lenient"]))
        # retrieval decision: compare each branch at its own best answering choice
        bw = max(("WW", "WS"), key=lambda c: R(em[c], C[c], a, lam))
        bs = max(("SW", "SS"), key=lambda c: R(em[c], C[c], a, lam))
        sret += abs(R(em[bs], C[bs], a, lam) - R(em[bw], C[bw], a, lam))
        sqw += abs(R(em["WS"], C["WS"], a, lam) - R(em["WW"], C["WW"], a, lam))
        sqs += abs(R(em["SS"], C["SS"], a, lam) - R(em["SW"], C["SW"], a, lam))
    n = len(val)
    print(f"{name:26s} {sret/n:10.3f} {sqw/n:9.3f} {sqs/n:9.3f}")
print("  (bigger = that decision matters more under this R)\n")

# ---------------- E3/E4: reward vs the existing frontier ----------------
print("E3/E4  MEAN R PER FRONTIER ROW (linear shapes are exact from aggregates)")
front = list(csv.DictReader(open(FRONTIER)))
print(f"frontier rows: {len(front)} (Wednesday's sweep, class-weighted routers)")
for name, a, lam in [s for s in SHAPES if s[1] is None]:
    best, bval = None, -1e9
    for f in front:
        meanR = float(f["val_accuracy"]) - lam * float(f["cost"]) / C_SS
        if meanR > bval: bval, best = meanR, f
    print(f"{name:26s} argmax-R row -> acc={float(best['val_accuracy'])*100 if float(best['val_accuracy'])<1.5 else float(best['val_accuracy']):.2f} "
          f"cost={float(best['cost']):.1f} ({100*float(best['cost'])/C_SS:.1f}% of SS) "
          f"theta_ret={float(best['theta_ret']):.3f} theta_qa={float(best['theta_qa']):.3f}")
print("  compare with: max-accuracy row 53.87 @ 184.8 | SS-parity row 52.79 @ 161.2")
