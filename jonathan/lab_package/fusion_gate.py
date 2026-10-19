#!/usr/bin/env python3
# fusion_gate.py v1.0 — val-fitted logistic fusion of the four verifier signals (23 Sep 2026)
# Signals per instance (all already on disk, zero GPU):
#   f1 = p_supported (greedy weak answer vs evidence)      verifier_scores.jsonl
#   f2 = bok_score   (best of 4 sampled answers)           bok_summary.jsonl
#   f3 = p_weak_better (greedy weak vs strong, ranked)     verifier_compare.jsonl
#   f4 = p_weak_better (best-of-4 vs strong, ranked)       verifier_compare_bok.jsonl
# DESIGN:
#   keep-decision: logistic on [logit(f1),logit(f2),logit(f3),logit(f4)] fit on VAL,
#     target = "keeping is at least as good as escalating" (em of the best-of-4 pick
#     >= em of SS) — kept instances use the best-of-4 answer (bok_em), matching the
#     53.41 champion's keep tier.
#   escalated instances: SS outcome, with the champion's comparative rescue kept as-is
#     (p_weak_better(greedy) >= 0.999999 -> keep SW outcome), so fusion replaces ONLY
#     the keep gate and the comparison stays fixed at its proven setting.
#   PROTOCOL: fit + threshold sweep on VAL only; sealed TEST read once at the chosen
#     threshold. 5 fitted parameters on 1,110 val rows. Honest label: "val-fitted".
# Cost model: keep 5.7, escalate 9.4 (as the champion; sampling & checker uncounted, noted).
import json, math, argparse

def jl(p): return [json.loads(l) for l in open(p)]
def lg(x): x = min(max(x, 1e-6), 1 - 1e-6); return math.log(x / (1 - x))

ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
ap.add_argument("--f1", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl")
ap.add_argument("--f2", default="/mnt/hdd/jonathan/runs/verifier_scores/bok_summary.jsonl")
ap.add_argument("--f3", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_compare.jsonl")
ap.add_argument("--f4", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_compare_bok.jsonl")
ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
ap.add_argument("--t3", type=float, default=0.999999, help="comparative rescue threshold (champion's val-selected value, held fixed)")
a = ap.parse_args()

grid = jl(a.grid)
F1 = {r["idx"]: r["p_supported"] for r in jl(a.f1)}
BOK = {r["idx"]: (r["bok_score"], r["bok_em"]) for r in jl(a.f2)}
F3 = {r["idx"]: r["p_weak_better"] for r in jl(a.f3)}
F4 = {r["idx"]: r["p_weak_better"] for r in jl(a.f4)}
sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
SW, SS = 2, 3

def rows(ids):
    out = []
    for i in ids:
        if i in F1 and i in BOK and i in F3 and i in F4:
            e = grid[i]["em_lenient"]
            x = [lg(F1[i]), lg(BOK[i][0]), lg(F3[i]), lg(F4[i])]
            out.append((x, BOK[i][1], e[SW], e[SS], F3[i]))
    return out

V, T = rows(sp["val"]), rows(sp["test"])
print(f"[coverage] val {len(V)}  test {len(T)}")

# ---- fit logistic on val: target = keep-is-safe (bok_em >= em_SS) ----
w = [0.0]*4; b = 0.0; lr = 0.05
for ep in range(300):
    for x, bok_em, _sw, e_ss, _f3 in V:
        y = 1.0 if bok_em >= e_ss else 0.0
        z = b + sum(wi*xi for wi, xi in zip(w, x))
        p = 1/(1+math.exp(-z))
        g = p - y
        b -= lr*g
        for k in range(4): w[k] -= lr*g*x[k]
print(f"[fit] weights f1..f4 = {[round(wi,3) for wi in w]}  bias {b:.3f}  (target: keep-is-safe, val n={len(V)})")

def run(rws, thr):
    n = len(rws); acc = 0; cost = 0.0
    for x, bok_em, e_sw, e_ss, f3 in rws:
        z = b + sum(wi*xi for wi, xi in zip(w, x))
        p = 1/(1+math.exp(-z))
        if p >= thr: acc += bok_em; cost += 5.7
        else:
            acc += (e_sw if f3 >= a.t3 else e_ss); cost += 9.4
    return 100*acc/n, 100*cost/(n*9.4)

grid_t = [x/100 for x in range(50, 100)] + [0.995, 0.999, 0.9995, 0.9999, 0.99999]
best = None
for t in grid_t:
    va, vc = run(V, t)
    if best is None or (va, -vc) > best[0]: best = ((va, -vc), t)
t_star = best[1]
va, vc = run(V, t_star)
ta, tc = run(T, t_star)
print(f"\n[VAL-SELECTED fusion threshold {t_star:.5f}]  val: {va:.2f} @ {vc:.1f}%")
print(f"[TEST, read once] {ta:.2f} @ {tc:.1f}% of SS   (champion to beat: 53.41 @ 88.9 | always-SS 53.54 @ 100)")
print(f"verdict: {'ABOVE SS, CHEAPER — band crossed' if ta > 53.54 and tc < 100 else ('above champion' if ta > 53.41 else 'no gain over champion')}")
print("\nval shape (reference, not for selection):")
for t in [0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]:
    va_, vc_ = run(V, t)
    print(f"  thr {t:6.3f}  val {va_:5.2f} @ {vc_:5.1f}%")
