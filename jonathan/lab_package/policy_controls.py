#!/usr/bin/env python3
# =============================================================================
# policy_controls.py — VERSION: policy_controls v1.0 2026-09-06
#
# The "compared to what?" ladder (slide 11): router-FREE policies, all pushed
# through the SAME protocol as Run 1c (tune on val, freeze, report test), so
# every rung is one more curve on the same cost–accuracy plot. CPU-only.
#
#   RUNG 1 — base-rate policy (no per-question information at all).
#     With identical probabilities for every question, a τ-policy can only
#     pick ONE cell for all traffic -> the honest rung-1 frontier is the four
#     fixed-cell points {WW, WS, SW, SS} plus their random-mixing hull. This
#     is the "no discrimination" frontier the trained router must beat.
#   RUNG 2 — one-feature rules (question-only features; each rule = a
#     per-instance score fed to the τ sweep): dataset 'level' (easy/med/hard),
#     'type' (comparison/bridge), question length.
#   RUNG 3 — shallow classifier (gradient boosting on question-only features)
#     -> predicted per-stage probabilities -> τ sweep; AUC directly comparable
#     to Run 1c's 0.580 / 0.550. Skipped with a message if sklearn is absent.
#
#   SCOPE NOTE (cost honesty): features derived from the weak retriever's
#   OUTPUT (evidence_W length, number of selected refs) assume weak retrieval
#   has already run (+~1 cost unit, cascade-shaped). Those are reported as
#   AUC ONLY, labeled "cascade preview"; their full cost-correct policy is
#   Run 3's job. Rungs 2–3 policy sweeps use question-only features, whose
#   cost semantics are identical to Run 1c's.
#
# USAGE (lab, venv; no GPU, minutes):
#   python3 policy_controls.py --data .../full_run_p100/router_data.jsonl \
#       --splits splits.json --run1c /mnt/hdd/jonathan/runs/run1c \
#       --out /mnt/hdd/jonathan/runs/controls
# Outputs: controls_result.json · ladder.png (all rungs + SS/WW + Run 1c τ*)
# =============================================================================
import os, json, argparse
from collections import Counter

VERSION = "policy_controls v1.0 2026-09-06"
COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
SS = 3

# reuse Run 1c's machinery verbatim (import is safe: both scripts are main-guarded)
from threshold_policy import auc, eval_policy, baselines, cell_of


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--run1c", default=None, help="Run 1c out dir; overlays its frozen τ* points")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau_step", type=float, default=0.02)
    return ap.parse_args()


def load_rows(path):
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["idx"]] = r
    return rows


# ------------------------- features (question-only) --------------------------
def q_features(r):
    q = r["question"]
    t = 1.0 if r.get("type") == "comparison" else 0.0
    lv = {"easy": 0.0, "medium": 0.5, "hard": 1.0}.get(r.get("level"), 0.5)
    return {"qlen": len(q.split()), "type_comparison": t, "level": lv}

# cascade-preview features (weak retrieval assumed already run; AUC only)
def wk_features(r):
    ev = r["evidence_W"]
    return {"evW_chars": len(ev), "evW_refs": ev.count("["), }


def rank01(vals):
    """Rank-normalize a list to [0,1] (ties get mean rank). Turns any feature
    into a probability-shaped score the τ sweep can threshold."""
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    r = [0.0] * len(vals); i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        avg = (i + j) / 2 / max(1, len(vals) - 1)
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def scores_from(vals_by_idx, ids, invert):
    """Map one feature to the 3-score interface used by eval_policy:
    high score = 'weak suffices'. invert=True flips the direction."""
    ids = list(ids)
    ranked = rank01([vals_by_idx[i] for i in ids])
    if invert:
        ranked = [1 - v for v in ranked]
    return {i: {"p_ret_W": s, "p_qaW_W": s, "p_qaS_W": s} for i, s in zip(ids, ranked)}


def sweep(rows, scores, va, te, taus, ss_val):
    grid = {}
    for tr in taus:
        for tq in taus:
            grid[(tr, tq)] = eval_policy(rows, scores, va, tr, tq)
    parity = max((g for g in grid if grid[g][0] >= ss_val), key=lambda g: -grid[g][2], default=None)
    best = max(grid, key=lambda g: (grid[g][0], -grid[g][2]))
    out = {}
    for name, g in (("parity", parity), ("best_acc", best)):
        if g is None:
            out[name] = None; continue
        lt, st, ct = eval_policy(rows, scores, te, *g)
        out[name] = {"tau": g, "test_lenient": lt, "test_strict": st, "test_cost_pct": ct}
    # full test curve for the plot
    curve = sorted({(eval_policy(rows, scores, te, tr, tq)[2],
                     eval_policy(rows, scores, te, tr, tq)[0]) for tr in taus for tq in taus})
    return out, curve


def main():
    a = build_args()
    os.makedirs(a.out, exist_ok=True)
    print(VERSION, flush=True)
    rows = load_rows(a.data)
    sp = json.load(open(a.splits))["splits"]
    va = [i for i in sp["val"] if i in rows]
    te = [i for i in sp["test"] if i in rows]
    base_va, base_te = baselines(rows, va), baselines(rows, te)
    ss_val = base_va["always_SS"]["lenient"]
    taus = [round(a.tau_step * k, 4) for k in range(int(1 / a.tau_step) + 1)] + [1.01]
    result = {"version": VERSION, "baselines_test": base_te, "rungs": {}}

    # ---- RUNG 1: the four fixed cells (no per-question information)
    fixed = {}
    for k, c in enumerate(COMBOS):
        lens = sum(rows[i]["em_lenient"][k] for i in te) / len(te)
        strict = sum(rows[i]["em_strict"][k] for i in te) / len(te)
        fixed[c] = {"test_lenient": round(100 * lens, 2), "test_strict": round(100 * strict, 2),
                    "test_cost_pct": round(100 * COST[c] / COST["strong-strong"], 1)}
        print(f"[rung1 fixed {c}] {fixed[c]}", flush=True)
    result["rungs"]["1_base_rate_fixed_cells"] = fixed

    # ---- RUNG 2: one-feature rules (question-only), each swept both directions
    feats = {i: q_features(rows[i]) for i in va + te}
    rung2 = {}
    for fname in ("qlen", "type_comparison", "level"):
        vals = {i: feats[i][fname] for i in va + te}
        for direction, inv in (("high=weak", False), ("high=strong", True)):
            sc = scores_from(vals, va + te, inv)
            chosen, curve = sweep(rows, sc, va, te, taus, ss_val)
            key = f"{fname} ({direction})"
            rung2[key] = chosen
            best = chosen["best_acc"]
            print(f"[rung2 {key}] best_acc test {best['test_lenient']} @ {best['test_cost_pct']}%", flush=True)
    result["rungs"]["2_one_feature_rules"] = rung2

    # ---- RUNG 3: shallow classifier on question-only features (if sklearn)
    rung3 = {"skipped": True}
    try:
        from sklearn.ensemble import GradientBoostingClassifier
        import numpy as np
        def X_of(ids): return np.array([[feats[i]["qlen"], feats[i]["type_comparison"], feats[i]["level"]] for i in ids])
        tr_ids = [i for i in sp["train"] if i in rows and rows[i]["ret_label"]]
        feats.update({i: q_features(rows[i]) for i in tr_ids})
        y_tr = np.array([1 if rows[i]["ret_label"] == "W" else 0 for i in tr_ids])
        clf = GradientBoostingClassifier(random_state=0).fit(X_of(tr_ids), y_tr)
        pW = {i: float(p) for i, p in zip(va + te, clf.predict_proba(X_of(va + te))[:, 1])}
        lab_te = [i for i in te if rows[i]["ret_label"]]
        rung3 = {"skipped": False,
                 "ret_auc_test": auc([(1 - pW[i], 1 if rows[i]["ret_label"] == "S" else 0) for i in lab_te])}
        sc = {i: {"p_ret_W": pW[i], "p_qaW_W": pW[i], "p_qaS_W": pW[i]} for i in va + te}
        chosen, curve3 = sweep(rows, sc, va, te, taus, ss_val)
        rung3["policy"] = chosen
        print(f"[rung3 classifier] ret AUC {rung3['ret_auc_test']} (Run 1c router: 0.580) | "
              f"best_acc test {chosen['best_acc']['test_lenient']} @ {chosen['best_acc']['test_cost_pct']}%", flush=True)
    except ImportError:
        print("[rung3] sklearn not installed — skipped (pip install scikit-learn to enable)", flush=True)
    result["rungs"]["3_feature_classifier"] = rung3

    # ---- cascade-preview AUCs (weak-evidence features; AUC only, no policy)
    wk = {i: wk_features(rows[i]) for i in te}
    prev = {}
    for fname in ("evW_chars", "evW_refs"):
        lab = [i for i in te if rows[i]["ret_label"]]
        vals = rank01([wk[i][fname] for i in lab])
        prev[fname] = auc([(1 - v, 1 if rows[i]["ret_label"] == "S" else 0) for v, i in zip(vals, lab)])
    result["cascade_preview_auc"] = prev
    print(f"[cascade preview] retrieval-label AUC from weak-evidence features: {prev} "
          f"(cost-correct policy = Run 3)", flush=True)

    # ---- plot: all rungs on one frontier
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
        for c, d in fixed.items():
            ax.scatter([d["test_cost_pct"]], [d["test_lenient"]], marker="D", s=70,
                       color="#8a8a8a", zorder=5)
            ax.annotate(c, (d["test_cost_pct"], d["test_lenient"]), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")
        hull = sorted([(d["test_cost_pct"], d["test_lenient"]) for d in fixed.values()])
        ax.plot([p[0] for p in hull], [p[1] for p in hull], ls=":", color="#8a8a8a",
                lw=1, label="rung 1: fixed cells (+mix)")
        if not rung3.get("skipped"):
            b = rung3["policy"]["best_acc"]
            ax.scatter([b["test_cost_pct"]], [b["test_lenient"]], marker="^", s=90,
                       color="#6B4E8E", zorder=6, label="rung 3: classifier τ*")
        if a.run1c and os.path.exists(os.path.join(a.run1c, "policy_result.json")):
            r1 = json.load(open(os.path.join(a.run1c, "policy_result.json")))
            ch = r1["chosen"].get("parity")
            if ch:
                ax.scatter([ch["test"]["cost_pct_of_SS"]], [ch["test"]["lenient"]], marker="o",
                           s=90, color="#D97A3A", zorder=6, label="Run 1c router τ* (parity)")
        ax.set_xlabel("cost, % of always-SS"); ax.set_ylabel("lenient accuracy (test)")
        ax.set_title("Controls ladder — what router-free policies achieve")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "ladder.png"))
        print("wrote ladder.png", flush=True)
    except Exception as e:
        print(f"(plot skipped: {e})", flush=True)

    json.dump(result, open(os.path.join(a.out, "controls_result.json"), "w"), indent=2)
    print(f"saved -> {a.out}/controls_result.json", flush=True)


if __name__ == "__main__":
    main()
