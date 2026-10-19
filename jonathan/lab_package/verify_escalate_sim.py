#!/usr/bin/env python3
# verify_escalate_sim.py v1.6.1 (fix: sbok fallback instead of row drop — full population kept) — CPU simulation of the verify-then-escalate architecture (21 Sep 2026)
#
# ARCHITECTURE UNDER TEST (no training, no prediction):
#   For every instance: run the cheap pipeline first; VERIFY its answer against the
#   evidence with the independent verifier; ESCALATE to the strong answerer only when
#   the verifier's confidence is below a threshold t. Sweep t -> cost/accuracy curve.
#
# Because the grid already contains all four measured outcomes per instance, and the
# verifier scores already exist on disk, the whole architecture can be SIMULATED:
#   kept  (score >= t):  outcome = em[base_cell],     cost = COST[base_cell]
#   escal (score <  t):  outcome = em[escal_cell],    cost = COST[base_cell] + answerer-only surcharge
#
# Two variants simulated:
#   A) QA-side cascade on strong evidence:  base SW  -> escalate to SS   (answerer swap; evidence reused)
#   B) QA-side cascade on weak evidence:    base WW  -> escalate to WS   (cheapest base)
#
# Escalation surcharge: strong answerer pass only = COST[SS]-COST[SW] (=3.7) on strong
# evidence; COST[WS]-COST[WW] (=3.7) on weak evidence. Retrieval is NOT rerun.
#
# Inputs: the measured grid + a verifier-scores file (path passed in; format probed at
# runtime and printed before use so nothing is silently assumed).
# Output: threshold sweep table per variant + the best operating points vs the ladder.
import json, argparse, sys

COST = {"WW": 2.0, "WS": 5.7, "SW": 5.7, "SS": 9.4}
CELL = {"WW": 0, "WS": 1, "SW": 2, "SS": 3}
LADDER = "ladder: WW 31.80 @21.3% | SW 41.73 @60.6% | SS 53.54 @100% | oracle 61.24 @~24%"

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
    ap.add_argument("--scores", required=True, help="verifier scores file (jsonl or json)")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--probe_only", action="store_true", help="print score-file structure and exit")
    ap.add_argument("--val_select", action="store_true", help="choose threshold on VAL, then report TEST once at that threshold (meeting-grade protocol)")
    ap.add_argument("--compare_scores", default="", help="verifier_compare.jsonl: escalated instances keep the WEAK outcome when p_weak_better >= t3 (comparative tier-3); t3 joins the val-selected sweep")
    ap.add_argument("--bok", default="", help="bok_summary.jsonl: keep-tier uses best-of-4 (gate on bok_score; kept outcome = bok_em instead of the grid SW cell)")
    ap.add_argument("--sbok", default="", help="sbok_summary.jsonl: escalated (non-rescued) outcome = strong best-of-4 (sbok_em) instead of the plain SS cell")
    a = ap.parse_args()

    # ---- probe the scores file structure first; assume nothing ----
    txt = open(a.scores).read()
    try:
        obj = json.loads(txt)          # whole-file json
        rows_s = obj if isinstance(obj, list) else None
    except json.JSONDecodeError:
        rows_s = [json.loads(l) for l in txt.splitlines() if l.strip()]
    if rows_s is None:
        sys.exit("scores file is a dict; inspect manually")
    print(f"[scores] {a.scores}: {len(rows_s)} rows; keys of row0: {sorted(rows_s[0].keys())}")
    print(f"[scores] row0 sample: {json.dumps(rows_s[0])[:300]}")
    if a.probe_only:
        return

    # ---- key mapping: edit here after the probe if names differ ----
    IDX_KEY = "idx"
    # candidate per-branch verifier-score keys, tried in order:
    W_KEYS = ["p_supported_weakbranch", "p_qaW_W", "ver_W", "verifier_W", "p_supported_W", "score_W"]
    S_KEYS = ["p_supported", "p_qaS_W", "ver_S", "verifier_S", "p_supported_S", "score_S"]
    def get(r, keys):
        for k in keys:
            if k in r: return r[k]
        return None

    smap = {}
    for r in rows_s:
        if IDX_KEY not in r: continue
        smap[r[IDX_KEY]] = (get(r, W_KEYS), get(r, S_KEYS))
    n_w = sum(1 for v in smap.values() if v[0] is not None)
    n_s = sum(1 for v in smap.values() if v[1] is not None)
    print(f"[scores] mapped: {len(smap)} idx; weak-branch scores {n_w}, strong-branch {n_s}")

    grid = load_jsonl(a.grid)
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    test = [i for i in sp["test"]]

    val = [i for i in sp["val"]]
    sbok = {}
    if a.sbok:
        for l in open(a.sbok):
            r = json.loads(l); sbok[r["idx"]] = r["sbok_em"]
        print(f"[sbok] loaded strong best-of-4 rows for {len(sbok)} instances")
    bok = {}
    if a.bok:
        for l in open(a.bok):
            r = json.loads(l); bok[r["idx"]] = (r["bok_score"], r["bok_em"])
        print(f"[bok] loaded best-of-4 rows for {len(bok)} instances")
    cmap = {}
    if a.compare_scores:
        for l in open(a.compare_scores):
            r = json.loads(l); cmap[r["idx"]] = r["p_weak_better"]
        print(f"[compare] loaded p_weak_better for {len(cmap)} rows")

    def rows_for(ids, base, escal, branch):
        out = []
        for i in ids:
            sc = smap.get(i, (None, None))[branch]
            if sc is None: continue
            if a.compare_scores and i not in cmap: continue
            if a.bok and i not in bok: continue
            e = grid[i]["em_lenient"]
            e_base = e[CELL[base]]
            if a.bok:
                sc, e_base = bok[i]      # gate on max-of-4 score; kept outcome = picked sample
            # sbok covers only the escalated pool; instances without a row (p_supported
            # = 1.0, never escalated) fall back to the plain SS outcome — NEVER dropped.
            e_esc = sbok.get(i, e[CELL[escal]]) if a.sbok else e[CELL[escal]]
            out.append((sc, e_base, e_esc, cmap.get(i, -1.0)))
        return out

    def point(rows, base, escal, t, t3=None):
        surcharge = COST[escal] - COST[base]
        n = len(rows); acc = 0; cost = 0.0
        for sc, e_b, e_e, pw in rows:
            if sc >= t: acc += e_b; cost += COST[base]
            else:
                if t3 is not None and pw >= t3: acc += e_b   # comparative rescue: keep WEAK outcome
                else:                            acc += e_e
                cost += COST[base] + surcharge
        return 100*acc/n, 100*cost/(n*COST["SS"])

    def val_select(base, escal, branch):
        vr = rows_for(val, base, escal, branch)
        tr = rows_for(test, base, escal, branch)
        if not vr or not tr:
            print(f"  [{base}->{escal}] missing val or test scores"); return
        grid_t = [x/100 for x in range(101)] + [0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999, 1.0 - 1e-9]
        T3 = ([None] if not a.compare_scores else
              [None, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999, 0.99999, 0.999999, 1.0 - 1e-9])
        best = None
        for t in grid_t:
            for t3 in T3:
                a_, c_ = point(vr, base, escal, t, t3)
                if best is None or (a_, -c_) > best[0]: best = ((a_, -c_), t, t3)
        t_star, t3_star = best[1], best[2]
        va, vc = point(vr, base, escal, t_star, t3_star)
        ta, tc = point(tr, base, escal, t_star, t3_star)
        t3txt = "none (blind escalation)" if t3_star is None else f"{t3_star:.6f}"
        print(f"\n[{base} -> {escal}] VAL-SELECTED keep-gate {t_star:.2f}, comparative t3 = {t3txt} (val: {va:.2f} @ {vc:.1f}%)")
        print(f"  TEST (read once): {ta:.2f} @ {tc:.1f}% of SS   ({LADDER})")
        print(f"  verdict: {'ABOVE SS, CHEAPER  <-- GOAL LINE CROSSED' if ta>53.54 and tc<100 else 'below SS at this threshold'}")
        print(f"  fine sweep near 1.0 (VAL -> chosen | TEST shown for the val-best only):")
        fine = [0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99995, 0.99999, 0.999995, 0.999999, 1.0 - 1e-9]
        fbest = None
        for t in fine:
            a_, c_ = point(vr, base, escal, t)
            k_ = 100*sum(1 for sc,_,_,_ in vr if sc >= t)/len(vr)
            print(f"    thr {t:.6f}  val kept {k_:5.1f}%  val acc {a_:5.2f} @ {c_:5.1f}%")
            if fbest is None or (a_, -c_) > fbest[0]: fbest = ((a_, -c_), t)
        t2 = fbest[1]
        ta2, tc2 = point(tr, base, escal, t2)
        print(f"  fine VAL-best thr {t2:.6f} -> TEST {ta2:.2f} @ {tc2:.1f}%   "
              f"{'<-- GOAL LINE CROSSED' if ta2>53.54 and tc2<100 else ''}")

    def sweep(base, escal, branch):  # branch: 0 = weak-evidence scores, 1 = strong-evidence
        surcharge = COST[escal] - COST[base]
        rows = rows_for(test, base, escal, branch)
        if not rows:
            print(f"  [{base}->{escal}] no scored test rows — check key mapping"); return
        n = len(rows)
        print(f"\n[{base} -> {escal}] scored test rows: {n}   ({LADDER})")
        print(f"  {'thr':>5} {'kept%':>6} {'acc':>6} {'cost%':>6}")
        best = None
        for t in [x/20 for x in range(21)]:
            acc = 0; cost = 0.0
            for sc, e_base, e_esc, _pw in rows:
                if sc >= t: acc += e_base; cost += COST[base]
                else:       acc += e_esc;  cost += COST[base] + surcharge
            accp = 100*acc/n; costp = 100*cost/(n*COST["SS"])
            kept = 100*sum(1 for sc,_,_,_ in rows if sc >= t)/n
            marker = ""
            if accp > 53.54 and costp < 100: marker = "  <-- BEATS SS, CHEAPER"
            print(f"  {t:5.2f} {kept:6.1f} {accp:6.2f} {costp:6.1f}{marker}")
            if best is None or (accp, -costp) > best[0]: best = ((accp, -costp), t)
        print(f"  best acc point: thr {best[1]:.2f} -> {best[0][0]:.2f} @ {-best[0][1]:.1f}% cost")

    if a.val_select:
        val_select("SW", "SS", branch=1)
        print("\n--- reference full-test sweeps below (not for threshold choice) ---")
    sweep("SW", "SS", branch=1)   # variant A: strong evidence, answerer cascade
    sweep("WW", "WS", branch=0)   # variant B: weak evidence, answerer cascade

if __name__ == "__main__":
    main()
