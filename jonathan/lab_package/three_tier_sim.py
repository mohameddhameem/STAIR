#!/usr/bin/env python3
# three_tier_sim.py v1.2 (jaccard read from the v2 label file: overlap_jaccard) — full cascade simulation: WW -> SW -> SS (23 Sep 2026)
# --tier1 mode: "verify" (default) gates tier 1 by WW-answer verification;
#               "overlap" gates tier 1 by selection-overlap (jaccard): high agreement
#               between the two retrievers' selections -> stay on WEAK evidence (the
#               overlap-derived retrieval signal), answer there, verify the answer;
#               low agreement -> escalate retrieval. Combines the project's two proven
#               signals: overlap decides RETRIEVAL, verification decides ANSWERING.
# In overlap mode tier 1 still verifies the WW answer (t1v gate) before keeping it —
# overlap picks the EVIDENCE, verification still guards the ANSWER (both stages gated).
# Tier 1: weak retrieval + weak answerer (2.0 units). Checker verified? -> done.
# Tier 2: escalate RETRIEVAL: strong retrieval + weak answerer (the SW attempt, +5.7).
#         Verified? -> done.
# Tier 3: escalate the ANSWERER on the same strong evidence (+3.7) -> SS outcome.
# Optional --two_sided: at tier 3, keep whichever of the SW/SS answers verified higher
# (rescues strong-side failures instead of trusting SS blindly).
# COST MODEL (honest, cumulative attempts): tier1-only 2.0; +tier2 -> 7.7; +tier3 -> 11.4
# (11.4 > SS 9.4: a fully escalated instance costs MORE than always-SS — the bet is that
#  enough instances stop early. The sweep decides.)
# PROTOCOL: gate thresholds (t1, t2) chosen on VAL by best accuracy (tie: lower cost);
# sealed TEST read once at the chosen pair. Reference val Pareto printed for shape.
import json, argparse

CELL = {"WW":0, "WS":1, "SW":2, "SS":3}
LADDER = "ladder: WW 31.80 @21.3% | SW 41.73 @60.6% | SS 53.54 @100% | oracle 61.24 @~24%"

def jl(p): return [json.loads(l) for l in open(p)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
    ap.add_argument("--ww_scores", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores_ww.jsonl")
    ap.add_argument("--sw_scores", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl")
    ap.add_argument("--ss_scores", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores_strongans.jsonl")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--two_sided", action="store_true", help="tier 3 keeps the better-verified of SW/SS answers")
    ap.add_argument("--tier1", default="verify", choices=["verify","overlap"], help="tier-1 gate signal")
    ap.add_argument("--jaccard_from", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_v2_tau0.50.jsonl",
                    help="file carrying overlap_jaccard per idx (the v2 label file)")
    a = ap.parse_args()

    grid = jl(a.grid)
    ww = {r["idx"]: r["p_supported"] for r in jl(a.ww_scores)}
    sw = {r["idx"]: r["p_supported"] for r in jl(a.sw_scores)}
    ss = {r["idx"]: r["p_supported"] for r in jl(a.ss_scores)}
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    jmap = {}
    if a.tier1 == "overlap":
        for l in open(a.jaccard_from):
            r = json.loads(l); jmap[r["idx"]] = r.get("overlap_jaccard")
        got = sum(1 for v in jmap.values() if v is not None)
        print(f"[jaccard] loaded overlap_jaccard for {got}/{len(jmap)} rows from {a.jaccard_from}")

    def rows(ids):
        out = []
        for i in ids:
            if i in ww and i in sw and i in ss:
                e = grid[i]["em_lenient"]
                j = jmap.get(i) if a.tier1 == "overlap" else 0.0
                if a.tier1 == "overlap" and j is None: continue
                out.append((ww[i], sw[i], ss[i], e, j))
        return out

    def run(rws, t1, t2, t1v=0.999):
        n = len(rws); acc = 0; cost = 0.0
        for s1, s2, s3, e, j in rws:
            gate1 = (s1 >= t1) if a.tier1 == "verify" else (j >= t1 and s1 >= t1v)
            if gate1: acc += e[CELL["WW"]]; cost += 2.0; continue
            if s2 >= t2: acc += e[CELL["SW"]]; cost += 7.7; continue
            if a.two_sided and s2 > s3: acc += e[CELL["SW"]]
            else:                        acc += e[CELL["SS"]]
            cost += 11.4
        return 100*acc/n, 100*cost/(n*9.4)

    V = rows(sp["val"]); T = rows(sp["test"])
    print(f"[coverage] val {len(V)}/{len(sp['val'])}  test {len(T)}/{len(sp['test'])}  two_sided={a.two_sided} tier1={a.tier1}")
    G = [0.0,0.2,0.4,0.6,0.8,0.9,0.95,0.99,0.999,0.9999,0.99999,0.999999,1.0-1e-9]
    if a.tier1 == "overlap":
        G1 = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]      # jaccard scale
    else:
        G1 = G
    best = None
    for t1 in G1:
        for t2 in G:
            va, vc = run(V, t1, t2)
            if best is None or (va, -vc) > best[0]: best = ((va, -vc), t1, t2)
    _, t1s, t2s = best
    va, vc = run(V, t1s, t2s)
    ta, tc = run(T, t1s, t2s)
    print(f"\n[VAL-SELECTED] t1={t1s:.6f} t2={t2s:.6f}  (val: {va:.2f} @ {vc:.1f}%)")
    print(f"[TEST, read once] {ta:.2f} @ {tc:.1f}% of always-SS   ({LADDER})")
    print(f"verdict: {'IN THE BAND (above SS, below oracle)' if 53.54 < ta else 'below SS'}"
          f"{' AND CHEAPER' if tc < 100 and ta > 53.54 else ''}")
    print("\nreference VAL Pareto shape (not for selection):")
    print(f"  {'t1':>9} {'t2':>9} {'val acc':>8} {'val cost%':>9}")
    pts = sorted({(round(run(V,t1,t2)[1],1), round(run(V,t1,t2)[0],2), t1, t2) for t1 in G1 for t2 in G})
    keep = []; hi = -1
    for c_, a_, t1_, t2_ in pts:
        if a_ > hi: keep.append((t1_, t2_, a_, c_)); hi = a_
    for t1_, t2_, a_, c_ in keep:
        print(f"  {t1_:9.6f} {t2_:9.6f} {a_:8.2f} {c_:9.1f}")

if __name__ == "__main__":
    main()
