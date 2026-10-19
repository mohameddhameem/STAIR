#!/usr/bin/env python3
# =============================================================================
# calibrate_policy.py — VERSION: calibrate_policy v2.1 2026-09-15 (deployment-cost accounting: judge FLOPs included; ret-gate-only variant)
#   (rebuilt against the CONFIRMED router_scores.jsonl schema:
#    {"idx", "split", "p_ret_W", "p_qaW_W", "p_qaS_W"} — p_S = 1 - p_W)
#
# OPTION B, quick verdict: can the zero-shot 7B judge's scores, plus a
# threshold policy (no training), route better than always-weak / always-
# strong? Steps:
#   1. Report score saturation (the pathology seen raw on 15 Sep: p=1.0).
#   2. Recover logits via clipped log-odds; fit temperature T on VAL
#      (minimize NLL vs ret labels) — honest note: saturated inputs clip,
#      so T has limited purchase; reported, not hidden.
#   3. Sweep thresholds tau_ret / tau_qa on VAL (composed lenient
#      accuracy); apply the best pair to TEST once.
#   4. Report TEST: policy vs WW vs SS vs oracle, capture %, cost % of SS.
#
# USAGE (lab, CPU, seconds):
#   python3 calibrate_policy.py \
#     --scores /mnt/hdd/jonathan/runs/zs_evidence/router_scores.jsonl \
#     --data   /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl
# =============================================================================
import json, math, argparse, collections

VERSION = "calibrate_policy v2.1 2026-09-15 (deployment-cost accounting: judge FLOPs included; ret-gate-only variant)"
CELLS = ["WW", "WS", "SW", "SS"]
COST = {"WW": 2.0, "WS": 5.7, "SW": 5.7, "SS": 9.4}   # FLOP units (manifest currency)
CLIP = 1e-6


def logit(p):
    p = min(max(p, CLIP), 1 - CLIP)
    return math.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--judge_cost", type=float, default=4.7, help="FLOP units per judge call (default: one strong-stage pass)")
    a = ap.parse_args()
    print(VERSION, flush=True)

    S = {}
    for l in open(a.scores):
        r = json.loads(l)
        S[r["idx"]] = r
    D = {}
    for l in open(a.data):
        r = json.loads(l)
        D[r["idx"]] = r
    splits = collections.Counter(S[i]["split"] for i in S)
    print(f"scores: {len(S)} rows by split {dict(splits)} | data rows: {len(D)}")

    # ---- 1. saturation report -------------------------------------------------
    for key in ("p_ret_W", "p_qaW_W", "p_qaS_W"):
        vals = [S[i][key] for i in S if key in S[i]]
        sat = sum(1 for v in vals if v >= 0.999)
        lo = sum(1 for v in vals if v <= 0.001)
        print(f"  {key}: n={len(vals)} | >=0.999: {sat} ({100*sat/len(vals):.1f}%) | <=0.001: {lo} | distinct: {len(set(vals))}")

    # ---- 2. temperature on VAL ret --------------------------------------------
    val = [i for i in S if S[i]["split"] == "val" and D.get(i, {}).get("ret_label")]
    test = [i for i in S if S[i]["split"] == "test" and i in D]
    print(f"val rows with ret label: {len(val)} | test rows: {len(test)}")

    def nll(T):
        tot = 0.0
        for i in val:
            z = logit(S[i]["p_ret_W"]) / T
            pW = 1 / (1 + math.exp(-z))
            p = pW if D[i]["ret_label"] == "W" else 1 - pW
            tot += -math.log(min(max(p, CLIP), 1 - CLIP))
        return tot / max(len(val), 1)

    Ts = [0.5, 1, 2, 4, 8, 16, 32, 64, 128]
    best_T = min(Ts, key=nll)
    print(f"temperature sweep NLL: " + " ".join(f"T={t}:{nll(t):.3f}" for t in Ts))
    print(f"  best T = {best_T} (note: saturation clips most logits to +/-{logit(1-CLIP):.1f}, so T mostly rescales a constant)")

    # ---- 3/4. threshold policy ------------------------------------------------
    def route(i, tr, tq):
        ch = "W" if S[i]["p_ret_W"] >= tr else "S"
        pq = S[i]["p_qaW_W"] if ch == "W" else S[i]["p_qaS_W"]
        qa = "W" if pq >= tq else "S"
        return ch + qa

    def em(i, cell):
        e = D[i].get("em_lenient") or [0, 0, 0, 0]
        return e[CELLS.index(cell)]

    def evaluate(ids, tr, tq):
        n = len(ids)
        acc = sum(em(i, route(i, tr, tq)) for i in ids) / n
        ww = sum(em(i, "WW") for i in ids) / n
        ss = sum(em(i, "SS") for i in ids) / n
        oracle = sum(1 for i in ids if any((D[i].get("em_lenient") or [0]*4))) / n
        cost = sum(COST[route(i, tr, tq)] for i in ids) / (COST["SS"] * n)
        picks = collections.Counter(route(i, tr, tq) for i in ids)
        cap = 100 * (acc - ss) / (oracle - ss) if oracle != ss else float("nan")
        return acc, ww, ss, oracle, cost, picks, cap

    cands = sorted({round(S[i]["p_ret_W"], 6) for i in val} | {0.5, 0.9, 0.99, 0.999, 0.9999, 1.0})
    if len(cands) > 60:
        cands = cands[:: max(1, len(cands) // 60)]
    best = None
    for tr in cands:
        for tq in cands:
            accv = sum(em(i, route(i, tr, tq)) for i in val) / max(len(val), 1)
            if best is None or accv > best[0]:
                best = (accv, tr, tq)
    accv, tr, tq = best
    print(f"\nVAL sweep: best tau_ret={tr} tau_qa={tq} (val composed lenient {100*accv:.2f}%) over {len(cands)}^2 candidates")

    acc, ww, ss, oracle, cost, picks, cap = evaluate(test, tr, tq)
    print(f"\nTEST (n={len(test)}), thresholds frozen from val:")
    print(f"  policy lenient {100*acc:.2f} | WW {100*ww:.2f} | SS {100*ss:.2f} | oracle {100*oracle:.2f}")
    print(f"  capture {cap:.1f}% | PIPELINE cost {100*cost:.1f}% of SS | picks {dict(picks)}")

    # ---- deployment-cost accounting (the honest Option-B economics) ----------
    # FLOP currency derived from the manifest costs: wr = wq = 1.0, sr = sq = 4.7
    # (ww=2.0, ws=5.7, sw=5.7, ss=9.4). A judge call = a 7B pass over
    # question+evidence ~= a strong-stage call = 4.7 (tunable via --judge_cost).
    jc = a.judge_cost
    # Variant FULL (today's policy): run weak retrieval to produce evidence for
    # the judge (1.0, wasted if S chosen), judge ret (jc), judge qa (jc), then
    # the chosen pipeline cells. Weak-ret cost double-counted for W* picks
    # (already in cell cost) -> add 1.0 only when ret choice is S.
    def deploy_cost(ids, tr, tq, variant):
        tot = 0.0
        for i in ids:
            cell = route(i, tr, tq)
            c = COST[cell] + jc                      # ret judge call
            if cell[0] == "S":
                c += 1.0                              # wasted weak retrieval (judge input)
            if variant == "full":
                c += jc                               # qa judge call
            tot += c
        return tot / (COST["SS"] * len(ids))

    def route_retgate(i, tr):
        ch = "W" if S[i]["p_ret_W"] >= tr else "S"
        return ch + ch                                # one judge call; qa mirrors ret
    accg = sum(em(i, route_retgate(i, tr)) for i in test) / len(test)
    costg = 0.0
    for i in test:
        cell = route_retgate(i, tr)
        c = COST[cell] + jc
        if cell[0] == "S":
            c += 1.0
        costg += c
    costg /= (COST["SS"] * len(test))

    print(f"\nDEPLOYMENT ECONOMICS (judge_cost={jc} FLOP units per call, = one strong-stage pass):")
    print(f"  FULL policy   : lenient {100*acc:.2f} | TOTAL cost {100*deploy_cost(test,tr,tq,'full'):.1f}% of SS  (2 judge calls + wasted weak-ret on S picks)")
    print(f"  RET-GATE only : lenient {100*accg:.2f} | TOTAL cost {100*costg:.1f}% of SS  (1 judge call; qa mirrors ret choice)")
    print(f"  reference     : always-SS = 100% cost, {100*ss:.2f} lenient | always-WW = {100*COST['WW']/COST['SS']:.1f}% cost, {100*ww:.2f} lenient")
    print(f"  reading: if TOTAL cost approaches/exceeds 100%, Option B's accuracy survives but its economics argue for")
    print(f"           DISTILLATION (Option A): move the judge's signal into the 1.5B so the judge fee disappears.")
    if acc > max(ww, ss) + 1e-9:
        verdict = "POLICY BEATS BOTH BASELINES"
    elif acc >= max(ww, ss) - 1e-9 and cost < 0.999:
        verdict = f"POLICY MATCHES BEST BASELINE AT {100*cost:.0f}% OF SS COST (a routing win if it holds)"
    elif abs(acc - ww) < 0.005:
        verdict = "POLICY ~ ALWAYS-WEAK (saturation leaves thresholds no purchase)"
    else:
        verdict = "POLICY BELOW BEST BASELINE"
    print("\nVERDICT:", verdict)


if __name__ == "__main__":
    main()
