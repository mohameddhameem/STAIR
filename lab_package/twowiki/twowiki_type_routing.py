#!/usr/bin/env python3
"""
twowiki_type_routing.py -- does routing the RETRIEVAL step by question type help? VALIDATION ONLY (val400), read-only,
CPU, seconds. Trains nothing, never opens test600. DRAFT v1.0 2026-11-10 for Jonathan's line-by-line review.

Idea (Jonathan, 12:04-12:06): comparison questions often do fine with weak retrieval; inference and compositional do not.
Policies compared (accuracy = reviewed scorer's lenient verdicts; cost = measured tokens in AgentTTS units, % of standalone
always-SS; a path that uses the router pays the weak calls the router needs as inputs -- the same weak-first prices as
twowiki_pair_sweep.py; a path that skips the router pays only its own two calls):
  P0  always-SS (no router)
  P1  fixed by type: comparison -> WS, all other types -> SS            (Jonathan's rule, no router)
  P1b fixed by type: comparison and bridge-comparison -> WS, others -> SS
  P2  refinement: non-comparison -> SS directly (no router, no weak calls); comparison -> the trained router
        P2-locked : the router at the already-locked global thresholds (0.15/0.10/0.20) -- nothing tuned
        P2-tuned  : thresholds chosen for comparison questions only, by Rule B, 2-fold half-split (see below)
  P3  per-type thresholds for every type (router everywhere), each type's thresholds by Rule B within that type, 2-fold
  REF the router's locked global point on all questions (the run we read on test)
HONEST ESTIMATE for anything tuned: val400 is split in two halves (by position parity); thresholds are chosen on one half
and scored on the other, then the two held-out halves are pooled -- every reported "tuned" number is out-of-sample.
TYPE FROM THE QUESTION TEXT: at deployment nobody hands us the 2Wiki type. A fixed keyword rule (written before seeing
its accuracy; shown below) predicts the type from the wording; its accuracy and confusion table are printed, and P1 / P2
are re-scored with the PREDICTED type -- the deployable version.
Inputs (lab): <RUN>/diag_val400.csv (from twowiki_router_diag.py), LOCKED_thresholds.json, val400 adapted json + meta stores.
Usage (lab):  python3 -I twowiki_type_routing.py
"""
import csv, json, math, os, re, importlib.util

_spec = importlib.util.spec_from_file_location("stair_cost", "/mnt/hdd/jonathan/STAIR/common/stair_common/cost.py")
_cost = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_cost)
units = _cost.units

L   = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/val400"
RUN = "/mnt/hdd/jonathan/runs/twowiki/grpo_2w_a1_fresh_ep6_objective"
P   = {"retr": {"W": 7.0, "S": 72.0}, "qa": {"W": 3.0, "S": 70.0}}
CELLS = ["WW", "WS", "SW", "SS"]
T = [i / 20 for i in range(21)]


def store(name):
    d = {}
    for line in open(os.path.join(L, name + ".jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def u(kind, w, m): return units(P[kind][w], m["n_in"], m["n_out"])


# ---- the fixed keyword rule for question type (from 2Wiki's question templates) ------------------------------------
KIN = re.compile(r"\b(grandfather|grandmother|grandparent|grandchild|grandson|granddaughter|uncle|aunt|nephew|niece|"
                 r"father-in-law|mother-in-law|son-in-law|daughter-in-law|brother-in-law|sister-in-law|"
                 r"stepfather|stepmother|step-father|step-mother|paternal|maternal)\b", re.I)
CMP = re.compile(r"(\bor\b|\bboth\b|\bsame\b|\bfirst\b|\bearlier\b|\blater\b|\bolder\b|\byounger\b|\blonger\b|"
                 r"\bmore recently\b|\bfirst\b|\blast\b|\bborn first\b|\bdied first\b|\bdied later\b)", re.I)
ROLE = re.compile(r"\b(director|directed|composer|performer|producer|creator|author|writer|founder|editor)\b", re.I)


def predict_type(q):
    if KIN.search(q):
        return "inference"
    if CMP.search(q):
        return "bridge_comparison" if ROLE.search(q) else "comparison"
    return "compositional"


# ---- load ----------------------------------------------------------------------------------------------------------
data = json.load(open(os.path.join(L, "tts_test_2wiki_adapted.json"), encoding="utf-8"))
diag = {int(r["idx"]): r for r in csv.DictReader(open(os.path.join(RUN, "diag_val400.csv"), encoding="utf-8"))}
lock = json.load(open(os.path.join(RUN, "LOCKED_thresholds.json")))
LOCK = (lock["t_ret"], lock["t_qaW"], lock["t_qaS"])
mr = {w: store(f"meta_retr_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
mq = {w: store(f"meta_qa_{'weak' if w == 'W' else 'strong'}") for w in "WS"}
rows = []
for i in range(400):
    d = diag[i]
    rW, rS = u("retr", "W", mr["W"][str(i)]), u("retr", "S", mr["S"][str(i)])
    q = {(a, b): u("qa", a, mq[a][f"{i}|{'weak' if b == 'W' else 'strong'}"]) for a in "WS" for b in "WS"}
    rows.append({"i": i, "type": d["type"], "ptype": predict_type(data[i]["question"]),
                 "em": [int(d[c]) for c in CELLS],
                 "routed": [rW + q["W", "W"], rW + q["W", "W"] + q["S", "W"], rW + rS + q["W", "S"], rW + rS + q["W", "S"] + q["S", "S"]],
                 "alone": [rW + q["W", "W"], rW + q["S", "W"], rS + q["W", "S"], rS + q["S", "S"]],
                 "pr": float(d["pW_ret"]), "pqw": float(d["pW_qa_onW"]), "pqs": float(d["pW_qa_onS"])})
C_SS = sum(r["alone"][3] for r in rows) / len(rows)


def route(r, t):
    tr, tw, ts = t
    return (0 if r["pqw"] >= tw else 1) if r["pr"] >= tr else (2 if r["pqs"] >= ts else 3)


def score(rs, decide):
    """decide(r) -> (cell, uses_router). Returns (accuracy, cost % of standalone SS, n)."""
    acc = cost = 0.0
    for r in rs:
        c, routed = decide(r)
        acc += r["em"][c]; cost += (r["routed"] if routed else r["alone"])[c]
    return acc / len(rs), cost / len(rs) / C_SS, len(rs)


def rule_b(rs, decide_t):
    """Rule B over the 21^3 grid on rows rs: best accuracy, 1-SE ties, cheapest tie wins. Returns thresholds."""
    res = [(*score(rs, lambda r, t=(a, b, c): decide_t(r, t))[:2], (a, b, c)) for a in T for b in T for c in T]
    best = max(x[0] for x in res); se = math.sqrt(best * (1 - best) / len(rs))
    return min((x for x in res if x[0] >= best - se), key=lambda x: (x[1], -x[0], x[2]))[2]


def cv(decide_with):
    """2-fold half-split: tune on one half (parity), score the other; pool both held-out halves."""
    halves = [[r for r in rows if r["i"] % 2 == k] for k in (0, 1)]
    picks, out = [], []
    for k in (0, 1):
        t = decide_with("tune", halves[k], None)
        picks.append(t)
        out += [(r, t) for r in halves[1 - k]]
    acc = sum(r["em"][decide_with("apply", None, t)(r)[0]] for r, t in out) / len(out)
    cost = sum((r["routed"] if decide_with("apply", None, t)(r)[1] else r["alone"])[decide_with("apply", None, t)(r)[0]]
               for r, t in out) / len(out) / C_SS
    return acc, cost, picks


def show(name, a, c, extra=""):
    print(f"  {name:52s} {100 * a:6.2f} %  at {100 * c:6.1f} %   {extra}")


ss_acc = score(rows, lambda r: (3, False))[0]
print(f"VAL400 type-routing analysis (n={len(rows)}) -- validation only, test600 untouched | C_SS {C_SS:.1f} units")
print(f"  {'policy':52s} {'accuracy':>8s}    {'cost (% of SS)':>14s}")
show("P0 always-SS", ss_acc, 1.0)
for label, tkey in (("gold type", "type"), ("PREDICTED type (deployable)", "ptype")):
    print(f"\n  -- {label} --")
    a, c, _ = score(rows, lambda r: (1, False) if r[tkey] == "comparison" else (3, False)); show("P1  comparison -> WS, others -> SS", a, c, f"vs SS {100 * (a - ss_acc):+.2f}")
    a, c, _ = score(rows, lambda r: (1, False) if r[tkey] in ("comparison", "bridge_comparison") else (3, False))
    show("P1b comparison + bridge -> WS, others -> SS", a, c, f"vs SS {100 * (a - ss_acc):+.2f}")
    a, c, _ = score(rows, lambda r: (route(r, LOCK), True) if r[tkey] == "comparison" else (3, False))
    show("P2-locked comparison -> router (locked t), others -> SS", a, c, f"vs SS {100 * (a - ss_acc):+.2f}")

    def p2(mode, rs, t, tkey=tkey):
        if mode == "tune":
            return rule_b(rs, lambda r, tt: (route(r, tt), True) if r[tkey] == "comparison" else (3, False))
        return lambda r: (route(r, t), True) if r[tkey] == "comparison" else (3, False)
    a, c, picks = cv(p2)
    show("P2-tuned comparison -> router (Rule B, 2-fold), others -> SS", a, c, f"vs SS {100 * (a - ss_acc):+.2f} | t per fold {picks}")

print("\n  -- router everywhere --")
a, c, _ = score(rows, lambda r: (route(r, LOCK), True)); show("REF router, locked global t (the test-read run)", a, c, f"vs SS {100 * (a - ss_acc):+.2f}")
TYPES = sorted({r["type"] for r in rows})


def p3(mode, rs, t):
    if mode == "tune":
        return {ty: rule_b([r for r in rs if r["type"] == ty], lambda r, tt: (route(r, tt), True)) for ty in TYPES}
    return lambda r: (route(r, t[r["type"]]), True)
a, c, picks = cv(p3)
show("P3  per-type thresholds (Rule B per type, 2-fold)", a, c, f"vs SS {100 * (a - ss_acc):+.2f}")
for k, pk in enumerate(picks):
    print(f"      fold {k} thresholds: " + " | ".join(f"{ty} {pk[ty]}" for ty in TYPES))

# ---- per-type view of P2's ingredients ------------------------------------------------------------------------------
print("\n  per type (gold): n | SS acc | router-locked acc / cost on that type | WS acc / cost")
for ty in TYPES:
    g = [r for r in rows if r["type"] == ty]
    a_r, c_r, _ = score(g, lambda r: (route(r, LOCK), True)); a_w, c_w, _ = score(g, lambda r: (1, False))
    print(f"    {ty:18s} {len(g):4d} | {100 * score(g, lambda r: (3, False))[0]:5.1f} | {100 * a_r:5.1f} / {100 * c_r:5.1f} %"
          f" | {100 * a_w:5.1f} / {100 * c_w:5.1f} %")

# ---- type predictor accuracy ----------------------------------------------------------------------------------------
ok = sum(r["type"] == r["ptype"] for r in rows)
print(f"\n  TYPE FROM WORDING: keyword rule accuracy {ok}/{len(rows)} = {100 * ok / len(rows):.1f} %")
PT = sorted({r["ptype"] for r in rows} | set(TYPES))
print("    gold \\ predicted  " + " ".join(f"{p[:10]:>10s}" for p in PT))
for ty in TYPES:
    print(f"    {ty:18s} " + " ".join(f"{sum(r['type'] == ty and r['ptype'] == p for r in rows):10d}" for p in PT))
bad = [(r["type"], r["ptype"], data[r["i"]]["question"]) for r in rows if r["type"] != r["ptype"]][:8]
for g_, p_, qq in bad:
    print(f"      miss: gold {g_} / predicted {p_}: {qq[:110]}")
