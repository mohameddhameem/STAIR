# CPU re-analysis run on 13 Oct 2026 from saved notebook outputs (no GPU). Paths at the top/arguments point to the
# working copies used then; point them at the notebook 26/27 run folders to re-run. Results are in docs/Experiment_Plan_and_Log.md (Set 9).
"""Notebook 27 re-analysis (CPU, saved outputs): fixed decision rule, does the model switch effectively?"""
import argparse, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import route9 as R9
from sklearn.metrics import roc_auc_score as AUC

P = "/content/drive/MyDrive/0_EngD_assignments/0_Small_Big_Model_Switching/data/hotpotqa_q"
W = sys.argv[1]
args = argparse.Namespace(old_data=f"{P}/dev100_test100_c20_s42_first", fresh_data=f"{P}/fresh200_c20_s43", run_dir=f"{W}/run",
                          label_dir=f"{W}/label", models_json=f"{W}/run/models.json")
A = R9.load_answers(args)
SM = {(r["qid"], R9.key(r["sel"])): r for r in R9.S.read_jsonl(f"{W}/run/step/samples_q32.jsonl")}
Wc, Wt = R9.World(args, "calib", A, SM), R9.World(args, "test", A, SM)
cal = R9.calibrate(Wc)
_rs, _as = {}, {}
_ret_state, _ans_state = R9.ret_state, R9.ans_state
def ret_state_m(Wx, q, m):
    kk = (id(Wx), q, tuple(m))
    if kk not in _rs:
        _rs[kk] = _ret_state(Wx, q, m)
    return _rs[kk]
def ans_state_m(Wx, q, sel, start, step, rs):
    kk = (id(Wx), q, sel, start, step, tuple(rs["members"]))
    if kk not in _as:
        _as[kk] = _ans_state(Wx, q, sel, start, step, rs)
    return _as[kk]
R9.ret_state, R9.ans_state = ret_state_m, ans_state_m
k = R9.key
# reference: cheap fixed setup (Qwen 7B retrieval 46.71 TF mean from notebook 26, Qwen 32B answer on its shortlist)
ref = {r["qid"]: k(r["sel"]) for r in R9.S.read_jsonl(f"{W}/run/step/test_ref_shortlists.jsonl")}
fixed_y = np.array([A["q32"][(q, ref[q])]["lenient"] for q in Wt.qids], float)
fixed_c = np.array([46.71e12 + A["q32"][(q, ref[q])]["flops"] for q in Wt.qids])
r3 = {r["qid"]: r for r in R9.S.read_jsonl(f"{W}/run/step/test_retrieval_r3.jsonl")}
r08 = {r["qid"]: r for r in R9.S.read_jsonl(f"{W}/run/step/test_retrieval_r08.jsonl")}
y3 = np.array([A["q32"][(q, k(r3[q]["sel"]))]["lenient"] for q in Wt.qids], float)
c3 = np.array([r3[q]["flops"] + A["q32"][(q, k(r3[q]["sel"]))]["flops"] for q in Wt.qids])
y8 = np.array([A["q32"][(q, k(r08[q]["sel"]))]["lenient"] for q in Wt.qids], float)
c8 = np.array([r08[q]["flops"] + A["q32"][(q, k(r08[q]["sel"]))]["flops"] for q in Wt.qids])


def run(Wx, q, th, start_r=None, start_a=None, escalate=True):
    """th = (ts, ta0, tr, ta): start bigger retriever if P(0.5B suffices) < ts; start with Qwen if P(Llama right) < ta0;
    retrieval / answer score thresholds tr / ta for escalation."""
    ts, ta0, tr, ta = th
    rr = Wx.route[q]
    if start_r is None:
        others = {r: rr["p_ret"][r] for r in ("r06", "r08", "r3")}
        start_r = "r05" if rr["p_ret"]["r05"] >= ts else max(others, key=others.get)
    if start_a is None:
        start_a = "l8" if rr["p_ans"]["Llama 8B"] >= ta0 else "q32"
    states = R9.ret_states(start_r) if escalate else R9.ret_states(start_r)[:1]
    for m in states:
        rs = R9.ret_state(Wx, q, m)
        kk = ("r", q, tuple(m))
        if kk not in Wx.cache:
            Wx.cache[kk] = float(cal["ret"]([rs["feats"]])[0])
        rsc = Wx.cache[kk]
        if rsc >= tr:
            break
    steps = [1, 2, 3] if escalate else [1]
    for st in steps:
        a = R9.ans_state(Wx, q, rs["sel"], start_a, st, rs)
        kk = ("a", q, tuple(rs["members"]), start_a, st)
        if kk not in Wx.cache:
            Wx.cache[kk] = float(cal["ans"]([a["feats"] + [rsc]])[0])
        asc = Wx.cache[kk]
        if asc >= ta:
            break
    return a["len"], rr["flops"] + rs["flops"] + a["flops"], start_r != "r05", start_a == "q32", asc >= ta, a["em"]


def table(Wx, th, **kw):
    return np.array([run(Wx, q, th, **kw) for q in Wx.qids], float)


def qcut(Wx, f, share):
    v = np.array([f(Wx.route[q]) for q in Wx.qids])
    return -np.inf if share == 0 else float(np.quantile(v, share))


p05 = lambda r: r["p_ret"]["r05"]
pL = lambda r: r["p_ans"]["Llama 8B"]
# ---- 1. signals: do the switch decisions carry information? (test questions)
print("\n1. SIGNALS on the 400 test questions (AUC, 0.5 = none)")
found05 = [float(Wt.gold(q) <= set(Wt.route[q]["sel"])) for q in Wt.qids]
llama_ok = [A["l8"][(q, k(Wt.route[q]["sel"]))]["lenient"] for q in Wt.qids]
print(f"   learned route P(0.5B suffices)      -> 0.5B shortlist holds both gold : {AUC(found05, [p05(Wt.route[q]) for q in Wt.qids]):.2f}")
print(f"   0.5B token probability (own output) -> same                          : {AUC(found05, [Wt.route[q]['mean_logprob'] for q in Wt.qids]):.2f}")
print(f"   calibrated retrieval score [0.5B]   -> same                          : {AUC(found05, [cal['ret']([R9.ret_state(Wt, q, ['r05'])['feats']])[0] for q in Wt.qids]):.2f}")
print(f"   learned route P(Llama right)        -> Llama right                   : {AUC(llama_ok, [pL(Wt.route[q]) for q in Wt.qids]):.2f}")
s1 = []
for q in Wt.qids:
    rs = R9.ret_state(Wt, q, ["r05"])
    rsc = float(cal["ret"]([rs["feats"]])[0])
    a = R9.ans_state(Wt, q, rs["sel"], "l8", 1, rs)
    s1.append(float(cal["ans"]([a["feats"] + [rsc]])[0]))
print(f"   calibrated answer score after Llama -> Llama right                   : {AUC(llama_ok, s1):.2f}")

# ---- 2. choose thresholds on the calibration questions: cheapest within 0.5 point of the best accuracy
SH = [0.0, 0.15, 0.3]
G = [0.7, 0.8, 0.85, 0.9, 0.95, 1.01]
cands = []
for ss, sa, tr, ta in itertools.product(SH, SH, G, G):
    th = (qcut(Wc, p05, ss), qcut(Wc, pL, sa), tr, ta)
    t = table(Wc, th)
    cands.append((t[:, 0].mean(), t[:, 1].mean(), (ss, sa, tr, ta), th))
best = max(c[0] for c in cands)
for tol in (0.005, 0.01, 0.02):
    c = min([c for c in cands if c[0] >= best - tol], key=lambda c: c[1])
    print(f"   calibration: within {100*tol:.1f} pt of best ({100*best:.1f}%): start-shares {c[2][:2]}, thresholds {c[2][2:]} -> {100*c[0]:.1f}%")
chosen = {tol: min([c for c in cands if c[0] >= best - tol], key=lambda c: c[1]) for tol in (0.005, 0.01, 0.02)}


def boot(a, b, B=4000):
    rng = np.random.default_rng(0)
    n = len(a)
    d = [a[w].mean() - b[w].mean() for w in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * np.percentile(d, 2.5), 100 * np.percentile(d, 97.5)


def show(name, t):
    lo, hi = boot(t[:, 0], y3)
    lo8, hi8 = boot(t[:, 0], y8)
    print(f"   {name:<74} {100*t[:,0].mean():5.1f}%  {100*(t[:,1].sum()/fixed_c.sum()-1):+6.1f}%  vs3B {100*(t[:,0].mean()-y3.mean()):+5.1f} ({lo:+.1f},{hi:+.1f})"
          f"  vs0.8B {100*(t[:,0].mean()-y8.mean()):+5.1f} ({lo8:+.1f},{hi8:+.1f})  bigger-retriever start {100*t[:,2].mean():3.0f}%  Qwen start {100*t[:,3].mean():3.0f}%  Sure {100*t[:,4].mean():3.0f}%")


print("\n2. TEST QUESTIONS (accuracy, compute vs cheap fixed setup)")
print(f"   {'cheap fixed setup':<74} {100*fixed_y.mean():5.1f}%  {0:+6.1f}%")
print(f"   {'0.8B retrieves -> Qwen 32B':<74} {100*y8.mean():5.1f}%  {100*(c8.sum()/fixed_c.sum()-1):+6.1f}%")
print(f"   {'3B retrieves -> Qwen 32B':<74} {100*y3.mean():5.1f}%  {100*(c3.sum()/fixed_c.sum()-1):+6.1f}%")
rng = np.random.default_rng(5)
for tol, c in chosen.items():
    th = c[3]
    t = table(Wt, th)
    show(f"FIXED cascade, thresholds from calibration (within {100*tol:.1f} pt)", t)
    tn = table(Wt, th, start_r="r05", start_a="l8")
    show("   same thresholds, no learned start (always 0.5B, Llama)", tn)
    # random start decisions with the same counts
    rr = []
    for rep in range(8):
        perm_r = rng.permutation(t[:, 2].astype(bool))
        perm_a = rng.permutation(t[:, 3].astype(bool))
        rows = []
        for i, q in enumerate(Wt.qids):
            others = {r: Wt.route[q]["p_ret"][r] for r in ("r06", "r08", "r3")}
            sr = max(others, key=others.get) if perm_r[i] else "r05"
            rows.append(run(Wt, q, th, start_r=sr, start_a="q32" if perm_a[i] else "l8"))
        rr.append(np.array(rows, float))
    show("   same thresholds, start decisions at random (same counts, 8 draws)", np.mean(rr, 0))

# ---- 3. the simplest switch: Llama answers on the 0.5B shortlist; Qwen replaces it when the calibrated score is low
print("\n3. ONE SWITCH: 0.5B retrieves, Llama answers; low calibrated score -> Qwen 32B answers instead (threshold set on calibration)")
def one(Wx, q, thr, rand=None):
    rs = R9.ret_state(Wx, q, ["r05"])
    rsc = float(cal["ret"]([rs["feats"]])[0])
    a = R9.ans_state(Wx, q, rs["sel"], "l8", 1, rs)
    sc = float(cal["ans"]([a["feats"] + [rsc]])[0])
    sw = (sc < thr) if rand is None else rand
    aq = A["q32"][(q, rs["sel"])]
    if sw:
        return aq["lenient"], Wx.route[q]["flops"] + a["flops"] + aq["flops"], 1.0, sc
    return a["len"], Wx.route[q]["flops"] + a["flops"], 0.0, sc
for share in (0.2, 0.4, 0.6):
    sc_c = np.array([one(Wc, q, -1)[3] for q in Wc.qids])
    thr = float(np.quantile(sc_c, share))
    t = np.array([one(Wt, q, thr) for q in Wt.qids], float)
    rnd = np.mean([np.array([one(Wt, q, thr, rand=b)[0] for q, b in zip(Wt.qids, rng.permutation(t[:, 2].astype(bool)))]) for _ in range(50)], 0)
    lo, hi = boot(t[:, 0], rnd)
    print(f"   switch ~{int(100*share)}% (test {100*t[:,2].mean():.0f}%): {100*t[:,0].mean():.1f}% at {100*(t[:,1].sum()/fixed_c.sum()-1):+.1f}%   "
          f"random switching same share {100*rnd.mean():.1f}%  -> {100*(t[:,0].mean()-rnd.mean()):+.1f} ({lo:+.1f},{hi:+.1f})")
ql = np.array([A["q32"][(q, k(Wt.route[q]['sel']))]["lenient"] for q in Wt.qids])
print(f"   (Llama alone {100*np.mean(llama_ok):.1f}%, Qwen alone on the same shortlists {100*ql.mean():.1f}%)")
