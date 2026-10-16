"""Set 8 - cascades of small fine-tuned retrievers and of answerers, with agreement and tie-break votes (notebook 26).

The retrievers are the LoRA-trained models of notebooks 23a / 25a / 25b / 25c (Qwen 3B, Qwen3 0.6B, Qwen3.5 0.8B, Qwen2.5 0.5B).
The answerers are unchanged (Llama 8B, Qwen 32B; one greedy step-by-step answer; Qwen 32B also gives 4 sampled answers for votes).

The chain asked for: Qwen2.5 0.5B shortlists; when it is unsure (low token probability) Qwen3 0.6B also shortlists - agreement is
accepted; when they differ Qwen3.5 0.8B breaks the tie (no majority -> union of the three). Then Llama 8B answers; when it is unsure
Qwen 32B answers too - agreement is accepted; when they differ, 4 sampled Qwen 32B answers join a majority vote.

Commands:
  shortlists  every shortlist a cascade can end on (each retriever, and the union of any group of them) -> step/cascade_shortlists.jsonl;
              the 0.5B shortlists and the chain's shortlists -> step/sample_shortlists.jsonl (Qwen 32B samples are made there);
              the cheap fixed setup's shortlists (Qwen 7B retrieves) -> step/ref_shortlists.jsonl
  analyze     the policies, thresholds fitted inside training folds (5-fold cross-validation, repeated), compute charged for every model run

Nothing here looks at the answer key except to score: escalation decisions use token probabilities, stated confidence and agreement only.
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S               # noqa: E402
import sweep as W             # noqa: E402
import retriever_check as RC  # noqa: E402

VERSION = "8.0"
log = S.log
RETS = ["r05", "r06", "r08", "r3"]            # cheapest first
RLAB = {"r05": "Qwen2.5 0.5B", "r06": "Qwen3 0.6B", "r08": "Qwen3.5 0.8B", "r3": "Qwen2.5 3B"}
TOL = 0.005                                   # fitted thresholds: the cheapest setting within 0.5 point of the best training accuracy
N_SAMPLES_VOTE = 4
KEY_ROWS = ("Cheap fixed setup", "Qwen2.5 3B alone -> Qwen", "Qwen2.5 0.5B alone -> Qwen", "Qwen2.5 0.5B alone -> Llama",
            "0.5B -> 0.6B -> 0.8B chain, always entered (agree -> accept; 0.8B breaks ties; else union) -> Qwen", "CHAIN (retrieval)",
            "FULL CASCADE", "   random gate, same share: FULL CASCADE", "[0.5B->0.6B->0.8B always] Oracle")


def sets(args):
    return [("old", args.old_sweep, args.old_data), ("fresh", args.fresh_run, args.fresh_data)]


def key(sel):
    return tuple(sorted(int(i) for i in sel))


def write(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")


def load_rets(args):
    R = {}
    for r in RETS:
        R[r] = {x["qid"]: x for x in S.read_jsonl(os.path.join(args.run_dir, "step", f"retrieval_{r}.jsonl"))}
    return R


def ref_rows(args, models):
    """The cheap fixed setup's retrieval (Qwen 7B, one step-by-step try, as AgentTTS picked)."""
    out = {}
    for _, run, d in sets(args):
        rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RC.RET, RC.COT, RC.PICK)))}
        for q in S.load_questions(d):
            row = rets[q["qid"]]
            sel = W.shortlists(row, RC.COT, len(q["ctxs_candidate"]))[(1, RC.MERGE)]
            g = {i + 1 for i in q["ret_answers"]}
            out[q["qid"]] = {"sel": key(sel), "flops": 2 * models[RC.RET]["params"] * (row["prompt_tokens"] + row["samples"][0]["out_tokens"]),
                             "found_both": float(g <= set(sel))}
    return out


def chain(R, q):
    """0.5B, then 0.6B: they agree -> that shortlist. Else 0.8B: a majority of the three -> it; all differ -> union of the three.
    Returns (shortlist, retrieval flops, retrievers run) when the chain is entered (the 0.5B model was unsure)."""
    s = {r: key(R[r][q]["sel"]) for r in RETS}
    f = {r: R[r][q]["flops"] for r in RETS}
    if s["r05"] == s["r06"]:
        return s["r05"], f["r05"] + f["r06"], 2
    three = [s["r05"], s["r06"], s["r08"]]
    maj = [x for x in three if three.count(x) >= 2]
    return (maj[0] if maj else key(set().union(*three))), f["r05"] + f["r06"] + f["r08"], 3


def cmd_shortlists(args):
    R = load_rets(args)
    qids = [q["qid"] for _, _, d in sets(args) for q in S.load_questions(d)]
    allr, smp = [], []
    for qid in qids:
        seen = set()
        for grp in [g for k in (1, 2, 3) for g in itertools.combinations(RETS[:3], k)] + [("r3",)]:   # 3B: a reference only
            if True:
                s = key(set().union(*[R[r][qid]["sel"] for r in grp]))
                if s not in seen:
                    seen.add(s)
                    allr.append({"qid": qid, "sel": list(s)})
        for s in {key(R["r05"][qid]["sel"]), chain(R, qid)[0]}:
            smp.append({"qid": qid, "sel": list(s)})
    models = W.load_models(args.models_json)
    ref = ref_rows(args, models)
    write(os.path.join(args.run_dir, "step", "cascade_shortlists.jsonl"), allr)
    write(os.path.join(args.run_dir, "step", "sample_shortlists.jsonl"), smp)
    write(os.path.join(args.run_dir, "step", "ref_shortlists.jsonl"), [{"qid": q, "sel": list(ref[q]["sel"])} for q in qids])
    log(f"{len(allr)} cascade shortlists ({len(allr) / len(qids):.2f} per question), {len(smp)} for samples, {len(qids)} reference")


# ==================================================================================
# answers, agreement and votes
# ==================================================================================
def agree(a, b):
    na, nb = S.normalize_answer(a or ""), S.normalize_answer(b or "")
    if not na or not nb:
        return False
    return na == nb or S.em_f1(a, b)[1] >= 0.8


def vote(votes):
    """Majority by agreement; ties go to the earlier vote (the order says who is preferred)."""
    sup = [sum(agree(v["prediction"], w["prediction"]) or i == j for j, w in enumerate(votes)) for i, v in enumerate(votes)]
    best = max(range(len(votes)), key=lambda i: (sup[i], -i))
    return votes[best], sup[best]


class Data:
    def __init__(self, args):
        self.models = W.load_models(args.models_json)
        self.Q = [q for _, _, d in sets(args) for q in S.load_questions(d)]
        self.qids = [q["qid"] for q in self.Q]
        self.origin = np.array([nm for nm, _, d in sets(args) for _ in S.load_questions(d)])
        self.R = load_rets(args)
        self.ref = ref_rows(args, self.models)
        self.A = {}
        for m in ("l8", "q32"):
            d = {}
            for path in [os.path.join(args.run_dir, "step", f"answers_{m}.jsonl")] + [os.path.join(x, "step", f"answers_{m}.jsonl") for x in args.ref_answers]:
                for r in S.read_jsonl(path):
                    k = (r["qid"], key(r["sel"]))
                    if k not in d or (d[k].get("mean_logprob") is None and r.get("mean_logprob") is not None):
                        d[k] = r
            self.A[m] = d
        self.SM = {(r["qid"], key(r["sel"])): r for r in S.read_jsonl(os.path.join(args.run_dir, "step", "samples_q32.jsonl"))}
        self.gold = {q["qid"]: {i + 1 for i in q["ret_answers"]} for q in self.Q}

    def ans(self, m, qid, sel):
        return self.A[m][(qid, key(sel))]

    def samples(self, qid, sel):
        r = self.SM[(qid, key(sel))]
        per = r["flops"] / max(1, len(r["samples"]))
        return r["samples"][:N_SAMPLES_VOTE], per * min(N_SAMPLES_VOTE, len(r["samples"]))


def out(v, flops, esc=0.0):
    return (float(v["lenient"]), float(v["EM"]), float(flops), float(esc))


# ---------------------------------------------------------------------------- retrieval policies (no fitting)
def retrieval_policies(D):
    """name -> {qid: (shortlist, retrieval flops, retrievers run)}"""
    P = {}
    for r in RETS:
        P[f"{RLAB[r]} alone"] = {q: (key(D.R[r][q]["sel"]), D.R[r][q]["flops"], 1) for q in D.qids}
    pol_v, pol_all = {}, {}
    for q in D.qids:
        pol_v[q] = chain(D.R, q)
        pol_all[q] = (key(set().union(*[D.R[r][q]["sel"] for r in ("r05", "r06", "r08")])), sum(D.R[r][q]["flops"] for r in ("r05", "r06", "r08")), 3)
    P["0.5B -> 0.6B -> 0.8B chain, always entered (agree -> accept; 0.8B breaks ties; else union)"] = pol_v
    P["0.5B + 0.6B + 0.8B, union of all three"] = pol_all
    return P


# ---------------------------------------------------------------------------- gated policies: (no-escalation outcome, escalation outcome, scores)
def gated(D, base_sel, esc_fn, base_ret_flops):
    """Llama answers on base_sel; esc_fn(q) gives the escalated outcome (its extra compute included)."""
    no, es = [], []
    for q in D.qids:
        aL = D.ans("l8", q, base_sel[q])
        no.append(out(aL, base_ret_flops[q] + aL["flops"]))
        es.append(esc_fn(q))
    return np.array(no), np.array(es)


def both_then_vote(D, q, sel, ret_flops, llama_run=True):
    aL, aQ = D.ans("l8", q, sel), D.ans("q32", q, sel)
    c = ret_flops + aQ["flops"] + (aL["flops"] if llama_run else 0.0)
    if agree(aL["prediction"], aQ["prediction"]):
        return out(aQ, c, 1.0)
    ss, cs = D.samples(q, sel)
    v, _ = vote([aQ, aL] + ss)
    return out(v, c + cs, 1.0)


def score_of(r, what):
    if what == "logprob":
        x = r.get("mean_logprob")
        return -9.0 if x is None else float(x)
    x = W.stated_conf(r.get("text", "")) if r.get("stated_conf") is None else r.get("stated_conf")
    return 0.5 if x is None else float(x)


# ---------------------------------------------------------------------------- fitting thresholds in folds
# A gate: outs[c] = per-question outcome (lenient, EM, flops, escalated) for outcome c; decide(sig, th) -> c per question.
class Gate:
    def __init__(self, outs, sig, decide, qcols):
        self.outs, self.sig, self.decide, self.qcols = np.asarray(outs, float), np.asarray(sig, float), decide, qcols

    def apply(self, idx, th, c=None):
        c = self.decide(self.sig[idx], th) if c is None else c
        r = self.outs[c, idx].copy()
        r[:, 3] = c > 0
        return r, c


def one_gate(no, es, sc):
    return Gate([no, es], np.asarray(sc, float).reshape(-1, 1), lambda s, th: (s[:, 0] < th[0]).astype(int), [0])


def two_stage(outs4, sig3):
    """Stage 1 on sig[:,0]; stage 2 on sig[:,1] (stage 1 not escalated) or sig[:,2] (stage 1 escalated). outcome = e1 + 2 * e2."""
    def decide(s, th):
        e1 = s[:, 0] < th[0]
        e2 = np.where(e1, s[:, 2], s[:, 1]) < th[1]
        return e1.astype(int) + 2 * e2.astype(int)
    return Gate(outs4, sig3, decide, [0, 1])


def grid_for(k):
    return [np.round(np.arange(0, 1.0001, 0.025), 3)] if k == 1 else [np.round(np.arange(0, 1.0001, 0.05), 2)] * k


def fit(G, idx):
    """Thresholds as shares of the training questions (per signal); the cheapest within TOL of the best training accuracy."""
    cand = []
    for shares in itertools.product(*grid_for(len(G.qcols))):
        th = [np.quantile(G.sig[idx, j], sh) if 0 < sh < 1 else (-np.inf if sh == 0 else np.inf) for j, sh in zip(G.qcols, shares)]
        r, _ = G.apply(idx, th)
        cand.append((r[:, 0].mean(), r[:, 2].mean(), th))
    best = max(c[0] for c in cand)
    return min([c for c in cand if c[0] >= best - TOL], key=lambda c: c[1])[2]


def cv(G, repeats, seed=0):
    n = G.outs.shape[1]
    acc, rnd = np.zeros((repeats, n, 4)), np.zeros((repeats, n, 4))
    for rep in range(repeats):
        rng = np.random.default_rng(seed + rep)
        perm = rng.permutation(n)
        for f in range(5):
            te = np.sort(perm[f::5])
            tr = np.setdiff1d(perm, te)
            r, c = G.apply(te, fit(G, tr))
            acc[rep, te] = r
            rnd[rep, te] = G.apply(te, None, c=rng.permutation(c))[0]      # random gate: the same escalation counts, shuffled
    return acc.mean(0), rnd.mean(0)


def transfer(G, origin):
    res = {}
    for a, b in (("old", "fresh"), ("fresh", "old")):
        tr, te = np.where(origin == a)[0], np.where(origin == b)[0]
        res[f"{a} -> {b}"] = (te, G.apply(te, fit(G, tr))[0])
    return res


def boot(a, b, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    d = [a[w].mean() - b[w].mean() for w in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))


# ==================================================================================
def cmd_analyze(args):
    import pandas as pd
    D = Data(args)
    n = len(D.qids)
    res_dir = os.path.join(args.run_dir, "results")
    os.makedirs(os.path.join(res_dir, "charts"), exist_ok=True)
    fixed = np.array([out(D.ans("q32", q, D.ref[q]["sel"]), D.ref[q]["flops"] + D.ans("q32", q, D.ref[q]["sel"])["flops"]) for q in D.qids])
    RP = retrieval_policies(D)
    rows_r = []
    pipes = {}                      # name -> per-question (lenient, EM, flops, escalated)
    for name, P in RP.items():
        for m in ("q32", "l8"):
            pipes[f"{name} -> {W.LABEL.get(m, m)}"] = np.array([out(D.ans(m, q, P[q][0]), P[q][1] + D.ans(m, q, P[q][0])["flops"]) for q in D.qids])
        rows_r.append({"Retrieval": name, "Both gold passages found": float(np.mean([float(D.gold[q] <= set(P[q][0])) for q in D.qids])),
                       "Mean shortlist size": float(np.mean([len(P[q][0]) for q in D.qids])), "Mean retrievers run": float(np.mean([P[q][2] for q in D.qids])),
                       "Retrieval TFLOPs per question": float(np.mean([P[q][1] for q in D.qids]) / 1e12)})
    rows_r.append({"Retrieval": "Qwen 7B, step-by-step (the cheap fixed setup)", "Both gold passages found": float(np.mean([D.ref[q]["found_both"] for q in D.qids])),
                   "Mean shortlist size": float(np.mean([len(D.ref[q]["sel"]) for q in D.qids])), "Mean retrievers run": 1.0,
                   "Retrieval TFLOPs per question": float(np.mean([D.ref[q]["flops"] for q in D.qids]) / 1e12)})
    s05 = {q: key(D.R["r05"][q]["sel"]) for q in D.qids}
    f05 = {q: D.R["r05"][q]["flops"] for q in D.qids}
    ch = {q: chain(D.R, q) for q in D.qids}
    rsig = np.array([score_of(D.R["r05"][q], "logprob") for q in D.qids])
    fitted = {}
    # (A) retrieval chain only, gated by the 0.5B model's token probability; Qwen 32B answers (fitted on final accuracy)
    no = np.array([out(D.ans("q32", q, s05[q]), f05[q] + D.ans("q32", q, s05[q])["flops"]) for q in D.qids])
    es = np.array([out(D.ans("q32", q, ch[q][0]), ch[q][1] + D.ans("q32", q, ch[q][0])["flops"], 1.0) for q in D.qids])
    fitted["CHAIN (retrieval): 0.5B; unsure -> 0.6B, then 0.8B on disagreement -> Qwen 32B answers"] = one_gate(no, es, rsig)
    no = np.array([out(D.ans("l8", q, s05[q]), f05[q] + D.ans("l8", q, s05[q])["flops"]) for q in D.qids])
    es = np.array([out(D.ans("l8", q, ch[q][0]), ch[q][1] + D.ans("l8", q, ch[q][0])["flops"], 1.0) for q in D.qids])
    fitted["CHAIN (retrieval): 0.5B; unsure -> 0.6B, then 0.8B on disagreement -> Llama 8B answers"] = one_gate(no, es, rsig)

    # (B) answer cascades on fixed retrieval: 0.5B alone, and the retrieval chain always entered
    for rname, RS in (("0.5B", {q: (s05[q], f05[q]) for q in D.qids}), ("0.5B->0.6B->0.8B always", {q: ch[q][:2] for q in D.qids})):
        sel = {q: RS[q][0] for q in D.qids}
        rf = {q: RS[q][1] for q in D.qids}
        pipes[f"[{rname}] Qwen 32B, greedy + 4 samples, vote"] = np.array([out(vote([D.ans("q32", q, sel[q])] + D.samples(q, sel[q])[0])[0],
                                                                            rf[q] + D.ans("q32", q, sel[q])["flops"] + D.samples(q, sel[q])[1]) for q in D.qids])
        pipes[f"[{rname}] Llama and Qwen always; agree -> accept, else Qwen samples vote"] = np.array([both_then_vote(D, q, sel[q], rf[q]) for q in D.qids])
        orc = []
        for q in D.qids:
            aL, aQ = D.ans("l8", q, sel[q]), D.ans("q32", q, sel[q])
            orc.append((max(aL["lenient"], aQ["lenient"]), max(aL["EM"], aQ["EM"]), rf[q] + aL["flops"] + (aQ["flops"] if not aL["lenient"] else 0), 0))
        pipes[f"[{rname}] Oracle: Llama if right, else Qwen (needs the answer key)"] = np.array(orc, float)
        for sig in ("logprob", "stated"):
            sc = np.array([score_of(D.ans("l8", q, sel[q]), sig) for q in D.qids])
            sigl = "token probability" if sig == "logprob" else "stated confidence"
            no, es = gated(D, sel, lambda q: out(D.ans("q32", q, sel[q]), rf[q] + D.ans("l8", q, sel[q])["flops"] + D.ans("q32", q, sel[q])["flops"], 1.0), rf)
            fitted[f"[{rname}] Llama; unsure ({sigl}) -> Qwen's answer replaces it"] = one_gate(no, es, sc)
            no, es = gated(D, sel, lambda q: both_then_vote(D, q, sel[q], rf[q]), rf)
            fitted[f"[{rname}] Llama; unsure ({sigl}) -> Qwen; agree -> accept, else Qwen samples vote"] = one_gate(no, es, sc)

    # (C) THE FULL CASCADE: retrieval chain gated by the 0.5B token probability, then the answer cascade gated by Llama's token probability
    outs4 = np.zeros((4, n, 4))
    for i, q in enumerate(D.qids):
        for e1 in (0, 1):
            sel, rf = (ch[q][0], ch[q][1]) if e1 else (s05[q], f05[q])
            aL = D.ans("l8", q, sel)
            outs4[e1, i] = out(aL, rf + aL["flops"], e1)
            outs4[e1 + 2, i] = both_then_vote(D, q, sel, rf)
    sig3 = np.array([[rsig[i], score_of(D.ans("l8", q, s05[q]), "logprob"), score_of(D.ans("l8", q, ch[q][0]), "logprob")] for i, q in enumerate(D.qids)])
    fitted["FULL CASCADE: 0.5B; unsure -> 0.6B -> 0.8B; Llama answers; unsure -> Qwen; agree -> accept, else Qwen samples vote"] = two_stage(outs4, sig3)
    pipes["FULL CASCADE, every step always run"] = outs4[3].copy()

    rows_a = []
    ref3 = pipes[f"{RLAB['r3']} alone -> {W.LABEL.get('q32', 'q32')}"]

    def add(name, r, kind, rnd=None, sub=None):
        idx = np.arange(n) if sub is None else sub
        f, b3 = fixed[idx], ref3[idx]
        row = {"Policy": name, "Kind": kind, "Questions": "all 400" if sub is None else "", "Accuracy (corrected label)": float(r[:, 0].mean()),
               "Exact match": float(r[:, 1].mean()), "Escalated": float(r[:, 3].mean()), "TFLOPs per question": float(r[:, 2].mean() / 1e12),
               "Compute vs the cheap fixed setup": float(r[:, 2].sum() / f[:, 2].sum() - 1)}
        lo, hi = boot(r[:, 0], f[:, 0])
        row |= {"vs cheap fixed setup (points)": 100 * float(r[:, 0].mean() - f[:, 0].mean()), "low": lo, "high": hi}
        lo, hi = boot(r[:, 0], b3[:, 0])
        row |= {"vs 3B fine-tuned -> Qwen 32B (points)": 100 * float(r[:, 0].mean() - b3[:, 0].mean()), "low ": lo, "high ": hi}
        if rnd is not None:
            lo, hi = boot(r[:, 0], rnd[:, 0])
            row |= {"vs random gate, same share (points)": 100 * float(r[:, 0].mean() - rnd[:, 0].mean()), "low  ": lo, "high  ": hi}
        rows_a.append(row)

    add("Cheap fixed setup: Qwen 7B retrieves, Qwen 32B answers (AgentTTS's pick)", fixed, "fixed")
    for k, v in pipes.items():
        add(k, v, "oracle" if "Oracle" in k else "fixed / always")
    rows_t = []
    for k, G in fitted.items():
        a, rnd = cv(G, args.repeats)
        add(k, a, "fitted in folds", rnd)
        add("   random gate, same share: " + k, rnd, "random gate")
        for tname, (te, r) in transfer(G, D.origin).items():
            rows_t.append({"Policy": k, "Fit -> test": tname, "Accuracy (corrected label)": float(r[:, 0].mean()), "Exact match": float(r[:, 1].mean()),
                           "Escalated": float(r[:, 3].mean()), "Cheap fixed setup on the same questions": float(fixed[te, 0].mean()),
                           "3B fine-tuned -> Qwen 32B on the same questions": float(ref3[te, 0].mean()),
                           "Compute vs the cheap fixed setup": float(r[:, 2].sum() / fixed[te, 2].sum() - 1)})
    # how often the answerers agree, and how right they are then
    rows_g = []
    for rname, sel in (("0.5B", s05), ("0.5B->0.6B->0.8B chain", {q: ch[q][0] for q in D.qids})):
        ag = np.array([agree(D.ans("l8", q, sel[q])["prediction"], D.ans("q32", q, sel[q])["prediction"]) for q in D.qids])
        yq = np.array([D.ans("q32", q, sel[q])["lenient"] for q in D.qids], float)
        yl = np.array([D.ans("l8", q, sel[q])["lenient"] for q in D.qids], float)
        rows_g.append({"Shortlists": rname, "Llama and Qwen agree": float(ag.mean()), "Qwen right when they agree": float(yq[ag].mean()) if ag.any() else None,
                       "Qwen right when they disagree": float(yq[~ag].mean()) if (~ag).any() else None, "Llama right when they disagree": float(yl[~ag].mean()) if (~ag).any() else None})
    # signal quality: how well each confidence signal separates right from wrong (AUC)
    rows_s = []
    from sklearn.metrics import roc_auc_score
    for lab_, vals, y in (("0.5B retriever token probability -> both gold passages found", [score_of(D.R["r05"][q], "logprob") for q in D.qids],
                           [float(D.gold[q] <= set(s05[q])) for q in D.qids]),
                          ("Llama token probability -> Llama right", [score_of(D.ans("l8", q, s05[q]), "logprob") for q in D.qids], [D.ans("l8", q, s05[q])["lenient"] for q in D.qids]),
                          ("Llama stated confidence -> Llama right", [score_of(D.ans("l8", q, s05[q]), "stated") for q in D.qids], [D.ans("l8", q, s05[q])["lenient"] for q in D.qids]),
                          ("Qwen token probability -> Qwen right", [score_of(D.ans("q32", q, s05[q]), "logprob") for q in D.qids], [D.ans("q32", q, s05[q])["lenient"] for q in D.qids])):
        y = np.array(y, float)
        rows_s.append({"Signal (on 0.5B shortlists)": lab_, "Separation (AUC; 0.5 = none)": float(roc_auc_score(y, vals)) if 0 < y.mean() < 1 else None,
                       "Base rate": float(y.mean())})
    def df(rr):
        d = pd.DataFrame(rr)
        for c in d.columns:
            if d[c].isna().any():                      # empty cells (e.g. the random-gate columns of fixed policies) shown as '-'
                d[c] = d[c].astype(object).where(d[c].notna(), "-")
        return d
    W.write_excel(os.path.join(res_dir, "set8_cascades.xlsx"), {"Policies": df(rows_a), "Retrieval": df(rows_r), "Transfer old-fresh": df(rows_t),
                                                               "Agreement": df(rows_g), "Signals": df(rows_s)})
    for nm, rr in (("policies.jsonl", rows_a), ("retrieval.jsonl", rows_r), ("transfer.jsonl", rows_t), ("agreement.jsonl", rows_g), ("signals.jsonl", rows_s)):
        write(os.path.join(res_dir, nm), rr)
    # chart: accuracy against compute
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6.5))
    for r in rows_a:
        if r["Kind"] == "random gate":
            continue
        c = {"fixed": "black", "fixed / always": "tab:blue", "oracle": "tab:gray", "fitted in folds": "tab:red"}[r["Kind"]]
        ax.scatter(100 * (1 + r["Compute vs the cheap fixed setup"]), 100 * r["Accuracy (corrected label)"], color=c, s=28)
        if r["Kind"] != "oracle" and ("CASCADE" in r["Policy"] or "CHAIN" in r["Policy"] or "fixed setup" in r["Policy"] or "3B alone -> Qwen" in r["Policy"] or "agree" in r["Policy"]):
            ax.annotate(r["Policy"][:48], (100 * (1 + r["Compute vs the cheap fixed setup"]), 100 * r["Accuracy (corrected label)"]), fontsize=6.5)
    ax.set_xlabel("Compute, % of the cheap fixed setup")
    ax.set_ylabel("Accuracy, corrected label (%)")
    ax.set_title("Set 8: retrieval and answer cascades (red = thresholds fitted in folds; blue = fixed; grey = oracle)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(res_dir, "charts", "accuracy_vs_compute.png"), dpi=150)
    # summary
    T = lambda rr, cols, pct=(), sg=(): ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join(("-" if r.get(c) is None else (r[c] if isinstance(r[c], str) else (f"{100 * r[c]:.1f}%" if c in pct else (f"{r[c]:+.1f}" if c in sg else f"{r[c]:.2f}"))))
                          for c in cols) + " |" for r in rr))
    L = [f"# Set 8: cascades of small fine-tuned retrievers and answerers\n",
         f"Code version {VERSION}. 400 test questions. Thresholds fitted inside training folds (5-fold cross-validation x {args.repeats}); "
         f"the cheapest setting within {100 * TOL:.1f} point of the best training accuracy. Compute counts every model that runs.\n",
         "## Headline\n",
         T([r for r in rows_a if any(r["Policy"].startswith(h) for h in KEY_ROWS)],
           ["Policy", "Accuracy (corrected label)", "Exact match", "Escalated", "Compute vs the cheap fixed setup", "vs cheap fixed setup (points)", "low", "high",
            "vs 3B fine-tuned -> Qwen 32B (points)", "low ", "high "],
           pct={"Accuracy (corrected label)", "Exact match", "Escalated", "Compute vs the cheap fixed setup"},
           sg={"vs cheap fixed setup (points)", "low", "high", "vs 3B fine-tuned -> Qwen 32B (points)", "low ", "high "}), "",
         "## Retrieval\n", T(rows_r, list(rows_r[0]), pct={"Both gold passages found"}), "",
         "## Final accuracy, all policies\n",
         T(rows_a, ["Policy", "Kind", "Accuracy (corrected label)", "Exact match", "Escalated", "Compute vs the cheap fixed setup", "vs cheap fixed setup (points)", "low", "high",
                    "vs 3B fine-tuned -> Qwen 32B (points)", "low ", "high "],
           pct={"Accuracy (corrected label)", "Exact match", "Escalated", "Compute vs the cheap fixed setup"},
           sg={"vs cheap fixed setup (points)", "low", "high", "vs 3B fine-tuned -> Qwen 32B (points)", "low ", "high "}), "",
         "## Fitted policies against a random gate escalating the same share\n",
         T([r for r in rows_a if r["Kind"] == "fitted in folds"], ["Policy", "Escalated", "vs random gate, same share (points)", "low  ", "high  "],
           pct={"Escalated"}, sg={"vs random gate, same share (points)", "low  ", "high  "}), "",
         "## Fit on one half, test on the other\n",
         T(rows_t, list(rows_t[0]), pct={c for c in rows_t[0] if c not in ("Policy", "Fit -> test")}), "",
         "## Agreement between Llama 8B and Qwen 32B\n", T(rows_g, list(rows_g[0]), pct={c for c in rows_g[0] if c != "Shortlists"}), "",
         "## Confidence signals\n", T(rows_s, list(rows_s[0]), pct={"Base rate"}), ""]
    open(os.path.join(res_dir, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log(f"results -> {res_dir}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("shortlists", "analyze"):
        p = sub.add_parser(c)
        for a in ("old_sweep", "old_data", "fresh_run", "fresh_data", "run_dir", "models_json"):
            p.add_argument("--" + a, required=True)
        p.add_argument("--ref_answers", nargs="*", default=[])
        p.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    {"shortlists": cmd_shortlists, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
