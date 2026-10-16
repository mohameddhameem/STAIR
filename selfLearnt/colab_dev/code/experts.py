"""Set 5g - the ideas from Routing_Experiment_Ideas.docx, tested on this task (notebook 20, CPU only).

Step 0  Scoring audit. Exact match marks 'Gardnerville, Nevada' wrong when the gold answer is 'Gardnerville'. Every idea is therefore
        scored twice: with exact match (as before) and with a corrected label, 'substantially right' = token F1 of at least 0.5, or the
        gold answer contained in the model's answer.
Ideas   1 five-voter lexical gate      2 rare routing margins      3 benefit-targeted probes (helps / hurts / expected gain)
        4 mixture of experts (MoE): fixed weights against question-dependent weights
        5 bundle of experts (BoE): the same experts, every output kept, joint model with named interactions
        6 adaptive weighting and sharpening (per-expert temperatures)
Sequence-aware transitions (bridge retrieval) need new model runs and are not in this notebook.

Two decisions, as in notebook 19: keep the Llama 8B answer or ask the Qwen 32B (answerer); stop after one retrieval try or try twice
more (retriever). Three evaluations: old -> fresh, fresh -> old, all 400 cross-validated. Every idea is compared with one fixed
hidden-state probe, with a random gate at the same budget, and with the cheap fixed setup. One variant per idea was named beforehand.
No model is run: everything is read from notebooks 11, 16, 17 and 18.
"""
import argparse
import copy
import json
import os
import re
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S               # noqa: E402
import sweep as W             # noqa: E402
import hidden as H            # noqa: E402
import final as F             # noqa: E402
import retriever_check as RC  # noqa: E402
import ensemble as E          # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "5g.1"
log = H.log
FIXED = E.FIXED
VOTERS = ["TF-IDF cosine", "Jaccard", "rare-token overlap", "Dice", "overlap coefficient"]
ORIG_THR = np.array([0.05, 0.05, 0.15, 0.07, 0.15])           # the thresholds of the earlier project
IDEAS = {  # idea -> the variant named beforehand as the one that decides
    "1 Five-voter gate": "Five voters: majority vote, original thresholds",
    "2 Rare routing margins": "Rare margins: rarity, margin and their interaction",
    "3 Benefit-targeted probes": "Benefit probe: expected gain (helps minus hurts)",
    "4 Mixture of experts (MoE)": "MoE: question-dependent weights (learned gate)",
    "5 Bundle of experts (BoE)": "BoE: joint model on every output with named interactions",
    "6 Adaptive weighting and sharpening": "Sharpening: per-expert temperatures and weights",
}
PLUS = "Fixed probe + "
_STOP = None


def toks(t):
    global _STOP
    if _STOP is None:
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        _STOP = set(ENGLISH_STOP_WORDS)
    return [w for w in re.findall(r"[a-z0-9]+", t.lower()) if w not in _STOP and len(w) > 1]


def ents(t):
    """Crude entity words: capitalised words of 4+ letters that do not start the text."""
    return set(w.lower() for w in re.findall(r"(?<!^)(?<![.!?]\s)\b[A-Z][a-z]{3,}\b", t))


def lenient(pred, gold):
    """Substantially right: exact match, or token F1 >= 0.5, or the gold answer contained in the answer. Yes/no must match exactly."""
    em, f1 = S.em_f1(pred, gold)
    if em:
        return 1.0
    p, g = S.normalize_answer(pred), S.normalize_answer(gold)
    if g in ("yes", "no", "noanswer") or p in ("yes", "no", "noanswer", ""):
        return 0.0
    return float(f1 >= 0.5 or (" " + g + " ") in (" " + p + " "))


def load_set(name, sweep_run, data_dir, hidden_run):
    s = E.load_set(name, sweep_run, data_dir, os.path.join(sweep_run, "models.json"), hidden_run)
    rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(sweep_run, W.vid(RC.RET, RC.COT, RC.PICK)))}
    answ = {(r["qid"], tuple(r["sel"])): r for r in S.read_jsonl(W.ans_path(sweep_run, RC.BIG, RC.COT))}
    sel1, L, rows = [], {k: [] for k in ("e1", "e3", "l8", "f1e1", "f1l8")}, []
    for i, q in enumerate(s["qs"]):
        sl = W.shortlists(rets[q["qid"]], RC.COT, len(q["ctxs_candidate"]))
        s1, s3 = sl[(1, RC.MERGE)], sl[(3, RC.MERGE)]
        sel1.append([j - 1 for j in s1])
        p1, p3 = answ[(q["qid"], s1)]["fused"]["1"]["prediction"], answ[(q["qid"], s3)]["fused"]["1"]["prediction"]
        pl = s["Ea"]["rows"][i]["prediction"]
        L["e1"].append(lenient(p1, q["answer"])); L["e3"].append(lenient(p3, q["answer"])); L["l8"].append(lenient(pl, q["answer"]))
        L["f1e1"].append(S.em_f1(p1, q["answer"])[1]); L["f1l8"].append(S.em_f1(pl, q["answer"])[1])
        rows.append({"set": name, "qid": q["qid"], "question": q["question"], "gold": q["answer"], "Qwen 32B answer": p1, "Llama 8B answer": pl,
                     "Qwen 32B: exact match": int(s["A"]["e1"][i]), "Qwen 32B: F1": round(L["f1e1"][-1], 3), "Qwen 32B: substantially right": int(L["e1"][-1]),
                     "Llama 8B: exact match": int(s["l8"][i]), "Llama 8B: F1": round(L["f1l8"][-1], 3), "Llama 8B: substantially right": int(L["l8"][-1]),
                     "first try found both passages": int(s["A"]["f1"][i])})
    s["sel1"], s["len"], s["label_rows"] = sel1, {k: np.array(v, float) for k, v in L.items()}, rows
    return s


# ==================================================================================
# Features of the ideas (everything that is fitted uses the training rows only)
# ==================================================================================
class Feats:
    def __init__(self, P, tr):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.P = P
        n = P.n
        self.vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit([p for i in tr for p in P.passages[i]] + [P.text[i] for i in tr])
        docs = [set(toks(p)) for i in tr for p in P.passages[i]]
        N = max(1, len(docs))
        df = {}
        for d in docs:
            for w in d:
                df[w] = df.get(w, 0) + 1
        self.idf = lambda w: float(np.log((N + 1) / (df.get(w, 0) + 1)) + 1)
        rare = lambda w: df.get(w, 0) / N < 0.01                       # in fewer than 1% of the training passages (or never seen)
        idf_max = float(np.log(N + 1) + 1)
        V, VL, RAR, MAR, STR = [], [], [], [], []
        for i in range(n):
            q, ps, sel = P.text[i], P.passages[i], P.sel1[i]
            Q = set(toks(q))
            Qr = {w for w in Q if rare(w)}
            Pt = [set(toks(p)) for p in ps]
            cos = (self.vec.transform(ps) @ self.vec.transform([q]).T).toarray().ravel()
            sc = np.zeros((len(ps), 5))
            for j, T in enumerate(Pt):
                inter = len(Q & T)
                sc[j] = [cos[j], inter / max(1, len(Q | T)), (len(Qr & T) / len(Qr)) if Qr else 0.0, 2 * inter / max(1, len(Q) + len(T)),
                         inter / max(1, min(len(Q), len(T)))]
            S_ = sc[sel] if len(sel) else np.zeros((1, 5))
            V.append(S_)                                               # kept per question: votes need thresholds
            union = set().union(*[Pt[j] for j in sel]) if sel else set()
            cover = len(Q & union) / max(1, len(Q))
            pair = [len(Pt[a] & Pt[b]) / max(1, len(Pt[a] | Pt[b])) for x, a in enumerate(sel) for b in sel[x + 1:]]
            VL.append(np.concatenate([S_.max(0), S_.mean(0), S_.min(0), [cover, float(np.mean(pair)) if pair else 0.0, float(len(sel))]]))
            idfs = [self.idf(w) for w in Q] or [0.0]
            rar = [float(np.mean(idfs)), float(np.max(idfs)), len(Qr) / max(1, len(Q)), float(np.mean([w not in df for w in Q])) if Q else 0.0, float(len(ents(q)))]
            srt = np.sort(cos)[::-1]
            mar = [srt[0] - srt[1], srt[1] - srt[2], srt[0], srt[1]]
            if P.sim_emb is not None:
                e = P.sim_emb[i]                                       # [s0, s1, s0-s2, ...]
                mar += [e[0] - e[1], e[1] - (e[0] - e[2]), e[0]]
            RAR.append(rar); MAR.append(mar)
            Eq = ents(q)
            Ep = [ents(p) for p in ps]
            bridge = max([len((Ep[a] & Ep[b]) - Eq) for x, a in enumerate(sel) for b in sel[x + 1:]] or [0])
            lead = [set(toks(" ".join(p.split()[:4]))) for p in ps]     # a passage usually opens with its title
            links = sum(1 for a in sel for b in sel if a != b and lead[a] and lead[a] <= Pt[b])
            unsel = [j for j in range(len(ps)) if j not in sel]
            left_out = (cos[unsel].max() if unsel else 0.0) - (cos[sel].min() if sel else 0.0)
            ecover = len(Eq & set().union(*[Ep[j] for j in sel])) / max(1, len(Eq)) if sel else 0.0
            STR.append([float(bridge), float(links), 1.0 - cover, len(Q - union) if sel else float(len(Q)), ecover, float(np.mean(pair)) if pair else 0.0,
                        float(left_out), float(len(sel)), P.facts[i, 0], P.facts[i, 1], float(len(Q))])
        self.V = V
        self.voters_table = np.array(VL)
        self.rarity, self.margin, self.structure = np.array(RAR), np.array(MAR), np.array(STR)
        self.rm = np.column_stack([self.rarity, self.margin, self.rarity[:, :1] * self.margin[:, 1:2], self.rarity[:, 2:3] * self.margin[:, 1:2],
                                   self.rarity[:, :1] * self.margin[:, :1]])

    def vote_score(self, thr, which=(0, 1, 2, 3, 4)):
        """Evidence support: how many selected passages win a majority of the voters; ties broken by the mean number of votes."""
        need = len(which) // 2 + 1
        out = np.zeros(len(self.V))
        for i, S_ in enumerate(self.V):
            votes = (S_[:, list(which)] >= thr[list(which)]).sum(1)
            out[i] = (votes >= need).sum() + votes.mean() / (len(which) + 1)
        return out

    def retune(self, tr, y, which=(0, 1, 2, 3, 4)):
        """Thresholds retuned on the training questions: one pass, each voter's threshold chosen from the training quantiles."""
        thr = ORIG_THR.copy()
        allv = np.vstack([self.V[i] for i in tr])
        for k in which:
            best = (-1, thr[k])
            for qv in np.quantile(allv[:, k], np.linspace(0.1, 0.9, 9)):
                t2 = thr.copy(); t2[k] = qv
                a = E.fauc(y[tr], self.vote_score(t2, which)[tr])
                a = a if a == a else 0.5
                if a > best[0]:
                    best = (a, qv)
            thr[k] = best[1]
        return thr


def softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_gate(Z, G, y, lam=1.0):
    """Mixture of experts: p = sigmoid(a * sum_k g_k(x) z_k + b), g = softmax(W [x, 1]). A small, regularised gate."""
    from scipy.optimize import minimize
    n, k = Z.shape
    Gb = np.column_stack([G, np.ones(n)])
    d = Gb.shape[1]

    def unpack(t):
        return t[:k * d].reshape(k, d), t[-2], t[-1]

    def loss(t):
        Wm, a, b = unpack(t)
        m = (softmax(Gb @ Wm.T) * Z).sum(1)
        p = 1 / (1 + np.exp(-(a * m + b)))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return -(y * np.log(p) + (1 - y) * np.log(1 - p)).mean() + lam * (Wm[:, :-1] ** 2).sum() / n
    t0 = np.concatenate([np.zeros(k * d), [1.0, float(np.log(y.mean() / (1 - y.mean() + 1e-9) + 1e-9))]])
    r = minimize(loss, t0, method="L-BFGS-B")
    Wm, a, b = unpack(r.x)

    def predict(Z2, G2, hard=False):
        g = softmax(np.column_stack([G2, np.ones(len(G2))]) @ Wm.T)
        if hard:
            g = np.eye(k)[g.argmax(1)]
        return (g * Z2).sum(1), g
    return predict


def pipeline(P, tr, te, decision, reps=2):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    y = P.y[decision]
    ytr = y[tr].astype(float)
    Ft = Feats(P, tr)
    base, ens, notes = [], {}, {}
    prior = float(ytr.mean())

    def reader(X, fn=H.f_linear, yy=None):
        yy = ytr if yy is None else yy
        if min(yy.sum(), len(yy) - yy.sum()) < 8:
            return None
        po = np.nanmean([H.cv(fn, X[tr], yy, 10 + r) for r in range(reps)], 0)
        try:
            pn = fn(X[tr], yy.astype(int), X[te])
        except Exception:
            pn = np.full(len(te), yy.mean())
        return np.nan_to_num(po, nan=float(yy.mean())), np.nan_to_num(pn, nan=float(yy.mean()))

    def add(name, po, pn, group="idea", note=""):
        ens[name] = (np.asarray(po, float), np.asarray(pn, float))
        if note:
            notes[name] = note
        base.append({"Signal family": name, "Group": group, "g": "Q", "Reader": "-", "train": E.fauc(ytr, po), "test": E.fauc(y[te], pn)})
    z = lambda a, ref: (a - np.mean(ref)) / (np.std(ref) + 1e-9)
    # ---------------------------------------------------------------- the fixed probe (the benchmark of notebooks 17-19)
    if decision == "answerer":
        Xfix = F.apply_reduce(F.fit_reduce(P.ha["h_answer_line"][tr, P.ha["h_answer_line"].shape[1] - 2, :]), P.ha["h_answer_line"][:, P.ha["h_answer_line"].shape[1] - 2, :])
    else:
        Xfix = F.apply_reduce(F.fit_reduce(P.hr[tr, P.hr.shape[1] // 2, :]), P.hr[:, P.hr.shape[1] // 2, :])
    fo, fn_ = reader(Xfix)
    add(FIXED, fo, fn_, "reference")
    ens[E.REFERENCE] = ens[FIXED]

    def plus(name):
        po, pn = ens[name]
        ens[PLUS + name] = ((z(fo, fo) + z(po, po)) / 2, (z(fn_, fo) + z(pn, po)) / 2)
    # ---------------------------------------------------------------- 1. five voters
    for nm, which, thr in (("Five voters: majority vote, original thresholds", (0, 1, 2, 3, 4), ORIG_THR),
                           ("Five voters: three voters only (TF-IDF, rare overlap, overlap coefficient)", (0, 2, 4), ORIG_THR)):
        s = Ft.vote_score(thr, which)
        add(nm, s[tr], s[te])
    thr = Ft.retune(tr, y)
    s = Ft.vote_score(thr)
    add("Five voters: majority vote, thresholds retuned on training questions", s[tr], s[te], note="thresholds " + ", ".join(f"{v}: {t:.3f}" for v, t in zip(VOTERS, thr)))
    r = reader(Ft.voters_table)
    add("Five voters: learned model on the continuous scores, coverage and redundancy", *r)
    plus(IDEAS["1 Five-voter gate"]); plus("Five voters: learned model on the continuous scores, coverage and redundancy")
    # ---------------------------------------------------------------- 2. rare margins
    add("Rare margins: rarity only", *reader(Ft.rarity))
    add("Rare margins: margin only", *reader(Ft.margin))
    add(IDEAS["2 Rare routing margins"], *reader(Ft.rm))
    plus(IDEAS["2 Rare routing margins"])
    # ---------------------------------------------------------------- experts for MoE / BoE
    ev = reader(np.column_stack([Ft.voters_table, Ft.rarity]))
    st = reader(Ft.structure)
    subs = [reader(P.meta), reader(F.apply_reduce(F.fit_reduce(P.hr[tr, P.hr.shape[1] // 2, :]), P.hr[:, P.hr.shape[1] // 2, :]))]
    if decision == "answerer":
        subs += [(fo, fn_), reader(P.tok)]
    co = (np.mean([z(a, a) for a, _ in subs], 0), np.mean([z(b, a) for a, b in subs], 0))
    add("Expert: evidence (lexical support, five voters, rare entities)", *ev, group="expert")
    add("Expert: structure (bridges, coverage gaps, redundancy, left-out passages)", *st, group="expert")
    add("Expert: confidence (retriever facts and hidden state" + (", answer probe, token probabilities)" if decision == "answerer" else ")"), *co, group="expert")
    Zo = np.column_stack([z(ev[0], ev[0]), z(st[0], st[0]), z(co[0], co[0])])
    Zn = np.column_stack([z(ev[1], ev[0]), z(st[1], st[0]), z(co[1], co[0])])
    Gall = np.column_stack([P.facts[:, 0], P.facts[:, 1], Ft.structure[:, -1], Ft.rarity[:, 0], Ft.margin[:, 1]])
    gmu, gsd = Gall[tr].mean(0), Gall[tr].std(0) + 1e-9
    Go = np.column_stack([(Gall[tr] - gmu) / gsd, Zo.std(1)])
    Gn = np.column_stack([(Gall[te] - gmu) / gsd, Zn.std(1)])
    yi = ytr.astype(int)
    # ---------------------------------------------------------------- 4. MoE
    add("MoE: equal weights", Zo.mean(1), Zn.mean(1))
    mk = lambda C=1.0, cw=None: (lambda X, yy: (lambda m: (lambda Z: m.predict_proba(Z)[:, 1]))(LogisticRegression(C=C, max_iter=500, class_weight=cw).fit(X, yy.astype(int))))
    add("MoE: fixed weights tuned on training questions", E.cvp(mk(), Zo, ytr), mk()(Zo, ytr)(Zn))

    def gate_cv(hard):
        from sklearn.model_selection import StratifiedKFold
        p = np.zeros(len(ytr))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=0).split(Zo, yi):
            p[b] = fit_gate(Zo[a], Go[a], ytr[a])(Zo[b], Go[b], hard)[0]
        g = fit_gate(Zo, Go, ytr)
        pn, w = g(Zn, Gn, hard)
        return p, pn, w
    po, pn, w = gate_cv(False)
    add(IDEAS["4 Mixture of experts (MoE)"], po, pn, note="mean weight on test questions - evidence %.2f, structure %.2f, confidence %.2f; weight spread across questions %.2f"
        % (w[:, 0].mean(), w[:, 1].mean(), w[:, 2].mean(), float(w.std(0).mean())))
    po, pn, w = gate_cv(True)
    add("MoE: the gate selects one expert per question", po, pn, note="share of test questions sent to each expert - evidence %.2f, structure %.2f, confidence %.2f" % tuple(w.mean(0)))
    plus(IDEAS["4 Mixture of experts (MoE)"])
    # ---------------------------------------------------------------- 6. adaptive rule and sharpening
    med = np.median(Zo.std(1))
    rule = lambda Z: np.where(Z.std(1) > med, Z.min(1), Z.mean(1))        # experts disagree -> trust the most worried one
    add("Adaptive rule: when the experts disagree, follow the most worried one", rule(Zo), rule(Zn))
    lg = lambda p: np.log(np.clip(p, 1e-4, 1 - 1e-4) / (1 - np.clip(p, 1e-4, 1 - 1e-4)))
    sig = lambda v: 1 / (1 + np.exp(-v))
    Lo = np.column_stack([lg(ev[0]), lg(st[0]), lg(sig(co[0]))])
    Ln = np.column_stack([lg(ev[1]), lg(st[1]), lg(sig(co[1]))])
    add(IDEAS["6 Adaptive weighting and sharpening"], E.cvp(mk(10.0), Lo, ytr), mk(10.0)(Lo, ytr)(Ln))
    plus(IDEAS["6 Adaptive weighting and sharpening"])
    # ---------------------------------------------------------------- 5. BoE
    share = E.SHARE[decision]
    vo = (Zo <= np.quantile(Zo, share, 0)).sum(1)
    vn = (Zn <= np.quantile(Zo, share, 0)).sum(1)
    add("BoE: every expert votes, ties broken by the mean score", -vo + Zo.mean(1) / 10, -vn + Zn.mean(1) / 10)
    joint = lambda Z, i: np.column_stack([Z, Z.std(1), Z.max(1) - Z.min(1), Ft.rarity[i, 0], Ft.margin[i, 1], Z[:, 0] * Z[:, 2], Z[:, 1] * Z[:, 2], Z[:, 0] * Z[:, 1],
                                          Ft.rarity[i, 0] * Ft.margin[i, 1]])
    Jo, Jn = joint(Zo, tr), joint(Zn, te)
    jm, js = Jo.mean(0), Jo.std(0) + 1e-9
    Jo, Jn = (Jo - jm) / js, (Jn - jm) / js
    add(IDEAS["5 Bundle of experts (BoE)"], E.cvp(mk(0.3), Jo, ytr), mk(0.3)(Jo, ytr)(Jn))
    gb = lambda X, yy: (lambda m: (lambda Z: m.predict_proba(Z)[:, 1]))(GradientBoostingClassifier(n_estimators=40, max_depth=2, subsample=0.8, random_state=0).fit(X, yy.astype(int)))
    add("BoE: joint boosted trees on every output", E.cvp(gb, Jo, ytr), gb(Jo, ytr)(Jn))
    plus(IDEAS["5 Bundle of experts (BoE)"])
    # ---------------------------------------------------------------- 3. benefit targets
    if decision == "answerer":
        cheap, big = P.l8, P.A["e1"]
    else:
        cheap, big = P.A["e1"], P.A["e3"]
    helps, hurts = ((big == 1) & (cheap == 0)).astype(float), ((big == 0) & (cheap == 1)).astype(float)
    for tag, X in (("", Xfix), (" on every expert output", np.column_stack([Zo_full(Zo, Zn, tr, te, P.n), ]))):
        rh, ru = reader(X, yy=helps[tr]), reader(X, yy=hurts[tr])
        if rh is None:                                                # too few examples to learn from: the fixed probe's score stands in
            add("Benefit probe: predicts 'helps' only" + tag, fo, fn_, note=f"'helps' has only {int(helps[tr].sum())} training questions: the fixed probe was used instead")
            add("Benefit probe: expected gain (helps minus hurts)" + tag, fo, fn_, note=f"'helps' has only {int(helps[tr].sum())} training questions: the fixed probe was used instead")
            continue
        add("Benefit probe: predicts 'helps' only" + tag, -rh[0], -rh[1], note=f"'helps' on {int(helps[tr].sum())} training questions")
        if ru is None:
            ru = (np.full(len(tr), hurts[tr].mean()), np.full(len(te), hurts[tr].mean()))
            nt = f"'hurts' has only {int(hurts[tr].sum())} training questions: a constant was used"
        else:
            nt = f"'helps' on {int(helps[tr].sum())}, 'hurts' on {int(hurts[tr].sum())} training questions"
        add("Benefit probe: expected gain (helps minus hurts)" + tag, -(rh[0] - ru[0]), -(rh[1] - ru[1]), note=nt)
    plus(IDEAS["3 Benefit-targeted probes"])
    corr = np.corrcoef(np.array([ens[b["Signal family"]][1] for b in base]))
    return base, ens, notes, corr


def Zo_full(Zo, Zn, tr, te, n):
    out = np.zeros((n, Zo.shape[1]))
    out[tr], out[te] = Zo, Zn
    return out


# ==================================================================================
# Run
# ==================================================================================
def relabel(P, sets, kind):
    """A copy of the pool scored with exact match ('strict') or with the corrected label ('lenient')."""
    Q = copy.copy(P)
    Q.A = dict(P.A)
    if kind == "lenient":
        cat = lambda k: np.concatenate([s["len"][k] for s in sets])
        Q.A["e1"], Q.A["e3"], Q.l8 = cat("e1"), cat("e3"), cat("l8")
    Q.y = {"answerer": (Q.l8 >= Q.A["e1"]).astype(float), "retriever": P.A["f1"].astype(float)}
    return Q


def cmd_run(args):
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    t0 = time.time()
    out = os.path.join(args.run_dir, "results")
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    log(f"experts code version {VERSION}")
    old = load_set("old", args.old_sweep, args.old_data, args.old_hidden)
    new = load_set("fresh", args.fresh_run, args.fresh_data, args.fresh_run)
    sets = [old, new]
    P0 = E.Pool(sets, use_embedding=not args.no_embedding)
    P0.sel1 = [x for s in sets for x in s["sel1"]]
    log(f"{P0.n} questions; question embedding: {P0.emb_note}")
    E.pipeline = pipeline                                             # the evaluation of notebook 19, with this file's models
    io, inew = np.where(P0.origin == "old")[0], np.where(P0.origin == "fresh")[0]
    rows, base_rows, extra = [], [], {}
    for kind, lab in (("lenient", "corrected label (substantially right)"), ("strict", "exact match")):
        P = relabel(P0, sets, kind)
        for decision in ("answerer", "retriever"):
            y = P.y[decision]
            r0, b0, x0 = [], [], {}
            E.evaluate(P, decision, "old -> fresh", [(io, inew)], r0, b0, x0)
            E.evaluate(P, decision, "fresh -> old", [(inew, io)], r0, b0, x0)
            strat = np.array([f"{o}{int(v)}" for o, v in zip(P.origin, y)])
            folds = []
            for r in range(args.cv_repeats):
                folds += [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(P.n), strat)]
            E.evaluate(P, decision, "all 400, cross-validated", folds, r0, b0, x0)
            for r in r0:
                r["Scoring"] = lab
            rows += [r for r in r0 if r["Ensemble"] != E.REFERENCE]
            log(f"{lab} | {decision} done ({time.time() - t0:.0f}s)")
    key_of = lambda d: "accuracy" if d == "answerer" else "both passages found"
    # ---------------------------------------------------------------- audit tables
    cat = lambda f: np.concatenate([f(s) for s in sets])
    audit = []
    for nm, s in (("Old 200", old), ("Fresh 200", new)):
        b, l = s["A"]["e1"], s["l8"]
        bl, ll = s["len"]["e1"], s["len"]["l8"]
        audit.append({"Questions": nm, "Qwen 32B: exact match": b.mean(), "Qwen 32B: F1": s["len"]["f1e1"].mean(), "Qwen 32B: substantially right": bl.mean(),
                      "Llama 8B: exact match": l.mean(), "Llama 8B: F1": s["len"]["f1l8"].mean(), "Llama 8B: substantially right": ll.mean(),
                      "Gap, exact match (points)": 100 * (b.mean() - l.mean()), "Gap, substantially right (points)": 100 * (bl.mean() - ll.mean()),
                      "'Escalating helps', exact match": int(((b == 1) & (l == 0)).sum()), "'Escalating helps', corrected": int(((bl == 1) & (ll == 0)).sum()),
                      "'Escalating hurts', exact match": int(((b == 0) & (l == 1)).sum()), "'Escalating hurts', corrected": int(((bl == 0) & (ll == 1)).sum()),
                      "Share of exact-match 'helps' that is only wording": float((((b == 1) & (l == 0)) & ~((bl == 1) & (ll == 0))).sum() / max(1, ((b == 1) & (l == 0)).sum())),
                      "Found both: Qwen 32B right (corrected)": bl[s["A"]["f1"] == 1].mean(), "Missed a passage: Qwen 32B right (corrected)": bl[s["A"]["f1"] == 0].mean(),
                      "Missed a passage, 3 tries: Qwen 32B right (corrected)": s["len"]["e3"][s["A"]["f1"] == 0].mean()})
    labels = old["label_rows"] + new["label_rows"]
    # ---------------------------------------------------------------- verdicts per idea
    verdicts = []
    for lab in ("corrected label (substantially right)", "exact match"):
        for decision in ("answerer", "retriever"):
            key = key_of(decision)
            g = {(r["Ensemble"], r["Evaluation"]): r for r in rows if r["Scoring"] == lab and r["Decision"] == decision}
            for idea, name in IDEAS.items():
                if (name, "all 400, cross-validated") not in g:
                    continue
                cv, a, b = g[(name, "all 400, cross-validated")], g[(name, "old -> fresh")], g[(name, "fresh -> old")]
                pl = {ev: g.get((PLUS + name, ev)) for ev in ("all 400, cross-validated", "old -> fresh", "fresh -> old")}
                fx = g[(FIXED, "all 400, cross-validated")]
                kr = f"Gate minus random: {key} (points)"
                verdicts.append({
                    "Scoring": lab, "Decision": decision, "Idea": idea, "Variant named beforehand": name, "Separation": cv["Separation"], "Fixed probe: separation": fx["Separation"],
                    "Idea minus fixed probe": cv["Minus the fixed probe"], "low": cv["Minus the fixed probe: low"], "high": cv["Minus the fixed probe: high"],
                    "Gate vs random (points)": cv[kr], "random: low": cv[f"Gate minus random: {key}, low"], "random: high": cv[f"Gate minus random: {key}, high"],
                    "Works alone (beats a random gate in all three evaluations)": bool(cv[f"Gate minus random: {key}, low"] > 0 and a[kr] > 0 and b[kr] > 0),
                    "Fixed probe + idea, minus fixed probe": pl["all 400, cross-validated"]["Minus the fixed probe"] if pl["all 400, cross-validated"] else None,
                    "added: low": pl["all 400, cross-validated"]["Minus the fixed probe: low"] if pl["all 400, cross-validated"] else None,
                    "added: high": pl["all 400, cross-validated"]["Minus the fixed probe: high"] if pl["all 400, cross-validated"] else None,
                    "Adds to the fixed probe (interval excludes zero, ahead in both transfers)": bool(
                        pl["all 400, cross-validated"] and pl["all 400, cross-validated"]["Minus the fixed probe: low"] > 0
                        and pl["old -> fresh"]["Minus the fixed probe"] > 0 and pl["fresh -> old"]["Minus the fixed probe"] > 0),
                    "Gate accuracy vs cheap fixed setup (points)": cv["Gate accuracy minus the cheap fixed setup (points)"], "bar: low": cv["vs cheap fixed setup, low"],
                    "bar: high": cv["vs cheap fixed setup, high"], "Compute vs cheap fixed setup": cv["Gate compute vs the cheap fixed setup"],
                    "Fixed probe: gate vs random (points)": fx[kr], "Fixed probe: accuracy vs cheap fixed setup (points)": fx["Gate accuracy minus the cheap fixed setup (points)"]})

    def dump(nm, rr):
        with open(os.path.join(out, nm), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    dump("all_models.jsonl", rows); dump("verdicts.jsonl", verdicts); dump("scoring_audit.jsonl", audit); dump("labels.jsonl", labels)
    S.write_json(os.path.join(out, "verdict.json"), {"version": VERSION, "embedding": P0.emb_note, "seconds": round(time.time() - t0),
                                                      "ideas": [{k: v for k, v in r.items()} for r in verdicts if r["Scoring"].startswith("corrected")]})
    charts(os.path.join(out, "charts"), rows, verdicts, audit)
    # ---------------------------------------------------------------- summary
    pctc = {"Qwen 32B: exact match", "Qwen 32B: F1", "Qwen 32B: substantially right", "Llama 8B: exact match", "Llama 8B: F1", "Llama 8B: substantially right",
            "Share of exact-match 'helps' that is only wording", "Found both: Qwen 32B right (corrected)", "Missed a passage: Qwen 32B right (corrected)",
            "Missed a passage, 3 tries: Qwen 32B right (corrected)", "Share acted on"}
    sgn = {"Idea minus fixed probe", "low", "high", "Gate vs random (points)", "random: low", "random: high", "Fixed probe + idea, minus fixed probe", "added: low", "added: high",
           "Gate accuracy vs cheap fixed setup (points)", "bar: low", "bar: high", "Gap, exact match (points)", "Gap, substantially right (points)", "Minus the fixed probe",
           "Minus the fixed probe: low", "Minus the fixed probe: high", "Gate accuracy minus the cheap fixed setup (points)", "vs cheap fixed setup, low", "vs cheap fixed setup, high",
           "Fixed probe: gate vs random (points)", "Fixed probe: accuracy vs cheap fixed setup (points)"}
    spc = {"Compute vs cheap fixed setup", "Gate compute vs the cheap fixed setup"}
    sep_diff = {"Idea minus fixed probe", "low", "high", "Fixed probe + idea, minus fixed probe", "added: low", "added: high", "Minus the fixed probe",
                "Minus the fixed probe: low", "Minus the fixed probe: high"}

    def T(rr, cols, ren=None):
        ren = ren or {}

        def f(r, c):
            v = r.get(c)
            if v is None or (isinstance(v, float) and v != v):
                return "-"
            if isinstance(v, (bool, np.bool_)):
                return "yes" if v else "no"
            if isinstance(v, str):
                return v
            if isinstance(v, (int, np.integer)):
                return str(v)
            if c in spc:
                return f"{100 * v:+.0f}%"
            if c in pctc:
                return f"{100 * v:.1f}%"
            if c in sep_diff:
                return f"{v:+.2f}"
            if c in sgn or c.startswith("Gate minus random"):
                return f"{v:+.1f}"
            return f"{v:.2f}"
        return "| " + " | ".join(ren.get(c, c) for c in cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rr)
    tr_ = lambda rr: [{"Measure": k, **{r["Questions"]: (f"{100 * r[k]:.1f}%" if k in pctc else (f"{r[k]:+.1f}" if k in sgn else str(r[k]))) for r in rr}} for k in rr[0] if k != "Questions"]
    L = ["# Set 5g - the ideas from Routing_Experiment_Ideas.docx, tested on this task\n",
         f"Code version {VERSION}. {P0.n} questions (200 old, 200 fresh). No model was run. Question embedding: {P0.emb_note}.",
         "Both question sets have already been used for development, so nothing here is an untouched test. Three evaluations are reported so that no single sample decides.\n",
         "## 0. Scoring audit\n",
         "Exact match marks an answer wrong when it differs from the gold string in any way ('Gardnerville, Nevada' against 'Gardnerville'). The corrected label counts an "
         "answer as substantially right when it matches exactly, or its token F1 with the gold answer is at least 0.5, or it contains the gold answer; yes/no answers must match "
         "exactly. A hand check of about 90 borderline answers on the fresh questions found roughly 1 in 20 accepted wrongly and a similar share rejected wrongly, so it is "
         "much closer to correctness than exact match, but not perfect. Every answer is listed in `results/labels.jsonl` for checking.\n",
         T(tr_(audit), ["Measure", "Old 200", "Fresh 200"]),
         "\nBecause the routing target ('the Llama 8B answer is no worse than the Qwen 32B answer') is built from these labels, a large part of what the earlier probes were "
         "trained to find was a difference in wording. All results below are given first with the corrected label and then with exact match.\n"]
    vc = ["Idea", "Variant named beforehand", "Separation", "Idea minus fixed probe", "low", "high", "Gate vs random (points)", "random: low", "random: high",
          "Works alone (beats a random gate in all three evaluations)", "Fixed probe + idea, minus fixed probe", "added: low", "added: high",
          "Adds to the fixed probe (interval excludes zero, ahead in both transfers)", "Gate accuracy vs cheap fixed setup (points)", "bar: low", "bar: high", "Compute vs cheap fixed setup"]
    ren = {"low": "95% low", "high": "95% high", "random: low": "95% low", "random: high": "95% high", "added: low": "95% low", "added: high": "95% high",
           "bar: low": "95% low", "bar: high": "95% high"}
    sec = 1
    for lab in ("corrected label (substantially right)", "exact match"):
        for decision, title in (("answerer", "keep the Llama 8B answer, or ask the Qwen 32B?"), ("retriever", "stop after one retrieval try, or try twice more?")):
            vv = [v for v in verdicts if v["Scoring"] == lab and v["Decision"] == decision]
            if not vv:
                continue
            key = key_of(decision)
            L += [f"## {sec}. {decision.capitalize()} decision ({title}) - {lab}\n",
                  f"All 400 questions, cross-validated. The fixed probe scores {vv[0]['Fixed probe: separation']:.2f}; its gate is {vv[0]['Fixed probe: gate vs random (points)']:+.1f} "
                  f"points against a random gate and {vv[0]['Fixed probe: accuracy vs cheap fixed setup (points)']:+.1f} points of accuracy against the cheap fixed setup. "
                  f"For the retriever decision 'gate vs random' counts how often both passages are found.\n" if decision == "retriever" else
                  f"All 400 questions, cross-validated. The fixed probe scores {vv[0]['Fixed probe: separation']:.2f}; its gate is {vv[0]['Fixed probe: gate vs random (points)']:+.1f} "
                  f"points against a random gate and {vv[0]['Fixed probe: accuracy vs cheap fixed setup (points)']:+.1f} points of accuracy against the cheap fixed setup.\n",
                  "### One row per idea (the variant named beforehand)\n", T(vv, vc, ren), "\n### Every variant\n"]
            for ev in ("all 400, cross-validated", "old -> fresh", "fresh -> old"):
                rr = sorted([r for r in rows if r["Scoring"] == lab and r["Decision"] == decision and r["Evaluation"] == ev], key=lambda r: (r["Ensemble"] != FIXED, -r["Separation"]))
                L += [f"**{ev}**\n",
                      T(rr, ["Ensemble", "Separation", "Minus the fixed probe", "Minus the fixed probe: low", "Minus the fixed probe: high", "Share acted on",
                             f"Gate minus random: {key} (points)", f"Gate minus random: {key}, low", f"Gate minus random: {key}, high",
                             "Gate accuracy minus the cheap fixed setup (points)", "vs cheap fixed setup, low", "vs cheap fixed setup, high", "Gate compute vs the cheap fixed setup"],
                        {"Ensemble": "Model"}), ""]
            nt = [r for r in rows if r["Scoring"] == lab and r["Decision"] == decision and r["Evaluation"] == "old -> fresh" and r.get("Built from")]
            if nt:
                L += ["Details (old -> fresh):\n"] + [f"- {r['Ensemble']}: {r['Built from']}" for r in nt] + [""]
            sec += 1
    n_rows = len(rows)
    L += ["## How to read the many rows\n",
          f"There are {n_rows} model-by-evaluation rows. At a 95% level about {round(0.05 * n_rows)} of them would show an interval that excludes zero by chance alone. "
          "That is why one variant per idea was named beforehand, and why an idea counts only if it holds in all three evaluations.\n",
          "Not tested here: sequence-aware transitions (bridge retrieval), which need new model runs.\n",
          "Charts: `results/charts/`. Excel: `results/set5g_ideas.xlsx`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    W.write_excel(os.path.join(out, "set5g_ideas.xlsx"), {"One row per idea": pd.DataFrame(verdicts), "Every model": pd.DataFrame(rows), "Scoring audit": pd.DataFrame(tr_(audit)),
                                                          "Every answer, three scores": pd.DataFrame(labels)})
    for v in verdicts:
        if v["Scoring"].startswith("corrected"):
            log(f"{v['Decision']:9s} | {v['Idea']:36s} | sep {v['Separation']:.2f} (fixed {v['Fixed probe: separation']:.2f}) | vs random {v['Gate vs random (points)']:+.1f} | "
                f"works alone {v['Works alone (beats a random gate in all three evaluations)']} | adds to fixed probe {v['Adds to the fixed probe (interval excludes zero, ahead in both transfers)']} | "
                f"vs cheap fixed setup {v['Gate accuracy vs cheap fixed setup (points)']:+.1f} at {100 * v['Compute vs cheap fixed setup']:+.0f}%")
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def charts(out, rows, verdicts, audit):
    plt = S._plt()
    C = S.SERIES
    # 0. scoring audit
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    x = np.arange(4)
    lab = ["Qwen 32B\nold", "Llama 8B\nold", "Qwen 32B\nfresh", "Llama 8B\nfresh"]
    em = [audit[0]["Qwen 32B: exact match"], audit[0]["Llama 8B: exact match"], audit[1]["Qwen 32B: exact match"], audit[1]["Llama 8B: exact match"]]
    le = [audit[0]["Qwen 32B: substantially right"], audit[0]["Llama 8B: substantially right"], audit[1]["Qwen 32B: substantially right"], audit[1]["Llama 8B: substantially right"]]
    ax.bar(x - 0.2, 100 * np.array(em), 0.38, color=S.MUTED, label="exact match")
    ax.bar(x + 0.2, 100 * np.array(le), 0.38, color=C[0], label="substantially right (corrected label)")
    for i in range(4):
        ax.text(x[i] - 0.2, 100 * em[i] + 1, f"{100 * em[i]:.1f}", ha="center", fontsize=8)
        ax.text(x[i] + 0.2, 100 * le[i] + 1, f"{100 * le[i]:.1f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(lab)
    ax.set_ylabel("answers counted right (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=8, loc="upper right", ncol=2)
    ax.set_title("Scoring audit: exact match against the corrected label (one answer per question)", fontsize=10)
    fig.savefig(os.path.join(out, "0_scoring_audit.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 1. one row per idea
    for li, lab_ in enumerate(("corrected label (substantially right)", "exact match")):
        vv = [v for v in verdicts if v["Scoring"] == lab_]
        if not vv:
            continue
        fig, axs = plt.subplots(2, 3, figsize=(15.5, 7.2))
        for r_, decision in enumerate(("answerer", "retriever")):
            g = [v for v in vv if v["Decision"] == decision][::-1]
            yy = np.arange(len(g))
            specs = [("Idea minus fixed probe", "low", "high", "separation: the idea alone minus the fixed probe", 0.0),
                     ("Fixed probe + idea, minus fixed probe", "added: low", "added: high", "separation: fixed probe + idea, minus the fixed probe", 0.0),
                     ("Gate vs random (points)", "random: low", "random: high", "the idea's gate minus a random gate (points)", 0.0)]
            for ax, (k, lo, hi, xl, ref) in zip(axs[r_], specs):
                for i, v in enumerate(g):
                    if v.get(k) is None:
                        continue
                    good = v[lo] > 0
                    ax.errorbar([v[k]], [i], xerr=[[v[k] - v[lo]], [v[hi] - v[k]]], fmt="o", color=C[2] if good else (C[7] if v[hi] < 0 else S.INK2), ms=5, lw=1.3)
                ax.axvline(ref, color=S.INK, lw=1, ls="--")
                ax.set_yticks(yy)
                ax.set_yticklabels([v["Idea"] for v in g] if ax is axs[r_][0] else [], fontsize=8)
                ax.set_xlabel(xl, fontsize=8)
                ax.set_title(f"{decision} decision", fontsize=9)
        fig.suptitle(f"The six ideas, one named variant each, all 400 questions cross-validated - {lab_} (green = interval above zero, red = below)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"{1 + li}_ideas_{'corrected' if li == 0 else 'exact_match'}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 3. accuracy change against compute change, corrected label, answerer
    for li, lab_ in enumerate(("corrected label (substantially right)", "exact match")):
        rr = [r for r in rows if r["Scoring"] == lab_ and r["Decision"] == "answerer" and r["Evaluation"] == "all 400, cross-validated"]
        if not rr:
            continue
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
        named = set(IDEAS.values())
        for r in rr:
            is_f, is_n = r["Ensemble"] == FIXED, r["Ensemble"] in named
            ax.scatter([100 * r["Gate compute vs the cheap fixed setup"]], [r["Gate accuracy minus the cheap fixed setup (points)"]], s=90 if (is_f or is_n) else 22,
                       color=S.INK if is_f else (C[0] if is_n else S.MUTED), marker="s" if is_f else "o", zorder=4 if (is_f or is_n) else 2, alpha=1 if (is_f or is_n) else 0.6)
            if is_n:
                ax.annotate([k for k, v in IDEAS.items() if v == r["Ensemble"]][0][:22], (100 * r["Gate compute vs the cheap fixed setup"], r["Gate accuracy minus the cheap fixed setup (points)"]),
                            textcoords="offset points", xytext=(6, 4), fontsize=7)
        ax.axhline(0, color=S.INK, lw=1)
        ax.axhline(-2, color=C[2], ls=":", lw=1)
        ax.scatter([0], [0], marker="*", s=220, color=C[1], zorder=5)
        ax.annotate("cheap fixed setup", (0, 0), textcoords="offset points", xytext=(-8, 8), ha="right", fontsize=8)
        ax.set_xlabel("compute against the cheap fixed setup (%)")
        ax.set_ylabel("accuracy against the cheap fixed setup (points)")
        ax.set_title(f"Answerer gates on all 400 questions - {lab_}\n(black square = fixed probe, blue = the named variant of each idea, grey = other variants; dotted = 2 points below)", fontsize=9)
        fig.savefig(os.path.join(out, f"{3 + li}_accuracy_vs_compute_{'corrected' if li == 0 else 'exact_match'}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    for a in ("old_sweep", "old_hidden", "old_data", "fresh_run", "fresh_data", "run_dir"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--cv_repeats", type=int, default=2)
    p.add_argument("--no_embedding", action="store_true")
    cmd_run(ap.parse_args())


if __name__ == "__main__":
    main()
