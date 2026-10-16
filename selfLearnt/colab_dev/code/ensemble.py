"""Set 5f - do signals from different families add up? Cross-family ensembles (notebook 19, CPU only).

Every earlier step combined signals only within one family. This file combines all of them: the question itself (facts, words, topics,
similarity to the passages, embedding), the retriever's first pass (facts about the try, hidden state) and one cheap answer (hidden
states, attention, token probabilities, stated confidence). No model is run: everything is read from notebooks 11, 16, 17 and 18.

Two decisions are studied:
  answerer   after one Llama 8B answer: keep it, or let the Qwen 32B answer?        target = the Llama 8B answer is no worse
  retriever  after one Qwen 7B retrieval try: stop, or try twice more?              target = the first try found both passages

Three evaluations, so that no single sample decides:
  old -> fresh   everything fitted on the 200 old questions, tested on the 200 fresh ones
  fresh -> old   the reverse
  all 400        5-fold cross-validation over both sets, the whole pipeline refitted inside every fold

The primary ensemble was named before running: the average of every base model that separates better than chance on the training
questions. It 'adds value' only if it beats the best single base model in the 400-question evaluation with a 95% interval that excludes
zero, and is ahead in both transfer directions. Because that comparator is picked among dozens of noisy candidates, every ensemble is also
compared with one fixed hidden-state probe (the kind tested in notebooks 17 and 18), with a random gate, and with the cheap fixed setup.
"""
import argparse
import json
import os
import re
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S              # noqa: E402
import sweep as W            # noqa: E402
import hidden as H           # noqa: E402
import final as F            # noqa: E402
import retriever_check as RC  # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "5f.3"
log = H.log
GROUPS = {"Q": "the question (free, before anything runs)", "R": "the retriever's first pass (free once it has run)", "A": "one cheap Llama 8B answer"}
READERS = {"linear": H.f_linear, "trees": H.f_boost, "neighbours": H.f_knn}
PRIMARY = "PRIMARY: average of every base model above chance on the training questions"
REFERENCE = "Reference: the best single base model on the training questions"
FIXED = "Reference: one fixed hidden-state probe, as in notebooks 17 and 18"
SIM_NAMES = 8
K_ADD, K_GATE = "primary ensemble adds value over the best single base model", "primary ensemble beats a random gate"
K_FIX, K_BAR = "primary ensemble beats one fixed hidden-state probe", "primary ensemble's gate against the cheap fixed setup"


def sim_features(s):
    s = np.sort(np.asarray(s, float))[::-1]
    p = np.clip(s, 1e-9, None)
    p = p / p.sum()
    return [s[0], s[1], s[0] - s[2], s[:2].mean() - s[2:].mean(), s.mean(), s.std(), float((s > 0.7 * s[0]).sum()), float(-(p * np.log(p + 1e-12)).sum())]


def embed(texts):
    try:
        from sentence_transformers import SentenceTransformer
        if not hasattr(embed, "m"):
            embed.m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return np.asarray(embed.m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=64), float), "all-MiniLM-L6-v2"
    except Exception as ex:
        return None, f"not available here ({type(ex).__name__})"


def load_set(name, sweep_run, data_dir, models_json, hidden_run):
    H_log, H.log = H.log, (lambda *a: None)
    qs, A, meta, Er = RC.load_side(sweep_run, data_dir, models_json, hidden_run)
    H.log = H_log
    Ea = H.load_extract(hidden_run, "l8", "answer", [q["qid"] for q in qs])
    assert Er is not None and Ea is not None and Er["have"].all() and Ea["have"].all(), f"hidden states missing for the {name} questions"
    P = {k: v["params"] for k, v in W.load_models(models_json).items()}
    rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(sweep_run, W.vid(RC.RET, RC.COT, RC.PICK)))}
    fr1 = np.array([2 * P[RC.RET] * (rets[q["qid"]]["prompt_tokens"] + rets[q["qid"]]["samples"][0]["out_tokens"]) for q in qs], float)
    cl8 = np.array([2 * P["l8"] * (r["prompt_tokens"] + r["out_tokens"]) for r in Ea["rows"]], float)
    A = dict(A, fr1=fr1, cl8=cl8)
    return {"name": name, "qs": qs, "A": A, "meta": meta, "Er": Er, "Ea": Ea, "l8": np.array([r["EM"] for r in Ea["rows"]], float)}


class Pool:
    """All questions of both sets in one table; every feature transform is fitted on the training rows it is given, never on test rows."""

    def __init__(self, sets, use_embedding=True):
        self.qs = [q for s in sets for q in s["qs"]]
        self.n = len(self.qs)
        self.origin = np.array([s["name"] for s in sets for _ in s["qs"]])
        cat = lambda f: np.concatenate([f(s) for s in sets])
        self.A = {k: cat(lambda s: s["A"][k]) for k in sets[0]["A"]}
        self.l8 = cat(lambda s: s["l8"])
        self.text = [q["question"] for q in self.qs]
        self.passages = [list(q["ctxs_candidate"]) for q in self.qs]
        self.meta = np.vstack([s["meta"] for s in sets])
        self.hr = np.vstack([s["Er"]["h_prompt"] for s in sets])                    # [n, layers, d]
        self.ha = {k: np.vstack([s["Ea"][k] for s in sets]) for k in ("h_prompt", "h_answer", "h_answer_line")}
        tabs = [F.tables(s["Ea"]) for s in sets]
        self.att = np.vstack([t["attention"] for t in tabs])
        self.tok = np.vstack([t["tokens"] for t in tabs])
        self.conf = np.array([[np.nan if r["stated_conf"] is None else r["stated_conf"], r["n_passages"], r["prompt_tokens"]] for s in sets for r in s["Ea"]["rows"]], float)
        f = [S.question_features(t) for t in self.text]
        wh = ["what", "which", "who", "when", "where", "how"]
        first = [next((w for w in wh if re.search(rf"\b{w}\b", t.lower())), "other") for t in self.text]
        self.facts = np.column_stack([[x["f_yesno"] for x in f], [x["f_compare"] for x in f], [x["f_words"] for x in f], [x["f_digit"] for x in f],
                                      [x["f_caps"] for x in f], [x["f_quote"] for x in f], [len(t) for t in self.text]] + [[float(a == w) for a in first] for w in wh])
        self.emb, self.emb_note, self.sim_emb = None, "not used", None
        if use_embedding:
            self.emb, self.emb_note = embed(self.text)
            if self.emb is not None:
                pe, _ = embed([p for ps in self.passages for p in ps])
                sims, i = [], 0
                for e, ps in zip(self.emb, self.passages):
                    sims.append(sim_features(pe[i:i + len(ps)] @ e))
                    i += len(ps)
                self.sim_emb = np.array(sims)
        self.y = {"answerer": (self.l8 >= self.A["e1"]).astype(float), "retriever": self.A["f1"].astype(float)}

    def families(self, tr, decision):
        """name -> (X for all rows, group). Unsupervised transforms are fitted on rows `tr` only."""
        from sklearn.decomposition import NMF, PCA, TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        out = {}
        ttr = [self.text[i] for i in tr]
        out["question facts"] = (self.facts, "Q")
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english").fit(ttr)
        tf_tr, tf_all = vec.transform(ttr), vec.transform(self.text)
        k = int(min(16, tf_tr.shape[1] - 1, len(tr) - 1))
        out["question words (TF-IDF)"] = (TruncatedSVD(k, random_state=0).fit(tf_tr).transform(tf_all), "Q")
        for nt in (6, 12):
            out[f"question topics (topic model, {nt} topics)"] = (NMF(nt, init="nndsvda", random_state=0, max_iter=400).fit(tf_tr).transform(tf_all), "Q")
        pv = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit([p for i in tr for p in self.passages[i]] + ttr)
        sims = [sim_features((pv.transform(ps) @ pv.transform([t]).T).toarray().ravel()) for t, ps in zip(self.text, self.passages)]
        names = [float(sum(1 for p in ps if any(w in p for w in set(re.findall(r"\b[A-Z][a-z]{3,}\b", t))))) for t, ps in zip(self.text, self.passages)]
        out["question-passage similarity (words)"] = (np.column_stack([np.array(sims), names]), "Q")
        if self.emb is not None:
            out["question embedding"] = (PCA(int(min(16, len(tr) - 1)), random_state=0).fit(self.emb[tr]).transform(self.emb), "Q")
            out["question-passage similarity (embedding)"] = (self.sim_emb, "Q")
        out["retriever: facts about its first try"] = (self.meta, "R")
        nl = self.hr.shape[1]
        for li, lab in ((nl // 2, "middle layer"), (nl - 1, "last layer")):
            out[f"retriever hidden state ({lab})"] = (F.apply_reduce(F.fit_reduce(self.hr[tr, li, :]), self.hr[:, li, :]), "R")
        if decision == "answerer":
            na = self.ha["h_answer_line"].shape[1]
            for key, lab in (("h_answer_line", "final answer line"), ("h_answer", "all answer tokens"), ("h_prompt", "before answering")):
                for li, ll in ((na // 2, "middle layer"), (na - 2, "late layer")):
                    out[f"answerer hidden state: {lab} ({ll})"] = (F.apply_reduce(F.fit_reduce(self.ha[key][tr, li, :]), self.ha[key][:, li, :]), "A")
            out["answerer: evidence-use attention"] = (self.att, "A")
            out["answerer: token probabilities"] = (self.tok, "A")
            out["answerer: stated confidence and lengths"] = (self.conf, "A")
        return out


def zs(a, ref):
    return (a - np.nanmean(ref, 0)) / (np.nanstd(ref, 0) + 1e-9)


def fauc(y, s):
    """Fast separation (area under the ROC curve) by ranks; nan when only one class is present."""
    from scipy.stats import rankdata
    y = np.asarray(y) > 0.5
    n1 = int(y.sum())
    n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def chance_bar(y):
    n1, n0 = float(y.sum()), float(len(y) - y.sum())
    return 0.5 + 1.645 * np.sqrt((n1 + n0 + 1) / (12 * n1 * n0))            # 95th percentile of separation when there is no signal


def cvp(make, X, y, seed=0):
    """Out-of-fold predictions of a combining model on the training side (so that thresholds are not set on fitted scores)."""
    from sklearn.model_selection import StratifiedKFold
    p = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        p[b] = make(X[a], y[a])(X[b])
    return p


def pipeline(P, tr, te, decision, reps=2):
    """Fits everything on rows tr and scores rows te. Returns the base table and, per ensemble, (training-side scores, test scores)."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    y = P.y[decision]
    ytr = y[tr]
    fams = P.families(tr, decision)
    base, So, Sn = [], [], []
    for name, (X, group) in fams.items():
        for rd, fn in READERS.items():
            Xtr, Xte = X[tr], X[te]
            po = np.nanmean([H.cv(fn, Xtr, ytr, 10 + r) for r in range(reps)], 0)
            try:
                pn = fn(Xtr, ytr.astype(int), Xte)
            except Exception:
                pn = np.full(len(te), ytr.mean())
            base.append({"Signal family": name, "Group": GROUPS[group], "g": group, "Reader": rd, "train": H.auc(ytr, po), "test": H.auc(y[te], pn)})
            So.append(po)
            Sn.append(pn)
    So, Sn = np.nan_to_num(np.array(So).T, nan=float(ytr.mean())), np.nan_to_num(np.array(Sn).T, nan=float(ytr.mean()))
    Zo, Zn = zs(So, So), zs(Sn, So)
    tr_auc = np.array([b["train"] if b["train"] == b["train"] else 0.5 for b in base])
    lin = np.array([b["Reader"] == "linear" for b in base])
    grp = np.array([b["g"] for b in base])
    ens, notes = {}, {}
    best = int(np.argmax(tr_auc))
    ens[REFERENCE] = (Zo[:, best], Zn[:, best])
    notes[REFERENCE] = f"{base[best]['Signal family']} | {base[best]['Reader']}"
    fixed_name = ("answerer hidden state: final answer line (late layer)" if decision == "answerer" else "retriever hidden state (middle layer)")
    fx = [i for i, b_ in enumerate(base) if b_["Signal family"] == fixed_name and b_["Reader"] == "linear"][0]
    ens[FIXED] = (Zo[:, fx], Zn[:, fx])
    notes[FIXED] = f"{fixed_name} | linear (the kind of probe tested in notebooks 17 and 18)"
    bar = chance_bar(ytr)
    keep = tr_auc > bar
    if keep.sum() == 0:
        keep = tr_auc >= np.sort(tr_auc)[-3]
    ens[PRIMARY] = (Zo[:, keep].mean(1), Zn[:, keep].mean(1))
    notes[PRIMARY] = f"{int(keep.sum())} of {len(base)} base models above {bar:.2f}; groups: " + ", ".join(f"{g} {int((keep & (grp == g)).sum())}" for g in "QRA" if (grp == g).any())
    ens["Average of all base models"] = (Zo.mean(1), Zn.mean(1))
    ens["Average of the linear reader of every family"] = (Zo[:, lin].mean(1), Zn[:, lin].mean(1))
    gs = [g for g in "QRA" if (grp == g).any()]
    ens["Average with equal weight for each group (question, retriever, answerer)"] = (np.mean([Zo[:, grp == g].mean(1) for g in gs], 0), np.mean([Zn[:, grp == g].mean(1) for g in gs], 0))
    for g in gs:
        ens[f"Only {GROUPS[g].split(' (')[0]}: average of its base models"] = (Zo[:, grp == g].mean(1), Zn[:, grp == g].mean(1))
    chosen, cur = [], -1.0                                                     # greedy selection: add the base model that helps the average most
    for _ in range(8):
        cand = [(fauc(ytr, Zo[:, chosen + [j]].mean(1)), j) for j in range(len(base)) if j not in chosen]
        a, j = max(cand)
        if a <= cur + 1e-4:
            break
        chosen.append(j)
        cur = a
    ens["Greedy selection of base models that add to each other"] = (Zo[:, chosen].mean(1), Zn[:, chosen].mean(1))
    notes["Greedy selection of base models that add to each other"] = "; ".join(f"{base[j]['Signal family']} | {base[j]['Reader']}" for j in chosen)

    def stack(name, make):
        try:
            ens[name] = (cvp(make, Zo, ytr), make(Zo, ytr)(Zn))
        except Exception as ex:
            notes[name] = f"failed: {type(ex).__name__}"
    stack("Stacked logistic regression over the base scores", lambda X, yy: (lambda m: (lambda Z: m.predict_proba(Z)[:, 1]))(LogisticRegression(C=0.1, max_iter=500).fit(X, yy.astype(int))))
    stack("Stacked logistic regression, classes weighted equally",
          lambda X, yy: (lambda m: (lambda Z: m.predict_proba(Z)[:, 1]))(LogisticRegression(C=0.1, max_iter=500, class_weight="balanced").fit(X, yy.astype(int))))

    def smote_lr(X, yy):
        from imblearn.over_sampling import SMOTE
        Xr, yr = SMOTE(random_state=0, k_neighbors=min(5, int(min(yy.sum(), len(yy) - yy.sum())) - 1)).fit_resample(X, yy.astype(int))
        m = LogisticRegression(C=0.1, max_iter=500).fit(Xr, yr)
        return lambda Z: m.predict_proba(Z)[:, 1]
    stack("Stacked logistic regression with SMOTE", smote_lr)
    stack("Stacked boosted trees over the base scores",
          lambda X, yy: (lambda m: (lambda Z: m.predict_proba(Z)[:, 1]))(GradientBoostingClassifier(n_estimators=40, max_depth=2, subsample=0.8, random_state=0).fit(X, yy.astype(int))))
    Xall = np.column_stack([X for X, _ in fams.values()])
    for nm, fn in (("All features of all families in one linear model", H.f_linear), ("All features of all families in one boosted-tree model", H.f_boost)):
        po = np.nanmean([H.cv(fn, Xall[tr], ytr, 10 + r) for r in range(reps)], 0)
        ens[nm] = (np.nan_to_num(po, nan=float(ytr.mean())), fn(Xall[tr], ytr.astype(int), Xall[te]))
    corr = np.corrcoef(Sn.T)
    return base, ens, notes, corr


SHARE = {"answerer": 0.40, "retriever": 0.20}


def outcomes(P, decision):
    """(what you get if the gate acts, what you get if it does not) for the outcome the decision is meant to improve, and for accuracy."""
    if decision == "answerer":
        return {"accuracy": (P.A["e1"], P.l8)}
    return {"both passages found": (P.A["f3"], P.A["f1"]), "accuracy": (P.A["e3"], P.A["e1"])}


def evaluate(P, decision, name, folds, rows, base_rows, extra, B=1000):
    """folds: list of (train rows, test rows). Scores are pooled over the test rows of all folds."""
    y = P.y[decision]
    acc = {}
    for fi, (tr, te) in enumerate(folds):
        base, ens, notes, corr = pipeline(P, tr, te, decision)
        for b in base:
            base_rows.append(dict({k: v for k, v in b.items() if k != "g"}, Decision=decision, Evaluation=name, Fold=fi))
        for k, (po, pn) in ens.items():
            thr = np.quantile(po, SHARE[decision])
            d = acc.setdefault(k, {"idx": [], "p": [], "act": [], "fold": [], "note": []})
            d["idx"].append(te); d["p"].append(pn); d["act"].append(pn <= thr); d["fold"].append(np.full(len(te), fi)); d["note"].append(notes.get(k, ""))
        extra.setdefault((decision, name), []).append({"corr": corr, "base": base})
        log(f"  {decision} | {name} | fold {fi + 1}/{len(folds)} done")
    rng = np.random.default_rng(0)
    ref, fxd = acc[REFERENCE], acc[FIXED]
    ridx, rp, rfold = np.concatenate(ref["idx"]), np.concatenate(ref["p"]), np.concatenate(ref["fold"])
    xp = np.concatenate(fxd["p"])                                   # same row order as every other ensemble

    def pooled_auc(idx, p, fold):
        """Mean over folds of the separation inside each fold (scores of different folds are not comparable)."""
        v = []
        for f_ in np.unique(fold):
            m = fold == f_
            a = fauc(y[idx[m]], p[m])
            if a == a:
                v.append(a)
        return float(np.mean(v)) if v else float("nan")
    # resample QUESTIONS, not rows: with repeated cross-validation every question appears more than once
    qlist = np.unique(ridx)
    where = {q: np.where(ridx == q)[0] for q in qlist}
    reps_ix = [np.concatenate([where[q] for q in rng.choice(qlist, len(qlist))]) for _ in range(B)]
    ci = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5)))
    for k, d in acc.items():
        idx, p, act, fold = np.concatenate(d["idx"]), np.concatenate(d["p"]), np.concatenate(d["act"]).astype(bool), np.concatenate(d["fold"])
        assert (idx == ridx).all()
        a = pooled_auc(idx, p, fold)
        boots, dboots, xboots = [], [], []
        for rep in reps_ix:
            ab = pooled_auc(idx[rep], p[rep], fold[rep])
            boots.append(ab)
            dboots.append(ab - pooled_auc(idx[rep], rp[rep], fold[rep]))
            xboots.append(ab - pooled_auc(idx[rep], xp[rep], fold[rep]))
        row = {"Decision": decision, "Evaluation": name, "Ensemble": k, "Separation": a, "95% interval, low": ci(boots)[0], "95% interval, high": ci(boots)[1],
               "Minus the best single base model": a - pooled_auc(ridx, rp, rfold), "Difference: 95% interval, low": ci(dboots)[0], "Difference: 95% interval, high": ci(dboots)[1],
               "Minus the fixed probe": a - pooled_auc(ridx, xp, rfold), "Minus the fixed probe: low": ci(xboots)[0], "Minus the fixed probe: high": ci(xboots)[1],
               "Share acted on": float(act.mean()), "Built from": " || ".join(sorted(set(n for n in d["note"] if n)))[:400]}
        for oname, (if_act, if_not) in outcomes(P, decision).items():
            ia, inn = if_act[idx], if_not[idx]
            got = np.where(act, ia, inn)
            rnd = act.mean() * ia.mean() + (1 - act.mean()) * inn.mean()
            db = [got[w].mean() - (act[w].mean() * ia[w].mean() + (1 - act[w].mean()) * inn[w].mean()) for w in reps_ix]
            row[f"Gate: {oname}"] = float(got.mean())
            row[f"Random gate, same share: {oname}"] = float(rnd)
            row[f"Gate minus random: {oname} (points)"] = 100 * float(got.mean() - rnd)
            row[f"Gate minus random: {oname}, low"] = 100 * ci(db)[0]
            row[f"Gate minus random: {oname}, high"] = 100 * ci(db)[1]
        # against the cheap fixed setup (Qwen 7B retrieves once, Qwen 32B answers once)
        bar_em, bar_c = P.A["e1"][idx], P.A["c1"][idx]
        if decision == "answerer":
            got = np.where(act, P.A["e1"][idx], P.l8[idx])
            cost = P.A["fr1"][idx] + P.A["cl8"][idx] + act * (P.A["c1"][idx] - P.A["fr1"][idx])
        else:
            got = np.where(act, P.A["e3"][idx], P.A["e1"][idx])
            cost = np.where(act, P.A["c3"][idx], P.A["c1"][idx])
        db = [got[w].mean() - bar_em[w].mean() for w in reps_ix]
        row.update({"Cheap fixed setup: accuracy": float(bar_em.mean()), "Gate accuracy minus the cheap fixed setup (points)": 100 * float(got.mean() - bar_em.mean()),
                    "vs cheap fixed setup, low": 100 * ci(db)[0], "vs cheap fixed setup, high": 100 * ci(db)[1], "Gate compute vs the cheap fixed setup": float(cost.mean() / bar_c.mean() - 1)})
        rows.append(row)


def cmd_run(args):
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    t0 = time.time()
    out = os.path.join(args.run_dir, "results")
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    log(f"ensemble code version {VERSION}")
    old = load_set("old", args.old_sweep, args.old_data, os.path.join(args.old_sweep, "models.json"), args.old_hidden)
    new = load_set("fresh", args.fresh_run, args.fresh_data, os.path.join(args.fresh_run, "models.json"), args.fresh_run)
    P = Pool([old, new], use_embedding=not args.no_embedding)
    log(f"{P.n} questions ({(P.origin == 'old').sum()} old, {(P.origin == 'fresh').sum()} fresh); question embedding: {P.emb_note}")
    io, inew = np.where(P.origin == "old")[0], np.where(P.origin == "fresh")[0]
    rows, base_rows, extra = [], [], {}
    for decision in ("answerer", "retriever"):
        y = P.y[decision]
        evaluate(P, decision, "old -> fresh", [(io, inew)], rows, base_rows, extra)
        evaluate(P, decision, "fresh -> old", [(inew, io)], rows, base_rows, extra)
        folds = []
        strat = np.array([f"{o}{int(v)}" for o, v in zip(P.origin, y)])
        for r in range(args.cv_repeats):
            folds += [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=100 + r).split(np.zeros(P.n), strat)]
        evaluate(P, decision, "all 400, cross-validated", folds, rows, base_rows, extra)
    bdf = pd.DataFrame(base_rows)
    fam = bdf.groupby(["Decision", "Evaluation", "Group", "Signal family", "Reader"], sort=False)[["train", "test"]].mean().reset_index()
    fam = fam.rename(columns={"train": "Separation on the training questions (held-out)", "test": "Separation on the test questions"})
    # verdict for the primary ensemble
    verdict = {}
    for decision in ("answerer", "retriever"):
        g = {r["Evaluation"]: r for r in rows if r["Decision"] == decision and r["Ensemble"] == PRIMARY}
        cvr = g["all 400, cross-validated"]
        ok = bool(cvr["Difference: 95% interval, low"] > 0 and g["old -> fresh"]["Minus the best single base model"] > 0 and g["fresh -> old"]["Minus the best single base model"] > 0)
        key = "accuracy" if decision == "answerer" else "both passages found"
        gate_ok = bool(cvr[f"Gate minus random: {key}, low"] > 0 and g["old -> fresh"][f"Gate minus random: {key} (points)"] > 0 and g["fresh -> old"][f"Gate minus random: {key} (points)"] > 0)
        fix_ok = bool(cvr["Minus the fixed probe: low"] > 0)
        bar_txt = (f"{cvr['Gate accuracy minus the cheap fixed setup (points)']:+.1f} points of accuracy (95% interval {cvr['vs cheap fixed setup, low']:+.1f} to "
                   f"{cvr['vs cheap fixed setup, high']:+.1f}) at {100 * cvr['Gate compute vs the cheap fixed setup']:+.0f}% compute")
        verdict[decision] = {K_ADD: ok, K_GATE: gate_ok, K_FIX: fix_ok, K_BAR: bar_txt,
                             "rows": {k: {kk: vv for kk, vv in v.items() if kk != "Built from"} for k, v in g.items()}}
    overlap = []
    for (decision, name), lst in extra.items():
        C = np.mean([e["corr"] for e in lst], 0)
        base = lst[0]["base"]
        grp = np.array([b["g"] for b in base])
        for a in "QRA":
            for b in "QRA":
                if a <= b and (grp == a).any() and (grp == b).any():
                    M = C[np.ix_(grp == a, grp == b)]
                    v = M[np.triu_indices(len(M), 1)] if a == b else M.ravel()
                    overlap.append({"Decision": decision, "Evaluation": name, "Between": f"{GROUPS[a].split(' (')[0]} and {GROUPS[b].split(' (')[0]}",
                                    "Mean correlation of scores on the test questions": float(np.nanmean(v))})

    def dump(nm, rr):
        with open(os.path.join(out, nm), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    dump("ensembles.jsonl", rows); dump("families.jsonl", fam.to_dict("records")); dump("overlap.jsonl", overlap)
    S.write_json(os.path.join(out, "verdict.json"), {"version": VERSION, "embedding": P.emb_note, "seconds": round(time.time() - t0), **verdict})
    charts(os.path.join(out, "charts"), rows, fam, extra)
    # ---------------------------------------------------------------- summary
    def T(rr, cols, pct=(), signed=(), spct=()):
        def f(r, c):
            v = r.get(c)
            if v is None or (isinstance(v, float) and v != v):
                return "-"
            if isinstance(v, str):
                return v
            if c in signed:
                return f"{v:+.2f}" if abs(v) < 1 else f"{v:+.1f}"
            if c in spct:
                return f"{100 * v:+.1f}%"
            if c in pct:
                return f"{100 * v:.1f}%"
            return f"{v:.2f}"
        return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rr)
    L = ["# Set 5f - do signals from different families add up? Cross-family ensembles\n",
         f"Code version {VERSION}. {P.n} questions (200 old, 200 fresh). No model was run; everything comes from notebooks 11, 16, 17 and 18. Question embedding: {P.emb_note}.",
         "Separation: 0.5 = useless, 1.0 = perfect. Every number is on questions that the models and thresholds did not see.\n",
         "The fresh questions were already used to develop ideas in steps 5c to 5e, so none of this is an untouched test; the three evaluations guard against "
         "reading too much into one sample.\n"]
    for decision, title, key in (("answerer", "Decision 1: after one Llama 8B answer, keep it or let the Qwen 32B answer?", "accuracy"),
                                 ("retriever", "Decision 2: after one Qwen 7B retrieval try, stop or try twice more?", "both passages found")):
        v = verdict[decision]
        L += [f"## {title}\n",
              f"**Does the primary ensemble add value over the best single base model?** {'Yes' if v[K_ADD] else 'No'}. "
              f"**Does it beat one fixed hidden-state probe (the kind tested in notebooks 17 and 18)?** {'Yes' if v[K_FIX] else 'No'}. "
              f"**Does its gate beat a random gate?** {'Yes' if v[K_GATE] else 'No'}. "
              f"**Against the cheap fixed setup (all 400 questions):** {v[K_BAR]}.\n",
              "The 'best single base model' is picked among dozens on the training questions, which makes it a noisy and rather weak comparator; "
              "the fixed probe is the fairer one.\n",
              "### Each signal family on its own (best reader of each family; mean over the three evaluations)\n"]
        g = fam[fam.Decision == decision]
        piv = g.groupby(["Group", "Signal family", "Evaluation"], sort=False)["Separation on the test questions"].max().unstack("Evaluation").reset_index()
        piv["Mean"] = piv[[c for c in piv.columns if c not in ("Group", "Signal family")]].mean(1)
        fr = piv.sort_values(["Group", "Mean"], ascending=[True, False]).to_dict("records")
        L += [T(fr, ["Group", "Signal family", "old -> fresh", "fresh -> old", "all 400, cross-validated", "Mean"]),
              "\n'Best reader' is picked per cell after the fact, so these single-family numbers are slightly flattered; the ensembles below are not.\n",
              "### How much do the groups overlap? (correlation of their scores; low = they see different things)\n",
              T([o for o in overlap if o["Decision"] == decision and o["Evaluation"] == "all 400, cross-validated"], ["Between", "Mean correlation of scores on the test questions"]),
              "\n### The ensembles\n"]
        for ev in ("all 400, cross-validated", "old -> fresh", "fresh -> old"):
            rr = sorted([r for r in rows if r["Decision"] == decision and r["Evaluation"] == ev], key=lambda r: (r["Ensemble"] != REFERENCE, r["Ensemble"] != FIXED, r["Ensemble"] != PRIMARY, -r["Separation"]))
            L += [f"**{ev}**\n",
                  "Separation, and whether the ensemble is ahead of a single model:\n",
                  T(rr, ["Ensemble", "Separation", "95% interval, low", "95% interval, high", "Minus the best single base model", "Difference: 95% interval, low",
                         "Difference: 95% interval, high", "Minus the fixed probe", "Minus the fixed probe: low", "Minus the fixed probe: high"],
                    signed={"Minus the best single base model", "Difference: 95% interval, low", "Difference: 95% interval, high", "Minus the fixed probe",
                            "Minus the fixed probe: low", "Minus the fixed probe: high"}),
                  "\nUsed as a gate:\n",
                  T(rr, ["Ensemble", "Share acted on", f"Gate: {key}", f"Random gate, same share: {key}", f"Gate minus random: {key} (points)",
                         f"Gate minus random: {key}, low", f"Gate minus random: {key}, high", "Gate accuracy minus the cheap fixed setup (points)",
                         "vs cheap fixed setup, low", "vs cheap fixed setup, high", "Gate compute vs the cheap fixed setup"],
                    pct={"Share acted on", f"Gate: {key}", f"Random gate, same share: {key}"}, spct={"Gate compute vs the cheap fixed setup"},
                    signed={f"Gate minus random: {key} (points)", f"Gate minus random: {key}, low", f"Gate minus random: {key}, high",
                            "Gate accuracy minus the cheap fixed setup (points)", "vs cheap fixed setup, low", "vs cheap fixed setup, high"}), ""]
        pr = [r for r in rows if r["Decision"] == decision and r["Ensemble"] in (PRIMARY, REFERENCE, FIXED, "Greedy selection of base models that add to each other") and r["Evaluation"] == "old -> fresh"]
        L += ["What they were built from (old -> fresh):\n"] + [f"- {r['Ensemble']}: {r['Built from']}" for r in pr] + [""]
    n_e = len([r for r in rows if r["Ensemble"] not in (REFERENCE, FIXED)])
    L += ["## How to read the many rows\n",
          f"There are {n_e} ensemble-by-evaluation rows. At a 95% level about {round(0.05 * n_e)} of them would show an interval that excludes zero by chance alone, "
          "which is why only the primary ensemble, named beforehand, decides the answer.\n",
          "Charts: `results/charts/`. Excel: `results/set5f_ensembles.xlsx`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    W.write_excel(os.path.join(out, "set5f_ensembles.xlsx"), {"Ensembles": pd.DataFrame(rows), "Signal families": fam, "Overlap between groups": pd.DataFrame(overlap),
                                                              "Every base model, every fold": bdf.drop(columns=["g"], errors="ignore")})
    for d_, v in verdict.items():
        log(f"{d_}: primary ensemble beats the best single base model = {v[K_ADD]}; beats one fixed probe = {v[K_FIX]}; its gate beats random = {v[K_GATE]}; vs cheap fixed setup: {v[K_BAR]}")
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def charts(out, rows, fam, extra):
    plt = S._plt()
    C = S.SERIES
    evs = ["all 400, cross-validated", "old -> fresh", "fresh -> old"]
    for decision in ("answerer", "retriever"):
        # 1. each family on its own
        g = fam[fam.Decision == decision]
        piv = g.groupby(["Group", "Signal family", "Evaluation"], sort=False)["Separation on the test questions"].max().unstack("Evaluation")
        piv = piv.loc[piv.mean(1).sort_values().index]
        fig, ax = plt.subplots(figsize=(9.5, 0.36 * len(piv) + 1.8))
        yy = np.arange(len(piv))
        gcol = {GROUPS["Q"]: C[0], GROUPS["R"]: C[2], GROUPS["A"]: C[1]}
        for j, ev in enumerate(evs):
            ax.scatter(piv[ev], yy, marker=["o", "s", "^"][j], s=42, color=[gcol[i[0]] for i in piv.index], alpha=0.9, label=ev, zorder=3)
        ax.hlines(yy, piv.min(1), piv.max(1), color=S.MUTED, lw=1, zorder=2)
        ax.axvline(0.5, color=S.INK2, ls=":", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([i[1] for i in piv.index], fontsize=8)
        for t, i in zip(ax.get_yticklabels(), piv.index):
            t.set_color(gcol[i[0]])
        ax.set_xlim(0.35, 0.9)
        ax.set_xlabel("separation on unseen questions (0.5 = useless)")
        ax.legend(frameon=False, fontsize=8, loc="lower right", title="marker = evaluation; colour = group", title_fontsize=7)
        ax.set_title(f"{decision.capitalize()} decision: each signal family on its own (blue = question, green = retriever, orange = answerer)", fontsize=9)
        fig.savefig(os.path.join(out, f"1_families_{decision}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        # 2. overlap heatmap (cross-validated)
        lst = extra[(decision, "all 400, cross-validated")]
        Cm = np.mean([e["corr"] for e in lst], 0)
        base = lst[0]["base"]
        keep = [i for i, b in enumerate(base) if b["Reader"] == "linear"]
        fig, ax = plt.subplots(figsize=(0.42 * len(keep) + 4.5, 0.42 * len(keep) + 2))
        im = ax.imshow(Cm[np.ix_(keep, keep)], cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(keep)))
        ax.set_yticks(range(len(keep)))
        ax.set_xticklabels([base[i]["Signal family"] for i in keep], rotation=90, fontsize=7)
        ax.set_yticklabels([base[i]["Signal family"] for i in keep], fontsize=7)
        ax.grid(False)
        fig.colorbar(im, ax=ax, shrink=0.7)
        ax.set_title(f"{decision.capitalize()} decision: how much the families' scores overlap (correlation; linear readers)", fontsize=9)
        fig.savefig(os.path.join(out, f"2_overlap_{decision}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
        # 3. ensembles against the best single base model
        names = [r["Ensemble"] for r in rows if r["Decision"] == decision and r["Evaluation"] == evs[0]]
        names = sorted(names, key=lambda k: (k == REFERENCE, k == FIXED, k == PRIMARY))
        fig, axs = plt.subplots(1, 3, figsize=(15, 0.36 * len(names) + 2.2), sharey=True)
        for ax, ev in zip(axs, evs):
            rr = {r["Ensemble"]: r for r in rows if r["Decision"] == decision and r["Evaluation"] == ev}
            ref = rr[REFERENCE]["Separation"]
            for i, k in enumerate(names):
                r = rr[k]
                col = S.INK if k in (REFERENCE, FIXED) else (C[0] if k == PRIMARY else S.MUTED)
                ax.errorbar([r["Separation"]], [i], xerr=[[r["Separation"] - r["95% interval, low"]], [r["95% interval, high"] - r["Separation"]]], fmt="o", color=col, ms=6 if k in (PRIMARY, REFERENCE, FIXED) else 4, lw=1.2)
            ax.axvline(ref, color=S.INK, ls="--", lw=1)
            ax.axvline(0.5, color=S.INK2, ls=":", lw=1)
            ax.set_title(ev, fontsize=9)
            ax.set_xlabel("separation on unseen questions")
            ax.set_xlim(0.35, 0.9)
        axs[0].set_yticks(range(len(names)))
        axs[0].set_yticklabels([n[:70] for n in names], fontsize=7)
        fig.suptitle(f"{decision.capitalize()} decision: every ensemble with its 95% interval (dashed = the best single base model; blue = primary ensemble)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, f"3_ensembles_{decision}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    for a in ("old_sweep", "old_hidden", "old_data", "fresh_run", "fresh_data", "run_dir"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--cv_repeats", type=int, default=2)
    p.add_argument("--no_embedding", action="store_true")
    args = ap.parse_args()
    cmd_run(args)


if __name__ == "__main__":
    main()
