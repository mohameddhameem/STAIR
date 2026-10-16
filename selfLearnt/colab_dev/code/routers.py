"""Set 5a - offline switching policies on the saved Set 4 sweep (no GPU, no new model answers).

Every policy is built from results that already exist in the sweep's results/per_question.jsonl:
  A1 answerer cascade        cheap answerer first, bigger answerer only if its 5 answers disagree (retrieval shared)
  A2 retriever cascade       small retriever first, bigger retriever only if its 3 tries disagree
  A3 both cascades           A2 then A1
  A4 big model, 1 try        A1 / A3 with a single answer try for the bigger answerer
  A5 learned router, text    reads the question text; chooses between a cheap and a big setup before running anything
  A6 learned router, 5 facts the Set 4 inputs (yes/no, comparison, words, digit, names)
  A8 learned cascade         like A1 but a small trained model decides from the cheap run's metadata
  A9 learned cascade + text  A8 with the question text added
  A7 Set 4 rules, smoothed   the planner's Set 4 rules, with each rule's dev accuracy shrunk by an m-estimate

Selection rule (the same for fixed setups and for every policy family): on the training questions, take the
cheapest candidate whose accuracy is within TOL of the most accurate fixed setup. Scores are always held-out.
"""
import argparse
import warnings
import json
import os
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S          # noqa: E402
import sweep as W        # noqa: E402

warnings.filterwarnings("ignore", message="Mean of empty slice")
VERSION = "5a.1"
TOL = 0.02                                   # "within 2 points of the most accurate fixed setup"
ANS_T = [0.4, 0.6, 0.8, 1.0]                 # accept the cheap answer if agreement of its 5 answers >= t
RET_T = [0.4, 0.5, 0.6, 0.7, 0.8, 1.0]       # keep the small retriever's shortlist if agreement of its 3 tries >= t
CONDS = [(c, p, m) for c in (0, 1) for p in ("silent", "3") for m in ("vote", "all")]
RET_PAIRS = [("q3", "q7"), ("q3", "q32"), ("q7", "q32")]
ANS_PAIRS = [("l3", "l8"), ("l3", "q32"), ("l8", "q32")]
TAUS = [0.2, 0.35, 0.5, 0.65, 0.8]           # learned gates: escalate the questions below this quantile of the gate's score
FAMILY_LABEL = {
    "fixed_best": "Fixed: most accurate setup",
    "fixed_cheap": "Fixed: cheapest setup within 2 points of it",
    "A1": "A1 answerer cascade",
    "A2": "A2 retriever cascade",
    "A3": "A3 retriever + answerer cascade",
    "A4": "A4 cascade, big answerer with 1 try",
    "A5_text": "A5 learned router, question text (TF-IDF)",
    "A5_emb": "A5 learned router, question text (embedding)",
    "A6_facts": "A6 learned router, 5 facts",
    "A8_meta": "A8 learned cascade, cheap run's metadata",
    "A9_meta_text": "A9 learned cascade, metadata + question text",
}
ORDER = list(FAMILY_LABEL)


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def agree_words(t, n=5):
    k = int(round(t * n))
    return "all 5 agree" if k >= n else f"at least {k} of {n} agree"


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~np.isnan(s)
    y, s = y[ok], s[ok]
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


# ==================================================================================
# Data
# ==================================================================================
class Data:
    def __init__(self, sweep_run, data_dir, keep_df=False):
        import pandas as pd
        qs = S.load_questions(data_dir)
        self.qids = [q["qid"] for q in qs]
        self.split = np.array([q["split"] for q in qs])
        self.text = [q["question"] for q in qs]
        self.n = len(qs)
        log(f"reading {sweep_run}/results/per_question.jsonl ...")
        df = pd.read_json(os.path.join(sweep_run, "results", "per_question.jsonl"), lines=True, dtype={"qid": str})
        df["found_both"] = (df["found"] == "both").astype(float)
        df["flops_tail"] = df["flops_ans"] + df["flops_fuse"]
        for c in ("ans_conf", "ret_conf", "ans_token_prob", "ret_token_prob"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        self.setups = sorted(df.setup.unique())
        self.col = {s: i for i, s in enumerate(self.setups)}

        def piv(v):
            return df.pivot(index="qid", columns="setup", values=v).reindex(index=self.qids, columns=self.setups).to_numpy(float)
        self.EM, self.FL, self.FR, self.FA = piv("EM"), piv("flops"), piv("flops_ret"), piv("flops_tail")
        self.AG, self.RG, self.AC, self.RC = piv("ans_agree"), piv("ret_agree"), piv("ans_conf"), piv("ret_conf")
        self.TP, self.FOUND = piv("ans_token_prob"), piv("found_both")
        assert not np.isnan(self.EM).any(), "the sweep is not complete: some setups have no result for some questions"
        m = df.drop_duplicates("setup").set_index("setup")
        self.meta = {s: (m.loc[s, "retriever"], m.loc[s, "answerer"], int(m.loc[s, "cot"]), str(m.loc[s, "pick"]), str(m.loc[s, "merge"]),
                         int(m.loc[s, "rk"]), int(m.loc[s, "ak"])) for s in self.setups}
        self.df = df if keep_df else None
        self.ref = float(self.FL[:, self.j("q3", "l3", 0, "silent", "vote")].mean())      # the cheapest main setup = 1.0x
        self.dev = np.where(self.split == "dev")[0]
        self.test = np.where(self.split == "test")[0]
        log(f"{self.n} questions ({len(self.dev)} dev, {len(self.test)} test), {len(self.setups)} setups")

    def j(self, r, a, c, p, m, rk=3, ak=5):
        return self.col.get(W.setup_id(r, a, c, p, m, rk, ak))

    def words(self, s):
        r, a, c, p, m, rk, ak = self.meta[s]
        return (f"{W.LABEL[r]} retrieves x{rk}, {W.LABEL[a]} answers x{ak} "
                f"(CoT {'on' if c else 'off'}, per-attempt shortlist {W.PICK_LABEL[p]}, final shortlist {W.MERGE_LABEL[m]})")


def cond_words(cond):
    c, p, m = cond
    return f"CoT {'on' if c else 'off'}, per-attempt shortlist {W.PICK_LABEL[p]}, final shortlist {W.MERGE_LABEL[m]}"


# ==================================================================================
# Candidates. A candidate = a list of paths (what is run, with per-question EM / cost / found-both) and,
# for every question, which path the policy takes.
# ==================================================================================
def cand(family, name, params, paths, choice, signal=None, target=None):
    idx = np.arange(len(choice))
    em = np.stack([p["em"] for p in paths])[choice, idx]
    cost = np.stack([p["cost"] for p in paths])[choice, idx]
    found = np.stack([p["found"] for p in paths])[choice, idx]
    return {"family": family, "name": name, "params": params, "paths": paths, "choice": choice, "em": em, "cost": cost, "found": found,
            "signal": signal, "target": target}


def cascade(D, family, rs, rb, a_s, a_b, cond, t_r, t_a, ak_big):
    """Small retriever rs (3 tries) -> optionally bigger retriever rb; answerer a_s (5 tries) -> optionally bigger answerer a_b."""
    c, p, m = cond
    js = D.j(rs, a_s, c, p, m)
    if js is None:
        return None
    go_rb = (D.RG[:, js] < t_r) if rb else np.zeros(D.n, bool)
    paths, choice = [], np.zeros(D.n, int)
    signal, target = None, None
    for r, use, extra in ((rs, ~go_rb, 0.0), (rb, go_rb, D.FR[:, js])):
        if r is None:
            continue
        j1 = D.j(r, a_s, c, p, m)
        k = len(paths)
        tag = f"{W.LABEL[r]} retrieval" + (" (after the small retriever's tries)" if r == rb else "")
        paths.append({"name": f"{tag}; {W.LABEL[a_s]} answer kept", "em": D.EM[:, j1], "cost": extra + D.FL[:, j1], "found": D.FOUND[:, j1],
                      "ret": r, "esc_ans": False})
        choice[use] = k
        if a_b:
            j2 = D.j(r, a_b, c, p, m, 3, ak_big)
            esc = D.AG[:, j1] < t_a
            paths.append({"name": f"{tag}; escalated to {W.LABEL[a_b]} x{ak_big}", "em": D.EM[:, j2],
                          "cost": extra + D.FL[:, j1] + D.FA[:, j2], "found": D.FOUND[:, j2], "ret": r, "esc_ans": True})
            choice[use & esc] = k + 1
            if signal is None:
                signal, target = np.full(D.n, np.nan), np.full(D.n, np.nan)
            signal[use] = D.AG[use, j1]
            target[use] = (D.EM[use, j1] >= D.EM[use, j2]).astype(float)
    if signal is None:                                        # retriever cascade only: the signal is the retrieval agreement
        signal = D.RG[:, js].copy()
        target = (D.EM[:, js] >= D.EM[:, D.j(rb, a_s, c, p, m)]).astype(float)
    bits = []
    if rb:
        bits.append(f"{W.LABEL[rs]} retrieves; {W.LABEL[rb]} retrieves again if agreement of the 3 tries is below {t_r:g}")
    else:
        bits.append(f"{W.LABEL[rs]} retrieves")
    if a_b:
        bits.append(f"{W.LABEL[a_s]} answers x5; {W.LABEL[a_b]} x{ak_big} answers unless {agree_words(t_a)}")
    else:
        bits.append(f"{W.LABEL[a_s]} answers x5")
    params = {"small_retriever": rs, "big_retriever": rb, "cheap_answerer": a_s, "big_answerer": a_b, "cot": c, "pick": p, "merge": m,
              "retrieval_agreement_threshold": t_r if rb else None, "answer_agreement_threshold": t_a if a_b else None,
              "big_answer_tries": ak_big if a_b else None}
    return cand(family, "; ".join(bits) + f" ({cond_words(cond)})", params, paths, choice, signal, target)


def threshold_candidates(D):
    out = defaultdict(list)
    for cond in CONDS:
        for r in W.RETRIEVERS:
            for a_s, a_b in ANS_PAIRS:
                for t in ANS_T:
                    out["A1"].append(cascade(D, "A1", r, None, a_s, a_b, cond, None, t, 5))
                    out["A4"].append(cascade(D, "A4", r, None, a_s, a_b, cond, None, t, 1))
        for rs, rb in RET_PAIRS:
            for a in W.ANSWERERS:
                for t in RET_T:
                    out["A2"].append(cascade(D, "A2", rs, rb, a, None, cond, t, None, None))
            for a_s, a_b in ANS_PAIRS:
                for tr in RET_T:
                    for ta in ANS_T:
                        out["A3"].append(cascade(D, "A3", rs, rb, a_s, a_b, cond, tr, ta, 5))
                        out["A4"].append(cascade(D, "A4", rs, rb, a_s, a_b, cond, tr, ta, 1))
    return {k: [c for c in v if c is not None] for k, v in out.items()}


# ---------------------------------------------------------------------------------- learned gates
def make_lr():
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(), LogisticRegression(C=0.3, max_iter=500))


def fit_predict(Xtr, ytr, Xte):
    if ytr.min() == ytr.max():
        return np.full(len(Xte), float(ytr[0]))
    return make_lr().fit(Xtr, ytr).predict_proba(Xte)[:, 1]


def oof(X, y, seed, k=5):
    from sklearn.model_selection import KFold
    p = np.zeros(len(y))
    for tr, te in KFold(k, shuffle=True, random_state=seed).split(X):
        p[te] = fit_predict(X[tr], y[tr], X[te])
    return p


def gate_candidates(D, family, configs, Xfn, train, evalq, seed):
    """configs: (label, params, cheap path, big path). The gate predicts 'the cheap path is no worse than the big one'."""
    out = []
    for label, params, cheap, big in configs:
        X = Xfn(params)
        y = (cheap["em"] >= big["em"]).astype(int)
        p = np.full(D.n, np.nan)
        p[train] = oof(X[train], y[train], seed)
        if evalq is not None and len(evalq):
            p[evalq] = fit_predict(X[train], y[train], X[evalq])
        for tq in TAUS:
            thr = float(np.quantile(p[train], tq))
            choice = np.where(np.nan_to_num(p, nan=1.0) < thr, 1, 0)
            pr = dict(params, gate_quantile=tq, gate_threshold=thr)
            out.append(cand(family, f"{label}; the lowest-scoring {round(100 * tq)}% of questions go to the big option", pr,
                            [cheap, big], choice, p.copy(), y.astype(float)))
    return out


def post_configs(D):
    cfgs = []
    for cond in CONDS:
        c, p, m = cond
        for r in W.RETRIEVERS:
            for a_s, a_b in ANS_PAIRS:
                j1, j2 = D.j(r, a_s, c, p, m), D.j(r, a_b, c, p, m)
                cheap = {"name": f"{W.LABEL[r]} retrieval; {W.LABEL[a_s]} answer kept", "em": D.EM[:, j1], "cost": D.FL[:, j1],
                         "found": D.FOUND[:, j1], "ret": r, "esc_ans": False}
                big = {"name": f"{W.LABEL[r]} retrieval; escalated to {W.LABEL[a_b]} x5", "em": D.EM[:, j2],
                       "cost": D.FL[:, j1] + D.FA[:, j2], "found": D.FOUND[:, j2], "ret": r, "esc_ans": True}
                label = f"{W.LABEL[r]} retrieves; {W.LABEL[a_s]} answers x5; a trained gate decides whether {W.LABEL[a_b]} x5 also answers ({cond_words(cond)})"
                cfgs.append((label, {"small_retriever": r, "cheap_answerer": a_s, "big_answerer": a_b, "cot": c, "pick": p, "merge": m, "j_cheap": j1},
                             cheap, big))
    return cfgs


def pre_configs(D, train, cap=6):
    acc, cost = D.EM[train].mean(0), D.FL[train].mean(0)
    big = int(np.lexsort((cost, -acc))[0])
    fr = [i for i in S.pareto(list(zip(cost, acc))) if cost[i] < cost[big]]
    fr = sorted(fr, key=lambda i: cost[i])
    if len(fr) > cap:
        fr = [fr[i] for i in sorted(set(np.linspace(0, len(fr) - 1, cap).round().astype(int)))]
    cfgs = []
    for i in fr:
        mk = lambda jx, tag: {"name": f"{tag}: {D.words(D.setups[jx])}", "em": D.EM[:, jx], "cost": D.FL[:, jx], "found": D.FOUND[:, jx],
                              "ret": D.meta[D.setups[jx]][0], "esc_ans": tag == "big"}
        label = f"a trained router reads the question and sends it to [{D.words(D.setups[i])}] or to [{D.words(D.setups[big])}]"
        cfgs.append((label, {"cheap_setup": D.setups[i], "big_setup": D.setups[big], "j_cheap": i}, mk(i, "cheap"), mk(big, "big")))
    return cfgs


class Features:
    def __init__(self, D, try_embedding=True):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        f = [S.question_features(t) for t in D.text]
        self.facts = np.array([[x["f_yesno"], x["f_compare"], x["f_words"], x["f_digit"], x["f_caps"]] for x in f], float)
        tf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True).fit_transform(D.text)
        k = int(min(16, tf.shape[1] - 1, D.n - 1))
        self.text = TruncatedSVD(k, random_state=0).fit_transform(tf)
        self.emb, self.emb_note = None, "not used"
        if try_embedding:
            try:
                from sklearn.decomposition import PCA
                from sentence_transformers import SentenceTransformer
                model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
                e = model.encode(D.text, normalize_embeddings=True, show_progress_bar=False)
                self.emb = PCA(int(min(16, D.n - 1)), random_state=0).fit_transform(np.asarray(e, float))
                self.emb_note = "all-MiniLM-L6-v2 (22.7M parameters), reduced to 16 dimensions"
            except Exception as ex:                                   # no internet / package: the TF-IDF text features still run
                self.emb_note = f"not available here ({type(ex).__name__}); the TF-IDF text features are used instead"
        self.D = D

    def meta(self, params):
        j = params["j_cheap"]
        D = self.D
        return np.column_stack([D.AG[:, j], D.AC[:, j], D.RG[:, j], D.TP[:, j]])


# ==================================================================================
# Selection and scoring
# ==================================================================================
def pick(cands, train, target):
    if not cands:
        return None
    acc = np.array([c["em"][train].mean() for c in cands])
    cost = np.array([c["cost"][train].mean() for c in cands])
    ok = acc >= target - TOL - 1e-9
    order = np.lexsort((-acc, cost))                          # cheapest first, more accurate first among equals
    i = int(order[np.argmax(ok[order])]) if ok.any() else int(np.lexsort((cost, -acc))[0])
    return cands[i]


def fixed_cand(D, j, family):
    s = D.setups[j]
    path = {"name": D.words(s), "em": D.EM[:, j], "cost": D.FL[:, j], "found": D.FOUND[:, j], "ret": D.meta[s][0], "esc_ans": False}
    return cand(family, D.words(s), {"setup": s}, [path], np.zeros(D.n, int))


def select_all(D, F, casc, train, evalq, seed, learned=True):
    """-> {family: selected candidate}, using only the training questions."""
    acc, cost = D.EM[train].mean(0), D.FL[train].mean(0)
    best = int(np.lexsort((cost, -acc))[0])
    target = float(acc[best])
    ok = acc >= target - TOL - 1e-9
    cheap = int(np.where(ok)[0][np.argmin(cost[ok])])
    sel = {"fixed_best": fixed_cand(D, best, "fixed_best"), "fixed_cheap": fixed_cand(D, cheap, "fixed_cheap")}
    for fam in ("A1", "A2", "A3", "A4"):
        sel[fam] = pick(casc[fam], train, target)
    if learned:
        pre = pre_configs(D, train)
        sel["A5_text"] = pick(gate_candidates(D, "A5_text", pre, lambda p: F.text, train, evalq, seed), train, target)
        if F.emb is not None:
            sel["A5_emb"] = pick(gate_candidates(D, "A5_emb", pre, lambda p: F.emb, train, evalq, seed), train, target)
        sel["A6_facts"] = pick(gate_candidates(D, "A6_facts", pre, lambda p: F.facts, train, evalq, seed), train, target)
        post = post_configs(D)
        sel["A8_meta"] = pick(gate_candidates(D, "A8_meta", post, F.meta, train, evalq, seed), train, target)
        sel["A9_meta_text"] = pick(gate_candidates(D, "A9_meta_text", post, lambda p: np.column_stack([F.meta(p), F.text]), train, evalq, seed),
                                   train, target)
    return {k: v for k, v in sel.items() if v is not None}, target


def describe(D, c, q):
    """Scores of one selected candidate on the questions q."""
    em, cost, found, ch = c["em"][q], c["cost"][q], c["found"][q], c["choice"][q]
    P = c["paths"]
    share = np.array([(ch == i).mean() for i in range(len(P))])
    rand = float(sum(share[i] * P[i]["em"][q].mean() for i in range(len(P))))
    allem = np.stack([p["em"][q] for p in P])
    esc_ans = np.array([P[i]["esc_ans"] for i in ch])
    big_ret = np.array([len({p["ret"] for p in P}) > 1 and P[i]["ret"] != P[0]["ret"] for i in ch])
    d = {"acc": float(em.mean()), "cost": float(cost.mean() / D.ref), "found": float(found.mean()),
         "acc_found": float(em[found == 1].mean()) if (found == 1).any() else float("nan"),
         "acc_not_found": float(em[found == 0].mean()) if (found == 0).any() else float("nan"),
         "esc_answer": float(esc_ans.mean()), "esc_retrieval": float(big_ret.mean()), "random_mix": rand, "perfect": float(allem.max(0).mean()),
         "separation": auc(c["target"][q], c["signal"][q]) if c["signal"] is not None else float("nan")}
    return d


def run_protocol(D, F, casc, splits, learned=True, label=""):
    """splits: list of (train idx, eval idx). -> per-family rows + per-question held-out arrays."""
    S_em, S_cost, S_n = defaultdict(lambda: np.zeros(D.n)), defaultdict(lambda: np.zeros(D.n)), defaultdict(lambda: np.zeros(D.n))
    scal, names, last = defaultdict(list), defaultdict(Counter), {}
    for si, (tr, ev) in enumerate(splits):
        sel, _ = select_all(D, F, casc, tr, ev, seed=si, learned=learned)
        for fam, c in sel.items():
            S_em[fam][ev] += c["em"][ev]
            S_cost[fam][ev] += c["cost"][ev]
            S_n[fam][ev] += 1
            scal[fam].append(describe(D, c, ev))
            names[fam][c["name"]] += 1
            last[fam] = (c, ev)
        if (si + 1) % 5 == 0 or si + 1 == len(splits):
            log(f"  {label}: split {si + 1}/{len(splits)}")
    rows, perq = [], {}
    for fam in ORDER:
        if fam not in scal:
            continue
        seen = S_n[fam] > 0
        em_q, cost_q = S_em[fam][seen] / S_n[fam][seen], S_cost[fam][seen] / S_n[fam][seen]
        perq[fam] = (seen, em_q, cost_q)
        mean = lambda k: float(np.nanmean([d[k] for d in scal[fam]]))
        top, cnt = names[fam].most_common(1)[0]
        rows.append({"family": fam, "Policy": FAMILY_LABEL[fam], "Accuracy": float(em_q.mean()), "Compute (x cheapest)": float(cost_q.mean() / D.ref),
                     "Share escalated to the bigger answerer": mean("esc_answer"), "Share escalated to the bigger retriever": mean("esc_retrieval"),
                     "Retrieval found both": mean("found"), "Accuracy when found": mean("acc_found"), "Accuracy when not found": mean("acc_not_found"),
                     "Gate separation (0.5 = useless)": mean("separation"), "Accuracy of a random router with the same mix": mean("random_mix"),
                     "Accuracy of a perfect router over the same options": mean("perfect"),
                     "Most often selected": top, "Selected in share of splits": cnt / len(splits)})
    base = {f: perq[f] for f in ("fixed_best", "fixed_cheap")}
    for r in rows:
        seen, em_q, cost_q = perq[r["family"]]
        for bf, tag in (("fixed_best", "most accurate fixed"), ("fixed_cheap", "cheapest fixed within 2 points")):
            bs, bem, bcost = base[bf]
            d, lo, hi = S.bootstrap_diff(em_q, bem)
            r[f"Accuracy vs {tag} [95% CI]"] = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            r[f"Accuracy vs {tag} (points)"] = 100 * d
            r[f"Compute vs {tag}"] = float(cost_q.mean() / bcost.mean() - 1)
        da, dc = r["Accuracy vs most accurate fixed (points)"], r["Compute vs most accurate fixed"]
        da2, dc2 = r["Accuracy vs cheapest fixed within 2 points (points)"], r["Compute vs cheapest fixed within 2 points"]
        r["Provisional verdict (final test is on fresh questions, step 5c)"] = (
            "baseline" if r["family"].startswith("fixed") else
            "beats the fixed line: within 2 points of the cheapest good fixed setup at 15%+ less compute" if da2 >= -2 and dc2 <= -0.15 else
            "meets the test against the most accurate fixed setup only (a cheaper fixed setup does as well)" if da >= -2 and dc <= -0.15 else
            "no gain over a fixed setup")
    return rows, perq, last


# ==================================================================================
# A7: the Set 4 rules with smoothed statistics
# ==================================================================================
def smoothed_rules(D, sweep_run, lam=0.03, ms=(0, 2, 5, 10)):
    path = os.path.join(sweep_run, "plan", "rules.jsonl")
    if not os.path.exists(path):
        return [], [], "no plan/rules.jsonl in the sweep run (notebook 12 has not been run)"
    recs = [json.loads(line) for line in open(path)]
    chosen = {}
    for r in recs:
        if str(r.get("chosen")).lower() == "true":
            chosen[(r["scope"], r["setting"])] = r
    qcol = {"yesno": 0, "compare": 1, "words": 2, "digit": 3, "names": 4}
    facts = np.array([[x["f_yesno"], x["f_compare"], x["f_words"], x["f_digit"], x["f_caps"]] for x in map(S.question_features, D.text)], float)
    dev, test = D.dev, D.test
    main = [s for s in D.setups if D.meta[s][5] == W.MAIN_RK and D.meta[s][6] == W.MAIN_AK]
    scopes = {}
    for c in W.COTS:
        for p in W.PICKS:
            for m in W.MERGES:
                ids = {f"{r}-{a}": W.setup_id(r, a, c, p, m, W.MAIN_RK, W.MAIN_AK) for r in W.RETRIEVERS for a in W.ANSWERERS}
                scopes[W.cond_name((c, p, m))] = (ids, ids["q3-l3"])
    stats = {s: (D.FL[dev, D.col[s]].mean(), D.EM[dev, D.col[s]].mean()) for s in main}
    g = W.frontier_routes(stats, 5)
    scopes["all conditions"] = ({f"G{i + 1}": s for i, s in enumerate(g)}, W.setup_id("q3", "l3", 0, "silent", "vote", W.MAIN_RK, W.MAIN_AK))
    sig = {"ret_agree": D.RG, "ans_agree": D.AG, "ans_conf": D.AC, "ret_conf": D.RC}

    def value(q, name, jp):
        if name in qcol:
            x = facts[q, qcol[name]]
            return ("yes" if x >= 0.5 else "no") if W.FEATURES[name][0] == "bool" else int(x)
        x = sig[name][q, jp]
        if x != x:
            x = float(np.nanmedian(sig[name][dev, jp]))
        return round(float(x), 2)

    def holds(x, o, v):
        if isinstance(x, str) or v in ("yes", "no"):
            return str(x) == v if o == "=" else False
        try:
            v = float(v)
        except ValueError:
            return False
        return {"<": x < v, "<=": x <= v, ">": x > v, ">=": x >= v, "=": abs(x - v) < 1e-9}[o]

    rows, rule_rows = [], []
    for (scope, setting), rec in sorted(chosen.items()):
        if scope not in scopes:
            continue
        ids, probe = scopes[scope]
        jp, use_probe = D.col[probe], setting != "question only"
        names = W.PRE if not use_probe else W.POST
        rules, default, _, _, dconf = W.parse_rules(rec["rules"], list(ids), names, 99)
        col = {k: D.col[s] for k, s in ids.items()}
        cost = lambda k, q: D.FL[q, col[k]] + (D.FL[q, jp] if use_probe and col[k] != jp else 0.0)
        used = sorted({r for _, r, _ in rules} | {default})
        util = {k: D.EM[dev, col[k]].mean() - lam * cost(k, dev).mean() / D.ref for k in ids}
        fallback = max(used, key=lambda k: util[k])                    # the best single route among those the rules use

        def fire(q, keep):
            for i, (conds, route, _) in enumerate(rules):
                if keep[i] and all(holds(value(q, a, jp), o, v) for a, o, v in conds):
                    return route, i
            return (default if keep[-1] else fallback), len(rules)
        all_on = [True] * (len(rules) + 1)
        fired_dev = [fire(q, all_on)[1] for q in dev]
        variants = [("as written in Set 4", None)] + [(f"pruned with m-estimate, m = {m}" if m else "pruned on raw counts (m = 0)", m) for m in ms] \
            + [("pruned with Laplace smoothing", "laplace")]
        for vname, m in variants:
            keep = list(all_on)
            for i in range(len(rules) + 1):
                route = rules[i][1] if i < len(rules) else default
                qs = dev[[f == i for f in fired_dev]]
                n = len(qs)
                k_r, k_f = (D.EM[qs, col[route]].sum(), D.EM[qs, col[fallback]].sum()) if n else (0.0, 0.0)
                raw = k_r / n if n else float("nan")
                if m is None:
                    sm_r = sm_f = float("nan")
                elif m == "laplace":
                    sm_r, sm_f = (k_r + 1) / (n + 2), (k_f + 1) / (n + 2)
                else:
                    pr, pf = D.EM[dev, col[route]].mean(), D.EM[dev, col[fallback]].mean()
                    sm_r = (k_r + m * pr) / (n + m) if n + m else pr
                    sm_f = (k_f + m * pf) / (n + m) if n + m else pf
                if m is not None:
                    c_r = cost(route, qs).mean() if n else cost(route, dev).mean()
                    c_f = cost(fallback, qs).mean() if n else cost(fallback, dev).mean()
                    keep[i] = bool(sm_r - lam * c_r / D.ref >= sm_f - lam * c_f / D.ref - 1e-12)
                text = ("IF " + " AND ".join(f"{a} {o} {v}" for a, o, v in rules[i][0]) + f" THEN {route}") if i < len(rules) else f"OTHERWISE {route}"
                conf = rules[i][2] if i < len(rules) else dconf
                rule_rows.append({"scope": scope, "setting": setting, "variant": vname, "rule": text, "route means": D.words(ids[route]),
                                  "planner's confidence": conf, "dev questions it fired on": n, "raw dev accuracy": raw,
                                  "smoothed dev accuracy": sm_r, "smoothed dev accuracy of the fallback route on the same questions": sm_f,
                                  "kept": keep[i], "fallback route": fallback})
            for part, qs in (("dev", dev), ("test", test)):
                ch = [fire(q, keep)[0] for q in qs]
                em = np.array([D.EM[q, col[k]] for q, k in zip(qs, ch)])
                co = np.array([cost(k, q) for q, k in zip(qs, ch)])
                if part == "dev":
                    de, dc = em.mean(), co.mean() / D.ref
            rows.append({"scope": scope, "setting": setting, "variant": vname, "rules kept": int(sum(keep)), "of rules": len(keep),
                         "dev accuracy": float(de), "dev compute (x cheapest)": float(dc), "test accuracy": float(em.mean()),
                         "test compute (x cheapest)": float(co.mean() / D.ref),
                         "fallback route test accuracy": float(D.EM[test, col[fallback]].mean()),
                         "fallback route test compute (x cheapest)": float(cost(fallback, test).mean() / D.ref)})
    return rows, rule_rows, ""


# ==================================================================================
# Headroom, threshold curves, learning curve
# ==================================================================================
def headroom(D):
    acc, cost = D.EM.mean(0), D.FL.mean(0)
    main = [i for i, s in enumerate(D.setups) if D.meta[s][5] == 3 and D.meta[s][6] == 5]
    fr = W.frontier_routes({D.setups[i]: (cost[i], acc[i]) for i in main}, 5)
    cols = [D.col[s] for s in fr]
    best = max(cols, key=lambda i: acc[i])
    r, a, c, _, _, _, _ = D.meta[D.setups[best]]
    twins = [D.j(r, a, c, p, m) for p in W.PICKS for m in W.MERGES]
    rng = np.random.default_rng(0)
    luck = float(np.mean([D.EM[:, rng.choice(twins, len(cols), replace=False)].max(1).mean() for _ in range(300)]))
    M = D.EM[:, cols]
    return [
        {"Quantity": "Most accurate of the 5 routes on the cost-accuracy frontier", "Value": float(acc[best]), "Note": D.words(D.setups[best])},
        {"Quantity": "Perfect router over those 5 routes", "Value": float(M.max(1).mean()), "Note": "hindsight: right if any route is right"},
        {"Quantity": "Perfect router over 5 near-identical copies of the best route (luck only)", "Value": luck,
         "Note": "same models, only the shortlist settings differ; mean of 300 random draws of 5 from 9"},
        {"Quantity": "Questions where the best route is right", "Value": float(D.EM[:, best].mean()), "Note": ""},
        {"Quantity": "Questions where the best route is wrong and a cheaper route is right", "Value": float(((D.EM[:, best] == 0) & (M.max(1) == 1)).mean()), "Note": ""},
        {"Quantity": "Questions where no route is right", "Value": float((M.max(1) == 0).mean()), "Note": ""},
        {"Quantity": "Questions where every route is right", "Value": float((M.min(1) == 1).mean()), "Note": ""},
    ]


def threshold_curves(D, frozen):
    rows = []
    for fam in ("A1", "A2", "A4"):
        c = frozen.get(fam)
        if c is None:
            continue
        p = c["params"]
        cond = (p["cot"], p["pick"], p["merge"])
        gate = "answer" if p["big_answerer"] else "retrieval"
        for t in ([0.0] + ANS_T + [1.01] if gate == "answer" else [0.0] + RET_T + [1.01]):
            x = cascade(D, fam, p["small_retriever"], p["big_retriever"], p["cheap_answerer"], p["big_answerer"], cond,
                        t if gate == "retrieval" else p["retrieval_agreement_threshold"],
                        t if gate == "answer" else p["answer_agreement_threshold"], p["big_answer_tries"])
            rows.append({"family": fam, "Policy": FAMILY_LABEL[fam], "gate": gate, "threshold": t,
                         "threshold in words": "never escalate" if t == 0 else "always escalate" if t > 1 else f"accept if agreement >= {t:g}",
                         "Accuracy": float(x["em"].mean()), "Compute (x cheapest)": float(x["cost"].mean() / D.ref),
                         "Share escalated": float((x["choice"] != 0).mean()), "is the frozen threshold": bool(
                             abs(t - (p["answer_agreement_threshold"] if gate == "answer" else p["retrieval_agreement_threshold"])) < 1e-9)})
    return rows


def learning_curve(D, F, sizes=(40, 80, 120, 160), reps=30):
    """How well can a gate predict 'the cheap answer is no worse than the big one', as the number of training questions grows?"""
    j1, j2 = D.j("q7", "l8", 1, "silent", "all"), D.j("q7", "q32", 1, "silent", "all")
    y = (D.EM[:, j1] >= D.EM[:, j2]).astype(int)
    sets = {"5 facts about the question": F.facts, "question text (TF-IDF)": F.text,
            "cheap run's metadata": F.meta({"j_cheap": j1}), "metadata + question text": np.column_stack([F.meta({"j_cheap": j1}), F.text])}
    if F.emb is not None:
        sets["question text (embedding)"] = F.emb
    rows = []
    rng = np.random.default_rng(0)
    for n in sizes:
        res = defaultdict(list)
        for _ in range(reps):
            perm = rng.permutation(D.n)
            tr, te = perm[:n], perm[n:]
            for name, X in sets.items():
                res[name].append(auc(y[te], fit_predict(X[tr], y[tr], X[te])))
        for name, v in res.items():
            rows.append({"Inputs the gate sees": name, "Training questions": n, "Separation on held-out questions (0.5 = useless)": float(np.nanmean(v)),
                         "standard error": float(np.nanstd(v) / np.sqrt(np.sum(~np.isnan(v))))})
    ref = f"cheap = {D.words(D.setups[j1])}; big = {D.words(D.setups[j2])}; share of questions where the cheap answer is no worse: {y.mean():.0%}"
    return rows, ref


# ==================================================================================
# Charts
# ==================================================================================
def charts(D, out, rows_cv, frozen, curves, lc, last_cv):
    plt = S._plt()
    os.makedirs(out, exist_ok=True)
    fams = [r["family"] for r in rows_cv]
    colors = {f: (S.INK2 if f.startswith("fixed") else S.SERIES[i % len(S.SERIES)]) for i, f in enumerate(fams)}
    short = {f: FAMILY_LABEL[f].replace("Fixed: ", "fixed: ") for f in fams}
    # 1. accuracy against compute
    acc, cost = D.EM.mean(0), D.FL.mean(0) / D.ref
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    ax.scatter(cost, acc, s=10, color=S.INK2, alpha=0.25, zorder=2, label="every fixed setup in the sweep (all 200 questions)")
    fr = sorted(S.pareto(list(zip(cost, acc))), key=lambda i: cost[i])
    ax.plot(cost[fr], acc[fr], color=S.INK2, ls="--", lw=1.2, zorder=3, label="fixed setups that are best for their cost")
    for r in rows_cv:
        f = r["family"]
        ax.scatter(r["Compute (x cheapest)"], r["Accuracy"], s=95, marker={"fixed_best": "s", "fixed_cheap": "D"}.get(f, "o"), color=colors[f], zorder=5,
                   edgecolor=S.SURFACE, linewidth=1.2, label=short[f] + " (held-out)")
    ax.set_xscale("log")
    ax.set_xlabel("compute per question, relative to the cheapest setup (log scale)")
    ax.set_ylabel("accuracy (exact match)")
    ax.set_title("Accuracy against compute: switching policies and fixed setups")
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(os.path.join(out, "1_accuracy_vs_compute.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 2. cascade threshold curve
    if curves:
        fs = sorted({r["family"] for r in curves}, key=ORDER.index)
        fig, axs = plt.subplots(1, len(fs), figsize=(6.2 * len(fs), 4.6), squeeze=False)
        for ax, f in zip(axs[0], fs):
            g = [r for r in curves if r["family"] == f]
            ax.plot([r["Compute (x cheapest)"] for r in g], [r["Accuracy"] for r in g], marker="o", color=colors.get(f, S.SERIES[0]), lw=1.6)
            seen_pts = set()
            for r in g:
                key = (round(r["Compute (x cheapest)"], 2), round(r["Accuracy"], 3))
                if key in seen_pts and not r["is the frozen threshold"]:
                    continue
                seen_pts.add(key)
                lab = "never" if r["threshold"] == 0 else "always" if r["threshold"] > 1 else f"{r['threshold']:g}"
                ax.annotate(lab, (r["Compute (x cheapest)"], r["Accuracy"]), textcoords="offset points", xytext=(5, -11), fontsize=8,
                            fontweight="bold" if r["is the frozen threshold"] else "normal")
            ax.set_title(f"{FAMILY_LABEL[f]} ({g[0]['gate']} gate)\nlabel = agreement needed to accept; bold = chosen", fontsize=9)
            ax.set_xlabel("compute (x cheapest)")
            ax.set_ylabel("accuracy, all 200 questions")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "2_cascade_threshold_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 3. learning curve
    if lc:
        fig, ax = plt.subplots(figsize=(7.6, 4.6))
        for i, name in enumerate(dict.fromkeys(r["Inputs the gate sees"] for r in lc)):
            g = [r for r in lc if r["Inputs the gate sees"] == name]
            x = [r["Training questions"] for r in g]
            y = np.array([r["Separation on held-out questions (0.5 = useless)"] for r in g])
            e = np.array([r["standard error"] for r in g])
            ax.plot(x, y, marker="o", color=S.SERIES[i % len(S.SERIES)], label=name, lw=1.6)
            ax.fill_between(x, y - e, y + e, color=S.SERIES[i % len(S.SERIES)], alpha=0.15, lw=0)
        ax.axhline(0.5, color=S.INK2, ls=":", lw=1)
        ax.set_xlabel("training questions")
        ax.set_ylabel("separation on held-out questions (0.5 = useless)")
        ax.set_title("Does more training data help the gate? (band = 1 standard error)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(out, "3_learning_curve.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 4. route mix (frozen policies, all questions)
    pol = [f for f in ORDER if f in frozen and not f.startswith("fixed")]
    if pol:
        fig, ax = plt.subplots(figsize=(9.5, 0.62 * len(pol) + 1.6))
        kinds = [("cheap path kept", S.SERIES[0]), ("escalated: bigger answerer", S.SERIES[1]), ("escalated: bigger retriever", S.SERIES[2]),
                 ("escalated: both", S.SERIES[3 % len(S.SERIES)])]
        left = np.zeros(len(pol))
        for kind, colr in kinds:
            v = []
            for f in pol:
                c = frozen[f]
                P = c["paths"]
                lab = []
                for i in c["choice"]:
                    br = P[i]["ret"] != P[0]["ret"]
                    ba = P[i]["esc_ans"]
                    lab.append("escalated: both" if br and ba else "escalated: bigger retriever" if br else "escalated: bigger answerer" if ba else "cheap path kept")
                v.append(np.mean(np.array(lab) == kind))
            ax.barh(range(len(pol)), v, left=left, color=colr, label=kind, edgecolor=S.SURFACE)
            left += np.array(v)
        ax.set_yticks(range(len(pol)))
        ax.set_yticklabels([FAMILY_LABEL[f] for f in pol], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("share of the 200 questions")
        ax.set_title("Where each policy sends the questions (policy chosen on all 200 questions)")
        ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.savefig(os.path.join(out, "4_route_mix.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 5. calibration of the gates
    gated = [f for f in pol if frozen[f]["signal"] is not None]
    if gated:
        ncol = min(4, len(gated))
        nrow = int(np.ceil(len(gated) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(3.9 * ncol, 3.7 * nrow), squeeze=False)
        for ax in axs.ravel()[len(gated):]:
            ax.axis("off")
        for ax, f in zip(axs.ravel(), gated):
            s, t = frozen[f]["signal"], frozen[f]["target"]
            ok = ~np.isnan(s) & ~np.isnan(t)
            s, t = s[ok], t[ok]
            edges = np.unique(np.quantile(s, [0, 0.2, 0.4, 0.6, 0.8, 1.0]))
            xs, ys, ns = [], [], []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mk = (s >= lo) & ((s < hi) if hi < edges[-1] else (s <= hi))
                if mk.sum():
                    xs.append(s[mk].mean()); ys.append(t[mk].mean()); ns.append(mk.sum())
            if len(edges) == 1:
                xs, ys, ns = [s.mean()], [t.mean()], [len(s)]
            ax.plot([0, 1], [0, 1], color=S.INK2, ls=":", lw=1)
            ax.plot(xs, ys, marker="o", color=colors.get(f, S.SERIES[0]), lw=1.6)
            for x, y, n in zip(xs, ys, ns):
                ax.annotate(str(int(n)), (x, y), textcoords="offset points", xytext=(4, 5), fontsize=7)
            ax.set_xlim(0, 1.02); ax.set_ylim(0, 1.02)
            ax.set_title(FAMILY_LABEL[f].split(",")[0], fontsize=8)
            ax.set_xlabel("gate's signal (agreement or score)", fontsize=8)
            ax.set_ylabel("share where the cheap option was enough", fontsize=8)
        fig.suptitle("Does the gate's signal tell when the cheap option is enough? (numbers = questions per point)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "5_gate_calibration.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 6. where the gain comes from
    if pol:
        groups = ["cheap option was enough", "needed the big option", "neither works"]
        outcomes = [("kept cheap, right (compute saved)", S.SERIES[2]), ("escalated, right (answer rescued)", S.SERIES[0]),
                    ("escalated needlessly or without success (compute wasted)", "#b9b9b4"), ("kept cheap, wrong", S.SERIES[1])]
        fig, axs = plt.subplots(1, 3, figsize=(15, 0.55 * len(pol) + 1.8), sharey=True)
        for ax, gname in zip(axs, groups):
            left = np.zeros(len(pol))
            for oname, colr in outcomes:
                v = []
                for f in pol:
                    c = frozen[f]
                    P = c["paths"]
                    cheap, big = P[0]["em"], np.max(np.stack([p["em"] for p in P[1:]]), 0) if len(P) > 1 else P[0]["em"]
                    grp = np.where(cheap == 1, 0, np.where(big == 1, 1, 2))
                    esc, em = c["choice"] != 0, c["em"]
                    o = np.where(~esc & (em == 1), 0, np.where(esc & (em == 1) & (grp == 1), 1, np.where(esc, 2, 3)))
                    mk = grp == groups.index(gname)
                    v.append(float(np.mean(mk & (o == [x[0] for x in outcomes].index(oname)))))
                ax.barh(range(len(pol)), v, left=left, color=colr, label=oname, edgecolor=S.SURFACE)
                left += np.array(v)
            ax.set_title(gname, fontsize=9)
            ax.set_xlabel("share of the 200 questions")
        axs[0].set_yticks(range(len(pol)))
        axs[0].set_yticklabels([FAMILY_LABEL[f] for f in pol], fontsize=8)
        axs[0].invert_yaxis()
        axs[2].legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.suptitle("Where each policy gains and loses, by kind of question", fontsize=10, y=1.02)
        fig.savefig(os.path.join(out, "6_where_the_gain_comes_from.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


# ==================================================================================
# Report
# ==================================================================================
def md_table(rows, cols, pct=(), x=(), num=()):
    def f(r, c):
        v = r.get(c)
        if v is None or (isinstance(v, float) and v != v):
            return "-"
        if c in pct:
            return f"{100 * v:.1f}%"
        if c in x:
            return f"{v:.1f}x"
        if c in num:
            return f"{v:.2f}"
        return str(v)
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rows)


def cmd_offline(args):
    import pandas as pd
    t0 = time.time()
    out = os.path.join(args.run_dir, "results")
    os.makedirs(out, exist_ok=True)
    D = Data(args.sweep_run, args.data_dir)
    F = Features(D, try_embedding=not args.no_embedding)
    log("text embedding:", F.emb_note)
    casc = threshold_candidates(D)
    log("threshold candidates:", {k: len(v) for k, v in casc.items()})
    from sklearn.model_selection import KFold
    cv = [(tr, te) for rep in range(args.repeats) for tr, te in KFold(5, shuffle=True, random_state=100 + rep).split(np.arange(D.n))]
    rows_cv, perq_cv, last_cv = run_protocol(D, F, casc, cv, label="cross-validation")
    rows_dt, perq_dt, last_dt = run_protocol(D, F, casc, [(D.dev, D.test)], label="dev -> test")
    allq = np.arange(D.n)
    frozen, target = select_all(D, F, casc, allq, None, seed=999)
    log("frozen policies chosen on all questions")
    curves = threshold_curves(D, frozen)
    lc, lc_ref = learning_curve(D, F, reps=10 if args.repeats < 3 else 30)
    a7, a7_rules, a7_note = smoothed_rules(D, args.sweep_run)
    head = headroom(D)

    # ---- files
    def dump(name, rows):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    for r in rows_cv:
        r["Protocol"] = f"5-fold cross-validation x{args.repeats}, all 200 questions"
    for r in rows_dt:
        r["Protocol"] = "chosen on dev (100), scored on test (100)"
    dump("policies.jsonl", rows_cv + rows_dt)
    dump("threshold_curves.jsonl", curves)
    dump("learning_curve.jsonl", lc)
    dump("rules_smoothed.jsonl", a7)
    dump("rules_smoothed_per_rule.jsonl", a7_rules)
    dump("headroom.jsonl", head)
    froz_rows, dec = [], []
    for fam in ORDER:
        if fam not in frozen:
            continue
        c = frozen[fam]
        d = describe(D, c, allq)
        froz_rows.append({"family": fam, "Policy": FAMILY_LABEL[fam], "Exact policy": c["name"], "Accuracy (all 200, in-sample)": d["acc"],
                          "Compute (x cheapest)": d["cost"], "Share escalated to the bigger answerer": d["esc_answer"],
                          "Share escalated to the bigger retriever": d["esc_retrieval"],
                          "parameters": json.dumps({k: v for k, v in c["params"].items() if k != "j_cheap"}, default=float)})
        for q in allq:
            p = c["paths"][c["choice"][q]]
            dec.append({"family": fam, "Policy": FAMILY_LABEL[fam], "qid": D.qids[q], "split": D.split[q], "question": D.text[q],
                        "path taken": p["name"], "gate signal": None if c["signal"] is None or c["signal"][q] != c["signal"][q] else float(c["signal"][q]),
                        "correct": float(c["em"][q]), "compute (x cheapest)": float(c["cost"][q] / D.ref),
                        "cheap option would have been right": float(c["paths"][0]["em"][q])})
    dump("decisions.jsonl", dec)
    json.dump({"version": VERSION, "tolerance": TOL, "chosen_on": "all 200 Set 4 questions", "sweep_run": args.sweep_run,
               "policies": [{k: v for k, v in r.items()} for r in froz_rows]}, open(os.path.join(args.run_dir, "frozen_policies.json"), "w"), indent=1)
    charts(D, os.path.join(out, "charts"), rows_cv, frozen, curves, lc, last_cv)

    # ---- A7 summary: mean over scopes
    a7_sum = []
    if a7:
        df7 = pd.DataFrame(a7)
        for (setting, variant), g in df7.groupby(["setting", "variant"], sort=False):
            a7_sum.append({"Rules applied": setting, "Variant": variant, "Scopes": len(g), "Mean rules kept": g["rules kept"].mean(),
                           "Mean test accuracy": g["test accuracy"].mean(), "Mean test compute (x cheapest)": g["test compute (x cheapest)"].mean(),
                           "Mean test accuracy of the single fallback route": g["fallback route test accuracy"].mean(),
                           "Mean test compute of the fallback route": g["fallback route test compute (x cheapest)"].mean()})
    dump("rules_smoothed_summary.jsonl", a7_sum)

    # ---- summary.md
    main_cols = ["Policy", "Accuracy", "Compute (x cheapest)", "Share escalated to the bigger answerer", "Share escalated to the bigger retriever",
                 "Retrieval found both", "Accuracy when found", "Accuracy when not found"]
    cmp_cols = ["Policy", "Accuracy vs most accurate fixed [95% CI]", "Compute vs most accurate fixed",
                "Accuracy vs cheapest fixed within 2 points [95% CI]", "Compute vs cheapest fixed within 2 points",
                "Provisional verdict (final test is on fresh questions, step 5c)"]
    skill_cols = ["Policy", "Accuracy", "Accuracy of a random router with the same mix", "Accuracy of a perfect router over the same options",
                  "Gate separation (0.5 = useless)"]
    pct = {"Accuracy", "Share escalated to the bigger answerer", "Share escalated to the bigger retriever", "Retrieval found both", "Accuracy when found",
           "Accuracy when not found", "Compute vs most accurate fixed", "Compute vs cheapest fixed within 2 points",
           "Accuracy of a random router with the same mix", "Accuracy of a perfect router over the same options", "Selected in share of splits",
           "Accuracy (all 200, in-sample)", "Mean test accuracy", "Mean test accuracy of the single fallback route", "Value",
           "raw dev accuracy", "smoothed dev accuracy", "smoothed dev accuracy of the fallback route on the same questions", "planner's confidence"}
    xs = {"Compute (x cheapest)", "Mean test compute (x cheapest)", "Mean test compute of the fallback route"}
    nums = {"Gate separation (0.5 = useless)", "Separation on held-out questions (0.5 = useless)", "Mean rules kept", "standard error"}

    def signed(rows):
        rr = [dict(r) for r in rows]
        for r in rr:
            for c in ("Compute vs most accurate fixed", "Compute vs cheapest fixed within 2 points"):
                r[c] = f"{100 * r[c]:+.0f}%"
        return rr
    L = [f"# Set 5a - offline switching policies on the Set 4 results\n",
         f"Sweep used: `{os.path.basename(args.sweep_run.rstrip('/'))}`. {D.n} questions, {len(D.setups)} setups. No new model answers were generated.",
         f"Selection rule for every row: on the training questions, the cheapest candidate whose accuracy is within {round(100 * TOL)} points of the most "
         "accurate fixed setup. All scores below are on questions the selection did not see, except where marked in-sample.",
         f"Question-text embedding: {F.emb_note}.\n",
         f"## 1. Main result: 5-fold cross-validation x{args.repeats} over all 200 questions\n", md_table(rows_cv, main_cols, pct, xs, nums),
         "\n### Against the fixed setups\n", md_table(signed(rows_cv), cmp_cols, pct - {"Compute vs most accurate fixed", "Compute vs cheapest fixed within 2 points"}, xs, nums),
         "\n### Does the router have skill?\n\nA policy has skill only if it beats a random router that escalates the same share of questions.\n",
         md_table([r for r in rows_cv if not r["family"].startswith("fixed")], skill_cols, pct, xs, nums),
         "\n### Which exact policy was selected most often\n", md_table(rows_cv, ["Policy", "Most often selected", "Selected in share of splits"], pct, xs, nums),
         "\n## 2. Same policies, chosen on dev (100) and scored on test (100), as in Set 4\n", md_table(rows_dt, main_cols, pct, xs, nums),
         "\n", md_table(signed(rows_dt), cmp_cols, pct - {"Compute vs most accurate fixed", "Compute vs cheapest fixed within 2 points"}, xs, nums),
         "\n## 3. Frozen policies for step 5c (chosen on all 200 questions; these numbers are in-sample)\n",
         md_table(froz_rows, ["Policy", "Exact policy", "Accuracy (all 200, in-sample)", "Compute (x cheapest)"], pct, xs, nums),
         "\n## 4. How much is there to gain? (all 200 questions)\n", md_table(head, ["Quantity", "Value", "Note"], pct, xs, nums),
         "\n## 5. Does more training data help the gate?\n", f"Reference pair: {lc_ref}.\n",
         "Values at or below 0.5 mean no usable signal; with few held-out questions an input without signal often scores below 0.5.\n",
         md_table(lc, ["Inputs the gate sees", "Training questions", "Separation on held-out questions (0.5 = useless)", "standard error"], pct, xs, nums),
         "\n## 6. A7: the Set 4 rules with smoothed statistics (chosen on dev, scored on test)\n",
         (a7_note or "A rule is dropped when its smoothed dev accuracy, net of cost, is below that of the single best route on the same questions. "
          "m-estimate: (correct + m x the route's overall dev accuracy) / (fired + m). Mean over the 19 scopes of Set 4.\n"),
         md_table(a7_sum, ["Rules applied", "Variant", "Scopes", "Mean rules kept", "Mean test accuracy", "Mean test compute (x cheapest)",
                           "Mean test accuracy of the single fallback route", "Mean test compute of the fallback route"], pct, xs, nums) if a7_sum else ""]
    whole = [r for r in a7_rules if r["scope"] == "all conditions" and r["variant"] == "pruned with m-estimate, m = 5"]
    if whole:
        L += ["\n### The planner's rules for the whole sweep, with raw and smoothed accuracy (m = 5)\n"]
        for setting in dict.fromkeys(r["setting"] for r in whole):
            L += [f"\n**{setting}**\n", md_table([r for r in whole if r["setting"] == setting],
                                                ["rule", "route means", "planner's confidence", "dev questions it fired on", "raw dev accuracy",
                                                 "smoothed dev accuracy", "smoothed dev accuracy of the fallback route on the same questions", "kept"], pct, xs, nums)]
    L += ["\nCharts: `results/charts/`. Excel: `results/set5a_results.xlsx`. Every question's path under each frozen policy: `results/decisions.jsonl`. "
          "Frozen policies for step 5c: `frozen_policies.json`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    drop = lambda rows: pd.DataFrame([{k: v for k, v in r.items() if k not in ("family",) and not k.endswith("(points)")} for r in rows])
    W.write_excel(os.path.join(out, "set5a_results.xlsx"), {
        "Policies (cross-validation)": drop(rows_cv), "Policies (dev to test)": drop(rows_dt), "Frozen policies": drop(froz_rows),
        "Headroom": pd.DataFrame(head), "Threshold curves": drop(curves), "Learning curve": pd.DataFrame(lc),
        "Rules smoothed (summary)": pd.DataFrame(a7_sum), "Rules smoothed (by scope)": pd.DataFrame(a7), "Rules (each rule)": pd.DataFrame(a7_rules),
        "Decisions": pd.DataFrame(dec)})
    json.dump({"version": VERSION, "seconds": round(time.time() - t0), "questions": D.n, "setups": len(D.setups), "repeats": args.repeats,
               "embedding": F.emb_note, "tolerance": TOL}, open(os.path.join(args.run_dir, "offline_status.json"), "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("offline")
    p.add_argument("--sweep_run", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--no_embedding", action="store_true")
    args = ap.parse_args()
    {"offline": cmd_offline}[args.cmd](args)


if __name__ == "__main__":
    main()
