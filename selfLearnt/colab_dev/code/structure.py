"""Set 5d part 1 - is there hidden structure that separates easy questions from hard ones? (CPU, no new model answers)

Several ways of looking for structure are scored on one scoreboard, so they can be compared:

  spaces (what is looked at)          question facts, question text, question topics, question embedding,
                                      question-passage similarity, metadata of a cheap run, cross-model agreement, all cheap signals
  strategies (how structure is read)  linear probe, first 2 principal components, nearest neighbours (the neighbourhoods t-SNE draws),
                                      k-means clusters, a small decision tree, boosted trees, association rules
  targets (what "easy" means)         small pipeline right; cheap answer no worse than big; best setup right; small retriever finds both;
                                      question is easy for most setups

Every score is on held-out questions (5-fold cross-validation), next to the score the same method gets when the labels are shuffled.
PCA / t-SNE pictures are drawn too, but a picture is never the evidence: the held-out number is.
"""
import argparse
import itertools
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S          # noqa: E402
import sweep as W        # noqa: E402
import routers as R      # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "5d1.1"
REF = dict(ret="q7", cheap="l8", big="q32", cot=1, pick="silent", merge="all")       # reference pair for "cheap answer is enough"
SMALL = dict(ret="q3", ans="l3", cot=1, pick="silent", merge="vote")                  # the small pipeline
log = R.log


def norm(t):
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", " ", str(t).lower())).strip()


# ==================================================================================
# Spaces
# ==================================================================================
def embed(texts):
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return np.asarray(m.encode(list(texts), normalize_embeddings=True, show_progress_bar=False, batch_size=64), float), \
            "all-MiniLM-L6-v2 (22.7M parameters)"
    except Exception as ex:
        return None, f"not available here ({type(ex).__name__}); TF-IDF is used instead"


def sim_features(sims):
    """sims: similarity of the question to each of its passages -> a few numbers describing the 'retrieval landscape'."""
    s = np.sort(np.asarray(sims, float))[::-1]
    p = np.exp((s - s.max()) / 0.1)
    p = p / p.sum()
    return [s[0], s[1], s[0] - s[2], s[:2].mean() - s[2:].mean(), s.mean(), s.std(), float((s > 0.7 * s[0]).sum()), float(-(p * np.log(p + 1e-12)).sum())]


SIM_NAMES = ["most similar passage", "second most similar passage", "gap: 1st minus 3rd", "gap: top 2 minus the rest", "mean similarity",
             "spread of similarity", "passages close to the best one", "entropy of similarity"]


def build_spaces(D, qs, try_embedding=True):
    import pandas as pd
    from sklearn.decomposition import NMF, PCA, TruncatedSVD
    from sklearn.feature_extraction.text import TfidfVectorizer
    spaces, info = {}, {}
    text = D.text
    f = [S.question_features(t) for t in text]
    wh = ["what", "which", "who", "when", "where", "how"]
    first = [next((w for w in wh if re.search(rf"\b{w}\b", t.lower())), "other") for t in text]
    facts = np.column_stack([[x["f_yesno"] for x in f], [x["f_compare"] for x in f], [x["f_words"] for x in f], [x["f_digit"] for x in f],
                             [x["f_caps"] for x in f], [x["f_quote"] for x in f], [len(t) for t in text]] +
                            [[float(a == w) for a in first] for w in wh])
    fact_names = ["yes/no question", "comparison", "words", "has digit", "capitalised words", "has quote", "characters"] + [f"asks '{w}'" for w in wh]
    spaces["question facts"] = (facts, fact_names, "free")
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words="english")
    tf = vec.fit_transform(text)
    k = int(min(16, tf.shape[1] - 1, D.n - 1))
    spaces["question text (TF-IDF)"] = (TruncatedSVD(k, random_state=0).fit_transform(tf), [f"text component {i + 1}" for i in range(k)], "free")
    nmf = NMF(6, init="nndsvda", random_state=0, max_iter=600)
    Wt = nmf.fit_transform(tf)
    vocab = np.array(vec.get_feature_names_out())
    topics = [", ".join(vocab[np.argsort(-c)[:7]]) for c in nmf.components_]
    spaces["question topics (NMF, 6 topics)"] = (Wt, [f"topic {i + 1}: {t}" for i, t in enumerate(topics)], "free")
    info["topics"], info["topic_of"] = topics, Wt.argmax(1)
    # question-passage similarity: TF-IDF always, embedding when available
    passages = [list(q["ctxs_candidate"]) for q in qs]
    pv = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit([p for ps in passages for p in ps] + text)
    simt = []
    for t, ps in zip(text, passages):
        A, B = pv.transform([t]), pv.transform(ps)
        simt.append(sim_features((B @ A.T).toarray().ravel()))
    overlap = [[sum(1 for p in ps if any(w in p for w in set(re.findall(r"\b[A-Z][a-z]{3,}\b", t)))), len(ps)] for t, ps in zip(text, passages)]
    X = np.column_stack([np.array(simt), np.array(overlap, float)[:, :1]])
    spaces["question-passage similarity (TF-IDF)"] = (X, [n + " (TF-IDF)" for n in SIM_NAMES] + ["passages sharing a name with the question"],
                                                      "free")
    emb, note = (embed(text) if try_embedding else (None, "not used"))
    info["embedding"] = note
    info["emb_raw"] = emb
    if emb is not None:
        spaces["question embedding"] = (PCA(int(min(16, D.n - 1)), random_state=0).fit_transform(emb), [f"embedding component {i + 1}" for i in range(16)],
                                        "22.7M-parameter model reads the question: under 0.001x")
        pe, _ = embed([p for ps in passages for p in ps])
        if pe is not None:
            sime, i = [], 0
            for e, ps in zip(emb, passages):
                sime.append(sim_features(pe[i:i + len(ps)] @ e))
                i += len(ps)
            spaces["question-passage similarity (embedding)"] = (np.array(sime), [n + " (embedding)" for n in SIM_NAMES],
                                                                 "22.7M-parameter model reads the 20 passages: about 0.005x")
    # metadata of runs that have already happened
    df = D.df

    def run_meta(setup, tag):
        g = df[df.setup == setup].set_index("qid").reindex(D.qids)
        pred = g["prediction"].fillna("").astype(str)
        cols = {"agreement of the 5 answers": g["ans_agree"], "stated answer confidence": pd.to_numeric(g["ans_conf"], errors="coerce"),
                "answer token probability": pd.to_numeric(g["ans_token_prob"], errors="coerce"), "agreement of the 3 retrieval tries": g["ret_agree"],
                "stated retrieval confidence": pd.to_numeric(g["ret_conf"], errors="coerce"),
                "retrieval token probability": pd.to_numeric(g["ret_token_prob"], errors="coerce"), "passages in the shortlist": g["n_passages"],
                "answer output tokens": g["ans_out_tokens"], "retrieval output tokens": g["ret_out_tokens"],
                "words in the answer": pred.str.split().str.len(), "answer is yes/no": pred.str.lower().str.strip().isin(["yes", "no"]).astype(float),
                "answer has a digit": pred.str.contains(r"\d").astype(float)}
        return np.column_stack([np.asarray(v, float) for v in cols.values()]), [f"{k} [{tag}]" for k in cols], g
    s_small = W.setup_id(SMALL["ret"], SMALL["ans"], SMALL["cot"], SMALL["pick"], SMALL["merge"], 3, 5)
    s_cheap = W.setup_id(REF["ret"], REF["cheap"], REF["cot"], REF["pick"], REF["merge"], 3, 5)
    Xs, ns, _ = run_meta(s_small, "small pipeline")
    Xc, nc, gc = run_meta(s_cheap, "cheap run")
    spaces["metadata of the small pipeline (Qwen 3B - Llama 3B)"] = (Xs, ns, "needs the small pipeline to have run: about 1.3x")
    spaces["metadata of the cheap run (Qwen 7B - Llama 8B)"] = (Xc, nc, "needs the cheap run to have run: about 3.5x")
    # cross-model agreement on the same shortlist
    preds = {}
    for a in ("l3", "l8"):
        sid = W.setup_id(REF["ret"], a, REF["cot"], REF["pick"], REF["merge"], 3, 5)
        g = df[df.setup == sid].set_index("qid").reindex(D.qids)
        preds[a] = ([norm(x) for x in g["prediction"].fillna("")], np.asarray(g["ans_agree"], float))
    same = np.array([float(a == b and a != "") for a, b in zip(preds["l3"][0], preds["l8"][0])])
    spaces["cross-model agreement (Llama 3B vs Llama 8B answers)"] = (
        np.column_stack([same, preds["l3"][1], preds["l8"][1], same * preds["l8"][1]]),
        ["Llama 3B and Llama 8B give the same answer", "agreement of Llama 3B's 5 answers", "agreement of Llama 8B's 5 answers",
         "same answer x Llama 8B agreement"], "needs both small answerers on the shared shortlist: about 3.9x")
    cheap_all = ["question facts", "question-passage similarity (TF-IDF)", "metadata of the cheap run (Qwen 7B - Llama 8B)"]
    spaces["all signals available after the cheap run"] = (np.column_stack([spaces[k][0] for k in cheap_all]),
                                                           sum([spaces[k][1] for k in cheap_all], []), "about 3.5x")
    return spaces, info


def build_targets(D):
    j = D.j
    small = D.EM[:, j(SMALL["ret"], SMALL["ans"], SMALL["cot"], SMALL["pick"], SMALL["merge"])]
    jc = j(REF["ret"], REF["cheap"], REF["cot"], REF["pick"], REF["merge"])
    jb = j(REF["ret"], REF["big"], REF["cot"], REF["pick"], REF["merge"])
    best = int(np.argmax(D.EM.mean(0)))
    main = [i for i, s in enumerate(D.setups) if D.meta[s][5] == 3 and D.meta[s][6] == 5]
    share = D.EM[:, main].mean(1)
    T = {
        "small pipeline is right": small,
        "cheap answer is no worse than the big one": (D.EM[:, jc] >= D.EM[:, jb]).astype(float),
        "the best setup is right": D.EM[:, best],
        "small retriever finds both passages": D.FOUND[:, j("q3", "l3", 1, "3", "all")],
        "easy for most setups (above the median)": (share > np.median(share)).astype(float),
    }
    return {k: np.asarray(v, int) for k, v in T.items()}, (jc, jb), share


# ==================================================================================
# Strategies: each returns held-out scores for one (space, target)
# ==================================================================================
def _prep(Xtr, Xte, cap=16):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    a, b = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))
    if cap and a.shape[1] > cap:
        p = PCA(cap, random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
    return a, b


def s_linear(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    a, b = _prep(Xtr, Xte)
    return LogisticRegression(C=0.3, max_iter=500).fit(a, ytr).predict_proba(b)[:, 1]


def s_pca2(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    a, b = _prep(Xtr, Xte, cap=2)
    return LogisticRegression(C=1.0, max_iter=500).fit(a[:, :2], ytr).predict_proba(b[:, :2])[:, 1]


def s_knn(Xtr, ytr, Xte):
    from sklearn.neighbors import KNeighborsClassifier
    a, b = _prep(Xtr, Xte)
    return KNeighborsClassifier(min(15, len(ytr) - 1)).fit(a, ytr).predict_proba(b)[:, 1]


def s_kmeans(Xtr, ytr, Xte, k=6):
    from sklearn.cluster import KMeans
    a, b = _prep(Xtr, Xte)
    km = KMeans(k, n_init=5, random_state=0).fit(a)
    base = ytr.mean()
    rate = np.array([(ytr[km.labels_ == c].sum() + 2 * base) / ((km.labels_ == c).sum() + 2) for c in range(k)])   # m-estimate, m = 2
    return rate[km.predict(b)]


def s_tree(Xtr, ytr, Xte):
    from sklearn.impute import SimpleImputer
    from sklearn.tree import DecisionTreeClassifier
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    return DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0).fit(imp.transform(Xtr), ytr).predict_proba(imp.transform(Xte))[:, 1]


def s_boost(Xtr, ytr, Xte):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    return GradientBoostingClassifier(n_estimators=40, max_depth=2, learning_rate=0.1, subsample=0.8, random_state=0).fit(
        imp.transform(Xtr), ytr).predict_proba(imp.transform(Xte))[:, 1]


def items_of(X, names, edges=None):
    """Turns numbers into yes/no items: a yes/no feature gives one item; a numeric one gives low / high (split at the training terciles)."""
    cols, labels, new_edges = [], [], []
    for jx in range(X.shape[1]):
        x = X[:, jx]
        ok = ~np.isnan(x)
        vals = np.unique(x[ok])
        if len(vals) <= 2:
            cols.append(np.nan_to_num(x, nan=0.0) > 0.5)
            labels.append(f"{names[jx]} = yes")
            new_edges.append(None)
            continue
        lo, hi = edges[jx] if edges is not None and edges[jx] is not None else (np.quantile(x[ok], 1 / 3), np.quantile(x[ok], 2 / 3))
        new_edges.append((lo, hi))
        cols.append(np.nan_to_num(x, nan=np.inf) <= lo)
        labels.append(f"{names[jx]} is low (<= {lo:.3g})")
        cols.append(np.nan_to_num(x, nan=-np.inf) > hi)
        labels.append(f"{names[jx]} is high (> {hi:.3g})")
    return np.column_stack(cols), labels, new_edges


def mine(I, y, min_support=0.08, max_len=2):
    """Association rules  items -> y = 1 / y = 0.  Returns (item ids, consequent, support, confidence, lift)."""
    n, base = len(y), y.mean()
    need = max(5, int(np.ceil(min_support * n)))
    out = []
    single = [i for i in range(I.shape[1]) if need <= I[:, i].sum() <= n - 1]
    sets = [(i,) for i in single]
    if max_len >= 2:
        sets += [c for c in itertools.combinations(single, 2)]
    for c in sets:
        m = I[:, c].all(1)
        sup = int(m.sum())
        if sup < need:
            continue
        conf = y[m].mean()
        for cons, cf, b in ((1, conf, base), (0, 1 - conf, 1 - base)):
            if b > 0:
                out.append((c, cons, sup, float(cf), float(cf / b)))
    return out


def s_rules(Xtr, ytr, Xte, names=None, top=12):
    """Score = the m-smoothed confidence of the strongest matching rule mined on the training questions."""
    names = names or [str(i) for i in range(Xtr.shape[1])]
    Itr, _, edges = items_of(Xtr, names)
    Ite, _, _ = items_of(Xte, names, edges)
    base = ytr.mean()
    rules = sorted(mine(Itr, ytr), key=lambda r: -abs(r[3] - (base if r[1] == 1 else 1 - base)) * np.sqrt(r[2]))[:top]
    score = np.full(len(Xte), base)
    best = np.zeros(len(Xte))
    for c, cons, sup, conf, lift in rules:
        m = Ite[:, c].all(1)
        sm = (conf * sup + 5 * (base if cons == 1 else 1 - base)) / (sup + 5)
        p1 = sm if cons == 1 else 1 - sm
        strength = abs(p1 - base)
        upd = m & (strength > best)
        score[upd], best[upd] = p1, strength
    return score


STRATEGIES = {
    "linear probe": s_linear,
    "first 2 principal components (PCA)": s_pca2,
    "nearest neighbours (t-SNE-style neighbourhoods)": s_knn,
    "k-means clusters (6)": s_kmeans,
    "decision tree (depth 3)": s_tree,
    "boosted trees": s_boost,
    "association rules": s_rules,
}


def cv_scores(fn, X, y, seed, names=None):
    from sklearn.model_selection import StratifiedKFold
    p = np.zeros(len(y))
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        if y[tr].min() == y[tr].max():
            p[te] = y[tr].mean()
            continue
        try:
            p[te] = fn(X[tr], y[tr], X[te], names=names) if fn is s_rules else fn(X[tr], y[tr], X[te])
        except Exception:
            p[te] = y[tr].mean()
    return p


def scoreboard(spaces, targets, reps, perms, quick=False):
    rows, oof = [], {}
    rng = np.random.default_rng(0)
    total = len(spaces) * len(STRATEGIES) * len(targets)
    done = 0
    for sp, (X, names, cost) in spaces.items():
        for st, fn in STRATEGIES.items():
            if st == "association rules" and sp.startswith(("question text", "question embedding", "question topics")):
                done += len(targets)
                continue                                   # association rules need named, readable features
            for tg, y in targets.items():
                done += 1
                if y.min() == y.max():
                    continue
                ps = [cv_scores(fn, X, y, 10 + r, names) for r in range(reps)]
                p = np.mean(ps, 0)
                a = R.auc(y, p)
                null = [R.auc(yp, cv_scores(fn, X, yp, 500 + i, names)) for i, yp in enumerate(rng.permutation(y) for _ in range(perms))]
                null = np.array([v for v in null if v == v])
                rows.append({"What is looked at": sp, "How structure is read": st, "What 'easy' means": tg,
                             "Separation on held-out questions (0.5 = useless)": a,
                             "Same method with shuffled labels: mean": float(null.mean()) if len(null) else float("nan"),
                             "Same method with shuffled labels: 95th percentile": float(np.percentile(null, 95)) if len(null) else float("nan"),
                             "Beats shuffled labels": bool(len(null) and a > np.percentile(null, 95)),
                             "Share of questions that are 'easy'": float(y.mean()), "Cost of having this signal": cost})
                oof[(sp, st, tg)] = p
        log(f"  scoreboard: {done}/{total}")
    return rows, oof


# ==================================================================================
# Detailed looks: clusters, topics, association rules, tree, importance, value
# ==================================================================================
def cluster_table(spaces, info, targets, D):
    from sklearn.cluster import KMeans
    rows, labels = [], {}
    for sp in ("question embedding", "question text (TF-IDF)", "metadata of the cheap run (Qwen 7B - Llama 8B)", "question-passage similarity (TF-IDF)"):
        if sp not in spaces:
            continue
        X = spaces[sp][0]
        a, _ = _prep(X, X)
        km = KMeans(6, n_init=10, random_state=0).fit(a)
        labels[sp] = km.labels_
        for c in range(6):
            m = km.labels_ == c
            ex = [D.text[i] for i in np.where(m)[0][:2]]
            r = {"What is clustered": sp, "Cluster": c + 1, "Questions": int(m.sum()), "Example questions": " / ".join(ex)}
            for tg, y in targets.items():
                r[f"Share where: {tg}"] = float(y[m].mean()) if m.any() else float("nan")
            rows.append(r)
    return rows, labels


def topic_table(spaces, info, targets, D):
    rows = []
    for t, words in enumerate(info["topics"]):
        m = info["topic_of"] == t
        r = {"Topic": t + 1, "Top words": words, "Questions": int(m.sum()), "Example question": D.text[np.where(m)[0][0]] if m.any() else ""}
        for tg, y in targets.items():
            r[f"Share where: {tg}"] = float(y[m].mean()) if m.any() else float("nan")
        rows.append(r)
    return rows


def rules_report(spaces, targets, D, top=15):
    """Mines on the Set 4 dev half, checks on the test half. Also mines with shuffled labels, to show how many rules survive by chance."""
    X, names, _ = spaces["all signals available after the cheap run"]
    dev, test = D.dev, D.test
    Idev, labels, edges = items_of(X[dev], names)
    Itest, _, _ = items_of(X[test], names, edges)
    rows, summary = [], []
    rng = np.random.default_rng(1)
    for tg in ("cheap answer is no worse than the big one", "small pipeline is right", "the best setup is right"):
        y = targets[tg]
        for kind, yd in (("real labels", y[dev]), ("shuffled labels (control)", rng.permutation(y[dev]))):
            base_d, base_t = yd.mean(), y[test].mean()
            rules = sorted([r for r in mine(Idev, yd) if r[4] > 1.0], key=lambda r: -(r[4] - 1) * np.sqrt(r[2]))[:top]
            held = 0
            for c, cons, sup, conf, lift in rules:
                m = Itest[:, c].all(1)
                st = int(m.sum())
                bt = base_t if cons == 1 else 1 - base_t
                ct = float((y[test][m] == cons).mean()) if st else float("nan")
                lt = ct / bt if st and bt > 0 else float("nan")
                held += int(st >= 5 and lt > 1.1)
                bd = base_d if cons == 1 else 1 - base_d
                rows.append({"Target": tg, "Mined on": kind, "IF": " AND ".join(labels[i] for i in c),
                             "THEN": tg if cons == 1 else f"NOT ({tg})", "dev questions matching": sup, "dev confidence": conf,
                             "dev confidence, m-smoothed (m = 5)": (conf * sup + 5 * bd) / (sup + 5), "dev lift": lift,
                             "test questions matching": st, "test confidence": ct, "test lift": lt,
                             "holds on test (lift > 1.1, at least 5 questions)": bool(st >= 5 and lt > 1.1)})
            summary.append({"Target": tg, "Mined on": kind, "Rules kept (strongest on dev)": len(rules), "Rules that hold on test": held,
                            "Share that hold on test": held / len(rules) if rules else float("nan")})
    return rows, summary


def tree_rules(spaces, targets, D):
    from sklearn.impute import SimpleImputer
    from sklearn.tree import DecisionTreeClassifier, export_text
    out, rows = [], []
    for sp in ("all signals available after the cheap run", "question facts", "question-passage similarity (TF-IDF)"):
        X, names, _ = spaces[sp]
        for tg in ("cheap answer is no worse than the big one", "small pipeline is right"):
            y = targets[tg]
            imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[D.dev])
            t = DecisionTreeClassifier(max_depth=3, min_samples_leaf=10, random_state=0).fit(imp.transform(X[D.dev]), y[D.dev])
            p = t.predict_proba(imp.transform(X[D.test]))[:, 1]
            a = R.auc(y[D.test], p)
            acc = float(((p >= 0.5) == y[D.test]).mean())
            out.append(f"=== What is looked at: {sp}\n=== Predicting: {tg}\n=== Learned on dev (100); on test: separation {a:.2f}, "
                       f"accuracy {acc:.0%} (always guessing the majority: {max(y[D.test].mean(), 1 - y[D.test].mean()):.0%})\n"
                       + export_text(t, feature_names=list(names), decimals=2) + "\n")
            rows.append({"What is looked at": sp, "Predicting": tg, "Test separation": a, "Test accuracy": acc,
                         "Test accuracy of always guessing the majority": float(max(y[D.test].mean(), 1 - y[D.test].mean()))})
    return "\n".join(out), rows


def importance(spaces, targets):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import StratifiedKFold
    X, names, _ = spaces["all signals available after the cheap run"]
    rows = []
    for tg in ("cheap answer is no worse than the big one", "small pipeline is right"):
        y = targets[tg]
        imp_sum = np.zeros(X.shape[1])
        for tr, te in StratifiedKFold(5, shuffle=True, random_state=0).split(X, y):
            im = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr])
            g = GradientBoostingClassifier(n_estimators=40, max_depth=2, subsample=0.8, random_state=0).fit(im.transform(X[tr]), y[tr])
            pi = permutation_importance(g, im.transform(X[te]), y[te], scoring="roc_auc", n_repeats=10, random_state=0)
            imp_sum += pi.importances_mean / 5
        for i in np.argsort(-imp_sum):
            rows.append({"Predicting": tg, "Signal": names[i], "Drop in held-out separation when this signal is scrambled": float(imp_sum[i])})
    return rows


def value_table(D, oof, rows, pair, targets, share=0.35):
    """What a signal would buy as a gate on the reference pair: escalate the questions the signal is least sure about."""
    jc, jb = pair
    tg = "cheap answer is no worse than the big one"
    cheap_em, big_em = D.EM[:, jc], D.EM[:, jb]
    cheap_cost, esc_cost = D.FL[:, jc], D.FL[:, jc] + D.FA[:, jb]
    out = [{"Gate": "always the cheap answerer", "Accuracy": float(cheap_em.mean()), "Compute (x cheapest)": float(cheap_cost.mean() / D.ref),
            "Share escalated": 0.0, "Separation": float("nan")},
           {"Gate": "always the big answerer (fixed)", "Accuracy": float(big_em.mean()), "Compute (x cheapest)": float(D.FL[:, jb].mean() / D.ref),
            "Share escalated": 1.0, "Separation": float("nan")},
           {"Gate": f"escalate a random {round(100 * share)}%", "Accuracy": float((1 - share) * cheap_em.mean() + share * big_em.mean()),
            "Compute (x cheapest)": float(((1 - share) * cheap_cost.mean() + share * esc_cost.mean()) / D.ref), "Share escalated": share,
            "Separation": 0.5}]
    cand = sorted([r for r in rows if r["What 'easy' means"] == tg], key=lambda r: -r["Separation on held-out questions (0.5 = useless)"])
    seen = set()
    for r in cand:
        if r["What is looked at"] in seen:
            continue
        seen.add(r["What is looked at"])
        p = oof[(r["What is looked at"], r["How structure is read"], tg)]
        thr = np.quantile(p, share)
        esc = p <= thr
        em = np.where(esc, big_em, cheap_em)
        co = np.where(esc, esc_cost, cheap_cost)
        out.append({"Gate": f"{r['What is looked at']} | {r['How structure is read']}", "Accuracy": float(em.mean()),
                    "Compute (x cheapest)": float(co.mean() / D.ref), "Share escalated": float(esc.mean()),
                    "Separation": r["Separation on held-out questions (0.5 = useless)"], "Cost of having this signal": r["Cost of having this signal"]})
    return out


# ==================================================================================
# Charts
# ==================================================================================
def charts(out, D, spaces, info, targets, rows, clus, clabels, topics, rrows, rsum, imp, val, quick):
    plt = S._plt()
    os.makedirs(out, exist_ok=True)
    tnames = list(targets)
    key = "Separation on held-out questions (0.5 = useless)"
    # 1. scoreboard: best strategy per space, per target
    sps = list(spaces)
    M = np.full((len(sps), len(tnames)), np.nan)
    star = np.zeros_like(M, bool)
    for i, sp in enumerate(sps):
        for j, tg in enumerate(tnames):
            g = [r for r in rows if r["What is looked at"] == sp and r["What 'easy' means"] == tg]
            if g:
                b = max(g, key=lambda r: r[key])
                M[i, j], star[i, j] = b[key], b["Beats shuffled labels"]
    fig, ax = plt.subplots(figsize=(11.5, 0.55 * len(sps) + 2.2))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
    for i in range(len(sps)):
        for j in range(len(tnames)):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if star[i, j] else ""), ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j] - 0.5) > 0.2 else "black")
    ax.set_xticks(range(len(tnames)))
    ax.set_xticklabels([t.replace(" (", "\n(").replace(" is no worse", "\nis no worse").replace(" finds", "\nfinds") for t in tnames], fontsize=8)
    ax.set_yticks(range(len(sps)))
    ax.set_yticklabels(sps, fontsize=8)
    ax.grid(False)
    ax.set_title("Where is there structure? Best held-out separation per space and target\n(0.5 = useless; * = beats the same method with shuffled labels)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(out, "1_scoreboard_spaces.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 2. strategies x spaces for the gate target
    tg = "cheap answer is no worse than the big one"
    sts = list(STRATEGIES)
    M = np.full((len(sps), len(sts)), np.nan)
    star = np.zeros_like(M, bool)
    for r in rows:
        if r["What 'easy' means"] == tg:
            i, j = sps.index(r["What is looked at"]), sts.index(r["How structure is read"])
            M[i, j], star[i, j] = r[key], r["Beats shuffled labels"]
    fig, ax = plt.subplots(figsize=(12.5, 0.55 * len(sps) + 2.4))
    im = ax.imshow(M, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
    for i in range(len(sps)):
        for j in range(len(sts)):
            if M[i, j] == M[i, j]:
                ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if star[i, j] else ""), ha="center", va="center", fontsize=9,
                        color="white" if abs(M[i, j] - 0.5) > 0.2 else "black")
    ax.set_xticks(range(len(sts)))
    ax.set_xticklabels([s.replace(" (", "\n(") for s in sts], fontsize=8)
    ax.set_yticks(range(len(sps)))
    ax.set_yticklabels(sps, fontsize=8)
    ax.grid(False)
    ax.set_title(f"Which way of reading structure works? Target: '{tg}'\n(held-out separation; * = beats shuffled labels; blank = not applicable)", fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.savefig(os.path.join(out, "2_scoreboard_strategies.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 3 + 4. PCA and t-SNE pictures
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    show = [s for s in ("question embedding", "question text (TF-IDF)", "question-passage similarity (TF-IDF)",
                        "metadata of the cheap run (Qwen 7B - Llama 8B)", "all signals available after the cheap run") if s in spaces][:4]
    col_t = ["cheap answer is no worse than the big one", "small pipeline is right"]
    for fname, kind in (("3_pca_maps.png", "PCA"), ("4_tsne_maps.png", "t-SNE")):
        fig, axs = plt.subplots(len(col_t), len(show), figsize=(4.2 * len(show), 3.9 * len(col_t)), squeeze=False)
        for j, sp in enumerate(show):
            a, _ = _prep(spaces[sp][0], spaces[sp][0])
            Z = PCA(2, random_state=0).fit_transform(a) if kind == "PCA" else \
                TSNE(2, perplexity=20 if not quick else 10, init="pca", random_state=0, max_iter=500 if quick else 1000).fit_transform(a)
            for i, tgn in enumerate(col_t):
                y = targets[tgn]
                ax = axs[i, j]
                ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=16, color=S.SERIES[0], alpha=0.75, label="yes")
                ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=16, color=S.SERIES[1], alpha=0.75, label="no")
                g = [r for r in rows if r["What is looked at"] == sp and r["What 'easy' means"] == tgn]
                best = max(g, key=lambda r: r[key]) if g else None
                ax.set_title(f"{sp}\ncoloured by: {tgn}\nbest held-out separation {best[key]:.2f}" + ("*" if best and best["Beats shuffled labels"] else ""),
                             fontsize=7.5)
                ax.set_xticks([]); ax.set_yticks([])
                if i == 0 and j == 0:
                    ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"{kind} maps of the questions. A picture is not evidence: read the held-out number in each title.", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 5. clusters
    cs = list(dict.fromkeys(r["What is clustered"] for r in clus))
    if cs:
        fig, axs = plt.subplots(1, len(cs), figsize=(4.6 * len(cs), 4.0), squeeze=False)
        for ax, sp in zip(axs[0], cs):
            g = [r for r in clus if r["What is clustered"] == sp]
            x = np.arange(len(g))
            for k, (tgn, colr) in enumerate(zip(col_t, (S.SERIES[0], S.SERIES[2]))):
                ax.bar(x + (k - 0.5) * 0.38, [r[f"Share where: {tgn}"] for r in g], 0.38, color=colr, label=tgn)
                ax.axhline(targets[tgn].mean(), color=colr, ls=":", lw=1)
            ax.set_xticks(x)
            ax.set_xticklabels([f"{r['Cluster']}\n(n={r['Questions']})" for r in g], fontsize=8)
            ax.set_ylim(0, 1.05)
            ax.set_title(sp, fontsize=8)
            ax.set_xlabel("cluster (k-means, 6)")
        axs[0][0].set_ylabel("share of the cluster's questions")
        axs[0][0].legend(frameon=False, fontsize=7, loc="lower left")
        fig.suptitle("Do clusters differ in difficulty? (dotted = average over all questions; in-sample, see the scoreboard for held-out)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "5_clusters.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 6. topics
    fig, ax = plt.subplots(figsize=(10.5, 4.4))
    x = np.arange(len(topics))
    for k, (tgn, colr) in enumerate(zip(["small pipeline is right", "cheap answer is no worse than the big one", "the best setup is right"],
                                         (S.SERIES[1], S.SERIES[0], S.SERIES[2]))):
        ax.bar(x + (k - 1) * 0.27, [r[f"Share where: {tgn}"] for r in topics], 0.27, color=colr, label=tgn)
        ax.axhline(targets[tgn].mean(), color=colr, ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"topic {r['Topic']} (n={r['Questions']})\n" + "\n".join(r["Top words"].split(", ")[:4]) for r in topics], fontsize=7.5)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("share of the topic's questions")
    ax.set_title("Are some topics easier? (topic model on the question text; dotted = average)", fontsize=10)
    ax.legend(frameon=False, fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(os.path.join(out, "6_topics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 7. association rules: dev lift against test lift
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for kind, colr, mk in (("real labels", S.SERIES[0], "o"), ("shuffled labels (control)", S.INK2, "x")):
        g = [r for r in rrows if r["Mined on"] == kind and r["test lift"] == r["test lift"]]
        ax.scatter([r["dev lift"] for r in g], [r["test lift"] for r in g], s=26, color=colr, marker=mk, alpha=0.7, label=kind)
    lim = [0.4, max(2.2, max([r["dev lift"] for r in rrows] + [1]) * 1.05)]
    ax.plot(lim, lim, color=S.INK2, ls=":", lw=1)
    ax.axhline(1, color=S.INK2, lw=0.8)
    ax.set_xlabel("lift on dev (where the rule was found)")
    ax.set_ylabel("lift on test (new questions)")
    ax.set_title("Do association rules survive on new questions?\n(above 1 = the rule still helps; crosses = rules found in shuffled labels)", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(out, "7_association_rules.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 8. importance
    tgn = "cheap answer is no worse than the big one"
    g = [r for r in imp if r["Predicting"] == tgn][:12][::-1]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.barh(range(len(g)), [r["Drop in held-out separation when this signal is scrambled"] for r in g], color=S.SERIES[0])
    ax.set_yticks(range(len(g)))
    ax.set_yticklabels([r["Signal"] for r in g], fontsize=8)
    ax.axvline(0, color=S.INK2, lw=0.8)
    ax.set_xlabel("drop in held-out separation when the signal is scrambled")
    ax.set_title(f"Which single signals matter? Target: '{tgn}'", fontsize=10)
    fig.savefig(os.path.join(out, "8_signal_importance.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 9. value
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    for i, r in enumerate(val):
        base = r["Gate"].startswith(("always", "escalate a random"))
        ax.scatter(r["Compute (x cheapest)"], r["Accuracy"], s=80, marker="s" if base else "o", color=S.INK2 if base else S.SERIES[(i - 3) % len(S.SERIES)],
                   label=r["Gate"], zorder=3, edgecolor=S.SURFACE)
    ax.set_xlabel("compute (x cheapest)")
    ax.set_ylabel("accuracy, held-out gate scores, all 200 questions")
    ax.set_title("What each signal would buy as a gate between the cheap and the big answerer\n(35% of questions escalated; squares = baselines)", fontsize=10)
    ax.legend(frameon=False, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.savefig(os.path.join(out, "9_what_a_signal_buys.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ==================================================================================
def cmd_offline(args):
    import pandas as pd
    t0 = time.time()
    out = os.path.join(args.run_dir, "results")
    os.makedirs(out, exist_ok=True)
    D = R.Data(args.sweep_run, args.data_dir, keep_df=True)
    qs = S.load_questions(args.data_dir)
    spaces, info = build_spaces(D, qs, try_embedding=not args.no_embedding)
    log("embedding:", info["embedding"])
    targets, pair, share = build_targets(D)
    log("spaces:", {k: v[0].shape[1] for k, v in spaces.items()})
    rows, oof = scoreboard(spaces, targets, args.repeats, args.permutations)
    clus, clabels = cluster_table(spaces, info, targets, D)
    topics = topic_table(spaces, info, targets, D)
    rrows, rsum = rules_report(spaces, targets, D)
    ttext, trows = tree_rules(spaces, targets, D)
    imp = importance(spaces, targets)
    val = value_table(D, oof, rows, pair, targets)
    charts(os.path.join(out, "charts"), D, spaces, info, targets, rows, clus, clabels, topics, rrows, rsum, imp, val, args.quick)

    def dump(name, rr):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    dump("scoreboard.jsonl", rows); dump("clusters.jsonl", clus); dump("topics.jsonl", topics); dump("association_rules.jsonl", rrows)
    dump("association_rules_summary.jsonl", rsum); dump("tree_rules.jsonl", trows); dump("signal_importance.jsonl", imp); dump("what_a_signal_buys.jsonl", val)
    open(os.path.join(out, "tree_rules.txt"), "w", encoding="utf-8").write(ttext)
    perq = []
    for i, q in enumerate(D.qids):
        r = {"qid": q, "split": D.split[i], "question": D.text[i], "share of main setups that answer it": float(share[i]),
             "topic": int(info["topic_of"][i]) + 1}
        r.update({f"target: {k}": int(v[i]) for k, v in targets.items()})
        r.update({f"cluster in: {k}": int(v[i]) + 1 for k, v in clabels.items()})
        perq.append(r)
    dump("per_question_structure.jsonl", perq)

    key = "Separation on held-out questions (0.5 = useless)"
    pct = {"Share of questions that are 'easy'", "Accuracy", "Share escalated", "dev confidence", "test confidence", "dev confidence, m-smoothed (m = 5)",
           "Share that hold on test", "Test accuracy", "Test accuracy of always guessing the majority"} | {f"Share where: {t}" for t in targets}
    nums = {key, "Same method with shuffled labels: mean", "Same method with shuffled labels: 95th percentile", "Separation", "dev lift", "test lift",
            "Test separation", "Drop in held-out separation when this signal is scrambled"}
    pct = pct | {"Share of its combinations that beat shuffled labels"}
    xs = {"Compute (x cheapest)"}
    T = lambda rr, cols: R.md_table(rr, cols, pct, xs, nums)
    best_space = []
    for tg in targets:
        for sp in spaces:
            g = [r for r in rows if r["What is looked at"] == sp and r["What 'easy' means"] == tg]
            if g:
                best_space.append(max(g, key=lambda r: r[key]))
    wins = sum(r["Beats shuffled labels"] for r in rows)
    L = ["# Set 5d part 1 - looking for hidden structure in the Set 4 results\n",
         f"Sweep used: `{os.path.basename(args.sweep_run.rstrip('/'))}`. {D.n} questions. No new model answers were generated. "
         f"Embedding model: {info['embedding']}.",
         f"Every separation score is on held-out questions (5-fold cross-validation x{args.repeats}). 0.5 = useless, 1.0 = perfect. "
         f"'Beats shuffled labels' = higher than the 95th percentile of the same method run on {args.permutations} shuffles of the labels.",
         f"{wins} of {len(rows)} combinations beat shuffled labels; by chance alone about {round(0.05 * len(rows))} would.\n",
         "## 1. What 'easy' means here\n",
         R.md_table([{"Target": k, "Share of the 200 questions": float(v.mean())} for k, v in targets.items()], ["Target", "Share of the 200 questions"],
                    {"Share of the 200 questions"}, xs, nums),
         "\n## 2. Top 25 combinations on the scoreboard\n",
         T(sorted(rows, key=lambda r: -r[key])[:25], ["What is looked at", "How structure is read", "What 'easy' means", key,
                                                      "Same method with shuffled labels: 95th percentile", "Beats shuffled labels", "Cost of having this signal"]),
         "\n## 3. Best way of reading each space, for the routing target ('cheap answer is no worse than the big one')\n",
         T(sorted([r for r in best_space if r["What 'easy' means"] == "cheap answer is no worse than the big one"], key=lambda r: -r[key]),
           ["What is looked at", "How structure is read", key, "Same method with shuffled labels: 95th percentile", "Beats shuffled labels",
            "Cost of having this signal"]),
         "\n## 4. Which way of reading structure works best? (mean over all spaces and targets)\n",
         T([{"How structure is read": st, key: float(np.nanmean([r[key] for r in rows if r["How structure is read"] == st])),
             "Share of its combinations that beat shuffled labels": float(np.mean([r["Beats shuffled labels"] for r in rows if r["How structure is read"] == st]))}
            for st in STRATEGIES], ["How structure is read", key, "Share of its combinations that beat shuffled labels"]).replace("True", "yes"),
         "\n## 5. What the best signal of each space would buy as a gate (35% of questions escalated, reference pair)\n",
         T(val, ["Gate", "Accuracy", "Compute (x cheapest)", "Share escalated", "Separation"]),
         "\n## 6. Topics in the question text\n",
         T(topics, ["Topic", "Top words", "Questions"] + [f"Share where: {t}" for t in ("small pipeline is right", "cheap answer is no worse than the big one",
                                                                                         "the best setup is right")]),
         "\n## 7. Association rules: found on dev, checked on test\n",
         "A rule 'holds on test' if it still has lift above 1.1 on at least 5 test questions. The control row shows how many rules "
         "hold when they were mined from shuffled labels, that is, by chance.\n",
         T(rsum, ["Target", "Mined on", "Rules kept (strongest on dev)", "Rules that hold on test", "Share that hold on test"]),
         "\n### The strongest rules for 'cheap answer is no worse than the big one'\n",
         T([r for r in rrows if r["Mined on"] == "real labels" and r["Target"] == "cheap answer is no worse than the big one"][:12],
           ["IF", "THEN", "dev questions matching", "dev confidence", "dev confidence, m-smoothed (m = 5)", "dev lift", "test questions matching",
            "test confidence", "test lift", "holds on test (lift > 1.1, at least 5 questions)"]),
         "\n## 8. Small decision trees (readable rules), learned on dev and scored on test\n",
         T(trows, ["What is looked at", "Predicting", "Test separation", "Test accuracy", "Test accuracy of always guessing the majority"]),
         "\nThe trees themselves are in `results/tree_rules.txt`.\n",
         "## 9. Which single signals matter (for 'cheap answer is no worse than the big one')\n",
         T([r for r in imp if r["Predicting"] == "cheap answer is no worse than the big one"][:10],
           ["Signal", "Drop in held-out separation when this signal is scrambled"]),
         "\nCharts: `results/charts/` (9). Excel: `results/set5d_structure.xlsx`. Clusters: `results/clusters.jsonl`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    W.write_excel(os.path.join(out, "set5d_structure.xlsx"), {
        "Scoreboard": pd.DataFrame(rows), "What a signal buys": pd.DataFrame(val), "Topics": pd.DataFrame(topics), "Clusters": pd.DataFrame(clus),
        "Association rules": pd.DataFrame(rrows), "Rules summary": pd.DataFrame(rsum), "Decision trees": pd.DataFrame(trows),
        "Signal importance": pd.DataFrame(imp), "Per question": pd.DataFrame(perq)})
    json.dump({"version": VERSION, "seconds": round(time.time() - t0), "questions": D.n, "embedding": info["embedding"], "spaces": list(spaces),
               "strategies": list(STRATEGIES), "targets": list(targets), "repeats": args.repeats, "permutations": args.permutations,
               "combinations": len(rows), "combinations beating shuffled labels": int(wins)},
              open(os.path.join(args.run_dir, "structure_status.json"), "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("offline")
    p.add_argument("--sweep_run", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--permutations", type=int, default=20)
    p.add_argument("--no_embedding", action="store_true")
    p.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    cmd_offline(args)


if __name__ == "__main__":
    main()
