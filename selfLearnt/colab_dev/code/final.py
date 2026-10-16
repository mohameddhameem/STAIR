"""Set 5c - the final test of the frozen gate on fresh questions (notebook 17).

The gate found in Set 5d part 2 (notebook 16): the Llama 8B answers ONCE; a probe reads its hidden state on the final answer line;
the questions the probe is least sure about go to the Qwen 32B. On the 200 Set 4 questions it matched the cheap fixed setup
(Qwen 7B retrieves once, Qwen 32B answers once) at about a quarter less compute - but the layer and the share sent to the big model
were chosen on those same questions. This file tests it properly:

  build_fresh  CPU  200 HotpotQA questions that no earlier set has used, built exactly like the earlier ones (seed 43)
  freeze       CPU  fits the gate on the OLD 200 questions only; layer and threshold chosen by a written rule; saved with a checksum
  retrieve     GPU  Qwen 7B retrieves on the fresh questions (CoT, select 3; the Set 4 code, unchanged)
  answer       GPU  Qwen 32B answers on the 'keep all' shortlists (the Set 4 code, unchanged)
  (hidden.py extract --role answer --model l8 gives the Llama 8B's single answer and its hidden state, unchanged)
  evaluate     CPU  applies the frozen gate; nothing is fitted or chosen on the fresh questions

The success test was fixed in advance: accuracy within 2 points of the cheap fixed setup at 15% or more less compute.
"""
import argparse
import hashlib
import json
import os
import random
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S          # noqa: E402
import sweep as W        # noqa: E402
import hidden as H       # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "5c.1"
RET, COT, PICK, RK, MERGE, BIG, CHEAP = H.RET, H.COT, H.PICK, H.RK, H.MERGE, H.BIG, "l8"
SHARES = [round(0.05 * i, 2) for i in range(1, 20)]          # escalation shares for which a threshold is frozen
TOL_POINTS, MIN_SAVING = 2.0, 0.15                           # the success test
SIGNALS = {"su": "semantic uncertainty probe on the final answer line",
           "probe": "plain linear probe on the final answer line",
           "attention": "evidence-use attention signals",
           "tokens": "token probabilities of the single answer"}
TOKEN_COLS = ["mean_logprob", "min_logprob", "mean_entropy", "answer_line_logprob", "out_tokens"]
log = H.log


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ==================================================================================
# Fresh questions
# ==================================================================================
def cmd_build_fresh(args):
    """Same construction as ssw.cmd_build_data (gold passages + BM25 distractors from the whole HotpotQA dev corpus, shuffled),
    but sampled only from questions that no earlier question set contains."""
    with open(args.hotpot_dev, encoding="utf-8") as f:
        data = json.load(f)
    used = set()
    for d in args.exclude:
        for sp in ("dev", "test"):
            p = os.path.join(d, f"{sp}.json")
            if os.path.exists(p):
                used |= {e["qid"] for e in json.load(open(p, encoding="utf-8"))}
    seen, corpus = set(), []
    for item in data:
        for _, sents in item["context"]:
            t = " ".join(sents)
            if t not in seen:
                seen.add(t)
                corpus.append(t)
    pool = [i for i, ex in enumerate(data) if ex["_id"] not in used]
    log(f"HotpotQA dev: {len(data)} questions, {len(used)} used by earlier sets, {len(pool)} never used; building BM25 over {len(corpus)} paragraphs ...")
    bm25 = S.BM25(corpus)
    rng = random.Random(args.seed)
    picked = rng.sample(pool, args.n)
    half = args.n // 2
    os.makedirs(args.out_dir, exist_ok=True)
    for split, idxs in (("dev", picked[:half]), ("test", picked[half:])):
        out = []
        for i in idxs:
            ex = data[i]
            ctx = {t: " ".join(s) for t, s in ex["context"]}
            gold = []
            for title, _ in ex["supporting_facts"]:
                if ctx[title] not in gold:
                    gold.append(ctx[title])
            cands = gold.copy()
            for j in bm25.search(ex["question"], args.n_candidates + 10):
                if len(cands) >= args.n_candidates:
                    break
                if corpus[j] not in cands:
                    cands.append(corpus[j])
            rng.shuffle(cands)
            out.append({"qid": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type"),
                        "ret_answers": [cands.index(g) for g in gold], "ctxs_candidate": cands})
        assert not ({e["qid"] for e in out} & used)
        with open(os.path.join(args.out_dir, f"{split}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1, ensure_ascii=False)
        log(f"wrote {split}.json: {len(out)} questions x {args.n_candidates} paragraphs")
    S.write_json(os.path.join(args.out_dir, "data_config.json"),
                 {"source": "HotpotQA dev (distractor), same construction as the earlier sets", "n": args.n, "n_candidates": args.n_candidates,
                  "seed": args.seed, "excluded_question_sets": [os.path.basename(d.rstrip("/")) for d in args.exclude], "excluded_questions": len(used),
                  "note": "the two files (dev.json, test.json) are just two halves of one fresh sample; nothing is tuned on either"})


# ==================================================================================
# Frozen linear readers (plain numpy, so that nothing depends on a library version when they are applied)
# ==================================================================================
def fit_reduce(X, k=16):
    """The same unsupervised reduction as hidden.reduce_once, returned as numbers that can be applied to new questions."""
    from sklearn.decomposition import PCA
    X = np.asarray(X, np.float64)
    ok = ~np.isnan(X).any(1)
    mu, sd = X[ok].mean(0), X[ok].std(0) + 1e-6
    p = PCA(min(k, max(1, ok.sum() - 1), X.shape[1]), random_state=0).fit((X[ok] - mu) / sd)
    return {"r_mu": mu, "r_sd": sd, "r_mean": p.mean_, "r_comp": p.components_}


def apply_reduce(fz, X):
    X = np.asarray(X, np.float64)
    ok = ~np.isnan(X).any(1)
    out = np.full((len(X), fz["r_comp"].shape[0]), np.nan)
    out[ok] = ((X[ok] - fz["r_mu"]) / fz["r_sd"] - fz["r_mean"]) @ fz["r_comp"].T
    return out


def fit_linear(X, y, k=16, C=0.3):
    """The same reader as hidden.f_linear (median fill, standardise, at most k principal components, logistic regression)."""
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    X, y = np.asarray(X, np.float64), np.asarray(y, float)
    ok = ~np.isnan(y)
    X, y = X[ok], y[ok].astype(int)
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X)
    a = imp.transform(X)
    sc = StandardScaler().fit(a)
    a = sc.transform(a)
    fz = {"l_med": imp.statistics_, "l_mu": sc.mean_, "l_sd": sc.scale_}
    if a.shape[1] > k:
        p = PCA(min(k, len(a) - 1), random_state=0).fit(a)
        a = p.transform(a)
        psd = a.std(0) + 1e-9
        a = a / psd
        fz.update({"l_pmean": p.mean_, "l_pcomp": p.components_, "l_psd": psd})
    lr = LogisticRegression(C=C, max_iter=500).fit(a, y)
    fz.update({"l_w": lr.coef_[0], "l_b": lr.intercept_})
    return fz


def apply_linear(fz, X):
    X = np.asarray(X, np.float64).copy()
    med = np.broadcast_to(fz["l_med"], X.shape)
    X[np.isnan(X)] = med[np.isnan(X)]
    a = (X - fz["l_mu"]) / fz["l_sd"]
    if "l_pcomp" in fz:
        a = ((a - fz["l_pmean"]) @ fz["l_pcomp"].T) / fz["l_psd"]
    return 1.0 / (1.0 + np.exp(-(a @ fz["l_w"] + fz["l_b"][0])))


def tables(E):
    """The two small tables read from one cheap answer, built exactly as in hidden.cmd_analyze."""
    tab = lambda cols: np.array([[np.nan if r is None or r.get(c) is None else float(r[c]) for c in cols] for r in E["rows"]], float)
    att0 = E["attention"]
    n_att = len(H.ATT_NAMES)
    rel = []
    for o in (0, n_att):
        den = att0[:, o] + att0[:, o + 1] + att0[:, o + 2]
        rel += [att0[:, o] / den, att0[:, o + 1] / den, att0[:, o] / (att0[:, o] + att0[:, o + 2])]
    return {"attention": np.column_stack([att0] + rel + [tab(["n_passages"])]), "tokens": tab(TOKEN_COLS)}


def scores_from(E, FZ, meta):
    """Applies every frozen reader to an extraction (old or fresh) -> {signal: score per question}; low score = send to the big model."""
    T = tables(E)
    out = {}
    for sig in SIGNALS:
        fz = {k[len(sig) + 1:]: v for k, v in FZ.items() if k.startswith(sig + "_")}
        if sig in ("su", "probe"):
            X = apply_reduce(fz, E["h_answer_line"][:, meta["signals"][sig]["layer_index"], :])
        else:
            X = T[sig]
        p = apply_linear(fz, X)
        p[~E["have"]] = np.nan
        out[sig] = p
    return out


# ==================================================================================
# Freeze: everything is decided here, on the old questions only
# ==================================================================================
def cmd_freeze(args):
    import routers as R
    t0 = time.time()
    D = R.Data(args.sweep_run, args.data_dir)
    P = {k: v["params"] for k, v in W.load_models(args.models_json).items()}
    E = H.load_extract(args.hidden_run, CHEAP, "answer", D.qids)
    assert E is not None and E["have"].all(), "the old extraction for the Llama 8B is not complete"
    j_big = D.j(RET, BIG, COT, PICK, MERGE, RK, 1)
    big = D.EM[:, j_big]
    em = np.array([r["EM"] for r in E["rows"]], float)
    y_route = (em >= big).astype(float)                                           # the cheap answer is no worse than the big one
    y_agree = (D.AG[:, D.j(RET, CHEAP, COT, PICK, MERGE, RK, 5)] >= 0.999).astype(float)   # the Llama 8B's 5 Set 4 answers all agree
    pt = np.array([r["prompt_tokens"] for r in E["rows"]], float)
    ot = np.array([r["out_tokens"] for r in E["rows"]], float)
    cheap = D.FR[:, j_big] + 2 * P[CHEAP] * (pt + ot)
    tail = D.FA[:, j_big]
    bar_acc, bar_cost = float(big.mean()), float(D.FL[:, j_big].mean() / D.ref)
    T = tables(E)
    reps = args.repeats
    meta = {"version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "frozen_on": "the 200 Set 4 questions only",
            "sweep_run": os.path.basename(args.sweep_run.rstrip("/")), "hidden_run": os.path.basename(args.hidden_run.rstrip("/")),
            "cheap_answerer": CHEAP, "big_answerer": BIG, "retriever": RET, "layers": E["layers"], "ref_flops": D.ref, "shares": SHARES,
            "old": {"questions": int(D.n), "bar_accuracy": bar_acc, "bar_compute_x": bar_cost, "cheap_single_accuracy": float(em.mean()),
                    "cheap_single_compute_x": float(cheap.mean() / D.ref), "share where the cheap answer is no worse": float(y_route.mean()),
                    "share where the 5 answers agree": float(y_agree.mean())},
            "signals": {}}
    FZ, OLD = {}, {"y_route": y_route, "em_cheap": em, "em_big": big}

    def oof(X, y):
        return np.nanmean([H.cv(H.f_linear, X, y, 10 + r) for r in range(reps)], 0)

    def curve(p):
        rows = []
        for s in SHARES:
            thr = float(np.nanquantile(p, s))
            esc = p <= thr
            rows.append({"share": s, "threshold": thr, "share_escalated": float(esc.mean()), "accuracy": float(np.where(esc, big, em).mean()),
                         "compute_x": float(np.where(esc, cheap + tail, cheap).mean() / D.ref)})
        return rows

    for sig, label in SIGNALS.items():
        train_y = y_agree if sig == "su" else y_route
        if sig in ("su", "probe"):
            cands = []
            for li in range(len(E["layers"])):
                Xr = H.reduce_once(E["h_answer_line"][:, li, :])
                p = oof(Xr, train_y)
                cands.append((H.auc(y_route, p), li, p))
            a, li, p = max(cands, key=lambda c: c[0])
            red = fit_reduce(E["h_answer_line"][:, li, :])
            Xr = apply_reduce(red, E["h_answer_line"][:, li, :])
            assert np.allclose(Xr, H.reduce_once(E["h_answer_line"][:, li, :]), atol=1e-6), "frozen reduction differs from the analysis"
            fz = dict(red, **fit_linear(Xr, train_y))
            info = {"layer_index": li, "layer": E["layers"][li], "of_layers": max(E["layers"]),
                    "held-out separation at every layer": {str(E["layers"][c[1]]): c[0] for c in cands}}
        else:
            X = T[sig]
            p = oof(X, train_y)
            a = H.auc(y_route, p)
            fz = fit_linear(X, train_y)
            Xr, info = X, {}
        full = apply_linear(fz, Xr)                                               # the frozen reader on its own training questions
        ref_p = H.f_linear(Xr, train_y.astype(int), Xr)
        assert np.allclose(full, ref_p, atol=1e-6), f"frozen reader for {sig} does not reproduce the analysis code"
        meta["signals"][sig] = dict(info, label=label, trained_to_predict="whether the Llama 8B's 5 answers agree" if sig == "su" else "whether the cheap answer is no worse than the big one",
                                    **{"held-out separation (routing target)": a, "held-out separation (cheap answer is right)": H.auc(em, p),
                                       "held-out curve": curve(p)})
        for k, v in fz.items():
            FZ[f"{sig}_{k}"] = np.asarray(v, np.float64)
        OLD[f"oof_{sig}"] = p
        log(f"{label}: held-out separation {a:.3f}" + (f" (layer {info['layer']} of {info['of_layers']})" if info else ""))
    # the primary gate, chosen by a rule written before looking at any fresh question
    cv_rows = meta["signals"]["su"]["held-out curve"]
    ok = [r for r in cv_rows if r["accuracy"] >= bar_acc - 1e-9]
    pick = min(ok, key=lambda r: r["share"]) if ok else max(cv_rows, key=lambda r: r["accuracy"])
    meta["primary"] = {"signal": "su", "share": pick["share"], "threshold": pick["threshold"], "held-out accuracy (old questions)": pick["accuracy"],
                       "held-out compute_x (old questions)": pick["compute_x"],
                       "rule for the layer": "the layer whose semantic uncertainty probe separates the routing target best on held-out old questions",
                       "rule for the threshold": "the smallest share sent to the big model at which held-out accuracy on the old questions is at least "
                                                 "the cheap fixed setup's; the threshold is that share's score quantile on the old questions"}
    s = meta["signals"]["su"]
    meta["rule in words"] = (f"The Llama 8B answers once on the Qwen 7B's shortlist. Read its hidden state on the final answer line at layer {s['layer']} of "
                             f"{s['of_layers']} and compute the semantic uncertainty score (the probability, from 0 to 1, that five answers would agree). "
                             f"IF the score is at most {pick['threshold']:.3f} THEN discard that answer and let the Qwen 32B answer once; "
                             f"OTHERWISE keep the Llama 8B's answer.")
    meta["success test"] = f"accuracy within {TOL_POINTS:.0f} points of the cheap fixed setup on the fresh questions, at {MIN_SAVING:.0%} or more less compute (gate cost included)"
    os.makedirs(args.out_dir, exist_ok=True)
    npz = os.path.join(args.out_dir, "frozen_gate.npz")
    np.savez_compressed(npz, **FZ, **{f"old_{k}": v for k, v in OLD.items()})
    meta["npz_sha256"] = sha256(npz)
    S.write_json(os.path.join(args.out_dir, "frozen_gate.json"), meta)
    log(f"primary gate: share {pick['share']:.0%}, threshold {pick['threshold']:.4f}; held-out on the old questions {100 * pick['accuracy']:.1f}% at "
        f"{pick['compute_x']:.2f}x vs the bar {100 * bar_acc:.1f}% at {bar_cost:.2f}x")
    log(meta["rule in words"])
    log(f"frozen in {time.time() - t0:.0f}s -> {args.out_dir} (checksum {meta['npz_sha256'][:16]}...)")


def load_frozen(d):
    meta = json.load(open(os.path.join(d, "frozen_gate.json"), encoding="utf-8"))
    npz = os.path.join(d, "frozen_gate.npz")
    got = sha256(npz)
    assert got == meta["npz_sha256"], f"the frozen gate file has changed since it was frozen (checksum {got[:16]} != {meta['npz_sha256'][:16]})"
    a = np.load(npz)
    return meta, {k: a[k] for k in a.files}


# ==================================================================================
# GPU stages on the fresh questions: the Set 4 code, restricted to what the test needs
# ==================================================================================
def _restrict():
    W.COTS, W.PICKS, W.RETRIEVERS = [COT], [PICK], [RET]


def cmd_retrieve(args):
    _restrict()
    args.model = RET
    W.cmd_retrieve(args)


def cmd_answer(args):
    _restrict()

    def needed(run, qs, cot):                       # only the 'keep all' shortlists, after 1 try and after 3 tries
        need = {q["qid"]: set() for q in qs}
        n_refs = {q["qid"]: len(q["ctxs_candidate"]) for q in qs}
        for row in S.read_jsonl(W.ret_path(run, W.vid(RET, cot, PICK))):
            if row["qid"] in need:
                sl = W.shortlists(row, cot, n_refs[row["qid"]])
                need[row["qid"]].update({sl[(1, MERGE)], sl[(3, MERGE)]})
        return {k: sorted(v) for k, v in need.items()}
    W.needed_shortlists = needed
    args.cot = COT
    qs = S.load_questions(args.data_dir)
    assert len(S.read_jsonl(W.ret_path(args.run_dir, W.vid(RET, COT, PICK)))) >= len(qs), "retrieval is not complete yet"
    W.cmd_answer(args)


# ==================================================================================
# Evaluate: apply the frozen gate, fit nothing
# ==================================================================================
def boot(fn, n, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)], float)
    v = v[~np.isnan(v)]
    return (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) if len(v) else (float("nan"), float("nan"))


def cmd_evaluate(args):
    import pandas as pd
    import routers as R
    t0 = time.time()
    run, out = args.run_dir, os.path.join(args.run_dir, "results")
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    meta, FZ = load_frozen(args.frozen_dir)
    log(f"evaluation code version {VERSION}; frozen gate {meta['version']} made {meta['created']} (checksum {meta['npz_sha256'][:16]}...)")
    P = {k: v["params"] for k, v in W.load_models(args.models_json).items()}
    qs = S.load_questions(args.data_dir)
    qids = [q["qid"] for q in qs]
    n = len(qs)
    ref = meta["ref_flops"]
    rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RET, COT, PICK)))}
    answ = {(r["qid"], tuple(r["sel"])): r for r in S.read_jsonl(W.ans_path(run, BIG, COT))}
    E = H.load_extract(run, CHEAP, "answer", qids)
    assert E is not None, "the Llama 8B extraction is missing"
    missing = [q["qid"] for q in qs if q["qid"] not in rets or not E["have"][qids.index(q["qid"])]]
    assert not missing, f"{len(missing)} questions are not finished yet - run the notebook again to continue"
    sc = scores_from(E, FZ, meta)
    em_c = np.array([r["EM"] for r in E["rows"]], float)
    f1_c = np.array([r["F1"] for r in E["rows"]], float)
    fl_c = 2 * P[CHEAP] * np.array([r["prompt_tokens"] + r["out_tokens"] for r in E["rows"]], float)
    A = {k: np.zeros(n) for k in ("em11", "f11", "fl11", "em15", "fl15", "em31", "fl31", "em35", "fl35", "fr1", "tail1", "found1", "found3", "np1")}
    for i, q in enumerate(qs):
        row = rets[q["qid"]]
        sl = W.shortlists(row, COT, len(q["ctxs_candidate"]))
        gold = set(q["ret_answers"])
        for rk in (1, 3):
            sel = sl[(rk, MERGE)]
            x = answ.get((q["qid"], sel))
            assert x is not None, f"the big answerer has not answered question {q['qid']} yet - run the notebook again to continue"
            fr = 2 * P[RET] * (row["prompt_tokens"] + sum(s["out_tokens"] for s in row["samples"][:rk]))
            A[f"found{rk}"][i] = float(gold <= {j - 1 for j in sel})
            for ak in (1, 5):
                fz = x["fused"][str(ak)]
                tail = 2 * P[BIG] * (x["prompt_tokens"] + sum(x["out_tokens"][:ak])) + 2 * P[BIG] * (fz["prompt_tokens"] + fz["out_tokens"])
                A[f"em{rk}{ak}"][i], A[f"fl{rk}{ak}"][i] = fz["EM"], fr + tail
                if (rk, ak) == (1, 1):
                    A["f11"][i], A["fr1"][i], A["tail1"][i], A["np1"][i] = fz["F1"], fr, tail, len(sel)
        assert tuple(E["rows"][i]["sel"]) == tuple(sl[(1, MERGE)]), "the Llama 8B did not answer on the same shortlist as the big answerer"
    big, f1_b = A["em11"], A["f11"]
    cheap_cost = A["fr1"] + fl_c
    y_route = (em_c >= big).astype(float)
    bar_acc, bar_cost = float(big.mean()), float(A["fl11"].mean() / ref)

    def policy(esc):
        esc = np.asarray(esc, bool)
        return np.where(esc, big, em_c), np.where(esc, f1_b, f1_c), np.where(esc, cheap_cost + A["tail1"], cheap_cost)

    def describe(name, em, f1, fl, esc=None, kind="gate"):
        d = float(em.mean() - bar_acc)
        lo, hi = boot(lambda ix: em[ix].mean() - big[ix].mean(), n)
        clo, chi = boot(lambda ix: fl[ix].mean() / A["fl11"][ix].mean() - 1, n, seed=1)
        cost = float(fl.mean() / ref)
        return {"Policy": name, "Kind": kind, "Share sent to the big answerer": float(np.mean(esc)) if esc is not None else None, "Accuracy": float(em.mean()),
                "F1": float(f1.mean()) if f1 is not None else None, "Compute (x cheapest)": cost,
                "Accuracy vs the cheap fixed setup (points)": 100 * d, "95% interval, low (points)": 100 * lo, "95% interval, high (points)": 100 * hi,
                "Compute vs the cheap fixed setup": cost / bar_cost - 1, "Compute 95% interval, low": clo, "Compute 95% interval, high": chi,
                "Right where the cheap fixed setup is wrong": int(((em == 1) & (big == 0)).sum()), "Wrong where the cheap fixed setup is right": int(((em == 0) & (big == 1)).sum()),
                "Within 2 points": bool(d >= -TOL_POINTS / 100 - 1e-9), "15% or more less compute": bool(cost <= (1 - MIN_SAVING) * bar_cost + 1e-9),
                "Passes the success test": bool(d >= -TOL_POINTS / 100 - 1e-9 and cost <= (1 - MIN_SAVING) * bar_cost + 1e-9)}

    pr = meta["primary"]
    esc_p = sc[pr["signal"]] <= pr["threshold"]
    pol = [describe("Cheap fixed setup: Qwen 7B retrieves once, Qwen 32B answers once (the bar)", big, f1_b, A["fl11"], np.ones(n), "fixed"),
           describe("PRIMARY frozen gate: Llama 8B answers once; least-sure questions go to Qwen 32B", *policy(esc_p), esc_p, "primary"),
           describe("Llama 8B answers once, never escalated", em_c, f1_c, cheap_cost, np.zeros(n), "fixed")]
    primary = pol[1]
    for k in ("Within 2 points", "15% or more less compute", "Passes the success test", "95% interval, low (points)", "95% interval, high (points)",
              "Compute 95% interval, low", "Compute 95% interval, high"):
        pol[0][k] = None                                                # the bar is not tested against itself
    # the whole frozen curve of every signal (every threshold was fixed in advance)
    curve = []
    for sig, label in SIGNALS.items():
        info = meta["signals"][sig]
        for c in info["held-out curve"]:
            esc = sc[sig] <= c["threshold"]
            em, f1, fl = policy(esc)
            d = describe(f"{label} | threshold frozen at the {c['share']:.0%} share", em, f1, fl, esc, "frozen curve")
            d.update({"Signal": label, "Planned share": c["share"], "Threshold": c["threshold"], "Old questions, held-out: accuracy": c["accuracy"],
                      "Old questions, held-out: compute (x cheapest)": c["compute_x"],
                      "Random gate with the same share: accuracy": float(esc.mean() * bar_acc + (1 - esc.mean()) * em_c.mean()),
                      "Is the primary gate": bool(sig == pr["signal"] and abs(c["share"] - pr["share"]) < 1e-9)})
            curve.append(d)
    for s in (0.30, 0.50):                                              # two more points of the primary signal, named in advance
        c = [r for r in curve if r["Signal"] == SIGNALS[pr["signal"]] and abs(r["Planned share"] - s) < 1e-9 and not r["Is the primary gate"]]
        if c:
            pol.append(dict(c[0], Policy=f"Same gate, threshold frozen at the {s:.0%} share", Kind="secondary"))
    for sig in SIGNALS:
        if sig != pr["signal"]:
            c = [r for r in curve if r["Signal"] == SIGNALS[sig] and abs(r["Planned share"] - pr["share"]) < 1e-9][0]
            pol.append(dict(c, Policy=f"Other frozen signal: {SIGNALS[sig]} ({pr['share']:.0%} share)", Kind="secondary"))
    s_p = float(esc_p.mean())
    rnd_acc = s_p * bar_acc + (1 - s_p) * float(em_c.mean())
    rnd_cost = float((cheap_cost.mean() + s_p * A["tail1"].mean()) / ref)
    pol.append({"Policy": "Control: random gate sending the same share", "Kind": "control", "Share sent to the big answerer": s_p, "Accuracy": rnd_acc,
                "Compute (x cheapest)": rnd_cost, "Accuracy vs the cheap fixed setup (points)": 100 * (rnd_acc - bar_acc), "Compute vs the cheap fixed setup": rnd_cost / bar_cost - 1})
    perfect = (big == 1) & (em_c == 0)
    pol.append(describe("Ceiling: perfect gate (escalates only where it helps; needs the answer key)", *policy(perfect), perfect, "ceiling"))
    for rk, ak, nm in ((1, 5, "Qwen 7B retrieves once, Qwen 32B answers 5 times + fusion"), (3, 1, "Qwen 7B retrieves 3 times, Qwen 32B answers once"),
                       (3, 5, "Qwen 7B retrieves 3 times, Qwen 32B answers 5 times + fusion")):
        pol.append(describe("Fixed: " + nm, A[f"em{rk}{ak}"], None, A[f"fl{rk}{ak}"], np.ones(n), "fixed"))

    # does each frozen signal still separate on the fresh questions?
    sig_rows = []
    for sig, label in SIGNALS.items():
        info = meta["signals"][sig]
        for tg, y, key in (("the cheap answer is no worse than the big one (routing target)", y_route, "held-out separation (routing target)"),
                           ("the cheap answer is right", em_c, "held-out separation (cheap answer is right)")):
            a = H.auc(y, sc[sig])
            lo, hi = boot(lambda ix: H.auc(y[ix], sc[sig][ix]), n, B=1000, seed=2)
            sig_rows.append({"Signal": label, "Layer": info.get("layer"), "What is predicted": tg, "Old questions (held-out)": info[key], "Fresh questions": a,
                             "Fresh: 95% interval, low": lo, "Fresh: 95% interval, high": hi,
                             "Share below the primary threshold, old": pr["share"] if sig == pr["signal"] else None,
                             "Share below the primary threshold, fresh": s_p if sig == pr["signal"] else None})
    # where the primary gate gained and lost
    groups = [("Qwen 32B right, Llama 8B wrong (escalating helps)", (big == 1) & (em_c == 0)), ("both right (escalating is wasted)", (big == 1) & (em_c == 1)),
              ("both wrong (escalating is wasted)", (big == 0) & (em_c == 0)), ("Llama 8B right, Qwen 32B wrong (escalating hurts)", (big == 0) & (em_c == 1))]
    where = [{"Kind of question": g, "Questions": int(m.sum()), "Share of all questions": float(m.mean()), "Sent to Qwen 32B by the gate": int((m & esc_p).sum()),
              "Kept with Llama 8B": int((m & ~esc_p).sum()), "Share of this kind sent to Qwen 32B": float(esc_p[m].mean()) if m.any() else None} for g, m in groups]
    base = [{"Measure": "Cheap fixed setup: accuracy", "Old 200 questions": meta["old"]["bar_accuracy"], "Fresh questions": bar_acc},
            {"Measure": "Cheap fixed setup: compute (x cheapest)", "Old 200 questions": meta["old"]["bar_compute_x"], "Fresh questions": bar_cost},
            {"Measure": "Llama 8B, one answer: accuracy", "Old 200 questions": meta["old"]["cheap_single_accuracy"], "Fresh questions": float(em_c.mean())},
            {"Measure": "Llama 8B, one answer: compute (x cheapest)", "Old 200 questions": meta["old"]["cheap_single_compute_x"], "Fresh questions": float(cheap_cost.mean() / ref)},
            {"Measure": "Share of questions where the Llama 8B answer is no worse", "Old 200 questions": meta["old"]["share where the cheap answer is no worse"], "Fresh questions": float(y_route.mean())},
            {"Measure": "Qwen 7B finds both passages in 1 try", "Old 200 questions": None, "Fresh questions": float(A["found1"].mean())},
            {"Measure": "Primary gate: share sent to Qwen 32B", "Old 200 questions": pr["share"], "Fresh questions": s_p},
            {"Measure": "Primary gate: accuracy", "Old 200 questions": pr["held-out accuracy (old questions)"], "Fresh questions": primary["Accuracy"]},
            {"Measure": "Primary gate: compute (x cheapest)", "Old 200 questions": pr["held-out compute_x (old questions)"], "Fresh questions": primary["Compute (x cheapest)"]},
            {"Measure": "Primary gate: accuracy vs the cheap fixed setup (points)", "Old 200 questions": 100 * (pr["held-out accuracy (old questions)"] - meta["old"]["bar_accuracy"]),
             "Fresh questions": primary["Accuracy vs the cheap fixed setup (points)"]},
            {"Measure": "Primary gate: compute vs the cheap fixed setup", "Old 200 questions": pr["held-out compute_x (old questions)"] / meta["old"]["bar_compute_x"] - 1,
             "Fresh questions": primary["Compute vs the cheap fixed setup"]}]
    perq = []
    for i, q in enumerate(qs):
        r = E["rows"][i]
        x = answ[(q["qid"], tuple(r["sel"]))]
        perq.append({"qid": q["qid"], "half": q["split"], "question": q["question"], "type": q.get("type"), "gold": q["answer"],
                     "passages in the shortlist": int(A["np1"][i]), "both gold passages found": bool(A["found1"][i]),
                     "Llama 8B answer": r["prediction"], "Llama 8B right": int(em_c[i]), "Llama 8B stated confidence": r["stated_conf"],
                     "Qwen 32B answer": x["fused"]["1"]["prediction"], "Qwen 32B right": int(big[i]),
                     **{f"score: {SIGNALS[s]}": None if sc[s][i] != sc[s][i] else round(float(sc[s][i]), 4) for s in SIGNALS},
                     "primary threshold": pr["threshold"], "sent to Qwen 32B by the primary gate": bool(esc_p[i]),
                     "primary gate right": int(np.where(esc_p[i], big[i], em_c[i])), "compute, gate (x cheapest)": float(policy(esc_p)[2][i] / ref),
                     "compute, cheap fixed setup (x cheapest)": float(A["fl11"][i] / ref)})
    from scipy.stats import binomtest
    b, c = primary["Right where the cheap fixed setup is wrong"], primary["Wrong where the cheap fixed setup is right"]
    p_mc = float(binomtest(b, b + c, 0.5).pvalue) if b + c else 1.0
    em_p = policy(esc_p)[0]
    p_pass = float(np.mean([em_p[ix].mean() - big[ix].mean() >= -TOL_POINTS / 100 - 1e-9 for ix in np.random.default_rng(3).integers(0, n, (4000, n))]))
    verdict = {"version": VERSION, "frozen gate checksum": meta["npz_sha256"], "questions": n, "passes the success test": primary["Passes the success test"],
               "primary": {k: primary[k] for k in primary if k not in ("Policy", "Kind")}, "bar": {"accuracy": bar_acc, "compute_x": bar_cost},
               "random gate with the same share": {"accuracy": rnd_acc, "compute_x": rnd_cost},
               "discordant pairs (gate right & bar wrong, gate wrong & bar right)": [b, c], "exact sign test p-value": p_mc,
               "share of bootstrap resamples in which the gate is within 2 points": p_pass, "seconds": round(time.time() - t0)}

    def dump(name, rr):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    dump("policies.jsonl", pol); dump("frozen_curve.jsonl", curve); dump("signals.jsonl", sig_rows); dump("where.jsonl", where)
    dump("old_vs_fresh.jsonl", base); dump("per_question.jsonl", perq)
    S.write_json(os.path.join(out, "verdict.json"), verdict)
    old = {k[4:]: v for k, v in FZ.items() if k.startswith("old_")}
    charts(os.path.join(out, "charts"), meta, pol, curve, sig_rows, where, sc, y_route, old, bar_acc, bar_cost, float(em_c.mean()), float(cheap_cost.mean() / ref))

    pct = {"Accuracy", "F1", "Share sent to the big answerer", "Compute vs the cheap fixed setup", "Compute 95% interval, low", "Compute 95% interval, high",
           "Share of all questions", "Share of this kind sent to Qwen 32B", "Old questions, held-out: accuracy", "Random gate with the same share: accuracy", "Planned share"}
    xs = {"Compute (x cheapest)", "Old questions, held-out: compute (x cheapest)"}
    nums = {"Accuracy vs the cheap fixed setup (points)", "95% interval, low (points)", "95% interval, high (points)", "Old questions (held-out)", "Fresh questions",
            "Fresh: 95% interval, low", "Fresh: 95% interval, high", "Threshold"}
    signed = {"Accuracy vs the cheap fixed setup (points)", "95% interval, low (points)", "95% interval, high (points)"}
    spct = {"Compute vs the cheap fixed setup", "Compute 95% interval, low", "Compute 95% interval, high"}

    def T(rr, cols):
        def f(r, c):
            v = r.get(c)
            if v is None or (isinstance(v, float) and v != v):
                return "-"
            if isinstance(v, (bool, np.bool_)):
                return "yes" if v else "no"
            if c in signed:
                return f"{v:+.1f}"
            if c in spct:
                return f"{100 * v:+.1f}%"
            if c in pct:
                return f"{100 * v:.1f}%"
            if c in xs:
                return f"{v:.2f}x"
            if c in nums:
                return f"{v:.2f}"
            return str(v)
        return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rr)

    def fmt_base(r):
        m = r["Measure"]
        f = (lambda v: "-" if v is None else (f"{v:.2f}x" if "(x cheapest)" in m else (f"{v:+.1f}" if "(points)" in m else (f"{100 * v:+.1f}%" if "compute vs" in m else f"{100 * v:.1f}%"))))
        return {"Measure": m, "Old 200 questions (held-out estimate)": f(r["Old 200 questions"]), "Fresh questions": f(r["Fresh questions"])}
    P_ = primary
    answer = ("**Yes.**" if P_["Passes the success test"] else "**No.**")
    why = (f"The frozen gate scored {100 * P_['Accuracy']:.1f}% at {P_['Compute (x cheapest)']:.2f}x; the cheap fixed setup scored {100 * bar_acc:.1f}% at {bar_cost:.2f}x. "
           f"That is {P_['Accuracy vs the cheap fixed setup (points)']:+.1f} points (95% interval {P_['95% interval, low (points)']:+.1f} to {P_['95% interval, high (points)']:+.1f}) "
           f"and {100 * P_['Compute vs the cheap fixed setup']:+.1f}% compute (95% interval {100 * P_['Compute 95% interval, low']:+.1f}% to {100 * P_['Compute 95% interval, high']:+.1f}%). "
           f"A random gate sending the same share would score about {100 * rnd_acc:.1f}%.")
    cols = ["Policy", "Share sent to the big answerer", "Accuracy", "Compute (x cheapest)", "Accuracy vs the cheap fixed setup (points)", "95% interval, low (points)",
            "95% interval, high (points)", "Compute vs the cheap fixed setup", "Passes the success test"]
    L = ["# Set 5c - final test of the frozen gate on fresh questions\n", f"Evaluation code version {VERSION}. Frozen gate version {meta['version']}, made {meta['created']} "
         f"on {meta['frozen_on']}; checksum `{meta['npz_sha256'][:16]}`. {n} fresh questions that no earlier set used. Nothing was fitted or chosen on them.\n",
         "## 1. Did the frozen gate pass the success test?\n",
         f"The test, fixed in advance: {meta['success test']}.\n", f"{answer} {why}\n",
         f"The gate, in words: {meta['rule in words']}\n",
         T([pol[0], pol[1]] + [r for r in pol if r["Kind"] == "control"] + [pol[2]] + [r for r in pol if r["Kind"] == "ceiling"], cols),
         f"\nPaired comparison with the cheap fixed setup: the gate is right on {b} questions where the cheap fixed setup is wrong, and wrong on {c} where it is right "
         f"(exact sign test p = {p_mc:.2f}). In {100 * p_pass:.0f}% of bootstrap resamples the gate is within 2 points. "
         f"With {n} questions the interval on the accuracy difference is several points wide, so a pass means 'no sign of a loss', not 'proven equal'.\n",
         "## 2. Old questions versus fresh questions\n", "The old-question numbers are the held-out estimates the gate was chosen on.\n",
         R.md_table([fmt_base(r) for r in base], ["Measure", "Old 200 questions (held-out estimate)", "Fresh questions"]),
         "\n## 3. Do the frozen signals still separate on fresh questions? (0.5 = useless)\n",
         T(sig_rows, ["Signal", "Layer", "What is predicted", "Old questions (held-out)", "Fresh questions", "Fresh: 95% interval, low", "Fresh: 95% interval, high"]),
         "\n## 4. Where the primary gate gained and lost\n",
         T(where, ["Kind of question", "Questions", "Share of all questions", "Sent to Qwen 32B by the gate", "Kept with Llama 8B", "Share of this kind sent to Qwen 32B"]),
         "\n## 5. Secondary results (named in advance; they do not decide the test)\n", T([r for r in pol if r["Kind"] == "secondary"], cols),
         "\n## 6. Fixed setups on the fresh questions, for reference\n", T([r for r in pol if r["Kind"] == "fixed"], cols[:-1]),
         "\n## 7. The whole frozen curve of the primary signal\n", "Every threshold was fixed on the old questions. 'Planned share' is the share it sent on the old questions.\n",
         T([r for r in curve if r["Signal"] == SIGNALS[pr["signal"]]],
           ["Planned share", "Threshold", "Share sent to the big answerer", "Accuracy", "Compute (x cheapest)", "Random gate with the same share: accuracy",
            "Old questions, held-out: accuracy", "Old questions, held-out: compute (x cheapest)", "Passes the success test"]),
         "\nCharts: `results/charts/`. Excel: `results/set5c_final_test.xlsx`. Every question: `results/per_question.jsonl`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    def xl(rr):
        d = pd.DataFrame(rr)
        for col in d.columns:
            if d[col].isna().all():
                d[col] = ""
            elif d[col].map(lambda v: isinstance(v, (list, dict))).any():
                d[col] = d[col].astype(str)
        return d
    fr = pd.DataFrame([{"Item": k, "Value": json.dumps(v, ensure_ascii=False) if isinstance(v, (dict, list)) else v} for k, v in meta.items() if k != "signals"]
                      + [{"Item": f"signal: {SIGNALS[s]}", "Value": json.dumps({k: v for k, v in i.items() if k != "held-out curve"}, ensure_ascii=False)} for s, i in meta["signals"].items()])
    W.write_excel(os.path.join(out, "set5c_final_test.xlsx"), {
        "Policies": xl(pol), "Old vs fresh": xl(base), "Signals": xl(sig_rows), "Where gained and lost": xl(where), "Frozen curve": xl(curve),
        "Per question": xl(perq), "Frozen gate": fr.astype(str)})
    log(("PASS" if P_["Passes the success test"] else "FAIL") + " - " + why)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def charts(out, meta, pol, curve, sig_rows, where, sc, y_route, old, bar_acc, bar_cost, cheap_acc, cheap_costx):
    plt = S._plt()
    os.makedirs(out, exist_ok=True)
    pr = meta["primary"]
    C = S.SERIES
    # 1. accuracy against compute: old (held-out) and fresh, side by side
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    panels = [("Old 200 questions (held-out estimate, used to choose the gate)", meta["old"]["bar_accuracy"], meta["old"]["bar_compute_x"],
               meta["old"]["cheap_single_accuracy"], meta["old"]["cheap_single_compute_x"], None),
              ("Fresh questions (the test; every threshold frozen beforehand)", bar_acc, bar_cost, cheap_acc, cheap_costx, curve)]
    for ax, (title, ba, bc, ca, cc, cv) in zip(axs, panels):
        ax.axvspan(0, (1 - MIN_SAVING) * bc, ymin=0, ymax=1, color=C[2], alpha=0.05)
        ax.axhline(100 * ba - TOL_POINTS, color=C[2], ls=":", lw=1)
        ax.axvline((1 - MIN_SAVING) * bc, color=C[2], ls=":", lw=1)
        ax.plot([cc, bc], [100 * ca, 100 * ba], color=S.MUTED, ls="--", lw=1.2, label="random gate")
        for k, (sig, label) in enumerate(SIGNALS.items()):
            if cv is None:
                h = meta["signals"][sig]["held-out curve"]
                xs_, ys_ = [r["compute_x"] for r in h], [100 * r["accuracy"] for r in h]
            else:
                h = [r for r in cv if r["Signal"] == label]
                xs_, ys_ = [r["Compute (x cheapest)"] for r in h], [100 * r["Accuracy"] for r in h]
            ax.plot(xs_, ys_, color=C[k], lw=2.0 if sig == pr["signal"] else 1.0, alpha=1.0 if sig == pr["signal"] else 0.55, marker="o", ms=3, label=label)
        if cv is None:
            px, py = pr["held-out compute_x (old questions)"], 100 * pr["held-out accuracy (old questions)"]
        else:
            px, py = pol[1]["Compute (x cheapest)"], 100 * pol[1]["Accuracy"]
        ax.scatter([px], [py], marker="*", s=320, color=C[0], edgecolor=S.INK, zorder=6, label="primary gate")
        ax.scatter([bc], [100 * ba], marker="s", s=90, color=S.INK, zorder=6, label="cheap fixed setup (the bar)")
        ax.scatter([cc], [100 * ca], marker="D", s=70, color=S.MUTED, zorder=6, label="Llama 8B alone, one answer")
        ax.annotate(f"primary gate\n{py:.1f}% at {px:.2f}x", (px, py), textcoords="offset points", xytext=(-14, 8), ha="right", va="bottom", fontsize=9)
        ax.annotate(f"the bar\n{100 * ba:.1f}% at {bc:.2f}x", (bc, 100 * ba), textcoords="offset points", xytext=(8, -8), va="top", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("compute (x cheapest setup)")
        ax.set_xlim(min(cc, px) - 0.35, max(bc, max(xs_)) + 0.75)
        ax.margins(y=0.12)
    axs[0].set_ylabel("accuracy (%)")
    hd, lb = axs[1].get_legend_handles_labels()
    fig.legend(hd, lb, frameon=False, fontsize=8, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.09))
    fig.suptitle("Final test: frozen gate against the cheap fixed setup (shaded: 15% or more less compute; dotted line: 2 points below the bar)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_accuracy_vs_compute.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 2. the primary score, old and fresh, split by whether escalating was needed
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2), sharey=False)
    bins = np.linspace(0, 1, 21)
    for ax, (title, p, y) in zip(axs, (("Old questions (held-out scores)", old.get(f"oof_{pr['signal']}"), old.get("y_route")), ("Fresh questions", sc[pr["signal"]], y_route))):
        if p is None:
            ax.axis("off")
            continue
        ax.hist([p[y == 1], p[y == 0]], bins=bins, stacked=True, color=[C[0], C[1]],
                label=["Llama 8B answer is no worse (keep it)", "Qwen 32B is needed (escalate)"])
        ax.axvline(pr["threshold"], color=S.INK, lw=1.4, label="frozen threshold (scores to its left go to Qwen 32B)")
        ax.set_title(f"{title}: {100 * float(np.mean(p <= pr['threshold'])):.0f}% below the threshold", fontsize=10)
        ax.set_xlabel("semantic uncertainty score (probability that 5 answers would agree)" if pr["signal"] == "su" else "score")
        ax.set_ylabel("questions")
    hd, lb = axs[1].get_legend_handles_labels()
    fig.legend(hd, lb, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07))
    fig.suptitle("The primary signal: does the frozen threshold cut the same way on fresh questions?", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2_score_distribution.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 3. separation, old against fresh
    g = [r for r in sig_rows if "routing target" in r["What is predicted"]]
    fig, ax = plt.subplots(figsize=(8.5, 0.8 * len(g) + 1.8))
    yy = np.arange(len(g))[::-1]
    ax.barh(yy + 0.18, [r["Old questions (held-out)"] for r in g], height=0.34, color=S.MUTED, label="old questions (held-out)")
    ax.barh(yy - 0.18, [r["Fresh questions"] for r in g], height=0.34, color=C[0], label="fresh questions",
            xerr=[[max(0, r["Fresh questions"] - r["Fresh: 95% interval, low"]) for r in g], [max(0, r["Fresh: 95% interval, high"] - r["Fresh questions"]) for r in g]],
            error_kw={"ecolor": S.INK2, "lw": 1})
    for y_, r in zip(yy, g):
        ax.text(0.305, y_ + 0.18, f"{r['Old questions (held-out)']:.2f}", va="center", fontsize=8, color="white")
        ax.text(0.305, y_ - 0.18, f"{r['Fresh questions']:.2f}", va="center", fontsize=8, color="white")
    ax.axvline(0.5, color=S.INK2, ls=":", lw=1)
    ax.set_yticks(yy)
    ax.set_yticklabels([r["Signal"] for r in g], fontsize=8)
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("separation of 'the cheap answer is no worse than the big one' (0.5 = useless)")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Do the frozen signals still separate on fresh questions?", fontsize=10)
    fig.savefig(os.path.join(out, "3_separation_old_vs_fresh.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    # 4. where the gate gained and lost
    fig, ax = plt.subplots(figsize=(9, 3.6))
    yy = np.arange(len(where))[::-1]
    ax.barh(yy, [w["Sent to Qwen 32B by the gate"] for w in where], color=C[1], label="sent to Qwen 32B")
    ax.barh(yy, [w["Kept with Llama 8B"] for w in where], left=[w["Sent to Qwen 32B by the gate"] for w in where], color=C[0], label="kept with Llama 8B")
    for y_, w in zip(yy, where):
        ax.text(w["Questions"] + 1, y_, f"{w['Sent to Qwen 32B by the gate']} sent / {w['Kept with Llama 8B']} kept", va="center", fontsize=8)
    ax.set_yticks(yy)
    ax.set_yticklabels([w["Kind of question"] for w in where], fontsize=8)
    ax.set_xlabel("fresh questions")
    ax.set_xlim(0, max(w["Questions"] for w in where) * 1.35 + 5)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Where the primary gate sent questions (a perfect gate sends only the first row)", fontsize=10)
    fig.savefig(os.path.join(out, "4_where_gained_and_lost.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build_fresh")
    p.add_argument("--hotpot_dev", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--exclude", nargs="+", required=True)
    p.add_argument("--n", type=int, default=200)
    p.add_argument("--n_candidates", type=int, default=20)
    p.add_argument("--seed", type=int, default=43)
    p = sub.add_parser("freeze")
    p.add_argument("--sweep_run", required=True)
    p.add_argument("--hidden_run", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--models_json", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--repeats", type=int, default=3)
    for name in ("retrieve", "answer"):
        p = sub.add_parser(name)
        p.add_argument("--run_dir", required=True)
        p.add_argument("--data_dir", required=True)
        p.add_argument("--models_json", required=True)
        p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
        p.add_argument("--time_limit_min", type=float, default=None)
        p.add_argument("--max_model_len", type=int, default=8192)
        p.add_argument("--gpu_util", type=float, default=0.88)
        p.add_argument("--chunk", type=int, default=10)
        if name == "answer":
            p.add_argument("--model", default=BIG)
    p = sub.add_parser("evaluate")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--models_json", required=True)
    p.add_argument("--frozen_dir", required=True)
    args = ap.parse_args()
    if getattr(args, "run_dir", None):
        os.makedirs(os.path.join(args.run_dir, "stages"), exist_ok=True)
    {"build_fresh": cmd_build_fresh, "freeze": cmd_freeze, "retrieve": cmd_retrieve, "answer": cmd_answer, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
