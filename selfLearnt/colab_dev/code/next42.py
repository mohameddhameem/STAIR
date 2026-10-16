"""Set 6 - the experiments of '4_2_Next_Routing_Experiments_Plan.docx'.

CPU part (this file, command `cpu`), on the existing 400 questions:
  item 6  Check what the experts add: ablation of the MoE / BoE inputs against the fixed probe
  item 5  Stable escalation benefit: repeated answers -> per-question success rates -> a regression probe for the benefit
  item 4  Evidence support (cheap variant): does the cheap answer appear in the selected evidence?
GPU part (command `generate`, then `actions`): items 1-3, see the bottom of this file.

Scoring: exact match, token F1, and the notebook 20 corrected label ('substantially right'), which is a secondary correctness proxy.
References everywhere: the cheap fixed setup (Qwen 7B retrieves once, Qwen 32B answers once), the fixed probe gate, Llama 8B alone,
and a random policy with the same action mix. Intervals resample questions, never rows.
"""
import argparse
import json
import os
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
import experts as X           # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "6.4"
log = H.log
FIXED = E.FIXED


def z(a, ref):
    return (a - np.mean(ref)) / (np.std(ref) + 1e-9)


def lr_maker(C=1.0):
    from sklearn.linear_model import LogisticRegression
    return lambda Xa, ya: (lambda m: (lambda Zt: m.predict_proba(Zt)[:, 1]))(LogisticRegression(C=C, max_iter=500).fit(Xa, ya.astype(int)))


def support_features(P, tr):
    """Item 4, cheap variant: does the Llama 8B answer appear in the passages it was given? (absence is a warning, not proof)"""
    rows = []
    for i in range(P.n):
        ans = P.l8_pred[i]
        a_norm = S.normalize_answer(ans)
        at = set(X.toks(ans))
        sel = P.sel1[i]
        txt = " ".join(P.passages[i][j] for j in sel)
        tn = " " + S.normalize_answer(txt) + " "
        tt = set(X.toks(txt))
        yesno = float(a_norm in ("yes", "no"))
        rows.append([float(bool(a_norm) and (" " + a_norm + " ") in tn), len(at & tt) / max(1, len(at)), float(len(at)), yesno,
                     float(any(c.isdigit() for c in ans)), float(any(c.isdigit() for c in ans) and any(w in tt for w in at if any(c.isdigit() for c in w)))])
    return np.array(rows)


def ablation_pipeline(P, tr, te, decision, reps=2):
    """Item 6: the same experts as notebook 20, added to the fixed probe one at a time; regularised linear combiner (primary) and equal weights."""
    y = P.y[decision]
    ytr = y[tr].astype(float)
    Ft = X.Feats(P, tr)
    ens, base = {}, []

    def reader(Xm, fn=H.f_linear):
        po = np.nanmean([H.cv(fn, Xm[tr], ytr, 10 + r) for r in range(reps)], 0)
        pn = fn(Xm[tr], ytr.astype(int), Xm[te])
        return np.nan_to_num(po, nan=ytr.mean()), np.nan_to_num(pn, nan=ytr.mean())

    def add(name, po, pn):
        ens[name] = (np.asarray(po, float), np.asarray(pn, float))
        base.append({"Signal family": name, "Group": "ablation", "g": "Q", "Reader": "-", "train": E.fauc(ytr, po), "test": E.fauc(y[te], pn)})
    if decision == "answerer":
        li = P.ha["h_answer_line"].shape[1] - 2
        Xfix = F.apply_reduce(F.fit_reduce(P.ha["h_answer_line"][tr, li, :]), P.ha["h_answer_line"][:, li, :])
    else:
        li = P.hr.shape[1] // 2
        Xfix = F.apply_reduce(F.fit_reduce(P.hr[tr, li, :]), P.hr[:, li, :])
    probe = reader(Xfix)
    add(FIXED, *probe)
    ens[E.REFERENCE] = ens[FIXED]
    parts = {"evidence": reader(np.column_stack([Ft.voters_table, Ft.rarity])), "structure": reader(Ft.structure), "facts about the first retrieval try": reader(P.meta)}
    if decision == "answerer":
        parts["retriever hidden state"] = reader(F.apply_reduce(F.fit_reduce(P.hr[tr, P.hr.shape[1] // 2, :]), P.hr[:, P.hr.shape[1] // 2, :]))
        parts["token probabilities"] = reader(P.tok)
        parts["support of the answer in the evidence"] = reader(support_features(P, tr))
    for k, v in parts.items():
        add("Expert alone: " + k, *v)
    combos = [("evidence",), ("structure",), ("evidence", "structure"), ("facts about the first retrieval try",)]
    if decision == "answerer":
        combos += [("token probabilities",), ("support of the answer in the evidence",), ("retriever hidden state",)]
    full = tuple(parts)
    combos += [tuple(k for k in full if k in ("evidence", "structure", "facts about the first retrieval try")), full]
    for c in combos:
        cols = [probe] + [parts[k] for k in c]
        Zo = np.column_stack([z(a, a) for a, _ in cols])
        Zn = np.column_stack([z(b, a) for a, b in cols])
        nm = "Probe + " + " + ".join(c)
        add(nm + " (linear combiner)", E.cvp(lr_maker(1.0), Zo, ytr), lr_maker(1.0)(Zo, ytr)(Zn))
        add(nm + " (equal weights)", Zo.mean(1), Zn.mean(1))
    # the notebook 20 MoE / BoE inputs, rebuilt so that their errors can be compared
    ev, st = parts["evidence"], parts["structure"]
    subs = [parts["facts about the first retrieval try"], probe] if decision == "retriever" else [parts["facts about the first retrieval try"], parts["retriever hidden state"], probe, parts["token probabilities"]]
    co = (np.mean([z(a, a) for a, _ in subs], 0), np.mean([z(b, a) for a, b in subs], 0))
    Zo = np.column_stack([z(ev[0], ev[0]), z(st[0], st[0]), z(co[0], co[0])])
    Zn = np.column_stack([z(ev[1], ev[0]), z(st[1], st[0]), z(co[1], co[0])])
    add("Notebook 20 MoE, equal weights (evidence + structure + confidence)", Zo.mean(1), Zn.mean(1))
    share = E.SHARE[decision]
    bo = -(Zo <= np.quantile(Zo, share, 0)).sum(1) + Zo.mean(1) / 10
    bn = -(Zn <= np.quantile(Zo, share, 0)).sum(1) + Zn.mean(1) / 10
    add("Notebook 20 BoE, every expert votes", bo, bn)
    add("Notebook 20 MoE without the confidence expert (evidence + structure only)", Zo[:, :2].mean(1), Zn[:, :2].mean(1))
    add("MoE + BoE averaged", (z(Zo.mean(1), Zo.mean(1)) + z(bo, bo)) / 2, (z(Zn.mean(1), Zo.mean(1)) + z(bn, bo)) / 2)
    corr = np.corrcoef(np.array([ens[b["Signal family"]][1] for b in base]))
    return base, ens, {}, corr


def cmd_cpu(args):
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    t0 = time.time()
    out = os.path.join(args.run_dir, "results_cpu")
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    log(f"next-experiments code version {VERSION}")
    old = X.load_set("old", args.old_sweep, args.old_data, args.old_hidden)
    new = X.load_set("fresh", args.fresh_run, args.fresh_data, args.fresh_run)
    sets = [old, new]
    P0 = E.Pool(sets, use_embedding=not args.no_embedding)
    P0.sel1 = [x for s in sets for x in s["sel1"]]
    P0.l8_pred = [r["Llama 8B answer"] for s in sets for r in s["label_rows"]]
    io, inew = np.where(P0.origin == "old")[0], np.where(P0.origin == "fresh")[0]
    P = X.relabel(P0, sets, "lenient")
    # ---------------------------------------------------------------- item 6 (and item 4, cheap variant, inside it for the answerer)
    E.pipeline = ablation_pipeline
    rows = []
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
        rows += [r for r in r0 if r["Ensemble"] != E.REFERENCE]
        log(f"item 6 | {decision} done ({time.time() - t0:.0f}s)")
    # paired errors of MoE and BoE (cross-validated predictions are not stored per question by evaluate, so use old -> fresh and fresh -> old)
    # ---------------------------------------------------------------- item 5: stable escalation benefit
    st_rows, st_policy = stable_benefit(args, P, old, new, io, inew)
    log(f"item 5 done ({time.time() - t0:.0f}s)")

    def dump(nm, rr):
        with open(os.path.join(out, nm), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    dump("item6_ablation.jsonl", rows); dump("item5_stable_benefit.jsonl", st_rows); dump("item5_policies.jsonl", st_policy)
    summary_cpu(out, rows, st_rows, st_policy, P0)
    W.write_excel(os.path.join(out, "set6_cpu.xlsx"), {"Item 6 ablation": pd.DataFrame(rows), "Item 5 label stability": pd.DataFrame(st_rows),
                                                        "Item 5 policies": pd.DataFrame(st_policy)})
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def draft_rates(run, qs, sel1, model):
    """Per-question share of the 5 sampled Set 4 answers that are substantially right (same shortlist, same prompt)."""
    ans = {(r["qid"], tuple(r["sel"])): r for r in S.read_jsonl(W.ans_path(run, model, RC.COT))}
    rates, em_rates = [], []
    for q, s in zip(qs, sel1):
        x = ans.get((q["qid"], tuple(j + 1 for j in s)))
        if x is None:
            rates.append(np.nan); em_rates.append(np.nan)
            continue
        rates.append(float(np.mean([X.lenient(a, q["answer"]) for a in x["draft_answers"]])))
        em_rates.append(float(np.mean([S.em_f1(a, q["answer"])[0] for a in x["draft_answers"]])))
    return np.array(rates), np.array(em_rates)


def stable_benefit(args, P, old, new, io, inew):
    """Item 5. Old questions: 5 sampled answers from each model on the same shortlist -> benefit = Qwen 32B rate - Llama 8B rate.
    Fresh questions: Qwen 32B has 5 samples, the Llama 8B only its one greedy answer, so they are used for testing only."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold
    pb_o, _ = draft_rates(args.old_sweep, old["qs"], old["sel1"], "q32")
    pl_o, _ = draft_rates(args.old_sweep, old["qs"], old["sel1"], "l8")
    pb_n, _ = draft_rates(args.fresh_run, new["qs"], new["sel1"], "q32")
    ben = pb_o - pl_o
    stab = []
    for nm, v in (("Qwen 32B", pb_o), ("Llama 8B", pl_o)):
        stab.append({"Model (old questions, 5 sampled answers each)": nm, "All 5 right": float(np.mean(v == 1)), "All 5 wrong": float(np.mean(v == 0)),
                     "Mixed": float(np.mean((v > 0) & (v < 1))), "Mean success rate": float(np.nanmean(v))})
    stab.append({"Model (old questions, 5 sampled answers each)": "Qwen 32B (fresh questions)", "All 5 right": float(np.mean(pb_n == 1)), "All 5 wrong": float(np.mean(pb_n == 0)),
                 "Mixed": float(np.mean((pb_n > 0) & (pb_n < 1))), "Mean success rate": float(np.nanmean(pb_n))})
    one_pair = P.y["answerer"][io]                                       # 1 = the deployed Llama answer is no worse than the deployed Qwen answer
    stab.append({"Model (old questions, 5 sampled answers each)": "Benefit = Qwen rate - Llama rate", "All 5 right": float(np.mean(ben >= 0.6)),
                 "All 5 wrong": float(np.mean(ben <= -0.6)), "Mixed": float(np.mean(np.abs(ben) < 0.6)), "Mean success rate": float(np.mean(ben)),
                 "Stable helps (benefit >= 0.6)": int((ben >= 0.6).sum()), "Stable hurts (benefit <= -0.6)": int((ben <= -0.6).sum()),
                 "One-pair 'helps' labels that are stable helps": float(np.mean(ben[one_pair == 0] >= 0.6)) if (one_pair == 0).any() else None,
                 "One-pair 'helps' labels where the benefit is at most 0": float(np.mean(ben[one_pair == 0] <= 0)) if (one_pair == 0).any() else None})
    li = P.ha["h_answer_line"].shape[1] - 2
    Hh = P.ha["h_answer_line"][:, li, :]
    y_bin = P.y["answerer"]
    big, cheap = P.A["e1"], P.l8
    pol = []

    def gate_eval(name, score, idx, thr_scores, share=0.40):
        thr = np.quantile(thr_scores, share)
        esc = score <= thr
        got = np.where(esc, big[idx], cheap[idx])
        s = esc.mean()
        rnd = s * big[idx].mean() + (1 - s) * cheap[idx].mean()
        rng = np.random.default_rng(0)
        d_r, d_b = [], []
        n = len(idx)
        for _ in range(2000):
            w = rng.integers(0, n, n)
            ss = esc[w].mean()
            d_r.append(got[w].mean() - (ss * big[idx][w].mean() + (1 - ss) * cheap[idx][w].mean()))
            d_b.append(got[w].mean() - big[idx][w].mean())
        return {"Policy": name, "Share sent to Qwen 32B": float(s), "Accuracy": float(got.mean()), "Cheap fixed setup": float(big[idx].mean()),
                "vs cheap fixed setup (points)": 100 * float(got.mean() - big[idx].mean()), "low": 100 * float(np.percentile(d_b, 2.5)), "high": 100 * float(np.percentile(d_b, 97.5)),
                "vs random gate (points)": 100 * float(got.mean() - rnd), "random: low": 100 * float(np.percentile(d_r, 2.5)), "random: high": 100 * float(np.percentile(d_r, 97.5)),
                "Separation of the one-pair routing target": E.fauc(y_bin[idx], score), "Separation of the stable benefit (old only)": (E.fauc((ben < 0.2).astype(float), score[:len(io)]) if idx is io else None)}
    # (a) old questions, cross-validated: binary target vs benefit regression, same input
    for rep in range(1):
        pb_cv, pr_cv = np.zeros(len(io)), np.zeros(len(io))
        for a, b in KFold(5, shuffle=True, random_state=7).split(io):
            ta, tb = io[a], io[b]
            red = F.fit_reduce(Hh[ta])
            Xa, Xb = F.apply_reduce(red, Hh[ta]), F.apply_reduce(red, Hh[tb])
            pb_cv[b] = H.f_linear(Xa, y_bin[ta].astype(int), Xb)
            mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
            pr_cv[b] = -Ridge(alpha=10.0).fit((Xa - mu) / sd, ben[a]).predict((Xb - mu) / sd)     # low score = escalate
    pol.append(dict(gate_eval("Old questions, cross-validated: probe trained on the one-pair routing target (as before)", pb_cv, io, pb_cv), Evaluation="old, cross-validated"))
    pol.append(dict(gate_eval("Old questions, cross-validated: regression probe for the stable benefit (item 5 primary)", pr_cv, io, pr_cv), Evaluation="old, cross-validated"))
    # (b) trained on all old questions, tested on the fresh questions (deployed outcomes: one Llama greedy answer, Qwen fused answer)
    red = F.fit_reduce(Hh[io])
    Xo, Xn = F.apply_reduce(red, Hh[io]), F.apply_reduce(red, Hh[inew])
    s_bin_n = H.f_linear(Xo, y_bin[io].astype(int), Xn)
    mu, sd = Xo.mean(0), Xo.std(0) + 1e-9
    rg = Ridge(alpha=10.0).fit((Xo - mu) / sd, ben)
    s_reg_n = -rg.predict((Xn - mu) / sd)
    s_reg_o = -rg.predict((Xo - mu) / sd)
    pol.append(dict(gate_eval("Old -> fresh: probe trained on the one-pair routing target (as before)", s_bin_n, inew, pb_cv), Evaluation="old -> fresh"))
    pol.append(dict(gate_eval("Old -> fresh: regression probe for the stable benefit (item 5 primary)", s_reg_n, inew, pr_cv), Evaluation="old -> fresh"))
    return stab, pol


def summary_cpu(out, rows, st_rows, st_policy, P0):
    def T(rr, cols, pct=(), signed=(), sep=()):
        def f(r, c):
            v = r.get(c)
            if v is None or (isinstance(v, float) and v != v):
                return "-"
            if isinstance(v, str):
                return v
            if isinstance(v, (int, np.integer)):
                return str(v)
            if c in sep:
                return f"{v:+.2f}"
            if c in signed:
                return f"{v:+.1f}"
            if c in pct:
                return f"{100 * v:.1f}%"
            return f"{v:.2f}"
        return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rr)
    L = ["# Set 6, CPU part - items 6, 5 and 4 (cheap variant) of the next-experiments plan\n",
         f"Code version {VERSION}. 400 questions. Scoring: the notebook 20 corrected label ('substantially right') as a secondary proxy; the retriever target "
         "('first try finds both passages') does not depend on it. Both question sets are development data.\n",
         "## Item 6 - what the experts add to the fixed probe\n",
         "Primary: a regularised linear combiner trained on out-of-fold expert scores; equal weights as a control. Separation: 0.5 = useless.\n"]
    for decision, key in (("answerer", "accuracy"), ("retriever", "both passages found")):
        for ev in ("all 400, cross-validated", "old -> fresh", "fresh -> old"):
            rr = sorted([r for r in rows if r["Decision"] == decision and r["Evaluation"] == ev], key=lambda r: (r["Ensemble"] != FIXED, -r["Separation"]))
            L += [f"**{decision} decision, {ev}**\n",
                  T(rr, ["Ensemble", "Separation", "Minus the fixed probe", "Minus the fixed probe: low", "Minus the fixed probe: high", f"Gate minus random: {key} (points)",
                         f"Gate minus random: {key}, low", f"Gate minus random: {key}, high", "Gate accuracy minus the cheap fixed setup (points)", "Gate compute vs the cheap fixed setup"],
                    signed={f"Gate minus random: {key} (points)", f"Gate minus random: {key}, low", f"Gate minus random: {key}, high", "Gate accuracy minus the cheap fixed setup (points)"},
                    sep={"Minus the fixed probe", "Minus the fixed probe: low", "Minus the fixed probe: high"}, pct={"Gate compute vs the cheap fixed setup"}), ""]
    L += ["## Item 5 - stable escalation benefit\n", "Label stability from the 5 sampled Set 4 answers per model on the same shortlist (old questions):\n",
          T(st_rows, list(dict.fromkeys(k for r in st_rows for k in r)), pct={"All 5 right", "All 5 wrong", "Mixed", "Mean success rate", "One-pair 'helps' labels that are stable helps",
                                                                            "One-pair 'helps' labels where the benefit is at most 0"}),
          "\nThe probe input is unchanged (the greedy Llama 8B answer's hidden state). Only the training target changes.\n",
          T(st_policy, ["Policy", "Share sent to Qwen 32B", "Accuracy", "Cheap fixed setup", "vs cheap fixed setup (points)", "low", "high", "vs random gate (points)", "random: low",
                        "random: high", "Separation of the one-pair routing target"],
            pct={"Share sent to Qwen 32B", "Accuracy", "Cheap fixed setup"}, signed={"vs cheap fixed setup (points)", "low", "high", "vs random gate (points)", "random: low", "random: high"})]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


# ==================================================================================
# GPU part: items 1-3 need new answers (command `generate`)
# ==================================================================================
REVISE = ("\nYour earlier answer to this query, written with fewer references, was: {prev}\n"
          "Check that earlier answer against the references above. If they support it, keep it; if not, correct it. "
          "Think briefly, then end with one final line in exactly this form: 'Answer: <short answer>' - the answer only "
          "(a name, date, number or short phrase; 'yes' or 'no' for yes/no questions).")
LAMBDA_COVER, MU_REDUNDANT = 0.5, 0.3          # fixed in advance, not tuned


def rerank(q, cand, k):
    """Bounded shortlist from the retry union: greedy, relevance to the question + new coverage of the question's words - redundancy.
    Unsupervised and per question (TF-IDF fitted on that question's own 20 candidates), so nothing is learned from labels."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    if len(cand) <= k:
        return tuple(sorted(cand))
    ps = q["ctxs_candidate"]
    vec = TfidfVectorizer(sublinear_tf=True, stop_words="english").fit(ps + [q["question"]])
    rel = (vec.transform([ps[j - 1] for j in cand]) @ vec.transform([q["question"]]).T).toarray().ravel()
    Q = set(X.toks(q["question"]))
    T = [set(X.toks(ps[j - 1])) for j in cand]
    chosen, covered = [], set()
    while len(chosen) < k:
        best, bi = -1e9, None
        for i, j in enumerate(cand):
            if j in chosen:
                continue
            gain = len((T[i] & Q) - covered) / max(1, len(Q))
            red = max([len(T[i] & T[cand.index(c)]) / max(1, len(T[i] | T[cand.index(c)])) for c in chosen] or [0.0])
            sc = rel[i] + LAMBDA_COVER * gain - MU_REDUNDANT * red
            if sc > best:
                best, bi = sc, j
        chosen.append(bi)
        covered |= T[cand.index(bi)] & Q
    return tuple(sorted(chosen))


def contexts(q, row):
    sl = W.shortlists(row, RC.COT, len(q["ctxs_candidate"]))
    c0, cu = sl[(1, RC.MERGE)], sl[(3, RC.MERGE)]
    out = {"initial shortlist": c0, "retry union": cu, "gold passages only (diagnostic)": tuple(sorted(i + 1 for i in q["ret_answers"]))}
    for k in (2, 3, 4):
        out[f"retry, reranked to {k}"] = rerank(q, list(cu), k)
    return out


def arm_path(run, model):
    return os.path.join(run, "step", f"arms_{model}.jsonl")


def cmd_generate(args):
    models = W.load_models(args.models_json)
    os.makedirs(os.path.join(args.run_dir, "step"), exist_ok=True)
    sets = [("old", args.old_sweep, args.old_data), ("fresh", args.fresh_run, args.fresh_data)]
    items = []
    for name, run, dd in sets:
        qs = S.load_questions(dd)
        rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RC.RET, RC.COT, RC.PICK)))}
        for q in qs:
            items.append((name, q, contexts(q, rets[q["qid"]])))
    if args.limit:
        items = items[:args.limit] + items[200:200 + args.limit]
    path = arm_path(args.run_dir, args.model)
    done = {(r["qid"], tuple(r["sel"]), r["kind"]) for r in S.read_jsonl(path)}
    prev = {}
    if args.model == "l8":
        for r in S.read_jsonl(path):
            if r["kind"] == "fresh" and r["context"] == "initial shortlist":
                prev[r["qid"]] = r["prediction"]
    t0 = time.time()
    kinds = ["fresh", "revise"] if args.model == "l8" else ["fresh"]

    def pending(kind):
        jobs, seen = [], set()
        for name, q, ctx in items:
            for cname, sel in ctx.items():
                if kind == "revise" and (cname in ("initial shortlist", "gold passages only (diagnostic)") or q["qid"] not in prev
                                         or tuple(sel) == tuple(ctx["initial shortlist"])):      # nothing new to revise against
                    continue
                key = (q["qid"], tuple(sel), kind)
                if key in done or key in seen:
                    continue
                seen.add(key)
                jobs.append((name, q, cname, sel))
        return jobs
    n_fresh = len(pending("fresh"))
    if args.count_only:
        n_rev = len(pending("revise")) if args.model == "l8" else 0
        print(json.dumps({"model": args.model, "pending": n_fresh + n_rev}))
        return
    if n_fresh == 0 and (args.model != "l8" or len(pending("revise")) == 0):
        log(f"{args.model}: every answer is already there")
        return
    eng = W.make_engine(args.backend, models, args.model, max_model_len=8192, gpu_util=0.88)
    log(f"{args.model} ready in {time.time() - t0:.0f}s")
    P_ = models[args.model]["params"]
    for kind in kinds:
        jobs = pending(kind)
        log(f"{args.model} {kind}: {len(jobs)} answers to generate")
        for c in range(0, len(jobs), args.chunk):
            chunk = jobs[c:c + args.chunk]
            prompts = []
            for name, q, cname, sel in chunk:
                user = S.qa_prompt(q["question"], S.selected_refs(q, sel), cot=True)
                if kind == "revise":
                    user = user.rsplit(S.SHORT_ANSWER, 1)[0] + REVISE.format(prev=prev[q["qid"]])
                else:
                    user = user + W.CONF_ANS
                prompts.append(eng.encode(S.QA_SYSTEM, user))
            meta = [{"kind": "qa", "gold": q["answer"], "hit": 1.0, "cot": True, "conf": kind == "fresh"} for _, q, _, _ in chunk]
            outs = eng.generate(prompts, 1, 0.0, 512, [0] * len(chunk), meta)
            rows = []
            for (name, q, cname, sel), p, o in zip(chunk, prompts, outs):
                text, k_out = o[0][0], o[0][1]
                pred = W.answer_of(text)
                em, f1 = S.em_f1(pred, q["answer"])
                rows.append({"set": name, "qid": q["qid"], "context": cname, "sel": list(sel), "kind": kind, "model": args.model, "prompt_tokens": len(p), "out_tokens": k_out,
                             "flops": 2 * P_ * (len(p) + k_out), "text": text, "prediction": pred, "EM": em, "F1": f1, "lenient": X.lenient(pred, q["answer"]),
                             "stated_conf": W.stated_conf(text), "hit_cap": k_out >= 512})
                if kind == "fresh" and cname == "initial shortlist":
                    prev[q["qid"]] = pred
            S.append_jsonl(path, rows)
            log(f"  {args.model} {kind}: {min(c + args.chunk, len(jobs))}/{len(jobs)}")
            if args.time_limit_min and time.time() - t0 > 60 * args.time_limit_min:
                log(f"{args.model}: time limit reached - run again to continue")
                return
    log(f"{args.model} done in {time.time() - t0:.0f}s")


# ==================================================================================
# Analysis of the new answers: items 1-3 (command `actions`, CPU)
# ==================================================================================
CTX_RETRY = ["retry union", "retry, reranked to 2", "retry, reranked to 3", "retry, reranked to 4"]
GOLD = "gold passages only (diagnostic)"
QR_GRID = [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]          # share of questions whose retrieval is retried
QA_GRID = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]    # share whose answer is escalated to Qwen 32B
REMEDIES = {"l8": "retry, then Llama 8B answers again from scratch (primary)", "rev": "retry, then Llama 8B revises its first answer",
            "q32": "retry, then Qwen 32B answers"}


class Arms:
    """Per question: the outcome and the full compute of every action, for one correctness measure."""

    def __init__(self, args, P, sets):
        params = {k: v["params"] for k, v in W.load_models(os.path.join(args.old_sweep, "models.json")).items()}
        arms = {}
        for m in ("l8", "q32"):
            for r in S.read_jsonl(arm_path(args.run_dir, m)):
                arms[(m, r["kind"], r["qid"], tuple(r["sel"]))] = r
        self.n = P.n
        self.ctx, self.gold_in, self.r1, self.r3, self.raw = [], [], np.zeros(P.n), np.zeros(P.n), {}
        self.missing = 0
        i = 0
        for s, run in zip(sets, (args.old_sweep, args.fresh_run)):
            rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RC.RET, RC.COT, RC.PICK)))}
            for q in s["qs"]:
                row = rets[q["qid"]]
                c = contexts(q, row)
                self.ctx.append(c)
                gold = {j + 1 for j in q["ret_answers"]}
                self.gold_in.append({k: float(gold <= set(v)) for k, v in c.items()})
                self.r1[i] = 2 * params[RC.RET] * (row["prompt_tokens"] + row["samples"][0]["out_tokens"])
                self.r3[i] = 2 * params[RC.RET] * (row["prompt_tokens"] + sum(x["out_tokens"] for x in row["samples"][:3]))
                for m, kind in (("l8", "fresh"), ("l8", "revise"), ("q32", "fresh")):
                    for cname, sel in c.items():
                        if kind == "revise" and (cname in ("initial shortlist", GOLD) or tuple(sel) == tuple(c["initial shortlist"])):
                            continue
                        r = arms.get((m, kind, q["qid"], tuple(sel)))
                        if r is None:
                            self.missing += 1
                        self.raw[(m, kind, i, cname)] = r
                i += 1
        self.P = P
        self.orig = {"EM": P.A_em_e1, "lenient": P.A["e1"]}                  # the cheap fixed setup as run before: one Qwen 32B draft + the fuse step
        self.c_orig = P.A["c1"]
        self.first_from_extraction = bool(getattr(args, "first_from_extraction", False))
        self.first_ok = {"EM": P.l8_em, "lenient": P.l8}
        self.first_cost = P.A["cl8"]
        self.l8_agree = float(np.mean([S.normalize_answer(self.raw[("l8", "fresh", i, "initial shortlist")]["prediction"]) == S.normalize_answer(P.l8_pred[i])
                                       for i in range(P.n) if self.raw[("l8", "fresh", i, "initial shortlist")] is not None]))

    def get(self, m, kind, i, cname, metric):
        """(correct, flops) of one generation; a revision on an unchanged context is the first answer, at no cost."""
        if self.first_from_extraction and m == "l8" and kind == "fresh" and cname == "initial shortlist":
            return float(self.first_ok[metric][i]), float(self.first_cost[i])      # the answer whose hidden state the probe reads
        if kind == "revise" and (cname == "initial shortlist" or tuple(self.ctx[i][cname]) == tuple(self.ctx[i]["initial shortlist"])):
            return self.get("l8", "fresh", i, "initial shortlist", metric)[0], 0.0
        r = self.raw[(m, kind, i, cname)]
        return float(r[metric]), float(r["flops"])

    def table(self, metric, kctx):
        """Arrays (correct, compute) per question for each action; kctx = the context used after a retry, per question."""
        n = self.n
        o = {a: (np.zeros(n), np.zeros(n)) for a in ("fixed", "original fixed", "keep", "escalate", "retry l8", "retry rev", "retry q32", "gold l8", "gold q32")}
        for i in range(n):
            k = kctx[i] if isinstance(kctx, (list, np.ndarray)) else kctx
            same = tuple(self.ctx[i][k]) == tuple(self.ctx[i]["initial shortlist"])
            y0, c0 = self.get("l8", "fresh", i, "initial shortlist", metric)
            yb, cb = self.get("q32", "fresh", i, "initial shortlist", metric)
            base = self.r1[i] + c0
            retry = self.r3[i] + c0                                         # the extra tries are charged even if they add nothing
            vals = {"fixed": (yb, self.r1[i] + cb), "original fixed": (self.orig[metric][i], self.c_orig[i]), "keep": (y0, base), "escalate": (yb, base + cb)}
            yl, cl = (y0, 0.0) if same else self.get("l8", "fresh", i, k, metric)
            yr, cr = self.get("l8", "revise", i, k, metric)
            yq, cq = self.get("q32", "fresh", i, k, metric)
            vals.update({"retry l8": (yl, retry + cl), "retry rev": (yr, retry + cr), "retry q32": (yq, retry + cq)})
            if self.raw.get(("l8", "fresh", i, GOLD)) is not None:
                vals["gold l8"] = (self.get("l8", "fresh", i, GOLD, metric)[0], np.nan)
                vals["gold q32"] = (self.get("q32", "fresh", i, GOLD, metric)[0], np.nan)
            else:
                vals["gold l8"] = vals["gold q32"] = (np.nan, np.nan)
            for a, (y, c) in vals.items():
                o[a][0][i], o[a][1][i] = y, c
        return o


def probe_scores(P, tr, te, decision):
    if decision == "answerer":
        Hh = P.ha["h_answer_line"][:, P.ha["h_answer_line"].shape[1] - 2, :]
    else:
        Hh = P.hr[:, P.hr.shape[1] // 2, :]
    red = F.fit_reduce(Hh[tr])
    return H.f_linear(F.apply_reduce(red, Hh[tr]), P.y[decision][tr].astype(int), F.apply_reduce(red, Hh[te]))


def oof_scores(P, tr, decision, seed=0):
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros(len(tr))
    y = P.y[decision][tr]
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(tr)), y):
        out[b] = probe_scores(P, tr[a], tr[b], decision)
    return out


def apply_rule(sr, sa, tr_thr, ta_thr, T, idx, remedy):
    """Three actions: retry where the retrieval looks incomplete; otherwise escalate where the answer probe is low; otherwise keep."""
    retry = sr <= tr_thr
    esc = (~retry) & (sa <= ta_thr)
    y = np.where(retry, T[f"retry {remedy}"][0][idx], np.where(esc, T["escalate"][0][idx], T["keep"][0][idx]))
    c = np.where(retry, T[f"retry {remedy}"][1][idx], np.where(esc, T["escalate"][1][idx], T["keep"][1][idx]))
    return y, c, retry, esc


_RCACHE = {}


def retriever_scores(P, tr, te, seed=0):
    """(out-of-fold scores on tr, scores on te) for the retry decision; low = the first try looks incomplete.
    RETRIEVER_SCORE = 'probe': the plain retriever probe. 'combined': the probe + evidence + structure + facts about the first try,
    each read by its own linear probe fitted on tr, z-scored and averaged with equal weights (item 6's best retriever score, 0.74)."""
    key = (RETRIEVER_SCORE, seed, tuple(tr[:5]), len(tr), tuple(te[:5]), len(te))
    if key in _RCACHE:
        return _RCACHE[key]
    if RETRIEVER_SCORE == "probe":
        out = (oof_scores(P, tr, "retriever", seed), probe_scores(P, tr, te, "retriever"))
    else:
        ytr = P.y["retriever"][tr].astype(float)
        Ft = X.Feats(P, tr)
        li = P.hr.shape[1] // 2
        mats = [F.apply_reduce(F.fit_reduce(P.hr[tr, li, :]), P.hr[:, li, :]), np.column_stack([Ft.voters_table, Ft.rarity]), Ft.structure, P.meta]
        zo, zn = [], []
        for Xm in mats:
            po = np.nanmean([H.cv(H.f_linear, Xm[tr], ytr, 10 + seed + r) for r in range(2)], 0)
            pn = H.f_linear(Xm[tr], ytr.astype(int), Xm[te])
            po, pn = np.nan_to_num(po, nan=ytr.mean()), np.nan_to_num(pn, nan=ytr.mean())
            zo.append(z(po, po)); zn.append(z(pn, po))
        out = (np.mean(zo, 0), np.mean(zn, 0))
    _RCACHE[key] = out
    return out


RETRIEVER_SCORE = "probe"


def split_policies(A, P, tr, te, metric, remedy, seed=0):
    """Everything fitted on `tr` (probes, thresholds, the retry context k); applied once to `te`."""
    # the bounded context: chosen on the training questions by the accuracy of the retry action
    acc_k = {}
    for k in CTX_RETRY:
        Tk = A.table(metric, k)
        acc_k[k] = float(np.mean(Tk[f"retry {remedy}"][0][tr]))
    kbest = max(CTX_RETRY, key=lambda k: (acc_k[k], k != "retry union"))
    T = A.table(metric, kbest)
    oa, sa = oof_scores(P, tr, "answerer", seed), probe_scores(P, tr, te, "answerer")
    orr, sr = retriever_scores(P, tr, te, seed)
    budget = T["fixed"][1][tr].mean()
    res = {}

    def choose(grid):
        best = None
        for qr, qa in grid:
            ty, tc, _, _ = apply_rule(orr, oa, np.quantile(orr, qr) if qr else -np.inf, np.quantile(oa, qa) if qa else -np.inf, T, tr, remedy)
            key = (tc.mean() <= budget, ty.mean(), -tc.mean())
            if best is None or key > best[0]:
                best = (key, qr, qa)
        return best[1], best[2]
    grids = {"three actions (primary)": [(a, b) for a in QR_GRID for b in QA_GRID],
             "answerer-only gate": [(0.0, b) for b in QA_GRID], "retriever-only gate": [(a, 0.0) for a in QR_GRID]}
    for name, g in grids.items():
        qr, qa = choose(g)
        y, c, rt, es = apply_rule(sr, sa, np.quantile(orr, qr) if qr else -np.inf, np.quantile(oa, qa) if qa else -np.inf, T, te, remedy)
        res[name] = dict(y=y, c=c, retry=rt, esc=es, qr=qr, qa=qa)
    y, c, rt, es = apply_rule(sr, sa, -np.inf, np.quantile(oa, 0.40), T, te, remedy)
    res["answerer probe at 40% (notebook 20 reference)"] = dict(y=y, c=c, retry=rt, esc=es, qr=0.0, qa=0.40)
    curve = {}
    for qr in QR_GRID:
        for qa in QA_GRID:
            y, c, _, _ = apply_rule(sr, sa, np.quantile(orr, qr) if qr else -np.inf, np.quantile(oa, qa) if qa else -np.inf, T, te, remedy)
            curve[(qr, qa)] = (y, c)
    return res, curve, kbest, T


def cmd_actions(args):
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    t0 = time.time()
    global RETRIEVER_SCORE
    RETRIEVER_SCORE = args.retriever_score
    out = os.path.join(args.run_dir, "results_actions" + ("_probe_answer" if args.first_from_extraction else "")
                       + ("_combined_retriever" if args.retriever_score == "combined" else ""))
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    log(f"actions analysis, code version {VERSION}, retriever score: {args.retriever_score}" + (" | first Llama answer = the extraction pass the probe reads" if args.first_from_extraction else ""))
    old = X.load_set("old", args.old_sweep, args.old_data, args.old_hidden)
    new = X.load_set("fresh", args.fresh_run, args.fresh_data, args.fresh_run)
    sets = [old, new]
    P0 = E.Pool(sets, use_embedding=False)
    P0.l8_pred = [r["Llama 8B answer"] for s in sets for r in s["label_rows"]]
    P = X.relabel(P0, sets, "lenient")          # probe targets as in notebook 20
    P.A_em_e1 = P0.A["e1"]
    P.l8_em = P0.l8
    P.sel1 = [x for s in sets for x in s["sel1"]]
    io, inew = np.where(P.origin == "old")[0], np.where(P.origin == "fresh")[0]
    A = Arms(args, P, sets)
    assert A.missing == 0, f"{A.missing} answers are missing - run `generate` for both models first"
    log(f"answers loaded; the Llama 8B first answer here matches the earlier extraction pass on {100 * A.l8_agree:.0f}% of questions")
    n = P.n
    fr = lambda a: a.mean()
    # ------------------------------------------------------------ item 2 (and the action table) descriptive, every question
    desc = []
    init_gold = np.array([g["initial shortlist"] for g in A.gold_in])
    for metric in ("EM", "lenient"):
        y0 = A.table(metric, "retry union")["keep"][0]
        for cname in ["initial shortlist"] + CTX_RETRY + [GOLD]:
            union_gold = np.array([g["retry union"] for g in A.gold_in])
            for subset, mask in (("all 400", np.ones(n, bool)), ("context changed by the retry", np.array([tuple(c[cname]) != tuple(c["initial shortlist"]) for c in A.ctx])),
                                 ("diagnostic: first try missed a gold passage, the retry union has both", (init_gold == 0) & (union_gold == 1))):
                if cname in ("initial shortlist",) and subset == "context changed by the retry":
                    continue
                for m in ("l8", "q32"):
                    ys = np.array([A.get(m, "fresh", i, cname, metric)[0] if A.raw[(m, "fresh", i, cname)] is not None else np.nan for i in range(n)])
                    ok = mask & ~np.isnan(ys)
                    if not ok.any():
                        continue
                    toks = np.array([A.raw[(m, "fresh", i, cname)]["prompt_tokens"] for i in np.where(ok)[0]])
                    d = {"Metric": metric, "Context": cname, "Model": "Llama 8B" if m == "l8" else "Qwen 32B", "Questions": subset, "n": int(ok.sum()),
                         "Accuracy": float(ys[ok].mean()), "Both gold passages in the context": float(np.mean([A.gold_in[i][cname] for i in np.where(ok)[0]])),
                         "Mean prompt tokens": float(toks.mean())}
                    if m == "l8":
                        d["Repairs (Llama first answer wrong -> right)"] = int(((y0 == 0) & (ys == 1) & ok).sum())
                        d["Damage (right -> wrong)"] = int(((y0 == 1) & (ys == 0) & ok).sum())
                    desc.append(d)
        for cname in CTX_RETRY:
            ch = np.array([tuple(c[cname]) != tuple(c["initial shortlist"]) for c in A.ctx])
            for subset, mask in (("context changed by the retry", ch), ("diagnostic: first try missed a gold passage, the retry union has both", ch & (init_gold == 0) & (np.array([g["retry union"] for g in A.gold_in]) == 1))):
                ys = np.array([A.get("l8", "revise", i, cname, metric)[0] for i in range(n)])
                desc.append({"Metric": metric, "Context": cname, "Model": "Llama 8B revises its first answer", "Questions": subset, "n": int(mask.sum()),
                             "Accuracy": float(ys[mask].mean()) if mask.any() else None, "Both gold passages in the context": float(np.mean([A.gold_in[i][cname] for i in np.where(mask)[0]])) if mask.any() else None,
                             "Repairs (Llama first answer wrong -> right)": int(((y0 == 0) & (ys == 1) & mask).sum()), "Damage (right -> wrong)": int(((y0 == 1) & (ys == 0) & mask).sum())})
    # ------------------------------------------------------------ item 1: action table and oracle, per retry context
    act = []
    for metric in ("EM", "lenient"):
        for k in CTX_RETRY:
            T = A.table(metric, k)
            ref = T["fixed"][1].mean()
            for a in ("original fixed", "fixed", "keep", "escalate", "retry l8", "retry rev", "retry q32"):
                act.append({"Metric": metric, "Retry context": k, "Action (every question)": a, "Accuracy": fr(T[a][0]), "Compute vs the cheap fixed setup": T[a][1].mean() / ref - 1,
                            "Right where keeping the first answer was wrong": int(((T["keep"][0] == 0) & (T[a][0] == 1)).sum()),
                            "Wrong where keeping the first answer was right": int(((T["keep"][0] == 1) & (T[a][0] == 0)).sum())})
            for nm, acts in (("ORACLE keep / escalate (upper bound, needs the answer key)", ("keep", "escalate")),
                             ("ORACLE keep / escalate / retry with Llama 8B (upper bound)", ("keep", "escalate", "retry l8")),
                             ("ORACLE keep / escalate / retry with Llama 8B or revision (upper bound)", ("keep", "escalate", "retry l8", "retry rev"))):
                Y = np.column_stack([T[a][0] for a in acts]); C = np.column_stack([T[a][1] for a in acts])
                Cm = np.where(Y == 1, C, np.inf)
                pick = np.where(np.isfinite(Cm.min(1)), Cm.argmin(1), 0)          # cheapest right action; keep when none is right
                act.append({"Metric": metric, "Retry context": k, "Action (every question)": nm, "Accuracy": float(Y[np.arange(n), pick].mean()),
                            "Compute vs the cheap fixed setup": C[np.arange(n), pick].mean() / ref - 1,
                            "Share escalated": float(np.mean(np.array(acts)[pick] == "escalate")), "Share retried": float(np.mean(np.char.startswith(np.array(acts)[pick], "retry")))})
    log(f"action table done ({time.time() - t0:.0f}s)")
    # ------------------------------------------------------------ item 1 policies (and items 2-3 inside them), nested selection
    splits = {"old -> fresh": [(io, inew)], "fresh -> old": [(inew, io)]}
    strat = np.array([f"{o}{int(v)}" for o, v in zip(P.origin, P.y["retriever"])])
    cvf = []
    for r in range(args.cv_repeats):
        cvf += [(a, b) for a, b in StratifiedKFold(5, shuffle=True, random_state=200 + r).split(np.zeros(n), strat)]
    splits["all 400, cross-validated"] = cvf
    pol, curves, perq, ks = [], [], [], []
    for metric in ("EM", "lenient"):
        for remedy in REMEDIES:
            for ev, folds in splits.items():
                acc = {}                              # name -> per-question arrays averaged over repeats
                cur = {}
                cnt = np.zeros(n)
                fixed_y, fixed_c, keep_y, esc_y, ret_y, orig_y, orig_c = (np.zeros(n) for _ in range(7))
                for f_i, (tr, te) in enumerate(folds):
                    res, curve, kb, T = split_policies(A, P, tr, te, metric, remedy, seed=f_i)
                    ks.append({"Metric": metric, "Remedy": remedy, "Evaluation": ev, "Fold": f_i, "Retry context chosen on the training questions": kb})
                    cnt[te] += 1
                    fixed_y[te] += T["fixed"][0][te]; fixed_c[te] += T["fixed"][1][te]
                    orig_y[te] += T["original fixed"][0][te]; orig_c[te] += T["original fixed"][1][te]
                    keep_y[te] += T["keep"][0][te]; esc_y[te] += T["escalate"][0][te]; ret_y[te] += T[f"retry {remedy}"][0][te]
                    for nm, d in res.items():
                        a = acc.setdefault(nm, {k: np.zeros(n) for k in ("y", "c", "retry", "esc", "rnd")})
                        a["y"][te] += d["y"]; a["c"][te] += d["c"]; a["retry"][te] += d["retry"]; a["esc"][te] += d["esc"]
                        sr_, se_ = d["retry"].mean(), d["esc"].mean()
                        a["rnd"][te] += sr_ * T[f"retry {remedy}"][0][te] + se_ * T["escalate"][0][te] + (1 - sr_ - se_) * T["keep"][0][te]   # random, same action mix
                        a.setdefault("q", []).append((d["qr"], d["qa"]))
                    for key, (y, c) in curve.items():
                        cc = cur.setdefault(key, (np.zeros(n), np.zeros(n)))
                        cc[0][te] += y; cc[1][te] += c
                idx = np.where(cnt > 0)[0]
                m_ = lambda v: v[idx] / cnt[idx]
                fy, fc = m_(fixed_y), m_(fixed_c)
                aoy = m_(acc["answerer-only gate"]["y"])
                for nm, a in acc.items():
                    y, c, rnd = m_(a["y"]), m_(a["c"]), m_(a["rnd"])
                    lo_b, hi_b = F.boot(lambda w: y[w].mean() - fy[w].mean(), len(idx), B=2000)
                    lo_r, hi_r = F.boot(lambda w: y[w].mean() - rnd[w].mean(), len(idx), B=2000, seed=1)
                    lo_a, hi_a = F.boot(lambda w: y[w].mean() - aoy[w].mean(), len(idx), B=2000, seed=2)
                    shares = a["q"]
                    pol.append({"Metric": metric, "Remedy after a retry": REMEDIES[remedy], "Evaluation": ev, "Policy": nm, "Accuracy": float(y.mean()),
                                "Cheap fixed setup": float(fy.mean()), "vs cheap fixed setup (points)": 100 * float(y.mean() - fy.mean()), "low": 100 * lo_b, "high": 100 * hi_b,
                                "Compute vs the cheap fixed setup": float(c.mean() / fc.mean() - 1),
                                "vs random, same action mix (points)": 100 * float(y.mean() - rnd.mean()), "random: low": 100 * lo_r, "random: high": 100 * hi_r,
                                "vs answerer-only gate (points)": 100 * float(y.mean() - aoy.mean()), "answerer-only: low": 100 * lo_a, "answerer-only: high": 100 * hi_a,
                                "Share retried": float(m_(a["retry"]).mean()), "Share escalated": float(m_(a["esc"]).mean()),
                                "Planned shares chosen in training (retry, escalate)": ", ".join(sorted({f"{x:.2f}/{z:.2f}" for x, z in shares})),
                                "Llama 8B alone": float(m_(keep_y).mean()),
                                "Original fixed setup (draft + fuse, as in notebooks 11-20)": float(m_(orig_y).mean()),
                                "vs original fixed setup (points)": 100 * float(y.mean() - m_(orig_y).mean()),
                                "Compute vs the original fixed setup": float(c.mean() / m_(orig_c).mean() - 1)})
                    if ev == "old -> fresh" or (ev == "all 400, cross-validated"):
                        for j, i in enumerate(idx):
                            if nm == "three actions (primary)":
                                perq.append({"Metric": metric, "Remedy": remedy, "Evaluation": ev, "qid": P.qs[i]["qid"], "set": P.origin[i], "retried (share of repeats)": float(m_(a["retry"])[j]),
                                             "escalated (share of repeats)": float(m_(a["esc"])[j]), "correct": float(y[j]), "compute": float(c[j]), "fixed setup correct": float(fy[j])})
                for (qr, qa), (y, c) in cur.items():
                    curves.append({"Metric": metric, "Remedy": remedy, "Evaluation": ev, "Retry share": qr, "Escalate share": qa,
                                   "Accuracy": float(m_(y).mean()), "Compute vs the cheap fixed setup": float(m_(c).mean() / fc.mean() - 1)})
                log(f"policies | {metric} | {remedy} | {ev} done ({time.time() - t0:.0f}s)")
    # ------------------------------------------------------------ item 3: on the questions the retriever probe would retry (20%), all three arms
    rev = []
    for metric in ("EM", "lenient"):
        for ev, folds in (("old -> fresh", [(io, inew)]), ("all 400, cross-validated", cvf[:5])):
            trig = np.zeros(n, bool)
            kchosen = np.array(["retry union"] * n, dtype=object)          # overwritten for the test questions of each split
            for tr, te in folds:
                orr, sr = retriever_scores(P, tr, te, 0)
                trig[te] = sr <= np.quantile(orr, 0.20)
                acc_k = {k: float(np.mean(A.table(metric, k)["retry l8"][0][tr])) for k in CTX_RETRY}
                kchosen[te] = max(CTX_RETRY, key=lambda k: (acc_k[k], k != "retry union"))
            idx = np.where(trig)[0] if ev == "all 400, cross-validated" else inew[trig[inew]]
            for kk in ("chosen bounded context", "retry union"):
                T = A.table(metric, list(kchosen) if kk == "chosen bounded context" else "retry union")
                ref = T["fixed"][1][idx].mean()
                y0 = T["keep"][0][idx]
                rec = np.array([A.gold_in[i]["initial shortlist"] == 0 and A.gold_in[i][kchosen[i] if kk == "chosen bounded context" else "retry union"] == 1 for i in idx])
                for a, lab in (("keep", "Llama 8B first answer (no retry)"), ("escalate", "Qwen 32B on the first shortlist (no retry)"),
                               ("retry l8", "Llama 8B from scratch on the updated context"), ("retry rev", "Llama 8B revises with its first answer"),
                               ("retry q32", "Qwen 32B on the updated context")):
                    y = T[a][0][idx]
                    rev.append({"Metric": metric, "Evaluation": ev, "Updated context": kk, "Questions retried by the probe": len(idx), "Arm": lab, "Accuracy": float(y.mean()),
                                "Repairs (wrong -> right)": int(((y0 == 0) & (y == 1)).sum()), "Regressions (right -> wrong)": int(((y0 == 1) & (y == 0)).sum()),
                                "Evidence recovered (a gold passage missing at first is now in)": int(rec.sum()),
                                "Recovered and repaired": int((rec & (y0 == 0) & (y == 1)).sum()), "Recovered, not repaired (still wrong)": int((rec & (y == 0)).sum()),
                                "Compute vs the cheap fixed setup (on these questions)": float(T[a][1][idx].mean() / ref - 1)})
    log(f"item 3 done ({time.time() - t0:.0f}s)")

    def dump(nm, rr):
        with open(os.path.join(out, nm), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    for nm, rr in (("item2_contexts.jsonl", desc), ("item1_action_table.jsonl", act), ("item1_policies.jsonl", pol), ("item1_curves.jsonl", curves),
                   ("item1_per_question.jsonl", perq), ("item3_revision.jsonl", rev), ("chosen_contexts.jsonl", ks)):
        dump(nm, rr)
    summary_actions(out, desc, act, pol, rev, ks, A)
    charts_actions(out, curves, pol)
    W.write_excel(os.path.join(out, "set6_actions.xlsx"), {"Item 1 policies": pd.DataFrame(pol), "Item 1 action table": pd.DataFrame(act),
                                                           "Item 1 curves": pd.DataFrame(curves), "Item 2 contexts": pd.DataFrame(desc), "Item 3 revision": pd.DataFrame(rev),
                                                           "Chosen contexts": pd.DataFrame(ks)})
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def _T(rr, cols, pct=(), signed=()):
    def f(r, c):
        v = r.get(c)
        if v is None or (isinstance(v, float) and v != v):
            return "-"
        if isinstance(v, str):
            return v
        if isinstance(v, (int, np.integer)):
            return str(v)
        if c in signed:
            return f"{v:+.1f}"
        if c in pct:
            return f"{100 * v:.1f}%"
        return f"{v:.2f}"
    return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rr)


def summary_actions(out, desc, act, pol, rev, ks, A):
    from collections import Counter
    L = ["# Set 6, GPU part - items 1-3 of the next-experiments plan (different remedies, adaptive context, answer revision)\n",
         f"Code version {VERSION}. 400 development questions (200 old + 200 fresh). Every answer is a new greedy generation with the same prompt and a 512-token limit. "
         "Exact match (EM) is the primary score; the notebook 20 corrected label ('lenient') is a secondary proxy. Compute charges the first retrieval try, the Llama 8B first answer "
         "(for every gate), the two extra retrieval tries when retried, and every further answer. The cheap fixed setup has no Llama pass.\n",
         f"The Llama 8B first answer here matches the earlier extraction pass (whose hidden state the probe reads) on {100 * A.l8_agree:.0f}% of questions.\n",
         "## Item 1 - routing between remedies (thresholds, probes and the retry context fitted on training questions only)\n"]
    cols = ["Policy", "Accuracy", "vs cheap fixed setup (points)", "low", "high", "Compute vs the cheap fixed setup", "vs answerer-only gate (points)", "answerer-only: low",
            "answerer-only: high", "vs random, same action mix (points)", "random: low", "random: high", "vs original fixed setup (points)", "Compute vs the original fixed setup", "Share retried", "Share escalated", "Planned shares chosen in training (retry, escalate)"]
    sg = {"vs cheap fixed setup (points)", "low", "high", "vs answerer-only gate (points)", "answerer-only: low", "answerer-only: high", "vs random, same action mix (points)", "random: low", "random: high"}
    for metric in ("EM", "lenient"):
        for remedy, lab in REMEDIES.items():
            for ev in ("all 400, cross-validated", "old -> fresh", "fresh -> old"):
                rr = [r for r in pol if r["Metric"] == metric and r["Remedy after a retry"] == lab and r["Evaluation"] == ev]
                if not rr:
                    continue
                L += [f"**{metric} | remedy: {lab} | {ev}** (cheap fixed setup, one greedy Qwen 32B answer: {100 * rr[0]['Cheap fixed setup']:.1f}%; "
                      f"as run before with the fuse step: {100 * rr[0]['Original fixed setup (draft + fuse, as in notebooks 11-20)']:.1f}%; Llama 8B alone {100 * rr[0]['Llama 8B alone']:.1f}%)\n",
                      _T(rr, cols, pct={"Accuracy", "Compute vs the cheap fixed setup", "Share retried", "Share escalated", "Compute vs the original fixed setup"}, signed=sg | {"vs original fixed setup (points)"}), ""]
    L += ["### Action table (every question takes the same action) and oracles\n"]
    for metric in ("EM", "lenient"):
        rr = [r for r in act if r["Metric"] == metric and r["Retry context"] in ("retry union", "retry, reranked to 3")]
        L += [f"**{metric}**\n", _T(rr, ["Retry context", "Action (every question)", "Accuracy", "Compute vs the cheap fixed setup", "Right where keeping the first answer was wrong",
                                         "Wrong where keeping the first answer was right", "Share escalated", "Share retried"],
                                     pct={"Accuracy", "Compute vs the cheap fixed setup", "Share escalated", "Share retried"}), ""]
    L += ["## Item 2 - context size after a retry\n", "Retry contexts chosen on the training side of each split: " +
          ", ".join(f"{k} x{v}" for k, v in Counter(r["Retry context chosen on the training questions"] for r in ks).items()) + "\n"]
    for metric in ("EM", "lenient"):
        for subset in ("all 400", "context changed by the retry", "diagnostic: first try missed a gold passage, the retry union has both"):
            rr = [r for r in desc if r["Metric"] == metric and r["Questions"] == subset]
            L += [f"**{metric} | {subset}**\n", _T(rr, ["Context", "Model", "n", "Accuracy", "Both gold passages in the context", "Mean prompt tokens",
                                                       "Repairs (Llama first answer wrong -> right)", "Damage (right -> wrong)"],
                                                   pct={"Accuracy", "Both gold passages in the context"}), ""]
    L += ["## Item 3 - revision vs fresh answer vs Qwen 32B, on the questions the retriever probe retries (20%)\n"]
    for metric in ("EM", "lenient"):
        rr = [r for r in rev if r["Metric"] == metric]
        L += [f"**{metric}**\n", _T(rr, ["Evaluation", "Updated context", "Questions retried by the probe", "Arm", "Accuracy", "Repairs (wrong -> right)", "Regressions (right -> wrong)",
                                         "Evidence recovered (a gold passage missing at first is now in)", "Recovered and repaired", "Recovered, not repaired (still wrong)",
                                         "Compute vs the cheap fixed setup (on these questions)"], pct={"Accuracy", "Compute vs the cheap fixed setup (on these questions)"}), ""]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def charts_actions(out, curves, pol):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for metric in ("EM", "lenient"):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
        for ax, remedy in zip(axes, REMEDIES):
            rr = [r for r in curves if r["Metric"] == metric and r["Remedy"] == remedy and r["Evaluation"] == "all 400, cross-validated"]
            if not rr:
                continue
            x = np.array([100 * r["Compute vs the cheap fixed setup"] for r in rr]); y = np.array([100 * r["Accuracy"] for r in rr])
            qr = np.array([r["Retry share"] for r in rr]); qa = np.array([r["Escalate share"] for r in rr])
            ax.scatter(x[qr > 0], y[qr > 0], c=qr[qr > 0], cmap="viridis", s=26, label="three actions (colour = retry share)")
            o = np.argsort(x[qr == 0])
            ax.plot(x[qr == 0][o], y[qr == 0][o], "o-", color="k", ms=4, label="answerer-only gate")
            fx = [r for r in pol if r["Metric"] == metric and r["Remedy after a retry"] == REMEDIES[remedy] and r["Evaluation"] == "all 400, cross-validated"]
            if fx:
                ax.axhline(100 * fx[0]["Cheap fixed setup"], color="tab:red", ls="--", lw=1, label="cheap fixed setup")
            ax.axvline(0, color="tab:red", ls=":", lw=1)
            ax.set_title(REMEDIES[remedy], fontsize=9)
            ax.set_xlabel("compute vs the cheap fixed setup (%)")
        axes[0].set_ylabel(f"accuracy ({metric}, %)")
        axes[0].legend(fontsize=7)
        fig.suptitle(f"Accuracy-compute frontier, all 400 cross-validated ({metric}); fixed planned shares, thresholds from training folds", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "charts", f"frontier_{metric}.png"), dpi=130)
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("cpu")
    for a in ("old_sweep", "old_hidden", "old_data", "fresh_run", "fresh_data", "run_dir"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--cv_repeats", type=int, default=2)
    p.add_argument("--no_embedding", action="store_true")
    p = sub.add_parser("generate")
    for a in ("old_sweep", "old_data", "fresh_run", "fresh_data", "run_dir", "models_json", "model"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
    p.add_argument("--chunk", type=int, default=64)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--time_limit_min", type=float, default=0)
    p.add_argument("--count_only", action="store_true")
    p = sub.add_parser("actions")
    for a in ("old_sweep", "old_hidden", "old_data", "fresh_run", "fresh_data", "run_dir"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--cv_repeats", type=int, default=2)
    p.add_argument("--retriever_score", choices=["probe", "combined"], default="probe")
    p.add_argument("--first_from_extraction", action="store_true",
                   help="use the Llama 8B first answer from the hidden-state extraction pass (the one the probe reads) instead of the regenerated one")
    args = ap.parse_args()
    {"cpu": cmd_cpu, "generate": cmd_generate, "actions": cmd_actions}[args.cmd](args)


if __name__ == "__main__":
    main()
