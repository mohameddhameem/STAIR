"""Set 5e - does the retriever's own hidden state tell us when one retrieval try is not enough? (notebook 18)

The idea: the Qwen 7B retrieves once. A probe reads its hidden state after it has read the question and the 20 passages (free: it is
part of the retrieval pass). Where the probe says the first try probably missed a passage, the retriever tries twice more and every
passage picked in any try is kept. The Qwen 32B then answers once, as in the cheap fixed setup.

  freeze    CPU  fits the probe on the OLD 200 questions only (layer and threshold by a written rule), saved with a checksum
  evaluate  CPU  applies it to the fresh questions; nothing is fitted or chosen on them
  (hidden.py extract --role retrieve --model q7 gives the hidden states; retrieval tries and Qwen 32B answers already exist from notebook 17)

Test fixed in advance (stricter than before): the gate must find both passages more often than a random gate that retries the same
share of questions, with a 95% interval that excludes zero. Accuracy is reported, but 200 questions cannot settle a 1-2 point difference.
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S          # noqa: E402
import sweep as W        # noqa: E402
import hidden as H       # noqa: E402
import final as F        # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "5e.1"
RET, COT, PICK, MERGE, BIG = "q7", 1, "3", "all", "q32"
SHARES = [round(0.05 * i, 2) for i in range(1, 20)]
SIGNALS = {"hidden": "retriever hidden state after reading the question and passages",
           "meta": "facts about the first retrieval try (stated confidence, token probability, length, passages picked)"}
META_COLS = ["stated confidence", "token probability", "output tokens", "passages picked", "wrote the final line"]
log = H.log


def load_side(run, data_dir, models_json, hidden_run):
    """Everything known about one question set: retrieval outcome after 1 and 3 tries, the big answerer's result on each, compute."""
    P = {k: v["params"] for k, v in W.load_models(models_json).items()}
    qs = S.load_questions(data_dir)
    qids = [q["qid"] for q in qs]
    rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RET, COT, PICK)))}
    answ = {(r["qid"], tuple(r["sel"])): r for r in S.read_jsonl(W.ans_path(run, BIG, COT))}
    n = len(qs)
    A = {k: np.zeros(n) for k in ("f1", "f3", "e1", "e3", "c1", "c3", "n1", "n3")}
    meta = np.full((n, len(META_COLS)), np.nan)
    for i, q in enumerate(qs):
        row = rets[q["qid"]]
        nr = len(q["ctxs_candidate"])
        sl = W.shortlists(row, COT, nr)
        gold = set(q["ret_answers"])
        for rk in (1, 3):
            sel = sl[(rk, MERGE)]
            x = answ[(q["qid"], sel)]
            fz = x["fused"]["1"]
            fr = 2 * P[RET] * (row["prompt_tokens"] + sum(s["out_tokens"] for s in row["samples"][:rk]))
            tail = 2 * P[BIG] * (x["prompt_tokens"] + x["out_tokens"][0]) + 2 * P[BIG] * (fz["prompt_tokens"] + fz["out_tokens"])
            A[f"f{rk}"][i], A[f"e{rk}"][i], A[f"c{rk}"][i], A[f"n{rk}"][i] = float(gold <= {j - 1 for j in sel}), fz["EM"], fr + tail, len(sel)
        s0 = row["samples"][0]
        v = [s0.get("stated_conf"), s0.get("token_prob"), s0["out_tokens"], len(W.picks_of(W.sel_text(s0, COT), nr)), float(s0.get("has_final_line", False))]
        meta[i] = [np.nan if x is None else float(x) for x in v]
    E = H.load_extract(hidden_run, RET, "retrieve", qids)
    return qs, A, meta, E


def cmd_freeze(args):
    t0 = time.time()
    qs, A, meta_x, E = load_side(args.sweep_run, args.data_dir, args.models_json, args.hidden_run)
    assert E is not None and E["have"].all(), "the old retriever extraction is not complete"
    ref = json.load(open(os.path.join(args.gate_dir, "frozen_gate.json")))["ref_flops"]          # the same 'x cheapest' unit as everywhere else
    y = A["f1"]                                                                                  # the first try finds both passages
    out = {"version": VERSION, "created": time.strftime("%Y-%m-%d %H:%M:%S"), "frozen_on": "the 200 Set 4 questions only", "ref_flops": ref,
           "layers": E["layers"], "shares": SHARES, "signals": {},
           "old": {"first try finds both passages": float(y.mean()), "three tries find both passages": float(A["f3"].mean()),
                   "accuracy, 1 try": float(A["e1"].mean()), "accuracy, 3 tries": float(A["e3"].mean()),
                   "compute_x, 1 try": float(A["c1"].mean() / ref), "compute_x, 3 tries": float(A["c3"].mean() / ref),
                   "accuracy where the first try missed a passage: 1 try": float(A["e1"][y == 0].mean()),
                   "accuracy where the first try missed a passage: 3 tries": float(A["e3"][y == 0].mean()),
                   "accuracy where the first try found both: 1 try": float(A["e1"][y == 1].mean()),
                   "accuracy where the first try found both: 3 tries": float(A["e3"][y == 1].mean())}}
    FZ, OLD = {}, {"f1": y}

    def oof(X):
        return np.nanmean([H.cv(H.f_linear, X, y, 10 + r) for r in range(args.repeats)], 0)

    def curve(p):
        rows = []
        for s in SHARES:
            thr = float(np.nanquantile(p, s))
            again = p <= thr
            rows.append({"share": s, "threshold": thr, "share_retried": float(again.mean()), "found_both": float(np.where(again, A["f3"], A["f1"]).mean()),
                         "accuracy": float(np.where(again, A["e3"], A["e1"]).mean()), "compute_x": float(np.where(again, A["c3"], A["c1"]).mean() / ref)})
        return rows
    cands = []
    for li in range(len(E["layers"])):
        Xr = H.reduce_once(E["h_prompt"][:, li, :])
        p = oof(Xr)
        cands.append((H.auc(y, p), li, p))
    a, li, p = max(cands, key=lambda c: c[0])
    red = F.fit_reduce(E["h_prompt"][:, li, :])
    Xr = F.apply_reduce(red, E["h_prompt"][:, li, :])
    fz = dict(red, **F.fit_linear(Xr, y))
    assert np.allclose(F.apply_linear(fz, Xr), H.f_linear(Xr, y.astype(int), Xr), atol=1e-6)
    out["signals"]["hidden"] = {"label": SIGNALS["hidden"], "layer_index": li, "layer": E["layers"][li], "of_layers": 28,
                                "held-out separation at every layer": {str(E["layers"][c[1]]): c[0] for c in cands},
                                "held-out separation (first try finds both)": a, "held-out curve": curve(p)}
    for k, v in fz.items():
        FZ[f"hidden_{k}"] = np.asarray(v, np.float64)
    OLD["oof_hidden"] = p
    pm = oof(meta_x)
    fzm = F.fit_linear(meta_x, y)
    out["signals"]["meta"] = {"label": SIGNALS["meta"], "held-out separation (first try finds both)": H.auc(y, pm), "held-out curve": curve(pm)}
    for k, v in fzm.items():
        FZ[f"meta_{k}"] = np.asarray(v, np.float64)
    OLD["oof_meta"] = pm
    miss = 1 - float(y.mean())
    share = min(s for s in SHARES if s >= miss - 1e-9)
    c = [r for r in out["signals"]["hidden"]["held-out curve"] if abs(r["share"] - share) < 1e-9][0]
    out["primary"] = {"signal": "hidden", "share": share, "threshold": c["threshold"], "held-out found_both (old questions)": c["found_both"],
                      "held-out accuracy (old questions)": c["accuracy"], "held-out compute_x (old questions)": c["compute_x"],
                      "rule for the layer": "the layer whose probe separates 'the first try finds both passages' best on held-out old questions",
                      "rule for the threshold": f"retry the share of questions equal to the first try's miss rate on the old questions ({miss:.1%}), "
                                                f"rounded up to the next 5% ({share:.0%}); the threshold is that share's score quantile on the old questions"}
    h = out["signals"]["hidden"]
    out["rule in words"] = (f"The Qwen 7B reads the question and the 20 passages. Read its hidden state at the last prompt token at layer {h['layer']} of 28 and compute "
                            f"the probability that one try will find both passages. IF it is at most {c['threshold']:.3f} THEN retrieve three times and keep every "
                            f"passage picked; OTHERWISE retrieve once. The Qwen 32B then answers once.")
    out["test"] = ("the gate finds both passages more often than a random gate retrying the same share, with a 95% interval that excludes zero; "
                   "accuracy is reported but 200 questions cannot settle a 1-2 point difference")
    os.makedirs(args.out_dir, exist_ok=True)
    npz = os.path.join(args.out_dir, "frozen_retriever_gate.npz")
    np.savez_compressed(npz, **FZ, **{f"old_{k}": v for k, v in OLD.items()})
    out["npz_sha256"] = F.sha256(npz)
    S.write_json(os.path.join(args.out_dir, "frozen_retriever_gate.json"), out)
    log("held-out separation by layer:", {k: round(v, 3) for k, v in h["held-out separation at every layer"].items()}, "| facts about try 1:", round(H.auc(y, pm), 3))
    log(out["rule in words"])
    log(f"old questions, held-out: found both {100 * c['found_both']:.1f}% (1 try {100 * y.mean():.1f}%, 3 tries {100 * A['f3'].mean():.1f}%), "
        f"accuracy {100 * c['accuracy']:.1f}% (1 try {100 * A['e1'].mean():.1f}%, 3 tries {100 * A['e3'].mean():.1f}%)")
    log(f"frozen in {time.time() - t0:.0f}s (checksum {out['npz_sha256'][:16]}...)")


def cmd_evaluate(args):
    import pandas as pd
    t0 = time.time()
    out = os.path.join(args.run_dir, "results_retriever")
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    meta = json.load(open(os.path.join(args.frozen_dir, "frozen_retriever_gate.json"), encoding="utf-8"))
    npz = os.path.join(args.frozen_dir, "frozen_retriever_gate.npz")
    assert F.sha256(npz) == meta["npz_sha256"], "the frozen retriever gate has changed since it was frozen"
    FZ = dict(np.load(npz))
    log(f"evaluation code version {VERSION}; frozen retriever gate made {meta['created']} (checksum {meta['npz_sha256'][:16]}...)")
    qs, A, meta_x, E = load_side(args.run_dir, args.data_dir, args.models_json, args.run_dir)
    assert E is not None and E["have"].all(), "the retriever extraction on the fresh questions is not complete - run the notebook again"
    n, ref = len(qs), meta["ref_flops"]
    y = A["f1"]
    sub = lambda sig: {k[len(sig) + 1:]: v for k, v in FZ.items() if k.startswith(sig + "_")}
    hs = meta["signals"]["hidden"]
    sc = {"hidden": F.apply_linear(sub("hidden"), F.apply_reduce(sub("hidden"), E["h_prompt"][:, hs["layer_index"], :])),
          "meta": F.apply_linear(sub("meta"), meta_x)}
    helps, hurts = (A["e3"] == 1) & (A["e1"] == 0), (A["e3"] == 0) & (A["e1"] == 1)
    gained = (A["f3"] == 1) & (A["f1"] == 0)

    def point(name, again, kind):
        again = np.asarray(again, bool)
        s = float(again.mean())
        fb, em, co = np.where(again, A["f3"], A["f1"]), np.where(again, A["e3"], A["e1"]), np.where(again, A["c3"], A["c1"])
        rnd_fb = s * A["f3"].mean() + (1 - s) * A["f1"].mean()
        rnd_em = s * A["e3"].mean() + (1 - s) * A["e1"].mean()

        def d_fb(ix):
            ss = again[ix].mean()
            return fb[ix].mean() - (ss * A["f3"][ix].mean() + (1 - ss) * A["f1"][ix].mean())

        def d_em(ix):
            ss = again[ix].mean()
            return em[ix].mean() - (ss * A["e3"][ix].mean() + (1 - ss) * A["e1"][ix].mean())
        lo, hi = F.boot(d_fb, n)
        elo, ehi = F.boot(d_em, n, seed=1)
        return {"Policy": name, "Kind": kind, "Share retried": s, "Both passages found": float(fb.mean()), "Random gate, same share: both passages found": float(rnd_fb),
                "Gate minus random: both passages found (points)": 100 * float(fb.mean() - rnd_fb), "95% interval, low (points)": 100 * lo, "95% interval, high (points)": 100 * hi,
                "Accuracy": float(em.mean()), "Random gate, same share: accuracy": float(rnd_em), "Gate minus random: accuracy (points)": 100 * float(em.mean() - rnd_em),
                "Accuracy 95% interval, low (points)": 100 * elo, "Accuracy 95% interval, high (points)": 100 * ehi, "Compute (x cheapest)": float(co.mean() / ref),
                "Accuracy vs one try (points)": 100 * float(em.mean() - A["e1"].mean()), "Compute vs one try": float(co.mean() / A["c1"].mean() - 1),
                "Missed-passage questions retried": int((again & (y == 0)).sum()), "Beats a random gate (interval excludes zero)": bool(lo > 0)}
    pr = meta["primary"]
    again_p = sc["hidden"] <= pr["threshold"]
    pol = [point("PRIMARY frozen gate: retry where the hidden state says the first try is likely incomplete", again_p, "primary"),
           point("Always one try (the cheap fixed setup)", np.zeros(n), "fixed"), point("Always three tries", np.ones(n), "fixed"),
           point("Ceiling: retry exactly where the first try missed a passage (needs the answer key)", y == 0, "ceiling")]
    for k in ("Gate minus random: both passages found (points)", "95% interval, low (points)", "95% interval, high (points)", "Gate minus random: accuracy (points)",
              "Accuracy 95% interval, low (points)", "Accuracy 95% interval, high (points)", "Beats a random gate (interval excludes zero)"):
        for r in pol[1:3]:
            r[k] = None
    c = [r for r in meta["signals"]["meta"]["held-out curve"] if abs(r["share"] - pr["share"]) < 1e-9][0]
    pol.append(point(f"Other frozen signal: facts about the first try ({pr['share']:.0%} share)", sc["meta"] <= c["threshold"], "secondary"))
    curve = []
    for sig, label in SIGNALS.items():
        for c in meta["signals"][sig]["held-out curve"]:
            d = point(f"{label} | {c['share']:.0%}", sc[sig] <= c["threshold"], "frozen curve")
            d.update({"Signal": label, "Planned share": c["share"], "Threshold": c["threshold"], "Old questions, held-out: both passages found": c["found_both"],
                      "Old questions, held-out: accuracy": c["accuracy"]})
            curve.append(d)
    sig_rows = []
    for sig, label in SIGNALS.items():
        for tg, yy in (("the first try finds both passages", y), ("retrying does not raise accuracy", 1 - helps.astype(float))):
            a = H.auc(yy, sc[sig])
            lo, hi = F.boot(lambda ix: H.auc(yy[ix], sc[sig][ix]), n, B=1000, seed=2)
            sig_rows.append({"Signal": label, "Layer": meta["signals"][sig].get("layer"), "What is predicted": tg,
                             "Old questions (held-out)": meta["signals"][sig]["held-out separation (first try finds both)"] if tg.startswith("the first") else None,
                             "Fresh questions": a, "Fresh: 95% interval, low": lo, "Fresh: 95% interval, high": hi})
    # exploratory: the probe refitted on the fresh questions with cross-validation (not part of the test)
    refit = []
    for li, L in enumerate(E["layers"]):
        p = np.nanmean([H.cv(H.f_linear, H.reduce_once(E["h_prompt"][:, li, :]), y, 10 + r) for r in range(3)], 0)
        refit.append({"Layer": L, "Old questions (held-out)": hs["held-out separation at every layer"][str(L)], "Fresh questions, probe refitted with cross-validation": H.auc(y, p)})
    o = meta["old"]
    base = [{"Measure": "First try finds both passages", "Old 200": o["first try finds both passages"], "Fresh": float(y.mean())},
            {"Measure": "Three tries find both passages", "Old 200": o["three tries find both passages"], "Fresh": float(A["f3"].mean())},
            {"Measure": "Accuracy, always 1 try", "Old 200": o["accuracy, 1 try"], "Fresh": float(A["e1"].mean())},
            {"Measure": "Accuracy, always 3 tries", "Old 200": o["accuracy, 3 tries"], "Fresh": float(A["e3"].mean())},
            {"Measure": "Where the first try missed a passage: accuracy with 1 try", "Old 200": o["accuracy where the first try missed a passage: 1 try"], "Fresh": float(A["e1"][y == 0].mean())},
            {"Measure": "Where the first try missed a passage: accuracy with 3 tries", "Old 200": o["accuracy where the first try missed a passage: 3 tries"], "Fresh": float(A["e3"][y == 0].mean())},
            {"Measure": "Where the first try found both: accuracy with 1 try", "Old 200": o["accuracy where the first try found both: 1 try"], "Fresh": float(A["e1"][y == 1].mean())},
            {"Measure": "Where the first try found both: accuracy with 3 tries", "Old 200": o["accuracy where the first try found both: 3 tries"], "Fresh": float(A["e3"][y == 1].mean())},
            {"Measure": "Primary gate: share retried", "Old 200": pr["share"], "Fresh": pol[0]["Share retried"]},
            {"Measure": "Primary gate: both passages found", "Old 200": pr["held-out found_both (old questions)"], "Fresh": pol[0]["Both passages found"]},
            {"Measure": "Primary gate: accuracy", "Old 200": pr["held-out accuracy (old questions)"], "Fresh": pol[0]["Accuracy"]}]
    perq = [{"qid": q["qid"], "question": q["question"], "type": q.get("type"), "first try finds both": int(A["f1"][i]), "three tries find both": int(A["f3"][i]),
             "passages after 1 try": int(A["n1"][i]), "passages after 3 tries": int(A["n3"][i]), "right with 1 try": int(A["e1"][i]), "right with 3 tries": int(A["e3"][i]),
             "score: hidden state": round(float(sc["hidden"][i]), 4), "score: facts about try 1": round(float(sc["meta"][i]), 4), "retried by the primary gate": bool(again_p[i])}
            for i, q in enumerate(qs)]

    def dump(name, rr):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    dump("policies.jsonl", pol); dump("frozen_curve.jsonl", curve); dump("signals.jsonl", sig_rows); dump("old_vs_fresh.jsonl", base)
    dump("per_question.jsonl", perq); dump("refit_by_layer.jsonl", refit)
    P_ = pol[0]
    S.write_json(os.path.join(out, "verdict.json"), {"version": VERSION, "checksum": meta["npz_sha256"], "questions": n,
                                                      "beats a random gate": P_["Beats a random gate (interval excludes zero)"], "primary": P_,
                                                      "retrying helps / hurts accuracy (questions)": [int(helps.sum()), int(hurts.sum())],
                                                      "retrying gains the missing passage (questions)": int(gained.sum())})
    charts(os.path.join(out, "charts"), meta, pol, curve, sc, y, FZ, A, ref, refit)
    pct = {"Share retried", "Both passages found", "Random gate, same share: both passages found", "Accuracy", "Random gate, same share: accuracy", "Planned share",
           "Old questions, held-out: both passages found", "Old questions, held-out: accuracy"}
    signed = {"Gate minus random: both passages found (points)", "95% interval, low (points)", "95% interval, high (points)", "Gate minus random: accuracy (points)",
              "Accuracy 95% interval, low (points)", "Accuracy 95% interval, high (points)", "Accuracy vs one try (points)"}
    nums = {"Old questions (held-out)", "Fresh questions", "Fresh: 95% interval, low", "Fresh: 95% interval, high", "Threshold", "Fresh questions, probe refitted with cross-validation"}

    def T(rr, cols):
        def f(r, c):
            v = r.get(c)
            if v is None or (isinstance(v, float) and v != v):
                return "-"
            if isinstance(v, (bool, np.bool_)):
                return "yes" if v else "no"
            if isinstance(v, str):
                return v
            if c in signed:
                return f"{v:+.1f}"
            if c == "Compute vs one try":
                return f"{100 * v:+.1f}%"
            if c in pct:
                return f"{100 * v:.1f}%"
            if c == "Compute (x cheapest)":
                return f"{v:.2f}x"
            if c in nums:
                return f"{v:.2f}"
            return str(v)
        return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rr)
    fb = lambda r: {"Measure": r["Measure"], "Old 200 questions": "-" if r["Old 200"] is None else f"{100 * r['Old 200']:.1f}%", "Fresh questions": f"{100 * r['Fresh']:.1f}%"}
    h_row = [r for r in sig_rows if r["Signal"] == SIGNALS["hidden"] and r["What is predicted"].startswith("the first")][0]
    verdict = "**Yes.**" if P_["Beats a random gate (interval excludes zero)"] else "**No.**"
    why = (f"The gate retried {100 * P_['Share retried']:.1f}% of the questions and found both passages on {100 * P_['Both passages found']:.1f}%; a random gate retrying the same share "
           f"would find them on {100 * P_['Random gate, same share: both passages found']:.1f}%. That is {P_['Gate minus random: both passages found (points)']:+.1f} points "
           f"(95% interval {P_['95% interval, low (points)']:+.1f} to {P_['95% interval, high (points)']:+.1f}). "
           f"Separation of 'the first try finds both passages': {h_row['Old questions (held-out)']:.2f} on the old questions, {h_row['Fresh questions']:.2f} on the fresh ones "
           f"(95% interval {h_row['Fresh: 95% interval, low']:.2f} to {h_row['Fresh: 95% interval, high']:.2f}).")
    cols = ["Policy", "Share retried", "Both passages found", "Random gate, same share: both passages found", "Gate minus random: both passages found (points)",
            "95% interval, low (points)", "95% interval, high (points)", "Accuracy", "Compute (x cheapest)", "Accuracy vs one try (points)", "Compute vs one try"]
    L = ["# Set 5e - does the retriever's hidden state tell us when one retrieval try is not enough?\n",
         f"Evaluation code version {VERSION}. Frozen on {meta['frozen_on']} at {meta['created']}; checksum `{meta['npz_sha256'][:16]}`. {n} fresh questions. "
         "Nothing was fitted or chosen on them, except the table marked exploratory.\n",
         "## 1. Does the frozen gate beat a random gate?\n", f"The test, fixed in advance: {meta['test']}.\n", f"{verdict} {why}\n",
         f"The gate, in words: {meta['rule in words']}\n", T(pol, cols),
         f"\nAccuracy of the primary gate against a random gate with the same share: {P_['Gate minus random: accuracy (points)']:+.1f} points "
         f"(95% interval {P_['Accuracy 95% interval, low (points)']:+.1f} to {P_['Accuracy 95% interval, high (points)']:+.1f}). "
         f"Retrying raised accuracy on {int(helps.sum())} questions and lowered it on {int(hurts.sum())}; it recovered the missing passage on {int(gained.sum())}.\n",
         "## 2. How much is there to win? Old questions against fresh questions\n", T([fb(r) for r in base], ["Measure", "Old 200 questions", "Fresh questions"]),
         "\n## 3. Do the frozen signals separate on fresh questions? (0.5 = useless)\n",
         T(sig_rows, ["Signal", "Layer", "What is predicted", "Old questions (held-out)", "Fresh questions", "Fresh: 95% interval, low", "Fresh: 95% interval, high"]),
         "\n## 4. Exploratory, not part of the test: the probe refitted on the fresh questions, every layer\n",
         "This shows whether the signal exists in the fresh hidden states at all, apart from whether the frozen probe carried over.\n",
         T(refit, ["Layer", "Old questions (held-out)", "Fresh questions, probe refitted with cross-validation"]),
         "\n## 5. The whole frozen curve of the hidden-state signal\n",
         T([r for r in curve if r["Signal"] == SIGNALS["hidden"]],
           ["Planned share", "Share retried", "Both passages found", "Random gate, same share: both passages found", "Gate minus random: both passages found (points)",
            "Accuracy", "Random gate, same share: accuracy", "Compute (x cheapest)", "Old questions, held-out: both passages found"]),
         "\nCharts: `results_retriever/charts/`. Excel: `results_retriever/set5e_retriever_gate.xlsx`. Every question: `results_retriever/per_question.jsonl`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")

    def xl(rr):
        d = pd.DataFrame(rr)
        for col in d.columns:
            if d[col].isna().all():
                d[col] = ""
        return d
    W.write_excel(os.path.join(out, "set5e_retriever_gate.xlsx"), {"Policies": xl(pol), "Old vs fresh": xl(base), "Signals": xl(sig_rows), "Refit by layer": xl(refit),
                                                                    "Frozen curve": xl(curve), "Per question": xl(perq)})
    log(("BEATS RANDOM" if P_["Beats a random gate (interval excludes zero)"] else "DOES NOT BEAT RANDOM") + " - " + why)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


def charts(out, meta, pol, curve, sc, y, FZ, A, ref, refit):
    plt = S._plt()
    C = S.SERIES
    pr = meta["primary"]
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.6))
    for ax, key, rkey, lab, lo_, hi_ in ((axs[0], "Both passages found", "Random gate, same share: both passages found", "both passages found (%)", A["f1"].mean(), A["f3"].mean()),
                                         (axs[1], "Accuracy", "Random gate, same share: accuracy", "accuracy (%)", A["e1"].mean(), A["e3"].mean())):
        ax.plot([0, 100], [100 * lo_, 100 * hi_], color=S.MUTED, ls="--", lw=1.2, label="random gate")
        for k, (sig, label) in enumerate(SIGNALS.items()):
            h = sorted([r for r in curve if r["Signal"] == label], key=lambda r: r["Share retried"])
            ax.plot([100 * r["Share retried"] for r in h], [100 * r[key] for r in h], color=C[k], marker="o", ms=3, lw=2 if sig == "hidden" else 1,
                    alpha=1 if sig == "hidden" else 0.6, label=label.split(" (")[0])
        ax.scatter([100 * pol[0]["Share retried"]], [100 * pol[0][key]], marker="*", s=300, color=C[0], edgecolor=S.INK, zorder=6, label="primary gate")
        ce = pol[3]
        ax.scatter([100 * ce["Share retried"]], [100 * ce[key]], marker="^", s=80, color=C[2], zorder=6, label="ceiling: retry exactly where try 1 missed")
        ax.set_xlabel("share of questions given two more retrieval tries (%)")
        ax.set_ylabel(lab)
    hd, lb = axs[0].get_legend_handles_labels()
    fig.legend(hd, lb, frameon=False, fontsize=8, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Fresh questions: retrying where the frozen signal says the first try is incomplete, against retrying at random", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "1_gate_vs_random.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.2))
    bins = np.linspace(0, 1, 21)
    for ax, (title, p, yy) in zip([axs[0]], [("Score on the old questions (held-out) and on the fresh ones", None, None)]):
        for k, (nm, pp, tt) in enumerate((("old", FZ.get("old_oof_hidden"), FZ.get("old_f1")), ("fresh", sc["hidden"], y))):
            ax.hist([pp[tt == 1], pp[tt == 0]], bins=bins, histtype="step", lw=1.8, ls="-" if nm == "fresh" else ":", color=[C[0], C[1]],
                    label=[f"{nm}: first try found both", f"{nm}: first try missed a passage"])
        ax.axvline(pr["threshold"], color=S.INK, lw=1.3, label="frozen threshold (retry to its left)")
        ax.set_xlabel("probability that one try finds both passages (frozen probe)")
        ax.set_ylabel("questions")
        ax.set_title(title, fontsize=9)
        ax.legend(frameon=False, fontsize=7)
    ax = axs[1]
    ax.plot([r["Layer"] for r in refit], [r["Old questions (held-out)"] for r in refit], marker="o", color=S.MUTED, label="old questions (held-out)")
    ax.plot([r["Layer"] for r in refit], [r["Fresh questions, probe refitted with cross-validation"] for r in refit], marker="o", color=C[0],
            label="fresh questions, probe refitted (exploratory)")
    ax.axhline(0.5, color=S.INK2, ls=":", lw=1)
    ax.set_ylim(0.35, 0.9)
    ax.set_xlabel("layer of the Qwen 7B (of 28)")
    ax.set_ylabel("separation of 'the first try finds both passages'")
    ax.set_title("Is the signal there at all? Layer by layer", fontsize=9)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "2_scores_and_layers.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("freeze")
    p.add_argument("--sweep_run", required=True)
    p.add_argument("--hidden_run", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--models_json", required=True)
    p.add_argument("--gate_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--repeats", type=int, default=3)
    p = sub.add_parser("evaluate")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_dir", required=True)
    p.add_argument("--models_json", required=True)
    p.add_argument("--frozen_dir", required=True)
    args = ap.parse_args()
    {"freeze": cmd_freeze, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
