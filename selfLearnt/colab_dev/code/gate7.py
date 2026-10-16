"""Set 7c - the answer gate on top of the fine-tuned retriever.

Question: with the fine-tuned Qwen 3B retriever (notebook 23a), can Llama 8B answer most questions and hand only its least-sure
answers to Qwen 32B, keeping most of the new pipeline's accuracy while saving more compute?

Why new model runs are needed: the hidden-state probe must read the exact Llama 8B answer it judges. The new retriever changes
the passages, hence Llama's answers and hidden states; notebook 21 showed that a probe reading a different generation loses its edge.

  extract   Llama 8B answers once (greedy, the earlier prompt, transformers) on the fine-tuned 3B's shortlists, and the hidden
            states of that same pass are saved (same code as notebooks 16-17, attention measurement off).
  analyze   Everything is fitted inside training folds: the probe (PCA 16 + logistic regression, layer 28 of 32, final answer line),
            its training target (Llama's answer is no worse than Qwen's on the same shortlist), and the share of questions to escalate.
            References: always Qwen 32B on the new shortlists (the improved pipeline), the cheap fixed setup (Qwen 7B + Qwen 32B),
            Llama 8B alone on the new shortlists, and a random gate with the same escalation share.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S               # noqa: E402
import sweep as W             # noqa: E402
import hidden as H            # noqa: E402
import final as F             # noqa: E402
import retriever_check as RC  # noqa: E402
import experts as X           # noqa: E402

VERSION = "7c.1"
log = S.log
TOL = 1.0              # predeclared: the 'retain' policy may lose at most 1 point (training estimate) against always-Qwen
SHARES = [round(x, 2) for x in np.arange(0.0, 1.0001, 0.05)]


def test_questions(args):
    out = []
    for name, d in (("old", args.old_data), ("fresh", args.fresh_data)):
        for q in S.load_questions(d):
            q["set"] = name
            out.append(q)
    return out


def load_llama(path):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16)
    except TypeError:
        m = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16)
    return m.to("cuda").eval(), tok


def cmd_extract(args):
    import torch
    H.MAX_NEW = args.max_new
    os.makedirs(os.path.join(args.run_dir, "step"), exist_ok=True)
    qs = test_questions(args)
    sels = {r["qid"]: tuple(r["sel"]) for r in S.read_jsonl(args.ret_file)}
    assert all(q["qid"] in sels for q in qs), "the fine-tuned retriever's shortlists are missing for some test questions"
    npz, jl = H.step_paths(args.run_dir, "l8", "answer")
    rows = S.read_jsonl(jl)
    arrs = dict(np.load(npz)) if os.path.exists(npz) and rows else {}
    done = {r["qid"] for r in rows}
    todo = [q for q in qs if q["qid"] not in done]
    log(f"extract: {len(done)} done, {len(todo)} to do (Llama 8B, shortlists from {args.ret_file})")
    if not todo:
        return
    t0 = time.time()
    if args.backend == "tiny":
        model, tok = H.tiny_model([q["question"] for q in qs] + [p for q in qs[:40] for p in q["ctxs_candidate"]] + [S.qa_prompt("x", ["y"], cot=True) + W.CONF_ANS])
    else:
        model, tok = load_llama(json.load(open(args.models_json))["l8"]["path"])
    log(f"model ready in {time.time() - t0:.0f}s")
    keys = ["h_prompt", "h_answer", "h_answer_line"]
    buf = {k: [arrs[k][i] for i in range(len(rows))] if k in arrs else [] for k in keys}
    lids = [int(x) for x in arrs["layers"]] if "layers" in arrs else None
    t1 = time.time()

    def save():
        with open(jl, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        np.savez_compressed(npz, layers=np.array(lids or []), **{k: np.array(v, np.float16) for k, v in buf.items()})
    for i, q in enumerate(todo):
        row, hp, ha, ht, _, lids = H.extract_answer_one(model, tok, q, sels[q["qid"]], want_attention=False)
        row.update({"model": "l8", "set": q["set"], "lenient": X.lenient(row["prediction"], q["answer"])})
        buf["h_prompt"].append(hp); buf["h_answer"].append(ha); buf["h_answer_line"].append(ht)
        rows.append(row)
        if torch.cuda.is_available() and i % 20 == 0:
            torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or i + 1 == len(todo):
            save()
            log(f"  {len(rows)}/{len(qs)} ({(time.time() - t1) / (i + 1):.1f}s per question)")
        if args.time_limit_min and (time.time() - t0) / 60 > args.time_limit_min:
            save()
            log("time limit reached - run again to continue")
            return
    log(f"extraction complete in {(time.time() - t0) / 60:.0f} min")


# ==================================================================================
# Analysis
# ==================================================================================
def boot(a, b, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    d = [a[w].mean() - b[w].mean() for w in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))


def probe(Htr, ytr, Hte):
    red = F.fit_reduce(Htr)
    return H.f_linear(F.apply_reduce(red, Htr), ytr.astype(int), F.apply_reduce(red, Hte))


def oof(Hm, y, seed):
    from sklearn.model_selection import StratifiedKFold
    out = np.zeros(len(y))
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(np.zeros(len(y)), y):
        out[b] = probe(Hm[a], y[a], Hm[b])
    return out


def gate(score, thr_scores, share):
    """Escalate the questions whose score is at or below the training quantile for this share."""
    if share <= 0:
        return np.zeros(len(score), bool)
    if share >= 1:
        return np.ones(len(score), bool)
    return score <= np.quantile(thr_scores, share)


def cmd_analyze(args):
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold
    out = os.path.join(args.run_dir, "results")
    os.makedirs(os.path.join(out, "charts"), exist_ok=True)
    models = W.load_models(args.models_json)
    qs = test_questions(args)
    qids = [q["qid"] for q in qs]
    n = len(qs)
    origin = np.array([q["set"] for q in qs])
    E = H.load_extract(args.run_dir, "l8", "answer", qids)
    assert E is not None and E["have"].all(), "Llama 8B hidden states are missing for some questions - run `extract` first"
    li = len(E["layers"]) - 2                                     # layer 28 of 32, as in notebooks 16-21
    Hm = E["h_answer_line"][:, li, :]
    rows = E["rows"]
    ft = {r["qid"]: r for r in S.read_jsonl(args.ret_file)}
    A = {}
    for d in [args.answers_run] + args.extra_answers:
        for r in S.read_jsonl(os.path.join(d, "step", "answers_q32.jsonl")):
            A[(r["qid"], tuple(r["sel"]))] = r
    # the fixed setup's shortlist (Qwen 7B, one try) and its compute
    ref_sel, ref_rc = {}, {}
    for run, d in ((args.old_sweep, args.old_data), (args.fresh_run, args.fresh_data)):
        rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RC.RET, RC.COT, RC.PICK)))}
        for q in S.load_questions(d):
            row = rets[q["qid"]]
            ref_sel[q["qid"]] = tuple(W.shortlists(row, RC.COT, len(q["ctxs_candidate"]))[(1, RC.MERGE)])
            ref_rc[q["qid"]] = 2 * models[RC.RET]["params"] * (row["prompt_tokens"] + row["samples"][0]["out_tokens"])
    missing = [q for q in qids if (q, tuple(ft[q]["sel"])) not in A or (q, ref_sel[q]) not in A]
    assert not missing, f"Qwen 32B answers missing for {len(missing)} questions"
    P8 = models["l8"]["params"]
    c_ret = np.array([ft[q]["flops"] for q in qids])
    c_l8 = np.array([2 * P8 * (r["prompt_tokens"] + r["out_tokens"]) for r in rows])
    c_q = np.array([A[(q, tuple(ft[q]["sel"]))]["flops"] for q in qids])
    c_fix = np.array([ref_rc[q] + A[(q, ref_sel[q])]["flops"] for q in qids])
    res, curves, summ = [], [], []
    for metric in ("lenient", "EM"):
        yl = np.array([float(r[metric]) for r in rows])
        yq = np.array([float(A[(q, tuple(ft[q]["sel"]))][metric]) for q in qids])
        yf = np.array([float(A[(q, ref_sel[q])][metric]) for q in qids])
        target = (yl >= yq).astype(int)                 # 1 = keeping Llama's answer is no worse
        cq = c_ret + c_q                                  # always Qwen on the new shortlists
        summ.append({"Metric": metric, "Llama 8B alone (new shortlists)": yl.mean(), "Always Qwen 32B (new shortlists)": yq.mean(), "Cheap fixed setup": yf.mean(),
                     "Llama right, Qwen wrong": int(((yl == 1) & (yq == 0)).sum()), "Qwen right, Llama wrong": int(((yq == 1) & (yl == 0)).sum()),
                     "Share where keeping Llama is no worse": target.mean()})
        evals = {"all 400, cross-validated": [], "old -> fresh": [(np.where(origin == "old")[0], np.where(origin == "fresh")[0])],
                 "fresh -> old": [(np.where(origin == "fresh")[0], np.where(origin == "old")[0])]}
        strat = np.array([f"{o}{t}" for o, t in zip(origin, target)])
        for r in range(args.repeats):
            evals["all 400, cross-validated"] += list(StratifiedKFold(5, shuffle=True, random_state=300 + r).split(np.zeros(n), strat))
        for ev, folds in evals.items():
            acc = {k: np.zeros(n) for k in ("retain", "s30", "s40", "s50")}
            esc = {k: np.zeros(n) for k in acc}
            cur = {s: [np.zeros(n), np.zeros(n)] for s in SHARES}
            sep, cnt, chosen = [], np.zeros(n), []
            for f_i, (tr, te) in enumerate(folds):
                o = oof(Hm[tr], target[tr], seed=f_i)
                st = probe(Hm[tr], target[tr], Hm[te])
                sep.append(H.auc(target[te], st))
                # 'retain': the smallest share whose training accuracy is within TOL points of always-Qwen (on training questions)
                ok = [s for s in SHARES if 100 * (np.where(gate(o, o, s), yq[tr], yl[tr]).mean() - yq[tr].mean()) >= -TOL]
                s_ret = min(ok)
                chosen.append(s_ret)
                cnt[te] += 1
                for k, s in (("retain", s_ret), ("s30", 0.30), ("s40", 0.40), ("s50", 0.50)):
                    e = gate(st, o, s)
                    acc[k][te] += np.where(e, yq[te], yl[te])
                    esc[k][te] += e
                for s in SHARES:
                    e = gate(st, o, s)
                    cur[s][0][te] += np.where(e, yq[te], yl[te])
                    cur[s][1][te] += e
            idx = np.where(cnt > 0)[0]
            m_ = lambda v: v[idx] / cnt[idx]
            for k, lab in (("retain", f"Gate, share chosen in training to stay within {TOL:.0f} point of always-Qwen"),
                           ("s30", "Gate, 30% escalated"), ("s40", "Gate, 40% escalated"), ("s50", "Gate, 50% escalated")):
                y, e = m_(acc[k]), m_(esc[k])
                c = c_ret[idx] + c_l8[idx] + e * c_q[idx]
                s = e.mean()
                rnd = s * yq[idx] + (1 - s) * yl[idx]
                lo_q, hi_q = boot(y, yq[idx])
                lo_f, hi_f = boot(y, yf[idx], seed=1)
                lo_r, hi_r = boot(y, rnd, seed=2)
                res.append({"Metric": metric, "Evaluation": ev, "Policy": lab, "Share escalated": float(s), "Accuracy": float(y.mean()),
                            "Always Qwen (new shortlists)": float(yq[idx].mean()), "vs always Qwen (points)": 100 * float(y.mean() - yq[idx].mean()), "q low": lo_q, "q high": hi_q,
                            "Compute vs always Qwen": float(c.sum() / cq[idx].sum() - 1),
                            "Cheap fixed setup": float(yf[idx].mean()), "vs cheap fixed setup (points)": 100 * float(y.mean() - yf[idx].mean()), "f low": lo_f, "f high": hi_f,
                            "Compute vs cheap fixed setup": float(c.sum() / c_fix[idx].sum() - 1),
                            "vs random gate, same share (points)": 100 * float(y.mean() - rnd.mean()), "r low": lo_r, "r high": hi_r,
                            "Separation (0.5 = useless)": float(np.nanmean(sep)), "Shares chosen in training": ", ".join(f"{x:.2f}" for x in sorted(set(chosen))) if k == "retain" else ""})
            for s in SHARES:
                y, e = m_(cur[s][0]), m_(cur[s][1])
                c = c_ret[idx] + c_l8[idx] + e * c_q[idx]
                curves.append({"Metric": metric, "Evaluation": ev, "Planned share": s, "Accuracy": float(y.mean()),
                               "Compute vs always Qwen": float(c.sum() / cq[idx].sum() - 1), "Compute vs cheap fixed setup": float(c.sum() / c_fix[idx].sum() - 1),
                               "Random gate, same share": float(e.mean() * yq[idx].mean() + (1 - e.mean()) * yl[idx].mean())})
        refc = {"Llama 8B alone": float((c_ret + c_l8).sum() / cq.sum() - 1), "Always Qwen": 0.0, "fixed": float(c_fix.sum() / cq.sum() - 1)}
        summ[-1].update({"Compute of Llama alone vs always Qwen": refc["Llama 8B alone"], "Compute of the cheap fixed setup vs always Qwen": refc["fixed"]})
        log(f"{metric} done")
    with open(os.path.join(out, "gate_policies.jsonl"), "w") as f:
        for r in res:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(out, "gate_curve.jsonl"), "w") as f:
        for r in curves:
            f.write(json.dumps(r) + "\n")
    W.write_excel(os.path.join(out, "set7c_answer_gate.xlsx"), {"Policies": pd.DataFrame(res), "Curve": pd.DataFrame(curves), "Baselines": pd.DataFrame(summ)})
    summary(out, res, summ, rows)
    chart(out, curves, summ)
    log(f"results -> {out}")


def summary(out, res, summ, rows):
    def T(rr, cols, pct=(), sg=()):
        f = lambda r, c: "-" if r.get(c) is None else (r[c] if isinstance(r[c], str) else (str(r[c]) if isinstance(r[c], int) else
                                                      (f"{100 * r[c]:.1f}%" if c in pct else (f"{r[c]:+.1f}" if c in sg else f"{r[c]:.2f}"))))
        return "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join("| " + " | ".join(f(r, c) for c in cols) + " |" for r in rr)
    L = ["# Set 7c: answer gate on top of the fine-tuned Qwen 3B retriever\n",
         f"Code version {VERSION}. 400 test questions. Llama 8B answers once on the fine-tuned 3B's shortlists; the probe reads the hidden state of "
         "that same answer. Probe, training target and escalation share are fitted inside training folds (5-fold cross-validation, repeated). "
         f"Answers cut off at the 512-token limit: {sum(r['hit_cap'] for r in rows)}.\n",
         "## Baselines\n",
         T(summ, ["Metric", "Llama 8B alone (new shortlists)", "Always Qwen 32B (new shortlists)", "Cheap fixed setup", "Llama right, Qwen wrong", "Qwen right, Llama wrong",
                  "Compute of Llama alone vs always Qwen", "Compute of the cheap fixed setup vs always Qwen"],
           pct={"Llama 8B alone (new shortlists)", "Always Qwen 32B (new shortlists)", "Cheap fixed setup", "Compute of Llama alone vs always Qwen",
                "Compute of the cheap fixed setup vs always Qwen"}), ""]
    for metric in ("lenient", "EM"):
        for ev in ("all 400, cross-validated", "old -> fresh", "fresh -> old"):
            rr = [r for r in res if r["Metric"] == metric and r["Evaluation"] == ev]
            L += [f"## {'Corrected label' if metric == 'lenient' else 'Exact match'} | {ev}\n",
                  T(rr, ["Policy", "Share escalated", "Accuracy", "vs always Qwen (points)", "q low", "q high", "Compute vs always Qwen", "vs cheap fixed setup (points)",
                         "f low", "f high", "Compute vs cheap fixed setup", "vs random gate, same share (points)", "r low", "r high", "Separation (0.5 = useless)",
                         "Shares chosen in training"],
                    pct={"Share escalated", "Accuracy", "Compute vs always Qwen", "Compute vs cheap fixed setup"},
                    sg={"vs always Qwen (points)", "q low", "q high", "vs cheap fixed setup (points)", "f low", "f high", "vs random gate, same share (points)", "r low", "r high"}), ""]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")


def chart(out, curves, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, metric in zip(axes, ("lenient", "EM")):
        rr = sorted([r for r in curves if r["Metric"] == metric and r["Evaluation"] == "all 400, cross-validated"], key=lambda r: r["Planned share"])
        x = [100 * (1 + r["Compute vs always Qwen"]) for r in rr]
        ax.plot(x, [100 * r["Accuracy"] for r in rr], "o-", color="#1f6feb", ms=4, label="hidden-state gate (share 0 to 100%)")
        ax.plot(x, [100 * r["Random gate, same share"] for r in rr], "--", color="#888888", label="random gate, same share")
        s = [r for r in summ if r["Metric"] == metric][0]
        ax.axhline(100 * s["Always Qwen 32B (new shortlists)"], color="#2da44e", lw=1, label="always Qwen 32B, new shortlists")
        ax.axhline(100 * s["Cheap fixed setup"], color="#cf222e", lw=1, ls=":", label="cheap fixed setup (AgentTTS pick)")
        ax.set_xlabel("compute, % of always-Qwen on the new shortlists")
        ax.set_ylabel(f"accuracy, % ({'corrected label' if metric == 'lenient' else 'exact match'})")
        ax.set_title("corrected label" if metric == "lenient" else "exact match", fontsize=10)
    axes[0].legend(fontsize=7)
    fig.suptitle("Answer gate on the fine-tuned 3B retriever: all 400 questions, cross-validated", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "charts", "gate_curve.png"), dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("extract", "analyze"):
        p = sub.add_parser(c)
        for a in ("old_data", "fresh_data", "run_dir", "models_json", "ret_file"):
            p.add_argument("--" + a, required=True)
        if c == "extract":
            p.add_argument("--backend", choices=["hf", "tiny"], default="hf")
            p.add_argument("--max_new", type=int, default=512)
            p.add_argument("--time_limit_min", type=float, default=0)
        else:
            for a in ("old_sweep", "fresh_run", "answers_run"):
                p.add_argument("--" + a, required=True)
            p.add_argument("--extra_answers", nargs="*", default=[])
            p.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()
    {"extract": cmd_extract, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
