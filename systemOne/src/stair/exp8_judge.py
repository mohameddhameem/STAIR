"""Experiment 8, phase 2 (docs/experiment-8-plan.md): a System One judge that picks the answer option.

The judge is experiment 7's nano-jev cross-encoder (`judge.Judge`, `judge.fine_tune`), asked one typed
question with one option per answering cell:

    Which answer option should be used for this question: {question}
    options: the cells of the plan, ordered from cheap to expensive

Its state is the question with nano-jev's top-k passages (`qp`, decided before any answer), and with
the cheapest option's proposed answer added (`qpa`). The training label is the cheapest option that
answered correctly (the cheapest option when none did). Its probabilities feed
`exp8_eval.cumulative_choice`, which picks the cheapest option whose cumulative probability reaches a
threshold t, so one judge gives a whole cost-accuracy curve.

    exp8-judge all --plan configs/exp8-judge.toml       # train every (inputs, seed), then evaluate
    exp8-judge train|eval --plan configs/exp8-judge.toml

Training resumes: a (inputs, seed) with a done.json is skipped. Outputs in `<out>/<judge_dir>/` (default `judge`):
`<inputs>/seed<k>/{calib,test}.parquet` (judge probabilities), `done.json` (training curve, timing),
and `eval/report.json`, `eval/curves.csv`.

Experiment 9: `init` may be a Strands Decider (`stair.decider`); `zero_shot = true` scores the checkpoint
as it is, without training; `reference_judges` (a list of `{name, judge_dir, inputs, seeds, init}`) adds
judges from other runs to the report, so the paired AIQ differences compare against them.
"""

import argparse
import json
import logging
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from stair import exp8_eval as ev
from stair import provenance
from stair.evaluate_routing import aiq
from stair.evidence import passages_state
from stair.experts import clean_answer
from stair.judge import fine_tune, judge_params_b, make_judge

log = logging.getLogger(__name__)

QUESTION = "Which answer option should be used for this question: {question}"
UNIT_GFLOPS = 2 * 3.0 * (128 + 64)  # AgentTTS's budget unit (pool_answer.UNIT_GFLOPS)
INPUTS = ("qp", "qpa")


def state(row, inputs: str) -> str:
    text = passages_state(row["passages"])
    return text if inputs == "qp" else f"Proposed answer: {clean_answer(str(row['pred0']))[:200]}\n{text}"


def items_for(df: pd.DataFrame, labels: list[str], inputs: str) -> list[dict]:
    return [{"question": QUESTION.format(question=r["question"]), "options": labels, "state": state(r, inputs)}
            for _, r in df.iterrows()]


def load_all(plan: dict) -> dict[str, tuple]:
    options = [o["cell"] for o in plan["options"]]
    return {s: ev.load_tables(plan.get("out", "runs/exp8"), s, options, plan["state_k"], plan.get("metric", "lenient"),
                              plan.get("rescore_dir"))
            for s in ("router_train", "calib", "test")}


def run_train(tables: dict, plan: dict, inputs: str, seed: int, out: Path, device: str | None) -> None:
    from sklearn.model_selection import train_test_split

    target = out / plan.get("judge_dir", "judge") / inputs / f"seed{seed}"
    if (target / "done.json").exists():
        log.info(f"train {inputs} seed {seed}: done")
        return
    hp, labels = plan.get("train", {}), [o["label"] for o in plan["options"]]
    df, ok, cost = tables["router_train"]
    y = ev.cheapest_correct_label(ok, cost)
    counts = np.bincount(y, minlength=len(labels))
    log.info(f"train {inputs} seed {seed}: {len(df)} questions, label counts {counts.tolist()}")
    tr, dev = train_test_split(np.arange(len(df)), test_size=hp["dev_fraction"], random_state=seed,
                               stratify=y if counts.min() >= 2 else None)
    items = items_for(df, labels, inputs)
    judge = make_judge(plan["init"], device, hp.get("max_length", 512), plan.get("score_batch"))
    start = time.perf_counter()
    if plan.get("zero_shot"):  # the checkpoint's own calibrated choice, no training
        history = []
    else:
        history = fine_tune(judge, [items[i] for i in tr], y[tr], [items[i] for i in dev], y[dev], hp["epochs"],
                            hp["lr"], hp["weight_decay"], hp["warmup"], hp["batch_size"], seed, hp.get("micro_batch"))
    train_seconds = time.perf_counter() - start
    target.mkdir(parents=True, exist_ok=True)
    scoring = {}
    for split in ("calib", "test"):
        d = tables[split][0]
        start = time.perf_counter()
        p, tokens = judge.probabilities(items_for(d, labels, inputs))
        scoring[split] = (time.perf_counter() - start) / len(d)
        pd.DataFrame({"id": d["id"], **{f"p{k}": p[:, k] for k in range(p.shape[1])}, "tokens": tokens}
                     ).to_parquet(target / f"{split}.parquet")
    (target / "done.json").write_text(json.dumps({
        "inputs": inputs, "seed": seed, "init": plan["init"], "zero_shot": bool(plan.get("zero_shot")),
        "options": labels, "train": hp, "n_train": len(tr),
        "n_dev": len(dev), "label_counts": counts.tolist(), "history": history, "train_seconds": train_seconds,
        "scoring_seconds_per_question": scoring, "init_revision": provenance.hub_revision(plan["init"]),
        "device": judge.device}, indent=1))


def judge_probs(root: Path, inputs: str, seeds: list[int], split: str, ids: list[str], init: str = "sdmlai/nano-jev"):
    """Mean probabilities over the seeds, the judge's cost in units per question, and the per-seed arrays."""
    per_seed, tokens = [], None
    for s in seeds:
        d = pd.read_parquet(root / inputs / f"seed{s}" / f"{split}.parquet").set_index("id").loc[ids]
        per_seed.append(d[[c for c in d.columns if c.startswith("p")]].to_numpy())
        tokens = d["tokens"].to_numpy()
    units = 2 * judge_params_b(init) * tokens * len(seeds) / UNIT_GFLOPS  # GFLOPs of every seed's pass, in units
    return np.mean(per_seed, axis=0), units, per_seed


def run_eval(tables: dict, plan: dict, out: Path, exclude: set[str] | None = None, tag: str = "") -> dict:
    seeds, labels = plan["seeds"], [o["label"] for o in plan["options"]]
    root = out / plan.get("judge_dir", "judge")
    (df_c, ok_c, cost_c), (df_t, ok_t, cost_t) = tables["calib"], tables["test"]
    keep = None
    if exclude:
        keep = ~df_t["id"].isin(exclude).to_numpy()
        df_t, ok_t, cost_t = df_t[keep].reset_index(drop=True), ok_t[keep], cost_t[keep]
    n, K = ok_t.shape
    order = np.argsort(cost_c.mean(0))
    if list(order) != list(range(K)):
        log.warning(f"options are not in cost order on calib: {order.tolist()}")
    strong = plan["strong_units"]
    budget = plan["budget_fraction"] * strong
    cmin, cmax = float(cost_t.mean(0).min()), float(cost_t.mean(0).max())
    report = {"n_test": n, "metric": plan.get("metric", "lenient"), "options": labels, "budget_units": budget,
              "strong_units": strong,
              "agenttts": {"lenient": plan["agenttts_lenient"], "units": plan["agenttts_units"]},
              "always": ev.always_points(ok_t, cost_t), "oracle": ev.oracle_point(ok_t, cost_t)}
    curves = {"hull of fixed policies": lambda ix: ev.always_points(ok_t[ix], cost_t[ix])}
    scored = {}  # name -> (calib points, test points function, lo, hi, post)

    def add_score(name, calib_score, test_score, post):
        calib_pts = ev.score_points(ok_c, cost_c, calib_score, 0, K - 1, post)
        curves[name] = lambda ix, s=test_score, p=post: ev.score_points(ok_t[ix], cost_t[ix], s[ix], 0, K - 1, p)
        scored[name] = (calib_pts, ev.score_points(ok_t, cost_t, test_score, 0, K - 1, post))

    add_score("entropy threshold", df_c["entropy0"].to_numpy(), df_t["entropy0"].to_numpy(), True)
    # logistic regression routers on the router_train questions: binary, cheapest versus strongest option
    df_r, ok_r, cost_r = tables["router_train"]
    feat = lambda d: np.stack([ev.evidence_features(s) for s in d["nanojev"]])
    X_r, X_c, X_t = feat(df_r), feat(df_c), feat(df_t)
    y_r = (ok_r[:, -1] & ~ok_r[:, 0]).astype(int)
    for name, cols, post in (("LR evidence (before)", [], False), ("LR evidence+answer (after)", ["entropy0"], True)):
        extra = lambda d, X: np.hstack([X] + [d[[c]].to_numpy() for c in cols]) if cols else X
        model, C = ev.fit_logistic(extra(df_r, X_r), y_r, 0)
        add_score(name, model.predict_proba(extra(df_c, X_c))[:, 1], model.predict_proba(extra(df_t, X_t))[:, 1], post)
        report.setdefault("lr_C", {})[name] = C

    judges = {}
    runs = [(f"judge {i}", root, i, seeds, plan["init"]) for i in plan["inputs"]]
    runs += [(r["name"], out / r["judge_dir"], r["inputs"], r["seeds"], r["init"]) for r in plan.get("reference_judges", [])]
    for name, jroot, inputs, jseeds, init in runs:
        pc, uc, _ = judge_probs(jroot, inputs, jseeds, "calib", df_c["id"].tolist(), init)
        pt, ut, per_seed = judge_probs(jroot, inputs, jseeds, "test", tables["test"][0]["id"].tolist(), init)
        if exclude:
            pt, ut, per_seed = pt[keep], ut[keep], [p[keep] for p in per_seed]
        curves[name] = lambda ix, p=pt, u=ut: ev.judge_points(ok_t[ix], cost_t[ix], p[ix], u[ix])
        scored[name] = (ev.judge_points(ok_c, cost_c, pc, uc), ev.judge_points(ok_t, cost_t, pt, ut))
        judges[name] = {"aiq_per_seed": [aiq(ev.judge_points(ok_t, cost_t, p), cmin, cmax) for p in per_seed],
                        "judge_units": float(ut.mean()), "init": init}

    boot = ev.bootstrap(curves, n, plan.get("bootstrap", 1000), 0, (cmin, cmax))
    report["aiq"] = {"hull of fixed policies": aiq(ev.always_points(ok_t, cost_t), cmin, cmax),
                     "oracle": aiq([report["oracle"], *report["always"]], cmin, cmax)}
    for name, (calib_pts, test_pts) in scored.items():
        report["aiq"][name] = aiq(test_pts, cmin, cmax)
        on_budget = ev.choose_threshold(calib_pts, budget=budget)
        floor = max(p["f1"] for p in calib_pts) - plan["max_drop"]
        near_best = ev.choose_threshold(calib_pts, floor=floor)
        report.setdefault("operating", {})[name] = {
            "budget": _apply(on_budget, calib_pts, test_pts), "near_best": _apply(near_best, calib_pts, test_pts)}
    report["aiq_ci95"] = boot["aiq_ci95"]
    report["aiq_diff"] = {f"{a} - {b}": ev.diff(boot, a, b) for a in judges for b in boot["aiq_ci95"] if b != a}
    report["judges"] = judges
    rows = [{"policy": name, **p} for name, (_, test_pts) in scored.items() for p in test_pts]
    pd.DataFrame(rows).to_csv(root / "eval" / f"curves{tag}.csv", index=False)
    (root / "eval" / f"report{tag}.json").write_text(json.dumps(report, indent=1))
    return report


def _apply(chosen: dict, calib_pts: list[dict], test_pts: list[dict]) -> dict:
    """The test point at the same position (threshold or rate) as the point chosen on calib."""
    i = calib_pts.index(chosen)
    return {"calib": chosen, "test": test_pts[i]}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "eval", "all"])
    parser.add_argument("--plan", type=Path, default=Path("configs/exp8-judge.toml"))
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--exclude", type=Path, help="eval: question ids to drop too, for the clean report")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = tomllib.loads(args.plan.read_text())
    out = Path(plan.get("out", "runs/exp8"))
    root = out / plan.get("judge_dir", "judge")
    (root / "eval").mkdir(parents=True, exist_ok=True)
    tables = load_all(plan)
    if args.step in ("train", "all"):
        provenance.record(root, "exp8-judge", plan=str(args.plan), **provenance.environment())
        for inputs in plan["inputs"]:
            for seed in plan["seeds"]:
                run_train(tables, plan, inputs, seed, out, args.device)
    if args.step in ("eval", "all"):
        run_eval(tables, plan, out)
        if args.exclude:
            run_eval(tables, plan, out, set(args.exclude.read_text().split()), "-clean")


if __name__ == "__main__":
    main()
