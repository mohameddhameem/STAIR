"""Experiment 11: the System One judge on MedAgentBench, by replay of Jonathan's recorded episodes
(docs/experiment-11-plan.md).

Jonathan's MedAgentBench campaign (STAIR branch `jonathan`, lab_package/medagentbench/) recorded a weak
(Qwen2.5-7B-Instruct) and a strong (Qwen2.5-72B-Instruct-AWQ) agent episode for each of the 300 tasks, and routes
weak-first: the weak attempt always runs, then a router delivers it or escalates to the strong model. His replay table
(`training1_table.json`, one row per task) holds the router's view (the instruction, then a summary of the weak attempt:
its final answer, capped at 400 characters, or that it did not finish), each model's pass/fail (`qw`, `qs`) and its
units (`cw`, `cs`). So a judge is trained and tested on the table alone; no episode runs.

    label       escalate when only the strong model succeeds (qw = 0, qs = 1), pair A's large-helps. A task neither
                model solves is delivered, at the lower cost
    accounting  his: deliver costs cw, escalate cw + cs. A judge call costs stair_common units at its measured tokens
    split       his, patient-disjoint: train = calib minus calib_sweep (90), val = calib_sweep (60), test (150)

A judge reads the task as its question and the weak attempt as its state, and gives P(escalate):

    zero-shot   "Is the proposed answer correct? Task: ..." {yes, no}: P(escalate) = P(no)
    fine-tuned  "Which model should answer this question: ..." {small model, large model}, on the train tasks, the
                epoch with the lowest val NLL kept (judge.fine_tune)

Three baselines score too, so they sweep like judges: whether the weak model finished (escalate if not); the task
category's escalation rate on train (the template lookup his analysis predicts: each category has one instruction
template); and that rate per (category, finished).

    exp11-mab check --plan configs/exp11-mab.toml                    # the table and the split (CPU, no model)
    exp11-mab zeroshot --plan configs/exp11-mab.toml                 # every zero-shot judge, on all 300 views
    exp11-mab train --judge nano-jev --plan configs/exp11-mab.toml   # one judge set, every seed; resumes
    exp11-mab select --plan configs/exp11-mab.toml                   # operating points from val (CPU)
    exp11-mab test --plan configs/exp11-mab.toml                     # the one test read (CPU); refuses a second

`select` writes <out>/select.json: for every router (each zero-shot judge, each seed, the mean of a set's seeds, the two
baselines), point A is the most accurate val threshold (the cheapest on ties) and point B the cheapest at or above
always-escalate's val accuracy. It reads no test labels, and refuses to run once the test has been read. `test` applies
those points to the test tasks and reports, for each router:
- accuracy, mean units, the share of always-escalate's units, and his reward R';
- AIQ over the test curve (every threshold), from always-deliver's units to always-escalate's, with each judge's own
  price and without;
- paired bootstrap intervals that resample patients, not tasks.

His routers' test decisions, when the plan names them, are compared with ours point to point.
"""

import argparse
import gc
import hashlib
import json
import logging
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from stair_common.cost import units
from stair_common.splits import load_split

from stair import exp8_eval as ev
from stair import provenance
from stair.evaluate_routing import aiq
from stair.judge import ROUTE, fine_tune, judge_params_b, make_judge, short_name

log = logging.getLogger(__name__)

CORRECT = ("Is the proposed answer correct? Task: {task}", ("yes", "no"))  # option 1 (no) = escalate
ATTEMPT = "\n\nThe weak model "  # where his view's summary of the weak attempt starts
SPLIT_SIZES = {"train": 90, "val": 60, "test": 150}
PER_CATEGORY = {"train": 9, "val": 6, "test": 15}  # tasks of each of the 10 categories


def split_view(view: str) -> tuple[str, str]:
    """(task, weak attempt) of his view: the instruction (and context), then the attempt's summary."""
    task, sep, rest = view.rpartition(ATTEMPT)
    if not sep:
        raise ValueError("a view ends with the weak attempt's summary, 'The weak model ...'")
    return task, sep.lstrip() + rest


def category(tid: str) -> str:
    return tid.rsplit("_", 1)[0]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def committed_split() -> dict[str, set[str]] | None:
    """His split as stair_common commits it (0.5.0, with calib_sweep), or None for an older stair_common."""
    try:
        sweep, calib = set(load_split("medagentbench", "calib_sweep")), set(load_split("medagentbench", "calib"))
    except FileNotFoundError:
        return None
    return {"train": calib - sweep, "val": sweep, "test": set(load_split("medagentbench", "test"))}


def check_split(df: pd.DataFrame) -> None:
    sizes = df["split"].value_counts().to_dict()
    if sizes != SPLIT_SIZES:
        raise ValueError(f"split sizes {sizes}, expected {SPLIT_SIZES}")
    for split, n in PER_CATEGORY.items():
        per = df.loc[df["split"] == split, "category"].value_counts()
        if len(per) != 10 or (per != n).any():
            raise ValueError(f"{split} must hold {n} tasks of each of the 10 categories")
    shared = set(df.loc[df["split"] != "test", "patient"]) & set(df.loc[df["split"] == "test", "patient"])
    if shared:
        raise ValueError(f"{len(shared)} patients have tasks in both test and train / val")
    committed = committed_split()
    if committed and any(set(df.loc[df["split"] == s, "tid"]) != ids for s, ids in committed.items()):
        raise ValueError("the table's split differs from stair_common's committed MedAgentBench split")


def load(plan: dict) -> pd.DataFrame:
    """His replay table, checked, with each task's text, weak attempt, category, patient and label."""
    table, tasks = Path(plan["table"]), Path(plan["tasks"])
    for path, key in ((table, "table_sha256"), (tasks, "tasks_sha256")):
        if plan.get(key) and _sha256(path) != plan[key]:
            raise ValueError(f"{path}: sha256 differs from the plan's {key}")
    df = pd.DataFrame(json.loads(table.read_text(encoding="utf-8")))
    patient = {t["id"]: t.get("eval_MRN") for t in json.loads(tasks.read_text(encoding="utf-8"))}
    if len(df) != 300 or df["tid"].nunique() != 300 or set(df["tid"]) != set(patient):
        raise ValueError("the table must hold each of MedAgentBench's 300 tasks once")
    parts = df["view"].map(split_view)
    df["task"], df["attempt"] = parts.str[0], parts.str[1]
    df["finished"] = df["attempt"].str.startswith("The weak model finished")
    df["category"] = df["tid"].map(category)
    df["patient"] = [patient[t] or f"none:{t}" for t in df["tid"]]  # two task1 items name no patient: one group each
    df["y"] = ((df["qw"] == 0) & (df["qs"] == 1)).astype(int)
    check_split(df)
    return df.sort_values("idx").reset_index(drop=True)


def correct_items(df: pd.DataFrame) -> list[dict]:
    question, options = CORRECT
    return [{"question": question.format(task=t), "options": options, "state": a}
            for t, a in zip(df["task"], df["attempt"])]


def route_items(df: pd.DataFrame) -> list[dict]:
    question, options = ROUTE
    return [{"question": question.format(question=t), "options": options, "state": a}
            for t, a in zip(df["task"], df["attempt"])]


def judge_units(params_b: float, tokens) -> np.ndarray:
    """A judge call's units: one forward pass over its tokens, nothing generated."""
    return np.array([units(params_b, int(t), 0) for t in tokens])


def reward(q: np.ndarray, cost: np.ndarray, cs: np.ndarray, lam: float) -> np.ndarray:
    """His R' per task: Q - lambda * x - max(0, x - 1), x = the policy's units over the strong episode's (cs)."""
    x = cost / cs
    return q - lam * x - np.maximum(0.0, x - 1.0)


# ---------------------------------------------------------------------------------------------
# Scoring


def zeroshot_dir(plan: dict, init: str) -> Path:
    return Path(plan["out"]) / "zeroshot" / short_name(init)


def seed_dir(plan: dict, spec: dict, seed: int) -> Path:
    return Path(plan["out"]) / spec["judge_dir"] / f"seed{seed}"


def run_zeroshot(df: pd.DataFrame, plan: dict, device: str | None) -> None:
    for init in plan.get("zero_shot", []):
        target = zeroshot_dir(plan, init)
        if (target / "done.json").exists():
            log.info(f"zero-shot {init}: done")
            continue
        judge = make_judge(init, device, plan.get("max_length", 512))
        start = time.perf_counter()
        p, tokens = judge.probabilities(correct_items(df))
        target.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"tid": df["tid"], "p_escalate": p[:, 1], "tokens": tokens}).to_parquet(target / "scores.parquet",
                                                                                             index=False)
        (target / "done.json").write_text(json.dumps({
            "init": init, "question": CORRECT[0], "seconds": time.perf_counter() - start,
            "init_revision": provenance.hub_revision(init), "device": judge.device}, indent=1))
        log.info(f"zero-shot {init}: mean P(escalate) {p[:, 1].mean():.3f}")
        del judge
        gc.collect()


def run_train(df: pd.DataFrame, plan: dict, spec: dict, seed: int, device: str | None,
              save_model: bool = False) -> None:
    """One judge seed: fine-tuned on the train tasks, the epoch picked on val NLL; scores the val and test tasks."""
    target = seed_dir(plan, spec, seed)
    if (target / "done.json").exists():
        log.info(f"{spec['name']} seed {seed}: done")
        return
    hp = spec["train"]
    train, val = (df[df["split"] == s].reset_index(drop=True) for s in ("train", "val"))
    log.info(f"{spec['name']} seed {seed}: {len(train)} tasks ({int(train['y'].sum())} escalate), {len(val)} val")
    judge = make_judge(spec["init"], device, hp.get("max_length", 512), spec.get("score_batch"))
    start = time.perf_counter()
    history = fine_tune(judge, route_items(train), train["y"].to_numpy(), route_items(val), val["y"].to_numpy(),
                        hp["epochs"], hp["lr"], hp["weight_decay"], hp["warmup"], hp["batch_size"], seed,
                        hp.get("micro_batch"), hp.get("min_steps", 0))
    train_seconds = time.perf_counter() - start
    rest = df[df["split"] != "train"].reset_index(drop=True)
    p, tokens = judge.probabilities(route_items(rest))
    target.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"tid": rest["tid"], "p_escalate": p[:, 1], "tokens": tokens}).to_parquet(target / "scores.parquet",
                                                                                           index=False)
    if save_model:
        judge.model.save_pretrained(target / "model")
        judge.tokenizer.save_pretrained(target / "model")
    (target / "done.json").write_text(json.dumps({
        "judge": spec["name"], "seed": seed, "init": spec["init"], "train": hp, "n_train": len(train),
        "n_val": len(val), "history": history, "train_seconds": train_seconds,
        "init_revision": provenance.hub_revision(spec["init"]), "device": judge.device}, indent=1))


# ---------------------------------------------------------------------------------------------
# Routers and operating points


class Router:
    """P(escalate) per task (indexed by tid), and the router's own units per task (0 for a baseline)."""

    def __init__(self, name: str, scores: pd.Series, price: pd.Series | None = None):
        self.name, self.scores = name, scores
        self.price = price if price is not None else pd.Series(0.0, index=scores.index)

    def on(self, part: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        return self.scores.loc[part["tid"]].to_numpy(), self.price.loc[part["tid"]].to_numpy()


def baselines(df: pd.DataFrame) -> list[Router]:
    """Whether the weak model finished (escalate if not), the category's escalation rate on train, and the rate per
    (category, finished), falling back to the category's where train has no task in that cell."""
    train = df[df["split"] == "train"]
    by_cat = train.groupby("category")["y"].mean()
    by_cell = train.groupby(["category", "finished"])["y"].mean()
    cell = [by_cell.get((c, f), by_cat[c]) for c, f in zip(df["category"], df["finished"])]
    return [Router("finished", pd.Series((~df["finished"]).astype(float).to_numpy(), index=df["tid"])),
            Router("category", pd.Series(df["category"].map(by_cat).to_numpy(), index=df["tid"])),
            Router("category x finished", pd.Series(cell, index=df["tid"], dtype=float))]


def judge_routers(plan: dict) -> list[Router]:
    """Each zero-shot judge, each fine-tuned seed, and the mean of a set's seeds (which pays every seed's calls)."""
    routers = []
    for init in plan.get("zero_shot", []):
        path = zeroshot_dir(plan, init) / "scores.parquet"
        if path.exists():
            s = pd.read_parquet(path).set_index("tid")
            routers.append(Router(f"zero-shot {short_name(init)}", s["p_escalate"],
                                  pd.Series(judge_units(judge_params_b(init), s["tokens"]), index=s.index)))
    for spec in plan.get("judges", []):
        params, seeds = judge_params_b(spec["init"]), []
        for seed in spec["seeds"]:
            path = seed_dir(plan, spec, seed) / "scores.parquet"
            if path.exists():
                s = pd.read_parquet(path).set_index("tid")
                seeds.append(Router(f"{spec['name']} seed{seed}", s["p_escalate"],
                                    pd.Series(judge_units(params, s["tokens"]), index=s.index)))
        routers += seeds
        if len(seeds) > 1:
            routers.append(Router(spec["name"], sum(r.scores for r in seeds) / len(seeds), sum(r.price for r in seeds)))
    return routers


def outcome(part: pd.DataFrame, escalate: np.ndarray, price: np.ndarray | None = None) -> tuple[float, float]:
    """(accuracy, mean units) of a set of decisions; `price` adds the router's own units."""
    q = np.where(escalate, part["qs"], part["qw"])
    cost = part["cw"].to_numpy() + escalate * part["cs"].to_numpy()
    return float(q.mean()), float((cost + (0 if price is None else price)).mean())


def thresholds(scores: np.ndarray) -> list[float]:
    """Every distinct decision: escalate when the score is at or above t; +inf delivers everything."""
    return sorted(set(np.asarray(scores, dtype=float).tolist())) + [float("inf")]


def operating_points(val: pd.DataFrame, scores: np.ndarray) -> dict:
    """A: the most accurate threshold on val, the cheapest on ties. B: the cheapest at or above always-escalate's val
    accuracy, or None."""
    pts = [(t, *outcome(val, scores >= t)) for t in thresholds(scores)]
    best = max(pts, key=lambda p: (p[1], -p[2]))
    bar = float(val["qs"].mean())
    above = [p for p in pts if p[1] >= bar]
    cheap = min(above, key=lambda p: (p[2], -p[1])) if above else None
    point = lambda p: {"t": p[0], "val_accuracy": p[1], "val_units": p[2]}  # noqa: E731
    return {"A": point(best), "B": point(cheap) if cheap else None}


def run_select(df: pd.DataFrame, plan: dict) -> dict:
    out = Path(plan["out"])
    if (out / "test" / "report.json").exists():
        raise SystemExit("the test has been read: operating points can no longer change")
    val = df[df["split"] == "val"].reset_index(drop=True)
    routers = baselines(df) + judge_routers(plan)
    chosen = {"always_escalate_val_accuracy": float(val["qs"].mean()), "table_sha256": _sha256(Path(plan["table"])),
              "routers": {r.name: operating_points(val, r.on(val)[0]) for r in routers}}
    (out / "select.json").write_text(json.dumps(chosen, indent=1) + "\n", encoding="utf-8")
    show = lambda p: f"t={p['t']:.4f} val {100 * p['val_accuracy']:.1f} at {p['val_units']:.1f}"  # noqa: E731
    for name, e in chosen["routers"].items():
        log.info(f"{name:34s} A {show(e['A'])} | B {show(e['B']) if e['B'] else 'none'}")
    return chosen


# ---------------------------------------------------------------------------------------------
# The test read


def his_decisions(plan: dict, test: pd.DataFrame) -> dict[str, np.ndarray]:
    """His routers' test decisions (a CSV: `tid`, then one 0/1 column per router, 1 = escalate), in test order."""
    if not plan.get("his_decisions"):
        return {}
    d = pd.read_csv(plan["his_decisions"]).set_index("tid").loc[test["tid"]]
    return {f"his {c}": d[c].astype(int).to_numpy().astype(bool) for c in d.columns}


def paired_points(test: pd.DataFrame, a: np.ndarray, b: np.ndarray, n_boot: int, seed: int) -> dict:
    """Accuracy and units of decisions a minus decisions b on the same test tasks, resampling patients."""
    rng, groups = np.random.default_rng(seed), test["patient"].to_numpy()
    q = lambda e: np.where(e, test["qs"], test["qw"]).astype(float)  # noqa: E731
    c = lambda e: test["cw"].to_numpy() + e * test["cs"].to_numpy()  # noqa: E731
    dq, dc = q(a) - q(b), c(a) - c(b)
    reps = [(dq[i].mean(), dc[i].mean()) for i in (ev.resample(len(test), rng, groups) for _ in range(n_boot))]
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]  # noqa: E731
    return {"accuracy": float(dq.mean()), "accuracy_ci95": ci([r[0] for r in reps]),
            "units": float(dc.mean()), "units_ci95": ci([r[1] for r in reps])}


def run_test(df: pd.DataFrame, plan: dict) -> dict:
    out = Path(plan["out"]) / "test"
    if (out / "report.json").exists():
        raise SystemExit(f"{out / 'report.json'} exists: the test is read once")
    chosen = json.loads((Path(plan["out"]) / "select.json").read_text(encoding="utf-8"))
    if chosen["table_sha256"] != _sha256(Path(plan["table"])):
        raise SystemExit("the table changed since the operating points were selected")
    test = df[df["split"] == "test"].reset_index(drop=True)
    routers = {r.name: r for r in baselines(df) + judge_routers(plan)}
    missing = sorted(set(chosen["routers"]) - set(routers))
    if missing:
        raise SystemExit(f"selected routers without scores: {missing}")
    lam, n, n_boot = plan.get("reward_lambda", 0.10), len(test), plan.get("bootstrap", 5000)
    deliver, escalate = np.zeros(n, dtype=bool), np.ones(n, dtype=bool)
    oracle = test["y"].to_numpy().astype(bool)
    qw, qs, cw, cs = (test[k].to_numpy() for k in ("qw", "qs", "cw", "cs"))
    lo, hi = outcome(test, deliver)[1], outcome(test, escalate)[1]

    def point(e: np.ndarray, price: np.ndarray | None = None) -> dict:
        acc, cost = outcome(test, e)
        entry = {"accuracy": acc, "units": cost, "share_of_always_escalate": cost / hi,
                 "reward": float(reward(np.where(e, qs, qw), cw + e * cs, cs, lam).mean()), "escalated": int(e.sum())}
        if price is not None:
            entry["units_with_judge"] = outcome(test, e, price)[1]
        return entry

    def curve(full: np.ndarray, price: np.ndarray, ix: np.ndarray) -> list[dict]:
        """(accuracy, mean units) of each row of decisions in `full` on the resampled tasks `ix`."""
        return [{"f1": float(np.where(e[ix], qs[ix], qw[ix]).mean()),
                 "cost": float((cw[ix] + e[ix] * cs[ix] + price[ix]).mean())} for e in full]

    report = {"n_test": n, "units_range": [lo, hi], "reward_lambda": lam,
              "references": {"always deliver": point(deliver), "always escalate": point(escalate),
                             "oracle": point(oracle)},
              "routers": {}}
    zero = np.zeros(n)
    curves = {"random": lambda ix: curve(np.stack([deliver, escalate]), zero, ix)}  # the line between the two
    decisions = {"tid": test["tid"]}
    for name, sel in chosen["routers"].items():
        scores, price = routers[name].on(test)
        entry = {}
        for rule in ("A", "B"):
            if sel[rule] is None:
                entry[rule] = None
                continue
            e = scores >= sel[rule]["t"]
            entry[rule] = {"t": sel[rule]["t"], **point(e, price)}
            decisions[f"{name} {rule}"] = e
        full = np.stack([scores >= t for t in thresholds(scores)])  # [thresholds, tasks]
        curves[name] = lambda ix, f=full: curve(f, zero, ix)
        curves[name + " +judge"] = lambda ix, f=full, p=price: curve(f, p, ix)
        rows = np.arange(n)
        entry["aiq"] = aiq(curves[name](rows), lo, hi)
        entry["aiq_with_judge_price"] = aiq(curves[name + " +judge"](rows), lo, hi)
        report["routers"][name] = entry
    report["aiq_random"] = aiq(curves["random"](np.arange(n)), lo, hi)
    boot = ev.bootstrap(curves, n, n_boot, 0, (lo, hi), groups=test["patient"].to_numpy())
    report["aiq_ci95"] = boot["aiq_ci95"]
    ours = list(chosen["routers"])
    report["aiq_diff_vs_category"] = {f"{a} - category": ev.diff(boot, a, "category") for a in ours if a != "category"}
    report["aiq_diff_contrasts"] = {f"{a} - {b}": ev.diff(boot, a, b) for a, b in plan.get("contrasts", [])
                                    if a in curves and b in curves}
    ft = [s["name"] for s in plan.get("judges", []) if s["name"] in report["routers"]]
    zs = [name for name in ours if name.startswith("zero-shot ")]
    report["gap_closed"] = {}  # H11a: how far a zero-shot judge gets from random's AIQ towards a fine-tuned set's
    for f in ft:
        span = report["routers"][f]["aiq"] - report["aiq_random"]
        for z in zs:
            gain = report["routers"][z]["aiq"] - report["aiq_random"]
            report["gap_closed"][f"{z} / {f}"] = gain / span if span > 1e-9 else None
    his = his_decisions(plan, test)
    report["his"] = {name: point(e) for name, e in his.items()}
    report["ours_A_minus_his"] = {f"{a} A - {h}": paired_points(test, decisions[f"{a} A"], e, n_boot, 0)
                                  for a in ours if f"{a} A" in decisions for h, e in his.items()}
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(decisions).to_parquet(out / "decisions.parquet", index=False)
    (out / "report.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    for name, e in report["routers"].items():
        a = e["A"]
        log.info(f"{name:34s} AIQ {e['aiq']:.4f} | A {100 * a['accuracy']:.2f} at {a['units']:.1f} "
                 f"({100 * a['share_of_always_escalate']:.1f}%)")
    return report


# ---------------------------------------------------------------------------------------------
# CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["check", "zeroshot", "train", "select", "test"])
    parser.add_argument("--plan", type=Path, default=Path("configs/exp11-mab.toml"))
    parser.add_argument("--judge", help="train: the judge set (a [[judges]] name; several, comma-separated)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--save-model", action="store_true", help="train: also save the fine-tuned weights")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = tomllib.loads(args.plan.read_text(encoding="utf-8"))
    df = load(plan)
    out = Path(plan["out"])
    provenance.record(out, f"exp11-mab {args.step}", plan=str(args.plan), judge=args.judge, save_model=args.save_model,
                      **provenance.environment())
    if args.step == "check":
        has_patient = ~df["patient"].str.startswith("none:")  # the two patient-less tasks are one group each, not patients
        counts = df.assign(has_patient=has_patient).groupby("split").agg(
            tasks=("tid", "size"),
            patients=("patient", lambda s: s[has_patient.loc[s.index]].nunique()),
            no_patient_tasks=("has_patient", lambda s: int((~s).sum())),
            escalate=("y", "sum"), weak_right=("qw", "sum"), strong_right=("qs", "sum"))
        log.info(f"{counts}\nsplit is patient-disjoint; table sha256 {_sha256(Path(plan['table']))}")
    elif args.step == "zeroshot":
        run_zeroshot(df, plan, args.device)
    elif args.step == "train":
        if not args.judge:
            raise SystemExit("train needs --judge <name>")
        for spec in [next(j for j in plan["judges"] if j["name"] == name) for name in args.judge.split(",")]:
            for seed in spec["seeds"]:
                run_train(df, plan, spec, seed, args.device, args.save_model)
    elif args.step == "select":
        run_select(df, plan)
    else:
        run_test(df, plan)


if __name__ == "__main__":
    main()
