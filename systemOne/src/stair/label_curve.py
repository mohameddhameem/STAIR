"""Jev novelty E1 (pre-registration H1): routing AIQ against the routing-label budget, per starting point, head and
inputs, from the judges `judge train` wrote for a label-budget plan (configs/e1-label-curve.toml).

For every (starting point, head, inputs, budget) arm and seed: the AIQ of that seed's test scores on the pair, with the
judge unpriced (nano-jev-size calls price at 0 under table5). An arm's AIQ is the mean over its seeds (one deployed
model). LE-AUC is the area under AIQ against log2(budget), divided by the log2 range: the arm's AIQ averaged over
budgets on a log scale.

Pre-registered tests, for each inputs (qp, qpa), Holm over the eight:
    H1a  LE-AUC(nano-jev v1.0) - LE-AUC(plain MiniLM) > 0                                    (typed heads)
    H1b  [v1.0 - plain] at budget 100 - [v1.0 - plain] at the full budget > 0                (typed heads)
    H1c  (LE(noHotpot) - LE(plain)) - 0.5 x (LE(v1.0) - LE(plain)) > 0                       (typed heads)
    H1d  LE-AUC(v1.0, typed head) - LE-AUC(v1.0, plain head) > 0
Two-level paired bootstrap: test questions resampled (same indices for every arm), then each arm's seeds resampled.
One-sided p = the share of replicates <= 0. Descriptive beside them: every pairwise LE-AUC difference, repro - v1.0
(the retraining noise), and each arm's share of the random-to-oracle gap.

    label-curve --plan configs/e1-label-curve.toml --out runs/e1-label-curve --exclude data/hotpotqa/nanojev_validation_ids.txt
writes <out>/label-curve/{report,report-clean}.json, cells.csv and curves.png (the clean population is the primary one).
"""

import argparse
import json
import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance
from stair.evaluate_routing import CURVE_COLUMNS, aiq, curve, expert_params, random_curve, with_cost_unit
from stair.judge import INPUTS, init_label, judge_name, load_plan
from stair.metrics import metric_view

log = logging.getLogger(__name__)

ROLES = {"v1": "nanojev", "plain": "minilm-l12-h384-uncased", "msmarco": "ms-marco-minilm-l12-v2",
         "repro": "nanojev-repro", "nohotpot": "nanojev-nohotpot"}  # starting point -> init_label
HYPOTHESES = ("H1a", "H1b", "H1c", "H1d")


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-adjusted p-values (step-down, monotone)."""
    order = sorted(pvalues, key=pvalues.get)
    out, running = {}, 0.0
    for k, name in enumerate(order):
        running = max(running, min(1.0, (len(order) - k) * pvalues[name]))
        out[name] = running
    return out


def load_test(plan: dict, pair: str, exclude: set[str]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    """The pair's test arrays in the plan's metric and cost unit, and the test ids (excluded ones dropped)."""
    run = Path(plan["pairs"][pair])
    cfg = tomllib.loads((run / "config.toml").read_text())
    params, models = expert_params(cfg), {e["name"]: e.get("model", "") for e in cfg["experts"]}
    df = with_cost_unit(metric_view(pd.read_parquet(run / "test.parquet"), plan.get("metric", "f1")),
                        plan.get("cost", "flops"), params, models)
    df = df[~df["id"].isin(exclude)].reset_index(drop=True)
    return {c: df[c].to_numpy(dtype=float) for c in CURVE_COLUMNS}, df["id"].to_numpy()


def arm_key(spec: dict) -> tuple[str, str, str, int]:
    return init_label(spec), spec.get("head", "typed"), spec["inputs"], int(spec["budget"])


def load_scores(out: Path, plan: dict, pair: str, ids: np.ndarray) -> dict[tuple, dict[int, np.ndarray]]:
    """(init, head, inputs, budget) -> seed -> test scores in `ids` order, for every finished judge seed."""
    arms: dict[tuple, dict[int, np.ndarray]] = {}
    for spec in plan["judges"]:
        folder = out / "judges" / judge_name(spec)
        for f in sorted(folder.glob(f"seed*/{pair}-test.parquet")):
            s = pd.read_parquet(f).set_index("id")["score"]
            arms.setdefault(arm_key(spec), {})[int(f.parent.name[4:])] = s.loc[ids].to_numpy(dtype=float)
    return arms


def arm_aiqs(a: dict[str, np.ndarray], arms: dict[tuple, dict[int, np.ndarray]], idx: np.ndarray | None = None
             ) -> dict[tuple, dict[int, float]]:
    s = a if idx is None else {k: v[idx] for k, v in a.items()}
    cmin, cmax = s["small_cost"].mean(), s["large_cost"].mean()
    return {key: {seed: aiq(curve(s, sc if idx is None else sc[idx], INPUTS[key[2]]), cmin, cmax)
                  for seed, sc in seeds.items()} for key, seeds in arms.items()}


def le_auc(by_budget: dict[int, float]) -> float:
    b = sorted(by_budget)
    if len(b) < 2:
        return float(by_budget[b[0]])
    x = np.log2(b)
    return float(np.trapezoid([by_budget[k] for k in b], x) / (x[-1] - x[0]))


def statistics(means: dict[tuple, float]) -> dict[str, float]:
    """Every LE-AUC and every pre-registered contrast, from arm means (seed-averaged AIQ per arm)."""
    curves: dict[tuple, dict[int, float]] = {}
    for (init, head, inputs, budget), v in means.items():
        curves.setdefault((init, head, inputs), {})[budget] = v
    out = {f"le_auc/{i}/{h}/{x}": le_auc(c) for (i, h, x), c in curves.items()}
    for x in INPUTS:
        le = lambda role, head="typed": out.get(f"le_auc/{ROLES[role]}/{head}/{x}")  # noqa: E731
        at = lambda role, b: curves.get((ROLES[role], "typed", x), {}).get(b)  # noqa: E731
        full = max((b for (i, h, xx), c in curves.items() if xx == x for b in c), default=None)
        if le("v1") is not None and le("plain") is not None:
            out[f"H1a/{x}"] = le("v1") - le("plain")
            if at("v1", 100) is not None and at("plain", 100) is not None and full:
                out[f"H1b/{x}"] = (at("v1", 100) - at("plain", 100)) - (at("v1", full) - at("plain", full))
            if le("nohotpot") is not None:
                out[f"H1c/{x}"] = (le("nohotpot") - le("plain")) - 0.5 * (le("v1") - le("plain"))
            if le("v1", "plain") is not None:
                out[f"H1d/{x}"] = le("v1") - le("v1", "plain")
            if le("repro") is not None:
                out[f"noise/repro-v1/{x}"] = le("repro") - le("v1")
            if le("msmarco") is not None:
                out[f"msmarco-plain/{x}"] = le("msmarco") - le("plain")
    return out


def seed_means(per_seed: dict[tuple, dict[int, float]], rng: np.random.Generator | None = None) -> dict[tuple, float]:
    """Each arm's mean over its seeds; with `rng`, over a resample of its seeds (the bootstrap's second level)."""
    out = {}
    for key, seeds in per_seed.items():
        v = np.array([seeds[s] for s in sorted(seeds)])
        out[key] = float((v if rng is None else v[rng.integers(0, len(v), len(v))]).mean())
    return out


def analyse(a: dict[str, np.ndarray], arms: dict[tuple, dict[int, np.ndarray]], n_boot: int, seed: int) -> dict:
    point_seeds = arm_aiqs(a, arms)
    point = statistics(seed_means(point_seeds))
    rng = np.random.default_rng(seed)
    n = len(a["small_f1"])
    reps: dict[str, list[float]] = {k: [] for k in point}
    for _ in range(n_boot):
        stats = statistics(seed_means(arm_aiqs(a, arms, rng.integers(0, n, n)), rng))
        for k in reps:
            reps[k].append(stats.get(k, np.nan))
    summary = {k: {"estimate": point[k], "ci95": [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))],
                   "p_le_0": float(np.mean(np.asarray(v) <= 0))} for k, v in reps.items()}
    tests = {k: summary[k]["p_le_0"] for k in summary if k.split("/")[0] in HYPOTHESES}
    for k, p in holm(tests).items():
        summary[k]["p_holm"] = p
        summary[k]["supported"] = bool(p < 0.05)
    cmin, cmax = a["small_cost"].mean(), a["large_cost"].mean()
    rand = aiq(random_curve(a), cmin, cmax)
    oracle_f1 = float(np.maximum(a["small_f1"], a["large_f1"]).mean())
    cells = [{"init": k[0], "head": k[1], "inputs": k[2], "budget": k[3], "n_seeds": len(v),
              "aiq_mean": float(np.mean(list(v.values()))), "aiq_sd": float(np.std(list(v.values()), ddof=1)) if len(v) > 1 else 0.0,
              "gap_closed": float((np.mean(list(v.values())) - rand) / (oracle_f1 - rand))}
             for k, v in sorted(point_seeds.items())]
    return {"n_test": n, "n_boot": n_boot, "random_aiq": rand, "oracle_f1": oracle_f1, "stats": summary, "cells": cells}


def plot(cells: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, inputs in zip(axes, ("qp", "qpa")):
        sub = cells[cells["inputs"] == inputs]
        for (init, head), g in sub.groupby(["init", "head"]):
            g = g.sort_values("budget")
            ax.errorbar(g["budget"], g["aiq_mean"], yerr=g["aiq_sd"], marker="o", capsize=2,
                        linestyle="-" if head == "typed" else "--", label=f"{init} ({head})")
        ax.set_xscale("log", base=2)
        ax.set_title(inputs)
        ax.set_xlabel("routing labels (budget, log scale)")
    axes[0].set_ylabel("AIQ (test)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, default=Path("configs/e1-label-curve.toml"))
    parser.add_argument("--out", type=Path, default=Path("runs/e1-label-curve"))
    parser.add_argument("--pair", default="A")
    parser.add_argument("--exclude", type=Path, action="append", default=[], help="test ids to drop: report-clean.json")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = load_plan(args.plan)
    target = args.out / "label-curve"
    target.mkdir(parents=True, exist_ok=True)
    exclude = set()
    for path in args.exclude:
        exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    for tag, drop in (("", set()), ("-clean", exclude)) if exclude else (("", set()),):
        a, ids = load_test(plan, args.pair, drop)
        arms = load_scores(args.out, plan, args.pair, ids)
        report = {"pair": args.pair, "excluded": len(drop), **analyse(a, arms, args.bootstrap, args.seed)}
        (target / f"report{tag}.json").write_text(json.dumps(report, indent=1))
        cells = pd.DataFrame(report["cells"])
        cells.to_csv(target / f"cells{tag}.csv", index=False)
        if tag == ("-clean" if exclude else ""):
            plot(cells, target / "curves.png")
        log.info(f"label curve{tag}: {len(arms)} arms, {sum(len(v) for v in arms.values())} judges, n = {report['n_test']}")
        for k, v in report["stats"].items():
            if k.split("/")[0] in HYPOTHESES:
                log.info(f"  {k:10s} {v['estimate']:+.4f} [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}] "
                         f"p {v['p_le_0']:.3f} Holm {v['p_holm']:.3f}")
    provenance.record(target, "label-curve", plan=str(args.plan), **provenance.environment())


if __name__ == "__main__":
    main()
