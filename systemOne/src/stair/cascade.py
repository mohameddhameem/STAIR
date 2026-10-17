"""Multi-stage cascades over cached experts from several runs.

Every run in this repo answers the same questions (same splits, same evidence), so experts from
different runs can be chained without any GPU: e.g. 1.5B (top 2) -> 7B (top 5) -> 14B (all 10).
A question starts at the first stage; after each stage a router decides whether to escalate to the
next one. A question pays for every stage it reaches and keeps the last stage's answer.

Router i (after stage i) is a logistic regression on the evidence features, the answer signals of
every stage so far (entropy, length, yes/no) and the agreement (token F1) between consecutive
answers. Its label is "a later stage helps": stage i's F1 < tau and some later stage's F1 >= tau.
The `entropy` variant of each cascade replaces the routers with the current stage's mean token
entropy, the multi-stage form of the small-entropy baseline.

Thresholds are never tuned on test. Each router's candidate thresholds are its calib-score
quantiles (escalation rates 0, 0.05, ..., 1). Every combination is scored on calib, the calib
cost-F1 frontier is kept, and only those policies are evaluated on test. AIQ is computed over
[cheapest, most expensive] single expert of the whole comparison, so cascades with the same
experts are comparable. The operating point is the cheapest calib-frontier policy within
max-drop of the cascade's strongest single expert on calib.

    cascade --expert small=results/gpu:small --expert mid=results/gpu-large-top5:large \\
            --expert large=results/gpu-large-14b:large --cascade small,large --cascade small,mid,large
"""

import argparse
import itertools
import json
import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance
from stair.evaluate_routing import COST_UNITS, RATES, aiq, expert_params
from stair.metrics import f1_score
from stair.router import EVIDENCE_COLUMNS, add_answer_cues, answer_columns, build_router, fit_tuned

log = logging.getLogger(__name__)

SPLITS = ("router_train", "calib", "test")
EXPERT_FIELDS = ("pred", "f1", "em", "cost", "uncertainty", "token_entropy", "generated_tokens", "prompt_tokens")
VARIANTS = ("router", "entropy")


def parse_expert(spec: str) -> tuple[str, Path, str]:
    """"name=run_dir:column" -> (name, run_dir, column), e.g. "mid=results/gpu-large-top5:large"."""
    name, _, rest = spec.partition("=")
    run, _, column = rest.rpartition(":")
    if not (name.isidentifier() and run and column):
        raise ValueError(f"--expert {spec!r}: expected name=run_dir:column with an identifier name")
    return name, Path(run), column


def load_split(experts: list[tuple[str, Path, str]], split: str, unit: str) -> pd.DataFrame:
    """One row per question: the first run's evidence columns plus <name>_* columns per expert."""
    first = pd.read_parquet(experts[0][1] / f"{split}.parquet")
    df = first[["id", "question", "answer", "type", *EVIDENCE_COLUMNS]].copy()
    for name, run, column in experts:
        cache = pd.read_parquet(run / f"{split}.parquet", columns=["id", *(f"{column}_{f}" for f in EXPERT_FIELDS)])
        if not (cache["id"].to_numpy() == df["id"].to_numpy()).all():
            raise ValueError(f"{run}/{split}.parquet has different questions or order than {experts[0][1]}")
        cache = cache.drop(columns="id").rename(columns=lambda c: name + c[len(column):])
        if unit == "flops":
            params = expert_params(tomllib.loads((run / "config.toml").read_text())).get(column)
            if params is None:
                raise ValueError(f"no parameter count for {column} in {run}/config.toml; use --cost tokens")
            cache[f"{name}_cost"] = 2 * params * cache[f"{name}_cost"]
        df = pd.concat([df, cache], axis=1)
        df = add_answer_cues(df, name)
    return df


def router_columns(stages: list[str], i: int) -> list[str]:
    cols = list(EVIDENCE_COLUMNS)
    for j in range(i + 1):
        cols += answer_columns(stages[j]) + ([f"agree_{stages[j - 1]}_{stages[j]}"] if j else [])
    return cols


def add_agreement(df: pd.DataFrame, stages: list[str]) -> pd.DataFrame:
    df = df.copy()
    for a, b in zip(stages, stages[1:]):
        df[f"agree_{a}_{b}"] = [f1_score(p, q) for p, q in zip(df[f"{a}_pred"], df[f"{b}_pred"])]
    return df


def later_helps(df: pd.DataFrame, stages: list[str], i: int, tau: float) -> np.ndarray:
    later = df[[f"{s}_f1" for s in stages[i + 1:]]].max(axis=1)
    return ((df[f"{stages[i]}_f1"] < tau) & (later >= tau)).astype(int).to_numpy()


def train_cascade_routers(train: pd.DataFrame, stages: list[str], tau: float, seed: int) -> list[dict]:
    routers = []
    for i in range(len(stages) - 1):
        cols = router_columns(stages, i)
        y = later_helps(train, stages, i, tau)
        model, tuning = fit_tuned(build_router(cols, text=False), train[cols].fillna(0.0), y, seed)
        routers.append({"model": model, "columns": cols, "tuning": tuning, "label_rate": float(y.mean())})
        log.info(f"    router {stages[i]} -> {stages[i + 1]}: label rate {y.mean():.3f} | {tuning}")
    return routers


def stage_scores(df: pd.DataFrame, stages: list[str], variant: str, routers: list[dict] | None) -> np.ndarray:
    """[stages - 1, questions] escalation scores (higher = escalate)."""
    if variant == "entropy":
        out = []
        for s in stages[:-1]:
            u = df[f"{s}_uncertainty"].to_numpy(dtype=float)
            out.append(np.where(np.isnan(u), np.nanmax(u), u))  # nothing generated: least sure
        return np.array(out)
    return np.array([r["model"].predict_proba(df[r["columns"]].fillna(0.0))[:, 1] for r in routers])


def matrices(df: pd.DataFrame, stages: list[str]) -> dict[str, np.ndarray]:
    """[stages, questions] F1, EM and cumulative cost (a question pays every stage it reaches)."""
    get = lambda f: np.array([df[f"{s}_{f}"].to_numpy(dtype=float) for s in stages])
    return {"f1": get("f1"), "em": get("em"), "cost": np.cumsum(get("cost"), axis=0)}


def stops(scores: np.ndarray, thresholds: tuple[float, ...]) -> np.ndarray:
    """Index of the stage each question ends at."""
    stop = np.zeros(scores.shape[1], dtype=int)
    reach = np.ones(scores.shape[1], dtype=bool)
    for i, t in enumerate(thresholds):
        reach &= scores[i] >= t
        stop[reach] = i + 1
    return stop


def per_question(m: dict[str, np.ndarray], stop: np.ndarray) -> dict[str, np.ndarray]:
    cols = np.arange(stop.size)
    return {k: v[stop, cols] for k, v in m.items()}


def threshold_grid(calib_scores: np.ndarray) -> list[list[float]]:
    """Per router: its calib-score quantiles, so each threshold escalates a rate in RATES of all calib questions."""
    grid = []
    for s in calib_scores:
        ts = [np.inf if r == 0 else -np.inf if r == 1 else float(np.quantile(s, 1 - r)) for r in RATES]
        grid.append(sorted(set(ts), reverse=True))
    return grid


def calib_frontier(calib_scores: np.ndarray, calib_m: dict[str, np.ndarray]) -> list[dict]:
    """Threshold combinations on calib's cost-F1 frontier (each strictly better than every cheaper one)."""
    points = []
    for thresholds in itertools.product(*threshold_grid(calib_scores)):
        q = per_question(calib_m, stops(calib_scores, thresholds))
        points.append({"thresholds": thresholds, "cost": float(q["cost"].mean()), "f1": float(q["f1"].mean())})
    points.sort(key=lambda p: (p["cost"], -p["f1"]))
    frontier, best = [], -np.inf
    for p in points:
        if p["f1"] > best:
            frontier.append(p)
            best = p["f1"]
    return frontier


def evaluate_variant(frontier: list[dict], scores: np.ndarray, m: dict[str, np.ndarray], stages: list[str],
                     calib_ref_f1: float, max_drop: float) -> tuple[dict, dict[str, np.ndarray]]:
    """Test points of the calib frontier, plus the operating point; also the per-question arrays."""
    arrays = {"f1": [], "cost": [], "em": []}
    points = []
    for p in frontier:
        stop = stops(scores, p["thresholds"])
        q = per_question(m, stop)
        for k in arrays:
            arrays[k].append(q[k])
        points.append({"calib_cost": p["cost"], "calib_f1": p["f1"], "cost": float(q["cost"].mean()),
                       "f1": float(q["f1"].mean()), "em": float(q["em"].mean()),
                       "stage_shares": {s: float((stop == i).mean()) for i, s in enumerate(stages)}})
    target = (1 - max_drop) * calib_ref_f1
    op = next((i for i, p in enumerate(frontier) if p["f1"] >= target), len(frontier) - 1)
    return {"points": points, "operating_point": points[op], "op_index": op}, {k: np.array(v) for k, v in arrays.items()}


def bootstrap(arrays: dict[str, dict[str, np.ndarray]], singles: dict[str, dict[str, np.ndarray]], ops: dict[str, int],
              refs: dict[str, str], diffs: list[tuple[str, str]], n_boot: int, seed: int) -> dict:
    """Paired bootstrap over test questions: AIQ intervals, AIQ differences, op F1 minus best single expert."""
    rng = np.random.default_rng(seed)
    n = next(iter(singles.values()))["f1"].size
    aiqs = {k: [] for k in arrays}
    gaps = {k: [] for k in arrays}
    for _ in range(n_boot):
        w = np.bincount(rng.integers(0, n, n), minlength=n) / n
        costs = [s["cost"] @ w for s in singles.values()]
        cmin, cmax = min(costs), max(costs)
        for k, a in arrays.items():
            f1s, cs = a["f1"] @ w, a["cost"] @ w
            aiqs[k].append(aiq([{"cost": c, "f1": f} for c, f in zip(cs, f1s)], cmin, cmax))
            gaps[k].append(f1s[ops[k]] - singles[refs[k]]["f1"] @ w)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return {
        "n": n_boot, "seed": seed,
        "aiq_ci95": {k: ci(v) for k, v in aiqs.items()},
        "aiq_diff": {f"{a} - {b}": {"ci95": ci(np.subtract(aiqs[a], aiqs[b])),
                                    "p_le_0": float((np.subtract(aiqs[a], aiqs[b]) <= 0).mean())} for a, b in diffs},
        "op_f1_minus_best_single": {k: {"ci95": ci(v), "p_le_0": float((np.asarray(v) <= 0).mean())}
                                    for k, v in gaps.items()},
    }


def run(experts: list[tuple[str, Path, str]], cascades: list[list[str]], tau: float, max_drop: float, unit: str,
        n_boot: int, seed: int, exclude: set[str]) -> tuple[dict, pd.DataFrame]:
    names = [e[0] for e in experts]
    for c in cascades:
        if len(c) < 2 or any(s not in names for s in c):
            raise ValueError(f"cascade {c}: needs 2+ stages, all from --expert ({names})")
    data = {split: load_split(experts, split, unit) for split in SPLITS}
    if exclude:
        data["test"] = data["test"][~data["test"]["id"].isin(exclude)].reset_index(drop=True)
    test, calib = data["test"], data["calib"]
    singles = {s: {k: test[f"{s}_{k}"].to_numpy(dtype=float) for k in ("f1", "em", "cost")} for s in names}
    single_costs = [v["cost"].mean() for v in singles.values()]
    cmin, cmax = min(single_costs), max(single_costs)
    cheapest = names[int(np.argmin(single_costs))]
    for c in cascades:  # AIQ integrates from cmin: a cascade starting higher would get credit it cannot reach
        if c[0] != cheapest:
            raise ValueError(f"cascade {c} must start with the cheapest expert, {cheapest}, for AIQ to be comparable")
    f1s = np.array([singles[s]["f1"] for s in names])
    costs = np.array([singles[s]["cost"] for s in names])
    best = f1s.max(axis=0)
    cheapest_best = np.where(f1s == best, costs, np.inf).argmin(axis=0)  # cheapest expert with the best F1
    cols = np.arange(len(test))
    report = {
        "cost_unit": unit, "tau": tau, "max_drop": max_drop, "n_test": len(test), "aiq_range": [cmin, cmax],
        "experts": {s: {"run": str(r), "column": c, **{k: float(v.mean()) for k, v in singles[s].items()}}
                    for (s, r, c) in experts},
        "oracle": {"f1": float(best.mean()), "cost": float(costs[cheapest_best, cols].mean()),
                   "shares": {s: float((cheapest_best == i).mean()) for i, s in enumerate(names)}},
        "cascades": {},
    }
    arrays, ops, refs, rows = {}, {}, {}, []
    for stages in cascades:
        cname = ">".join(stages)
        log.info(f"cascade {cname}")
        split = {k: add_agreement(v, stages) for k, v in data.items()}
        routers = train_cascade_routers(split["router_train"], stages, tau, seed)
        calib_m, test_m = matrices(split["calib"], stages), matrices(split["test"], stages)
        ref = max(stages, key=lambda s: calib[f"{s}_f1"].mean())  # strongest single expert on calib
        entry = {"strongest_expert": ref, "routers": [{k: r[k] for k in ("columns", "tuning", "label_rate")} for r in routers]}
        for variant in VARIANTS:
            frontier = calib_frontier(stage_scores(split["calib"], stages, variant, routers), calib_m)
            out, arr = evaluate_variant(frontier, stage_scores(split["test"], stages, variant, routers), test_m, stages,
                                        calib[f"{ref}_f1"].mean(), max_drop)
            key = f"{cname}/{variant}"
            out["aiq"] = aiq(out["points"], cmin, cmax)
            entry[variant] = out
            arrays[key], ops[key], refs[key] = arr, out["op_index"], ref
            rows += [{"cascade": cname, "variant": variant, **{k: v for k, v in p.items() if k != "stage_shares"}}
                     for p in out["points"]]
            op = out["operating_point"]
            log.info(f"  {variant:<8} AIQ {out['aiq']:.4f} | op F1 {op['f1']:.3f} EM {op['em']:.3f} cost {op['cost']:,.0f} "
                     f"| stops {', '.join(f'{s} {v:.0%}' for s, v in op['stage_shares'].items())}")
        report["cascades"][cname] = entry
    if n_boot:
        keys = list(arrays)
        reference = keys[0]
        diffs = [(k, f"{k.split('/')[0]}/entropy") for k in keys if k.endswith("/router")]
        diffs += [(k, reference) for k in keys if k != reference and k.endswith("/router")]
        report["bootstrap"] = bootstrap(arrays, singles, ops, refs, diffs, n_boot, seed)
    return report, pd.DataFrame(rows)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--expert", action="append", required=True, help="name=run_dir:column; repeatable")
    parser.add_argument("--cascade", action="append", required=True,
                        help="comma-separated expert names, cheapest first; repeatable. The first is the bootstrap reference")
    parser.add_argument("--tau", type=float, default=0.8)
    parser.add_argument("--max-drop", type=float, default=0.01)
    parser.add_argument("--cost", choices=COST_UNITS[:2], default="flops")
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--exclude", type=Path, action="append", default=[], help="test question ids to drop; writes *-clean")
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    experts = [parse_expert(s) for s in args.expert]
    exclude = set()
    for path in args.exclude:
        exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    report, curves = run(experts, [c.split(",") for c in args.cascade], args.tau, args.max_drop, args.cost,
                         args.bootstrap, args.seed, exclude)
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = "-clean" if exclude else ""
    (args.out / f"report{suffix}.json").write_text(json.dumps(report, indent=1))
    curves.to_csv(args.out / f"curves{suffix}.csv", index=False)
    provenance.record(args.out, "cascade", experts=args.expert, cascades=args.cascade, tau=args.tau,
                      excluded_files=[str(p) for p in args.exclude], **provenance.environment())
    for name, e in report["experts"].items():
        log.info(f"expert {name:<8} F1 {e['f1']:.3f} EM {e['em']:.3f} cost {e['cost']:,.0f}")
    log.info(f"oracle F1 {report['oracle']['f1']:.3f} cost {report['oracle']['cost']:,.0f}")
    for k, d in report.get("bootstrap", {}).get("aiq_diff", {}).items():
        log.info(f"  AIQ {k:<60} 95% CI [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] p(<=0) {d['p_le_0']:.3f}")
    log.info(f"Saved {args.out / f'report{suffix}.json'} and {args.out / f'curves{suffix}.csv'}")


if __name__ == "__main__":
    main()
