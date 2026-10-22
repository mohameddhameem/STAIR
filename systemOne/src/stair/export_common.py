"""Write a judge plan's test results in the team's shared layout (stair_common.schema), one row per question,
policy and seed, and validate them.

    export-common --plan configs/system-one-common.toml --out runs/common-judges --csv results/common/systemOne-hotpotqa-test.csv \
        --history results/common/training-history.csv --plot results/common/loss-curves.png

Policies: always-small, always-large, and every finished judge at its operating point, per seed. The operating
point is the plan's: the cheapest threshold whose calib score (the plan's metric) is within max_drop of always-large.
With --history, every trained judge's per-epoch train loss and dev NLL also go out in the team's training-history
layout (stair_common.training), and --plot draws the loss curves with the best dev epoch marked.

Each row is re-scored with the standard ruler and official EM/F1 on the answer actually returned, and costed in
AgentTTS units:
- `units_retrieve` is nano-jev ranking the paragraphs (paid on every question);
- `units_answer` is the expert answers (both, for a post-answer judge that escalates);
- `units_router` is the judge itself.
"""

import argparse
import json
import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from stair_common.cost import UNIT_GFLOPS
from stair_common.schema import COLUMNS, validate
from stair_common.score import score
from stair_common.training import HISTORY_COLUMNS, from_records, plot_history, validate_history

from stair.evaluate_routing import expert_params, pick_threshold, with_cost_unit
from stair.judge import INPUTS, judge_columns, judge_params_b, load_plan
from stair.metrics import metric_view, text

log = logging.getLogger(__name__)
TRACK = "systemOne"


def models_field(cfg: dict) -> str:
    parts = [f"retriever={cfg['scorer']['path']}"] if cfg.get("scorer", {}).get("path") else []
    parts += [f"{e['name']}={e['model']}" for e in cfg["experts"]]
    return ";".join(parts)


def rows_for(test: pd.DataFrame, escalate: np.ndarray, post: bool, policy: str, seed: int, router_units: np.ndarray,
             retrieve_units: np.ndarray, models: str) -> list[dict]:
    esc = escalate.astype(bool)
    answer_units = np.where(esc, test["large_cost"], test["small_cost"]) + (np.where(esc, test["small_cost"], 0) if post else 0)
    out = []
    for i, (qid, gold) in enumerate(zip(test["id"], test["answer"])):
        answer = text(test["large_pred"].iat[i] if esc[i] else test["small_pred"].iat[i])
        parts = {"units_retrieve": float(retrieve_units[i]), "units_answer": float(answer_units[i]), "units_fuse": 0.0,
                 "units_router": float(router_units[i])}
        out.append({"dataset": "hotpotqa", "split": "test", "id": qid, "track": TRACK, "policy": policy, "menu": "B",
                    "seed": seed, "answer": answer, **score(answer, [str(gold)]), "units": sum(parts.values()), **parts,
                    "ms_router": None, "models": models})
    return out


def export(plan: dict, out: Path, pair: str = "A") -> pd.DataFrame:
    run = Path(plan["pairs"][pair])
    cfg = tomllib.loads((run / "config.toml").read_text())
    params, metric = expert_params(cfg), plan.get("metric", "f1")
    data = {}
    for split in ("calib", "test"):
        df = with_cost_unit(metric_view(pd.read_parquet(run / f"{split}.parquet"), metric), "agenttts", params)
        cols, specs = judge_columns(out, pair, split, plan)
        data[split] = df.merge(cols, on="id", how="left", validate="1:1") if len(cols) else df
    calib, test = data["calib"], data["test"]
    n, models = len(test), models_field(cfg)
    scorer_params = cfg.get("scorer", {}).get("params") or 0.0
    retrieve = (2 * scorer_params * test["scorer_tokens"] / UNIT_GFLOPS).to_numpy() if "scorer_tokens" in test else np.zeros(n)
    zero = np.zeros(n)
    rows = rows_for(test, zero, False, "always-small", 0, zero, retrieve, models)
    rows += rows_for(test, np.ones(n), False, "always-large", 0, zero, retrieve, models)
    for col, spec in specs.items():
        if spec.get("zero_shot"):
            continue
        post = INPUTS[spec["inputs"]]
        tokens = test.get(f"{col}_tokens")
        router = (2 * judge_params_b(spec["init"]) * tokens / UNIT_GFLOPS).to_numpy() if tokens is not None else zero
        for seed_col in sorted(c for c in test.columns if c.startswith(f"{col}#")):
            seed = int(seed_col.rsplit("#seed", 1)[1])
            t = pick_threshold(calib, calib[seed_col].to_numpy(), plan.get("max_drop", 0.01))
            esc = test[seed_col].to_numpy() >= t
            rows += rows_for(test, esc, post, col.removeprefix("judge/"), seed, router, retrieve, models)
    return pd.DataFrame(rows, columns=COLUMNS)


def histories(out: Path) -> pd.DataFrame:
    """The trained judges' per-epoch train loss and dev NLL, one row per (judge, seed, epoch)."""
    frames = []
    for done in sorted(out.glob("judges/*/seed*/done.json")):
        history = json.loads(done.read_text()).get("history")
        if history:
            frames.append(from_records(history, "systemOne", done.parent.parent.name, "judge",
                                       int(done.parent.name.removeprefix("seed")), val_key="dev_nll"))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=HISTORY_COLUMNS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the judge run's --out (judges/<name>/seed<k>/)")
    parser.add_argument("--pair", default="A")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--history", type=Path, help="also write the judges' training histories here (CSV)")
    parser.add_argument("--plot", type=Path, help="and their loss curves here (needs matplotlib)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    df = export(load_plan(args.plan), args.out, args.pair)
    validate(df)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.csv, index=False)
    summary = df.groupby(["policy", "seed"])[["lenient", "em", "f1", "units"]].mean().groupby("policy").mean()
    log.info(f"{len(df)} rows -> {args.csv}\n{summary.round(3).to_string()}")
    if args.history or args.plot:
        hist = histories(args.out)
        validate_history(hist)
        if args.history:
            args.history.parent.mkdir(parents=True, exist_ok=True)
            hist.to_csv(args.history, index=False)
        if args.plot and len(hist):
            plot_history(hist, args.plot)
        log.info(f"{hist.groupby(['model', 'seed']).ngroups} training histories -> {args.history} {args.plot}")


if __name__ == "__main__":
    main()
