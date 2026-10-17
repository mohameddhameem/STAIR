"""Sensitivity of the routing results to tau, the F1 at which an expert counts as correct.

tau only changes the routers' training labels: the experts' answers, costs and the always-small,
always-large and oracle points do not move. For each tau this retrains every router on
router_train (same CV tuning as train-router), picks its operating point on calib and evaluates on
test, without the bootstrap. Routers are not saved; run.../routers.pkl and report.json are left
alone. Writes `<run>/tau-sweep.csv` (one row per tau and router) and `<run>/tau-sweep.json`.
"""

import argparse
import json
import logging
import tomllib
from pathlib import Path

import pandas as pd

from stair import provenance
from stair.evaluate_routing import COST_UNITS, evaluate, expert_params, with_cost_unit
from stair.router import LABELS, escalation_label, train_routers

log = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--tau", type=float, nargs="+", default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    parser.add_argument("--max-drop", type=float, default=0.01)
    parser.add_argument("--cost", choices=COST_UNITS, default="flops")
    parser.add_argument("--seed", type=int, default=0, help="seed for the cross-validation folds")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = expert_params(tomllib.loads((args.run / "config.toml").read_text()))
    train = pd.read_parquet(args.run / "router_train.parquet")
    test = with_cost_unit(pd.read_parquet(args.run / "test.parquet"), args.cost, params)
    calib = with_cost_unit(pd.read_parquet(args.run / "calib.parquet"), args.cost, params)

    rows, baselines = [], {}
    for tau in args.tau:
        log.info(f"tau {tau}")
        routers = train_routers(train, tau, args.seed)
        report, _ = evaluate(test, calib, routers, tau, args.max_drop)
        baselines = {name: report["aiq"][name] for name in ("random", "oracle", "small-entropy (post)")}
        for name, entry in report["routers"].items():
            op = entry["operating_point"]
            rows.append({
                "tau": tau, "router": name, "label": entry["label"], "post": entry["post"],
                "train_label_rate": float(escalation_label(train, tau, entry["label"]).mean()),
                "aiq": report["aiq"][f"router:{name}"],
                "op_f1": op["f1"], "op_cost": op["cost"], "op_rate": op["rate"],
                "ece": entry["ece"], "C": entry["tuning"]["C"],
            })
            log.info(f"  {name:<40} AIQ {rows[-1]['aiq']:.4f} | op F1 {op['f1']:.3f} cost {op['cost']:.0f} "
                     f"rate {op['rate']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(args.run / "tau-sweep.csv", index=False)
    summary = {"cost_unit": args.cost, "taus": args.tau, "labels": list(LABELS), "baseline_aiq": baselines,
               "best_by_tau": {str(t): g.loc[g["aiq"].idxmax(), ["router", "aiq"]].to_dict()
                               for t, g in df.groupby("tau")}}
    (args.run / "tau-sweep.json").write_text(json.dumps(summary, indent=1))
    provenance.record(args.run, "tau-sweep", taus=args.tau, **provenance.environment())
    log.info(f"baseline AIQ (independent of tau): {baselines}")
    log.info(f"Saved {args.run / 'tau-sweep.csv'} and {args.run / 'tau-sweep.json'}")


if __name__ == "__main__":
    main()
