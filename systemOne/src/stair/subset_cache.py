"""Copy the rows of a finished run's cache whose ids are in a split key into another run, with no model calls.

The team's common splits take validation and test from the HotpotQA dev file, which an earlier run has already
answered in full (`<run>/test.parquet`, 7,405 questions). This cuts those caches out of it:

    subset-cache --from runs/gpu --splits data/hotpotqa/splits-common.json --split calib --run runs/common --name calib
    subset-cache --from runs/gpu --splits data/hotpotqa/splits-common.json --split test  --run runs/common --name test

The target run takes the source's config.toml; if it already has one (e.g. from run-experts on router_train), the
settings that change outputs must match, so every cache of the run comes from the same experts.
"""

import argparse
import json
import logging
import shutil
import tomllib
from pathlib import Path

import pandas as pd

from stair import provenance
from stair.run_experts import output_settings

log = logging.getLogger(__name__)


def subset(source: Path, splits: Path, split: str, run: Path, name: str, cache: str = "test") -> pd.DataFrame:
    ids = json.loads(splits.read_text())[split]
    df = pd.read_parquet(source / f"{cache}.parquet")
    missing = set(ids) - set(df["id"])
    if missing:
        raise ValueError(f"{len(missing)} {split} ids are not in {source / f'{cache}.parquet'}, e.g. {sorted(missing)[:3]}")
    out = df.set_index("id").loc[ids].reset_index()  # the split's order
    run.mkdir(parents=True, exist_ok=True)
    src_cfg, cfg = source / "config.toml", run / "config.toml"
    if cfg.exists():
        if output_settings(tomllib.loads(cfg.read_text())) != output_settings(tomllib.loads(src_cfg.read_text())):
            raise ValueError(f"{cfg} and {src_cfg} differ in settings that change outputs")
    else:
        shutil.copy(src_cfg, cfg)
    out.to_parquet(run / f"{name}.parquet", index=False)
    provenance.record(run, "subset-cache", source=str(source / f"{cache}.parquet"), splits=str(splits), split=split,
                      name=name, n=len(out))
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--from", dest="source", type=Path, required=True, help="finished run with the full cache")
    parser.add_argument("--cache", default="test", help="cache name in the source run (default test: the dev file)")
    parser.add_argument("--splits", type=Path, required=True)
    parser.add_argument("--split", required=True, help="key in --splits")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--name", required=True, help="cache name to write: calib or test")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = subset(args.source, args.splits, args.split, args.run, args.name, args.cache)
    log.info(f"{args.run / (args.name + '.parquet')}: {len(out)} rows from {args.source / (args.cache + '.parquet')}")


if __name__ == "__main__":
    main()
