"""Split HotpotQA train questions into router-train and calibration sets.

The official validation set is the final test set and is never split here. Questions that
another component was trained on (e.g. nano-jev's HotpotQA training questions) can be
excluded with --exclude so that component's scores are not inflated on router data.

Every validation question is `level == "hard"`, while train mixes easy, medium and hard. By
default only hard train questions are used, so the router and its operating threshold are fit on
the same difficulty as the test set (about 15.6k questions before exclusions). `--level any`
keeps all levels.
"""

import argparse
import json
import logging
import random
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def make_splits(ids: list[str], exclude: set[str], n_router: int, n_calib: int, seed: int) -> dict[str, list[str]]:
    """Return disjoint, sorted id lists for router_train and calib."""
    pool = sorted(set(ids) - exclude)
    if n_router + n_calib > len(pool):
        raise ValueError(f"requested {n_router + n_calib} questions but only {len(pool)} are available")
    random.Random(seed).shuffle(pool)
    return {
        "router_train": sorted(pool[:n_router]),
        "calib": sorted(pool[n_router : n_router + n_calib]),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, default=Path("data/hotpotqa/distractor_train.parquet"))
    parser.add_argument("--exclude", type=Path, action="append", default=[], help="file with one question id per line; repeatable")
    parser.add_argument("--level", action="append", default=None,
                        help="keep train questions of this level (repeatable; default hard, 'any' for all)")
    parser.add_argument("--router-size", type=int, default=12_000)
    parser.add_argument("--calib-size", type=int, default=2_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("data/hotpotqa/splits.json"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    levels = args.level or ["hard"]
    df = pd.read_parquet(args.train, columns=["id", "type", "level"])
    exclude = set()
    for path in args.exclude:
        exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    unknown = exclude - set(df["id"])
    if unknown:
        log.warning(f"{len(unknown)} excluded ids are not in {args.train}")
    if "any" not in levels:
        before = len(df)
        df = df[df["level"].isin(levels)]
        log.info(f"kept {len(df)} of {before} train questions with level in {levels}")

    splits = make_splits(df["id"].tolist(), exclude, args.router_size, args.calib_size, args.seed)
    types = df.set_index("id")["type"]
    for name, ids in splits.items():
        log.info(f"{name}: {len(ids)} questions | types={types.loc[ids].value_counts().to_dict()}")

    manifest = {
        "source": str(args.train),
        "seed": args.seed,
        "levels": levels,
        "excluded_files": [str(p) for p in args.exclude],
        "num_excluded": len(exclude & set(df["id"])),
        **splits,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1))
    log.info(f"Saved {args.out}")


if __name__ == "__main__":
    main()
