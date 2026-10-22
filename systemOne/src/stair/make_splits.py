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
    parser.add_argument("--standard", action="store_true",
                        help="the team's common splits (stair_common): router_train = common train minus --exclude, "
                             "calib = common val (dev[0:500]), test = common test (dev[500:1000]); ignores the size, "
                             "level and seed options")
    parser.add_argument("--slr", action="store_true",
                        help="the HotpotQA golden set (stair_common 0.3.0, Jonathan's split): router_train = slr-train, "
                             "calib = slr-val, test = slr-test, all from the dev file; with --router-size / --calib-size "
                             "as caps (test is always whole) and --prefer, ignores the other options")
    parser.add_argument("--prefer", type=Path, action="append", default=[],
                        help="--slr: ids taken first when a part is capped (.txt one per line, or .parquet `id`), e.g. "
                             "questions already answered; repeatable")
    return parser.parse_args(argv)


def load_common_train() -> list[str]:
    from stair_common.splits import load_split

    return load_split("hotpotqa", "train")


def standard_splits(exclude: set[str]) -> dict[str, list[str]]:
    """The team's frozen HotpotQA splits; router_train drops `exclude` (e.g. questions nano-jev was trained on)."""
    from stair_common.splits import load_split

    return {"router_train": [i for i in load_common_train() if i not in exclude], "calib": load_split("hotpotqa", "val"),
            "test": load_split("hotpotqa", "test")}


def slr_splits(sizes: dict[str, int] | None = None, prefer: set[str] = frozenset()) -> dict[str, list[str]]:
    """The HotpotQA golden set's split, all dev questions: a run's caches cut from a full-dev run need no model calls.

    `sizes` caps router_train / calib: the `prefer` ids of the part come first, then the rest in the team's random order
    (stair_common.splits.shuffled); the chosen ids keep the part's order. Test is always whole."""
    from stair_common.splits import load_split, shuffled

    parts = {"router_train": load_split("hotpotqa", "slr-train"), "calib": load_split("hotpotqa", "slr-val"),
             "test": load_split("hotpotqa", "slr-test")}
    for name, n in (sizes or {}).items():
        ids = parts[name]
        if n < len(ids):
            ranked = [i for i in ids if i in prefer] + shuffled([i for i in ids if i not in prefer])
            chosen = set(ranked[:n])
            parts[name] = [i for i in ids if i in chosen]
    return parts


def read_ids(path: Path) -> set[str]:
    if path.suffix == ".parquet":
        return set(pd.read_parquet(path, columns=["id"])["id"])
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.slr:
        prefer = set().union(*(read_ids(p) for p in args.prefer))
        splits = slr_splits({"router_train": args.router_size, "calib": args.calib_size}, prefer)
        log.info(f"golden-set splits: router_train {len(splits['router_train'])} (slr-train), "
                 f"calib {len(splits['calib'])} (slr-val), test {len(splits['test'])} (slr-test); "
                 f"{sum(len(prefer & set(v)) for v in splits.values())} of them preferred")
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({"source": "stair_common.splits (hotpotqa: slr-train, slr-val, slr-test)",
                                        "prefer": [str(p) for p in args.prefer], **splits}, indent=1))
        log.info(f"Saved {args.out}")
        return

    if args.standard:
        exclude = set()
        for path in args.exclude:
            exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
        splits = standard_splits(exclude)
        dropped = len(load_common_train()) - len(splits["router_train"])
        log.info(f"standard splits: router_train {len(splits['router_train'])} (dropped {dropped} excluded), "
                 f"calib {len(splits['calib'])} (dev[0:500]), test {len(splits['test'])} (dev[500:1000])")
        manifest = {"source": "stair_common.splits (hotpotqa: train, val, test)", "excluded_files": [str(p) for p in args.exclude],
                    "num_excluded": dropped, **splits}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(manifest, indent=1))
        log.info(f"Saved {args.out}")
        return

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
