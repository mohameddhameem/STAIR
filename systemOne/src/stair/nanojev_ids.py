"""Write the HotpotQA question ids that nano-jev v1.0 was trained, tuned or tested on.

nano-jev's `build_all` shuffles each split with `seed` and then takes a range: train questions
[0, 6000) for training, validation questions [0, 500) for calibration and dev-NLL epoch selection,
and [500, 1000) for its own test. So those validation questions are the first 1,000 after the
shuffle, not the first 1,000 rows. The id lists written here feed `make-splits --exclude`
(train) and `eval-routing --exclude` (validation).

The lists are conservative: nano-jev skips a few selected questions (not exactly two gold
paragraphs), and they are listed anyway.
"""

import argparse
import logging
from pathlib import Path

from datasets import load_dataset

log = logging.getLogger(__name__)


def shuffled_ids(split, seed: int, n: int) -> list[str]:
    """Ids of the first n questions of a `datasets` split after `split.shuffle(seed=seed)`."""
    return list(split.shuffle(seed=seed).select(range(min(n, len(split))))["id"])


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="nano-jev's prepare_data.py --seed (v1.0: 0)")
    parser.add_argument("--train-questions", type=int, default=6_000, help="preset hotpot_train")
    parser.add_argument("--eval-questions", type=int, default=1_000, help="2 x preset hotpot_eval (calib + test)")
    parser.add_argument("--out", type=Path, default=Path("data/hotpotqa"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    dataset = load_dataset("hotpotqa/hotpot_qa", name="distractor")
    args.out.mkdir(parents=True, exist_ok=True)
    for split, n in (("train", args.train_questions), ("validation", args.eval_questions)):
        ids = shuffled_ids(dataset[split], args.seed, n)
        path = args.out / f"nanojev_{split}_ids.txt"
        path.write_text("\n".join(ids) + "\n")
        log.info(f"{split}: {len(ids)} ids -> {path}")


if __name__ == "__main__":
    main()
