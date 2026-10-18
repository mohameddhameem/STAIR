"""The team's frozen question splits, and the one function that builds them.

Each test set is AgentTTS's own (HotpotQA, CWQ, WebQSP), so every track's headline number is directly comparable with
the paper and no test question is in anyone's training data. Validation and training come from questions AgentTTS
does not test on. Where a split is random, it uses one rule: sort the IDs, then shuffle with
numpy.random.default_rng(20260729).

    python -m stair_common.splits build --sources <dir with the files below> --out common/splits

Sources (pinned): AgentTTS @ 3f0c2df (data/HotpotQA/hotpotqa-dev.json, data/2wikimultihopqa/tts_train.json,
data/cwq/data_with_ct_0.json, data/webqsp/data_with_ct_0.json); HotpotQA train (hotpot_train_v1.1.json, or the HF
`hotpot_qa` distractor train split, same IDs); MedAgentBench @ 9926011 (data/medagentbench/test_data_v2.json).
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 20260729
SPLITS_DIR = Path(__file__).resolve().parent / "splits"  # shipped inside the package
HOTPOTQA_TRAIN_DRAW = 12_000  # fine-tuning / router-training questions drawn from the official train file


def shuffled(ids, seed: int = SEED) -> list[str]:
    """The team's random rule: sort, then shuffle with default_rng(seed)."""
    ids = sorted(ids)
    return [ids[i] for i in np.random.default_rng(seed).permutation(len(ids))]


def cut(ids, sizes: dict[str, int], seed: int = SEED) -> dict[str, list[str]]:
    """Shuffle `ids` with the rule and cut consecutive parts of the given sizes (in dict order)."""
    order, out, i = shuffled(ids, seed), {}, 0
    if sum(sizes.values()) > len(order):
        raise ValueError(f"asked for {sum(sizes.values())} of {len(order)} ids")
    for name, n in sizes.items():
        out[name] = order[i:i + n]
        i += n
    return out


def hotpotqa(dev_ids: list[str], train_ids: list[str]) -> dict[str, list[str]]:
    """dev in file order: validation [0:500], test [500:1000] (AgentTTS's), extra training [1000:]."""
    if len(dev_ids) != 7405:
        raise ValueError(f"expected the 7,405-question dev file, got {len(dev_ids)}")
    return {"val": dev_ids[:500], "test": dev_ids[500:1000], "train_extra": dev_ids[1000:],
            "train": cut(train_ids, {"train": HOTPOTQA_TRAIN_DRAW})["train"]}


def twowiki(dev_ids: list[str]) -> dict[str, list[str]]:
    """Official dev (AgentTTS's tts_train.json, file order), the same slices as HotpotQA."""
    if len(dev_ids) != 12576:
        raise ValueError(f"expected 12,576 questions, got {len(dev_ids)}")
    return {"val": dev_ids[:500], "test": dev_ids[500:1000], "train_extra": dev_ids[1000:]}


def kgqa(ids: list[str]) -> dict[str, list[str]]:
    """CWQ / WebQSP data_with_ct_0.json (1,000 with gold triplets): test = the first 500, which is AgentTTS's
    *_test500.json exactly; the other 500 are cut 350 train / 150 validation with the rule."""
    if len(ids) != 1000:
        raise ValueError(f"expected 1,000 questions, got {len(ids)}")
    return {"test": ids[:500], **cut(ids[500:], {"train": 350, "val": 150})}


def medagentbench(ids: list[str]) -> dict[str, list[str]]:
    """300 tasks, 10 categories of 30: per category 15 calibration / 15 test, with the rule. Too few to fine-tune."""
    by_task = defaultdict(list)
    for i in ids:
        by_task[i.rsplit("_", 1)[0]].append(i)
    out = {"calib": [], "test": []}
    for task in sorted(by_task):
        parts = cut(by_task[task], {"calib": len(by_task[task]) // 2, "test": len(by_task[task]) - len(by_task[task]) // 2})
        out["calib"] += parts["calib"]
        out["test"] += parts["test"]
    return out


def load_split(dataset: str, part: str, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """IDs of one part (train, train_extra, val, calib, test) of a dataset, in the committed order."""
    return (splits_dir / dataset / f"{part}.txt").read_text().split()


def _json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hotpotqa_train_ids(path: Path) -> list[str]:
    if path.suffix == ".parquet":  # HF `hotpot_qa` distractor train split, saved as parquet (column `id`)
        import pandas as pd
        return pd.read_parquet(path, columns=["id"])["id"].tolist()
    return [e["_id"] for e in _json(path)]


def build(sources: Path, out: Path) -> dict:
    train = next((sources / n for n in ("hotpot_train_v1.1.json", "distractor_train.parquet") if (sources / n).exists()),
                 sources / "hotpot_train_v1.1.json")
    files = {
        "hotpotqa_dev": sources / "hotpotqa-dev.json",
        "hotpotqa_train": train,
        "2wikimultihopqa": sources / "tts_train.json",
        "cwq": sources / "cwq_data_with_ct_0.json",
        "webqsp": sources / "webqsp_data_with_ct_0.json",
        "medagentbench": sources / "test_data_v2.json",
    }
    splits = {
        "hotpotqa": hotpotqa([e["_id"] for e in _json(files["hotpotqa_dev"])],
                             _hotpotqa_train_ids(files["hotpotqa_train"])),
        "2wikimultihopqa": twowiki([e["_id"] for e in _json(files["2wikimultihopqa"])]),
        "cwq": kgqa([e["ID"] for e in _json(files["cwq"])]),
        "webqsp": kgqa([e["QuestionId"] for e in _json(files["webqsp"])]),
        "medagentbench": medagentbench([e["id"] for e in _json(files["medagentbench"])]),
    }
    manifest = {"seed": SEED, "sources": {k: {"file": v.name, "sha256": _sha(v)} for k, v in files.items()}, "parts": {}}
    for dataset, parts in splits.items():
        seen = set()
        for part, ids in parts.items():
            if seen & set(ids):
                raise ValueError(f"{dataset}/{part} overlaps another part")
            seen |= set(ids)
            target = out / dataset / f"{part}.txt"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("\n".join(ids) + "\n", newline="\n")
            manifest["parts"][f"{dataset}/{part}"] = {"n": len(ids), "sha256": _sha(target)}
    (out / "MANIFEST.json").write_text(json.dumps(manifest, indent=1) + "\n", newline="\n")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=SPLITS_DIR)
    args = parser.parse_args(argv)
    for name, part in build(args.sources, args.out)["parts"].items():
        print(f"{name:32s} {part['n']:6d}")


if __name__ == "__main__":
    main()
