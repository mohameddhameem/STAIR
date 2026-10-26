"""The team's frozen question splits, and the one function that builds them.

Each test set is AgentTTS's own (HotpotQA, CWQ, WebQSP), so every track's headline number is directly comparable with
the paper and no test question is in anyone's training data. Validation and training come from questions AgentTTS
does not test on. Where a split is random, it uses one rule: sort the IDs, then shuffle with
numpy.random.default_rng(20260729).

HotpotQA also carries the golden set the team follows (Jonathan's Stage-Level Routing split): parts `slr-train`,
`slr-val` and `slr-test` (4,814 / 1,110 / 1,481 of the 7,405 dev questions), taken by position from his committed
splits.json. It is a second scheme over the same dev questions, so it overlaps the dev slices above by design; parts
are disjoint within a scheme.

MedAgentBench is split by patient (from 0.5.0): all of a patient's tasks sit in `calib` or all in `test`, 15 tasks per
category on each side, and `calib_sweep` is a subset of calib. The split is Jonathan's, found by a seeded search on
his branch (`jonathan` @ 2a35c9b), so it is frozen in the committed files: `build` keeps them, after checking them
against test_data_v2.json.

    python -m stair_common.splits build --sources <dir with the files below> --out common/splits

Sources (pinned): AgentTTS @ 3f0c2df (data/HotpotQA/hotpotqa-dev.json, data/cwq/data_with_ct_0.json,
data/webqsp/data_with_ct_0.json); HotpotQA train (hotpot_train_v1.1.json, or the HF `hotpot_qa` distractor train split,
same IDs); 2WikiMultiHopQA's official dev set, the HF mirror `framolfese/2WikiMultihopQA` @ fe713bf
(data/validation-00000-of-00001.parquet, saved as 2wikimultihopqa_dev.parquet); MedAgentBench @ 9926011
(data/medagentbench/test_data_v2.json); STAIR branch `jonathan` @ 5b69cce (jonathan/lab_package/splits.json, saved as
slr_splits.json).
"""

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

SEED = 20260729
SPLITS_DIR = Path(__file__).resolve().parent / "splits"  # shipped inside the package
HOTPOTQA_TRAIN_DRAW = 12_000  # fine-tuning / router-training questions drawn from the official train file
TWOWIKI_DEV = "2wikimultihopqa_dev.parquet"  # the mirror's validation split, 12,576 answered questions


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


SLR_SIZES = {"train": 4814, "val": 1110, "test": 1481}


def hotpotqa_slr(dev_ids: list[str], slr: dict) -> dict[str, list[str]]:
    """Jonathan's Stage-Level Routing split: splits.json lists dev positions per part; IDs follow in position order."""
    if len(dev_ids) != 7405 or slr.get("n_total") != 7405:
        raise ValueError("the SLR split is over the 7,405-question dev file")
    parts = {f"slr-{name}": [dev_ids[i] for i in sorted(slr["splits"][name])] for name in SLR_SIZES}
    if {k: len(v) for k, v in parts.items()} != {f"slr-{k}": n for k, n in SLR_SIZES.items()}:
        raise ValueError(f"SLR part sizes {[len(v) for v in parts.values()]}, expected {list(SLR_SIZES.values())}")
    if set().union(*map(set, parts.values())) != set(dev_ids):
        raise ValueError("the SLR parts must cover the dev file exactly")
    return parts


def scheme(part: str) -> str:
    """The scheme a part belongs to: "slr" for slr-*, "calib_sweep" for MedAgentBench's subset of calib, "" for the
    rest. Parts are disjoint within a scheme."""
    if part == "calib_sweep":
        return part
    return part.split("-")[0] if "-" in part else ""


def twowiki(dev_ids: list[str], answers: list[str]) -> dict[str, list[str]]:
    """Official dev, cut with the rule: val = the first 500, test = the next 500, train_extra = the other 11,576.
    Every question must have a gold answer. Until 0.4.0 the source was AgentTTS's tts_train.json, which is 2Wiki's
    official test set: all 12,576 answers are empty, so nothing on it could be scored."""
    if len(dev_ids) != 12576:
        raise ValueError(f"expected 12,576 questions, got {len(dev_ids)}")
    unanswered = [i for i, a in zip(dev_ids, answers, strict=True) if not str(a or "").strip()]
    if unanswered:
        raise ValueError(f"{len(unanswered)} questions have no gold answer (first: {unanswered[0]}); "
                         "this is not the answered dev set")
    return cut(dev_ids, {"val": 500, "test": 500, "train_extra": len(dev_ids) - 1000})


def kgqa(ids: list[str]) -> dict[str, list[str]]:
    """CWQ / WebQSP data_with_ct_0.json (1,000 with gold triplets): test = the first 500, which is AgentTTS's
    *_test500.json exactly; the other 500 are cut 350 train / 150 validation with the rule."""
    if len(ids) != 1000:
        raise ValueError(f"expected 1,000 questions, got {len(ids)}")
    return {"test": ids[:500], **cut(ids[500:], {"train": 350, "val": 150})}


MEDAGENTBENCH_PER_CATEGORY = {"calib": 15, "test": 15, "calib_sweep": 6}  # tasks of each of the 10 categories


def medagentbench(tasks: list[dict], parts: dict[str, list[str]]) -> dict[str, list[str]]:
    """300 tasks, 10 categories of 30, split by patient (`eval_MRN`; two task1 items name none): each patient's tasks
    sit all in calib or all in test, 15 per category on each side. calib_sweep is 6 per category inside calib, for
    threshold sweeps and pilots. The split is Jonathan's, frozen in the committed files, so this checks `parts`
    against the tasks and returns them unchanged. Until 0.5.0 each category was cut with the rule, which put 68 of
    the 98 patients on both sides. Too few to fine-tune."""
    patient = {t["id"]: t.get("eval_MRN") for t in tasks}
    if len(patient) != 300:
        raise ValueError(f"expected 300 tasks, got {len(patient)}")
    if set(parts) != set(MEDAGENTBENCH_PER_CATEGORY):
        raise ValueError(f"expected the parts {sorted(MEDAGENTBENCH_PER_CATEGORY)}, got {sorted(parts)}")
    if sorted(parts["calib"] + parts["test"]) != sorted(patient):
        raise ValueError("calib and test must hold each of the 300 tasks once")
    if not set(parts["calib_sweep"]) <= set(parts["calib"]):
        raise ValueError("calib_sweep must lie inside calib")
    for part, n in MEDAGENTBENCH_PER_CATEGORY.items():
        per = Counter(i.rsplit("_", 1)[0] for i in parts[part])
        if len(per) != 10 or set(per.values()) != {n}:
            raise ValueError(f"{part} must hold {n} tasks of each of the 10 categories, got {dict(per)}")
    shared = ({patient[i] for i in parts["calib"]} & {patient[i] for i in parts["test"]}) - {None}
    if shared:
        raise ValueError(f"{len(shared)} patients have tasks in both calib and test")
    return parts


def load_split(dataset: str, part: str, splits_dir: Path = SPLITS_DIR) -> list[str]:
    """IDs of one part (train, train_extra, val, calib, test; HotpotQA also slr-train, slr-val, slr-test;
    MedAgentBench also calib_sweep) of a dataset, in the committed order."""
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


def _twowiki_dev(path: Path) -> tuple[list[str], list[str]]:
    import pandas as pd
    dev = pd.read_parquet(path, columns=["id", "answer"])  # the mirror names 2Wiki's `_id` column `id`
    return dev["id"].tolist(), dev["answer"].tolist()


def build(sources: Path, out: Path) -> dict:
    train = next((sources / n for n in ("hotpot_train_v1.1.json", "distractor_train.parquet") if (sources / n).exists()),
                 sources / "hotpot_train_v1.1.json")
    files = {
        "hotpotqa_dev": sources / "hotpotqa-dev.json",
        "hotpotqa_train": train,
        "2wikimultihopqa": sources / TWOWIKI_DEV,
        "cwq": sources / "cwq_data_with_ct_0.json",
        "webqsp": sources / "webqsp_data_with_ct_0.json",
        "medagentbench": sources / "test_data_v2.json",
        "hotpotqa_slr": sources / "slr_splits.json",
    }
    dev_ids = [e["_id"] for e in _json(files["hotpotqa_dev"])]
    splits = {
        "hotpotqa": hotpotqa(dev_ids, _hotpotqa_train_ids(files["hotpotqa_train"]))
                    | hotpotqa_slr(dev_ids, _json(files["hotpotqa_slr"])),
        "2wikimultihopqa": twowiki(*_twowiki_dev(files["2wikimultihopqa"])),
        "cwq": kgqa([e["ID"] for e in _json(files["cwq"])]),
        "webqsp": kgqa([e["QuestionId"] for e in _json(files["webqsp"])]),
        "medagentbench": medagentbench(_json(files["medagentbench"]),  # frozen: the committed parts, checked
                                       {p: load_split("medagentbench", p) for p in MEDAGENTBENCH_PER_CATEGORY}),
    }
    manifest = {"seed": SEED, "sources": {k: {"file": v.name, "sha256": _sha(v)} for k, v in files.items()}, "parts": {}}
    for dataset, parts in splits.items():
        seen = defaultdict(set)
        for part, ids in parts.items():
            if seen[scheme(part)] & set(ids):
                raise ValueError(f"{dataset}/{part} overlaps another part")
            seen[scheme(part)] |= set(ids)
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
