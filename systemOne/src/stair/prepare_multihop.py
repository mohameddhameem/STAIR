"""Prepare 2WikiMultiHopQA and MuSiQue in prepare-hotpotqa's schema (Jev novelty E2), so run-experts, the judges and
the evidence scorer read them unchanged.

    2wikimultihopqa  framolfese/2WikiMultihopQA: the official train / dev (167,454 / 12,576; Apache-2.0 repository),
                     already HotpotQA-shaped (10 paragraphs as title + sentences, supporting facts). No alias file.
                     Not the team's stair_common 2Wiki parts: their source, AgentTTS's tts_train.json, is the official
                     test set, whose answers are hidden (all 12,576 empty), so it cannot be scored.
    musique          dgslibisey/MuSiQue: MuSiQue-Ans v1.0 train / dev (19,938 / 2,417; CC BY 4.0). Each of the 20
                     paragraphs becomes one sentence; `type` is the hop pattern from the id (2hop, 3hop1, ...).

Writes data/<dataset>/{train,dev}.parquet with id, question, answer, answers (gold + aliases: the secondary ruler
only), type, level, supporting_context, full_context, num_context_paragraphs, num_supporting_facts, context_titles,
context_sentences, sp_titles, sp_sent_ids, gold_titles.

    prepare-multihop --dataset 2wikimultihopqa   # login node: download + light export (as prepare-hotpotqa)
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from stair.prepare_hotpotqa import flatten_example

log = logging.getLogger(__name__)

SOURCES = {"2wikimultihopqa": "framolfese/2WikiMultihopQA", "musique": "dgslibisey/MuSiQue"}
SIZES = {"2wikimultihopqa": {"train": 167_454, "dev": 12_576}, "musique": {"train": 19_938, "dev": 2_417}}
HF_SPLITS = {"train": "train", "dev": "validation"}


def twowiki_row(ex: dict) -> dict:
    row = flatten_example({**ex, "level": ""})
    return {**row, "answers": [ex["answer"]]}


def musique_row(ex: dict) -> dict:
    paragraphs = ex["paragraphs"]
    titles = [p["title"] for p in paragraphs]
    sentences = [[p["paragraph_text"].strip()] for p in paragraphs]
    support = [p["title"] for p in paragraphs if p["is_supporting"]]
    return {
        "id": ex["id"],
        "question": ex["question"],
        "answer": ex["answer"],
        "answers": [ex["answer"], *[a for a in ex.get("answer_aliases") or [] if a != ex["answer"]]],
        "type": ex["id"].split("__")[0],
        "level": "",
        "supporting_context": " ".join(p["paragraph_text"].strip() for p in paragraphs if p["is_supporting"]),
        "full_context": "\n\n".join(f"[{t}] {s[0]}" for t, s in zip(titles, sentences)),
        "num_context_paragraphs": len(paragraphs),
        "num_supporting_facts": len(support),
        "context_titles": titles,
        "context_sentences": sentences,
        "sp_titles": support,
        "sp_sent_ids": [0] * len(support),
        "gold_titles": list(dict.fromkeys(support)),
    }


ROW = {"2wikimultihopqa": twowiki_row, "musique": musique_row}


def convert(dataset: str, examples) -> pd.DataFrame:
    df = pd.DataFrame([ROW[dataset](ex) for ex in examples])
    if df["id"].duplicated().any():
        raise ValueError(f"{dataset}: duplicate ids")
    return df


def check(dataset: str, frames: dict[str, pd.DataFrame]) -> None:
    """Row counts as published, every question with a gold answer, and train and dev disjoint."""
    for part, n in SIZES[dataset].items():
        if len(frames[part]) != n:
            raise ValueError(f"{dataset} {part}: {len(frames[part])} rows, expected {n}")
        if (empty := int((frames[part]["answer"].astype(str).str.strip() == "").sum())):
            raise ValueError(f"{dataset} {part}: {empty} questions without a gold answer (a hidden-answer test set?)")
    if overlap := set(frames["train"]["id"]) & set(frames["dev"]["id"]):
        raise ValueError(f"{dataset}: {len(overlap)} ids in both train and dev")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", choices=sorted(SOURCES), required=True)
    parser.add_argument("--out", type=Path, help="default data/<dataset>")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    from datasets import load_dataset

    out = args.out or Path("data") / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    frames = {}
    for part, hf_split in HF_SPLITS.items():
        frames[part] = convert(args.dataset, load_dataset(SOURCES[args.dataset], split=hf_split))
        log.info(f"{args.dataset} {part}: {len(frames[part])} questions | types {frames[part]['type'].value_counts().to_dict()}")
    check(args.dataset, frames)
    for part, df in frames.items():
        df.to_parquet(out / f"{part}.parquet", index=False)
        log.info(f"  saved {out / f'{part}.parquet'}")


if __name__ == "__main__":
    main()
