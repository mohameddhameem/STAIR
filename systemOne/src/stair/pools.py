"""Experiment 8, phase 0 (docs/experiment-8-plan.md): AgentTTS-style candidate pools and their retrieval.

AgentTTS (Wang et al., NeurIPS 2025) gives each HotpotQA question a pool of ~100 candidate
paragraphs: the gold (supporting-fact) paragraphs, then the BM25 top 100 for the question from all
paragraphs of the split, deduplicated by text and shuffled. An LLM then retrieves from the pool.
This module rebuilds those pools (with a fixed seed: AgentTTS's shuffle had none), scores every
candidate with nano-jev, and measures how often a ranker's top k holds the gold paragraphs.

    pools build  --source validation --start 500 --count 500 --name test     # AgentTTS's test questions
    pools build  --source train --ids results/gpu/router_train.parquet --count 3000 --name router_train
    pools score  --name test --scorer sdmlai/nano-jev@v1.0 [--device cpu]     # resumable, in chunks
    pools recall --name test --exclude data/hotpotqa/nanojev_validation_ids.txt

BM25 follows Lucene's formula (Elasticsearch's default: k1 = 1.2, b = 0.75) over lowercase word
tokens of "title text". AgentTTS used Elasticsearch `multi_match` over the two fields, which scores
each field apart and keeps the best; one combined field is an approximation that only changes which
distractors enter a pool (gold always does).
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance

log = logging.getLogger(__name__)

POOL_SIZE = 100
NANOJEV_PARAMS_B = 0.0334
AGENTTTS_UNIT_GFLOPS = 2 * 3.0 * (128 + 64)  # one 3B pass on the cheapest task (128 prompt + 64 generated tokens)
KS = (1, 2, 3, 5, 10, 20)


class BM25:
    """Lucene BM25 over a fixed corpus, as a sparse weight matrix (documents x terms)."""

    def __init__(self, docs: list[str], k1: float = 1.2, b: float = 0.75):
        from sklearn.feature_extraction.text import CountVectorizer

        self.vectorizer = CountVectorizer(lowercase=True, token_pattern=r"(?u)\b\w+\b", dtype=np.float32)
        tf = self.vectorizer.fit_transform(docs).tocsr()  # documents x terms
        n = tf.shape[0]
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        self.idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        lengths = np.asarray(tf.sum(axis=1)).ravel()
        norm = k1 * (1 - b + b * lengths / lengths.mean())
        rows = np.repeat(np.arange(n), np.diff(tf.indptr))
        tf.data = tf.data * (k1 + 1) / (tf.data + norm[rows]) * self.idf[tf.indices]
        self.weights = tf.tocsc()  # column slices per query term

    def scores(self, query: str) -> np.ndarray:
        q = self.vectorizer.transform([query])  # a repeated query term counts once per occurrence (one Lucene clause each)
        if q.nnz == 0:
            return np.zeros(self.weights.shape[0], dtype=np.float32)
        return np.asarray(self.weights[:, q.indices] @ q.data).ravel()

    def top(self, query: str, k: int) -> tuple[np.ndarray, np.ndarray]:
        s = self.scores(query)
        k = min(k, len(s))
        idx = np.argpartition(-s, k - 1)[:k]
        idx = idx[np.argsort(-s[idx], kind="stable")]
        return idx, s


def paragraphs(example: dict) -> list[tuple[str, str]]:
    """(title, text) per context paragraph, text joined as AgentTTS's preprocess.py does."""
    ctx = example["context"]
    return [(t, " ".join(s)) for t, s in zip(ctx["title"], ctx["sentences"])]


def corpus_of(split) -> tuple[list[tuple[str, str]], dict[tuple[str, str], int]]:
    """Every distinct (title, text) paragraph of a dataset split, and its index."""
    docs, index = [], {}
    for ctx in split["context"]:
        for t, s in zip(ctx["title"], ctx["sentences"]):
            key = (t, " ".join(s))
            if key not in index:
                index[key] = len(docs)
                docs.append(key)
    return docs, index


def build_pool(example: dict, bm25: BM25, docs: list[tuple[str, str]], index: dict, rng: np.random.Generator,
               size: int = POOL_SIZE) -> dict:
    """Gold paragraphs + the BM25 top `size` (deduplicated by text), shuffled; as AgentTTS builds it."""
    by_title = dict(paragraphs(example))
    gold = []
    for t in dict.fromkeys(example["supporting_facts"]["title"]):  # in order, once each
        if t in by_title and (t, by_title[t]) not in gold:
            gold.append((t, by_title[t]))
    top, scores = bm25.top(example["question"], size)
    pool, seen = list(gold), {text for _, text in gold}
    for i in top:
        if docs[i][1] not in seen:
            pool.append(docs[i])
            seen.add(docs[i][1])
    order = rng.permutation(len(pool))
    pool = [pool[i] for i in order]
    gold_texts = {text for _, text in gold}
    return {
        "id": example["id"], "question": example["question"], "answer": example["answer"],
        "type": example["type"], "level": example["level"],
        "titles": [t for t, _ in pool], "texts": [x for _, x in pool],
        "gold_idx": [i for i, (_, x) in enumerate(pool) if x in gold_texts],
        "n_gold": len(gold),
        "bm25": [float(scores[index[p]]) if p in index else 0.0 for p in pool],
    }


# ---------------------------------------------------------------------------------------------


def cmd_build(args) -> None:
    from datasets import load_dataset

    split = load_dataset("hotpotqa/hotpot_qa", "distractor", split=args.source)
    if args.ids:
        wanted = (json.loads(args.ids.read_text())[args.name] if args.ids.suffix == ".json"  # a make-splits key
                  else pd.read_parquet(args.ids, columns=["id"])["id"].tolist())[: args.count]
        pos = {qid: i for i, qid in enumerate(split["id"])}
        rows = [pos[q] for q in wanted]
    else:
        rows = list(range(args.start, min(args.start + args.count, len(split))))
    log.info(f"{args.name}: {len(rows)} questions from {args.source}")
    start = time.perf_counter()
    docs, index = corpus_of(split)
    bm25 = BM25([f"{t} {x}" for t, x in docs])
    log.info(f"BM25 over {len(docs):,} distinct {args.source} paragraphs, built in {time.perf_counter() - start:.0f} s")
    out = []
    for j, r in enumerate(rows):
        rng = np.random.default_rng([args.seed, r])  # per question, so a subset rebuilds identically
        out.append(build_pool(split[r], bm25, docs, index, rng, args.pool_size))
        if (j + 1) % 500 == 0:
            log.info(f"  {j + 1}/{len(rows)} pools")
    df = pd.DataFrame(out)
    target = args.out / "pools" / f"{args.name}.parquet"
    target.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(target)
    sizes = df["titles"].map(len)
    log.info(f"saved {target}: pool size mean {sizes.mean():.1f} (min {sizes.min()}, max {sizes.max()}), "
             f"gold per question {df['n_gold'].mean():.2f}, questions with < 2 gold {(df['n_gold'] < 2).sum()}")
    provenance.record(args.out, "pools-build", name=args.name, source=args.source, rows=len(rows), seed=args.seed,
                      corpus=len(docs), **provenance.environment())


def cmd_score(args) -> None:
    from stair.evidence import CrossEncoderScorer

    pools = pd.read_parquet(args.out / "pools" / f"{args.name}.parquet")
    parts = args.out / "scores" / f"{args.name}.parts"
    parts.mkdir(parents=True, exist_ok=True)
    chunks = [pools.iloc[i: i + args.chunk_size] for i in range(0, len(pools), args.chunk_size)]
    todo = [c for c in range(len(chunks)) if not (parts / f"{c:05d}.parquet").exists()]
    log.info(f"{args.name}: {len(pools)} pools, {len(chunks)} chunks, {len(todo)} to score")
    if todo:
        scorer = CrossEncoderScorer(args.scorer, device=args.device)
        provenance.record(args.out, "pools-score", name=args.name, scorer=args.scorer,
                          revision=provenance.hub_revision(args.scorer), device=scorer.device,
                          **provenance.environment())
    for c in todo:
        rows, start = [], time.perf_counter()
        for _, p in chunks[c].iterrows():
            texts = [f"{t}: {x}" for t, x in zip(p["titles"], p["texts"])]
            tokens0, trunc0, t0 = scorer.tokens_seen, scorer.truncated, time.perf_counter()
            s = np.concatenate([scorer.score(p["question"], texts[i: i + args.batch])
                                for i in range(0, len(texts), args.batch)])  # independent per paragraph
            rows.append({"id": p["id"], "nanojev": s.astype(float).tolist(), "tokens": scorer.tokens_seen - tokens0,
                         "truncated": scorer.truncated - trunc0, "seconds": time.perf_counter() - t0})
        pd.DataFrame(rows).to_parquet(parts / f"{c:05d}.parquet")
        log.info(f"  chunk {c + 1}/{len(chunks)}: {(time.perf_counter() - start) / len(rows):.2f} s/question")
    done = sorted(parts.glob("*.parquet"))
    if len(done) == len(chunks):
        merged = pd.concat([pd.read_parquet(f) for f in done], ignore_index=True)
        merged.to_parquet(args.out / "scores" / f"{args.name}.parquet")
        log.info(f"saved {args.out / 'scores' / f'{args.name}.parquet'}")


def recall_at(scores: list, gold: list, ks=KS) -> dict:
    """Share of questions whose top-k (by score) holds ALL their gold paragraphs, and the mean share of gold found."""
    out = {}
    for k in ks:
        both, frac = [], []
        for s, g in zip(scores, gold):
            if not len(g):
                continue
            top = set(np.argsort(-np.asarray(s), kind="stable")[:k].tolist())
            hit = [i in top for i in g]
            both.append(all(hit))
            frac.append(np.mean(hit))
        out[k] = {"all_gold": float(np.mean(both)), "gold_share": float(np.mean(frac))}
    return out


def cmd_recall(args) -> None:
    pools = pd.read_parquet(args.out / "pools" / f"{args.name}.parquet")
    scores = pd.read_parquet(args.out / "scores" / f"{args.name}.parquet")
    df = pools.merge(scores, on="id", validate="1:1")
    exclude = set()
    for path in args.exclude:
        exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    report = {"name": args.name, "n": len(df), "pool_size_mean": float(df["titles"].map(len).mean()),
              "nanojev_cost": {"tokens_per_question": float(df["tokens"].mean()),
                               "gflops_per_question": float(2 * NANOJEV_PARAMS_B * df["tokens"].mean()),
                               "agenttts_units": float(2 * NANOJEV_PARAMS_B * df["tokens"].mean() / AGENTTTS_UNIT_GFLOPS),
                               "seconds_per_question": float(df["seconds"].mean()),
                               "pairs_truncated_share": float(df["truncated"].sum() / df["titles"].map(len).sum())},
              "subsets": {}}
    subsets = {"all": df, "bridge": df[df["type"] == "bridge"], "comparison": df[df["type"] == "comparison"]}
    if exclude:
        subsets["clean"] = df[~df["id"].isin(exclude)]
    for name, d in subsets.items():
        if d.empty:
            continue
        report["subsets"][name] = {"n": len(d), **{ranker: recall_at(d[col].tolist(), d["gold_idx"].tolist())
                                                   for ranker, col in (("nanojev", "nanojev"), ("bm25", "bm25"))}}
    target = args.out / f"recall-{args.name}.json"
    target.write_text(json.dumps(report, indent=1))
    log.info(f"{args.name}: {len(df)} pools of {report['pool_size_mean']:.1f} | nano-jev "
             f"{report['nanojev_cost']['gflops_per_question']:.0f} GFLOPs = {report['nanojev_cost']['agenttts_units']:.2f} "
             f"AgentTTS units per question, {report['nanojev_cost']['seconds_per_question']:.1f} s per question "
             f"(device: see provenance.jsonl)")
    for name, r in report["subsets"].items():
        line = " | ".join(f"@{k}: nano-jev {r['nanojev'][k]['all_gold']:.3f} BM25 {r['bm25'][k]['all_gold']:.3f}"
                          for k in (2, 3, 5, 10))
        log.info(f"  {name:<10} n={r['n']:<5} both gold in top-k  {line}")
    log.info(f"saved {target}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="step", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", choices=["train", "validation"], required=True)
    b.add_argument("--start", type=int, default=0, help="first row (file order) when --ids is not given")
    b.add_argument("--count", type=int, required=True)
    b.add_argument("--ids", type=Path, help="parquet whose `id` column (in order) picks the questions, or a make-splits "
                                            "JSON whose key --name does")
    b.add_argument("--name", required=True)
    b.add_argument("--pool-size", type=int, default=POOL_SIZE)
    b.add_argument("--seed", type=int, default=0)
    s = sub.add_parser("score")
    s.add_argument("--name", required=True)
    s.add_argument("--scorer", default="sdmlai/nano-jev@v1.0")
    s.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    s.add_argument("--chunk-size", type=int, default=25, help="questions per resumable chunk")
    s.add_argument("--batch", type=int, default=16, help="paragraphs per forward pass (memory only)")
    r = sub.add_parser("recall")
    r.add_argument("--name", required=True)
    r.add_argument("--exclude", type=Path, action="append", default=[], help="ids for a separate clean subset")
    for p in (b, s, r):
        p.add_argument("--out", type=Path, default=Path("runs/exp8"))
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)
    {"build": cmd_build, "score": cmd_score, "recall": cmd_recall}[args.step](args)


if __name__ == "__main__":
    main()
