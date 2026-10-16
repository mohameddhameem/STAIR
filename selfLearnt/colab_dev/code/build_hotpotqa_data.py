"""
Build the HotpotQA retrieval-QA splits used by AgentTTS (archon_hotpotqa.py).

The original repo builds them with data/retrieval_QA_data_generator.py, which needs a
running Elasticsearch server. This script reproduces the same logic with an in-memory
BM25 index (scikit-learn + scipy), so it runs on a plain Colab CPU in a couple of minutes.

Output format (identical keys to the repo):
  {question, answer, ret_answers, ctxs_candidate, ctxs_gt}

Files written (names match what archon_hotpotqa.py loads):
  tts_test.json                   -> dev/search split, examples [0:100]   (repo uses the first 50)
  tts_test_2_samples500.json      -> test split,       examples [500:1000]
"""
import argparse
import json
import os
import random
import re

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer

_TOKEN = re.compile(r"(?u)\b\w\w+\b")


class BM25:
    """Okapi BM25 over a sparse term matrix (k1=1.2, b=0.75, same defaults as Elasticsearch)."""

    def __init__(self, docs, k1=1.2, b=0.75):
        self.vec = CountVectorizer(lowercase=True, token_pattern=_TOKEN.pattern, dtype=np.float32)
        tf = self.vec.fit_transform(docs).tocsc()  # docs x terms
        n_docs = tf.shape[0]
        dl = np.asarray(tf.sum(axis=1)).ravel()
        avgdl = dl.mean()
        df = np.diff(tf.indptr)  # document frequency per term (csc)
        self.idf = np.log(1 + (n_docs - df + 0.5) / (df + 0.5)).astype(np.float32)
        # pre-compute the BM25 weight for every (doc, term) entry
        tf = tf.tocoo()
        denom = tf.data + k1 * (1 - b + b * dl[tf.row] / avgdl)
        w = tf.data * (k1 + 1) / denom * self.idf[tf.col]
        self.W = sparse.csr_matrix((w, (tf.row, tf.col)), shape=tf.shape).T.tocsr()  # terms x docs

    def search(self, query, size):
        q = self.vec.transform([query])
        q.data[:] = 1.0  # each query term counted once (multi_match-like)
        scores = (q @ self.W).toarray().ravel()
        if size >= len(scores):
            return np.argsort(-scores)
        top = np.argpartition(-scores, size)[:size]
        return top[np.argsort(-scores[top])]


def load_corpus(data):
    """Same corpus as data/HotpotQA/preprocess.py: every context paragraph of the dev set (deduplicated)."""
    seen, texts = set(), []
    for item in data:
        for title, sents in item["context"]:
            text = " ".join(sents)
            if text not in seen:
                seen.add(text)
                texts.append(text)
    return texts


def build_split(data, bm25, corpus, start, count, retrieve_number, seed):
    rng = random.Random(seed)
    out = []
    for ex in data[start:start + count]:
        ctx = {t: " ".join(s) for t, s in ex["context"]}
        gt = []
        for title, _ in ex["supporting_facts"]:
            if ctx[title] not in gt:
                gt.append(ctx[title])
        ctxs = gt.copy()
        for i in bm25.search(ex["question"], retrieve_number):
            if corpus[i] not in ctxs:
                ctxs.append(corpus[i])
        rng.shuffle(ctxs)
        out.append({
            "question": ex["question"],
            "answer": ex["answer"],
            "ret_answers": [ctxs.index(c) for c in gt],
            "ctxs_candidate": ctxs,
            "ctxs_gt": gt,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hotpot_dev", required=True, help="path to data/HotpotQA/hotpotqa-dev.json from the repo")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--retrieve_number", type=int, default=100, help="BM25 hits added to the gold paragraphs (repo: 100)")
    ap.add_argument("--dev_count", type=int, default=100)
    ap.add_argument("--test_start", type=int, default=500)
    ap.add_argument("--test_count", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.hotpot_dev, encoding="utf-8") as f:
        data = json.load(f)
    corpus = load_corpus(data)
    print(f"HotpotQA dev questions: {len(data)} | corpus paragraphs: {len(corpus)}")
    bm25 = BM25(corpus)
    print("BM25 index built.")

    os.makedirs(args.out_dir, exist_ok=True)
    dev = build_split(data, bm25, corpus, 0, args.dev_count, args.retrieve_number, args.seed)
    test = build_split(data, bm25, corpus, args.test_start, args.test_count, args.retrieve_number, args.seed + 1)
    for name, split in [("tts_test.json", dev), (f"tts_test_2_samples{args.test_count}.json", test)]:
        path = os.path.join(args.out_dir, name)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(split, f, indent=1, ensure_ascii=False)
        n_c = np.mean([len(e["ctxs_candidate"]) for e in split])
        chars = np.mean([sum(len(c) for c in e["ctxs_candidate"]) for e in split])
        print(f"wrote {path}: {len(split)} questions, {n_c:.0f} candidates/question, ~{chars/4:.0f} tokens of references/question")


if __name__ == "__main__":
    main()
