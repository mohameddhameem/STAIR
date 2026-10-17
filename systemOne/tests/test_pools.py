import math

import numpy as np
import pandas as pd
import pytest

from stair import pools


def example(i, gold_titles, extra):
    titles = gold_titles + [t for t, _ in extra]
    sentences = [[f"{t} is about topic {i}."] for t in gold_titles] + [[x] for _, x in extra]
    return {"id": f"q{i}", "question": f"Which topic links {gold_titles[0]} and {gold_titles[-1]}?", "answer": "x",
            "type": "bridge", "level": "hard",
            "context": {"title": titles, "sentences": sentences},
            "supporting_facts": {"title": gold_titles + [gold_titles[0]], "sent_id": [0] * (len(gold_titles) + 1)}}


def test_bm25_matches_the_lucene_formula():
    docs = ["apple pie recipe", "apple apple tart", "pear crumble"]
    bm = pools.BM25(docs)
    n, avgdl, k1, b = 3, 8 / 3, 1.2, 0.75
    def w(tf, dl, df):
        return math.log1p((n - df + 0.5) / (df + 0.5)) * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / avgdl))
    s = bm.scores("apple tart")
    assert s[0] == pytest.approx(w(1, 3, 2), rel=1e-5)
    assert s[1] == pytest.approx(w(2, 3, 2) + w(1, 3, 1), rel=1e-5)
    assert s[2] == 0
    assert bm.scores("apple apple")[0] == pytest.approx(2 * w(1, 3, 2), rel=1e-5)  # one clause per occurrence
    idx, _ = bm.top("apple tart", 2)
    assert idx.tolist() == [1, 0]


def test_pool_holds_gold_and_dedupes_and_is_seeded():
    exs = [example(i, [f"G{i}a", f"G{i}b"], [(f"D{i}{j}", f"distractor {i} {j} topic") for j in range(8)]) for i in range(6)]
    split = {"context": [e["context"] for e in exs]}
    docs, index = pools.corpus_of(split)
    bm = pools.BM25([f"{t} {x}" for t, x in docs])
    p1 = pools.build_pool(exs[0], bm, docs, index, np.random.default_rng([0, 0]), size=10)
    p2 = pools.build_pool(exs[0], bm, docs, index, np.random.default_rng([0, 0]), size=10)
    assert p1 == p2  # same seed, same pool
    assert p1["n_gold"] == 2 and sorted(p1["titles"][i] for i in p1["gold_idx"]) == ["G0a", "G0b"]
    assert len(set(p1["texts"])) == len(p1["texts"])  # no duplicate text, gold not repeated by BM25
    assert 10 <= len(p1["texts"]) <= 12
    assert len(p1["bm25"]) == len(p1["texts"])


def test_recall_at_k():
    scores = [[0.9, 0.1, 0.8, 0.2], [0.9, 0.8, 0.1, 0.2]]
    gold = [[0, 2], [0, 3]]
    r = pools.recall_at(scores, gold, ks=(1, 2, 4))
    assert r[2]["all_gold"] == 0.5 and r[2]["gold_share"] == 0.75
    assert r[1]["all_gold"] == 0 and r[4]["all_gold"] == 1


def test_agenttts_prompts_and_scoring():
    from stair import pool_answer as pa

    p = pa.qa_prompt("Who?", ["A text.", "B text."])
    assert p == ("You will be provided with a set of references related to the query. Your task is to answer the query "
                 "using the provided references. If the references are insufficient, supplement your response with your "
                 "own knowledge. Here is the query and references: \n Query: Who?\nReferences:\n[1] A text.\n[2] B text.\n.")
    f = pa.fuser_prompt(p, ["x", "y"])
    assert f.count(p) == 2 and f.endswith("Responses from models:\n1. x\n2. y")
    assert pa.lenient("It was directed by Kalo Mer.", "kalo mer") == 1.0
    assert pa.lenient("Kalomer", "kalo") == 0.0  # whole words only
    assert pa.lenient("anything", "") == 0.0
    assert pa.agreement(["Kalo", "Kalo", "Mer"]) == pytest.approx(1 / 3)
