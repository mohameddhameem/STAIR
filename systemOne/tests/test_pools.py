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


def test_jonathans_short_answer_prompt():
    from stair import pool_answer as pa

    block = pa.slr_block("Who?", ["A: a text", "B: b text"])
    assert block == "Query: Who?\nSelected references: [1] A: a text [2] B: b text"  # his build_block layout
    assert pa.slr_qa_prompt("Who?", block) == (
        "Your task is to answer the query using the provided references. If the references are insufficient, "
        "supplement your response with your own knowledge. Answer with ONLY the exact answer phrase - no explanation, "
        "no full sentence.\nQuery: Who?\n" + block)  # run_labels4.py qa_prompt, verbatim
    assert pa.slr_block("Who?", []).endswith("Selected references: (none)")


def test_pool_answer_reuses_earlier_answers_in_the_splits_order(tmp_path):
    from stair import pool_answer as pa

    out, old = tmp_path / "run", tmp_path / "old" / "test"
    ids = ["q0", "q1", "q2"]
    (out / "pools").mkdir(parents=True)
    (out / "scores").mkdir()
    pd.DataFrame({"id": ids, "question": ["a?", "b?", "c?"], "answer": ["x", "y", "z"],
                  "titles": [["T"]] * 3, "texts": [["t"]] * 3}).to_parquet(out / "pools" / "test.parquet")
    pd.DataFrame({"id": ids, "nanojev": [[1.0]] * 3, "tokens": [100, 200, 300]}).to_parquet(out / "scores" / "test.parquet")
    old.mkdir(parents=True)
    pd.DataFrame({"id": ["q2", "q0", "q1"], "pred": ["z", "x", "no"], "prompt_tokens": [10, 10, 10],
                  "gen_tokens": [5, 5, 5], "fuse_prompt_tokens": [0, 0, 0], "fuse_gen_tokens": [0, 0, 0],
                  "units": [9.0, 9.0, 9.0]}).to_parquet(old / "cell.parquet")
    plan = {"out": str(out), "reuse_answers": [str(old)], "cells": [{"name": "cell", "top_k": 1, "params": 3.0}]}
    args = type("Args", (), {"only": [], "split": "test", "limit": None, "chunk_size": 2, "device": "cpu"})()
    pa.cmd_run(args, plan)  # every question is reused: no model is loaded
    merged = pd.read_parquet(out / "answers" / "test" / "cell.parquet")
    assert merged["id"].tolist() == ids and merged["pred"].tolist() == ["x", "no", "z"]
    nanojev = 2 * pa.NANOJEV_PARAMS_B * np.array([100, 200, 300])
    assert merged["gflops"].to_numpy() == pytest.approx(2 * 3.0 * 15 + nanojev)  # recosted from the tokens, this split


def test_h2_block_is_nanojevs_top_k_with_titles():
    from stair import pool_answer as pa

    df = pd.DataFrame({"question": ["Who?"], "titles": [["A", "B", "C"]], "texts": [["a", "b", "c"]],
                       "nanojev": [[0.1, 0.9, 0.5]]})
    assert pa.passage_blocks(df, {"top_k": 2}) == ["Query: Who?\nSelected references: [1] B: b [2] C: c"]
