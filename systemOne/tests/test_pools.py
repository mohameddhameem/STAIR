import json
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


def test_rankers_pick_their_passages_and_pay_their_scoring(tmp_path):
    from stair import pool_answer as pa

    out = tmp_path / "run"
    (out / "pools").mkdir(parents=True)
    (out / "scores").mkdir()
    pd.DataFrame({"id": ["q0", "q1"], "question": ["a?", "b?"], "answer": ["x", "y"],
                  "titles": [["A", "B", "C", "D"]] * 2, "texts": [["a", "b", "c", "d"]] * 2,
                  "gold_idx": [[3, 1], [3, 1]], "bm25": [[4.0, 3.0, 2.0, 1.0]] * 2}).to_parquet(out / "pools" / "test.parquet")
    pd.DataFrame({"id": ["q0", "q1"], "nanojev": [[0.9, 0.1, 0.8, 0.2]] * 2, "tokens": [100, 100]}).to_parquet(
        out / "scores" / "test.parquet")
    pd.DataFrame({"id": ["q1", "q0"], "msmarco": [[0.0, 0.0, 0.0, 1.0], [0.1, 0.2, 0.3, 0.4]], "tokens": [50, 60]}).to_parquet(
        out / "scores" / "test-msmarco.parquet")
    nj = 2 * pa.NANOJEV_PARAMS_B * 100
    picks = {r: pa.load_split(out, "test", 2, None, r) for r in ("nanojev", "bm25", "msmarco", "gold")}
    assert picks["nanojev"]["passages"][0] == ["a", "c"] and picks["bm25"]["passages"][0] == ["a", "b"]
    assert picks["msmarco"]["passages"].tolist() == [["d", "c"], ["d", "a"]]  # by id, not by file order
    assert pa.load_split(out, "test", 3, None, "gold")["passages"][0] == ["d", "b", "a"]  # gold in nano-jev's order, then its top
    assert picks["nanojev"]["ranker_gflops"].tolist() == [nj, nj] == picks["gold"]["ranker_gflops"].tolist()
    assert picks["bm25"]["ranker_gflops"].tolist() == [0.0, 0.0]
    assert picks["msmarco"]["ranker_gflops"].tolist() == pytest.approx([2 * 0.0334 * 60, 2 * 0.0334 * 50])
    assert (picks["msmarco"]["nanojev_gflops"] == nj).all()  # nano-jev's own cost stays for exp8_eval
    pd.DataFrame({"id": ["q0"], "msmarco": [[0.1, 0.2, 0.3, 0.4]], "tokens": [60]}).to_parquet(
        out / "scores" / "test-msmarco.parquet")
    with pytest.raises(ValueError, match="1 of 2 pools"):
        pa.load_split(out, "test", 2, None, "msmarco")


def test_compare_pairs_every_cell_with_the_base(tmp_path):
    from stair import pool_answer as pa

    out = tmp_path / "run"
    (out / "pools").mkdir(parents=True)
    ids = [f"q{i}" for i in range(4)]
    pd.DataFrame({"id": ids, "answer": ["Kalo", "Mer", "Ann", "Bo"]}).to_parquet(out / "pools" / "test.parquet")
    folder = out / "answers" / "test"
    folder.mkdir(parents=True)
    pd.DataFrame({"id": ids[::-1], "pred": ["Bo", "no", "no", "Kalo"], "units": 10.0}).to_parquet(folder / "base.parquet")
    pd.DataFrame({"id": ids, "pred": ["Kalo", "Mer", "Ann", "Bo"], "units": 30.0}).to_parquet(folder / "all.parquet")
    plan = tmp_path / "plan.toml"
    plan.write_text(f'out = "{out.as_posix()}"\n[[cells]]\nname = "base"\ntop_k = 5\n[[cells]]\nname = "all"\ntop_k = 5\n'
                    'ranker = "gold"\nsamples = 20\nfuse = true\n[[cells]]\nname = "unfinished"\ntop_k = 5\n')
    pa.main(["compare", "--plan", str(plan), "--split", "test", "--base", "base", "--bootstrap", "200"])
    report = json.loads((out / "answers" / "compare-test.json").read_text())
    assert set(report["cells"]) == {"base", "all"} and report["n"] == 4
    assert report["cells"]["base"]["lenient_std"] == 0.5 and report["cells"]["base"]["ci95"] == [0.0, 0.0]
    a = report["cells"]["all"]
    assert a["diff"] == 0.5 and a["units"] == 30.0 and a["ranker"] == "gold" and a["samples"] == 20
    assert 0 <= a["ci95"][0] <= 0.5 <= a["ci95"][1] <= 1
