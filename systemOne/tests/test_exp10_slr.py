import numpy as np
import pandas as pd
import pytest

from stair import exp10_slr as e
from stair import slr


def _grid(ok_test, ok_val):
    """A grid with the columns evaluate() reads: verdicts per path, val first, then test."""
    ok = np.vstack([ok_val, ok_test]).astype(bool)
    split = ["val"] * len(ok_val) + ["test"] * len(ok_test)
    return pd.DataFrame({"idx": range(len(ok)), "id": [f"q{i}" for i in range(len(ok))], "split": split,
                         **{f"ok_{p}": ok[:, k] for k, p in enumerate(slr.PATHS)}})


def test_selection_drops_the_query_line():
    assert e.selection("Query: who?\nSelected references: [1] A: a\n[4] B: b") == "[1] A: a\n[4] B: b"
    assert e.selection("Query: who?\nSelected references:") == "(no passages selected)"


def test_labels_are_his_targets():
    df = pd.DataFrame({"y_ret": ["W", "S"], "y_qa_W": ["S", "W"], "y_qa_S": ["W", "W"]})
    assert {k: v.tolist() for k, v in e.labels(df).items()} == {"ret": [0, 1], "qa_W": [1, 0], "qa_S": [0, 0]}
    with pytest.raises(ValueError, match="targets"):
        e.labels(df.assign(y_ret=["W", None]))


def test_pareto_keeps_the_undominated_thresholds():
    points = [(0.5, 0.1, "a"), (0.6, 0.2, "b"), (0.55, 0.3, "c"), (0.6, 0.25, "d"), (0.7, 0.9, "e")]
    assert e.pareto(points) == [("a",), ("b",), ("e",)]


def test_router_pays_the_retrieval_call_and_the_answer_call_it_took():
    price = np.array([[1.0, 2.0, 30.0], [1.0, 2.0, 30.0]])  # ret call, qa under W, qa under S
    r = e.Router("ours", {}, {}, price)
    assert r.extra(np.array([1, 3])).tolist() == [3.0, 31.0]  # WS took the weak branch, SS the strong one
    assert e.Router("his", {}, {}).extra(np.array([0, 3])).tolist() == [0.0, 0.0]  # his routers: not charged


def test_evaluate_compares_ours_with_his_on_his_terms():
    rng = np.random.default_rng(0)
    ok_val, ok_test = rng.random((40, 4)) < [0.5, 0.6, 0.6, 0.8], rng.random((60, 4)) < [0.5, 0.6, 0.6, 0.8]
    df = _grid(ok_test, ok_val)

    def oracle_scores(ok):  # route each question to its cheapest correct path, by weak-first price
        c = e.ev.cheapest_correct_label(ok, np.tile(e.PRICES, (len(ok), 1)))
        return {"ret": (c < 2).astype(float), "qa_W": (c == 0).astype(float), "qa_S": (c == 2).astype(float)}

    keep_weak = lambda n: {k: np.ones(n) for k in e.SCORES}  # noqa: E731
    routers = [e.Router("SLR perfect", oracle_scores(ok_val), oracle_scores(ok_test)),
               e.Router("ours#seed0", keep_weak(40), keep_weak(60), np.full((60, 3), 2.0))]
    report = e.evaluate(df, routers, {"bootstrap": 20})
    best = report["routers"]["SLR perfect"]["peak"]
    rows = np.arange(60)
    c = e.ev.cheapest_correct_label(ok_test, np.tile(e.PRICES, (60, 1)))
    assert best["lenient"] == pytest.approx(ok_test[rows, c].mean())  # a perfect router reaches every solvable question
    ours = report["routers"]["ours#seed0"]["peak"]
    assert ours["paths"]["WW"] == 60 and ours["price"] == 80 and ours["judge_price"] == 4.0
    assert report["routers"]["ours#seed0"]["aiq_with_judge_price"] <= report["routers"]["ours#seed0"]["aiq"]
    assert set(report["aiq_diff"]) == {"ours#seed0 - SLR perfect", "ours#seed0 - fixed paths"}
    assert report["aiq_diff"]["ours#seed0 - SLR perfect"]["ci95"][1] < 0
    clean = e.evaluate(df, routers, {"bootstrap": 20}, keep=np.arange(60) < 30)
    assert clean["n_test"] == 30
