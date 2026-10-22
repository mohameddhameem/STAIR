import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stair_common.cost import AGENTTTS_TABLE5_HOTPOTQA, units

from stair import exp8_eval as ev

CONFIGS = Path(__file__).parent.parent / "configs"


def test_cheapest_correct_label():
    cost = np.tile([1.0, 5.0, 10.0], (5, 1))
    ok = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool)
    assert ev.cheapest_correct_label(ok, cost).tolist() == [0, 1, 2, 0, 0]  # none correct: the cheapest option
    cost[1] = [1.0, 12.0, 10.0]  # the label follows the cost, not the option's position
    assert ev.cheapest_correct_label(ok, cost)[1] == 2


def test_cumulative_choice_walks_up_the_options():
    p = np.array([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
    assert ev.cumulative_choice(p, 0.0).tolist() == [0, 0]
    assert ev.cumulative_choice(p, 0.5).tolist() == [0, 2]
    assert ev.cumulative_choice(p, 0.9).tolist() == [1, 2]
    assert ev.cumulative_choice(p, 1.0).tolist() == [2, 2]  # never past the last option


def test_judge_curve_cost_never_falls_as_t_rises():
    rng = np.random.default_rng(0)
    p = rng.dirichlet([1, 1, 1], 200)
    ok = rng.random((200, 3)) < [0.4, 0.6, 0.8]
    cost = np.tile([3.0, 13.0, 120.0], (200, 1))
    pts = ev.judge_points(ok, cost, p)
    costs = [q["cost"] for q in pts]
    assert costs == sorted(costs) and len(pts) == len(ev.THRESHOLDS)
    assert pts[0]["cost"] == pytest.approx(3.0)  # t = 0: the cheapest option everywhere
    assert ev.judge_points(ok, cost, p, extra=2.0)[0]["cost"] == pytest.approx(5.0)


def test_a_judge_that_read_the_cheapest_answer_pays_for_it():
    ok = np.ones((2, 3), dtype=bool)
    cost = np.array([[3.0, 10.0, 50.0], [3.0, 10.0, 50.0]])
    choice = np.array([0, 2])
    assert ev.point(ok, cost, choice)["cost"] == pytest.approx((3 + 50) / 2)
    assert ev.point(ok, cost, choice, post=True)["cost"] == pytest.approx((3 + 50 + 3) / 2)  # option 0 ran first
    # every option's cost includes a ranking call of 1 that ran once: option 0 adds only its answer (2)
    assert ev.point(ok, cost, choice, post=True, shared=1.0)["cost"] == pytest.approx((3 + 50 + 2) / 2)


def test_strongest_option_is_the_most_accurate_not_the_last():
    # the last option (70B under Eq. 8) costs most but is less accurate than option 2 (3B x40 fused)
    ok = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [1, 1, 1, 0], [0, 0, 1, 1]], dtype=bool)
    assert ev.strongest_option(ok) == 2
    assert ev.strongest_option(np.array([[1, 0, 1], [1, 0, 1]], dtype=bool)) == 2  # never the cheapest itself


def test_agenttts_reference_costs_are_in_stair_common_units():
    # Table 5 is AgentTTS's normalised budget (Eq. 3-4): the 3B samples, sharing one 128-token prompt, of equal FLOPs,
    # which is 3u - 2 for a call of u units. Its HotpotQA prices at its Table 4 lengths reproduce that exactly.
    params = {"Qwen2.5-72B": 72, "Qwen2.5-32B": 32, "Qwen2.5-7B": 7, "Llama-3.1-70B": 70, "Llama-3.1-8B": 8,
              "Llama-3.2-3B": 3}
    lengths = {"retrieve": (2048, 128), "answer": (256, 64)}
    for task, prices in AGENTTTS_TABLE5_HOTPOTQA.items():
        for model, price in prices.items():
            assert round(3 * units(params[model], *lengths[task]) - 2) == price
    assert round(3 * units(3, 256, 10 * 64) - 2) == 12  # Table 7, 10 samples: the prompt counts once (Eq. 8)
    retrieve = units(72, 2048, 128)
    recipe = retrieve + units(3, 256, 40 * 64) + units(3, 256 + 40 * 64, 64)  # 40 samples, then a fusion reading them
    for path in sorted(CONFIGS.glob("exp8-judge*-common.toml")):
        plan = tomllib.loads(path.read_text(encoding="utf-8"))
        assert (plan["strong_units"], plan["agenttts_units"]) == (round(retrieve + units(70, 256, 64)), round(recipe))


def test_score_points_between_two_options():
    ok = np.array([[0, 1], [1, 1], [0, 1], [1, 0]], dtype=bool)
    cost = np.tile([2.0, 10.0], (4, 1))
    score = np.array([0.9, 0.1, 0.8, 0.2])
    first, last = ev.score_points(ok, cost, score, 0, 1, post=False)[0], ev.score_points(ok, cost, score, 0, 1, post=False)[-1]
    assert (first["f1"], first["cost"]) == (0.5, 2.0) and (last["f1"], last["cost"]) == (0.75, 10.0)
    assert ev.score_points(ok, cost, score, 0, 1, post=True)[-1]["cost"] == 12.0  # post: escalated pay both calls (2 + 10)
    assert ev.score_points(ok, cost, score, 0, 1, post=True, shared=1.0)[-1]["cost"] == 11.0  # the shared 1 once
    half = [p for p in ev.score_points(ok, cost, score, 0, 1, post=False) if p["rate"] == 0.5][0]
    assert half["f1"] == 1.0  # the two highest scores are the ones the strong option fixes


def test_operating_points():
    pts = [{"f1": 0.5, "cost": 1.0}, {"f1": 0.6, "cost": 5.0}, {"f1": 0.7, "cost": 20.0}, {"f1": 0.71, "cost": 90.0}]
    assert ev.choose_threshold(pts, budget=10.0) == pts[1]
    assert ev.choose_threshold(pts, budget=0.5) == pts[0]  # nothing fits: the cheapest point
    assert ev.choose_threshold(pts, floor=0.70) == pts[2]
    assert ev.choose_threshold(pts, floor=0.99) == pts[3]  # unreachable: the most accurate point


def test_bootstrap_pairs_policies():
    rng = np.random.default_rng(1)
    ok = rng.random((100, 2)) < [0.3, 0.8]
    cost = np.tile([1.0, 10.0], (100, 1))
    better = lambda ix: ev.always_points(ok[ix], cost[ix])
    boot = ev.bootstrap({"a": better, "b": better}, 100, 30, 0, (1.0, 10.0))
    assert ev.diff(boot, "a", "b")["p_le_0"] == 1.0  # identical policies: never better
    assert boot["aiq_ci95"]["a"][0] <= boot["aiq_ci95"]["a"][1]
    again = ev.bootstrap({"a": better}, 100, 30, 0, (1.0, 10.0))
    assert again["aiq_ci95"]["a"] == boot["aiq_ci95"]["a"]  # seeded


def test_evidence_features_and_logistic():
    f = ev.evidence_features(list(np.linspace(1, 0, 100)))
    assert f.shape == (7,) and f[0] > f[1] > f[2] and f[3] == pytest.approx(f[0] - f[1])
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 3))
    y = (X[:, 0] > 0).astype(int)
    model, C = ev.fit_logistic(X, y, 0)
    assert model.predict_proba(X)[:, 1][y == 1].mean() > 0.7 and C in (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)


def _out_dir(tmp_path, n=6):
    """A minimal runs/exp8 layout: pools, nano-jev scores, two answer cells and their re-scored copies."""
    out = tmp_path / "exp8"
    for d in ("pools", "scores", "answers/test", "rescore/test"):
        (out / d).mkdir(parents=True)
    ids = [f"q{i}" for i in range(n)]
    pd.DataFrame({"id": ids, "question": "Who?", "answer": "Kalo", "titles": [["A", "B"]] * n,
                  "texts": [["Kalo did it.", "Other."]] * n, "gold_idx": [[0]] * n}).to_parquet(out / "pools" / "test.parquet")
    pd.DataFrame({"id": ids, "nanojev": [[0.9, 0.1]] * n, "tokens": [100] * n}).to_parquet(out / "scores" / "test.parquet")
    for cell, lenient, graded in (("cheap", [1, 1, 0, 0, 1, 0], [0, 1, 0, 0, 1, 0]), ("strong", [1, 1, 1, 1, 1, 0], [1, 1, 1, 0, 0, 0])):
        pd.DataFrame({"id": ids, "lenient": np.array(lenient, float), "units": 1.0 if cell == "cheap" else 9.0,
                      "entropy": 0.5, "pred": "x"}).to_parquet(out / "answers" / "test" / f"{cell}.parquet")
        pd.DataFrame({"id": ids, "correct": np.array(graded, float), "ext_em": 0.0}).to_parquet(
            out / "rescore" / "test" / f"{cell}.parquet")
    return out


def test_load_tables_uses_the_chosen_metric(tmp_path):
    out = _out_dir(tmp_path)
    _, ok_l, cost = ev.load_tables(out, "test", ["cheap", "strong"], 2)
    _, ok_g, _ = ev.load_tables(out, "test", ["cheap", "strong"], 2, metric="correct", rescore_dir=out / "rescore")
    assert ok_l[:, 0].tolist() == [True, True, False, False, True, False]
    assert ok_g[:, 0].tolist() == [False, True, False, False, True, False]
    assert ok_g[:, 1].tolist() == [True, True, True, False, False, False]
    assert cost[0].tolist() == [1.0, 9.0]
    with pytest.raises(ValueError, match="rescore_dir"):
        ev.load_tables(out, "test", ["cheap", "strong"], 2, metric="correct")
    (out / "rescore" / "test" / "strong.parquet").unlink()
    with pytest.raises(FileNotFoundError):
        ev.load_tables(out, "test", ["cheap", "strong"], 2, metric="correct", rescore_dir=out / "rescore")


def test_load_tables_standard_ruler(tmp_path):
    out = _out_dir(tmp_path)
    # gold "Kalo": contained; "about" is the ruler's quantity guard; whole words only; empty; case; wrong
    preds = ["It was Kalo.", "about Kalo and more", "Kalorama", "", "kalo", "no idea"]
    for cell in ("cheap", "strong"):
        f = out / "answers" / "test" / f"{cell}.parquet"
        pd.read_parquet(f).assign(pred=preds).to_parquet(f)
    _, ok, cost = ev.load_tables(out, "test", ["cheap", "strong"], 2, metric="lenient_std")
    assert ok[:, 0].tolist() == [True, False, False, False, True, False]
    assert cost[0].tolist() == [1.0, 9.0]  # no commit step under the response-level ruler


def test_the_ranking_every_option_includes_is_paid_once(tmp_path):
    out = _out_dir(tmp_path)
    f = out / "scores" / "test.parquet"
    pd.read_parquet(f).assign(tokens=20_000).to_parquet(f)  # nano-jev read 20k tokens: 1,336 GFLOPs
    df, ok, cost = ev.load_tables(out, "test", ["cheap", "strong"], 2, prices=[3, 115])
    ranking = 3 * 1336 / 1152 - 2  # budget(): 1.48
    assert df["shared_cost"].iat[0] == pytest.approx(ranking)
    assert cost[0].tolist() == pytest.approx([3 + ranking, 115 + ranking])
    escalate = np.ones(len(df), dtype=int)  # read the cheap answer, then ran the strong option
    assert ev.point(ok, cost, escalate, post=True, shared=df["shared_cost"].to_numpy())["cost"] == \
        pytest.approx(3 + 115 + ranking)
    measured, _, _ = ev.load_tables(out, "test", ["cheap", "strong"], 2)
    assert measured["shared_cost"].iat[0] == pytest.approx(1336 / 1152)  # inside each answer's measured units


def test_rescore_helpers():
    from stair.rescore import committed_text, grounded

    assert committed_text(' **"Forbidden Quest"**.\nBecause ...') == "Forbidden Quest"
    assert committed_text("\n\n  Kalo.") == "Kalo" and committed_text("") == ""
    assert grounded("Forbidden Quest", 'His debut was "Forbidden Quest" (2006).')
    assert grounded("the Forbidden Quest", "forbidden quest")  # normalised: articles and case
    assert grounded("Forbidden Quest film", "forbidden quest")  # 2 of 3 tokens >= 0.6
    assert grounded("Yes", "Both films were released in 2009.")  # yes/no exempt
    assert not grounded("Obsessed", "His debut was Forbidden Quest.") and not grounded("", "anything")


def test_load_tables_adds_commit_cost(tmp_path):
    out = _out_dir(tmp_path)
    for cell in ("cheap", "strong"):
        f = out / "rescore" / "test" / f"{cell}.parquet"
        pd.read_parquet(f).assign(commit_units=0.5).to_parquet(f)
    _, _, cost = ev.load_tables(out, "test", ["cheap", "strong"], 2, metric="correct", rescore_dir=out / "rescore")
    assert cost[0].tolist() == [1.5, 9.5]
    _, _, lenient_cost = ev.load_tables(out, "test", ["cheap", "strong"], 2)
    assert lenient_cost[0].tolist() == [1.0, 9.0]  # the lenient metric has no commit step


def test_golden_set_cost_prices_options_and_nanojev():
    from stair_common.cost import UNIT_GFLOPS

    cost = ev.golden_set_cost([3, 15, 115], np.array([0.0, 2 * UNIT_GFLOPS]))  # nano-jev: free, then 3 x 2 - 2 = 4
    assert cost.tolist() == [[3.0, 15.0, 115.0], [7.0, 19.0, 119.0]]


def test_slr_splits_cap_with_preferred_ids_first():
    from stair_common.splits import load_split

    from stair import make_splits

    train, val = load_split("hotpotqa", "slr-train"), load_split("hotpotqa", "slr-val")
    prefer = set(train[:7]) | set(val[-3:])
    parts = make_splits.slr_splits({"router_train": 10, "calib": 5}, prefer)
    assert len(parts["router_train"]) == 10 and set(train[:7]) <= set(parts["router_train"])
    assert len(parts["calib"]) == 5 and set(val[-3:]) <= set(parts["calib"])
    assert parts["router_train"] == [i for i in train if i in set(parts["router_train"])]  # the part's order
    assert len(parts["test"]) == 1481
    assert make_splits.slr_splits({"router_train": 10, "calib": 5}, prefer) == parts  # deterministic
