"""The team's common setup in this repo: the lenient metric view, AgentTTS units, the standard splits, cache
subsets and the shared results layout."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from stair_common.schema import COLUMNS, validate
from stair_common.splits import load_split

from stair import export_common, make_splits, subset_cache
from stair.evaluate_routing import judge_cost_in, table5_price, with_cost_unit
from stair.metrics import metric_view
from stair.router import escalation_label


def _cache(ids, small, large):
    return pd.DataFrame({"id": ids, "answer": "Greenwich Village, New York City", "small_pred": small,
                         "large_pred": large, "small_f1": 0.3, "large_f1": 0.9, "small_em": 0.0, "large_em": 0.0,
                         "small_cost": 192.0, "large_cost": 192.0})


def test_metric_view_uses_the_standard_ruler():
    df = _cache(["a", "b"], ["New York City", "Boston"], ["more than 20", "Greenwich Village"])
    view = metric_view(df, "lenient")
    assert view["small_f1"].tolist() == [1.0, 0.0] and view["large_f1"].tolist() == [0.0, 1.0]
    assert df["small_f1"].tolist() == [0.3, 0.3]  # the original frame is untouched
    assert escalation_label(view, 0.5, "large-helps").tolist() == [0, 1]
    assert metric_view(df, "f1") is df
    with pytest.raises(ValueError):
        metric_view(df, "bleu")


def test_agenttts_unit_is_one_3b_pass():
    df = with_cost_unit(_cache(["a"], ["x"], ["y"]), "agenttts", {"small": 3.0, "large": 7.0})
    assert df["small_cost"].iat[0] == pytest.approx(1.0) and df["large_cost"].iat[0] == pytest.approx(7 / 3)


def test_table5_prices_each_answering_call():
    models = {"small": "Qwen/Qwen2.5-1.5B-Instruct", "large": "Qwen/Qwen2.5-7B-Instruct"}
    df = with_cost_unit(_cache(["a"], ["x"], ["y"]), "table5", {"small": 1.5, "large": 7.0}, models)
    assert df["small_cost"].iat[0] == pytest.approx(0.5)  # not in Table 5: budget(1.5, 256, 64)
    assert df["large_cost"].iat[0] == 10                  # Table 5's price
    assert table5_price("Qwen/Qwen2.5-72B-Instruct-AWQ", 72) == 118  # "Qwen2.5-7B" must not match the 72B
    assert table5_price("Qwen/Qwen2.5-1.5B-Instruct", 1.54) == pytest.approx(0.5)  # nominal size, as Table 5 counts
    with pytest.raises(ValueError, match="model name"):
        with_cost_unit(_cache(["a"], ["x"], ["y"]), "table5", {"small": 1.5, "large": 7.0})


def test_judge_calls_priced_with_budget_under_table5():
    assert judge_cost_in("table5", np.array([2 * 1152.0, 10.0])).tolist() == [4.0, 0.0]  # 3u - 2, floored at 0
    assert judge_cost_in("agenttts", 1152.0) == pytest.approx(1.0)


def test_judge_only_refuses_a_name_no_judge_has(tmp_path):
    from stair import judge

    plan = Path(__file__).parent.parent / "configs" / "system-one-slr.toml"
    with pytest.raises(SystemExit, match="ms-marco-minilm-l12-v2"):  # judge names are lower case
        judge.main(["train", "--plan", str(plan), "--out", str(tmp_path), "--only", "qp-large-helps-ms-marco-MiniLM-L12-v2-A"])


def test_slr_splits_are_the_golden_set():
    splits = make_splits.slr_splits()
    assert [len(splits[k]) for k in ("router_train", "calib", "test")] == [4814, 1110, 1481]
    assert splits["test"] == load_split("hotpotqa", "slr-test")


def test_standard_splits_follow_the_team_files():
    splits = make_splits.standard_splits({load_split("hotpotqa", "train")[0]})
    assert len(splits["router_train"]) == 11_999
    assert splits["calib"] == load_split("hotpotqa", "val") and splits["test"] == load_split("hotpotqa", "test")
    assert not (set(splits["router_train"]) & set(splits["test"]))


def test_standard_test_is_agenttts_slice():
    path = Path("data/hotpotqa/distractor_validation.parquet")
    if not path.exists():
        pytest.skip("HotpotQA dev parquet not prepared")
    dev = pd.read_parquet(path, columns=["id"])
    assert load_split("hotpotqa", "test") == dev["id"].iloc[500:1000].tolist()
    assert load_split("hotpotqa", "val") == dev["id"].iloc[:500].tolist()


def test_subset_cache_copies_rows_in_split_order(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    _cache(["a", "b", "c"], ["1", "2", "3"], ["4", "5", "6"]).to_parquet(source / "test.parquet")
    (source / "config.toml").write_text('[[experts]]\nname = "small"\nmodel = "m"\n')
    splits = tmp_path / "splits.json"
    splits.write_text(json.dumps({"calib": ["c", "a"]}))
    out = subset_cache.subset(source, splits, "calib", tmp_path / "run", "calib")
    assert out["id"].tolist() == ["c", "a"] and (tmp_path / "run" / "config.toml").exists()
    (tmp_path / "run" / "config.toml").write_text('[[experts]]\nname = "small"\nmodel = "other"\n')
    with pytest.raises(ValueError, match="differ"):
        subset_cache.subset(source, splits, "calib", tmp_path / "run", "calib")
    splits.write_text(json.dumps({"calib": ["zzz"]}))
    with pytest.raises(ValueError, match="not in"):
        subset_cache.subset(source, splits, "calib", tmp_path / "run2", "calib")


def test_exported_rows_pass_the_team_validator():
    ids = load_split("hotpotqa", "test")[:4]
    test = _cache(ids, ["New York City", "Boston", "x", "y"], ["Boston", "New York City", "x", "y"])
    test["small_cost"], test["large_cost"] = 1.0, 3.0
    esc = np.array([False, True, False, True])
    rows = export_common.rows_for(test, esc, True, "judge", 0, np.full(4, 0.5), np.full(4, 0.1), "m")
    df = pd.DataFrame(rows, columns=COLUMNS)
    validate(df)
    assert df["lenient"].tolist() == [1, 1, 0, 0]
    assert df["units_answer"].tolist() == [1.0, 4.0, 1.0, 4.0]  # post judge: an escalation pays both experts
    assert df["units"].tolist() == pytest.approx([1.6, 4.6, 1.6, 4.6])


def test_missing_answers_are_empty_not_none():
    ids = load_split("hotpotqa", "test")[:3]
    test = _cache(ids, ["New York City", None, np.nan], [None, "Boston", "x"])
    rows = export_common.rows_for(test, np.array([True, False, False]), False, "p", 0, np.zeros(3), np.zeros(3), "m")
    assert [r["answer"] for r in rows] == ["", "", ""]  # a missing large answer is not the string "None"
    assert metric_view(test, "lenient")["small_lenient"].tolist() == [1.0, 0.0, 0.0]


def test_export_histories(tmp_path):
    import json

    from stair.export_common import histories
    for seed in (0, 1):
        d = tmp_path / "judges" / "qpa-large-helps-nanojev-A" / f"seed{seed}"
        d.mkdir(parents=True)
        d.joinpath("done.json").write_text(json.dumps({"history": [
            {"epoch": 1, "train_loss": 0.7, "dev_nll": 0.6, "seconds": 1.0},
            {"epoch": 2, "train_loss": 0.5, "dev_nll": 0.65, "seconds": 1.0}]}))
    hist = histories(tmp_path)
    assert len(hist) == 4 and set(hist["seed"]) == {0, 1} and hist["val_loss"].tolist()[:2] == [0.6, 0.65]
    assert histories(tmp_path / "none").empty
