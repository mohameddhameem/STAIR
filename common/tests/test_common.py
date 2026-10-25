import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import pytest

from stair_common import cost, schema, score, splits

EXPECTED = {
    "hotpotqa/val": 500, "hotpotqa/test": 500, "hotpotqa/train_extra": 6405, "hotpotqa/train": 12000,
    "hotpotqa/slr-train": 4814, "hotpotqa/slr-val": 1110, "hotpotqa/slr-test": 1481,
    "2wikimultihopqa/val": 500, "2wikimultihopqa/test": 500, "2wikimultihopqa/train_extra": 11576,
    "cwq/test": 500, "cwq/train": 350, "cwq/val": 150,
    "webqsp/test": 500, "webqsp/train": 350, "webqsp/val": 150,
    "medagentbench/calib": 150, "medagentbench/test": 150,
}


def test_committed_splits_match_manifest():
    manifest = json.loads((splits.SPLITS_DIR / "MANIFEST.json").read_text())
    assert manifest["seed"] == 20260729
    for name, n in EXPECTED.items():
        dataset, part = name.split("/")
        ids = splits.load_split(dataset, part)
        assert len(ids) == n == manifest["parts"][name]["n"]
        path = splits.SPLITS_DIR / dataset / f"{part}.txt"
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["parts"][name]["sha256"]


@pytest.mark.parametrize("dataset", ["hotpotqa", "2wikimultihopqa", "cwq", "webqsp", "medagentbench"])
def test_parts_are_disjoint(dataset):
    names = [n.split("/")[1] for n in EXPECTED if n.startswith(dataset + "/")]
    for scheme in {splits.scheme(n) for n in names}:  # the SLR split is a second scheme over HotpotQA's dev questions
        parts = [set(splits.load_split(dataset, n)) for n in names if splits.scheme(n) == scheme]
        assert sum(map(len, parts)) == len(set().union(*parts))


def test_slr_split_recuts_the_dev_questions():
    slr = set().union(*(splits.load_split("hotpotqa", f"slr-{p}") for p in ("train", "val", "test")))
    dev = set().union(*(splits.load_split("hotpotqa", p) for p in ("val", "test", "train_extra")))
    assert slr == dev and len(slr) == 7405
    assert not slr & set(splits.load_split("hotpotqa", "train"))  # all dev: none from the train file


def test_rule_is_deterministic_and_order_free():
    ids = [f"q{i}" for i in range(100)]
    assert splits.shuffled(ids) == splits.shuffled(list(reversed(ids)))
    parts = splits.cut(ids, {"train": 65, "val": 15, "test": 20})
    assert [len(p) for p in parts.values()] == [65, 15, 20]


def test_twowiki_needs_answered_questions():
    ids = [f"q{i:05d}" for i in range(12576)]
    parts = splits.twowiki(ids, ["an answer"] * len(ids))
    assert {k: len(v) for k, v in parts.items()} == {"val": 500, "test": 500, "train_extra": 11576}
    assert parts["val"] + parts["test"] + parts["train_extra"] == splits.shuffled(ids)  # the rule, then cut
    with pytest.raises(ValueError, match="no gold answer"):
        splits.twowiki(ids, ["an answer"] * (len(ids) - 1) + [" "])


def test_twowiki_source_is_the_answered_dev_set():
    # tts_train.json, the source until 0.4.0, is 2Wiki's official test set: every answer is empty (STAIR #9)
    source = json.loads((splits.SPLITS_DIR / "MANIFEST.json").read_text())["sources"]["2wikimultihopqa"]
    assert source == {"file": splits.TWOWIKI_DEV,
                      "sha256": "408e2dbb28edc6c8b9ca3ba0c94d4fc7bf17ffb923766593a3a7f546ab4cba59"}


@pytest.mark.skipif(not os.environ.get("STAIR_2WIKI_DEV"), reason="set STAIR_2WIKI_DEV to the pinned dev parquet")
def test_committed_twowiki_parts_rebuild_from_the_dev_file():
    ids, answers = splits._twowiki_dev(Path(os.environ["STAIR_2WIKI_DEV"]))
    for part, rebuilt in splits.twowiki(ids, answers).items():
        assert splits.load_split("2wikimultihopqa", part) == rebuilt


def test_medagentbench_is_stratified():
    test = splits.load_split("medagentbench", "test")
    tasks = pd.Series([i.rsplit("_", 1)[0] for i in test]).value_counts()
    assert len(tasks) == 10 and (tasks == 15).all()


def test_ruler_is_the_certified_file():
    assert hashlib.sha256(score.RULER.read_bytes()).hexdigest() == score.RULER_SHA256


def test_lenient_is_the_standard_ruler():
    # the examples on the team's ruler slide (13 Oct)
    assert score.lenient("New York City", ["Greenwich Village, New York City"]) == 1
    assert score.lenient("more than 20", ["20"]) == 0
    assert score.lenient("The Beatles!", ["the beatles"]) == 1


def test_em_f1_official():
    assert score.em("the Eiffel Tower", "Eiffel Tower") == 1.0
    assert score.f1("yes it is", "yes") == 0.0  # official rule: a yes/no gold gets no partial credit
    assert score.f1("New York City", "City of New York") == pytest.approx(6 / 7)  # P 3/3, R 3/4
    assert score.f1("no", "yes") == 0.0
    assert score.em("Paris", ["Lyon", "paris"]) == 1.0


def test_unit_is_one_3b_pass():
    assert cost.units(3.0, 128, 64) == pytest.approx(1.0)
    assert cost.units(72.0, 128, 64) == pytest.approx(24.0)


def test_samples_share_their_prompt():
    # AgentTTS App. A.2, Eq. 8: 40 samples of 64 tokens from one 128-token prompt pay the prompt once, not 40 times
    forty = cost.units(3.0, 128, 40 * 64)
    assert forty == pytest.approx((128 + 40 * 64) / 192)
    assert forty < 40 * cost.units(3.0, 128, 64)


def test_table5_is_agenttts_budget_not_units():
    # Table 5 counts the 3B samples, sharing one 128-token prompt, of equal FLOPs (Eq. 3-4): 3 x units - 2 for one call
    params = {"Qwen2.5-72B": 72, "Qwen2.5-32B": 32, "Qwen2.5-7B": 7, "Llama-3.1-70B": 70, "Llama-3.1-8B": 8,
              "Llama-3.2-3B": 3}
    lengths = {"retrieve": (2048, 128), "answer": (256, 64)}  # AgentTTS Table 4, HotpotQA
    for task, prices in cost.AGENTTTS_TABLE5_HOTPOTQA.items():
        for model, price in prices.items():
            assert round(3 * cost.units(params[model], *lengths[task]) - 2) == price
            assert round(cost.budget(params[model], *lengths[task])) == price
    assert cost.units(72.0, 2048, 128) == pytest.approx(272.0)


def test_budget_shares_the_prompt_and_is_floored():
    assert cost.budget(3.0, 256, 10 * 64) == pytest.approx(12.0)  # Table 7, 10 samples: the prompt counts once
    assert cost.budget(3.0, 256, 40 * 64) == pytest.approx(42.0)
    assert cost.budget(0.033, 200, 0) == 0.0                       # tiny calls would go negative
    assert cost.AGENTTTS_TABLE5_HOTPOTQA is cost.TABLE5_HOTPOTQA


def test_slr_path_prices_are_jonathans():
    assert cost.SLR_PATH_PRICES == {"WW": 80, "WS": 195, "SW": 894, "SS": 1009}  # configb_test_readout.py
    assert cost.STRONG_EVERYTHING == 929


def _row(**kw):
    row = {"dataset": "hotpotqa", "split": "test", "id": splits.load_split("hotpotqa", "test")[0], "track": "t",
           "policy": "p", "menu": "A", "seed": 0, "answer": "x", "lenient": 1, "em": 1.0, "f1": 1.0, "units": 3.0,
           "units_retrieve": 2.0, "units_answer": 1.0, "units_fuse": 0.0, "units_router": 0.0, "ms_router": None,
           "models": ""}
    return {**row, **kw}


def test_schema_accepts_a_good_table_and_rejects_bad_ones():
    schema.validate(pd.DataFrame([_row()]))
    schema.validate(pd.DataFrame([_row(split="slr-test", id=splits.load_split("hotpotqa", "slr-test")[0], price=80.0)]))
    with pytest.raises(ValueError, match="sum"):
        schema.validate(pd.DataFrame([_row(units=5.0)]))
    with pytest.raises(ValueError, match="not in splits"):
        schema.validate(pd.DataFrame([_row(id="not-a-test-id")]))


@pytest.mark.parametrize("bad, match", [
    ({"units": float("nan"), "units_retrieve": float("nan")}, "finite"),  # NaN also passes a sum check
    ({"units": -1.0, "units_retrieve": -2.0}, "finite"),                   # the stages still sum to units
    ({"f1": 1.5}, r"\[0, 1\]"),
    ({"em": float("nan")}, r"\[0, 1\]"),
    ({"seed": 0.5}, "integer"),
    ({"split": "tset"}, "unknown split"),
    ({"price": -1.0}, "price must be finite"),
    ({"price": float("nan")}, "price must be finite"),
])
def test_schema_rejects(bad, match):
    with pytest.raises(ValueError, match=match):
        schema.validate(pd.DataFrame([_row(**bad)]))


def test_quickstart_example_runs(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / "examples" / "quickstart.py"
    out = subprocess.run([sys.executable, str(example)], cwd=tmp_path, capture_output=True, text=True, check=True)
    assert "438 of 500" in out.stdout and (tmp_path / "results" / "demo-hotpotqa-test.csv").exists()


def test_package_data_ships_every_split_file():
    import fnmatch
    import tomllib
    from pathlib import Path

    from stair_common import load_models

    pkg = Path(splits.__file__).resolve().parent
    globs = tomllib.loads((pkg.parent / "pyproject.toml").read_text())["tool"]["setuptools"]["package-data"]["stair_common"]
    data = [p.relative_to(pkg).as_posix() for p in (pkg / "splits").rglob("*") if p.is_file()] + ["models.toml"]
    assert all(any(fnmatch.fnmatch(f, g) for g in globs) for f in data)
    assert load_models()["A"]["answerer"]["Llama-3.2-3B"]["params_b"] == 3.0
    training = load_models()["training"]["qlora"]
    assert {m["dtype"] for m in training.values()} == {"bf16"} and all(len(m["revision"]) == 40 for m in training.values())
