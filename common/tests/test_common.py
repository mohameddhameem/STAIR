import hashlib
import json

import pandas as pd
import pytest

from stair_common import cost, schema, score, splits

EXPECTED = {
    "hotpotqa/val": 500, "hotpotqa/test": 500, "hotpotqa/train_extra": 6405, "hotpotqa/train": 12000,
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
    parts = [set(splits.load_split(dataset, n.split("/")[1])) for n in EXPECTED if n.startswith(dataset + "/")]
    assert sum(map(len, parts)) == len(set().union(*parts))


def test_rule_is_deterministic_and_order_free():
    ids = [f"q{i}" for i in range(100)]
    assert splits.shuffled(ids) == splits.shuffled(list(reversed(ids)))
    parts = splits.cut(ids, {"train": 65, "val": 15, "test": 20})
    assert [len(p) for p in parts.values()] == [65, 15, 20]


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


def _row(**kw):
    row = {"dataset": "hotpotqa", "split": "test", "id": splits.load_split("hotpotqa", "test")[0], "track": "t",
           "policy": "p", "menu": "A", "seed": 0, "answer": "x", "lenient": 1, "em": 1.0, "f1": 1.0, "units": 3.0,
           "units_retrieve": 2.0, "units_answer": 1.0, "units_fuse": 0.0, "units_router": 0.0, "ms_router": None,
           "models": ""}
    return {**row, **kw}


def test_schema_accepts_a_good_table_and_rejects_bad_ones():
    schema.validate(pd.DataFrame([_row()]))
    with pytest.raises(ValueError, match="sum"):
        schema.validate(pd.DataFrame([_row(units=5.0)]))
    with pytest.raises(ValueError, match="not in splits"):
        schema.validate(pd.DataFrame([_row(id="not-a-test-id")]))
