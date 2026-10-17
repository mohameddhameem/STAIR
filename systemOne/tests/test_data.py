import json

import pandas as pd
import pytest

from stair import make_splits
from stair.prepare_hotpotqa import count_invalid_supporting_facts, flatten_example

EXAMPLE = {
    "id": "q1",
    "question": "Which band did the singer of X join?",
    "answer": "Y",
    "type": "bridge",
    "level": "hard",
    "context": {
        "title": ["A", "B", "C"],
        "sentences": [["a0 ", "a1"], ["b0"], ["c0"]],
    },
    "supporting_facts": {"title": ["B", "A", "B"], "sent_id": [0, 1, 5]},
}


def test_flatten_keeps_raw_structure():
    row = flatten_example(EXAMPLE)
    assert row["context_titles"] == ["A", "B", "C"]
    assert row["context_sentences"] == [["a0 ", "a1"], ["b0"], ["c0"]]
    assert row["sp_titles"] == ["B", "A", "B"]
    assert row["sp_sent_ids"] == [0, 1, 5]
    assert row["gold_titles"] == ["B", "A"]


def test_flatten_text_fields():
    row = flatten_example(EXAMPLE)
    assert row["supporting_context"] == "b0 a1"
    assert row["full_context"] == "[A] a0 a1\n\n[B] b0\n\n[C] c0"
    assert row["num_supporting_facts"] == 3


def test_flat_rows_round_trip_through_parquet(tmp_path):
    path = tmp_path / "x.parquet"
    pd.DataFrame([flatten_example(EXAMPLE)]).to_parquet(path, index=False)
    row = pd.read_parquet(path).iloc[0]
    assert [list(s) for s in row["context_sentences"]] == [["a0 ", "a1"], ["b0"], ["c0"]]


def test_count_invalid_supporting_facts():
    assert count_invalid_supporting_facts(EXAMPLE) == 1


def test_splits_are_disjoint_deterministic_and_respect_exclusions():
    ids = [f"q{i}" for i in range(100)]
    exclude = {f"q{i}" for i in range(10)}
    a = make_splits.make_splits(ids, exclude, n_router=60, n_calib=20, seed=1)
    b = make_splits.make_splits(list(reversed(ids)), exclude, n_router=60, n_calib=20, seed=1)
    assert a == b
    assert not set(a["router_train"]) & set(a["calib"])
    assert not (set(a["router_train"]) | set(a["calib"])) & exclude
    assert len(a["router_train"]) == 60 and len(a["calib"]) == 20


def test_splits_reject_oversized_request():
    with pytest.raises(ValueError):
        make_splits.make_splits(["a", "b"], set(), n_router=2, n_calib=1, seed=0)


def test_make_splits_cli(tmp_path):
    train = tmp_path / "train.parquet"
    pd.DataFrame({"id": [f"q{i}" for i in range(60)], "type": ["bridge", "comparison"] * 30,
                  "level": ["hard"] * 50 + ["easy"] * 10}).to_parquet(train)
    exclude = tmp_path / "exclude.txt"
    exclude.write_text("q0\nq1\n\n")
    out = tmp_path / "splits.json"
    make_splits.main(["--train", str(train), "--exclude", str(exclude), "--router-size", "30", "--calib-size", "10", "--out", str(out)])
    manifest = json.loads(out.read_text())
    assert manifest["num_excluded"] == 2 and manifest["levels"] == ["hard"]
    assert len(manifest["router_train"]) == 30 and len(manifest["calib"]) == 10
    chosen = manifest["router_train"] + manifest["calib"]
    assert "q0" not in chosen
    assert not {f"q{i}" for i in range(50, 60)} & set(chosen)  # easy questions are left out by default
    with pytest.raises(ValueError, match="only 48"):
        make_splits.main(["--train", str(train), "--exclude", str(exclude), "--router-size", "45", "--calib-size", "5",
                          "--out", str(out)])
    make_splits.main(["--train", str(train), "--level", "any", "--router-size", "50", "--calib-size", "10", "--out", str(out)])
    assert len(json.loads(out.read_text())["router_train"]) == 50


def test_nanojev_ids_follow_its_shuffle_not_row_order():
    from datasets import Dataset

    from stair.nanojev_ids import shuffled_ids

    split = Dataset.from_dict({"id": [f"q{i}" for i in range(50)]})
    ids = shuffled_ids(split, seed=0, n=10)
    # nano-jev's calib and test ranges: shuffle(seed).select(range(start, start + n))
    calib = list(split.shuffle(seed=0).select(range(0, 5))["id"])
    test = list(split.shuffle(seed=0).select(range(5, 10))["id"])
    assert ids == calib + test
    assert ids != list(split["id"])[:10]
    assert shuffled_ids(split, seed=0, n=100) == list(split.shuffle(seed=0)["id"])
