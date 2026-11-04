import json

import numpy as np
import pandas as pd
import pytest

from stair import exp11_mab as e
from stair import exp8_eval as ev


def _table(tmp_path, swap=None):
    """A synthetic replay table in Jonathan's layout and its task file. Patient k has task k of every category; patients
    1-9 are train, 10-15 val, 16-30 test (9 / 6 / 15 a category, patient-disjoint). task1_15 and task1_30 name no
    patient. The weak model solves task1 only; the strong one solves every other task except task10."""
    rows, tasks = [], []
    for c in range(1, 11):
        for k in range(1, 31):
            tid = f"task{c}_{k}"
            split = "train" if k <= 9 else "val" if k <= 15 else "test"
            finished = c in (1, 2)
            attempt = ("The weak model finished after 2 round(s). Its final answer: FINISH([1])" if finished
                       else "The weak model did NOT finish: it stopped after 8 round(s) without calling FINISH. "
                            "Its last action: GET x")
            rows.append({"tid": tid, "idx": (c - 1) * 30 + k - 1, "split": split,
                         "view": f"Instruction of category {c} for patient {k}\n\n{attempt}",
                         "qw": int(c == 1), "qs": int(c not in (1, 10)), "cw": 100.0, "cs": 1000.0, "weak_rounds": 2})
            tasks.append({"id": tid} | ({} if (c, k) in ((1, 15), (1, 30)) else {"eval_MRN": f"p{k}"}))
    if swap:  # exchange two tasks' sides: the sizes hold, but both patients are now on both sides
        a, b = (next(r for r in rows if r["tid"] == t) for t in swap)
        a["split"], b["split"] = b["split"], a["split"]
    (tmp_path / "table.json").write_text(json.dumps(rows))
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    return {"table": str(tmp_path / "table.json"), "tasks": str(tmp_path / "tasks.json"), "out": str(tmp_path / "out"),
            "bootstrap": 50}


def test_a_view_splits_into_the_task_and_the_weak_attempt():
    task, attempt = e.split_view("What's the MRN?\nContext: now\n\nThe weak model finished after 1 round(s). Its final "
                                 "answer: FINISH([\"S1\"])")
    assert task == "What's the MRN?\nContext: now" and attempt.startswith("The weak model finished")
    with pytest.raises(ValueError, match="weak attempt"):
        e.split_view("no summary here")


def test_load_labels_escalation_and_keeps_patients_on_one_side(tmp_path):
    df = e.load(_table(tmp_path))
    assert len(df) == 300 and df["y"].sum() == 240  # categories 2-9: only the strong model solves them
    assert df.loc[df["tid"] == "task1_30", "patient"].item() == "none:task1_30"
    assert df["finished"].sum() == 60
    with pytest.raises(ValueError, match="2 patients have tasks in both"):
        e.load(_table(tmp_path, swap=("task3_1", "task3_16")))
    plan = _table(tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        e.load(plan | {"table_sha256": "0" * 64})


def test_the_items_ask_the_typed_questions_on_the_attempt(tmp_path):
    df = e.load(_table(tmp_path)).head(1)
    zs, rt = e.correct_items(df)[0], e.route_items(df)[0]
    assert zs["question"].startswith("Is the proposed answer correct? Task: Instruction of category 1")
    assert zs["options"] == ("yes", "no") and rt["options"] == ("small model", "large model")
    assert zs["state"] == rt["state"] and zs["state"].startswith("The weak model finished")


def test_baselines_are_the_category_rate_on_train(tmp_path):
    df = e.load(_table(tmp_path))
    fin, cat, cell = e.baselines(df)
    assert fin.scores["task1_20"] == 0 and fin.scores["task3_20"] == 1  # task1 and task2 finish
    assert cat.scores["task1_20"] == 0 and cat.scores["task5_20"] == 1 and cat.scores["task10_20"] == 0
    assert cell.scores.equals(cat.scores)  # finishing is constant within each category here


def test_operating_points_pick_the_most_accurate_then_the_cheapest(tmp_path):
    df = e.load(_table(tmp_path))
    val = df[df["split"] == "val"].reset_index(drop=True)
    pts = e.operating_points(val, val["y"].to_numpy().astype(float))  # a perfect score: escalate exactly the y = 1
    assert pts["A"]["t"] == 1.0 and pts["A"]["val_accuracy"] == pytest.approx(0.9)  # task10 is hopeless
    assert pts["B"]["t"] == 1.0  # the oracle is also the cheapest point at always-escalate's accuracy (0.8)
    always = e.operating_points(val, np.zeros(len(val)))  # one score: escalate all (t 0) or none (inf)
    assert always["A"]["t"] == 0.0 and always["B"]["t"] == 0.0


def test_his_reward():
    q, cs = np.array([1, 0]), np.array([1000.0, 1000.0])
    r = e.reward(q, np.array([100.0, 1100.0]), cs, 0.10)
    assert r.tolist() == pytest.approx([1 - 0.01, 0 - 0.11 - 0.1])


def test_bootstrap_resamples_whole_patients():
    rng, groups = np.random.default_rng(0), np.array(["a", "a", "b", "c", "c", "c"])
    idx = ev.resample(6, rng, groups)
    drawn = pd.Series(groups[idx]).value_counts()
    sizes = {"a": 2, "b": 1, "c": 3}
    assert all(n % sizes[g] == 0 for g, n in drawn.items())  # a group comes with all its rows
    assert sum(n // sizes[g] for g, n in drawn.items()) == 3  # as many groups as there are
    assert len(ev.resample(6, rng)) == 6


def test_select_then_one_test_read(tmp_path, monkeypatch):
    class FakeJudge:
        device = "cpu"

        def probabilities(self, its):
            p = np.array([0.9 if "did NOT finish" in it["state"] else 0.1 for it in its])
            return np.stack([1 - p, p], axis=1), np.full(len(its), 100)

    monkeypatch.setattr(e, "make_judge", lambda *a, **k: FakeJudge())
    monkeypatch.setattr(e, "fine_tune", lambda *a, **k: [{"epoch": 1}])
    monkeypatch.setattr(e.provenance, "hub_revision", lambda init: "rev")
    plan = _table(tmp_path) | {
        "zero_shot": ["sdmlai/nano-jev@v1.0"],
        "judges": [{"name": "nano-jev", "judge_dir": "judge-nanojev", "init": "sdmlai/nano-jev@v1.0", "seeds": [0, 1],
                    "train": {"epochs": 1, "lr": 1e-5, "weight_decay": 0, "warmup": 0, "batch_size": 4}}],
        "contrasts": [["nano-jev", "category x finished"]]}
    df = e.load(plan)
    e.run_zeroshot(df, plan, "cpu")
    for seed in (0, 1):
        e.run_train(df, plan, plan["judges"][0], seed, "cpu")
    assert len(pd.read_parquet(tmp_path / "out" / "judge-nanojev" / "seed0" / "scores.parquet")) == 210  # val + test
    his = pd.DataFrame({"tid": df["tid"], "always": 1})
    his.to_csv(tmp_path / "his.csv", index=False)
    plan["his_decisions"] = str(tmp_path / "his.csv")

    chosen = e.run_select(df, plan)
    assert set(chosen["routers"]) == {"finished", "category", "category x finished", "zero-shot nano-jev",
                                      "nano-jev seed0", "nano-jev seed1", "nano-jev"}
    report = e.run_test(df, plan)
    nj = report["routers"]["nano-jev"]
    assert nj["A"]["accuracy"] == pytest.approx(0.8)  # escalates the unfinished, categories 3-10; task2 is lost
    assert nj["A"]["units_with_judge"] > nj["A"]["units"]  # two seeds' calls
    assert report["references"]["oracle"]["accuracy"] == pytest.approx(0.9)
    assert "nano-jev - category x finished" in report["aiq_diff_contrasts"]
    assert report["gap_closed"]["zero-shot nano-jev / nano-jev"] == pytest.approx(1.0)  # same scores
    assert report["ours_A_minus_his"]["nano-jev A - his always"]["units"] < 0  # cheaper than always escalating
    with pytest.raises(SystemExit, match="read once"):
        e.run_test(df, plan)
    with pytest.raises(SystemExit, match="can no longer change"):
        e.run_select(df, plan)
