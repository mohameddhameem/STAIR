import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from stair import decider, exp8_judge, judge
from test_exp8 import _out_dir


def test_params_and_backend_from_the_name():
    assert decider.params_b("StrandsAgents/strands-decider-2B-hobson-v21") == 2.0
    assert decider.params_b("StrandsAgents/strands-decider-E4B-gemma4-v1-2610") == 4.0
    assert decider.params_b("StrandsAgents/strands-decider-26B-A4B-gemma4-v1-2610") == 4.0  # active parameters
    assert decider.params_b("strands-decider-12B-gemma4-v1-2610") == 12.0
    with pytest.raises(ValueError):
        decider.params_b("StrandsAgents/strands-decider-tiny")
    assert decider.is_decider("StrandsAgents/strands-decider-2B-hobson-v21") and not decider.is_decider("sdmlai/nano-jev@v1.0")
    assert judge.judge_params_b("sdmlai/nano-jev@v1.0") == judge.JUDGE_PARAMS_B
    assert judge.judge_params_b("StrandsAgents/strands-decider-E4B-gemma4-v1-2610") == 4.0


def test_make_judge_picks_the_backend(monkeypatch):
    made = []
    monkeypatch.setattr(judge, "DeciderJudge", lambda *a: made.append(("decider", a[0])) or "D")
    monkeypatch.setattr(judge, "Judge", lambda *a: made.append(("cross", a[0])) or "J")
    assert judge.make_judge("StrandsAgents/strands-decider-2B-hobson-v21") == "D"
    assert judge.make_judge("sdmlai/nano-jev@v1.0") == "J"
    assert [m[0] for m in made] == ["decider", "cross"]


def test_zero_shot_judges_keep_the_old_layout_for_one_judge():
    assert judge.zero_shot_judges({"zero_shot_judge": "sdmlai/nano-jev@v1.0"}) == [("sdmlai/nano-jev@v1.0", "zs", Path("zeroshot"))]
    assert judge.zero_shot_judges({}) == []
    many = judge.zero_shot_judges({"zero_shot_judge": ["sdmlai/nano-jev@v1.0", "StrandsAgents/strands-decider-2B-hobson-v21"]})
    assert [m[1] for m in many] == ["zs_nano-jev", "zs_strands-decider-2b-hobson-v21"]
    assert many[1][2] == Path("zeroshot") / "strands-decider-2b-hobson-v21"


class StubJudge:
    """Prefers the strong option when the question mentions it; a decider's interface, no model."""

    device = "cpu"

    def probabilities(self, items, temperature=1.0):
        p = np.array([[0.3, 0.7] if "strong" in it["state"] else [0.8, 0.2] for it in items])
        return p, np.full(len(items), 50)


def _plan(out: Path, **extra) -> dict:
    return {"out": str(out), "judge_dir": "judge-decider", "seeds": [0], "inputs": ["qp"],
            "init": "StrandsAgents/strands-decider-2B-hobson-v21", "zero_shot": True, "metric": "correct",
            "rescore_dir": str(out / "rescore"), "state_k": 2, "strong_units": 10, "budget_fraction": 0.5,
            "max_drop": 0.01, "bootstrap": 20, "agenttts_lenient": 0.0, "agenttts_units": 0.0,
            "options": [{"cell": "cheap", "label": "cheap"}, {"cell": "strong", "label": "strong"}],
            "train": {"dev_fraction": 0.5, "max_length": 512}, **extra}


def test_exp8_zero_shot_decider_and_reference_judge(tmp_path, monkeypatch):
    out = _out_dir(tmp_path)
    ids = pd.read_parquet(out / "pools" / "test.parquet")["id"].tolist()
    texts = ["Kalo did it, the strong one."] + [f"Other {i}." for i in range(9)]
    for split in ("router_train", "calib", "test"):  # the same toy questions, with 10 passages each, on every split
        pd.DataFrame({"id": ids, "question": "Who?", "answer": "Kalo", "titles": [[f"T{i}" for i in range(10)]] * len(ids),
                      "texts": [texts] * len(ids), "gold_idx": [[0]] * len(ids)}).to_parquet(out / "pools" / f"{split}.parquet")
        pd.DataFrame({"id": ids, "nanojev": [list(np.linspace(0.9, 0.0, 10))] * len(ids), "tokens": 100}).to_parquet(
            out / "scores" / f"{split}.parquet")
        if split == "test":
            continue
        for sub in ("answers", "rescore"):
            (out / sub / split).mkdir()
            for f in (out / sub / "test").glob("*.parquet"):
                (out / sub / split / f.name).write_bytes(f.read_bytes())
    monkeypatch.setattr(exp8_judge, "make_judge", lambda *a, **k: StubJudge())

    class Constant:  # the LR baselines' 5-fold CV needs more than 6 toy questions; not under test here
        def predict_proba(self, X):
            return np.full((len(X), 2), 0.5)

    monkeypatch.setattr(exp8_judge.ev, "fit_logistic", lambda X, y, seed: (Constant(), 1.0))
    plan = _plan(out)
    tables = exp8_judge.load_all(plan)
    exp8_judge.run_train(tables, plan, "qp", 0, out, "cpu")
    done = json.loads((out / "judge-decider" / "qp" / "seed0" / "done.json").read_text())
    assert done["zero_shot"] and done["history"] == []
    probs = pd.read_parquet(out / "judge-decider" / "qp" / "seed0" / "test.parquet")
    assert np.allclose(probs[["p0", "p1"]].sum(axis=1), 1)

    # a reference judge from another run (nano-jev), compared in the same report
    ref = out / "judge-nanojev" / "qp" / "seed0"
    ref.mkdir(parents=True)
    for split in ("calib", "test"):
        probs.assign(p0=0.5, p1=0.5).to_parquet(ref / f"{split}.parquet")
    plan = _plan(out, reference_judges=[{"name": "nano-jev qp", "judge_dir": "judge-nanojev", "inputs": "qp",
                                         "seeds": [0], "init": "sdmlai/nano-jev@v1.0"}])
    (out / "judge-decider" / "eval").mkdir()
    report = exp8_judge.run_eval(tables, plan, out)
    assert {"judge qp", "nano-jev qp"} <= set(report["judges"])
    assert "judge qp - nano-jev qp" in report["aiq_diff"]
    # the decider pays 2B parameters per token, nano-jev 0.0334B
    ratio = report["judges"]["judge qp"]["judge_units"] / report["judges"]["nano-jev qp"]["judge_units"]
    assert ratio == pytest.approx(2.0 / judge.JUDGE_PARAMS_B)
