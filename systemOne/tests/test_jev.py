"""The Jev novelty programme (docs/experiments/2026-10-24-jev-preregistration.md): label budgets, judge grids and the
nano-jev pretraining arms."""

import json
from pathlib import Path

import numpy as np
import pytest

from stair import judge, nanojev_pretrain

CONFIGS = Path(__file__).parent.parent / "configs"
FAKE_EXAMPLE = '\ndef _example(decision, label, state, source, **fields):\n    return {"decision": decision, "question": str(fields), "options": ["a", "b", "c"], "state": state,\n            "label": label, "source": source}\n'


def test_budgets_nest_and_keep_the_label_rate():
    y = (np.random.default_rng(0).random(1000) < 0.2).astype(int)
    order = judge.budget_order(y, seed=3)
    assert sorted(order) == list(range(1000))  # a permutation
    for b in (50, 100, 500):
        assert abs(y[order[:b]].mean() - y.mean()) <= 1.5 / b + 0.01  # every prefix is stratified
    assert set(order[:50]) <= set(order[:100])  # nested by construction
    assert not np.array_equal(order, judge.budget_order(y, seed=4))  # another seed, another draw


def test_a_budget_counts_its_dev_hold_out():
    y = (np.random.default_rng(1).random(400) < 0.3).astype(int)
    tr, dev = judge.budget_split(y, 50, 0.1, seed=0)
    assert len(tr) + len(dev) == 50 and len(dev) == 10  # at least 10 to pick the epoch
    assert set(tr) | set(dev) == set(judge.budget_order(y, 0)[:50])
    tr, dev = judge.budget_split(y, 400, 0.1, seed=0)
    assert len(dev) == 40
    with pytest.raises(ValueError, match="budget"):
        judge.budget_split(y, 401, 0.1, seed=0)


def test_a_judge_entry_expands_to_its_grid_and_names_stay_apart(tmp_path):
    spec = {"inputs": ["qp", "qpa"], "label": "large-helps", "init": ["sdmlai/nano-jev@v1.0", "runs/x/model"],
            "head": ["typed", "plain"], "budget": [50, 4814], "train_pair": "A", "seeds": [0, 1]}
    specs = judge.expand(spec)
    assert len(specs) == 16 and all(s["seeds"] == [0, 1] for s in specs)
    assert judge.judge_name({**specs[0], "head": "typed", "budget": None}) == "qp-large-helps-nanojev-A"
    named = {**specs[-1], "init_name": "nanojev-repro"}
    assert judge.judge_name(named) == f"{named['inputs']}-large-helps-nanojev-repro-A-plain-b4814"

    def plan(names: str) -> str:
        return f"""
tau = 0.5
seeds = [0]
[pairs]
A = "runs/a"
{names}
[[judges]]
inputs = "qp"
label = "large-helps"
init = ["runs/one/model", "runs/two/model"]
train_pair = "A"
"""

    path = tmp_path / "plan.toml"
    path.write_text(plan(""))
    with pytest.raises(ValueError, match="share a name"):  # both local inits would be called "model"
        judge.load_plan(path)
    path.write_text(plan('[init_names]\n"runs/one/model" = "one"\n"runs/two/model" = "two"'))
    assert {judge.judge_name(s) for s in judge.load_plan(path)["judges"]} == {"qp-large-helps-one-A", "qp-large-helps-two-A"}


def test_the_e1_plan_is_the_preregistered_grid():
    plan = judge.load_plan(CONFIGS / "e1-label-curve.toml")
    specs = plan["judges"]
    assert len(specs) == 2 * 5 * 2 * 7  # inputs x starting points x heads x budgets
    runs = sum(len(s["seeds"]) for s in specs)
    assert runs == 2 * 5 * 2 * (4 * 10 + 3 * 5)
    assert {s["budget"] for s in specs} == {50, 100, 250, 500, 1000, 2000, 4814}


def _fake_recipe(root):
    """A stand-in for nano-jev's `nanojev.data`: its two non-HotpotQA builders, tagged by source."""
    pkg = root / "nanojev"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "data.py").write_text('''
def _ex(source, split, i):
    return {"decision": "sufficient", "question": f"{source} {split} {i}", "options": ["yes", "no"],
            "state": "s", "label": i % 2, "source": source}

def squad2_examples(split, n, rng):
    return [_ex("squad2", split, rng.randint(0, 10**6)) for _ in range(n)]

def mnli_examples(split, n, rng):
    return [_ex("mnli", split, rng.randint(0, 10**6)) for _ in range(n)]
''')


class _Rows:
    """A stand-in for a SQuAD 2.0 split: column access by name, row access by index."""

    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, key):
        return [r[key] for r in self.rows] if isinstance(key, str) else self.rows[key]


def _squad(n_titles=4, per_title=3):
    rows = []
    for t in range(n_titles):
        for p in range(per_title):
            context = f"paragraph {t}-{p} mentions answer{t}{p}"
            rows.append({"title": f"T_{t}", "context": context, "question": f"q{t}{p}?",
                         "answers": {"text": [f"answer{t}{p}"]}})
            rows.append({"title": f"T_{t}", "context": context, "question": f"u{t}{p}?", "answers": {"text": []}})
    return _Rows(rows)


def _example(decision, label, state, source, **fields):
    return {"decision": decision, "question": str(fields), "options": ["a", "b", "c"], "state": state,
            "label": label, "source": source}


def test_squad_relevance_mirrors_hotpotqa_four_examples():
    import random

    out = nanojev_pretrain.squad_relevance_examples(_squad(), 3, random.Random(0), _example)
    assert len(out) == 12 and [e["label"] for e in out[:4]] == [2, 1, 0, 0]
    gold, partial, other = out[0]["state"], out[1]["state"], out[2]["state"]
    assert gold.split(":")[0] == partial.split(":")[0] != other.split(":")[0]  # same article, then another one
    answer = gold.split("mentions ")[1]
    assert answer not in partial  # "partially relevant": same article, without the answer string


def test_the_nohotpot_arm_has_no_hotpotqa_anywhere(tmp_path, monkeypatch):
    import sys

    _fake_recipe(tmp_path / "src")
    data_py = tmp_path / "src" / "nanojev" / "data.py"
    data_py.write_text(data_py.read_text() + FAKE_EXAMPLE)
    monkeypatch.setattr(nanojev_pretrain, "NOHOTPOT", {"relevance_train": 2, "squad_train": 6, "mnli_train": 4,
                                                       "relevance_eval": 1, "squad_eval": 2, "mnli_eval": 3})
    monkeypatch.setattr(nanojev_pretrain, "load_squad", lambda split: _squad())
    monkeypatch.setattr("sys.path", list(sys.path))  # build_nohotpot prepends the recipe folder
    for mod in [m for m in sys.modules if m == "nanojev" or m.startswith("nanojev.")]:
        monkeypatch.delitem(sys.modules, mod)
    data = nanojev_pretrain.build_nohotpot(tmp_path / "src", seed=0)
    assert len(data["train"]) == 8 + 10 and len(data["calib"]) == 4 + 5 and len(data["test"]) == 4 + 5
    assert {ex["source"] for split in data.values() for ex in split} == {"squad2", "mnli"}
    for split, examples in data.items():
        nanojev_pretrain.write_jsonl(examples, tmp_path / "data" / f"{split}.jsonl")
    counts = nanojev_pretrain.source_counts(tmp_path / "data")
    assert counts["train"] == {"squad2/relevance": 8, "squad2/sufficient": 6, "mnli/sufficient": 4}
    again = nanojev_pretrain.build_nohotpot(tmp_path / "src", seed=0)
    assert json.dumps(again) == json.dumps(data)  # deterministic for a seed


def _musique_example(i: int) -> dict:
    paragraphs = [{"idx": k, "title": f"T{k}", "paragraph_text": f" text {i}-{k} ", "is_supporting": k in (3, 7)}
                  for k in range(20)]
    return {"id": f"2hop__{i}_{i + 1}", "paragraphs": paragraphs, "question": f"q{i}?", "answer": "A",
            "answer_aliases": ["A", "Alpha"], "answerable": True}


def test_musique_becomes_the_hotpotqa_schema():
    from stair.prepare_hotpotqa import flatten_example
    from stair.prepare_multihop import convert

    df = convert("musique", [_musique_example(1), _musique_example(2)])
    row = df.iloc[0]
    assert row["type"] == "2hop" and row["num_context_paragraphs"] == 20 and list(row["answers"]) == ["A", "Alpha"]
    assert list(row["context_sentences"][3]) == ["text 1-3"] and list(row["gold_titles"]) == ["T3", "T7"]
    hotpot_like = {"id": "x", "question": "q", "answer": "a", "type": "bridge", "level": "hard",
                   "context": {"title": ["A"], "sentences": [["s"]]}, "supporting_facts": {"title": ["A"], "sent_id": [0]}}
    assert set(flatten_example(hotpot_like)) | {"answers"} == set(df.columns)  # the same columns run-experts reads


def test_multihop_splits_follow_the_team_rule(tmp_path):
    import pandas as pd
    from stair_common.splits import shuffled

    from stair.make_splits import main as make_splits_main

    train = [f"t{i:05d}" for i in range(6000)]
    pd.DataFrame({"id": train}).to_parquet(tmp_path / "train.parquet")
    pd.DataFrame({"id": [f"d{i}" for i in range(300)]}).to_parquet(tmp_path / "dev.parquet")
    make_splits_main(["--dataset", "musique", "--data-dir", str(tmp_path)])
    splits = json.loads((tmp_path / "splits.json").read_text())
    assert splits["router_train"] == sorted(shuffled(train)[:4814]) and len(splits["calib"]) == 500
    assert not set(splits["router_train"]) & set(splits["calib"]) and len(splits["test"]) == 300


def test_a_dataset_without_gold_answers_is_refused():
    import pandas as pd

    from stair.prepare_multihop import SIZES, check

    frames = {part: pd.DataFrame({"id": [f"{part}{i}" for i in range(n)], "answer": ["a"] * n})
              for part, n in SIZES["2wikimultihopqa"].items()}
    check("2wikimultihopqa", frames)
    frames["dev"].loc[:, "answer"] = ""  # the team's 2Wiki source: the official test set, answers hidden
    with pytest.raises(ValueError, match="without a gold answer"):
        check("2wikimultihopqa", frames)


def test_twowiki_splits_cut_the_answered_dev_set(tmp_path):
    import pandas as pd
    from stair_common.splits import shuffled

    from stair.make_splits import main as make_splits_main

    pd.DataFrame({"id": [f"t{i}" for i in range(5000)]}).to_parquet(tmp_path / "train.parquet")
    dev = [f"d{i}" for i in range(4000)]
    pd.DataFrame({"id": dev}).to_parquet(tmp_path / "dev.parquet")
    make_splits_main(["--dataset", "2wikimultihopqa", "--data-dir", str(tmp_path)])
    splits = json.loads((tmp_path / "splits.json").read_text())
    order = shuffled(dev)
    assert splits["calib"] == sorted(order[:500]) and splits["test"] == sorted(order[500:1000])
    assert splits["test_ext"] == sorted(order[1000:3000])


def test_holm_and_le_auc():
    from stair.label_curve import holm, le_auc

    assert holm({"a": 0.01, "b": 0.04, "c": 0.03}) == {"a": 0.03, "c": 0.06, "b": 0.06}
    assert le_auc({50: 0.6, 100: 0.6, 4814: 0.6}) == pytest.approx(0.6)  # flat curve: its value
    assert le_auc({64: 0.5, 128: 0.7}) == pytest.approx(0.6)  # the mean over log2(budget)


def test_label_curve_finds_a_pretraining_gain_that_fades_with_labels():
    from stair import label_curve as lc

    rng = np.random.default_rng(0)
    n = 600
    small = (rng.random(n) < 0.55).astype(float)
    large = np.maximum(small, (rng.random(n) < 0.35).astype(float))
    a = {"small_f1": small, "large_f1": large, "small_em": small, "large_em": large,
         "small_cost": np.full(n, 0.5), "large_cost": np.full(n, 10.0), "small_uncertainty": rng.random(n)}
    helps = large - small  # 1 where escalating fixes the answer

    def judge(quality: float, seed: int) -> np.ndarray:  # a score that sees `quality` of the truth
        r = np.random.default_rng(seed)
        return quality * helps + (1 - quality) * r.random(n)

    arms = {}
    for budget, gain in ((50, 0.3), (100, 0.25), (4814, 0.02)):
        for init, base in (("nanojev", 0.4 + gain), ("minilm-l12-h384-uncased", 0.4), ("nanojev-nohotpot", 0.4 + gain)):
            arms[(init, "typed", "qp", budget)] = {s: judge(base, 100 * budget + s) for s in range(3)}
        arms[("nanojev", "plain", "qp", budget)] = {s: judge(0.4, 7 + s) for s in range(3)}
    report = lc.analyse(a, arms, n_boot=30, seed=0)
    stats = report["stats"]
    assert stats["H1a/qp"]["estimate"] > 0 and stats["H1a/qp"]["p_le_0"] < 0.05
    assert stats["H1b/qp"]["estimate"] > 0  # the gain shrinks as labels grow
    assert stats["H1c/qp"]["estimate"] > 0  # noHotpot keeps the gain here
    assert stats["H1d/qp"]["estimate"] > 0 and "p_holm" in stats["H1d/qp"]
    assert {c["budget"] for c in report["cells"]} == {50, 100, 4814}
    assert all(0 < c["gap_closed"] < 1 for c in report["cells"])


def _routing_arrays(n=500, seed=0):
    rng = np.random.default_rng(seed)
    small = (rng.random(n) < 0.55).astype(float)
    large = np.maximum(small, (rng.random(n) < 0.35).astype(float))
    return {"small_f1": small, "large_f1": large, "small_em": small, "large_em": large,
            "small_cost": np.full(n, 0.5), "large_cost": np.full(n, 10.0), "small_uncertainty": rng.random(n)}


def test_confirmatory_tests_find_what_was_planted():
    from stair.confirm import Policy, Test, run_tests, stronger

    a = _routing_arrays()
    helps = a["large_f1"] - a["small_f1"]
    rng = np.random.default_rng(1)

    def scores(q, k=3):
        return [q * helps + (1 - q) * rng.random(len(helps)) for _ in range(k)]

    pols = {"good": Policy("good", scores(0.6), False), "same": Policy("same", scores(0.6), False),
            "noise": Policy("noise", scores(0.0), False), "weak": Policy("weak", scores(0.3), False)}
    tests = [Test("good > noise", "superiority", "good", "noise"),
             Test("good >= same - delta", "noninferiority", "good", "same"),
             Test("noise ~ random", "equivalence", "noise"),
             Test("weak keeps 75% of good's gain", "ratio", "weak", "good", 0.75),
             Test("good transfers better than weak", "transfer-gap", "good", "same", 0.0, "weak", "same")]
    r = run_tests(a, pols, tests, n_boot=60, seed=0, delta=0.02)["tests"]
    assert r["good > noise"]["supported"] and r["good >= same - delta"]["supported"]
    assert r["noise ~ random"]["supported"]  # a score with no signal is equivalent to random routing
    assert not r["weak keeps 75% of good's gain"]["supported"] and r["weak keeps 75% of good's gain"]["estimate"] < 0
    assert r["good transfers better than weak"]["estimate"] > 0
    assert all("p_holm" in t for t in r.values())
    for p in pols.values():
        p.calib = p.seeds
    assert stronger(a, [pols["weak"], pols["good"], pols["noise"]]).name == "good"  # chosen on calib


def test_the_free_comparison_drops_only_the_signals_price():
    from stair.confirm import Policy, h3_free_tests, policy_aiq

    a = _routing_arrays()
    helps = a["large_f1"] - a["small_f1"]
    sc = [0.5 * helps + 0.5 * np.random.default_rng(2).random(len(helps))]
    pols = {"signal:self-consistency": Policy("signal:self-consistency", sc, True, np.full(len(helps), 3.0), calib=sc),
            "signal:msp": Policy("signal:msp", sc, True, calib=sc)}
    judge = Policy("judge", sc, True)
    tests, pols = h3_free_tests(pols, judge)
    assert [t.b for t in tests] == ["signal:self-consistency (free)"]  # only the signals that pay for extra calls
    free = pols["signal:self-consistency (free)"]
    assert free.extra is None and free.seeds is sc and free.post
    assert policy_aiq(a, free, None) == policy_aiq(a, judge, None) > policy_aiq(a, pols["signal:self-consistency"], None)


def test_zero_label_compositions_and_combiners():
    import pandas as pd

    from stair import zero_label as zl

    rng = np.random.default_rng(0)
    n = 300
    y = (rng.random(n) < 0.3).astype(int)
    df = pd.DataFrame({"id": [f"q{i}" for i in range(n)], "ev_sufficiency_k2": 1 - 0.6 * y - 0.3 * rng.random(n),
                       "ev_top1": -y + rng.random(n), "ev_top2_sum": -y + 2 * rng.random(n)})
    zs = pd.DataFrame({"id": df["id"], "zs_grounded": 0.5 * y + 0.5 * rng.random(n), "zs_correct": rng.random(n)})
    s = zl.typed_scores(df, zs, "zs")
    comps = zl.compositions(s)
    assert {"insufficient", "not-grounded", "max(pre)", "product(pre)", "max(all)"} <= set(comps)
    assert comps["max(pre)"][1] is False and comps["not-grounded"][1] is True  # pre-answer vs post
    assert np.corrcoef(comps["insufficient"][0], y)[0, 1] > 0.5  # oriented: higher = escalate
    test_draws, cal_draws = zl.combiner_draws(s, y, s, 25, seed=0)
    assert len(test_draws) == zl.DRAWS and all(len(d) == n for d in test_draws + cal_draws)


def test_risk_control_keeps_its_guarantee():
    from stair import risk_control as rc

    rng = np.random.default_rng(0)
    n = 2000
    score = rng.random(n)
    wrong = (rng.random(n) < 0.05 + 0.6 * score).astype(int)  # higher score, more errors: a useful router
    small, large = np.full(n, 0.5), np.full(n, 10.0)
    assert rc.upper_bound(0, 100, 0.1) == pytest.approx(1 - 0.1 ** (1 / 100))  # Clopper-Pearson, no errors
    for alpha in (0.1, 0.2):
        r = rc.resplits(score, wrong, small, large, alpha, 0.1, n_splits=200, seed=0)
        assert r["mean_risk"] <= alpha and r["violation_rate"] <= 0.1 and r["h6a"]
        assert 0 < r["mean_kept"] < 1  # it does keep questions, not just escalate everything
    assert rc.ltt_threshold(score, np.ones(n, dtype=int), 0.1, 0.1) == -np.inf  # all wrong: keep nothing


def test_signals_finish_a_run_whose_splits_are_done(tmp_path):
    import pandas as pd

    from stair import signals

    run = tmp_path / "pairA"
    (run / "signals").mkdir(parents=True)
    (run / "config.toml").write_text('[[experts]]\nname = "small"\nkind = "hf"\nmodel = "tiny/weak"\n')
    pd.DataFrame({"id": ["q1"], "sig_msp": [0.5], "sig_nli_entail": [0.9]}).to_parquet(run / "signals" / "test.parquet")
    signals.main(["--run", str(run), "--data-dir", str(tmp_path), "--split", "test"])  # loads no model
    record = json.loads((run / "signals" / "provenance.jsonl").read_text().splitlines()[-1])
    assert record["pair_run"] == str(run) and record["splits"] == ["test"]
    assert json.loads((run / "signals" / "settings.json").read_text())["k"] == 10  # confirm.py prices the samples by it


def test_e7_reuses_experiment_8s_cells_unchanged():
    import tomllib

    e7 = tomllib.loads((CONFIGS / "e7-evidence.toml").read_text(encoding="utf-8"))
    exp8 = {c["name"]: c for c in tomllib.loads((CONFIGS / "exp8-slr.toml").read_text(encoding="utf-8"))["cells"]}
    reused = [c for c in e7["cells"] if c["name"] in exp8]
    assert {c["name"] for c in reused} == {"llama3b-top5", "llama3b-s20fuse-top5", "llama70b-slrprompt-72bsel"}
    assert all(c == exp8[c["name"]] for c in reused)  # reused by name: the settings must be the ones that ran
    for c in e7["cells"]:
        if "ranker" in c:  # every new cell is its nano-jev twin with only the ranker changed
            twin = exp8["llama3b-s20fuse-top5" if c.get("fuse") else "llama3b-top5"]
            assert {k: v for k, v in c.items() if k not in ("name", "ranker")} == {k: v for k, v in twin.items() if k != "name"}
    assert e7["seed"] == 0 and sorted(c["ranker"] for c in e7["cells"] if "ranker" in c) == sorted(["msmarco", "bm25", "gold"] * 2)
