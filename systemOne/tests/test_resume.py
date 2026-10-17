import math
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from stair import experts, run_experts
from synthetic import make_split

SMOKE = Path(__file__).parent.parent / "configs" / "smoke.toml"


def run(tmp_path, *extra):
    run_experts.main(["--config", str(SMOKE), "--data", str(tmp_path / "q.parquet"), "--run", str(tmp_path / "run"),
                      "--name", "test", "--chunk-size", "10", *extra])


def test_stages_resume_and_match_single_pass(tmp_path, monkeypatch):
    make_split(35, seed=2).to_parquet(tmp_path / "q.parquet")
    run(tmp_path, "--stage", "evidence")
    parts = tmp_path / "run" / "test.parts"
    assert len(list(parts.glob("evidence-*.parquet"))) == 4

    # simulate a session dying after two chunks of the small expert
    calls = []
    real = run_experts.expert_rows
    def flaky(chunk, evidence, expert):
        if len(calls) == 2:
            raise KeyboardInterrupt
        calls.append(1)
        return real(chunk, evidence, expert)
    monkeypatch.setattr(run_experts, "expert_rows", flaky)
    with pytest.raises(KeyboardInterrupt):
        run(tmp_path, "--stage", "small")
    assert len(list(parts.glob("small-*.parquet"))) == 2
    assert not list(parts.glob("*.tmp"))

    monkeypatch.setattr(run_experts, "expert_rows", real)
    run(tmp_path)  # finishes small, runs large and merge
    staged = pd.read_parquet(tmp_path / "run" / "test.parquet")

    fresh = tmp_path / "fresh"
    fresh.mkdir()
    (fresh / "q.parquet").write_bytes((tmp_path / "q.parquet").read_bytes())
    run(fresh)
    single = pd.read_parquet(fresh / "run" / "test.parquet")
    cols = [c for c in single.columns if not c.endswith("_seconds")]
    pd.testing.assert_frame_equal(staged[cols], single[cols])
    assert len(staged) == 35 and "ranked" not in staged.columns


def test_merge_refuses_unfinished_stage(tmp_path):
    make_split(12, seed=3).to_parquet(tmp_path / "q.parquet")
    run(tmp_path, "--stage", "evidence")
    with pytest.raises(FileNotFoundError, match="small"):
        run(tmp_path, "--stage", "merge")


def test_changed_question_list_is_rejected(tmp_path):
    make_split(12, seed=3).to_parquet(tmp_path / "q.parquet")
    run(tmp_path, "--stage", "evidence")
    with pytest.raises(ValueError, match="different question list"):
        run(tmp_path, "--stage", "evidence", "--limit", "5")


def test_run_is_pinned_to_one_config(tmp_path):
    make_split(12, seed=3).to_parquet(tmp_path / "q.parquet")
    run(tmp_path, "--stage", "evidence")
    other = tmp_path / "other.toml"
    other.write_text(SMOKE.read_text().replace("top_k = 4", "top_k = 3"))
    with pytest.raises(ValueError, match="new --run"):
        run_experts.main(["--config", str(other), "--data", str(tmp_path / "q.parquet"), "--run", str(tmp_path / "run"),
                          "--name", "test", "--chunk-size", "10"])
    runtime_only = tmp_path / "runtime.toml"
    runtime_only.write_text(SMOKE.read_text().replace("top_k = 4", "top_k = 4\nbatch_size = 8"))
    run_experts.main(["--config", str(runtime_only), "--data", str(tmp_path / "q.parquet"), "--run",
                      str(tmp_path / "run"), "--name", "test", "--chunk-size", "10"])


def test_limit_is_a_seeded_random_sample(tmp_path):
    make_split(100, seed=4).to_parquet(tmp_path / "q.parquet")
    a = run_experts.load_questions(tmp_path / "q.parquet", None, None, 10, seed=0)
    b = run_experts.load_questions(tmp_path / "q.parquet", None, None, 10, seed=0)
    c = run_experts.load_questions(tmp_path / "q.parquet", None, None, 10, seed=1)
    assert a["id"].tolist() == b["id"].tolist() != c["id"].tolist()
    assert a["id"].tolist() != [f"syn{i:05d}" for i in range(10)]  # not the first rows
    assert len(run_experts.load_questions(tmp_path / "q.parquet", None, None, 500)) == 100


class FakeLogprob:
    def __init__(self, logprob):
        self.logprob = logprob


class FakeLLM:
    """Mimics vllm.LLM: generate() returns RequestOutput-like objects."""

    prompts: list[str] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def get_tokenizer(self):
        tok = types.SimpleNamespace(chat_template="x")
        tok.apply_chat_template = lambda msgs, tokenize, add_generation_prompt: f"<user>{msgs[0]['content']}<assistant>"
        tok.encode = lambda text: text.split()
        return tok

    def generate(self, prompts, params, use_tqdm):
        FakeLLM.prompts = prompts
        step = {1: FakeLogprob(math.log(0.5)), 2: FakeLogprob(math.log(0.5))}
        return [types.SimpleNamespace(
            prompt_token_ids=list(range(100)),
            outputs=[types.SimpleNamespace(text=" Dublin.\nExtra", token_ids=[1, 2], logprobs=[step, step])],
        ) for _ in prompts]


@pytest.fixture
def fake_vllm(monkeypatch):
    module = types.SimpleNamespace(LLM=FakeLLM, SamplingParams=lambda **kw: kw)
    monkeypatch.setitem(sys.modules, "vllm", module)


def test_vllm_expert_parses_outputs(fake_vllm):
    e = experts.make_expert({"name": "large", "kind": "vllm", "model": "m", "dtype": "float16",
                             "quantization": "awq", "max_model_len": 4096})
    assert e.llm.kwargs == {"model": "m", "dtype": "float16", "quantization": "awq", "max_model_len": 4096,
                            "gpu_memory_utilization": 0.85}
    outs = e.answer_batch(["Q1?", "Q2?"], [["P a"], ["P b", "P c"]])
    assert [o.answer for o in outs] == ["Dublin", "Dublin"]
    assert outs[0].cost == 102
    assert outs[0].uncertainty == pytest.approx(math.log(2))
    assert FakeLLM.prompts[1].startswith("<user>") and "P b\n\nP c" in FakeLLM.prompts[1]


def test_vllm_expert_rejects_overlong_prompts(fake_vllm):
    e = experts.make_expert({"name": "large", "kind": "vllm", "model": "m", "max_model_len": 40})
    with pytest.raises(ValueError, match="exceed"):
        e.answer_batch(["Q?"], [["word " * 50]])
