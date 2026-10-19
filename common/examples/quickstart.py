"""End-to-end walk-through of stair_common, runnable without any model: `python common/examples/quickstart.py`.

It fakes two policies' answers on the HotpotQA test split, then does what every track does for real:
score -> cost -> results rows -> validate -> save -> pool -> compare. Swap `fake_run` for your pipeline.
Needs Python >= 3.11 and the package installed: pip install -e common
"""

from pathlib import Path

import numpy as np
import pandas as pd

from stair_common import load_models
from stair_common.cost import units
from stair_common.schema import COLUMNS, validate
from stair_common.score import score
from stair_common.splits import load_split

# 1. Questions: the frozen split. Join the IDs to your copy of the data (see README "Get the questions").
test_ids = load_split("hotpotqa", "test")  # 500 IDs, AgentTTS's test questions, in file order
nanojev_seen = set(load_split("hotpotqa", "nanojev_v1_seen"))

# 2. Models: one pinned copy each; params_b feeds the cost.
models = load_models()
retriever, answerer = models["A"]["retriever"]["Qwen2.5-7B"], models["A"]["answerer"]["Llama-3.2-3B"]
print(f"menu A: {retriever['repo']}@{retriever['revision'][:7]} -> {answerer['repo']}@{answerer['revision'][:7]}")


def fake_run(qid: str, policy: str, rng) -> dict:
    """Stand-in for your pipeline: the answer, the gold answers and the measured tokens of every call."""
    gold = ["Greenwich Village, New York City"]
    answer = "New York City" if rng.random() < (0.75 if policy == "always-strong" else 0.70) else "Boston"
    calls = [("retrieve", retriever["params_b"], 3500, 20), ("answer", answerer["params_b"], 600, 16)]
    if policy == "router":
        calls.append(("router", 0.033, 512, 0))  # e.g. a nano-jev-sized judge reading 512 tokens
    return {"answer": answer, "gold": gold, "calls": calls, "ms_router": 2.1 if policy == "router" else None}


# 3. Score + cost each question, one row per (dataset, split, id, track, policy, menu, seed).
rows = []
for policy in ["always-strong", "router"]:
    for seed in [0, 1, 2]:
        rng = np.random.default_rng(seed)
        for qid in test_ids:
            out = fake_run(qid, policy, rng)
            stage = {"retrieve": 0.0, "answer": 0.0, "fuse": 0.0, "router": 0.0}
            for name, params_b, prompt_tokens, generated_tokens in out["calls"]:
                stage[name] += units(params_b, prompt_tokens, generated_tokens)
            rows.append({
                "dataset": "hotpotqa", "split": "test", "id": qid, "track": "demo", "policy": policy, "menu": "A",
                "seed": seed, "answer": out["answer"], **score(out["answer"], out["gold"]),
                "units": sum(stage.values()), **{f"units_{k}": v for k, v in stage.items()},
                "ms_router": out["ms_router"],
                "models": f"retriever={retriever['repo']}@{retriever['revision']};answerer={answerer['repo']}@{answerer['revision']}",
            })
results = pd.DataFrame(rows, columns=COLUMNS)

# 4. Validate before sharing: right columns, IDs inside the split, units = sum of stages, no duplicate rows.
validate(results)
out_dir = Path("results")
out_dir.mkdir(exist_ok=True)
results.to_csv(out_dir / "demo-hotpotqa-test.csv", index=False)

# 5. Pool everyone's files (here just ours) and make the headline table: ruler named, mean over seeds, range.
pooled = pd.concat([pd.read_csv(p) for p in sorted(out_dir.glob("*-test.csv"))])
per_seed = pooled.groupby(["track", "policy", "seed"]).agg(lenient=("lenient", "mean"), units=("units", "mean"))
table = per_seed.groupby(["track", "policy"]).agg(
    lenient_standard=("lenient", "mean"), lenient_min=("lenient", "min"), lenient_max=("lenient", "max"),
    units_per_question=("units", "mean"))
print(table.round(3).to_string())

# 6. Compare two policies paired on the same questions (seed 0): bootstrap interval of the accuracy difference.
a = results.query("policy == 'router' and seed == 0").set_index("id")["lenient"]
b = results.query("policy == 'always-strong' and seed == 0").set_index("id")["lenient"].reindex(a.index)
diff = (a - b).to_numpy()
boot = np.random.default_rng(0).choice(diff, size=(2000, len(diff))).mean(axis=1)
print(f"router - always-strong: {diff.mean():+.3f} lenient, 95% CI [{np.percentile(boot, 2.5):+.3f}, "
      f"{np.percentile(boot, 97.5):+.3f}] (paired, n={len(diff)})")

# 7. nano-jev saw some test questions in its own calibration: report its rows also on the clean subset.
clean = results[~results["id"].isin(nanojev_seen)]
print(f"clean subset without nano-jev's questions: {clean['id'].nunique()} of {results['id'].nunique()} test questions")
