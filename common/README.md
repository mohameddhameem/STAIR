# STAIR common setup: one split, one ruler, one cost, one results layout

This is the shared setup every track uses, so that results can be pooled. It settles the "Decisions needed" in
the team's proposal "STAIR shared run configuration" (17 Oct) and answers the questions asked in the Teams chat. It is a
proposal until the team agrees. Comment on the pull request.

## Quick start

```bash
git clone https://github.com/mohameddhameem/STAIR.git && cd STAIR
pip install -e "common[test]"          # numpy + pandas (+ pytest); Python >= 3.11
pytest common/tests                    # the splits, the ruler and the layout check out
python common/examples/quickstart.py   # the whole flow below on fake answers, no GPU needed
```

The five things a track needs:

| Import | What it does |
|---|---|
| `from stair_common.splits import load_split` | `load_split("hotpotqa", "test")` gives 500 question IDs. Parts are `train`, `train_extra`, `val`, `calib` and `test`. |
| `from stair_common.score import score` | `score(answer, golds)` gives `{"lenient": 0/1, "em": 0/1, "f1": 0–1}`. |
| `from stair_common.cost import units` | `units(params_b, prompt_tokens, generated_tokens)` gives AgentTTS units for one call. |
| `from stair_common import load_models` | The pinned copy of each model (`stair_common/models.toml`): `repo`, `revision`, `params_b`, `dtype`, for menus A, B and C. |
| `from stair_common.schema import validate` | `validate(df)` checks a results table before you share it. |

### 1. Get the questions of a split

The splits hold only IDs; join them to your copy of the data. For HotpotQA, the IDs are the dataset's `_id` / `id`.

```python
from datasets import load_dataset                     # pip install datasets
from stair_common.splits import load_split

dev = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation").to_pandas().set_index("id")
test = dev.loc[load_split("hotpotqa", "test")]         # 500 rows, AgentTTS's test questions, in their order
val = dev.loc[load_split("hotpotqa", "val")]           # tune thresholds / pick checkpoints here
print(test.iloc[0][["question", "answer"]])
```

```python
import json                                            # or AgentTTS's own files (same IDs, same order)
dev = {q["_id"]: q for q in json.load(open("hotpotqa-dev.json"))}
test = [dev[i] for i in load_split("hotpotqa", "test")]
```

ID fields per dataset:
- HotpotQA and 2WikiMultiHopQA: `_id`;
- CWQ: `ID`;
- WebQSP: `QuestionId`;
- MedAgentBench: `id`, as in `task3_17`.

### 2. Pick models from the menu

```python
from stair_common import load_models

models = load_models()
m = models["A"]["answerer"]["Llama-3.1-70B"]
print(m["repo"], m["revision"], m["params_b"], m["dtype"])   # hugging-quants/...AWQ-INT4 2123003... 70.0 awq

# vLLM: always load the pinned revision
from vllm import LLM
llm = LLM(model=m["repo"], revision=m["revision"], quantization="awq" if m["dtype"] == "awq" else None)
```

### 3. Score an answer

```python
from stair_common.score import score, lenient

score("New York City", ["Greenwich Village, New York City"])  # {'lenient': 1, 'em': 0.0, 'f1': 0.75}
lenient("more than 20", ["20"])                                # 0: the quantity guard
```

Pass every gold alias you have; each score takes the best match. Name the ruler next to every number: "lenient
(standard)", "EM", "F1".

### 4. Cost a question: sum every call

```python
from stair_common.cost import units

calls = [  # (stage, params_b, prompt_tokens, generated_tokens), with the tokens measured from each call
    ("retrieve", 7.0, 3500, 20),     # Qwen2.5-7B picks passages from the candidates
    ("answer",   3.0, 600, 40 * 16), # Llama-3.2-3B, 40 samples from one prompt: the prompt counts once
    ("fuse",     3.0, 1300, 16),     # fusing the samples is a new call with its own prompt
    ("router",   0.033, 512, 0),     # your router / judge, if any
]
per_stage = {}
for stage, p, prompt, gen in calls:
    per_stage[stage] = per_stage.get(stage, 0.0) + units(p, prompt, gen)
total = sum(per_stage.values())      # one 3B pass of 128+64 tokens = 1 unit
```

With vLLM, the tokens come from each output: `len(out.prompt_token_ids)` once, and
`sum(len(o.token_ids) for o in out.outputs)` over the samples. Samples drawn from one prompt pay that prompt once, as
AgentTTS counts them (App. A.2, Eq. 8); every fusion call pays its own prompt.

### 5. Write results in the shared layout, validate and save

```python
import pandas as pd
from stair_common.schema import COLUMNS, validate

rows.append({
    "dataset": "hotpotqa", "split": "test", "id": qid, "track": "nonLLM", "policy": "router@0.5", "menu": "A",
    "seed": 0, "answer": answer, **score(answer, golds),
    "units": total, "units_retrieve": per_stage.get("retrieve", 0.0), "units_answer": per_stage.get("answer", 0.0),
    "units_fuse": per_stage.get("fuse", 0.0), "units_router": per_stage.get("router", 0.0),
    "ms_router": 2.1, "models": "retriever=Qwen/Qwen2.5-7B-Instruct@a09a354...;answerer=...",
})
df = pd.DataFrame(rows, columns=COLUMNS)
validate(df)                                   # raises on a missing column, an ID outside the split, bad units
df.to_csv("results/nonLLM-hotpotqa-test.csv", index=False)
```

### 6. Pool the team's results and compare

```python
import glob
import numpy as np

pooled = pd.concat(pd.read_csv(p) for p in glob.glob("results/*-test.csv"))
per_seed = pooled.groupby(["dataset", "track", "policy", "menu", "seed"])[["lenient", "units"]].mean()
table = per_seed.groupby(["dataset", "track", "policy", "menu"]).agg(
    lenient_standard=("lenient", "mean"), lenient_min=("lenient", "min"), lenient_max=("lenient", "max"),
    units_per_question=("units", "mean"))

# Two policies, paired on the same questions: a 95% bootstrap interval of the difference
a = pooled.query("policy == 'router@0.5' and seed == 0").set_index("id")["lenient"]
b = pooled.query("policy == 'always-strong' and seed == 0").set_index("id")["lenient"].reindex(a.index)
boot = np.random.default_rng(0).choice((a - b).to_numpy(), size=(2000, len(a))).mean(axis=1)
print((a - b).mean(), np.percentile(boot, [2.5, 97.5]))
```

### 7. Train on train, tune on val, test once

```python
train_ids = load_split("hotpotqa", "train")            # 12,000 from the official train file (fine-tuning)
extra_ids = load_split("hotpotqa", "train_extra")      # dev[1000:], only if your labels exist only on dev
seen = set(load_split("hotpotqa", "nanojev_v1_seen"))  # nano-jev rows: also report on test minus these (438)
```

To rebuild the split files from the pinned sources, run `python -m stair_common.splits build --sources <dir>`. The
module docstring lists the files.

## 1. Splits: AgentTTS's test sets, fixed by position

**Test = AgentTTS's own test questions wherever the paper has them.** The headline numbers of every track are then
directly comparable with the paper, and no track has trained on a test question. The ID lists are committed in
`stair_common/splits/` (with `MANIFEST.json`: sizes, sha256, source files) and rebuilt exactly by
`python -m stair_common.splits build`.

| Dataset | Source (pinned) | train | val | test | Notes |
|---|---|---|---|---|---|
| HotpotQA | official train file; AgentTTS `hotpotqa-dev.json` (7,405) | 12,000 drawn from the train file; optional `train_extra` = dev[1000:7405] (6,405) | dev[0:500] | **dev[500:1000]**, AgentTTS's 500 | test is exactly what AgentTTS's code scores (`data[500:1000]`) |
| 2WikiMultiHopQA | AgentTTS `tts_train.json` (official dev, 12,576) | official train file; optional `train_extra` = dev[1000:] (11,576) | dev[0:500] | dev[500:1000] | AgentTTS cuts its test from a `queries.jsonl` it does not publish, so this is the same position, not provably the same questions |
| CWQ | AgentTTS `cwq/data_with_ct_0.json` (1,000 with gold triplets) | 350 | 150 | **[0:500]** = AgentTTS's `cwq_500_test500.json` | identical, checked question by question |
| WebQSP | AgentTTS `webqsp/data_with_ct_0.json` (1,000) | 350 | 150 | **[0:500]** = AgentTTS's `webqsp_500_test500.json` | identical, checked |
| MedAgentBench | `test_data_v2.json` (300 tasks, 10 categories × 30) | none | 150 `calib` | 150 | 15 / 15 per category |

- **Random parts use one rule:** sort the IDs, shuffle with `numpy.random.default_rng(20260729)`, then cut. These are
  the HotpotQA train draw, CWQ/WebQSP train/val, and MedAgentBench.
- **`train_extra`** is for tracks whose labels exist only on dev questions, such as nonLLM's stored outcome grid. Never
  train on `val` or `test`.
- **nano-jev** (menu B) saw 1,000 HotpotQA dev questions in its own calibration and test, not in its training. 62 of
  them are in our test and 66 in val. Report nano-jev rows on all 500 and also without
  `stair_common/splits/hotpotqa/nanojev_v1_seen.txt`; a track that tunes on val with nano-jev should know that 66 of
  those questions are not new to it.
- **500 test questions** give about ±4 points of interval per policy. Compare policies paired on the same questions,
  for example with a paired bootstrap, which is much tighter.

### Why not the 4,814 / 1,110 / 1,481 split (seed 20260729)

That split comes from the Config A archive (`Dataset/splits.json`), not from systemOne. systemOne trained on the
official train file with its own seed and tested on all 7,405. I checked whether the seed regenerates it:

- I tried numpy `default_rng`, `RandomState`, `random.shuffle` and sklearn, each with three part orders.
- None reproduces it. Every rule matches only about 300 of nonLLM's committed 1,481 test indices, which is chance.

Two more problems:
- Only about 100 of AgentTTS's 500 fall in that test split. The slide says 105; I count 94 against nonLLM's indices.
  The other 400 were training or validation data for the routers, so a fair comparison with the paper used about 100
  questions.
- It has no rule that would extend it to the other datasets.

**Moving existing results:** tracks with outcomes on all 7,405 dev questions don't need new model calls. Retrain the
router on `train_extra` (or the train file), pick thresholds on `val`, and re-score on `test`.

## 2. Scoring: one ruler, named every time

- **Headline: `lenient`**, the certified standard ruler `jonathan/final/lenient_match_agenttts.py` on branch
  [`jonathan`](https://github.com/mohameddhameem/STAIR/blob/jonathan/jonathan/final/lenient_match_agenttts.py). It
  reproduces AgentTTS's published 0.746. `stair_common/` carries a byte-for-byte copy, which a test pins by sha256, and
  `score.py` loads that copy as it is. Nobody re-implements it. If the original changes, copy it again and update the
  hash.
- **Also reported: `em` and `f1`**, the official HotpotQA scoring, best over the gold aliases.
- **The standing rule:** never mix rulers in one table, and name the ruler next to every accuracy. A track's own extra
  grader (for example systemOne's LLM grader) is reported beside these, never instead of them.

## 3. Cost: AgentTTS units only

`units = 2 × params × (prompt + generated tokens) / 1,152 GFLOPs`
- One unit is one Llama-3.2-3B pass of 128 + 64 tokens, as in AgentTTS Eq. 2–4.
- Tokens are measured per call and summed over every call: retrieval, answering, fusion, and the router or judge.
- **Samples share their prompt:** S samples from one prompt count that prompt once, plus every sample's generated
  tokens, as in AgentTTS App. A.2, Eq. 8 (FLOPs = prompt + S × decode). Counting the prompt once per sample overstates
  sampling badly: for 40 fused samples of Llama-3.2-3B on 5 passages it gives 220 units instead of 71, more than a
  70B answer (94).
- `params_b` is the nominal size in `models.toml`.
- **AgentTTS's own tables are on another scale.** Its Tables 5–9 give a normalised budget (Eq. 3–4): the number of 3B
  samples, sharing one 128-token prompt, with the same FLOPs. For a call of u units that is 3u − 2.
  - Example: Qwen2.5-72B retrieval at the paper's Table 4 lengths (2,048 + 128 tokens) is 272 units here, and 814 in
    Table 5.
  - Convert with 3u − 2 before comparing with the paper's tables, or with the team slide's AgentTTS ladder
    (strong-everything 929).
  - The slide's AgentTTS recipe (937) also predates Eq. 8. Under Eq. 8 it is 302 units: 72B retrieval 272, plus 40 3B
    samples from one prompt 15, plus a fusion call reading them 15. That is 97% of strong-everything (311).
- **Limitation, stated with the numbers:** like AgentTTS, it counts parameter FLOPs and not attention, so it undercounts
  small models on long prompts. We keep it, because it is the paper's unit.

## 4. Models: one copy each, 16-bit

- **Weights:** bf16 weights from Hugging Face, pinned to a revision in `models.toml`, served with vLLM.
- **AWQ 4-bit only for the 70B and 72B models**, so that they fit our two GPUs.
- **QLoRA training copies:** an AWQ model can't be fine-tuned, so QLoRA trains on the bf16 weights pinned in
  `models.toml` under `[training.qlora]`, loaded in 4-bit. Serve the adapter on the menu's AWQ copy; cost uses the
  menu entry's `params_b`.
- **Menus:**
  - A, the AgentTTS menu, runs on every dataset;
  - B, the full ladder, runs on every dataset;
  - C, the Gemma / Ministral swap, runs on HotpotQA only.
- GGUF copies are not used for pooled results, because a 4-bit GGUF is a different model from the one in the paper.

## 5. Training

See [TRAINING.md](TRAINING.md): the LoRA / QLoRA recipe, seeds 0, 1 and 2, and a table where each track records the
epochs it used (to be reconciled later, as agreed), and how to log and plot loss curves with
`stair_common.training`.

## 6. Results layout and metrics

Share results as one row per (dataset, split, id, track, policy, menu, seed). The columns are listed in
`stair_common/schema.py`: answer, lenient, em, f1, units with a split by stage, ms_router and the models.

- **Mandatory:** lenient accuracy and AgentTTS units, both on `test`.
- **Optional, per track:** EM, F1, ms per routing decision, AUC / APGR. These are to be finalised at the next meeting.

## Answers to the Teams questions

| Question | Answer |
|---|---|
| HotpotQA split code / IDs, plain random? | The 4,814/1,110/1,481 split is not systemOne's and can't be regenerated from its seed (§1). Proposed instead: AgentTTS's test slices, committed in `stair_common/splits/`. |
| GGUF or full precision? | bf16 from Hugging Face, pinned revisions; AWQ only for 70B/72B (§4, `models.toml`). |
| nano-jev as the smallest retriever? | Yes: v1.0 (revision `ffee345`), keep its top 5. On AgentTTS's 500 test questions it puts both gold passages in its top 5 for 77% of them, at 2.9 units against 272 for Qwen 72B at the paper's token lengths (814 on AgentTTS's Table 5 scale, §3). It ranks passages only. |
| One lenient rule and one script? | `lenient_match_agenttts.py`, through `stair_common.score` (§2). |
| Correct the 2 × size × tokens formula? | No. Keep AgentTTS units and state the limitation (§3). |
