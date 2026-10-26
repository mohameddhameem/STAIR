# STAIR common setup: one split, one ruler, one cost, one results layout

This is the shared setup every track uses, so that results can be pooled. It settles the "Decisions needed" in
the team's proposal "STAIR shared run configuration" (17 Oct) and answers the questions asked in the Teams chat. It is a
proposal until the team agrees. Comment on the pull request.

## HotpotQA golden set: Jonathan's Stage-Level Routing setup (from 21 Oct)

On HotpotQA the team follows Jonathan's SLR setup (branch `jonathan` at `5b69cce`). Its certified sealed-test numbers
are the reference every HotpotQA track reports against.

| | Golden set (HotpotQA headline) |
|---|---|
| Split | `slr-train` / `slr-val` / `slr-test`: 4,814 / 1,110 / 1,481 of the 7,405 dev questions (§1) |
| Ruler | `lenient`, unchanged (§2) |
| Cost | AgentTTS Table 5 prices, weak-first, every call that ran counted; the router is not charged; strong-everything = 929 (§3) |
| Reference rows (`slr-test`) | always-strong 76.77 at 929; SLR SFT (label) 76.70 at 834.8; SLR SFT + reward 75.96 at 664.7 |

- Pick thresholds and checkpoints on `slr-val`, report once on `slr-test`.
- nano-jev saw 178 of the `slr-test` questions. Report nano-jev rows on all 1,481 and also without
  `nanojev_v1_seen.txt`.
- **What stays:**
  - AgentTTS's 500 (`test` = dev[500:1000]) is for comparing with the paper only;
  - the other datasets keep their splits and AgentTTS units.
- **Moving existing results:** a track with outcomes on all 7,405 dev questions needs no new model calls. Retrain the
  router on `slr-train`, pick thresholds on `slr-val`, and re-score on `slr-test` at the Table 5 prices.
  - systemOne and selfLearnt entries made against the dev slices move this way.

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
| `from stair_common.splits import load_split` | `load_split("hotpotqa", "slr-test")` gives the golden set's 1,481 question IDs. Parts are `train`, `train_extra`, `val`, `calib` and `test`, and on HotpotQA also `slr-train`, `slr-val` and `slr-test`. |
| `from stair_common.score import score` | `score(answer, golds)` gives `{"lenient": 0/1, "em": 0/1, "f1": 0–1}`. |
| `from stair_common.cost import units, budget, TABLE5_HOTPOTQA, SLR_PATH_PRICES` | `units(params_b, prompt_tokens, generated_tokens)` gives AgentTTS units for one call. The golden set prices calls at Table 5 (`budget()` for a call it doesn't list). |
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
| 2WikiMultiHopQA | official dev (12,576, all answered): the HF mirror `framolfese/2WikiMultihopQA` @ `fe713bf`, `validation` (saved as `2wikimultihopqa_dev.parquet`) | official train file; optional `train_extra` = the other 11,576 dev questions | 500 dev, by the rule | the next 500 dev, by the rule | **Changed in 0.4.0** (STAIR #9): the source was AgentTTS `tts_train.json`, which is 2Wiki's official *test* set, with every answer empty. AgentTTS's own 2Wiki test comes from a `queries.jsonl` it does not publish, so no comparability is lost |
| CWQ | AgentTTS `cwq/data_with_ct_0.json` (1,000 with gold triplets) | 350 | 150 | **[0:500]** = AgentTTS's `cwq_500_test500.json` | identical, checked question by question |
| WebQSP | AgentTTS `webqsp/data_with_ct_0.json` (1,000) | 350 | 150 | **[0:500]** = AgentTTS's `webqsp_500_test500.json` | identical, checked |
| MedAgentBench | `test_data_v2.json` (300 tasks, 10 categories × 30) | none | 150 `calib` | 150 | 15 / 15 per category |

- **Random parts use one rule:** sort the IDs, shuffle with `numpy.random.default_rng(20260729)`, then cut. These are
  the HotpotQA train draw, all three 2Wiki parts, CWQ/WebQSP train/val, and MedAgentBench.
- **`train_extra`** is for tracks whose labels exist only on dev questions, such as nonLLM's stored outcome grid. Never
  train on `val` or `test`.
- **nano-jev** (menu B) saw 1,000 HotpotQA dev questions in its own calibration and test, not in its training. 62 of
  them are in our test and 66 in val. Report nano-jev rows on all 500 and also without
  `stair_common/splits/hotpotqa/nanojev_v1_seen.txt`; a track that tunes on val with nano-jev should know that 66 of
  those questions are not new to it.
- **500 test questions** give about ±4 points of interval per policy. Compare policies paired on the same questions,
  for example with a paired bootstrap, which is much tighter.

### The golden set's split (`slr-*`, HotpotQA)

Jonathan's 4,814 / 1,110 / 1,481 split of the 7,405 dev questions (65 / 15 / 20, seed 20260729).
- **How it was made:** his `make_splits.py` stratified it on question type, level and the Config A label. It can't
  be regenerated from the seed alone. So `stair_common` takes it by position from his committed
  `lab_package/splits.json` (branch `jonathan` at `5b69cce`), and the `MANIFEST.json` pins that file's sha256.
- **Overlap with AgentTTS's 500:** 94 of them are in `slr-test`; 335 are in `slr-train` and 71 in `slr-val`. That's
  why the paper comparison keeps its own slice (`test` = dev[500:1000]).
- **No rule for other datasets:** it covers HotpotQA only. The other datasets keep the splits above.

## 2. Scoring: one ruler, named every time

- **Headline: `lenient`**, the certified standard ruler `jonathan/final/lenient_match_agenttts.py` on branch
  [`jonathan`](https://github.com/mohameddhameem/STAIR/blob/jonathan/jonathan/final/lenient_match_agenttts.py). It
  reproduces AgentTTS's published 0.746. `stair_common/` carries a byte-for-byte copy, which a test pins by sha256, and
  `score.py` loads that copy as it is. Nobody re-implements it. If the original changes, copy it again and update the
  hash.
- **Also reported: `em` and `f1`**, the official HotpotQA scoring, best over the gold aliases.
- **The standing rule:** never mix rulers in one table, and name the ruler next to every accuracy. A track's own extra
  grader (for example systemOne's LLM grader) is reported beside these, never instead of them.

## 3. Cost: Table 5 prices for the HotpotQA golden set, AgentTTS units for the rest

**Golden set (HotpotQA `slr-*`):** price every call at AgentTTS's Table 5, as Jonathan's `configb_test_readout.py`
does.
- **Prices:** `cost.TABLE5_HOTPOTQA`, per call, on AgentTTS's normalised budget scale (not units). For example,
  Qwen2.5-72B retrieval is 814 and Llama-3.1-70B answering is 115.
- **Weak-first:** every call that ran is counted. Menu A's paths cost `cost.SLR_PATH_PRICES`: WW 80, WS 195, SW 894,
  SS 1,009. Strong-everything (`cost.STRONG_EVERYTHING`) is 929.
- **Samples:** S samples from one prompt are one call (Eq. 8). For example, 40 Llama-3.2-3B answers cost
  `budget(3, 256, 40 × 64)` = 42.
- **Calls Table 5 doesn't list:**
  - another model's retrieval or answering call is priced with `cost.budget()` at Table 4's lengths: retrieve
    2,048 + 128 tokens, answer 256 + 64;
  - a router or judge call is priced with `cost.budget()` at its measured tokens.
- **The router is not charged** in the golden set's rows. A track whose question needs it adds a second column
  "with router cost".
- **In the results:** prices go in the `price` column (§6).

**Everything else** (other datasets, AgentTTS's 500 slice, measured comparisons): AgentTTS units.

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

- **Mandatory:**
  - HotpotQA: lenient accuracy and `price` on `slr-test`, plus the subset without nano-jev's seen questions;
  - other datasets: lenient accuracy and AgentTTS units on `test`.
  - Keep the `units` columns on HotpotQA too (measured), so that magnitudes can be compared.
- **Optional, per track:** EM, F1, ms per routing decision, AUC / APGR. These are to be finalised at the next meeting.

## Answers to the Teams questions

| Question | Answer |
|---|---|
| HotpotQA split code / IDs, plain random? | Since 21 Oct the team follows Jonathan's 4,814 / 1,110 / 1,481 split as the golden set: `slr-train` / `slr-val` / `slr-test`, taken by position from his committed `splits.json` (§1). AgentTTS's test slice stays for comparing with the paper. |
| GGUF or full precision? | bf16 from Hugging Face, pinned revisions; AWQ only for 70B/72B (§4, `models.toml`). |
| nano-jev as the smallest retriever? | Yes: v1.0 (revision `ffee345`), keep its top 5. On AgentTTS's 500 test questions it puts both gold passages in its top 5 for 77% of them, at 2.9 units against 272 for Qwen 72B at the paper's token lengths (814 on AgentTTS's Table 5 scale, §3). It ranks passages only. |
| One lenient rule and one script? | `lenient_match_agenttts.py`, through `stair_common.score` (§2). |
| Correct the 2 × size × tokens formula? | No. Keep AgentTTS units and state the limitation (§3). |
