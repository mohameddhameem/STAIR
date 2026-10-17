# STAIR common setup: one split, one ruler, one cost, one results layout

This is the shared setup every track uses, so that results can be pooled. It settles the "Decisions needed" in
the team's proposal "STAIR shared run configuration" (17 Oct) and answers the questions asked in the Teams chat. It is a
proposal until the team agrees. Comment on the pull request.

```bash
pip install -e common            # numpy + pandas; python >= 3.10
```
```python
from stair_common.splits import load_split   # load_split("hotpotqa", "test") -> 500 question IDs
from stair_common.score import score         # score(answer, golds) -> {"lenient", "em", "f1"}
from stair_common.cost import units          # units(params_b, prompt_tokens, generated_tokens) -> AgentTTS units
from stair_common.schema import validate     # validate(results_df) before you share a results file
```

## 1. Splits: AgentTTS's test sets, fixed by position

**Test = AgentTTS's own test questions wherever the paper has them.** The headline numbers of every track are then
directly comparable with the paper, and no track has trained on a test question. The ID lists are committed in
`splits/` (with `MANIFEST.json`: sizes, sha256, source files) and rebuilt exactly by
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
  them are in our test. Report nano-jev rows on all 500 and also without `splits/hotpotqa/nanojev_v1_seen.txt`.
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
- Tokens are measured per call and summed over every call: retrieval, every sample, fusion, and the router or judge.
- `params_b` is the nominal size in `models.toml`.
- **Limitation, stated with the numbers:** like AgentTTS, it counts parameter FLOPs and not attention, so it undercounts
  small models on long prompts. We keep it, because it is the paper's unit.

## 4. Models: one copy each, 16-bit

- **Weights:** bf16 weights from Hugging Face, pinned to a revision in `models.toml`, served with vLLM.
- **AWQ 4-bit only for the 70B and 72B models**, so that they fit our two GPUs.
- **Menus:**
  - A, the AgentTTS menu, runs on every dataset;
  - B, the full ladder, runs on every dataset;
  - C, the Gemma / Ministral swap, runs on HotpotQA only.
- GGUF copies are not used for pooled results, because a 4-bit GGUF is a different model from the one in the paper.

## 5. Training

See [TRAINING.md](TRAINING.md): the LoRA / QLoRA recipe, seeds 0, 1 and 2, and a table where each track records the
epochs it used. That table is to be reconciled later, as agreed.

## 6. Results layout and metrics

Share results as one row per (dataset, split, id, track, policy, menu, seed). The columns are listed in
`stair_common/schema.py`: answer, lenient, em, f1, units with a split by stage, ms_router and the models.

- **Mandatory:** lenient accuracy and AgentTTS units, both on `test`.
- **Optional, per track:** EM, F1, ms per routing decision, AUC / APGR. These are to be finalised at the next meeting.

## Answers to the Teams questions

| Question | Answer |
|---|---|
| HotpotQA split code / IDs, plain random? | The 4,814/1,110/1,481 split is not systemOne's and can't be regenerated from its seed (§1). Proposed instead: AgentTTS's test slices, committed in `splits/`. |
| GGUF or full precision? | bf16 from Hugging Face, pinned revisions; AWQ only for 70B/72B (§4, `models.toml`). |
| nano-jev as the smallest retriever? | Yes: v1.0 (revision `ffee345`), keep its top 5. On AgentTTS's 500 test questions it puts both gold passages in its top 5 for 77% of them, at 2.9 units against 814 for Qwen 72B. It ranks passages only. |
| One lenient rule and one script? | `lenient_match_agenttts.py`, through `stair_common.score` (§2). |
| Correct the 2 × size × tokens formula? | No. Keep AgentTTS units and state the limitation (§3). |
