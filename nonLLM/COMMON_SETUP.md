> Historical18 October setup audit, superseded by upstream common0.3. It documents what the completed500-question run used. See [results/README.md](results/README.md) and current [common/README.md](../common/README.md).

# STAIR common setup sync — 18 October 2026

Synced `origin/main` commit `c906e3dcdb0a704e06a16061809799b08851ad6d` into branch `nonLLM` (merge `c02b09367718249af962140e605a91ddd392baa0`). Installed and checked `stair-common==0.2.0` locally. This document records the implementation currently merged into main. **The upstream `common/README.md` still calls it a proposal until team agreement**; it does not establish that every change has been approved in Teams. The pasted discussion and repository were available; the Word/PPT attachment itself was not.

Authority: [common setup](../common/README.md), [pinned models](../common/stair_common/models.toml), [training](../common/TRAINING.md), and [experiment submission instructions](../experiments/CLAUDE-INSTRUCTIONS.md).

## Dataset IDs take precedence over remembered seeds

| Dataset | Router training / calibration | Validation | Headline test |
|---|---|---|---|
| HotpotQA | 12,000 IDs drawn from official train; existing dev-grid tracks can use `train_extra` = 6,405 dev IDs | official dev positions `[0:500]` | official dev positions `[500:1000]`, 500 IDs |
| 2WikiMultiHopQA | official train, or provided `train_extra` = 11,576 | first500 of the specified dev source | next500 of that source |
| CWQ |350|150|first500 of the supplied1,000|
| WebQSP |350|150|first500 of the supplied1,000|
| MedAgentBench |150 stratified `calib`, no fine-tuning|use calibration as specified, not a separate invented split|150, fifteen per task category|

Use `stair_common.splits.load_split`, not local row numbers or a new random draw. Random draws sort IDs then use `numpy.random.default_rng(20260729)`. Sampling/training seeds are separately0/1/2.

The earlier chat described4,814/1,110/1,481 and a random500 from the old test using seed20261015. **Current main implements a different protocol**: fixed committed AgentTTS-position slices. It does not implement either the first500 as test or the random500-from-old-test plan. Our audit confirms current test IDs equal AgentTTS dev `[500:1000]` exactly. 2Wiki uses the same positions but upstream explicitly cannot prove the same question identities as the paper's unavailable queries file.

**Observed historical overlap:** new HotpotQA test500 = old train335 + old validation71 + old test94. New validation500 = old train317 + old validation78 + old test105. Reusing old trained routers would contaminate the new test; retrain on `train_extra` or official train.

## Model menus and copies

Run **A and B on every dataset, including MedAgentBench**. C is an additional HotpotQA-only swap menu.

| Menu | Retrieval choices | Answering choices |
|---|---|---|
| A — AgentTTS model menu | Qwen2.5-Instruct7B /32B /72B | Llama-3.2-3B-Instruct; Llama-3.1-8B /70B-Instruct |
| B — full ladder | nano-jev33M; Qwen2.5-Instruct0.5B; Qwen3.5-0.8B; Qwen2.5-Instruct1.5B /3B /7B /32B /72B | Llama-3.2-Instruct1B /3B; Llama-3.1-Instruct8B /70B |
| C — HotpotQA swap | Gemma3-270M /1B; Gemma4-E2B /E4B /31B | Ministral3-3B /8B /14B |

Load repository, revision, parameter count and dtype from `stair_common.load_models()`. Main uses bf16 served with vLLM; **only70B/72B use pinned AWQ copies**. Training copies for QLoRA are pinned separately and loaded from bf16 weights with bitsandbytes, not trained directly from AWQ files. These are the common pooling copies; matching the paper's model menu does not mean its original quantization/backend is identical.

Our six downloaded Q4_K_M GGUF models remain valid exploratory assets but **are not the common pooled-run copies**. Nothing has been deleted or silently replaced. Synchronizing settings does not download the new bf16/AWQ copies or start fresh generation.

nano-jev: `sdmlai/nano-jev`, revision `ffee345e97797bef0948561461ea54fe558588a0`, fp32, retain top5 passages; it ranks rather than generates. The standard lists1,000 dev IDs it previously saw:62 in common test,66 in validation. Report nano-jev results on all500 test IDs and on the438-ID unseen subset.

## Scoring

Import `stair_common.score.score` or `lenient`. The packaged author scorer is byte-identical to `origin/jonathan:jonathan/final/lenient_match_agenttts.py`; SHA256 `2f184ac252a26df49d402f22f9ba52af3db6ed4812adb42507d3bbb989a0c882`. Do not reimplement it. Its header claims reproduction of published74.6%; this sync verified code identity, not that published accuracy independently.

The newly available standard is **different from our historical reconstruction**: ASCII punctuation normalization, sequence containment in either direction, and quantity-prefix guards. In particular the common code accepts `1936` for gold `7 January 1936` and `Yes\nNo` for gold `yes`; it rejects `more than 20` for gold `20`. See exact machine-readable probes in the audit rather than inventing extra safeguards.

Audited29,620 cached Qwen1.5B/7B answers:1,861 outcome differences versus saved reconstructed lenient labels (WW405 / WS434 / SW518 / SS504). Therefore rebuild all grading-dependent labels, rewards, model selection and summaries for common reruns. Historical run directories and reported scores retain their original scorer identity.

Headline metric: final-answer standard lenient. Also emit official EM/F1 via the common wrapper. Retrieval-specific accuracy/coverage and clinical task-success semantics are not fully implemented by the generic string scorer; the pasted chat does not establish a final common definition for those.

## Cost

Use `stair_common.cost.units(params_b, prompt_tokens, generated_tokens)` for every model invocation:

`units = 2 * params_b * (prompt_tokens + generated_tokens) / 1152`

One unit = nominal3B with128 prompt +64 output tokens. For S samples sharing one prompt, charge the prompt once and sum all generated tokens. A fresh fusion/router/judge prompt is another charged call. Use measured tokens, nominal sizes from the pinned menu, and the per-stage output columns.

Do not substitute our historical mean token costs, percent-of-SS tables, or one-prompt-per-sample totals for absolute common units. Illustration:3B,600-token prompt,40 outputs of16 tokens costs6.4583 units with shared prompt versus128.3333 when charging the prompt40 times, excluding fusion. This is a formula example, not a recomputed experiment result.

Retain the parameter-FLOP approximation; state that attention FLOPs are excluded and small models with long prompts are undercounted. Record non-LLM routing latency separately; do not invent an LLM parameter cost for a scikit-learn router.

## Training and outputs

- Executor training and router training are distinct. Existing non-LLM runs did not fine-tune Qwen executors; their GRPO trains linear routing heads.
- Common executor recipe: <=1B LoRA12,000x2;1.5–14B LoRA12,000x1;24–72B QLoRA2,000x1, LoRA rank16/alpha32/lr1e-4. These are proposal recipes/max exposures; keep best validation loss and record the actual epochs. CWQ/WebQSP have only350 training IDs; MedAgentBench has no fine-tuning.
- Use seeds0/1/2 and report mean/range. Common does not require every method to have the same epoch count.
- Log training and validation losses in `stair_common.training` layout. Our old120/300-epoch GRPO streams logged expected validation accuracy/cost rather than a common validation-loss early-stopping history. A training/selection adapter is required.
- Closed-form or tree/classical fits have no meaningful neural training epoch. Record actual fit settings and validation selection rather than fabricate an epoch count.
- Emit one row per dataset/split/id/track/policy/menu/seed, including final answer, lenient/EM/F1, total and per-stage units, routing latency, model repo@revision. Run `stair_common.schema.validate` before pooling.
- Optional AUC/APGR/latency reporting is still described as needing team metric finalization. Common includes EM/F1 output columns even though README calls secondary metrics optional for reports.

## Migration status and remaining work

Completed here: merge main; install common0.2.0;30 common tests; experiment-template checker; read-only split/scorer/cost audit. No historical raw caches, checkpoints or result files were rewritten. No HPC jobs were submitted or models retrained as part of this settings sync.

Before a pooled result: use pinned model copies; generate caches as needed; rebuild standard labels/rewards; retrain routers on disjoint IDs with new seeds and validation selection; measure all actual calls; export and validate the common layout. Old52.79%,52.94% and GRPO results cannot be copied into the common table under new labels.

MedAgentBench A/B belongs in the plan early, but a real environment/executor and its task-success evaluation remain necessary. The common package provides task IDs and the generic result format, not a complete clinical execution harness. Do not claim that it has already run.

The repository's experiment-sheet workflow is separate: first identify which new/rerun experiments to include; write `experiments/<id>.md` with a pushed code SHA, executable portable commands, outputs, walltime and <=2GPUs; run `experiments/check.py`. Entries are reviewed before committing/pushing under that workflow. This settings sync does not publish entries or submit cluster work.

Audit artifacts: [read-only audit code](experiments/common_alignment/audit.py), [audit results](experiments/common_alignment/audit_20261018.json). Reproduce from repository root after installing common:

```bash
python nonLLM/experiments/common_alignment/audit.py
python -m pytest common/tests -q
python experiments/check.py
```
