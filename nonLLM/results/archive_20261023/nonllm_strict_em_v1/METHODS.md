# Corrected non-LLM experiments

Primary metric: official HotpotQA normalized answer exact match, maximized over
the supplied gold answer aliases. This is a deliberate, independently verifiable
metric choice. It is not claimed to reproduce AgentTTS's missing
`evaluation_2wiki.py` or the reference archive's missing `mint_labels.py`.

The reference value 52.79% is the supplied label grid's validation **lenient**
score (586/1,110). It is not an exact-match target. In the local regenerated
Instruct grid, validation SS EM is 521/1,110 = 46.94%, and test SS EM is
699/1,481 = 47.20%. The original supplied grid has validation SS EM 46.85%
and test SS EM 48.14%; those are different model outputs.

## Scope

The rerun covers five original-style classifier families, the full October
cascade and no-weak-answer ablation, the 27 initial improvement comparisons,
the extended exact-reward / GRPO grid (144 seed-specific training streams,
144 validation candidates, 24 selected operating points), the 12 architecture
operating points, and three combined objectives. All score-dependent targets,
sampled-answer labels, policy rewards and validation selections are rebuilt.
Config A's retrieval coverage labels are retained because coverage is not a
QA correctness metric. Original-style retrieval and QA labels use EM outcomes,
and unresolved rows are skipped during fitting.

No new LLM generation is required: the existing complete greedy grid uses
Qwen2.5-1.5B-Instruct / 7B-Instruct in FP16. Weak-answer confidence and random
samples were also generated with the 1.5B Instruct model. Early Base/GGUF
live-pipeline experiments are excluded. Qwen weights are never trained here.

All classifiers use the same six measured mean parameter-weighted token costs.
Direct routing skips unused weak calls; cascades include sunk weak calls;
consensus pays for all answers; post-answer choice pays for both answers.
CPU feature extraction / routing overhead and training cost are excluded.
Confidence cost assumes extracting logits from ordinary generation.

## Reproduction

Run from `<WORKSPACE>` into a fresh output directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .cascade-env/bin/python -u \
  STAIR/nonLLM/experiments/router_improvements/corrected_run.py \
  --out runs/nonllm_strict_em_v1
```

The coordinator generates an isolated source snapshot under `code/`, a strict-EM
grid under `base/`, provenance under `protocol.json`, independent logs, and a
`status.json`. Existing answers, samples and target-free extended-policy
features are reused through links; prior labels, weights and results are not
modified. Extended features are safe to reuse because their fitting was
unsupervised and training-only on identical observations. Every reward and
target array is newly written; no previous lenient reward array is reused.

Legacy scripts expect `em_lenient`, `lenient_accuracy` and a two-value scoring
tuple. In this isolated snapshot **both metric fields / tuple slots represent
strict EM**. This compatibility alias is declared in the protocol; it does not
perform containment matching. Human reports label the metric as exact match.
Primary aggregated results use `accuracy` with explicit primary-metric metadata.

After the coordinator finishes:

```bash
.cascade-env/bin/python \
  STAIR/nonLLM/experiments/router_improvements/build_strict_html_report.py \
  --run runs/nonllm_strict_em_v1
```

The English HTML replaces the report's main entry and preserves the previous
lenient page as `reports/router_methods_results_exploratory.html`.

## Validation and limits

The official normalization / equality functions are extracted from the local
official HotpotQA reference file. All 29,620 greedy QA judgments are checked
against stored strict outcomes. Sampled answers are graded again before target
construction. Model selection and operating points use train/validation only;
architecture model families use question-grouped training CV. Existing sampling
variants retain equal weight per training question.

This test split has already been examined. Results and paired intervals are
exploratory; independent data is needed to establish generalization. This rerun
fixes metric consistency; it does not certify identity to the original author's
full generation environment, prompts, precision or undistributed scorer.
