# Four further studies: paired routing, strict cross-fitting, risk gating, and cascades

Completed output: `runs/nonllm_common_ws_next_v1`. Source: the corrected four-executor GGUF cache; no new LLM calls or executor training. Lenient scoring and token-derived AgentTTS units remain unchanged. This is common0.2 test500, not the common0.3 SLR golden set.

## Fitting and calibration partition

The original6405 fitting IDs are split, seed20261024, into5125 model-fit and1280 calibration IDs. Validation500 and test500 remain the same. No calibration labels fit any new base/meta model. These calibration examples and the test were used in earlier research, so this run is exploratory and does not claim a new certified risk guarantee. Previous6405-fit routers have a larger fitting-label budget.

Features are raw deterministic numerical lexical statistics. There is no fitted vocabulary, global IDF, scaler, encoder LM, or reuse of earlier trained router weights. Each question's two QA branches share every fold.

## 1. Four-route paired rescue/harm learning

For WW/WS/SW versus SS, classify rescue, harm, or tie; predict expected accuracy change as P(rescue)-P(harm). Within each retrieval branch, a QA reviewer similarly compares weak and strong QA. Compare complete-path selection before execution (`pair_joint`) and evidence-conditioned second-stage selection (`pair_staged`). Include outcome-regression and signed-gain-regression experts as controls.

Separate regressors forecast future call costs from available features; actual unexecuted costs are never decision inputs. A fixed per-unit penalty grid is selected on validation or OOF, with learned-only and fixed-fallback versions separate.

## 2. Strict nested OOF stacking and model selection

Five outer folds generate honest held-out predictions for base and stacked policies. Each outer stack trains its meta models on three inner OOF folds confined to that outer training set. The outer held-out labels cannot influence inner base models or meta fitting. The final stack follows the same inner3-fold procedure over all5125 fit IDs.

Three seeds0/1/2,96 trees per estimator. Base experts are outcome regression, signed-gain regression, and rescue/harm/tie classification. Meta outcome regressors combine raw features and OOF expert predictions. Keep global method selection on outer OOF separate from global selection on validation. Select checkpoints/weights before calibration/test, not by the best test row.

## 3. Held-out CRC-style risk calibration

Joint pair/stack proposers freeze on validation before calibration labels are read. Confidence comes from predicted harm. A nested threshold switches to the proposed alternative only when confidence exceeds the threshold; otherwise execute direct SS before any weak call.

Loss is additional harm relative to SS: switch AND SS correct AND alternative wrong. It is averaged over ALL queries, not just switchers, and is not overall answer error. Calibration uses alpha0.005/0.01/0.02 and `(harm+1)/(n+1)<=alpha`, or known-zero no-switch fallback. All proposer/alpha policies are reported rather than choosing by test. This is an expected-risk criterion under assumptions, not a finite-test error cap; historical data reuse precludes a certified claim here.

## 4. Fully charged weak-first cascade

Run weak retrieval first. The retrieval reviewer sees question, supplied pool, and actual weak evidence/observed retrieval metadata; it does not see strong retrieval or weak QA yet. After keeping/escalating retrieval, run weak QA on the selected evidence. The QA reviewer sees only that paid weak answer/evidence and can keep or escalate.

Cost is W-retrieval + optional S-retrieval + W-QA(selected retrieval) + optional S-QA(selected retrieval). Every prompt/generation is counted. Router CPU work remains outside the executor FLOP proxy. Always-escalate produces the same final answers as SS but109.61% SS cost; it is not relabeled as100%.

## Results

[Full report](../../results/menu_ws_next_v1/REPORT.md), including all114 policies/57000 policy-question records and fold/risk/cost audits. The500 test questions are reused across policies, not57000 independent examples.

- OOF stack joint, per-family validation maximum-accuracy selection:72.0% /90.70% SS cost, one more correct than SS.
- Outer-OOF global match-SS selection:71.8% /89.77%, selected outcome_joint.
- Stack joint with alpha0.5% risk gate:72.0% /91.36%; this test realization has zero extra harms and one rescue, not a formal guarantee.
- Weak-first cascade maximum-accuracy selection:71.8% /98.21%; its95%-validation-target selection:71.4% /74.13%.
- Accuracy-focused pairwise router selects SS throughout this test; its cheaper selection is70.6% /81.39%.

Some calibrated policies' observed test harm rates exceed nominal alpha. All calibration criteria nevertheless passed; conditional sample results and repeated exploration must not be called certified risk control. Full online routing latency is unmeasured in this vectorized cache replay.

Commands from the workspace containing STAIR and the local caches:

```bash
python STAIR/nonLLM/experiments/menu_ws_gguf/test_next_studies.py
python STAIR/nonLLM/experiments/menu_ws_gguf/next_studies.py --out runs/nonllm_common_ws_next_v1
python STAIR/nonLLM/experiments/menu_ws_gguf/summarize_next_studies.py
```

The folds use local CPU subprocesses; no cluster jobs or outside messages are submitted. Large model artifacts, caches, and fitted forest bundles remain local.
