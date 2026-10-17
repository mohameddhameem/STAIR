# Experiment 9: Amazon's Strands Decider as the System One judge

Replaces the 33M nano-jev judge with a 2B or 4B decision model from Amazon's Strands Decider family, zero-shot and
fine-tuned, in both settings: experiment 7 (pair A, small or large) and experiment 8 (five options, graded labels).
Plan: [experiment-9-plan.md](../experiment-9-plan.md). Job sheet: [gpu-jobs-exp9.md](../gpu-jobs-exp9.md). Run on
omega on 2026-10-16 and 2026-10-17.

**Result:**

- **On experiment 7 the deciders are better judges than nano-jev, and the gain survives their cost.** The fine-tuned
  2B with the small model's answer (qpa) reaches AIQ **0.708** against nano-jev qp's **0.673**: paired difference
  **[+0.030, +0.039]**. Its judge cost (about 1,250 GFLOPs a question, 38× nano-jev and slightly more than the small model's answer) lowers it to 0.702, still
  0.029 above nano-jev. The E4B qpa is higher still before cost (0.713) but about the same after it (0.700).
- **Most of that gain needs the answer.** Reading only the question and passages (qp), the fine-tuned 2B gains
  **+0.007 [+0.004, +0.011]** over nano-jev qp, and +0.003 once its cost is paid. nano-jev did not gain from the
  answer (qpa 0.673); the deciders gain about 0.03 from it.
- **Zero-shot, the E4B is already better than any nano-jev judge** (0.693, [+0.016, +0.024] over nano-jev qp; 0.683
  with its cost). The zero-shot 2B ties nano-jev ([−0.002, +0.006]).
- **On experiment 8 no decider helps.** Zero-shot, both tie the fixed-policy hull (0.698–0.700 against 0.701), as
  nano-jev qp did. Fine-tuned, both fall below it, the E4B significantly (qp [−0.022, −0.004]). Every decider is
  below nano-jev qpa (0.713), with intervals below zero.

## Setup

| | |
|---|---|
| Models | `StrandsAgents/strands-decider-2B-hobson-v21` (Qwen3.5-2B torso, `choice` temperature 0.818) and `StrandsAgents/strands-decider-E4B-gemma4-v1-2610` (Gemma 4 E4B, 0.775). Package `strands-decider` at git `6039cf8b` |
| Wrapper | `stair.decider.DeciderJudge`: each item becomes a decider `choice` question on the state, rendered and cut as the package's own engine does. Smoke job 42761: our probabilities match the engine's to 3–4e-5 on both models |
| Zero-shot | The checkpoint's calibrated probabilities. Experiment 7: nano-jev's two zero-shot questions (`grounded`, `correct`). Experiment 8: the five-option "which answer option" question |
| Fine-tuned | nano-jev's protocol on the LoRA and the pointer head only: grouped cross-entropy, 2 epochs at lr 1e-4, the best dev-NLL epoch kept, 3 seeds; scores are the mean over the seeds |
| Judge cost | 2 × parameters × tokens: 2B for the Qwen model, 4B active for E4B, against nano-jev's 0.0334B. A fine-tuned judge pays all 3 seeds' passes, as nano-jev's did |
| Window | The deciders keep their 4,096-token window; nano-jev cut its input at 512 tokens |
| Evaluation | Unchanged from experiments 7 and 8 (AIQ, paired bootstrap with 1,000 resamples, the clean subset without nano-jev's validation questions). Experiment 7 adds `paired_references`, so every judge is bootstrapped against nano-jev qp and qpa |

## Execution

| | |
|---|---|
| Code | `feat/exp9-decider` at `76da48c` |
| Jobs | 42761 (smoke, passed), 42765 (experiment 7 zero-shot, 1 h), 42766 (experiment 7 2B fine-tunes, 3 h 30), 42767 (experiment 7 E4B fine-tunes, 5 h 18), 42773 (experiment 8, three plans; cancelled after the fourth ran out of memory), 42784 (experiment 8 E4B fine-tunes, 2 h 36), 42769 (experiment 7 evaluation, GPU), 42785 (experiment 7 evaluation with the paired references, CPU). All exit 0 |
| Fixes | 42768: zero-shot experiment 8 plans had no `max_length` (`a7c390b`). 42773: the E4B fine-tune on experiment 8 ran out of memory at `micro_batch` 4 on the 46 GB L40S; 1 fits (`60ef8ca`). No result-changing fix |
| Time per fine-tune | Experiment 7: 2B 17–18 min, E4B 41–42 min. Experiment 8: 2B 11 min, E4B 21–22 min. Zero-shot scoring: nano-jev 4 ms, 2B 61 ms, E4B 141 ms a question |
| Checks | 12 + 12 `done.json` (experiment 7 + 8 fine-tunes) and 2 + 4 zero-shot; nano-jev's experiment 7 AIQs reproduce the committed ones exactly (qp 0.6731, qpa 0.6726), as do experiment 8's reference rows and hull (0.7006, 0.7133, 0.7010). A CPU rerun on Windows reproduces all four experiment 8 reports exactly and experiment 7's AIQs exactly (bootstrap bounds within 5e-5, from logistic regression floating point) |

## Results: experiment 7 (pair A, test, 7,405 questions)

| Judge | AIQ | With judge cost | GFLOPs a question | Minus nano-jev qp, 95% interval |
|---|---|---|---|---|
| E4B fine-tuned, qpa | **0.713** (seeds 0.709 ± 0.001) | 0.700 | 2,461 | [+0.035, +0.045] |
| 2B fine-tuned, qpa | 0.708 (seeds 0.705 ± 0.001) | **0.702** | 1,250 | [+0.030, +0.039] |
| E4B zero-shot, `correct` | 0.693 | 0.683 | 2,446 | [+0.016, +0.024] |
| E4B zero-shot, `grounded` | 0.692 | 0.682 | 2,486 | [+0.015, +0.024] |
| E4B fine-tuned, qp | 0.681 (seeds 0.678 ± 0.001) | 0.672 | 2,401 | [+0.004, +0.011] |
| 2B fine-tuned, qp | 0.680 (seeds 0.678 ± 0.001) | 0.676 | 1,215 | [+0.004, +0.011] |
| 2B zero-shot, `correct` | 0.675 | 0.671 | 1,238 | [−0.001, +0.006] |
| 2B zero-shot, `grounded` | 0.675 | 0.671 | 1,258 | [−0.002, +0.006] |
| nano-jev fine-tuned, qp (experiment 7) | 0.673 (seeds 0.671 ± 0.000) | 0.673 | 32 | |
| nano-jev fine-tuned, qpa | 0.673 (seeds 0.669 ± 0.004) | 0.673 | 33 | |
| nano-jev zero-shot, `correct` | 0.664 | 0.664 | 33 | |
| Entropy | 0.662 | | | |

The intervals compare AIQ without the judge's cost; the cost-included column is a point estimate. The clean subset
(6,405 questions) gives the same order and margins: 2B qpa 0.708, E4B qpa 0.714, nano-jev qp 0.675.

## Results: experiment 8 (five options, graded, test, 500 questions)

| Judge | qp AIQ | qpa AIQ | Judge units a question | qp minus hull | qpa minus nano-jev qpa |
|---|---|---|---|---|---|
| 2B zero-shot | 0.699 | **0.700** | 2.5–2.6 | [−0.003, +0.005] | [−0.020, −0.005] |
| E4B zero-shot | 0.698 | 0.697 | 4.9–5.1 | [−0.005, +0.001] | [−0.024, −0.010] |
| 2B fine-tuned | 0.693 | 0.691 | 7.4–7.8 | [−0.013, +0.002] | [−0.034, −0.006] |
| E4B fine-tuned | 0.685 | 0.683 | 14.7–15.4 | [−0.022, −0.004] | [−0.041, −0.017] |
| nano-jev (experiment 8) | 0.701 | **0.713** | 0.4 | [−0.001, +0.003] | |
| Fixed-policy hull | 0.701 | | | | |
| Oracle | 0.800 (0.816 at 30 units) | | | | |

Experiment 8 includes the judge's units in every question's cost (the 70B costs about 96). Without them, the
zero-shot deciders score 0.702–0.703. The fine-tuned ones vary by seed: seed 1 is far below the others for both
models (2B qp 0.677 against 0.701 and 0.703; E4B 0.684 against 0.700 and 0.703). At the operating points every
decider still picks (almost) always the 70B: accuracy 0.726–0.748 at 85–106 units.

## Hypotheses

- **H9a (the fine-tuned 2B beats nano-jev qp by ≥ 0.005 AIQ, interval above zero): met, narrowly, and only before
  its cost.** The qp judge gains +0.007 [+0.004, +0.011]; after its cost, +0.003. The qpa judge gains +0.035
  [+0.030, +0.039], and +0.029 after its cost, which is the larger and more useful result.
- **H9b (a decider beats the fixed-policy hull on experiment 8): not met.** No decider is above the hull; the
  fine-tuned E4B is below it.
- **H9c (zero-shot deciders beat entropy with their judge cost included): met on experiment 7.** 2B 0.671 and E4B
  0.683, against 0.662 (intervals without cost: [+0.008, +0.018] and [+0.026, +0.036]).

## Findings and cautions

1. **Reading the small model's answer is what a bigger judge adds.** nano-jev qp and qpa tie, but both deciders gain
   about 0.03 from the answer. A 33M cross-encoder seems unable to use it; a 2B model can.
2. **The decider is worth its cost on experiment 7, but the cost is not small.** The 2B costs about 1,250 GFLOPs a
   question: more than the 1.5B's own answer (1,051 on average) and 5.5% of always-large (22,762). It still keeps
   0.029 of its 0.035 gain.
3. **Experiment 8 has too little to learn from.** 3,000 training questions, of which only 42 and 22 have the two
   most expensive options as the cheapest correct one; dev NLL rises in epoch 2 and one seed is much worse. With the
   70B this far ahead of the cheaper options, no judge here finds the oracle's headroom.
4. **The fine-tuned judges pay three passes.** As with nano-jev, the score is the mean over 3 seeds and the cost is
   three forward passes. One seed would cost a third, with a little less accuracy (the per-seed AIQs are 0.002–0.004
   below the mean on experiment 7).
5. **Windows differ.** The deciders read up to 4,096 tokens; nano-jev read 512. For experiment 7's top-2 passages
   this rarely matters; for experiment 8's top 5 it may favour the deciders.
6. **Two inputs, one pair.** Experiment 7 is pair A only (1.5B / 7B); pairs B and C were not run.
7. **500 test questions** in experiment 8 give intervals of about ±0.04 on AIQ; the paired intervals are narrower.

## Reproducing

```bash
qsub -l walltime=00:30:00 -v STEP=exp9-smoke scripts/omega.pbs
qsub -l walltime=03:00:00 -v STEP=exp9-judge7,J7STEP=zeroshot scripts/omega.pbs
qsub -l walltime=12:00:00 -v "STEP=exp9-judge7,J7STEP=train,ONLY=--only qp-large-helps-strands-decider-2b-hobson-v21-A --only qpa-large-helps-strands-decider-2b-hobson-v21-A" scripts/omega.pbs
qsub -l walltime=24:00:00 -v "STEP=exp9-judge7,J7STEP=train,ONLY=--only qp-large-helps-strands-decider-e4b-gemma4-v1-2610-A --only qpa-large-helps-strands-decider-e4b-gemma4-v1-2610-A" scripts/omega.pbs
qsub -l walltime=16:00:00 -v STEP=exp9-judge8 scripts/omega.pbs
qsub -l walltime=01:00:00 -v STEP=exp9-judge7,J7STEP=eval scripts/omega-cpu.pbs
```

Results: [results/exp9/](../../results/exp9/).
