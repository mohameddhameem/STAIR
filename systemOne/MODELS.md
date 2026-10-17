# Fine-tuned judges and the Hugging Face plan

**Status: not published yet.** The experiments saved each judge's scores and training history (`done.json`), not
its weights. Publishing means retraining the chosen judges with the weights saved. The training is seeded, so a
retrain reproduces the reported runs up to GPU nondeterminism.

## What to publish

| Candidate | Base model (licence) | Task | Result | Why |
|---|---|---|---|---|
| `stair-judge-nanojev-qp-A` | `sdmlai/nano-jev` v1.0, 33M (Apache-2.0) | Before any answer: will Qwen2.5-7B help over 1.5B? | AIQ 0.673, +0.004 to +0.011 over logistic regression | The report's main judge; 32 GFLOPs a question |
| `stair-judge-decider2b-qpa-A` | `StrandsAgents/strands-decider-2B-hobson-v21` (Apache-2.0), LoRA + head | After the 1.5B's answer: escalate to 7B? | AIQ 0.708, 0.702 with its cost | Best judge after cost |
| `stair-judge-decidere4b-qpa-A` | `StrandsAgents/strands-decider-E4B-gemma4-v1-2610` (Apache-2.0), LoRA + head | Same | AIQ 0.713 | Most accurate judge |
| optional: `stair-judge-nanojev-5opt-graded` | nano-jev | Pick one of 5 answer options (AgentTTS setting) | ties the fixed-policy hull | Only as a reference |

Training data: HotpotQA (CC BY-SA 4.0) questions and passages, with labels derived from our own expert runs. Each
candidate ships as 3 seeds (the reported score is their mean), or seed 0 alone with that stated.

## Retraining with weights saved (OMEGA, one L40S)

The experiment 7 trainer already has `--save-model`; it writes `model/` (weights + tokenizer, `save_pretrained`)
next to each seed's `done.json`, under `runs/system-one/judges/<judge>/seed<k>/`.

```bash
# nano-jev qp, pair A (3 seeds, minutes each)
uv run judge train --plan configs/system-one.toml --only qp-large-helps-nanojev-A --save-model
# decider 2B qpa (~18 min a seed) and E4B qpa (~42 min a seed)
uv run judge train --plan configs/system-one-decider.toml --only qpa-large-helps-strands-decider-2b-hobson-v21-A --save-model
uv run judge train --plan configs/system-one-decider.toml --only qpa-large-helps-strands-decider-e4b-gemma4-v1-2610-A --save-model
```

On OMEGA: `qsub -l walltime=03:00:00 -v "STEP=exp9-judge7,J7STEP=train,ONLY=--only <judge>" scripts/omega.pbs` after
adding `--save-model` to that step. A finished seed is skipped, so move the old `done.json` aside first.

**To do before publishing:**

- [ ] **Decider save:** `judge.run_train` calls `judge.model.save_pretrained(...)`. Check that `StrandsDeciderModel`
  supports it; otherwise save only the trainable LoRA + head tensors (`fine_tune` already snapshots them) and record
  the base checkpoint and the `strands-decider` commit (`6039cf8b`).
- [ ] **Experiment 8 judge:** `exp8-judge` has no `--save-model` yet; add it if that judge is published.
- [ ] **Check the retrain:** score the saved model on test and confirm the AIQ matches `results/` within the seed spread.
- [ ] **Usage snippet:** a few lines that load the judge and score one item (`judge.route_items` builds the input).

## Model card draft

```markdown
# stair-judge-nanojev-qp-A

A routing judge for multi-hop question answering. Given a HotpotQA question and its top-2 retrieved passages, it
outputs P(the large model helps), i.e. the small model (Qwen2.5-1.5B on the top 2 passages) is wrong (F1 < 0.8)
and the large model (Qwen2.5-7B on all 10) is right. Escalate when the probability exceeds a threshold chosen on
calibration data.

- Base model: sdmlai/nano-jev v1.0 (33M), fine-tuned with grouped cross-entropy, 3 epochs, lr 5e-5, 3 seeds.
- Training data: 12,000 HotpotQA train questions (router-train split), labels from our expert runs; 10% dev.
- Evaluation: 7,405 HotpotQA dev questions. AIQ 0.673 (always-small F1 61.3, always-large 66.9, oracle AIQ 0.755);
  judge cost 32 GFLOPs a question, 3% of the small model.
- Limits: trained for one expert pair; other pairs or retrievers need recalibration or retraining. English only.
  The labels depend on the F1 threshold τ = 0.8.
- Licence: Apache-2.0 (base model); training data CC BY-SA 4.0.
```
