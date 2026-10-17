# Fine-tuned judges and the Hugging Face plan

**Status: weights saved, not published yet.** The judges were retrained with `--save-model` on OMEGA (2026-10-17,
jobs 42789–42791). That gives 12 models: 4 judges × 3 seeds, 2.4 GB. They are kept on the cluster under
`runs/judge-models/judges/<judge>/seed<k>/model/`, next to each seed's `done.json` and scores.

## Check of the retrained weights

The saved models were re-evaluated on pair A's test set (7,405 questions; job 42797). They match the reported results.
GPU nondeterminism makes the per-seed scores differ slightly, but the AIQ is unchanged within the seed spread.

| Judge | Reported AIQ | Retrained AIQ (mean of seeds) | With judge cost | Judge cost a question |
|---|---|---|---|---|
| nano-jev qp | 0.673 | 0.673 | 0.673 | 32 GFLOPs |
| nano-jev qpa | — | 0.675 | 0.675 | 33 GFLOPs |
| decider 2B qpa | 0.708 | 0.709 | 0.702 | 1,250 GFLOPs |
| decider E4B qpa | 0.713 | 0.712 | 0.699 | 2,461 GFLOPs |

## What to publish

| Candidate | Base model (licence) | Task | Result | Why |
|---|---|---|---|---|
| `stair-judge-nanojev-qp-A` | `sdmlai/nano-jev` v1.0, 33M (Apache-2.0) | Before any answer: will Qwen2.5-7B help over 1.5B? | AIQ 0.673, +0.004 to +0.011 over logistic regression | The report's main judge; 32 GFLOPs a question |
| `stair-judge-decider2b-qpa-A` | `StrandsAgents/strands-decider-2B-hobson-v21` (Apache-2.0), LoRA + head | After the 1.5B's answer: escalate to 7B? | AIQ 0.708, 0.702 with its cost | Best judge after cost |
| `stair-judge-decidere4b-qpa-A` | `StrandsAgents/strands-decider-E4B-gemma4-v1-2610` (Apache-2.0), LoRA + head | Same | AIQ 0.713 | Most accurate judge |
| optional: `stair-judge-nanojev-5opt-graded` | nano-jev | Pick one of 5 answer options (AgentTTS setting) | ties the fixed-policy hull | Only as a reference; not saved yet |

Training data: HotpotQA (CC BY-SA 4.0) questions and passages, with labels derived from our own expert runs. Each
candidate ships as 3 seeds (the reported score is their mean), or seed 0 alone with that stated.

## Retraining with weights saved (OMEGA, one L40S)

`judge train … --save-model` writes `model/` next to each seed's `done.json`.
- **nano-jev:** `save_pretrained` writes the weights and the tokenizer.
- **Decider:** writes its config, `slot_head.pt`, `lora/` and the tokenizer. `StrandsDeciderModel.load` reads them
  back.

```bash
# on OMEGA: 3 seeds each; walltimes used for the 2026-10-17 run
qsub -l walltime=03:00:00 -v "STEP=save-judges,PLAN=configs/system-one.toml,ONLY=--only qp-large-helps-nanojev-A --only qpa-large-helps-nanojev-A" scripts/omega.pbs
qsub -l walltime=03:00:00 -v "STEP=save-judges,PLAN=configs/system-one-decider.toml,ONLY=--only qpa-large-helps-strands-decider-2b-hobson-v21-A" scripts/omega.pbs
qsub -l walltime=05:00:00 -v "STEP=save-judges,PLAN=configs/system-one-decider.toml,ONLY=--only qpa-large-helps-strands-decider-e4b-gemma4-v1-2610-A" scripts/omega.pbs
# then evaluate the saved judges (CPU job)
uv run judge eval --plan configs/system-one-decider.toml --out runs/judge-models --exclude data/hotpotqa/nanojev_validation_ids.txt
```

**To do before publishing:**

- [x] Decider save and load (`StrandsDeciderModel.save_pretrained` / `load`).
- [x] Check the retrain: AIQ matches (table above).
- [ ] **Experiment 8 judge:** `exp8-judge` has no `--save-model` yet; add it if that judge is published.
- [ ] **Usage snippet:** a few lines that load the judge and score one item (`judge.route_items` builds the input).
- [ ] Copy the weights off the cluster and upload them (`huggingface-cli upload`) with the model card below.

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
