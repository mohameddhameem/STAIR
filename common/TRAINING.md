# Training recipe (retrievers and answerers)

The proposal's recipe, from the selfLearnt Colab runs. Train only on a dataset's `train` (and `train_extra`) IDs;
select checkpoints and thresholds on `val`; touch `test` once.

| Model size | Method | Training questions × passes |
|---|---|---|
| ≤ 1B | LoRA | 12,000 × 2 |
| 1.5–14B | LoRA | 12,000 × 1 |
| 24–72B | QLoRA (4-bit) | 2,000 × 1 |

- LoRA rank 16, alpha 32, learning rate 1e-4. Keep the checkpoint with the best validation loss.
- HotpotQA: the 12,000 are `stair_common/splits/hotpotqa/train.txt`, drawn from the official train file with the team's seed; a
  2,000-question run takes its first 2,000 lines.
- CWQ / WebQSP: only 350 training questions carry gold triplets (`stair_common/splits/<dataset>/train.txt`).
- MedAgentBench: no fine-tuning (300 tasks).
- Run seeds 0, 1, 2; report the mean and the range.

## Epochs: record first, reconcile later

The team agreed (17 Oct) not to force one epoch count across methods yet, but to record what each run used. Router
or judge training has its own schedule and is recorded here too.

| Track | Dataset | Model / router | Method | Train questions | Epochs | Selected by | Seeds |
|---|---|---|---|---|---|---|---|
| systemOne | HotpotQA | nano-jev judge (33M) | full fine-tune | 12,000 (HotpotQA train, before this standard) | 3, best dev NLL | 10% dev | 0, 1, 2 |
| systemOne | HotpotQA | Strands Decider 2B / E4B judge | LoRA + head | 12,000 (same) | 2, best dev NLL | 10% dev | 0, 1, 2 |
| | | | | | | | |
