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
- MedAgentBench: no fine-tuning of the task models (300 tasks). Router training (the RightRoute track) is separate:
  a 1.5B LoRA router is trained on `calib` minus `calib_sweep` (90), checkpoints and thresholds are selected on
  `calib_sweep` (60), and `test` (150) is read once per trained router. All three groups are patient-disjoint;
  `python common/tools/medagentbench_patient_split.py --check` verifies this.
- Run seeds 0, 1, 2; report the mean and the range.

## Loss curves: stop at the elbow on val

Each run stops early on the **val** split, never on test, and shares its loss curve. Log one row per epoch in the
layout of `stair_common/training.py` (track, model, run, seed, epoch, train_loss, val_loss), then plot:

```python
from stair_common.training import from_records, validate_history, best_epoch, plot_history

rows = [{"epoch": 1, "train_loss": 0.71, "val_loss": 0.64}, {"epoch": 2, "train_loss": 0.52, "val_loss": 0.58}]
df = from_records(rows, track="selfLearnt", model="Qwen2.5-1.5B retriever", run="lora-r16", seed=0)
validate_history(df)
df.to_csv("history-seed0.csv", index=False)
print(best_epoch(df))                     # the early-stopping epoch of each run
plot_history(df, "loss-curves.png")       # needs the plot extra: stair-common[plot]
```

Or from the shell: `stair-plot-history history-*.csv --out loss-curves.png`. Each panel shows train and val loss, a
thin line per seed and their mean in bold, a dashed line at the best val epoch, and "val rises after" when the run
overfits past it. Training loss alone isn't a stopping signal: it keeps falling as the model overfits.

## Epochs: record first, reconcile later

The team agreed (17 Oct) not to force one epoch count across methods yet, but to record what each run used. Router
or judge training has its own schedule and is recorded here too.

| Track | Dataset | Model / router | Method | Train questions | Epochs | Selected by | Seeds |
|---|---|---|---|---|---|---|---|
| systemOne | HotpotQA | nano-jev judge (33M) | full fine-tune | 10,107: `train` minus nano-jev's 769 training questions, less the hold-out | 3; best at epoch 1 (the qa judge: 2) | lowest NLL on a 10% hold-out of train (1,124) | 0, 1, 2 |
| systemOne | HotpotQA | Strands Decider 2B / E4B judge | LoRA + head | 10,107 (same) | 2; best at epoch 1 | the same hold-out | 0, 1, 2 |
| RightRoute | MedAgentBench | Qwen2.5-1.5B router (LoRA, fresh) | GRPO(Reward), label-free | 90 (`calib` minus `calib_sweep`) | 40; best checkpoint kept | validation objective on `calib_sweep` (60) | 17 (single run; seed repeats are a declared follow-up) |
| | | | | | | | |

systemOne's common redo (18 Oct): 9 of its 10 judges have their lowest val loss at epoch 1, and val loss rises after
it while train loss keeps falling, so the later epochs only cost GPU time. For judge-sized fine-tunes, prefer 1–2
epochs with early stopping. Its `val_loss` is the hold-out of train above; the team `val` split picks the routing
threshold.
