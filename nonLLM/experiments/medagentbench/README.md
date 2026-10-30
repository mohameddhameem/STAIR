# MedAgentBench: non-LLM replay aligned with Jonathan

The completed local run is `../runs/nonllm_medagentbench_replay_v1` relative to
the repository. It pins Jonathan's branch at
`3ac054d1d72994cab5e8927c6cdba1449371aba4`. The source files are extracted into
the run's `reference/` tree, preserving this working branch and all local changes.
No raw patient views, dataset records, or model weights are copied into this folder.

Training1 andTraining3 each supply300 replay rows: task view, weak/strong
pass/fail, and measured costs. The task executors are Qwen2.5-7B and72B-AWQ,
not the HotpotQA retrieval/answering model ladder. No task models are trained or
called; no live FHIR container is required. Replay reproduces the committed
outcomes, not a fresh live-environment execution.

The committed patient split is used unchanged: train90, validation60, test150,
with30/19/49 patients and zero patient overlap across every boundary. Official
test_data_v2.json matches the team's manifest hash. The original patient split
generator was not preserved: frozen files/hashes, rather than regenerating a
new split with the same seed, define this comparison.

Three non-LLM architectures: linear softmax, MLP64 and MLP128→64. Features are
train-fitted TF-IDF and deterministic statistics of the exact view available to
the LLM router; no outcome/cost/risk/strong-output fields enter model features.
Fresh GRPO A1 starts,40epochs, lr1e-5, accumulation8, clip1, G8, entropy.03,
advantage=reward−group mean. No standard-deviation normalization, KL penalty,
PPO clipping or SFT warm start. This differs from the older HotpotQA GRPO
adaptation; it matches the latest MedAgentBench trainer's update. Same feature
contents and training mechanics do not make a small classifier identical to a
pretrained1.5B LLM: parameterization and available pretrained knowledge differ.

Training1 decides after the completed weak attempt. Training3 decides at round2,
then either finishes the weak episode (pay full cw) or escalates early (cw2+cs).
Reward is pass/fail minus.1×C/cs minus max(0,C/cs−1). The best checkpoint is
selected by rounded greedy validation reward, every10 optimizer steps, exactly
as the supplied trainer. Seed17 is the source-matching primary run;0/1/2 are
independent sensitivity repeats, without ensemble or test-based seed selection.

Fixed thresholds.50/.20 and a validation-reward-selected threshold are locked
before one test inference per model. All alternate threshold and risk-gate
readouts reuse the stored probabilities. Cost denominator is the full weak-first
always-escalate sum(cw+cs) on the same150 tasks, for both settings. It includes
sunk weak calls; it is not the direct-strong costcs or HotpotQA Table5 pricing.
Classifier CPU work and the frozen LLM risk-scorer's cost are excluded from the
proxy, consistently with the reference board. Feature+batch-inference latency is
recorded separately.

Key limitations: reference test results were already public; this is paired
replay, not a newly secret test. Substantial seed sensitivity prevents a stable
reproduction claim. Same tau>=.8 guardrail must be used when comparing risk
policies: ungated savings can result from delivering high-risk failures. Patient
bootstrap intervals respect clustering. The newer three-action Training5 human
handoff adds separate rewards and human-cost assumptions and is outside this
two-action campaign.

Validation commands (from repository root):

```bash
../.cascade-env/bin/python nonLLM/experiments/medagentbench/test_replay.py
../.cascade-env/bin/python nonLLM/experiments/medagentbench/audit.py --out ../runs/nonllm_medagentbench_replay_v1
```

For a new output directory, `fetch.py --out <new-run>` downloads the pinned
source artifacts, and `replay.py --out <new-run> --phase prepare` verifies the
split/reward equations and fits train-only features. Run `--phase train` with
each `--setting 1/3`, `--architecture linear/mlp64/mlp128_64`, and
`--seed 17/0/1/2`. After all24 fits finish, run `--phase lock` then
`--phase evaluate`, in that order. Never replace the committed patient split
with a freshly generated split, and never select a seed or architecture on test.

`replay.py --phase evaluate` intentionally refuses a second test inference.
Use `audit.py` for post-read integrity checks and the already-stored decisions.
