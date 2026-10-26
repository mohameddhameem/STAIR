# MedAgentBench routing campaign — lab scripts and registration (26 Oct 2026)

Jonathan's stage-level-routing experiments on MedAgentBench (weak-first cascade:
Qwen2.5-7B attempt -> 1.5B GRPO router decides deliver/escalate -> Qwen2.5-72B-AWQ).
All training is execution-free (replayed from recorded episodes); the
pre-registration file in this directory is the authoritative record of every
declared decision, prediction, and result.

- training1_build_table.py / training1_trainer.py — Training 1 (risk-free; risk
  tested as a guardrail at readout). Held-out test: 61.33% acc at 54.1% of
  always-escalate cost (test read once, both tallies pre-declared).
- tau_scoring.py — the Q1/Q2/Q3 risk rubric (v2), scored by the local 72B.
- training1_test_read.py — the single Training 1 test read (refuses re-runs).
- training1_calibration_latency.py — the registered calibration audit + decision
  latency (clones of it scored Trainings 2 and 3).
- training2_trainer.py — Training 2 (risk in the reward): reproduced the
  guardrail at higher cost. Verdict: risk belongs at the decision layer.
- training3_build_table.py / training3_trainer.py / training3_test_read.py —
  Training 3 (the round-2 router / mid-episode escalation, replay-trained).
- midepisode_headroom.py — Step A headroom analysis for the round-2 chapter.

Run-time stores (tables, adapters, scores, episode outputs) live on the lab
under /mnt/hdd/jonathan/runs/mab and are NOT committed.
