# MedAgentBench routing campaign — shared data and metrics (27 Oct 2026)

Files behind the held-out test results board (deck V14-137, slide 117; write-up
`lab_package/medagentbench/MedAgentBench_Results_Writeup_27Oct2026.md`). Shared for the
System One judges comparison by replay. Nothing here is a model or code: tables and scores only.

| File | sha256 (first 8) | Contents |
|---|---|---|
| training1_table.json | 186545f0 | Replay table, 300 rows (one per task). |
| training3_table.json | 4eff23f3 | Same, with the round-2 view and round-2 weak cost `cw2` (Training 3). |
| tau_scores_v2.json | 2d971783 | Per-task risk score τ (rubric v2, scored by the local 72B, frozen; hand-validated at tier level). |
| training1_test_scores.jsonl | c0155f50 | Training 1 router: `pDeliver` per held-out test task (150). |
| training3_test_scores.jsonl | 252abc19 | Training 3 (round-2) router: `pDeliver` per held-out test task (150). |

## Table columns
- `tid` task id (`task<type>_<n>`), `idx` 0-based index in `test_data_v2.json`, `split` ∈ {train, val, test}
- `view` what the router reads: task instruction (+ context) followed by a summary of the weak model's
  attempt (finished / not finished, rounds, final answer or last action capped at 400 chars).
  Training 3's view is the attempt cut at round 2.
- `qw`, `qs` pass/fail correctness of the weak (Qwen2.5-7B-Instruct) and strong (Qwen2.5-72B-Instruct-AWQ)
  episodes under the benchmark's own grader; `weak_rounds` rounds the weak episode used (cap 8)
- `cw`, `cs` measured cost of the weak / strong episode in normalized FLOPs units
  (`stair_common.cost.units`, served model's own tokenizer, cumulative input per round);
  `cw2` (Training 3 only) weak cost up to the round-2 cut
- Weak-first accounting: deliver pays `cw`; escalate pays `cw + cs` (Training 3: `cw2 + cs`).
  Always-strong on the same instance set = Σ(cw + cs).

## Splits (committed in common/stair_common/splits/medagentbench/, patient-disjoint at every boundary)
train = calib minus calib_sweep (90) · validation = calib_sweep (60) · held-out test = test (150).
Verify with `python common/tools/medagentbench_patient_split.py --check`.

## Operating points
Decisions follow from `pDeliver >= theta`: Training 1 primary theta = 0.50 (61.33 % at 54.1 % of
always-strong cost), cost-lean theta = 0.20 (60.00 % at 51.2 %); Training 3 theta = 0.50 (63.33 % at 81.5 %).
These scores are for paired comparison after a judge's own single test read; thresholds for any new
method are chosen on calib_sweep first. Intervals: resample by patient (49 patients on test), not by task.

## Data provenance
MedAgentBench patient profiles are de-identified real records publicly released by the authors
(Stanford STARR de-identified warehouse; patient-level date jittering; generated MRNs, names and contacts;
arXiv 2501.14654 §2.2; data and FHIR image on GitHub and Docker Hub). `view` text derives from that data.

## weak_exposure_high_tier.json (28 Oct 2026)
Weak wrong-action rate on the 60 high-risk (τ ≥ 0.8) instances, from a read-only replay of the canonical
weak store (sha256 db008f80…) with lab_package/medagentbench/weak_exposure_measure.py. Categories per
instance: correct / no_action / rejected_post / other_post / wrong_order ("accepted POST" per refsol.extract_posts).
Result: e_accepted = 0 on all 60 (task 5: no POST ever attempted, 30/30; task 9: every POST rejected by the
harness, 30/30, 189 attempts); e_attempted = 0 (task 5) / 1.0 (task 9). See the pre-registration entry of 28 Oct.

