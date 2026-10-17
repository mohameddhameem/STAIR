# exp8-rescore: v4 validation (PASSED)

Job 42694 (`exp8-rescore-validate`, 2 GPUs, exit 0, 8 min 35 s), code at `c181713`. Method: the answering model
commits a short final answer, then Qwen/Qwen2.5-14B-Instruct (revision `cf98f3b3`) grades it.

Job 42693 (1 GPU) failed first with `ValueError: Some modules are dispatched on the CPU or the disk`: the 70B in
4-bit does not fit one L40S (the same limit as experiment 8 phase 1). Resubmitted with
`-l select=1:ncpus=8:ngpus=2:mem=96gb`; no code was changed.

| Field | Result | Required |
|---|---|---|
| `agree_em1_correct` (n = 269) | 0.996 | >= 0.95 |
| `agree_f10_incorrect` (n = 109) | 0.917 | >= 0.80 |
| `written_agreement` | 0.994 | >= 0.95 |
| `verdict_mass_mean` | 0.998 | >= 0.90 |
| commit `grounded_when_correct`: 3B x1 / 3B x40 / 70B | 0.978 / 1.000 / 0.993 | >= 0.90 |
| commit `nonempty`: all three cells | 1.000 | >= 0.98 |
| `passed` | **true** | |

Not gating (200 calib answers per cell): graded correct 0.460 (3B x1), 0.585 (3B x40), 0.735 (70B), against
lenient-of-response 0.58, 0.76, 0.695. Commit EM 0.305, 0.245, 0.565. Mean commit cost 0.13, 0.16, 2.4 units.
The disagreements are aliases and formats the grader accepts correctly ("UDC", "writers", "animated"), plus
two grader errors ("no" against a long gold, an identical "Benedict of Nursia" at P = 0.38).
