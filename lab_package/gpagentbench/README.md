# GPAgentBench-2K — third dataset (decided 31 Oct 2026, Jonathan)

Source: Hugging Face `Frank-xd/GPAgentBench-2K` (gated; CC BY-NC 4.0; 12.4 MB; paper arXiv 2608.30188, EMNLP 2026).
The authors' GitHub repo (jianing-lab/GPAgentBench) is an empty placeholder as of 30 Aug 2026 — no harness,
simulator or judge prompts released. The data files are NOT committed here (gated licence): download with your
own HF account into `/mnt/hdd/jonathan/data/gpagentbench/` (lab copy exists).

## Why this set (screen of 30 Oct; see handoff / Project note)
Same recipe as MedAgentBench — weak-first cascade, GRPO(Reward) fresh 1.5B adapter, risk score τ frozen before
training, three options W / S / H on the high-risk tier — on a set that has what MedAgentBench lacked:
* a labelled risk gradient to VALIDATE τ on all 2,428 cases: referral urgency (none 1,422 / routine 135 / urgent 785 /
  emergency 86), destination (GP 1,422 / specialty 526 / Emergency Department 278), real disposition (Admitted 758,
  ICU 17, died 18, Transferred 46; 385 not reported);
* a weak model that is HARMFUL on the high-stakes subset, not just wrong: Qwen2.5-7B manages locally 89.6 % of the
  1,006 referral-needed cases (72B: 54.7 %) — accepted wrong actions exist (e > 0);
* a native human hand-off: `refer` is in the action space; missed referral / unnecessary referral = our needed /
  unnecessary pair;
* size (train 1,943 / test 485) and a published Qwen2.5 7B / 32B / 72B baseline.

## Profile (gpab_profile.py, lab run 31 Oct 00:04)
2,428 cases, case_id unique, 0 exact duplicate patient blocks, 40 repeated generic chief complaints, 11 shared
train/test. Fields: patient (29 keys, ~150 words), persona (personality / language A-B-C / recall / disorientation),
evidence (vital_signs, physical_exam, exactly 9 test_results, body_exam_targets, test_targets; ~290 words),
ground_truth (diagnosis — 1,077 distinct; treatment; disposition; referral {needed, urgency, destination}).
Labels: diagnose_and_manage 1,422 / abstain_and_refer 1,006; categories manage_in_gp 1,365 · manage_after_tests 57 ·
refer 940 · urgent_refer 66. Test split by category × urgency: manage_in_gp 273, manage_after_tests 11, refer
emergency 11 / routine 37 / urgent 140, urgent_refer emergency 4 / urgent 9.
Provenance note: treatments contain many TCM products — a translated Chinese primary-care network (unnamed in
the paper). Management / referral endpoints are deterministic; diagnosis / treatment are LLM-judged in the paper.

## Structure for routing
Two stages per episode: (1) evidence gathering — ask / body_exam / test turns with a simulated patient, cap 10 turns;
(2) terminal decision — refer (ends) or diagnose → treat. Ground truth is per INSTANCE (terminal decision); stage 1
has no label of its own — it is rewarded through the terminal outcome and its own cost (turns + tests), as the
MedAgentBench round-1 decision was. Risk τ is scored from the presentation only (never the ground truth) and
validated against urgency / destination / disposition.

## Gate before registration (the MedAgentBench lesson): 60-episode weak-first pilot
Local patient simulator and judge (72B) in place of GPT-4o-mini / GPT-5.4 (prompts not released → ours, stated).
Stop rules declared in advance: abandon if (a) the local setup does not reproduce the authors' Qwen rows within
~5 points (7B Dx 40.4 / 72B 54.1; missed referral 89.6 / 54.7), (b) weak/strong churn on repeats is beyond what
replay training tolerates, (c) a τ rubric scored from the presentation does not separate urgency ≥ urgent
(AUC < 0.75). Measure accuracy, e (missed-referral rate), τ spread and churn. Their 485-case test split is the
comparability set and is not opened until the single declared read.

Files: `gpab_profile.py` (read-only profiler; prints the counts above; `--csv` writes one row per case — that CSV
is derived from gated data and is NOT committed).
