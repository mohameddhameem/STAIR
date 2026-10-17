# exp8-rescore: grader v3 validation (FAILED: the extraction step)

Job 42689 (`exp8-rescore-validate`, `-l walltime=01:00:00`, exit 0, 10 min 20 s), Qwen/Qwen2.5-14B-Instruct at
revision `cf98f3b3`, code at `d8a2557`. Method: extract the short answer, then grade it. Gate slice: dev items
1000-1499; dumps from the calib pools.

| Field | Result | Required | |
|---|---|---|---|
| `extract_faithful` | **0.758** | >= 0.90 | **fail** |
| `agree_em1_correct` (n = 269) | **0.844** | >= 0.95 | **fail** |
| `agree_f10_incorrect` (n = 109) | 0.908 | >= 0.80 | pass |
| `dump_graded_correct` (n = 1,000) | **0.414** | <= 0.10 | **fail** |
| `written_agreement` | 0.994 | >= 0.95 | pass |
| `verdict_mass_mean` | 0.999 | >= 0.90 | pass |

Not gating: `known_extracted_none` 0.238, `dump_extracted_none` 0.522. Diagnostic slice (dev 500-999, test-pool
dumps): extract_faithful 0.748, agree_em1 0.794, dumps 0.454.

`passed` is false, so `exp8-rescore` and `exp8-judge-graded` were not run. Prompts, thresholds and code unchanged.

**Why.** The extractor returns "none" for about a quarter of already-short, correct answers (for example "Man Booker
Prize for Fiction", "David Dobkin", "Shut Up", "no"), and "none" is graded wrong. Dumps that open with a sentence
stating the gold still grade CORRECT 41% of the time.

| Version | Short answers | Pasted-passage control |
|---|---|---|
| v1 (`../README.md`) | identical answers graded wrong | 0.662 |
| v2 (`../v2/README.md`) | good (0.996 / 0.917) | 0.619 (fail) |
| v3 (this folder) | extractor drops 24% of correct short answers | 0.414 (fail) |

## Diagnosis and version 4 (project owner's local session, 2026-10-15)

**v3's two failures pull in opposite directions,** so a prompt tweak would trade one for the other:
- it extracted "none" for 23.8% of short answers that were already bare and correct;
- when it extracted anything from a dump (47.8% of dumps), 87% of those extractions were the gold. The 14B
  reads the passages.

**The project owner chose self-commit (version 4).**
- **Commit.** Each cached response's exact final conversation (the answering prompt, or the fuser prompt
  with its samples) is replayed with the response. The cue "So, in as few words as possible, the final
  answer is:" follows, and *the model that wrote it* continues greedily. The commit's cost (the cue plus
  the generated tokens) is added to each option.
- **Grade.** The committed short answer is graded with version 2's grader prompt, which passed on short
  answers, and scored with official EM / F1.
- **The pasted-passage control is dropped:** a dump is not a conversation that can be continued.
- **The new gate** keeps version 2's grader thresholds, and adds two checks on 200 calib answers of each of
  the 3B, the fused-40 and the 70B:
  - of the commits graded CORRECT, ≥ 90% must be grounded in the response (≥ 60% of the commit's tokens
    appear in it; yes/no exempt);
  - ≥ 98% of commits must be non-empty.
  - These thresholds were fixed before any version 4 run.

**Local check of the commit step,** with Qwen2.5-1.5B standing in for Llama, which is not cached locally:
- it runs on real cached answers, including 6k-token fused conversations, at 0.14–0.22 units per commit;
- honest abstentions come out as paraphrases ("I do not have enough information…"). That is why grounding
  is gated only among the correct commits.
