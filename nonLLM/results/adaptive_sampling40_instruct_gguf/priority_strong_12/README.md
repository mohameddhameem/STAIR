# Twelve-question matched reference batch

Completed on 2026-10-16. Twelve training questions were selected with seed
20261016 among questions with complete Weak results, without using correctness.
Strong execution produced 960 QA candidates and 960 prefix-fusion replies. All
four pairs then received 1,920 matched extraction records per evaluation version.
No sampling or fusion was repeated for evaluation v2. The full run is paused
for reference review; all generation caches remain available for resumption.

The first letter is the retrieval model, the second the answering/fusion model.
W is Qwen2.5-1.5B-Instruct GGUF, S is Qwen2.5-7B-Instruct GGUF, both Q4_K_M.

## Evaluation v2 reference results

Strict normalized EM hits out of twelve questions, after gold-independent answer
extraction. This is a small training diagnostic, not held-out accuracy.

| Pair | k=1 | k=5 | k=10 | k=20 | k=40 | Best fixed k on these twelve | Retrospective budget oracle |
|---|---:|---:|---:|---:|---:|---:|---:|
| WW | 2 | 2 | 4 | 4 | 4 | 5 | 5 |
| WS | 5 | 3 | 2 | 2 | 3 | 5 | 8 |
| SW | 2 | 2 | 3 | 2 | 2 | 4 | 5 |
| SS | 4 | 4 | 5 | 5 | 3 | 6 | 7 |

The best fixed k and oracle above both use the same twelve training questions
retrospectively; neither is a validation-selected AgentTTS baseline. The oracle
uses gold answers and is not a deployable policy or a cost-constrained result.
WS has three questions correct at k=1 but wrong at k=40, and one with the reverse
pattern under this scorer. SS has one k=1-correct/k=40-wrong question.

The individual outputs also show both patterns. For question 7188 about Leeds
United's formation year, WS answers 1919 at both k=1 and k=40. For question 6405,
WS answers the gold nationality American at k=1, but Canadian at k=40.
These examples motivate further checks; they do not establish a general gain.

## Output and scoring checks

All 960 Strong QA candidates are nonempty, with no replacement characters,
input truncation or output-limit stops; all 24 batches have forty distinct seeds.
Strong fusions are nonempty and have no input truncation. Fourteen of 960 fusions
hit the public 512-token output limit (1.46%).

The original deterministic extractor sometimes kept a whole answer clause.
Version 2 applies the existing grounded reader to every reply over six words,
using the question and model reply only. The rule was revised on training data,
and should be frozen before formal test evaluation. Version 1 remains in
report.json; evaluation_v2/report.json is separate and uses a new cache fingerprint.
Main inference/extraction caches were not changed.

Strict EM still rejects some equivalent names or forms, such as Australia versus
the supplied Australian, and Max Mosley versus Max Rufus Mosley. Grounded-reader
abstentions and extraction errors remain possible. These counts are therefore
pipeline scores, not a manually verified semantic-correctness assessment.

The initial summary failed because unused shard 3 had no database file. Opening
only represented shards fixed it without rerunning inference. final_validation.json
verifies all original cache counts were preserved, both evaluation versions have
1,920 matched records, their fingerprints differ, and all GPU compute stopped.

comparison_v2.csv contains all forty budgets and EM/F1/coverage for each pair.
Both JSON reports retain representative full replies and extracted answers.
