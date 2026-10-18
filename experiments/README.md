# Team experiment sheet

Use one entry for each experiment you want run on OMEGA. It can be a new experiment, or an old one to **rerun on
the common standard** (STAIR `common/`: team splits, lenient accuracy, AgentTTS cost units).

- Each entry is its own file, `<id>.md`, filled in by your Claude with [CLAUDE-INSTRUCTIONS.md](CLAUDE-INSTRUCTIONS.md).
- [example-exp8.md](example-exp8.md) is a filled-in entry.
- Commit them with your code on **your own branch** of STAIR, as `experiments/<id>.md`, and push the branch.
  Dhameem fetches your branch, checks the entries, runs them on OMEGA and sends you the results CSVs. Nobody else
  pushes to your branch.

## Template (copy it exactly: keep the keys, fill in the values)

```yaml
id: XX-01                 # your initials + a number
owner: <first name>       # no email, no student ID
track: <systemOne | systemTwo | kgqa | medagent | ...>
status: new               # new | rerun
title: <one line>
question: <the research question this answers>
hypothesis: <what you expect and why>

code:
  repo: <https URL, pushed and reachable>
  branch: <branch>
  commit: <full 40-char SHA, pushed>
  setup: <e.g. uv sync --extra llm>
  commands:               # in order; each must run non-interactively from the repo root
    - <command>

data:
  dataset: <hotpotqa | 2wikimultihopqa | cwq | webqsp | medagentbench>
  splits:                 # names from stair_common.splits; the sizes used
    train: <n or none>
    val: <n or none>
    test: <n>
  extra: <other data (source URL + licence) or none>

models:
  - id: <HF model ID>
    params_b: <billions>
    quant: <none | 4bit | 8bit>
    role: <small | large | judge | retriever | ...>

compute:
  gpus: <1 | 2>           # 2 is the per-user limit; a 70B model in 4-bit needs 2
  mem_gb: <n>
  walltime_h: <estimate>  # every job needs one
  epochs: <max epochs>    # training only
  early_stopping: best val loss epoch   # or say what else and why

outputs:
  results_csv: <path; stair_common.schema columns, one row per question/policy/seed>
  history_csv: <path; stair_common.training layout, or none>
  other: <none, or a list>

metrics:
  headline: lenient       # the team standard
  extra: [em, f1]         # add aiq / llm_grade if used
  cost_unit: agenttts

reproduce:                # status: rerun only; otherwise none
  old_results: <the numbers being reproduced>
  source: <the file/commit/report they came from>
  differences: <what changes under the common standard>

adopt:                    # code worth reusing team-wide, or none
  - path: <file in your branch>
    why: <one line>

open_questions: <anything Claude could not find out; the ONLY field that may say TBD>
```

After the YAML block, add at most 10 lines of prose with anything a reviewer needs (why this config, known pitfalls).

