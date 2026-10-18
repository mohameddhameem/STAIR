# Instructions for your Claude

Open Claude on your own branch of STAIR; Claude Code is best. First merge STAIR `main` into your branch, so that
`experiments/` is there. Then paste everything between the two lines below.

---

You are helping me fill in the team experiment sheet. Each experiment I want run on our GPU cluster becomes one file,
`experiments/<id>.md`. It follows the YAML template in `experiments/README.md` **exactly** (same keys, same order),
plus at most 10 lines of prose. `experiments/example-exp8.md` is a correctly filled-in entry.

Steps:
1. Ask me which experiments to include: new ones, and old ones to rerun on the common standard. Don't ask me
   anything else yet.
2. For each experiment, look for the answers yourself first:
   - my README, configs, scripts, notebooks, `git log` and `git remote -v`;
   - the STAIR `common/` docs: `stair_common.splits`, `schema`, `score`, `cost` and `training`;
   - for a rerun, the old numbers and where they came from.
3. Then ask me, in one list, only what you could not find.
4. Write each file, then check it:
   - `code.commit` is a full SHA that is on the pushed remote branch (`git branch -r --contains <sha>`). If it
     isn't, tell me to push first.
   - Every command in `commands`:
     - runs non-interactively from the repo root;
     - uses no local paths (`C:\...`, `/Users/...`) and no notebook-only steps;
     - has no hard-coded secrets. API keys and tokens are read from environment variables, and you list their
       names in `open_questions`.
   - The splits are named as in `stair_common.splits`, and the test sets are the team's standard ones.
   - The outputs are written in the `stair_common.schema` layout, and training curves in the `stair_common.training`
     layout. If my code doesn't produce that yet, say so in `open_questions`.
   - Cost follows `stair_common.cost`: several samples from one prompt pay that prompt once (AgentTTS Eq. 8). If my
     code counts the prompt once per sample, say so in `open_questions`.
   - `compute.walltime_h` is filled in, and `gpus` is at most 2.
   - Training stops early on validation loss (keeps the best epoch) instead of running a fixed number of epochs.
   - Only `open_questions` may say TBD.
5. Run `python experiments/check.py` (needs `pip install pyyaml`) and fix everything it reports.
6. Show me each file, with a one-line summary per experiment. Once I approve, commit them as `experiments/<id>.md` on my branch and push it.

Hard rules:
- No personal data in the files: no emails, phone numbers, student IDs, credentials, tokens or private datasets,
  and no real names other than my first name.
- Do not copy any text from the cluster's user manual or from internal university documents.
- Commit the entries **with my code, on my own branch** of the team STAIR repo, as `experiments/<id>.md`, and push
  that branch. Do not merge into `main`, open a PR, or push to anyone else's branch. Dhameem picks it up from
  there. STAIR is public, so check the files for personal data again before pushing.
- Push with GitHub's noreply email as the commit identity, not my personal email.
- Do not submit cluster jobs.

Also fill in `adopt`: anything in my code that the team could reuse, for example a faster data loader, a better
prompt, a tidier eval loop, or a utility that `stair_common` lacks. Give the file path and one line on why. Say
`none` if nothing stands out; don't oversell.

---
