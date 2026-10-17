"""Experiment 8: re-score the cached long-form answers by self-commit, then an LLM grader.

The lenient scorer (gold answer contained anywhere in the response) rewards long answers. Pasting
nano-jev's top 5 passages as the "answer" scores 0.798 lenient, above every model. Version 4 scores
each cached answer in `<out>/answers/<split>/<cell>.parquet` in two steps:

1. commit: the model that wrote the response finishes its own turn. The response's exact final
   conversation is replayed (the answering prompt, or for a fused cell the fuser prompt with its
   samples), followed by the response and the cue "So, in as few words as possible, the final answer
   is:". The same model continues greedily. Its short answer is the system's committed answer, and
   the official HotpotQA EM / F1 are computed on it (`stair.metrics`), as in experiments 1-7. In a
   deployed system this is a continuation of the same turn, so its cost is the cue plus the generated
   tokens of that model (`commit_units`). That cost is added to the cell's cost.
2. grade: a fixed grader (default Qwen2.5-14B-Instruct) reads (question, gold answer, committed short
   answer) and gives a one-word verdict. P(CORRECT) is the next-token mass of every vocabulary token that
   begins "CORRECT" against every token that begins "INCORRECT", case-insensitive. The gold answer is
   correct by definition. `correct` = P >= 0.5 is the primary metric.

    rescore validate                  # must pass before anything else
    rescore commit --split test       # the cells' own models (3B and 70B), every cell of that split
    rescore run    --split test       # the grader, on the committed answers
    rescore summary

`validate` writes `<out>/rescore/validation.json`:

- Grader, on known short answers: experiment 1's 7B answers to HotpotQA dev items 1000-1499, with
  official EM / F1.
  - EM = 1 should be graded CORRECT.
  - F1 = 0 should mostly be INCORRECT. HotpotQA's gold strings have aliases ("AdScam" for "The
    sponsorship scandal"), so a good grader cannot reach 100% there.
  - The written greedy verdict must agree with P(CORRECT), and almost all next-token mass must fall on
    the two verdicts.
- Commit, on calib answers of three cells (the 3B single, the 3B with 40 samples fused, the 70B): the
  commits graded CORRECT must be grounded in the response (at least 60% of the commit's normalised
  tokens appear in it; yes/no exempt), and commits must almost never be empty. An ungrounded correct
  commit is an answer the continuation added and the response did not give. Abstentions ("I do not have
  enough information") are often ungrounded paraphrases, but they are graded wrong and cannot inflate
  accuracy, so they do not count against the gate.

History (2026-10-15). Each version failed its validation; the results READMEs are in
`results/exp8-rescore/`.
- Version 1 graded the whole response with a "correct?" prompt. The 14B judged against its own
  knowledge.
- Version 2 graded the whole response against the gold. It was good on short answers (0.996 / 0.917),
  but passed 62% of pasted-passage dumps, because it treats "the gold is stated somewhere" as an answer.
- Version 3 had the 14B extract a short answer, then graded it. The extractor dropped 24% of correct
  bare answers as "none", and still read answers out of 41% of the dumps.

Version 4 keeps version 2's grader prompt, which passed on short answers, and lets the system commit
its own answer, so no outside model has to decide what a response "meant". The passage-dump control
does not apply to version 4, because a dump is not the output of a conversation that can be continued.
"""

import argparse
import json
import logging
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance
from stair.metrics import exact_match, f1_score, normalize_answer

log = logging.getLogger(__name__)

GRADER = "Qwen/Qwen2.5-14B-Instruct"
GRADE_SYSTEM = ("You compare a response with a reference answer. The reference (gold) answer is correct by definition: "
                "never use your own knowledge to second-guess it, and do not judge whether the response is true in general.")
GRADE_USER = """Question: {question}
Gold answer: {gold}
Response: {response}

Does the response give the same final answer as the gold answer?
- CORRECT: the response commits to one final answer, and it matches the gold answer: the same entity, date, number, place, or yes/no. Different wording, spelling, aliases, abbreviations, formatting, units or date order are fine, and so is extra explanation.
- INCORRECT: the response's final answer differs from the gold answer; or it says it cannot answer or that the information is missing; or it hedges between several candidates; or it gives no final answer at all (for example it only quotes, lists or summarises references or passages); or the gold answer appears only inside quoted material while the response concludes something else.
Reply with exactly one word: CORRECT or INCORRECT."""
VERDICTS = ("CORRECT", "INCORRECT")
COMMIT_CUE = "\n\nSo, in as few words as possible, the final answer is:"

# Version 4 validation thresholds, fixed on 2026-10-15 before any version 4 run. The grader ones are
# version 2's (its prompt is unchanged, and it passed them on this slice).
MIN_AGREE_EM1 = 0.95         # EM = 1 short answers graded CORRECT
MIN_AGREE_F10 = 0.80         # F1 = 0 short answers graded INCORRECT (gold aliases make 100% unreachable)
MIN_GEN_AGREEMENT = 0.95     # the written greedy verdict agrees with P(CORRECT) >= 0.5
MIN_VERDICT_MASS = 0.90      # mean next-token mass on the two verdicts (format adherence)
MIN_COMMIT_GROUNDED = 0.90   # per checked cell: of the commits graded CORRECT, the share grounded in the response.
                             # An ungrounded correct commit is an answer the continuation added (a "rescue")
MIN_COMMIT_NONEMPTY = 0.98   # per checked cell: committed answers that are not empty
GROUNDED_RECALL = 0.6        # grounded: this share of the commit's normalised tokens appears in the response
COMMIT_CHECK_CELLS = ("llama3b-top5", "llama3b-s40fuse-top5", "llama70b-top5")


def verdict_token_sets(tokenizer) -> tuple[np.ndarray, np.ndarray]:
    """Vocabulary ids whose text (stripped, upper-cased) is a non-empty prefix of CORRECT / of INCORRECT."""
    correct, incorrect = [], []
    for i in range(len(tokenizer)):
        text = tokenizer.decode([i]).strip().upper()
        if not text:
            continue
        if VERDICTS[0].startswith(text):
            correct.append(i)
        elif VERDICTS[1].startswith(text):
            incorrect.append(i)
    return np.asarray(correct, dtype=np.int64), np.asarray(incorrect, dtype=np.int64)


def release_memory() -> None:
    """Give a dropped model's GPU memory back before the next one loads (the caller `del`s its reference first)."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


class Grader:
    """A chat LLM that grades a short answer against the gold from its next-token verdict mass."""

    def __init__(self, model: str, device: str | None = None, revision: str | None = None, dtype: str = "bfloat16"):
        from stair.pool_answer import Generator

        self.gen = Generator(model, device, dtype=dtype, revision=revision)
        self.tok, self.model, self.torch = self.gen.tokenizer, self.gen.model, self.gen.torch
        self.correct_ids, self.incorrect_ids = verdict_token_sets(self.tok)

    def verdict_probs(self, prompts: list[str], batch: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """P(CORRECT) and the verdict mass (P(CORRECT-prefix) + P(INCORRECT-prefix)) for each prompt.

        Prompts run in length order, so a batch holds similar lengths. One prompt per pass (`batch = 1`,
        the default) makes each grade independent of the others: with heavy left padding in bfloat16
        the same answer's probability moved by up to 0.077 (Qwen2.5-0.5B; float32 agreed to 2e-5).
        """
        torch = self.torch
        p_correct = np.full(len(prompts), 0.5)
        mass = np.zeros(len(prompts))
        if not (len(self.correct_ids) and len(self.incorrect_ids)):
            log.warning("the tokenizer has no CORRECT / INCORRECT prefix tokens: every grade is 0.5")
            return p_correct, mass
        order = np.argsort([len(p) for p in prompts], kind="stable")
        dev = self.gen.input_device
        c_ids, i_ids = torch.tensor(self.correct_ids, device=dev), torch.tensor(self.incorrect_ids, device=dev)
        pad = self.tok.pad_token_id
        for s in range(0, len(order), batch):
            idx = order[s: s + batch]
            seqs = [self.tok(prompts[j], add_special_tokens=False)["input_ids"] for j in idx]
            width = max(map(len, seqs))
            ids = torch.full((len(seqs), width), pad, dtype=torch.long)
            mask = torch.zeros((len(seqs), width), dtype=torch.long)
            for r, q in enumerate(seqs):  # left padding: every prompt ends at the last position
                ids[r, width - len(q):] = torch.tensor(q)
                mask[r, width - len(q):] = 1
            positions = (mask.cumsum(-1) - 1).clamp(min=0)  # count from each row's first real token, as generate() does
            with torch.no_grad():
                logits = self.model(input_ids=ids.to(dev), attention_mask=mask.to(dev), position_ids=positions.to(dev),
                                    logits_to_keep=1).logits
            last = torch.softmax(logits[:, -1, :].float(), dim=-1)  # [:, -1] is the last position whether or not
            pc, pi = last[:, c_ids].sum(-1), last[:, i_ids].sum(-1)  # logits_to_keep was honoured
            p_correct[idx] = (pc / (pc + pi).clamp(min=1e-12)).cpu().numpy()
            mass[idx] = (pc + pi).cpu().numpy()
        return p_correct, mass

    def grade_prompts(self, questions, golds, responses) -> list[str]:
        return [self.gen.chat(GRADE_SYSTEM, GRADE_USER.format(question=q, gold=g, response=r))
                for q, g, r in zip(questions, golds, responses)]

    def grade(self, questions, golds, responses, batch: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """(P(CORRECT), verdict mass) per response. An empty response is never correct."""
        p, mass = self.verdict_probs(self.grade_prompts(questions, golds, responses), batch)
        empty = np.array([not str(r).strip() for r in responses], dtype=bool)
        return np.where(empty, 0.0, p), mass

    def written_verdicts(self, questions, golds, responses, batch: int = 8) -> np.ndarray:
        """The greedy written verdict: 1.0 CORRECT, 0.0 INCORRECT, nan when it wrote neither (a self-check)."""
        prompts = self.grade_prompts(questions, golds, responses)
        out = np.full(len(prompts), np.nan)
        order = np.argsort([len(p) for p in prompts], kind="stable")
        for s in range(0, len(order), batch):
            idx = order[s: s + batch]
            res = self.gen.generate([prompts[j] for j in idx], 1, 0.0, 4, seed=0)
            for j, r in zip(idx, res):
                text = r["texts"][0].strip().upper()
                out[j] = 0.0 if text.startswith(VERDICTS[1]) else 1.0 if text.startswith(VERDICTS[0]) else np.nan
        return out


# ---------------------------------------------------------------------------------------------
# commit: the answering model finishes its own turn


def committed_text(text: str) -> str:
    """The first non-empty line of the continuation, without quotes, markdown bold or a final full stop."""
    line = next((s.strip() for s in text.splitlines() if s.strip()), "")
    return line.strip('*"\' ').rstrip(".").strip('*"\' ')


def grounded(answer: str, response: str) -> bool:
    """At least GROUNDED_RECALL of the committed answer's normalised tokens appear in the response. A yes/no
    commit counts as grounded (a response can conclude "yes" without the word); an empty one does not."""
    tokens = normalize_answer(answer).split()
    if not tokens:
        return False
    if tokens[0] in ("yes", "no"):
        return True
    words = set(normalize_answer(response).split())
    return sum(t in words for t in tokens) / len(tokens) >= GROUNDED_RECALL


def commit_prompts(gen, rows: pd.DataFrame, cell: dict) -> list[str]:
    """The response's final conversation (pool_answer.answer_chunk), the response, then the cue."""
    from stair.pool_answer import FUSER_SYSTEM, SYSTEM, fuser_prompt, qa_prompt

    out = []
    for _, r in rows.iterrows():
        user = qa_prompt(r["question"], r["passages"])
        if cell.get("fuse"):
            head = gen.chat(FUSER_SYSTEM, fuser_prompt(user, list(r["samples"])))
        else:
            head = gen.chat(SYSTEM, user)
        out.append(head + str(r["pred"]).strip() + COMMIT_CUE)
    return out


def commit_rows(gen, rows: pd.DataFrame, cell: dict, batch: int, max_new_tokens: int = 24) -> pd.DataFrame:
    """Committed answers for `rows` (question, passages, pred and, for fused cells, samples)."""
    from stair.pool_answer import UNIT_GFLOPS

    prompts = commit_prompts(gen, rows, cell)
    cue_tokens = len(gen.tokenizer(COMMIT_CUE, add_special_tokens=False)["input_ids"])
    order = np.argsort([len(p) for p in prompts], kind="stable")
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s: s + batch]
        for j, r in zip(idx, gen.generate([prompts[j] for j in idx], 1, 0.0, max_new_tokens, seed=0)):
            res[j] = r
    committed = [committed_text(r["texts"][0]) for r in res]
    tokens = np.array([cue_tokens + r["gen_tokens"][0] for r in res])
    return pd.DataFrame({
        "id": rows["id"].to_numpy(), "committed": committed, "commit_raw": [r["texts"][0] for r in res],
        "commit_prompt_tokens": [r["prompt_tokens"] for r in res], "commit_tokens": tokens,
        "commit_units": 2 * cell["params"] * tokens / UNIT_GFLOPS,
        "grounded": [float(grounded(c, p)) for c, p in zip(committed, rows["pred"])],
    })


def answer_cells(plan_path: Path) -> dict[str, dict]:
    return {c["name"]: c for c in tomllib.loads(plan_path.read_text())["cells"]}


def commit_frame(out: Path, split: str, cell_name: str, k: int) -> pd.DataFrame:
    """The cell's cached answers joined with the question, the passages it read and its samples."""
    from stair.pool_answer import load_split

    q = load_split(out, split, k, None)[["id", "question", "answer", "passages"]]
    a = pd.read_parquet(out / "answers" / split / f"{cell_name}.parquet", columns=["id", "pred", "samples"])
    return a.merge(q, on="id", how="left", validate="1:1")


def load_generator(cell: dict, args):
    from stair.pool_answer import Generator

    model = args.commit_model or cell["model"]
    return Generator(model, args.device, load_in_4bit=bool(cell.get("load_in_4bit")) and not args.commit_model)


def commit_batch(cell: dict) -> int:
    return 1 if cell.get("fuse") else 8  # fused conversations run to 16k tokens


# ---------------------------------------------------------------------------------------------


def known_slice(path: Path, start: int, count: int) -> pd.DataFrame:
    """Rows [start, start + count) of experiment 1's dev cache (HotpotQA dev file order), with question and gold."""
    known = pd.read_parquet(path, columns=["id", "question", "answer", "large_pred", "large_em", "large_f1"])
    return known.iloc[start: start + count].reset_index(drop=True)


def check_grader(grader: Grader, known: pd.DataFrame, batch: int) -> tuple[dict, pd.DataFrame]:
    pk, mk = grader.grade(known["question"], known["answer"], known["large_pred"], batch)
    wk = grader.written_verdicts(known["question"], known["answer"], known["large_pred"])
    parsed = ~np.isnan(wk)
    em1, f10 = known["large_em"].to_numpy() == 1, known["large_f1"].to_numpy() == 0
    stats = {
        "known_n": int(len(known)), "known_em1_n": int(em1.sum()), "known_f10_n": int(f10.sum()),
        "agree_em1_correct": float((pk[em1] >= 0.5).mean()), "agree_f10_incorrect": float((pk[f10] < 0.5).mean()),
        "known_graded_correct": float((pk >= 0.5).mean()), "known_em": float(em1.mean()),
        "written_parsed": float(parsed.mean()),
        "written_agreement": float(((pk[parsed] >= 0.5) == (wk[parsed] == 1.0)).mean()) if parsed.any() else 0.0,
        "verdict_mass_mean": float(mk.mean()),
    }
    dis = known.assign(p_correct=pk, written=wk)
    dis = dis[(em1 & (pk < 0.5)) | (f10 & (pk >= 0.5))]
    return stats, dis[["id", "question", "answer", "large_pred", "large_em", "large_f1", "p_correct", "written"]]


def cmd_validate(args) -> None:
    out = args.out
    report = {"method": "v4: self-commit by the answering model, then grade the committed answer",
              "thresholds": {"min_agree_em1": MIN_AGREE_EM1, "min_agree_f10": MIN_AGREE_F10,
                             "min_gen_agreement": MIN_GEN_AGREEMENT, "min_verdict_mass": MIN_VERDICT_MASS,
                             "min_commit_grounded": MIN_COMMIT_GROUNDED, "min_commit_nonempty": MIN_COMMIT_NONEMPTY}}
    start = time.perf_counter()
    # 1. commit, on calib answers of the checked cells (each model loads once)
    cells = answer_cells(args.answer_plan)
    checked = [c for c in (args.commit_cells or COMMIT_CHECK_CELLS) if (out / "answers" / args.commit_split / f"{c}.parquet").exists()]
    commits, samples = {}, {}
    by_model: dict[str, list[str]] = {}
    for c in checked:
        by_model.setdefault(args.commit_model or cells[c]["model"], []).append(c)
    for model, names in by_model.items():
        gen = load_generator(cells[names[0]], args)
        provenance.record(out / "rescore", "rescore-validate-commit", model=model, cells=names,
                          revision=provenance.hub_revision(model, None), **provenance.environment())
        for c in names:
            rows = commit_frame(out, args.commit_split, c, cells[c]["top_k"]).iloc[: args.commit_count]
            commits[c] = commit_rows(gen, rows, cells[c], commit_batch(cells[c]))
            samples[c] = rows
        del gen
        release_memory()
    report["commit"] = {
        c: {"n": int(len(d)), "grounded": float(d["grounded"].mean()),
            "nonempty": float((d["committed"].str.strip() != "").mean()),
            "mean_commit_units": float(d["commit_units"].mean()),
            "lenient_of_response": float(np.mean([s >= 0.5 for s in pd.read_parquet(
                out / "answers" / args.commit_split / f"{c}.parquet", columns=["id", "lenient"]).set_index("id").loc[d["id"], "lenient"]])),
            "commit_em": float(np.mean([exact_match(a, g) for a, g in zip(d["committed"], samples[c]["answer"])]))}
        for c, d in commits.items()}
    # 2. the grader, on known short answers (the gate slice, and the earlier diagnostic slice)
    grader = Grader(args.grader, args.device, args.revision)
    revision = provenance.hub_revision(args.grader, args.revision)
    provenance.record(out / "rescore", "rescore-validate", grader=args.grader, revision=revision, **provenance.environment())
    gate, gate_dis = check_grader(grader, known_slice(args.known, args.known_start, args.known_count), args.batch)
    report |= {"grader": args.grader, "revision": revision,
               "gate": {"known": f"{args.known} rows {args.known_start}-{args.known_start + args.known_count - 1}", **gate}}
    if args.diagnostic_start is not None:
        diag, _ = check_grader(grader, known_slice(args.known, args.diagnostic_start, args.known_count), args.batch)
        report["diagnostic_not_gating"] = {"known": f"rows {args.diagnostic_start}-{args.diagnostic_start + args.known_count - 1}", **diag}
    # 3. the committed answers graded (reported, not gating: there is no ground truth for them)
    for c, d in commits.items():
        p, _ = grader.grade(samples[c]["question"], samples[c]["answer"], d["committed"], args.batch)
        ok = p >= 0.5
        commits[c] = d.assign(p_correct=p)
        report["commit"][c] |= {"graded_correct": float(ok.mean()), "n_graded_correct": int(ok.sum()),
                                "grounded_when_correct": float(d["grounded"].to_numpy()[ok].mean()) if ok.any() else 0.0}
    pd.concat([d.assign(cell=c, question=samples[c]["question"].to_numpy(), gold=samples[c]["answer"].to_numpy(),
                        response=samples[c]["pred"].to_numpy()) for c, d in commits.items()]).to_csv(
        out / "rescore" / "validation-commits.csv", index=False)
    report["seconds"] = time.perf_counter() - start
    commit_ok = bool(commits) and all(v["grounded_when_correct"] >= MIN_COMMIT_GROUNDED and v["nonempty"] >= MIN_COMMIT_NONEMPTY
                                      for v in report["commit"].values())
    report["passed"] = bool(gate["agree_em1_correct"] >= MIN_AGREE_EM1 and gate["agree_f10_incorrect"] >= MIN_AGREE_F10
                            and gate["written_agreement"] >= MIN_GEN_AGREEMENT
                            and gate["verdict_mass_mean"] >= MIN_VERDICT_MASS and commit_ok)
    gate_dis.head(40).to_csv(out / "rescore" / "validation-disagreements.csv", index=False)
    (out / "rescore" / "validation.json").write_text(json.dumps(report, indent=1))
    log.info(json.dumps(report, indent=1))
    log.info("VALIDATION " + ("PASSED" if report["passed"] else "FAILED") + f" (see {out / 'rescore' / 'validation.json'})")


def require_validation(args) -> None:
    if args.skip_validation_check:
        return
    v = args.out / "rescore" / "validation.json"
    if not v.exists() or not json.loads(v.read_text()).get("passed") or not json.loads(v.read_text()).get("method", "").startswith("v4"):
        raise SystemExit(f"{v} is missing, not version 4, or did not pass: run `rescore validate` first (or --skip-validation-check)")


def split_cells(args) -> list[Path]:
    cells = sorted((args.out / "answers" / args.split).glob("*.parquet"))
    return [c for c in cells if c.stem in args.only] if args.only else cells


def cmd_commit(args) -> None:
    require_validation(args)
    out, cells = args.out, answer_cells(args.answer_plan)
    todo = [f.stem for f in split_cells(args) if not (out / "commit" / args.split / f.name).exists()]
    by_model: dict[str, list[str]] = {}
    for c in todo:
        by_model.setdefault(args.commit_model or cells[c]["model"], []).append(c)
    for model, names in by_model.items():
        gen = load_generator(cells[names[0]], args)
        for c in names:
            provenance.record(out / "rescore", "rescore-commit", split=args.split, cell=c, model=model,
                              revision=provenance.hub_revision(model, None), **provenance.environment())
            frame = commit_frame(out, args.split, c, cells[c]["top_k"])
            parts = out / "commit" / args.split / f"{c}.parts"
            parts.mkdir(parents=True, exist_ok=True)
            start = time.perf_counter()
            for i in range(0, len(frame), args.chunk_size):
                pf = parts / f"{i // args.chunk_size:05d}.parquet"
                if not pf.exists():
                    commit_rows(gen, frame.iloc[i: i + args.chunk_size], cells[c], commit_batch(cells[c])).to_parquet(pf)
            merged = pd.concat([pd.read_parquet(p) for p in sorted(parts.glob("*.parquet"))], ignore_index=True)
            merged.to_parquet(out / "commit" / args.split / f"{c}.parquet")
            log.info(f"commit {args.split}/{c}: {len(merged)} answers in {time.perf_counter() - start:.0f} s | "
                     f"grounded {merged['grounded'].mean():.3f} empty {(merged['committed'] == '').mean():.3f} "
                     f"units {merged['commit_units'].mean():.2f}")
        del gen
        release_memory()


def questions_of(out: Path, split: str) -> pd.DataFrame:
    return pd.read_parquet(out / "pools" / f"{split}.parquet", columns=["id", "question", "answer"]).set_index("id")


def cmd_run(args) -> None:
    require_validation(args)
    out = args.out
    q = questions_of(out, args.split)
    grader = None
    for f in split_cells(args):
        target = out / "rescore" / args.split / f.name
        if target.exists():
            log.info(f"{args.split}/{f.stem}: done")
            continue
        commit_file = out / "commit" / args.split / f.name
        if not commit_file.exists():
            raise SystemExit(f"{commit_file} is missing: run `rescore commit --split {args.split}` first")
        grader = grader or Grader(args.grader, args.device, args.revision)
        provenance.record(out / "rescore", "rescore-run", split=args.split, cell=f.stem, grader=args.grader,
                          revision=provenance.hub_revision(args.grader, args.revision), **provenance.environment())
        c = pd.read_parquet(commit_file).merge(pd.read_parquet(f, columns=["id", "lenient"]), on="id", validate="1:1")
        questions, golds = q.loc[c["id"], "question"].tolist(), q.loc[c["id"], "answer"].tolist()
        start = time.perf_counter()
        p, mass = grader.grade(questions, golds, c["committed"].tolist(), args.batch)
        d = pd.DataFrame({
            "id": c["id"], "committed": c["committed"], "p_correct": p, "correct": (p >= 0.5).astype(float),
            "verdict_mass": mass, "commit_em": [exact_match(a, g) for a, g in zip(c["committed"], golds)],
            "commit_f1": [f1_score(a, g) for a, g in zip(c["committed"], golds)], "grounded": c["grounded"],
            "commit_units": c["commit_units"], "lenient": c["lenient"]})
        target.parent.mkdir(parents=True, exist_ok=True)
        d.to_parquet(target)
        log.info(f"{args.split}/{f.stem}: {len(d)} answers in {time.perf_counter() - start:.0f} s | "
                 f"lenient {d['lenient'].mean():.3f} | graded {d['correct'].mean():.3f} | "
                 f"EM {d['commit_em'].mean():.3f} F1 {d['commit_f1'].mean():.3f} | grounded {d['grounded'].mean():.3f}")


def cmd_summary(args) -> None:
    rows = []
    for split_dir in sorted((args.out / "rescore").iterdir()):
        if not split_dir.is_dir():
            continue
        for f in sorted(split_dir.glob("*.parquet")):
            d = pd.read_parquet(f)
            units = pd.read_parquet(args.out / "answers" / split_dir.name / f.name, columns=["units"])["units"].mean()
            rows.append({"split": split_dir.name, "cell": f.stem, "n": len(d), "lenient": d["lenient"].mean(),
                         "graded_correct": d["correct"].mean(), "commit_em": d["commit_em"].mean(),
                         "commit_f1": d["commit_f1"].mean(), "grounded": d["grounded"].mean(),
                         "verdict_mass": d["verdict_mass"].mean(), "answer_units": units,
                         "commit_units": d["commit_units"].mean(), "units": units + d["commit_units"].mean()})
    table = pd.DataFrame(rows)
    table.to_csv(args.out / "rescore" / "summary.csv", index=False)
    with pd.option_context("display.width", 200):
        log.info("\n" + table.round(3).to_string(index=False))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["validate", "commit", "run", "summary"])
    parser.add_argument("--out", type=Path, default=Path("runs/exp8"))
    parser.add_argument("--grader", default=GRADER)
    parser.add_argument("--revision", help="grader revision to pin (default: whatever the cache holds)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--answer-plan", type=Path, default=Path("configs/exp8.toml"),
                        help="the answering cells (model, params, top_k, fuse) that wrote the cached answers")
    parser.add_argument("--commit-model", help="use this model for every cell's commit (tests only; default: the cell's own)")
    parser.add_argument("--split", default="test", help="commit / run: answers/<split>/")
    parser.add_argument("--only", action="append", default=[], help="commit / run: cell names (default all); repeatable")
    parser.add_argument("--batch", type=int, default=1,
                        help="responses per grading forward pass (1 = exact; larger batches shift bf16 grades slightly)")
    parser.add_argument("--chunk-size", type=int, default=250, help="commit: answers per resumable chunk")
    parser.add_argument("--known", type=Path, default=Path("results/gpu/test.parquet"),
                        help="validate: experiment 1's dev cache with short answers and official EM/F1 (dev file order)")
    parser.add_argument("--known-start", type=int, default=1000, help="validate: first row of the grader's gate slice")
    parser.add_argument("--known-count", type=int, default=500)
    parser.add_argument("--diagnostic-start", type=int, default=500,
                        help="validate: also report (not gate on) this slice, the one version 1 was diagnosed on")
    parser.add_argument("--commit-split", default="calib", help="validate: answers checked for the commit step")
    parser.add_argument("--commit-cells", action="append", default=[], help="validate: cells checked (default: the 3B, fused-40, 70B)")
    parser.add_argument("--commit-count", type=int, default=200, help="validate: answers per checked cell")
    parser.add_argument("--skip-validation-check", action="store_true", help="commit / run: do not require a passed validation")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    (args.out / "rescore").mkdir(parents=True, exist_ok=True)
    {"validate": cmd_validate, "commit": cmd_commit, "run": cmd_run, "summary": cmd_summary}[args.step](args)


if __name__ == "__main__":
    main()
