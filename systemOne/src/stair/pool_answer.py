"""Experiment 8, phase 1 (docs/experiment-8-plan.md): answer pool questions from nano-jev's top-k passages.

Each *cell* of a plan (configs/exp8.toml) is one answering policy: a model, how many nano-jev
passages it reads, and how many samples it draws. With several samples it can fuse them into one
answer. Prompts follow AgentTTS's HotpotQA code (`archon_hotpotqa.py`) and Archon's fuser, so the
answering side matches the published setup; only retrieval differs (nano-jev top-k instead of an LLM
retriever). Two deliberate deviations:

- The passages are numbered texts without titles, as in AgentTTS. Our fusion prompt carries only
  the passages the answerer saw, whereas AgentTTS's fuser repeats the whole 100-candidate user
  message twice.
- Sampling is seeded per question, so a rerun reproduces the same answers.

    pool-answer run --plan configs/exp8.toml --split test [--only llama3b-top5] [--limit 20]
    pool-answer summary --plan configs/exp8.toml
    pool-answer recost --plan configs/exp8.toml --split test   # rewrite a finished split's cost under Eq. 8
    pool-answer compare --plan configs/e7-evidence.toml --split test --base llama3b-s20fuse-top5   # paired intervals

A cell's `ranker` (Jev novelty E7) picks its passages: nano-jev's scores (the default), `bm25` (the pool's BM25),
another scored ranker (`msmarco`: pools score --column msmarco), or the `gold` oracle (the gold paragraphs first).

Every call records its prompt and generated tokens. Cost is GFLOPs = 2 x params x tokens as AgentTTS
counts them (App. A.2, Eq. 8): the samples of one prompt share its prefill, so the prompt counts once
and every sample's generated tokens count; the fuser is a new call with its own prompt. Plus nano-jev's
measured scoring cost. It is also given in AgentTTS units (one 3B pass of 128 + 64 tokens = 1,152
GFLOPs). Until 2026-10-18 the prompt was counted once per sample; `recost` rewrites such files from
their stored tokens and keeps the old cost as `gflops_prompt_per_sample` / `units_prompt_per_sample`.
Answers are scored three ways:

- lenient: the normalised gold answer occurs in the normalised response (a placeholder for the
  team's lenient scorer);
- the official EM / F1 of the whole response;
- the official EM / F1 of its first line.
"""

import argparse
import json
import logging
import math
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance
from stair.experts import clean_answer
from stair.metrics import exact_match, f1_score, normalize_answer

log = logging.getLogger(__name__)

SYSTEM = "You are a helpful assistant."
FUSER_SYSTEM = "You are a helpful assistant who fuses multiple answers"
UNIT_GFLOPS = 2 * 3.0 * (128 + 64)  # AgentTTS's budget unit: one 3B pass on its cheapest task
NANOJEV_PARAMS_B = 0.0334
RANKER_PARAMS_B = {"msmarco": 0.0334}  # cross-encoder/ms-marco-MiniLM-L12-v2: nano-jev's MiniLM-L12-H384 architecture


def qa_prompt(question: str, passages: list[str]) -> str:
    """AgentTTS's answering prompt (archon_hotpotqa.py, CustomGenerator.generate), verbatim."""
    query_references = "Query: " + question + "\nReferences:\n"
    for i, p in enumerate(passages):
        query_references += f"[{i + 1}] " + p + "\n"
    return ("You will be provided with a set of references related to the query. "
            "Your task is to answer the query using the provided references. If the references are insufficient, "
            "supplement your response with your own knowledge. "
            f"Here is the query and references: \n {query_references}.")


SLR_QA = ("Your task is to answer the query using the provided references. "
          "If the references are insufficient, supplement your response with your own knowledge. "
          "Answer with ONLY the exact answer phrase - no explanation, no full sentence.\n")


def slr_qa_prompt(question: str, block: str) -> str:
    """Jonathan's grid answering prompt (STAIR jonathan/lab_package/run_labels4.py, qa_prompt), verbatim: AgentTTS's
    task with "answer with only the exact answer phrase". `block` is "Query: ...\\nSelected references: [n] ..."."""
    return SLR_QA + f"Query: {question}\n{block}"


def slr_block(question: str, passages: list[str]) -> str:
    """His build_block layout for passages we chose: "Query: ...\\nSelected references: [1] title: text [2] ..."."""
    refs = " ".join(f"[{i + 1}] {p}" for i, p in enumerate(passages)) or "(none)"
    return f"Query: {question}\nSelected references: {refs}"


def fuser_prompt(query: str, responses: list[str]) -> str:
    """Archon's make_fuser_prompt (no critiques), verbatim; `query` is the answerer's user message."""
    prompt = (f"You have been provided with a set of responses from various open-source models to the latest user query, "
              f"which is {query}."
              "            Your task is to synthesize these responses into a single, high-quality response. "
              "            It is crucial to critically evaluate the information provided in these responses, recognizing "
              "that some of it may be biased or incorrect. "
              "            Your response should not simply replicate the given answers but should offer a refined, accurate, "
              "and comprehensive reply to the instruction. "
              "            Ensure your response is well-structured, coherent, and adheres to the highest standards of "
              "accuracy and reliability.\n")
    prompt += f"Once again, the query is: {query}\n"
    prompt += "Responses from models:"
    for i, r in enumerate(responses):
        prompt += f"\n{i + 1}. {r}"
    return prompt


def lenient(prediction: str, gold: str) -> float:
    """Placeholder lenient match: the normalised gold answer occurs in the normalised prediction."""
    g, p = normalize_answer(gold), normalize_answer(prediction)
    return float(bool(g) and f" {g} " in f" {p} ")


def agreement(samples: list[str]) -> float:
    """Mean pairwise token F1 between the samples' first lines (1 = they all say the same)."""
    short = [clean_answer(s) for s in samples]
    pairs = [(a, b) for i, a in enumerate(short) for b in short[i + 1:]]
    return float(np.mean([f1_score(a, b) for a, b in pairs])) if pairs else float("nan")


class EntropyRecorder:
    """A logits processor that records each step's entropy of the raw next-token distribution.

    Keeping every step's logits (output_logits) would hold steps x sequences x vocabulary floats:
    for 40 sampled sequences of up to 512 tokens over Llama's 128k vocabulary, about 10 GB. In
    transformers 5, custom processors run before the sampling warpers (temperature, top-k, top-p),
    so `scores` are the raw logits here (checked in generation/utils.py, _get_logits_processor).
    """

    def __init__(self):
        self.steps = []

    def __call__(self, input_ids, scores):
        import torch

        logp = torch.log_softmax(scores.float(), dim=-1)
        self.steps.append(-(logp.exp() * logp).nan_to_num().sum(-1).cpu())
        return scores


class Generator:
    """A chat LLM with batched sampling; returns texts, token counts and mean raw-logit token entropy."""

    def __init__(self, model: str, device: str | None = None, dtype: str = "bfloat16", load_in_4bit: bool = False,
                 revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        kwargs = {"revision": revision}
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            kwargs |= {"quantization_config": BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                                                 bnb_4bit_compute_dtype=torch.bfloat16),
                       "device_map": "auto"}  # spread over every visible GPU when one is not enough
            self.model = AutoModelForCausalLM.from_pretrained(model, **kwargs)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, **kwargs).to(self.device)
        self.model.eval()
        self.input_device = next(self.model.parameters()).device

    def chat(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return f"{system}\n\n{user}\n\n"

    def generate(self, prompts: list[str], n: int, temperature: float, max_new_tokens: int, seed: int) -> list[dict]:
        """For each prompt: {'texts': [n], 'prompt_tokens', 'gen_tokens': [n], 'entropy': [n]}."""
        torch = self.torch
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=not self.tokenizer.chat_template)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        torch.manual_seed(seed)
        sample = temperature > 0
        recorder = EntropyRecorder()
        with torch.no_grad():
            out = self.model.generate(
                **enc.to(self.input_device), max_new_tokens=max_new_tokens, do_sample=sample,
                temperature=temperature if sample else None, top_p=1.0 if sample else None, top_k=0 if sample else None,
                repetition_penalty=1.0, num_return_sequences=n, logits_processor=[recorder],
                return_dict_in_generate=True, pad_token_id=self.tokenizer.pad_token_id)
        new = out.sequences[:, enc["input_ids"].shape[1]:].tolist()
        entropy = torch.stack(recorder.steps, dim=1) if recorder.steps else torch.zeros(len(new), 0)  # [batch * n, steps]
        stops = set(self.model.generation_config.eos_token_id if isinstance(self.model.generation_config.eos_token_id, list)
                    else [self.model.generation_config.eos_token_id]) | {self.tokenizer.eos_token_id}
        results = []
        for b in range(len(prompts)):
            texts, gens, ents = [], [], []
            for j in range(n):
                ids = new[b * n + j]
                m = next((i for i, t in enumerate(ids) if t in stops), len(ids))
                texts.append(self.tokenizer.decode(ids[:m], skip_special_tokens=True).strip())
                gens.append(m)
                steps = entropy[b * n + j, :m].tolist()
                ents.append(sum(steps) / len(steps) if steps else math.inf)
            results.append({"texts": texts, "prompt_tokens": int(lengths[b]), "gen_tokens": gens, "entropy": ents})
        return results


def ranked(pool: pd.Series, ranker: str = "nanojev") -> np.ndarray:
    """A pool's candidates in a ranker's order (Jev novelty E7): its scores' column (nanojev, bm25, or another ranker's
    scores merged by load_split), or `gold`, an oracle: the gold paragraphs, then the rest in nano-jev's order."""
    order = np.argsort(-np.asarray(pool["nanojev" if ranker == "gold" else ranker]), kind="stable")
    if ranker == "gold":
        gold = set(np.asarray(pool["gold_idx"]).tolist())
        order = np.array([i for i in order if i in gold] + [i for i in order if i not in gold])
    return order


def answer_chunk(gen: Generator, cell: dict, rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    n, fuse = cell.get("samples", 1), cell.get("fuse", False)
    if cell.get("prompt") == "slr":  # rows carry the references block (passage_blocks)
        users = [slr_qa_prompt(r["question"], r["block"]) for _, r in rows.iterrows()]
    else:
        users = [qa_prompt(r["question"], r["passages"]) for _, r in rows.iterrows()]
    out_rows, b = [], cell.get("batch_size", 8)
    for i in range(0, len(users), b):
        part = rows.iloc[i: i + b]
        start = time.perf_counter()
        res = gen.generate([gen.chat(SYSTEM, u) for u in users[i: i + b]], n, cell.get("temperature", 0.3),
                           cell.get("max_new_tokens", 512), seed + int(part.index[0]))
        fused = [None] * len(res)
        if fuse:
            fres = gen.generate([gen.chat(FUSER_SYSTEM, fuser_prompt(u, r["texts"])) for u, r in zip(users[i: i + b], res)],
                                1, cell.get("fuse_temperature", 0.3), cell.get("fuse_max_new_tokens", 1024),
                                seed + 7 + int(part.index[0]))
            fused = fres
        seconds = (time.perf_counter() - start) / len(res)
        for (_, r), g, f in zip(part.iterrows(), res, fused):
            pred = f["texts"][0] if f else g["texts"][0]
            gflops = cell_gflops(cell["params"], g["prompt_tokens"], sum(g["gen_tokens"]),
                                 f["prompt_tokens"] if f else 0, f["gen_tokens"][0] if f else 0)
            out_rows.append({
                "id": r["id"], "pred": pred, "samples": g["texts"] if n > 1 else None,
                "lenient": lenient(pred, r["answer"]), "em": exact_match(pred, r["answer"]), "f1": f1_score(pred, r["answer"]),
                "short_em": exact_match(clean_answer(pred), r["answer"]), "short_f1": f1_score(clean_answer(pred), r["answer"]),
                "prompt_tokens": g["prompt_tokens"], "gen_tokens": int(sum(g["gen_tokens"])),
                "fuse_prompt_tokens": f["prompt_tokens"] if f else 0, "fuse_gen_tokens": f["gen_tokens"][0] if f else 0,
                "gflops": gflops, "units": gflops / UNIT_GFLOPS,
                "entropy": (f or g)["entropy"][0], "agreement": agreement(g["texts"]) if n > 1 else float("nan"),
                "seconds": seconds,
            })
    return pd.DataFrame(out_rows)


def passage_blocks(df: pd.DataFrame, cell: dict) -> list[str]:
    """The references block of an `slr`-prompt cell: nano-jev's top k as "title: text" (passages = "nanojev"), or
    Jonathan's Qwen2.5-72B selection from his grid (passages = "slr-72b", read through `slr_plan`)."""
    if cell.get("passages", "nanojev") == "nanojev":
        return [slr_block(r["question"], [f"{r['titles'][i]}: {r['texts'][i]}"
                                          for i in np.argsort(-np.asarray(r["nanojev"]), kind="stable")[:cell["top_k"]]])
                for _, r in df.iterrows()]
    from stair import slr

    slr_plan = tomllib.loads(Path(cell["slr_plan"]).read_text(encoding="utf-8"))
    slr.check_files(slr_plan)
    position = {q: i for i, q in enumerate(pd.read_parquet(slr_plan["dev"], columns=["id"])["id"])}
    selected = {r["k"]: r["v"] for r in slr.read_jsonl(Path(slr_plan["out"]) / slr.GRID / "retr_strong.jsonl")}
    return [str(selected[str(position[q])]) for q in df["id"]]


def reused_answers(plan: dict, cell: str) -> pd.DataFrame:
    """Earlier answers of the same cell (plan `reuse_answers`: answer folders), by id; empty if none."""
    found = [pd.read_parquet(Path(d) / f"{cell}.parquet") for d in plan.get("reuse_answers", [])
             if (Path(d) / f"{cell}.parquet").exists()]
    return pd.concat(found).drop_duplicates("id").set_index("id") if found else pd.DataFrame()


def load_split(out: Path, split: str, k: int, limit: int | None, ranker: str = "nanojev") -> pd.DataFrame:
    """A split's pools with each question's top k passages by `ranker` (see `ranked`) and what ranking cost:
    ranker_gflops is nano-jev's scoring (also for the gold oracle, whose fill is in nano-jev's order), 0 for BM25, or
    the other ranker's own scoring (scores/<split>-<ranker>.parquet, pools score --column)."""
    pools = pd.read_parquet(out / "pools" / f"{split}.parquet")
    scores = pd.read_parquet(out / "scores" / f"{split}.parquet", columns=["id", "nanojev", "tokens"])
    df = pools.merge(scores, on="id", validate="1:1")
    df["nanojev_gflops"] = 2 * NANOJEV_PARAMS_B * df["tokens"]
    df["ranker_gflops"] = 0.0 if ranker == "bm25" else df["nanojev_gflops"]
    if ranker not in ("nanojev", "bm25", "gold"):
        other = pd.read_parquet(out / "scores" / f"{split}-{ranker}.parquet", columns=["id", ranker, "tokens"])
        df = df.merge(other.rename(columns={"tokens": "ranker_tokens"}), on="id", validate="1:1")
        if len(df) != len(pools):
            raise ValueError(f"{split}-{ranker}: scores for {len(df)} of {len(pools)} pools")
        df["ranker_gflops"] = 2 * RANKER_PARAMS_B[ranker] * df["ranker_tokens"]
    if limit:
        df = df.head(limit)
    df["passages"] = [[r["texts"][i] for i in ranked(r, ranker)[:k]] for _, r in df.iterrows()]
    return df.reset_index(drop=True)


def cmd_run(args, plan: dict) -> None:
    out = Path(plan.get("out", "runs/exp8"))
    for cell in plan["cells"]:
        if args.only and cell["name"] not in args.only:
            continue
        df = load_split(out, args.split, cell["top_k"], args.limit, cell.get("ranker", "nanojev"))
        if cell.get("prompt") == "slr":
            df["block"] = passage_blocks(df, cell)
        suffix = f"-limit{args.limit}" if args.limit else ""
        parts = out / "answers" / f"{args.split}{suffix}" / f"{cell['name']}.parts"
        parts.mkdir(parents=True, exist_ok=True)
        reused = reused_answers(plan, cell["name"])
        new = df[~df["id"].isin(reused.index)] if len(reused) else df  # keeps df's index: the sampling seeds
        chunks = [new.iloc[i: i + args.chunk_size] for i in range(0, len(new), args.chunk_size)]
        todo = [c for c in range(len(chunks)) if not (parts / f"{c:05d}.parquet").exists()]
        log.info(f"{args.split}/{cell['name']}: {len(df)} questions ({len(df) - len(new)} reused), "
                 f"{len(todo)} of {len(chunks)} chunks to run")
        if todo:
            gen = Generator(cell["model"], args.device, load_in_4bit=cell.get("load_in_4bit", False),
                            revision=cell.get("revision"))
            provenance.record(out, "pool-answer", split=args.split, cell=cell, limit=args.limit,
                              revision=provenance.hub_revision(cell["model"], cell.get("revision")),
                              **provenance.environment())
            for c in todo:
                start = time.perf_counter()
                answer_chunk(gen, cell, chunks[c], plan.get("seed", 0)).to_parquet(parts / f"{c:05d}.parquet")
                log.info(f"  chunk {c + 1}/{len(chunks)}: {(time.perf_counter() - start) / len(chunks[c]):.2f} s/question")
            del gen
            import gc

            gc.collect()
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass
        files = sorted(parts.glob("*.parquet"))
        if len(files) == len(chunks):
            merged = pd.concat([pd.read_parquet(f) for f in files]
                               + ([reused.loc[df["id"][df["id"].isin(reused.index)]].reset_index()] if len(reused) else []),
                               ignore_index=True).set_index("id").loc[df["id"]].reset_index()  # the split's order
            # from the tokens, so chunks written before the Eq. 8 change merge right; every cell pays its ranker's scoring
            merged["gflops"] = token_gflops(merged, cell["params"]) + df["ranker_gflops"].to_numpy()
            merged["units"] = merged["gflops"] / UNIT_GFLOPS
            merged.to_parquet(parts.parent / f"{cell['name']}.parquet")


def cell_gflops(params: float, prompt_tokens, gen_tokens, fuse_prompt_tokens, fuse_gen_tokens):
    """AgentTTS's FLOPs of one question's answer (App. A.2, Eq. 8, parameter terms): the samples share the prompt's
    prefill, so the prompt counts once and `gen_tokens` is the sum over the samples; the fuser is a call of its own."""
    return 2 * params * (prompt_tokens + gen_tokens + fuse_prompt_tokens + fuse_gen_tokens)


def token_gflops(answers: pd.DataFrame, params: float) -> np.ndarray:
    """cell_gflops over an answer file's stored token columns (without nano-jev's scoring)."""
    return cell_gflops(params, *(answers[c].to_numpy(dtype=float) for c in
                                 ("prompt_tokens", "gen_tokens", "fuse_prompt_tokens", "fuse_gen_tokens")))


def recost(answers: pd.DataFrame, cell: dict, nanojev_gflops: np.ndarray) -> pd.DataFrame:
    """A merged answer file with its cost under Eq. 8. The first recost of a file written with the prompt counted per
    sample keeps that cost as gflops/units_prompt_per_sample, after checking the file is what that rule produced."""
    out = answers.copy()
    new = token_gflops(out, cell["params"]) + nanojev_gflops
    if "gflops_prompt_per_sample" not in out:
        n = cell.get("samples", 1)
        old = new + 2 * cell["params"] * (n - 1) * out["prompt_tokens"].to_numpy(dtype=float)
        if not np.allclose(out["gflops"].to_numpy(dtype=float), old, rtol=1e-9, atol=1e-6):
            raise ValueError(f"{cell['name']}: stored gflops do not match the per-sample-prompt rule; not recosting")
        out["gflops_prompt_per_sample"], out["units_prompt_per_sample"] = out["gflops"], out["units"]
    out["gflops"], out["units"] = new, new / UNIT_GFLOPS
    return out


def cmd_recost(args, plan: dict) -> None:
    out = Path(plan.get("out", "runs/exp8"))
    scores = pd.read_parquet(out / "scores" / f"{args.split}.parquet", columns=["id", "tokens"]).set_index("id")
    for cell in plan["cells"]:
        path = out / "answers" / args.split / f"{cell['name']}.parquet"
        if (args.only and cell["name"] not in args.only) or not path.exists():
            continue
        answers = pd.read_parquet(path)
        nanojev = 2 * NANOJEV_PARAMS_B * scores.loc[answers["id"], "tokens"].to_numpy(dtype=float)
        done = recost(answers, cell, nanojev)
        done.to_parquet(path)
        log.info(f"{args.split}/{cell['name']}: {len(done)} answers, units {done['units_prompt_per_sample'].mean():.1f} "
                 f"(prompt per sample) -> {done['units'].mean():.1f} (Eq. 8)")
    provenance.record(out, "pool-answer recost", split=args.split, rule="AgentTTS App. A.2 Eq. 8: prompt once per call")


def cmd_summary(args, plan: dict) -> None:
    out = Path(plan.get("out", "runs/exp8"))
    rows = []
    from stair_common.score import lenient as lenient_std

    for split_dir in sorted((out / "answers").glob("*")):
        cells = {f.stem: pd.read_parquet(f) for f in sorted(split_dir.glob("*.parquet"))}
        pools = out / "pools" / f"{split_dir.name.split('-limit')[0]}.parquet"
        gold = pd.read_parquet(pools, columns=["id", "answer"]).set_index("id")["answer"] if pools.exists() else None
        for name, d in cells.items():
            std = (float(np.mean([lenient_std(str(p), [str(gold[i])]) for i, p in zip(d["id"], d["pred"])]))
                   if gold is not None else float("nan"))  # the team's standard ruler
            rows.append({"split": split_dir.name, "cell": name, "n": len(d), "lenient_std": std, "lenient": d["lenient"].mean(),
                         "em": d["em"].mean(), "f1": d["f1"].mean(), "short_em": d["short_em"].mean(),
                         "short_f1": d["short_f1"].mean(), "units": d["units"].mean(), "gflops": d["gflops"].mean(),
                         "s_per_q": d["seconds"].mean()})
        if len(cells) > 1:  # oracle: the cheapest cell that is lenient-correct, else the cheapest cell
            ids = set.intersection(*(set(d["id"]) for d in cells.values()))
            al = {k: d.set_index("id").loc[sorted(ids)] for k, d in cells.items()}
            ok = np.array([al[k]["lenient"].to_numpy() for k in al])
            cost = np.array([al[k]["units"].to_numpy() for k in al])
            pick = np.where(ok.any(0), np.where(ok > 0, cost, np.inf).argmin(0), cost.argmin(0))
            cols = np.arange(len(ids))
            rows.append({"split": split_dir.name, "cell": "oracle", "n": len(ids), "lenient": ok.max(0).mean(),
                         "units": cost[pick, cols].mean()})
    table = pd.DataFrame(rows)
    table["pct_of_agenttts_ss"] = 100 * table["units"] / plan.get("agenttts_strong_units", 929)
    table.to_csv(out / "answers" / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 20):
        log.info("\n" + table.round(3).to_string(index=False))


def cmd_compare(args, plan: dict) -> None:
    """Jev novelty E7: each finished cell of the plan against `--base` on all of the split's questions, by the team's
    lenient ruler, with a paired bootstrap over questions; writes answers/compare-<split>.json."""
    from stair_common.score import lenient as lenient_std

    out = Path(plan.get("out", "runs/exp8"))
    gold = pd.read_parquet(out / "pools" / f"{args.split}.parquet", columns=["id", "answer"]).set_index("id")["answer"]
    folder = out / "answers" / args.split
    specs = {c["name"]: c for c in plan["cells"] if (folder / f"{c['name']}.parquet").exists()}
    if args.base not in specs:
        raise SystemExit(f"{args.base}: no finished answers in {folder}")
    cells = {k: pd.read_parquet(folder / f"{k}.parquet").set_index("id").loc[gold.index] for k in specs}
    ok = {k: np.array([lenient_std(str(p), [str(g)]) for p, g in zip(d["pred"], gold)], dtype=float)
          for k, d in cells.items()}
    rows = np.random.default_rng(args.seed).integers(0, len(gold), size=(args.bootstrap, len(gold)))
    report = {"split": args.split, "base": args.base, "n": len(gold), "bootstrap": args.bootstrap, "cells": {}}
    for k, d in cells.items():
        diff = ok[k] - ok[args.base]
        boot = diff[rows].mean(axis=1)
        report["cells"][k] = {"ranker": specs[k].get("ranker", "nanojev"), "top_k": specs[k]["top_k"],
                              "samples": specs[k].get("samples", 1), "fuse": specs[k].get("fuse", False),
                              "lenient_std": float(ok[k].mean()), "units": float(d["units"].mean()),
                              "diff": float(diff.mean()), "ci95": np.percentile(boot, [2.5, 97.5]).tolist(),
                              "p_le_0": float((boot <= 0).mean())}
        lo, hi = report["cells"][k]["ci95"]
        log.info(f"{k:32s} lenient {ok[k].mean():.3f} at {d['units'].mean():6.1f} units | "
                 f"vs {args.base} {diff.mean():+.3f} [{lo:+.3f}, {hi:+.3f}]")
    (folder.parent / f"compare-{args.split}.json").write_text(json.dumps(report, indent=1))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["run", "summary", "recost", "compare"])
    parser.add_argument("--plan", type=Path, default=Path("configs/exp8.toml"))
    parser.add_argument("--split", default="test", help="run: pools/<split>.parquet")
    parser.add_argument("--only", action="append", default=[], help="run: cell names (default all); repeatable")
    parser.add_argument("--limit", type=int, help="run: first N questions only (a pilot); written to <split>-limitN")
    parser.add_argument("--chunk-size", type=int, default=50, help="run: questions per resumable chunk")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    parser.add_argument("--base", help="compare: the cell every other is set against")
    parser.add_argument("--bootstrap", type=int, default=2000, help="compare: paired resamples of the questions")
    parser.add_argument("--seed", type=int, default=0, help="compare: the bootstrap's seed")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = tomllib.loads(args.plan.read_text())
    {"run": cmd_run, "summary": cmd_summary, "recost": cmd_recost, "compare": cmd_compare}[args.step](args, plan)


if __name__ == "__main__":
    main()
