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

Every call records its prompt and generated tokens. Cost is GFLOPs = 2 x params x tokens, counting
each sample as its own call (as AgentTTS's API calls are), plus nano-jev's measured scoring cost.
It is also given in AgentTTS units (one 3B pass of 128 + 64 tokens = 1,152 GFLOPs). Answers are
scored three ways:

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


def qa_prompt(question: str, passages: list[str]) -> str:
    """AgentTTS's answering prompt (archon_hotpotqa.py, CustomGenerator.generate), verbatim."""
    query_references = "Query: " + question + "\nReferences:\n"
    for i, p in enumerate(passages):
        query_references += f"[{i + 1}] " + p + "\n"
    return ("You will be provided with a set of references related to the query. "
            "Your task is to answer the query using the provided references. If the references are insufficient, "
            "supplement your response with your own knowledge. "
            f"Here is the query and references: \n {query_references}.")


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


def top_passages(pool: pd.Series, scores: list[float], k: int) -> list[str]:
    order = np.argsort(-np.asarray(scores), kind="stable")[:k]
    return [pool["texts"][i] for i in order]


def answer_chunk(gen: Generator, cell: dict, rows: pd.DataFrame, seed: int) -> pd.DataFrame:
    n, fuse = cell.get("samples", 1), cell.get("fuse", False)
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
            tokens = n * g["prompt_tokens"] + sum(g["gen_tokens"]) + ((f["prompt_tokens"] + f["gen_tokens"][0]) if f else 0)
            gflops = 2 * cell["params"] * tokens
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


def load_split(out: Path, split: str, k: int, limit: int | None) -> pd.DataFrame:
    pools = pd.read_parquet(out / "pools" / f"{split}.parquet")
    scores = pd.read_parquet(out / "scores" / f"{split}.parquet", columns=["id", "nanojev", "tokens"])
    df = pools.merge(scores, on="id", validate="1:1")
    if limit:
        df = df.head(limit)
    df["passages"] = [top_passages(r, r["nanojev"], k) for _, r in df.iterrows()]
    df["nanojev_gflops"] = 2 * NANOJEV_PARAMS_B * df["tokens"]
    return df.reset_index(drop=True)


def cmd_run(args, plan: dict) -> None:
    out = Path(plan.get("out", "runs/exp8"))
    for cell in plan["cells"]:
        if args.only and cell["name"] not in args.only:
            continue
        df = load_split(out, args.split, cell["top_k"], args.limit)
        suffix = f"-limit{args.limit}" if args.limit else ""
        parts = out / "answers" / f"{args.split}{suffix}" / f"{cell['name']}.parts"
        parts.mkdir(parents=True, exist_ok=True)
        chunks = [df.iloc[i: i + args.chunk_size] for i in range(0, len(df), args.chunk_size)]
        todo = [c for c in range(len(chunks)) if not (parts / f"{c:05d}.parquet").exists()]
        log.info(f"{args.split}/{cell['name']}: {len(df)} questions, {len(todo)} of {len(chunks)} chunks to run")
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
            merged = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            merged["units"] += (df["nanojev_gflops"] / UNIT_GFLOPS).to_numpy()  # every cell pays nano-jev's scoring
            merged["gflops"] += df["nanojev_gflops"].to_numpy()
            merged.to_parquet(parts.parent / f"{cell['name']}.parquet")


def cmd_summary(args, plan: dict) -> None:
    out = Path(plan.get("out", "runs/exp8"))
    rows = []
    for split_dir in sorted((out / "answers").glob("*")):
        cells = {f.stem: pd.read_parquet(f) for f in sorted(split_dir.glob("*.parquet"))}
        for name, d in cells.items():
            rows.append({"split": split_dir.name, "cell": name, "n": len(d), "lenient": d["lenient"].mean(),
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["run", "summary"])
    parser.add_argument("--plan", type=Path, default=Path("configs/exp8.toml"))
    parser.add_argument("--split", default="test", help="run: pools/<split>.parquet")
    parser.add_argument("--only", action="append", default=[], help="run: cell names (default all); repeatable")
    parser.add_argument("--limit", type=int, help="run: first N questions only (a pilot); written to <split>-limitN")
    parser.add_argument("--chunk-size", type=int, default=50, help="run: questions per resumable chunk")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = tomllib.loads(args.plan.read_text())
    {"run": cmd_run, "summary": cmd_summary}[args.step](args, plan)


if __name__ == "__main__":
    main()
