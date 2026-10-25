"""Jev novelty E3 (pre-registration H3): the weak model's own escalation signals, the baselines a router is measured
against, for a pair's cached split.

The weak expert is re-loaded exactly as run-experts ran it (its config in the run folder: model, top-k passages from the
cached nano-jev ranking, experiment 1's prompt and chat template, transformers backend). Per question:

    teacher-forced on its cached answer (one forward pass, no generation)
        sig_msp            probability of the whole answer (product over its tokens)
        sig_mean_logprob   mean token log-probability
        sig_first_margin   top-1 minus top-2 probability of the answer's first token
        hidden             last layer at the last prompt token and its mean over the answer tokens: <split>-hidden.npy
                           (float16 [questions, 2, hidden]), for a logistic probe fitted on router_train
    P(True)                one extra prompt: "Is the proposed answer correct?", P(yes) / (P(yes) + P(no))
    self-consistency       k samples at temperature T (default 10, 0.7): the share agreeing with the cached answer
                           (normalised), the number of distinct answers, and discrete semantic entropy over exact-match
                           clusters of the samples plus the cached answer
    self-verification      AutoMix style: k_verify samples (default 8) of "does the context support the answer?";
                           the share answering yes
    NLI                    cross-encoder/nli-deberta-v3-xsmall (71M): P(entailment) that the weak model's passages
                           entail "The answer to <question> is <answer>." (sig_nli_entail; added to finished splits too)

Every extra call's tokens are kept (prompt once, generated tokens summed over samples, as AgentTTS Eq. 8 prices them),
so the analysis can charge each signal its own calls. Writes <run>/signals/<split>.parquet; resumes per split.

    signals --run runs/slr-pairA --data-dir data/hotpotqa --split-data distractor_validation.parquet --split test
"""

import argparse
import json
import logging
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from stair import provenance
from stair.experts import HFExpert, clean_answer
from stair.judge import load_pair_split
from stair.metrics import normalize_answer, text

log = logging.getLogger(__name__)

NLI = "cross-encoder/nli-deberta-v3-xsmall"
CLAIM = 'The answer to "{question}" is {answer}.'
PTRUE = ("Question: {question}\nProposed answer: {answer}\nIs the proposed answer correct? Reply with only yes or no.")
VERIFY = ("{context}\n\nQuestion: {question}\nProposed answer: {answer}\n"
          "Does the context above support the proposed answer? Reply with only yes or no.")


def semantic_entropy(answers: list[str]) -> float:
    """Discrete semantic entropy: entropy of the answer distribution over exact-match (normalised) clusters."""
    counts: dict[str, int] = {}
    for a in answers:
        counts[normalize_answer(a)] = counts.get(normalize_answer(a), 0) + 1
    p = np.array(list(counts.values()), dtype=float) / len(answers)
    return float(-(p * np.log(p)).sum())


def yes_share(texts: list[str]) -> float:
    votes = [t.strip().lower().lstrip("*\"' ") for t in texts]
    yes = sum(v.startswith("yes") for v in votes)
    no = sum(v.startswith("no") for v in votes)
    return yes / (yes + no) if yes + no else 0.5


class WeakModel:
    """The pair's small expert, for scoring its own answers."""

    def __init__(self, expert_cfg: dict, device: str | None = None, batch_size: int = 16):
        self.expert = HFExpert("small", expert_cfg.get("top_k"), expert_cfg["model"], device=device,
                               dtype=expert_cfg.get("dtype", "bfloat16"), revision=expert_cfg.get("revision"))
        self.torch, self.tok, self.model = self.expert.torch, self.expert.tokenizer, self.expert.model
        self.device, self.batch_size = self.expert.device, batch_size
        self.yes_ids = self._first_ids(["yes", "Yes", " yes", " Yes"])
        self.no_ids = self._first_ids(["no", "No", " no", " No"])

    def _first_ids(self, words: list[str]) -> list[int]:
        return sorted({self.tok(w, add_special_tokens=False)["input_ids"][0] for w in words})

    def chat(self, content: str) -> str:
        if self.tok.chat_template:
            return self.tok.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                                add_generation_prompt=True)
        return content

    def teacher_forced(self, prompts: list[str], answers: list[str]) -> list[dict]:
        """MSP, mean log-prob, first-token margin and hidden states of each cached answer under its prompt."""
        torch = self.torch
        out = []
        for i in range(0, len(prompts), self.batch_size):
            p_ids = [self.tok(p, add_special_tokens=not self.tok.chat_template)["input_ids"] for p in prompts[i: i + self.batch_size]]
            a_ids = [self.tok(a, add_special_tokens=False)["input_ids"] or [self.tok.eos_token_id]
                     for a in answers[i: i + self.batch_size]]
            seqs = [p + a for p, a in zip(p_ids, a_ids)]
            width = max(map(len, seqs))
            pad = self.tok.pad_token_id
            ids = torch.tensor([s + [pad] * (width - len(s)) for s in seqs], device=self.device)
            mask = torch.tensor([[1] * len(s) + [0] * (width - len(s)) for s in seqs], device=self.device)
            with torch.no_grad():
                res = self.model.base_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            last = res.hidden_states[-1].float()
            for b, (p, a) in enumerate(zip(p_ids, a_ids)):
                # only the positions that predict the answer go through the LM head: logits over every position x
                # the vocabulary (152k for Qwen), in float, ran out of memory on an L40S (jobs 43220, 43224, 43225)
                with torch.no_grad():
                    logits = self.model.get_output_embeddings()(res.last_hidden_state[b, len(p) - 1: len(p) + len(a) - 1])
                logp = torch.log_softmax(logits.float(), dim=-1)
                tok_lp = logp[torch.arange(len(a), device=self.device), torch.tensor(a, device=self.device)]
                first = logp[0].exp().topk(2).values
                out.append({"sig_msp": float(tok_lp.sum().exp()), "sig_mean_logprob": float(tok_lp.mean()),
                            "sig_first_margin": float(first[0] - first[1]), "answer_tokens": len(a),
                            "hidden": torch.stack([last[b, len(p) - 1], last[b, len(p): len(p) + len(a)].mean(0)])
                            .to(torch.float16).cpu().numpy()})
        return out

    def p_yes(self, prompts: list[str]) -> tuple[np.ndarray, np.ndarray]:
        """P(yes) / (P(yes) + P(no)) at the next token of each prompt, and the prompts' token counts."""
        torch = self.torch
        probs, tokens = [], []
        for i in range(0, len(prompts), self.batch_size):
            enc = self.tok(prompts[i: i + self.batch_size], return_tensors="pt", padding=True,
                           add_special_tokens=not self.tok.chat_template).to(self.device)  # left-padded (HFExpert)
            with torch.no_grad():
                p = torch.softmax(self.model(**enc).logits[:, -1].float(), dim=-1)
            yes, no = p[:, self.yes_ids].sum(-1), p[:, self.no_ids].sum(-1)
            probs += (yes / (yes + no)).tolist()
            tokens += enc["attention_mask"].sum(1).tolist()
        return np.asarray(probs), np.asarray(tokens)

    def sample(self, prompts: list[str], k: int, temperature: float, max_new_tokens: int) -> tuple[list[list[str]], np.ndarray, np.ndarray]:
        """k samples per prompt; the prompts' token counts and the generated tokens summed over the k samples."""
        torch = self.torch
        texts, p_tok, g_tok = [], [], []
        for i in range(0, len(prompts), max(1, self.batch_size // 2)):
            enc = self.tok(prompts[i: i + max(1, self.batch_size // 2)], return_tensors="pt", padding=True,
                           add_special_tokens=not self.tok.chat_template).to(self.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, do_sample=True, temperature=temperature, top_p=1.0, top_k=None,
                                          num_return_sequences=k, max_new_tokens=max_new_tokens,
                                          repetition_penalty=1.0, pad_token_id=self.tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[1]:].tolist()
            for q in range(enc["input_ids"].shape[0]):
                outs, n_gen = [], 0
                for ids in new[q * k: (q + 1) * k]:
                    n = next((j + 1 for j, t in enumerate(ids) if t in self.expert.stop_ids), len(ids))
                    n_gen += n
                    outs.append(clean_answer(self.tok.decode(ids[:n], skip_special_tokens=True)))
                texts.append(outs)
                g_tok.append(n_gen)
            p_tok += enc["attention_mask"].sum(1).tolist()
        return texts, np.asarray(p_tok), np.asarray(g_tok)


def nli_entailment(df: pd.DataFrame, device: str | None = None, batch_size: int = 64) -> np.ndarray:
    """P(entailment) that each question's passages entail the claim built from its weak answer."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(NLI)
    model = AutoModelForSequenceClassification.from_pretrained(NLI).to(device).eval()
    entail = next(i for i, name in model.config.id2label.items() if name.lower().startswith("entail"))
    premises = ["\n".join(p) for p in df["passages"]]
    claims = [CLAIM.format(question=q, answer=text(a)) for q, a in zip(df["question"], df["small_pred"])]
    out = []
    for i in range(0, len(claims), batch_size):
        enc = tok(premises[i: i + batch_size], claims[i: i + batch_size], truncation="only_first", max_length=512,
                  padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            out += torch.softmax(model(**enc).logits.float(), dim=-1)[:, entail].tolist()
    return np.asarray(out)


def run_split(model: WeakModel, df: pd.DataFrame, k: int, k_verify: int, temperature: float) -> tuple[pd.DataFrame, np.ndarray]:
    answers = [text(a) for a in df["small_pred"]]
    prompts = [model.expert._prompt(q, p) for q, p in zip(df["question"], df["passages"])]
    tf = model.teacher_forced(prompts, answers)
    hidden = np.stack([r.pop("hidden") for r in tf])
    out = pd.DataFrame({"id": df["id"].to_numpy(), **pd.DataFrame(tf).to_dict("list")})
    ptrue, ptrue_tokens = model.p_yes([model.chat(PTRUE.format(question=q, answer=a)) for q, a in zip(df["question"], answers)])
    out["sig_ptrue"], out["ptrue_prompt_tokens"] = ptrue, ptrue_tokens
    samples, sp_tok, sg_tok = model.sample(prompts, k, temperature, model.expert.max_new_tokens)
    norm = [normalize_answer(a) for a in answers]
    out["sig_sc_agree"] = [np.mean([normalize_answer(s) == g for s in ss]) for ss, g in zip(samples, norm)]
    out["sig_sc_distinct"] = [len({normalize_answer(s) for s in ss}) for ss in samples]
    out["sig_semantic_entropy"] = [semantic_entropy([a, *ss]) for a, ss in zip(answers, samples)]
    out["sample_prompt_tokens"], out["sample_generated_tokens"] = sp_tok, sg_tok
    verify = [model.chat(VERIFY.format(context="\n\n".join(p), question=q, answer=a))
              for q, p, a in zip(df["question"], df["passages"], answers)]
    votes, vp_tok, vg_tok = model.sample(verify, k_verify, temperature, 4)
    out["sig_verify_yes"] = [yes_share(v) for v in votes]
    out["verify_prompt_tokens"], out["verify_generated_tokens"] = vp_tok, vg_tok
    return out, hidden


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", type=Path, required=True, help="a pair's run folder (config.toml, <split>.parquet)")
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--split", action="append", required=True, help="repeatable: router_train, calib, test, ...")
    parser.add_argument("--split-data", action="append", default=[],
                        help="the dataset file of each --split, in order (default: as judge.SPLIT_DATA)")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-verify", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--device")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cfg = tomllib.loads((args.run / "config.toml").read_text())
    small = next(e for e in cfg["experts"] if e["name"] == "small")
    if small.get("kind") != "hf":
        raise SystemExit(f"{args.run}: the small expert ran on {small.get('kind')!r}; signals re-load it with transformers")
    target = args.run / "signals"
    target.mkdir(parents=True, exist_ok=True)
    files = dict(zip(args.split, args.split_data)) if args.split_data else {}
    model = None
    for split in args.split:
        path = target / f"{split}.parquet"
        if path.exists():
            done = pd.read_parquet(path)
            if "sig_nli_entail" not in done:
                df = load_pair_split(args.run, args.data_dir, split, split_data={split: files[split]} if split in files else None)
                done["sig_nli_entail"] = pd.Series(nli_entailment(df, args.device), index=df["id"]).loc[done["id"]].to_numpy()
                done.to_parquet(path, index=False)
                log.info(f"signals {split}: NLI added, mean entailment {done['sig_nli_entail'].mean():.3f}")
            else:
                log.info(f"signals {split}: done")
            continue
        df = load_pair_split(args.run, args.data_dir, split, split_data={split: files[split]} if split in files else None)
        model = model or WeakModel(small, args.device, args.batch_size)
        model.torch.manual_seed(args.seed)
        out, hidden = run_split(model, df, args.k, args.k_verify, args.temperature)
        out["sig_nli_entail"] = nli_entailment(df, args.device)
        np.save(target / f"{split}-hidden.npy", hidden)
        out.to_parquet(target / f"{split}.parquet", index=False)
        log.info(f"signals {split}: {len(out)} questions | MSP {out['sig_msp'].mean():.3f}, P(True) {out['sig_ptrue'].mean():.3f}, "
                 f"agree {out['sig_sc_agree'].mean():.3f}, verify {out['sig_verify_yes'].mean():.3f}")
    provenance.record(target, "signals", pair_run=str(args.run), splits=args.split, k=args.k, k_verify=args.k_verify,
                      temperature=args.temperature, model=small["model"], revision=provenance.hub_revision(small["model"]),
                      **provenance.environment())
    (target / "settings.json").write_text(json.dumps({"k": args.k, "k_verify": args.k_verify,
                                                      "temperature": args.temperature, "model": small["model"],
                                                      "params_b": small.get("params"), "ptrue": PTRUE, "verify": VERIFY},
                                                     indent=1))


if __name__ == "__main__":
    main()
