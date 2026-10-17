"""Answering experts. Each reads the question plus its top-k paragraphs.

`HeuristicExpert` needs no model and exists only to exercise the pipeline. `HFExpert` runs a
Hugging Face causal LM with greedy decoding, one question at a time. `VLLMExpert` does the same in
batches with vLLM and is the one to use on a GPU.
"""

import math
import re

from dataclasses import dataclass, field

import numpy as np

from stair.evidence import content_words, tokenize

AUX_VERBS = frozenset("is are was were do does did can could has have had will would".split())

# Two worked examples, chosen on calib questions: small-model F1 0.235 -> 0.466 vs asking for
# "yes/no for yes/no questions", which made Qwen2.5-0.5B answer "Yes" to open questions.
PROMPT = (
    "Answer the question using the paragraphs below. Reply with only the answer, a few words, not a sentence.\n\n"
    "Example: Question: Which city is the birthplace of the author of Dracula? Answer: Dublin\n"
    "Example: Question: Are Paris and Rome both capital cities? Answer: yes\n\n"
    "{context}\n\nQuestion: {question}\nAnswer:"
)


def clean_answer(text: str) -> str:
    return text.strip().split("\n")[0].strip().rstrip(".")


@dataclass
class ExpertOutput:
    answer: str
    cost: int  # prompt + generated tokens (whitespace tokens for the heuristic)
    uncertainty: float  # higher = less sure; mean token entropy for LMs
    prompt_tokens: int = 0
    generated_tokens: int = 0
    token_entropy: list[float] = field(default_factory=list)  # per generated token, for quantile features


class HeuristicExpert:
    """Picks the sentence with most question overlap and returns a capitalised phrase from it.

    With hops=1 it then follows that phrase to the paragraph of the same title and answers from
    there, which is how a bridge question is resolved when the paragraph is in its context.
    """

    def __init__(self, name: str, top_k: int | None, hops: int = 0):
        self.name, self.top_k, self.hops = name, top_k, hops

    @staticmethod
    def _best_sentence(words: set[str], paragraphs: list[str]) -> tuple[str, float]:
        best, best_overlap = "", -1.0
        for para in paragraphs:
            for sent in re.split(r"(?<=[.!?])\s+", para):
                overlap = len(words & content_words(sent)) / max(len(words), 1)
                if overlap > best_overlap:
                    best, best_overlap = sent, overlap
        return best, best_overlap

    @staticmethod
    def _new_phrase(sentence: str, known: set[str]) -> str | None:
        for phrase in re.findall(r"[A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*", sentence.split(":", 1)[-1]):
            if not content_words(phrase) <= known:
                return phrase
        return None

    def answer(self, question: str, paragraphs: list[str]) -> ExpertOutput:
        cost = len(question.split()) + sum(len(p.split()) for p in paragraphs)
        if tokenize(question)[:1] and tokenize(question)[0] in AUX_VERBS:
            return ExpertOutput("yes", cost, 0.5, cost - 1, 1)

        known = content_words(question)
        sentence, overlap = self._best_sentence(known, paragraphs)
        phrase = self._new_phrase(sentence, known)
        for _ in range(self.hops):
            if phrase is None:
                break
            linked = [p for p in paragraphs if p.startswith(f"{phrase}:")]
            if not linked:
                break
            known |= content_words(phrase)
            phrase = self._new_phrase(linked[0], known) or phrase
        answer, uncertainty = (sentence.split(":")[0], 1.0) if phrase is None else (phrase, 1.0 - overlap)
        generated = len(answer.split())
        return ExpertOutput(answer, cost, uncertainty, cost - generated, generated)


class HFExpert:
    """transformers causal LM, plain greedy decoding, left-padded batches.

    The model's own generation config is overridden: Qwen2.5 ships a repetition penalty, which
    transformers applies even without sampling, and vLLM does not. Uncertainty is the mean entropy
    of the raw (unprocessed) next-token distribution over the full vocabulary.

    `load_in_4bit` uses bitsandbytes NF4, the fallback for 7B models on a 16 GB GPU when vLLM
    is unavailable.
    """

    def __init__(self, name: str, top_k: int | None, model: str, device: str | None = None,
                 max_new_tokens: int = 16, dtype: str = "auto", batch_size: int = 1, load_in_4bit: bool = False,
                 revision: str | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.name, self.top_k, self.max_new_tokens, self.batch_size = name, top_k, max_new_tokens, batch_size
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model, padding_side="left", revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        if load_in_4bit:
            from transformers import BitsAndBytesConfig

            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(model, quantization_config=quant, device_map=self.device,
                                                              revision=revision)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model, dtype=dtype, revision=revision).to(self.device)
        self.model.eval()
        self.max_length = getattr(self.model.config, "max_position_embeddings", None)
        eos = self.model.generation_config.eos_token_id
        self.stop_ids = set(eos if isinstance(eos, list) else [eos]) | {self.tokenizer.eos_token_id}

    def _prompt(self, question: str, paragraphs: list[str]) -> str:
        text = PROMPT.format(context="\n\n".join(paragraphs), question=question)
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                      add_generation_prompt=True)
        return text

    def _generate(self, prompts: list[str]) -> list[ExpertOutput]:
        enc = self.tokenizer(prompts, return_tensors="pt", padding=True, add_special_tokens=not self.tokenizer.chat_template)
        lengths = enc["attention_mask"].sum(dim=1).tolist()
        if self.max_length:
            too_long = [i for i, n in enumerate(lengths) if n + self.max_new_tokens > self.max_length]
            if too_long:
                raise ValueError(f"{len(too_long)} prompts exceed {self.max_length - self.max_new_tokens} tokens "
                                 f"(batch positions {too_long[:5]}); use a longer-context model rather than truncating")
        enc = enc.to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **enc,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None, top_p=None, top_k=None,  # unused without sampling; silences config warnings
                repetition_penalty=1.0,
                output_logits=True,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        new = out.sequences[:, enc["input_ids"].shape[1]:].tolist()
        logp = self.torch.stack([self.torch.log_softmax(s.float(), dim=-1) for s in out.logits], dim=1)
        entropy = -(logp.exp() * logp).sum(-1).cpu()  # [batch, steps], raw logits: no processors applied
        results = []
        for b, ids in enumerate(new):
            n = len(ids)
            for j, tok in enumerate(ids):
                if tok in self.stop_ids:
                    n = j + 1
                    break
            steps = entropy[b, :n].tolist()
            text = self.tokenizer.decode(ids[:n], skip_special_tokens=True)
            uncertainty = sum(steps) / len(steps) if steps else math.inf
            prompt = int(lengths[b])
            results.append(ExpertOutput(clean_answer(text), prompt + n, uncertainty, prompt, n, steps))
        return results

    def answer_batch(self, questions: list[str], contexts: list[list[str]]) -> list[ExpertOutput]:
        prompts = [self._prompt(q, c) for q, c in zip(questions, contexts)]
        results = []
        for i in range(0, len(prompts), self.batch_size):
            results += self._generate(prompts[i: i + self.batch_size])
        return results

    def answer(self, question: str, paragraphs: list[str]) -> ExpertOutput:
        return self.answer_batch([question], [paragraphs])[0]


class VLLMExpert:
    """Greedy batched generation with vLLM.

    Uncertainty is the mean entropy of the renormalised top-`logprobs` distribution at each
    generated token, an approximation of HFExpert's full-vocabulary entropy, so the entropy
    baseline is not comparable across backends: keep one backend per run.
    """

    def __init__(self, name: str, top_k: int | None, model: str, dtype: str = "auto",
                 quantization: str | None = None, max_model_len: int = 8192,
                 gpu_memory_utilization: float = 0.85, max_new_tokens: int = 16, logprobs: int = 20,
                 revision: str | None = None):
        from vllm import LLM, SamplingParams

        self.name, self.top_k = name, top_k
        self.max_new_tokens, self.max_model_len = max_new_tokens, max_model_len
        pin = {"revision": revision, "tokenizer_revision": revision} if revision else {}
        self.llm = LLM(model=model, dtype=dtype, quantization=quantization, max_model_len=max_model_len,
                       gpu_memory_utilization=gpu_memory_utilization, **pin)
        self.tokenizer = self.llm.get_tokenizer()
        self.params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens, logprobs=logprobs,
                                     repetition_penalty=1.0)

    def _prompt(self, question: str, paragraphs: list[str]) -> str:
        text = PROMPT.format(context="\n\n".join(paragraphs), question=question)
        if self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False,
                                                      add_generation_prompt=True)
        return text

    def answer_batch(self, questions: list[str], contexts: list[list[str]]) -> list[ExpertOutput]:
        prompts = [self._prompt(q, c) for q, c in zip(questions, contexts)]
        limit = self.max_model_len - self.max_new_tokens
        too_long = [i for i, p in enumerate(prompts) if len(self.tokenizer.encode(p)) > limit]
        if too_long:
            raise ValueError(f"{len(too_long)} prompts exceed {limit} tokens (batch positions {too_long[:5]}); "
                             "raise max_model_len rather than truncating evidence")

        results = []
        for out in self.llm.generate(prompts, self.params, use_tqdm=False):
            completion = out.outputs[0]
            entropies = []
            for step in completion.logprobs or []:
                p = np.exp(np.array([lp.logprob for lp in step.values()]))
                p /= p.sum()
                entropies.append(float(-(p * np.log(p + 1e-12)).sum()))
            uncertainty = sum(entropies) / len(entropies) if entropies else math.inf
            prompt, generated = len(out.prompt_token_ids), len(completion.token_ids)
            results.append(ExpertOutput(clean_answer(completion.text), prompt + generated, uncertainty, prompt,
                                        generated, entropies))
        return results

    def answer(self, question: str, paragraphs: list[str]) -> ExpertOutput:
        return self.answer_batch([question], [paragraphs])[0]


def answer_batch(expert, questions: list[str], contexts: list[list[str]]) -> list[ExpertOutput]:
    if hasattr(expert, "answer_batch"):
        return expert.answer_batch(questions, contexts)
    return [expert.answer(q, c) for q, c in zip(questions, contexts)]


def make_expert(cfg: dict):
    kind = cfg.get("kind", "heuristic")
    top_k = cfg.get("top_k")
    if kind == "heuristic":
        return HeuristicExpert(cfg["name"], top_k, hops=cfg.get("hops", 0))
    if kind == "hf":
        options = ("device", "max_new_tokens", "dtype", "batch_size", "load_in_4bit", "revision")
        return HFExpert(cfg["name"], top_k, cfg["model"], **{k: cfg[k] for k in options if k in cfg})
    if kind == "vllm":
        options = ("dtype", "quantization", "max_model_len", "gpu_memory_utilization", "max_new_tokens", "logprobs",
                   "revision")
        return VLLMExpert(cfg["name"], top_k, cfg["model"], **{k: cfg[k] for k in options if k in cfg})
    raise ValueError(f"unknown expert kind: {kind}")
