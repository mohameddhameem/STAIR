"""nano-jev as a retriever rung (AL-05): the cross-encoder scorer from STAIR's systemOne track, used unchanged.

Copied from branch systemOne, systemOne/src/stair/evidence.py (commit f986d6b), class CrossEncoderScorer and
resolve_checkpoint, so that every track ranks passages with the same code. Only the module header differs.
"""
import json
from pathlib import Path

import numpy as np


def passages_state(paragraphs: list[str]) -> str:
    """Several paragraphs as one state, in nano-jev's `format_passages` layout."""
    return "\n".join(f"[{i + 1}] {p}" for i, p in enumerate(paragraphs))


def resolve_checkpoint(path: str) -> Path:
    """A local folder as is; otherwise a Hub id with optional "@revision", downloaded whole."""
    if Path(path).is_dir():
        return Path(path)
    from huggingface_hub import snapshot_download

    repo_id, _, revision = path.partition("@")
    return Path(snapshot_download(repo_id, revision=revision or None))


class CrossEncoderScorer:
    """nano-jev cross-encoder: one logit per (question + option, context) pair."""

    name = "cross-encoder"
    RELEVANCE = ("How relevant is this passage to the query: {q}", ["irrelevant", "partially relevant", "directly answers"])
    SUFFICIENT = ("Does the context contain enough information to answer: {q}", ["yes", "no"])

    def __init__(self, path: str, device: str | None = None, max_length: int = 512):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        path = resolve_checkpoint(path)
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForSequenceClassification.from_pretrained(path).to(self.device).eval()
        self.max_length = max_length
        self.temperatures = {"relevance": 1.0, "sufficient": 1.0}
        calib = path / "calibration.json"
        if calib.exists():
            self.temperatures.update(json.loads(calib.read_text()))
        self.truncated = 0
        self.pairs_seen = 0
        self.tokens_seen = 0  # model tokens processed, for the scorer's own cost

    def _decide(self, decision: str, question: str, states: list[str]) -> np.ndarray:
        """Return [len(states), num_options] temperature-scaled probabilities."""
        template, options = self.RELEVANCE if decision == "relevance" else self.SUFFICIENT
        text = template.format(q=question)
        firsts = [f"question: {text} option: {opt}" for _ in states for opt in options]
        seconds = [s for s in states for _ in options]
        enc = self.tokenizer(firsts, seconds, truncation="only_second", max_length=self.max_length, padding=True, return_tensors="pt")
        full_lengths = [len(ids) for ids in self.tokenizer(firsts, seconds)["input_ids"]]
        self.truncated += sum(n > self.max_length for n in full_lengths)
        self.pairs_seen += len(full_lengths)
        self.tokens_seen += sum(min(n, self.max_length) for n in full_lengths)
        with self.torch.no_grad():
            logits = self.model(**enc.to(self.device)).logits[:, 0].float().cpu().numpy()
        logits = logits.reshape(len(states), len(options)) / self.temperatures[decision]
        logits -= logits.max(axis=1, keepdims=True)
        probs = np.exp(logits)
        return probs / probs.sum(axis=1, keepdims=True)

    def score(self, question: str, paragraphs: list[str]) -> np.ndarray:
        """P(not irrelevant) per paragraph."""
        return 1.0 - self._decide("relevance", question, paragraphs)[:, 0]

    def sufficiency(self, question: str, paragraphs: list[str]) -> float:
        """P(yes) that the given paragraphs together are enough to answer."""
        return float(self._decide("sufficient", question, [passages_state(paragraphs)])[0, 0])
