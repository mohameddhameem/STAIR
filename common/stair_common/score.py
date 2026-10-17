"""The team's answer scores. Every accuracy is named by its ruler; never mix rulers in one table.

- `lenient`: THE STANDARD (headline). `lenient_match_agenttts.py` next to this file is a byte-for-byte copy of
  jonathan/final/lenient_match_agenttts.py on branch `jonathan` (git blob 846e780, sha256 below; a test checks it),
  used unchanged (certified: reproduces AgentTTS's published 0.746). Never re-implement or edit it; if the original
  changes, copy it again and update RULER_SHA256.
- `em`, `f1`: official HotpotQA answer scoring (hotpot_evaluate_v1.py normalisation and yes/no rule). Secondary.

A question has one or more gold answers (aliases); EM and F1 take the best over them, as `lenient` does.
"""

import importlib.util
import re
import string
from collections import Counter
from functools import cache
from pathlib import Path

RULER = Path(__file__).resolve().parent / "lenient_match_agenttts.py"
RULER_SHA256 = "2f184ac252a26df49d402f22f9ba52af3db6ed4812adb42507d3bbb989a0c882"
_NO_PARTIAL_CREDIT = {"yes", "no", "noanswer"}


@cache
def _ruler():
    spec = importlib.util.spec_from_file_location("lenient_match_agenttts", RULER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _golds(gold: str | list[str]) -> list[str]:
    return [gold] if isinstance(gold, str) else list(gold)


def lenient(answer: str, gold: str | list[str]) -> int:
    """1 if the answer matches any gold under the standard ruler (lenient_match_agenttts.LEN), else 0."""
    return int(_ruler().LEN(answer, _golds(gold)))


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def _f1(prediction: str, gold: str) -> float:
    pred, ref = normalize_answer(prediction), normalize_answer(gold)
    if (pred in _NO_PARTIAL_CREDIT or ref in _NO_PARTIAL_CREDIT) and pred != ref:
        return 0.0
    num_same = sum((Counter(pred.split()) & Counter(ref.split())).values())
    if num_same == 0:
        return 0.0
    precision, recall = num_same / len(pred.split()), num_same / len(ref.split())
    return 2 * precision * recall / (precision + recall)


def em(answer: str, gold: str | list[str]) -> float:
    return max(float(normalize_answer(answer) == normalize_answer(g)) for g in _golds(gold))


def f1(answer: str, gold: str | list[str]) -> float:
    return max(_f1(answer, g) for g in _golds(gold))


def score(answer: str, gold: str | list[str]) -> dict[str, float]:
    """All three scores of one answer: {"lenient", "em", "f1"}."""
    return {"lenient": lenient(answer, gold), "em": em(answer, gold), "f1": f1(answer, gold)}
