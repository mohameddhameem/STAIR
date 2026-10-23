"""Review-validated reconstruction of the reference lenient answer metric.

The original scoring source is unavailable. This rule agrees with all 4,267
available unique, untruncated original answer-review judgments (train/val),
but is not certified on unobserved original cases. Never call it exact match.
"""
import re
import string
import unicodedata


def gold_list(golds):
    return [golds] if isinstance(golds,str) else list(golds)


def normalize_lenient(text):
    text=''.join(c for c in str(text).lower() if not unicodedata.category(c).startswith('P'))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def normalize_strict(text):
    text=''.join(c for c in str(text).lower() if c not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',text).split())


def strict_match(prediction,golds):
    p=normalize_strict(prediction)
    return int(any(p==normalize_strict(g) for g in gold_list(golds)))


def lenient_match(prediction,golds):
    p=normalize_lenient(prediction)
    for gold in gold_list(golds):
        g=normalize_lenient(gold)
        if p==g:return 1
        if g in ['yes','no'] and {'yes','no'}<=set(p.split()):continue
        if g and re.search(r'(?<!\w)'+re.escape(g)+r'(?!\w)',p):return 1
    return 0


def score(prediction,golds):
    return strict_match(prediction,golds),lenient_match(prediction,golds)
