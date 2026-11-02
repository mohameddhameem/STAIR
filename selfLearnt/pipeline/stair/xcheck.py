"""Cross-check (configs/xcheck_old500.yaml): golden_v1's fine-tuned models with AgentTTS's passages on AgentTTS's 500 test
questions, next to the same pairs in the first run (AL-04 / AL-05: first-run models, our passages, same 500) and on the
golden set (AL-13 / AL-14: golden models, AgentTTS's passages, slr-test). Holds the questions fixed, so it separates the
questions from the passages and models. CPU only, reads saved outputs."""
import os

import pandas as pd

from .common import key, log, read_jsonl, write_json
from .states import f_ans, f_ret

PAIRS = (("q08:ft", "l8:ft", "AL-05", "AL-14"), ("q7:ft", "l8:ft", "AL-04", "AL-13"))


def _ref(cfg, exp, split, pol):
    d = cfg["paths"]["results_dir"]
    for base in (d, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "common")):
        p = os.path.join(base, f"{exp}-hotpotqa-{split}-seed0.csv")
        if os.path.exists(p):
            df = pd.read_csv(p)
            df = df[df.policy == pol]
            return f"{df.lenient.mean():.3f} @ {df.units.mean():.0f}"
    return "n/a"


def compare(cfg):
    A = {}
    for r in read_jsonl(f_ans(cfg, "l8:ft")):
        A[(r["qid"], key(r["sel"]))] = r
    out = {}
    log("pair                          | this check (golden models, AgentTTS passages, old 500) | first run (old 500) | golden set (slr-test)")
    for rv, av, old_exp, new_exp in PAIRS:
        R = read_jsonl(f_ret(cfg, "test", rv))
        rows = [(r, A.get((r["qid"], key(r["sel"])))) for r in R]
        rows = [(r, a) for r, a in rows if a]
        if not rows:
            log(f"{rv} -> {av}: no answers yet")
            continue
        acc = sum(a["lenient"] for _, a in rows) / len(rows)
        u = sum(r["units"] + a["units"] for r, a in rows) / len(rows)
        both = sum(r["found_both"] for r, _ in rows) / len(rows)
        pol = f"fixed:{rv}->{av}"
        log(f"{pol:29s} | {acc:.3f} @ {u:.0f} on {len(rows)} (both gold passages picked: {100 * both:.0f}%) | "
            f"{_ref(cfg, old_exp, 'test', pol)} ({old_exp}) | {_ref(cfg, new_exp, 'slr-test', pol)} ({new_exp})")
        out[pol] = {"acc": acc, "units": u, "n": len(rows), "found_both": both}
    write_json(os.path.join(cfg["run_dir"], "xcheck.json"), out)
