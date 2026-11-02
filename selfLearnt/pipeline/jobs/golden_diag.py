"""Why do our fine-tuned pairs score lower on the golden questions? Per question set: accuracy of two fixed pairs, how
often the retriever picked both gold passages, and accuracy by question type (bridge / comparison), next to the team's
always-strong rows on the same questions. CPU only, reads saved outputs, a few seconds.
  bash -c 'source jobs/cluster.env; source jobs/cluster.local.env; python jobs/golden_diag.py'"""
import json
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stair.common import key, read_jsonl          # noqa: E402

RUNS = os.path.expanduser(os.environ.get("STAIR_RUNS_DIR", "~/stair-runs"))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
SETS = [("full_v1", "val", "first run, dev[0:500] (old val)"), ("full_v1", "test", "first run, old 500 test"),
        ("xcheck_old500", "test", "golden models + AgentTTS passages, old 500"),
        ("golden_tts", "val", "golden, slr-val (1,110)"), ("golden_tts", "test", "golden, slr-test (1,481)")]
PAIRS = (("q08:ft", "l8:ft"), ("q7:ft", "l8:ft"))
SS = {"test": "jonathan-ss-hotpotqa-test.csv", "val": "jonathan-ss-hotpotqa-val.csv"}
SS_G = {"test": "jonathan-ss-hotpotqa-slr-test.csv", "val": "jonathan-ss-hotpotqa-slr-val.csv"}


def fn(v):
    return v.replace(":", "-")


for run, sp, label in SETS:
    d = os.path.join(RUNS, run, "seed0", "step")
    dq = os.path.join(RUNS, run, "data", f"{sp}.json")
    if not os.path.exists(dq):
        print(f"\n{label}: no data")
        continue
    qs = {q["qid"]: q for q in json.load(open(dq))}
    comp = sum(q.get("type") == "comparison" for q in qs.values()) / len(qs)
    print(f"\n=== {label}: {len(qs)} questions, {100 * comp:.0f}% comparison")
    for rv, av in PAIRS:
        R = read_jsonl(os.path.join(d, f"{sp}_ret_{fn(rv)}.jsonl"))
        A = {(r["qid"], key(r["sel"])): r for r in read_jsonl(os.path.join(d, f"answers_{fn(av)}.jsonl"))}
        rows = [(r, A.get((r["qid"], key(r["sel"])))) for r in R if r["qid"] in qs]
        rows = [(r, a) for r, a in rows if a]
        if not rows:
            print(f"  {rv} -> {av}: no outputs")
            continue
        acc = sum(a["lenient"] for _, a in rows) / len(rows)
        fb = sum(r["found_both"] for r, _ in rows) / len(rows)
        okfb = [a["lenient"] for r, a in rows if r["found_both"]]
        by = {}
        for r, a in rows:
            by.setdefault(qs[r["qid"]].get("type"), []).append(a["lenient"])
        typ = ", ".join(f"{t} {sum(v) / len(v):.3f}" for t, v in sorted(by.items(), key=lambda x: str(x[0])))
        print(f"  {rv:7s}-> {av:6s} acc {acc:.3f} | both gold passages picked {100 * fb:.0f}% (acc then "
              f"{sum(okfb) / max(1, len(okfb)):.3f}) | by type: {typ}")
    ssf = (SS_G if run == "golden_tts" else SS)[sp]
    p = os.path.join(REPO, "results", "common", ssf)
    if os.path.exists(p):
        s = pd.read_csv(p)
        s = s[s.id.isin(qs)]
        s["type"] = s.id.map(lambda i: qs[i].get("type"))
        print(f"  always-strong (team rows) acc {s.lenient.mean():.3f} on {len(s)} | by type: "
              + ", ".join(f"{t} {g.lenient.mean():.3f}" for t, g in s.groupby("type")))
