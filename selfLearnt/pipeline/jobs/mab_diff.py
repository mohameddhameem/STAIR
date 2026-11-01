"""MedAgentBench acceptance diagnosis: the tasks where our Qwen 72B and Jonathan's disagree, with what our model asked the
records server and what it answered. Shows whether the differences are in what the model looked up (GET), in the value
it found, or only in how it wrote the answer. CPU only, reads saved episodes, a few seconds.
  bash -c 'source jobs/cluster.env; source jobs/cluster.local.env; python jobs/mab_diff.py [q72-greedy]'"""
import json
import os
import re
import subprocess
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stair.common import read_jsonl                     # noqa: E402
from stair.config import load_config                    # noqa: E402
from stair.medagent import ROOT, wdir                   # noqa: E402

run = sys.argv[1] if len(sys.argv) > 1 else "q72-greedy"
cfg = load_config(os.environ.get("CONFIG", "configs/default.yaml"))
his = {r["tid"]: r for r in json.loads(subprocess.run(
    ["git", "-C", ROOT, "show", "origin/jonathan:results/medagentbench/training1_table.json"],
    capture_output=True, text=True, check=True).stdout)}
ep = {e["tid"]: e for e in read_jsonl(os.path.join(wdir(cfg), f"episodes_{run}.jsonl"))}
gr = {g["tid"]: g for g in json.load(open(os.path.join(wdir(cfg), f"graded_{run}.json")))}


def shape(res):
    """The answer's form with digits replaced: '[2.1]' -> '[9.9]', '["2.1 mg/dL"]' -> '["9.9 mg/dL"]'."""
    return re.sub(r"\d+", "9", str(res))[:60]


def gets(e):
    return [re.sub(r"^GET\s+\S*?/fhir/", "", h["content"].strip().replace("```", "").strip())[:110]
            for h in e["history"] if h["role"] == "agent" and h["content"].strip().replace("```", "").strip().startswith("GET")]


print(f"run {run}: {len(ep)} episodes")
for ty in ("task4", "task5", "task6", "task10"):
    ids = sorted((t for t in gr if t.startswith(ty + "_")), key=lambda x: int(x.split("_")[1]))
    diff = [t for t in ids if gr[t]["correct"] != his[t]["qs"]]
    print(f"\n=== {ty}: ours {sum(gr[t]['correct'] for t in ids)}/30, his {sum(his[t]['qs'] for t in ids)}/30; "
          f"{len(diff)} differ")
    print("  question:", ep[ids[0]]["history"][0]["content"].split("Question:")[-1].strip()[:300])
    print("  our answer forms, right:", dict(Counter(shape(ep[t]["result"]) for t in ids if gr[t]["correct"]).most_common(4)))
    print("  our answer forms, wrong:", dict(Counter(shape(ep[t]["result"]) for t in ids if not gr[t]["correct"]).most_common(4)))
    for t in diff[:5]:
        e = ep[t]
        print(f"  {t}: ours {'right' if gr[t]['correct'] else 'WRONG'}, his {'right' if his[t]['qs'] else 'WRONG'}; "
              f"{e['rounds']} rounds, {e['status']}; answer {str(e['result'])[:80]}")
        for g in gets(e):
            print(f"      GET {g}")
        last = [h["content"] for h in e["history"] if h["role"] == "user"][-1]
        print(f"      last reply from the server ({len(last)} chars): {last[:160]!r}")
