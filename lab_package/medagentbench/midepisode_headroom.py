# midepisode_headroom.py — Step A of the mid-episode escalation chapter (replay-only).
# QUESTION: how much of the weak model's doomed-episode cost could an earlier
# escalation decision recover, holding the Training 1 router's DECISIONS fixed?
# Restart semantics: escalation re-runs the strong model from scratch (its full
# episode is recorded), so cutting the weak episode at round k changes only the
# weak-side cost — fully computable from stored transcripts. UPPER BOUND caveat:
# assumes the same decisions would be made from a partial attempt; decision
# quality at round k is what a trained round-k router would have to demonstrate.
# Read-only. No GPU. Writes runs/mab/weak_costs_by_round.csv.
# Usage (lab): /mnt/hdd/jonathan/slr-env/bin/python midepisode_headroom.py
import json, statistics as st
from transformers import AutoTokenizer
from stair_common import cost as C

MAB   = "/mnt/hdd/jonathan/medagentbench"
WEAK  = f"{MAB}/outputs/WeakPilot7B_run1/qwen2.5-7b-local/medagentbench-std/overall.json"
TABLE = "/mnt/hdd/jonathan/runs/mab/training1_table.json"
VAL_S = "/mnt/hdd/jonathan/runs/mab/training1_router/val_scores.jsonl"
TEST_S= "/mnt/hdd/jonathan/runs/mab/training1_router/test_scores.jsonl"
OUT   = "/mnt/hdd/jonathan/runs/mab/weak_costs_by_round.csv"
THETA = 0.50   # OP-A, the declared primary operating point

tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
W = {r["index"]: r for r in json.load(open(WEAK))["custom"]["raw_results"]}
table = {t["idx"]: t for t in json.load(open(TABLE))}

# cumulative weak cost after each round, per instance (same tokenization as the
# cost store: cumulative-input convention, served model's tokenizer, params 7.0)
by_round = {}
for idx, rec in W.items():
    ctx = ""; n_in = n_out = 0; cum = []
    for m in rec["history"]:
        if m["role"] == "agent":
            n_in += len(tok.encode(ctx)); n_out += len(tok.encode(m["content"]))
            cum.append(round(C.units(7.0, n_in, n_out), 2))
        ctx += m["content"]
    by_round[idx] = cum
with open(OUT, "w") as f:
    f.write("tid,rounds," + ",".join(f"cum_units_r{k}" for k in range(1, 9)) + "\n")
    for idx in sorted(by_round):
        cum = by_round[idx]
        padded = [str(cum[k-1]) if k <= len(cum) else "" for k in range(1, 9)]
        f.write(f"{table[idx]['tid']},{len(cum)}," + ",".join(padded) + "\n")
print(f"wrote per-round cost store -> {OUT}")

# decisions of the Training 1 router at OP-A (val 60 + test 150; train stays out)
dec = {}
for path in (VAL_S, TEST_S):
    for line in open(path):
        s = json.loads(line)
        dec[s["idx"]] = (s["pDeliver"] >= THETA)   # True = deliver
rows = [table[i] for i in dec]
print(f"instances re-priced: {len(rows)} (val+test), decisions = Training 1 OP-A, held fixed")

def price(k=None):
    """Total cost of the OP-A policy; escalations cut the weak episode at round
    k (None = after the full episode, the current architecture)."""
    tot = 0.0
    for t in rows:
        cum = by_round[t["idx"]]
        if dec[t["idx"]]:
            tot += t["cw"]                        # deliver: full weak episode, unchanged
        else:
            cut = t["cw"] if k is None else (cum[min(k, len(cum)) - 1])
            tot += cut + t["cs"]                  # escalate: partial weak + full strong
    return tot

base = price(None)
ss   = sum(t["cw"] + t["cs"] for t in rows)       # always-escalate (full weak + strong)
acc  = sum((t["qw"] if dec[t["idx"]] else t["qs"]) for t in rows)
print(f"\ncurrent architecture (decide after the full weak attempt): "
      f"acc {100*acc/len(rows):.2f} | cost {100*base/ss:.1f}% of always-escalate")
print(f"\n{'escalate at':>12s} {'cost %SS':>9s} {'saving vs current':>18s}")
for k in (1, 2, 3, 4, 6, 8):
    p = price(k)
    print(f"{'round ' + str(k):>12s} {100*p/ss:9.1f} {100*(base-p)/ss:17.1f}pp")

# where the waste lives: mean rounds + cost of escalated instances by type
import collections
esc = collections.defaultdict(list)
for t in rows:
    if not dec[t["idx"]]:
        esc[t["tid"].split("_")[0]].append(t)
print(f"\nescalated instances by type (mean weak rounds / mean weak units burned):")
for ty in sorted(esc, key=lambda s: int(s[4:])):
    g = esc[ty]
    print(f"  {ty:8s} n={len(g):3d}  rounds {st.mean(len(by_round[t['idx']]) for t in g):4.1f}  "
          f"weak units {st.mean(t['cw'] for t in g):7.1f}")
