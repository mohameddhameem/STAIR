#!/usr/bin/env python3
# bestofk_analysis.py v1.0 — pilot verdict for best-of-K weak sampling (23 Sep 2026)
# VAL-ONLY (500 pilot instances; sealed test untouched).
# Joins: sampled answers (s0..s3) + verifier scores (s0 from the teacher-1 file;
# s1..s3 from bestofk_ver_*) + gold answers (tts_test.json 'answers' list, any-match).
# METRIC CALIBRATION: computes lenient under both variants (substring / whole-word,
# each with the HotpotQA yes/no guard) on the s0 answers and compares against the
# grid's stored em_lenient[SW]; uses whichever variant agrees best and PRINTS the
# agreement — if neither reaches ~99%, the verdict is not trustworthy and says so.
# Verdict table: s0 rate | majority-of-4 | best-of-4-by-verifier | oracle-of-4.
import json, re, string, collections

GRID   = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
DATA   = "/mnt/hdd/jonathan/data/tts_test.json"
PILOT  = "/mnt/hdd/jonathan/runs/verifier_scores/bestofk_pilot.jsonl"
VER_S0 = "/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl"
VER_S  = "/mnt/hdd/jonathan/runs/verifier_scores/bestofk_ver_{b}.jsonl"

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def guard(np_, ng):  # HotpotQA yes/no special case
    return (np_ in ("yes","no","noanswer") or ng in ("yes","no","noanswer")) and np_ != ng

def em_sub(pred, golds):
    np_ = norm(pred)
    for g in golds:
        ng = norm(g)
        if guard(np_, ng): continue
        if ng and ng in np_: return 1
    return 0

def em_word(pred, golds):
    np_ = norm(pred)
    for g in golds:
        ng = norm(g)
        if guard(np_, ng): continue
        if ng and re.search(r"(?<!\w)"+re.escape(ng)+r"(?!\w)", np_): return 1
    return 0

data = json.load(open(DATA))
golds = {i: (data[i]["answers"] if isinstance(data[i]["answers"], list) else [data[i]["answers"]]) for i in range(len(data))}
grid_em = {}
for l in open(GRID):
    r = json.loads(l); grid_em[r["idx"]] = r["em_lenient"]
ans = collections.defaultdict(dict)
for l in open(PILOT):
    r = json.loads(l); i, s = r["k"].split("|"); ans[int(i)][s] = (r["v"] or "").strip()
ver = collections.defaultdict(dict)
for l in open(VER_S0):
    r = json.loads(l); ver[r["idx"]]["s0"] = r["p_supported"]
for b in ("s1","s2","s3"):
    for l in open(VER_S.format(b=b)):
        r = json.loads(l); ver[r["idx"]][b] = r["p_supported"]

ids = [i for i in ans if all(f"s{j}" in ans[i] for j in range(4)) and all(f"s{j}" in ver.get(i, {}) for j in range(4))]
print(f"[join] complete instances: {len(ids)} (need samples s0..s3 + all 4 verifier scores)")

# ---- calibration on s0 vs the grid's SW cell ----
SW = 2
cal = {"substring": 0, "whole-word": 0}
for i in ids:
    cal["substring"] += (em_sub(ans[i]["s0"], golds[i]) == grid_em[i][SW])
    cal["whole-word"] += (em_word(ans[i]["s0"], golds[i]) == grid_em[i][SW])
best = max(cal, key=cal.get)
em = em_sub if best == "substring" else em_word
print(f"[metric calibration vs grid em_lenient[SW] on s0] substring {cal['substring']}/{len(ids)}  whole-word {cal['whole-word']}/{len(ids)}  -> using {best}")
if cal[best] < 0.99 * len(ids):
    print("!! WARNING: neither variant reaches ~99% agreement — verdict below is NOT trustworthy; investigate before using.")

# ---- verdict ----
n = len(ids)
r_s0   = sum(em(ans[i]["s0"], golds[i]) for i in ids)
r_maj  = 0; r_bok = 0; r_orc = 0
for i in ids:
    ems = {s: em(ans[i][s], golds[i]) for s in ("s0","s1","s2","s3")}
    votes = collections.Counter(norm(ans[i][s]) for s in ("s0","s1","s2","s3"))
    top = votes.most_common(1)[0][0]
    pick_maj = next(s for s in ("s0","s1","s2","s3") if norm(ans[i][s]) == top)
    r_maj += ems[pick_maj]
    pick_bok = max(("s0","s1","s2","s3"), key=lambda s: ver[i][s])
    r_bok += ems[pick_bok]
    r_orc += max(ems.values())
print(f"\n[PILOT VERDICT — {n} val instances, weak answerer on strong evidence]")
print(f"  single greedy (the grid's SW cell) : {100*r_s0/n:.2f}")
print(f"  majority vote of 4                 : {100*r_maj/n:.2f}")
print(f"  best-of-4 by verifier score        : {100*r_bok/n:.2f}")
print(f"  oracle of 4 (diversity ceiling)    : {100*r_orc/n:.2f}")
print(f"\nreading: oracle-minus-greedy = harvestable mass; best-of-4-minus-greedy = what the")
print(f"verifier actually harvests. A best-of-4 lift of +2 or more justifies the full run;")
print(f"a flat best-of-4 with a high oracle means selection (not sampling) is the bottleneck.")
