#!/usr/bin/env python3
# make_sbok_summary.py v1.0 — strong-side best-of-4 summary for the FINAL experiment (23 Sep 2026)
# For every escalated-pool instance (val+test) with complete data: candidates =
# {greedy SS answer (score from verifier_scores_strongans), t1..t3 (scores from
# pilot_strongbok_ver_t*)}; pick the highest-verified; emit
# {idx, split, sbok_em (= calibrated lenient correctness of the picked answer)}.
# Calibration hard-asserted on plain-SS answers vs the grid's SS cell (>=99%).
import json, re, string, collections

DATA  = "/mnt/hdd/jonathan/data/tts_test.json"
GRID  = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
SSANS = "/mnt/hdd/jonathan/runs/full_run_p100/qa_strong.jsonl"
SB    = "/mnt/hdd/jonathan/runs/verifier_scores/pilot_strongbok.jsonl"
VSS   = "/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores_strongans.jsonl"
VT    = "/mnt/hdd/jonathan/runs/verifier_scores/pilot_strongbok_ver_{b}.jsonl"
OUT   = "/mnt/hdd/jonathan/runs/verifier_scores/sbok_summary.jsonl"

def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em(pred, golds):
    np_ = norm(pred)
    for g in golds:
        ng = norm(g)
        if (np_ in ("yes","no","noanswer") or ng in ("yes","no","noanswer")) and np_ != ng: continue
        if ng and re.search(r"(?<!\w)"+re.escape(ng)+r"(?!\w)", np_): return 1
    return 0

data = json.load(open(DATA))
golds = {i: (data[i]["answers"] if isinstance(data[i]["answers"], list) else [data[i]["answers"]]) for i in range(len(data))}
grid = {}
for l in open(GRID):
    r = json.loads(l); grid[r["idx"]] = (r["split"], r["em_lenient"])
ss = {}
for l in open(SSANS):
    r = json.loads(l); i, br = r["k"].split("|")
    if br == "strong": ss[int(i)] = (r.get("v") or "").strip()
sb = collections.defaultdict(dict)
for l in open(SB):
    r = json.loads(l); i, s = r["k"].split("|"); sb[int(i)][s] = (r["v"] or "").strip()
vs = {json.loads(l)["idx"]: json.loads(l)["p_supported"] for l in open(VSS)}
vt = {b: {json.loads(l)["idx"]: json.loads(l)["p_supported"] for l in open(VT.format(b=b))} for b in ("t1","t2","t3")}

ids = sorted(i for i in sb if len(sb[i]) == 3 and i in vs and i in ss and all(i in vt[b] for b in vt))
cal = sum(em(ss[i], golds[i]) == grid[i][1][3] for i in ids)
print(f"[join] complete escalated instances: {len(ids)}   [calibration plain-SS vs grid] {cal}/{len(ids)}")
assert cal >= 0.99*len(ids), "calibration below 99% — investigate before trusting"

with open(OUT, "w") as f:
    for i in ids:
        cand = [(ss[i], vs[i])] + [(sb[i][b], vt[b][i]) for b in ("t1","t2","t3")]
        pick = max(cand, key=lambda c: c[1])[0]
        f.write(json.dumps({"idx": i, "split": grid[i][0], "sbok_em": em(pick, golds[i])}) + "\n")
by = collections.Counter(grid[i][0] for i in ids)
print(f"[done] wrote {len(ids)} rows -> {OUT}   coverage {dict(by)}")
