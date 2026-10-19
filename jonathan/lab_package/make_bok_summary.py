#!/usr/bin/env python3
# make_bok_summary.py v1.1 (also emits picked-answer store for the comparative scorer) — collapse best-of-4 samples+scores into one row per instance (23 Sep 2026)
# For each val/test instance with complete data: pick the sample with the HIGHEST verifier
# score; emit {idx, split, bok_score (=max score), bok_em (=lenient correctness of that
# picked sample)}. Metric = whole-word lenient with yes/no guard, any-gold — the variant
# that calibrated 500/500 against the grid; calibration is re-run here on all s0 rows and
# printed (must stay ~100% or the output is not written).
import json, re, string, collections, argparse

def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em_word(pred, golds):
    np_ = norm(pred)
    for g in golds:
        ng = norm(g)
        if (np_ in ("yes","no","noanswer") or ng in ("yes","no","noanswer")) and np_ != ng: continue
        if ng and re.search(r"(?<!\w)"+re.escape(ng)+r"(?!\w)", np_): return 1
    return 0

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/mnt/hdd/jonathan/data/tts_test.json")
ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
ap.add_argument("--pilot", default="/mnt/hdd/jonathan/runs/verifier_scores/bestofk_pilot.jsonl")
ap.add_argument("--ver_s0", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl")
ap.add_argument("--ver_tpl", default="/mnt/hdd/jonathan/runs/verifier_scores/bestofk_ver_{b}.jsonl")
ap.add_argument("--out", default="/mnt/hdd/jonathan/runs/verifier_scores/bok_summary.jsonl")
ap.add_argument("--answers_out", default="/mnt/hdd/jonathan/runs/verifier_scores/bok_answers.jsonl",
                help="also write the PICKED sample's answer text per idx, Store format k='idx|strong' (input for the comparative scorer)")
a = ap.parse_args()

data = json.load(open(a.data))
golds = {i: (data[i]["answers"] if isinstance(data[i]["answers"], list) else [data[i]["answers"]]) for i in range(len(data))}
grid = {}
for l in open(a.grid):
    r = json.loads(l); grid[r["idx"]] = (r["split"], r["em_lenient"])
ans = collections.defaultdict(dict)
for l in open(a.pilot):
    r = json.loads(l); i, s = r["k"].split("|"); ans[int(i)][s] = (r["v"] or "").strip()
ver = collections.defaultdict(dict)
for l in open(a.ver_s0):
    r = json.loads(l); ver[r["idx"]]["s0"] = r["p_supported"]
for b in ("s1","s2","s3"):
    for l in open(a.ver_tpl.format(b=b)):
        r = json.loads(l); ver[r["idx"]][b] = r["p_supported"]

ids = sorted(i for i in ans if all(f"s{j}" in ans[i] for j in range(4)) and all(f"s{j}" in ver.get(i, {}) for j in range(4)))
SW = 2
cal = sum(em_word(ans[i]["s0"], golds[i]) == grid[i][1][SW] for i in ids)
print(f"[join] complete: {len(ids)}   [calibration s0 vs grid SW] {cal}/{len(ids)}")
assert cal >= 0.99*len(ids), "metric calibration below 99% — do not trust; investigate first"

with open(a.out, "w") as f, open(a.answers_out, "w") as fa:
    for i in ids:
        pick = max(("s0","s1","s2","s3"), key=lambda s: ver[i][s])
        f.write(json.dumps({"idx": i, "split": grid[i][0],
                            "bok_score": ver[i][pick],
                            "bok_em": em_word(ans[i][pick], golds[i])}) + "\n")
        fa.write(json.dumps({"k": f"{i}|strong", "v": ans[i][pick]}) + "\n")
print(f"[done] wrote {len(ids)} rows -> {a.out}  (+ picked answers -> {a.answers_out})")
by = collections.Counter((grid[i][0]) for i in ids)
print(f"[coverage] {dict(by)}")
