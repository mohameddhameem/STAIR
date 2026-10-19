#!/usr/bin/env python3
# feedback_analysis.py v1.0 — cell-vs-cell verdict for feedback escalation (23 Sep 2026)
# VAL-ONLY, on the exact pilot instances. Compares, per instance:
#   plain SS answer  (qa_strong.jsonl, the grid's escalation endpoint)
#   feedback answer  (feedback_pilot.jsonl)
# under the calibrated whole-word lenient metric (yes/no guard, any-gold) — the variant
# that matched the grid 2591/2591. Also cross-checks em(plain SS) against the grid's
# stored em_lenient[SS] on these instances (must be ~100% or nothing is trusted).
# Prints: both cell rates, the delta, the flip table (fixed vs broken), and the
# composed projection if the delta were applied to the champion's escalated mass.
import json, re, string, collections

DATA  = "/mnt/hdd/jonathan/data/tts_test.json"
GRID  = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
SSANS = "/mnt/hdd/jonathan/runs/full_run_p100/qa_strong.jsonl"
FB    = "/mnt/hdd/jonathan/runs/verifier_scores/feedback_pilot.jsonl"

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
gem = {}
for l in open(GRID):
    r = json.loads(l); gem[r["idx"]] = r["em_lenient"]
ss = {}
for l in open(SSANS):
    r = json.loads(l); i, br = r["k"].split("|")
    if br == "strong": ss[int(i)] = (r.get("v") or "").strip()
fb = {}
for l in open(FB):
    r = json.loads(l); fb[int(r["k"].split("|")[0])] = (r["v"] or "").strip()

ids = sorted(fb.keys() & ss.keys())
SS_CELL = 3
cal = sum(em(ss[i], golds[i]) == gem[i][SS_CELL] for i in ids)
print(f"[pilot instances] {len(ids)}   [calibration em(plain SS) vs grid] {cal}/{len(ids)}")
assert cal >= 0.99*len(ids), "calibration below 99% — do not trust"

n = len(ids)
r_ss = sum(em(ss[i], golds[i]) for i in ids)
r_fb = sum(em(fb[i], golds[i]) for i in ids)
fixed  = sum(1 for i in ids if em(fb[i], golds[i]) == 1 and em(ss[i], golds[i]) == 0)
broken = sum(1 for i in ids if em(fb[i], golds[i]) == 0 and em(ss[i], golds[i]) == 1)
print(f"\n[CELL RATES on the escalated pool, n={n}]")
print(f"  plain SS answer     : {100*r_ss/n:.2f}")
print(f"  feedback SS answer  : {100*r_fb/n:.2f}   (delta {100*(r_fb-r_ss)/n:+.2f})")
print(f"  flips — feedback FIXED {fixed} | feedback BROKE {broken}")
esc_share = 0.755   # champion's escalated mass on test (1 - 24.5% kept)
print(f"\n[composed projection] champion 53.41 + delta x escalated-share({esc_share:.0%}) ="
      f" {53.41 + (100*(r_fb-r_ss)/n)*esc_share:.2f}   (bar: SS 53.54)")
print("pre-registered: delta >= +1.5 -> full run justified; ~0 or negative -> lever closed.")
