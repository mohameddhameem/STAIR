#!/usr/bin/env python3
# pilot2_analysis.py v1.0 — verdicts for the union and strong-best-of-K pilots (23 Sep 2026)
# VAL-ONLY, cell-vs-cell on each pilot's exact instances, calibrated metric
# (whole-word lenient + yes/no guard + any-gold; asserted vs the grid on plain SS).
# UNION: em(union answer) vs em(plain SS answer).
# STRONGBOK: best-of-4 = argmax verifier score over {greedy SS (scores from
# verifier_scores_strongans.jsonl), t1..t3 (scores from pilot_strongbok_ver_t*.jsonl)};
# em(picked) vs em(plain SS).
import json, re, string

DATA  = "/mnt/hdd/jonathan/data/tts_test.json"
GRID  = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
SSANS = "/mnt/hdd/jonathan/runs/full_run_p100/qa_strong.jsonl"
UN    = "/mnt/hdd/jonathan/runs/verifier_scores/pilot_union.jsonl"
SB    = "/mnt/hdd/jonathan/runs/verifier_scores/pilot_strongbok.jsonl"
VSS   = "/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores_strongans.jsonl"
VT    = "/mnt/hdd/jonathan/runs/verifier_scores/pilot_strongbok_ver_{b}.jsonl"

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
gem = {json.loads(l)["idx"]: json.loads(l)["em_lenient"] for l in open(GRID)}
ss = {}
for l in open(SSANS):
    r = json.loads(l); i, br = r["k"].split("|")
    if br == "strong": ss[int(i)] = (r.get("v") or "").strip()

def verdict(name, picked, ids):
    n = len(ids)
    cal = sum(em(ss[i], golds[i]) == gem[i][3] for i in ids)
    r_ss = sum(em(ss[i], golds[i]) for i in ids)
    r_x  = sum(em(picked[i], golds[i]) for i in ids)
    fx = sum(1 for i in ids if em(picked[i], golds[i]) == 1 and em(ss[i], golds[i]) == 0)
    br = sum(1 for i in ids if em(picked[i], golds[i]) == 0 and em(ss[i], golds[i]) == 1)
    d = 100*(r_x - r_ss)/n
    print(f"\n[{name}] n={n}  calibration {cal}/{n}")
    print(f"  plain SS : {100*r_ss/n:.2f}   {name}: {100*r_x/n:.2f}   delta {d:+.2f}   FIXED {fx} | BROKE {br}")
    print(f"  composed projection: 53.41 + delta x 0.76 = {53.41 + d*0.76:.2f}   (bar 53.54; pre-registered: >= +1.5 -> full run)")

# union
un = {}
try:
    for l in open(UN):
        r = json.loads(l); un[int(r["k"].split("|")[0])] = (r["v"] or "").strip()
    verdict("UNION-EVIDENCE", un, sorted(un.keys() & ss.keys()))
except FileNotFoundError:
    print("[union] file missing — skipped")

# strong best-of-4
try:
    sb = {}
    for l in open(SB):
        r = json.loads(l); i, s = r["k"].split("|"); sb.setdefault(int(i), {})[s] = (r["v"] or "").strip()
    vs = {json.loads(l)["idx"]: json.loads(l)["p_supported"] for l in open(VSS)}
    vt = {b: {json.loads(l)["idx"]: json.loads(l)["p_supported"] for l in open(VT.format(b=b))} for b in ("t1","t2","t3")}
    ids = sorted(i for i in sb if len(sb[i]) == 3 and i in vs and all(i in vt[b] for b in vt))
    picked = {}
    for i in ids:
        cand = [("ss", ss[i], vs[i])] + [(b, sb[i][b], vt[b][i]) for b in ("t1","t2","t3")]
        picked[i] = max(cand, key=lambda c: c[2])[1]
    verdict("STRONG-BEST-OF-4", picked, ids)
except FileNotFoundError:
    print("[strongbok] score files missing — run the verification chain first; skipped")
