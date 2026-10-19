#!/usr/bin/env python3
# fuser_analysis.py v1.0 — their combiner vs ours on IDENTICAL candidates (24 Sep 2026)
# Per pilot instance: fused answer (their method) vs best-verified pick (ours, from
# bok_summary) vs first-sample (greedy) vs oracle-of-4. Calibrated lenient throughout.
import json, re, string, collections, os

DATA="/mnt/hdd/jonathan/data/tts_test.json"
GRID="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
BOK ="/mnt/hdd/jonathan/runs/verifier_scores/bok_summary.jsonl"
QW  ="/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl"
SAMP="/mnt/hdd/jonathan/runs/verifier_scores/bestofk_pilot.jsonl"  # s0=greedy, s1..s3=samples
FU  ="/mnt/hdd/jonathan/runs/verifier_scores/fuser_{who}.jsonl"

def norm(s):
    s=s.lower(); s="".join(ch for ch in s if ch not in set(string.punctuation))
    s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def em(p,g):
    np_=norm(p or "")
    for x in g:
        ng=norm(x)
        if (np_ in ("yes","no","noanswer") or ng in ("yes","no","noanswer")) and np_!=ng: continue
        if ng and re.search(r"(?<!\w)"+re.escape(ng)+r"(?!\w)", np_): return 1
    return 0

data=json.load(open(DATA))
G={i:(data[i]["answers"] if isinstance(data[i]["answers"],list) else [data[i]["answers"]]) for i in range(len(data))}
gem={json.loads(l)["idx"]: json.loads(l)["em_lenient"] for l in open(GRID)}
bok={json.loads(l)["idx"]: json.loads(l) for l in open(BOK)}
greedy={}
for l in open(QW):
    r=json.loads(l); i,br=r["k"].split("|")
    if br=="strong": greedy[int(i)]=(r.get("v") or "").strip()
samp=collections.defaultdict(list)
for l in open(SAMP):
    r=json.loads(l); samp[int(str(r["k"]).split("|")[0])].append((r.get("v") or "").strip())

for who in ("weak","strong"):
    p=FU.format(who=who)
    if not os.path.exists(p): print(f"[fuser_{who}] missing — skipped"); continue
    fu={int(json.loads(l)["k"].split("|")[0]): json.loads(l)["v"] for l in open(p)}
    ids=sorted(i for i in fu if i in bok and i in samp and len(samp[i])>=4)
    n=len(ids)
    r_fu=sum(em(fu[i],G[i]) for i in ids)
    r_vp=sum(bok[i]["bok_em"] for i in ids)                       # our verified pick
    r_fi=sum(em(samp[i][0],G[i]) for i in ids)                    # first sample (s0 = greedy)
    r_or=sum(1 if any(em(c,G[i]) for c in samp[i][:4]) else 0 for i in ids)
    fx=sum(1 for i in ids if em(fu[i],G[i])==1 and bok[i]["bok_em"]==0)
    br=sum(1 for i in ids if em(fu[i],G[i])==0 and bok[i]["bok_em"]==1)
    print(f"\n[FUSER={who} fuses our 4 weak candidates] n={n}")
    print(f"  their method (fused)      : {100*r_fu/n:6.2f}")
    print(f"  our method (verified pick): {100*r_vp/n:6.2f}   delta (theirs - ours) {100*(r_fu-r_vp)/n:+.2f}   FIXED {fx} | BROKE {br}")
    print(f"  first sample (greedy)     : {100*r_fi/n:6.2f}   oracle-of-4: {100*r_or/n:6.2f}")
