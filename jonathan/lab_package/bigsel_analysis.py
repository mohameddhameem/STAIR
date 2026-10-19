#!/usr/bin/env python3
# bigsel_analysis.py v1.0 — do degenerate weak selections predict outcomes? (24 Sep 2026)
# Subgroup: instances where the WEAK retriever selects > --thr passages (select-all
# behavior). Reports, per split and subgroup-vs-rest: the four grid cells' lenient
# rates (WW WS SW SS) and gold-coverage of weak/strong selections — i.e., is
# "n_weak > thr -> just use strong retrieval" a sound RULE-BASED ret decision?
import json, re, string, argparse, os
os.chdir('/mnt/hdd/jonathan/slr/stage-level-routing/lab_package')
import run_labels4 as R

def norm(s):
    s=s.lower(); s="".join(ch for ch in s if ch not in set(string.punctuation))
    s=re.sub(r"\b(a|an|the)\b"," ",s); return " ".join(s.split())
def contains(t,golds):
    nt=norm(t)
    for g in golds:
        ng=norm(g)
        if ng and re.search(r"(?<!\w)"+re.escape(ng)+r"(?!\w)", nt): return True
    return False

ap=argparse.ArgumentParser()
ap.add_argument("--thr", type=int, default=20)
ap.add_argument("--data", default="/mnt/hdd/jonathan/data/tts_test.json")
ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
ap.add_argument("--retr_weak", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_weak.jsonl")
ap.add_argument("--retr_strong", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_strong.jsonl")
a=ap.parse_args()

data=json.load(open(a.data))
def rmap(p):
    m={}
    for l in open(p):
        r=json.loads(l); m[int(str(r["k"]).split("|")[0])]=r["v"]
    return m
rw,rs=rmap(a.retr_weak),rmap(a.retr_strong)
grid=[json.loads(l) for l in open(a.grid)]

def report(split):
    rows=[r for r in grid if r["split"]==split and r["idx"] in rw and r["idx"] in rs]
    def stats(sub,label):
        n=len(sub)
        if n==0: print(f"  {label:22s} n=0"); return
        e=[0,0,0,0]; covW=covS=0
        for r in sub:
            for c in range(4): e[c]+=r["em_lenient"][c]
            i=r["idx"]; inst=data[i]; nr=len(inst["ctxs_candidate"])
            golds=inst["answers"] if isinstance(inst["answers"],list) else [inst["answers"]]
            def ptext(j):
                k=j-1
                if k<0 or k>=nr: return ""
                t=inst["ctxs_candidate"][k]
                return t.get("text",str(t)) if isinstance(t,dict) else str(t)
            covW+=any(contains(ptext(j),golds) for j in R.parse_selection(rw[i],nr))
            covS+=any(contains(ptext(j),golds) for j in R.parse_selection(rs[i],nr))
        print(f"  {label:22s} n={n:5d}  WW {100*e[0]/n:5.1f}  WS {100*e[1]/n:5.1f}  SW {100*e[2]/n:5.1f}  SS {100*e[3]/n:5.1f}   covW {100*covW/n:5.1f}%  covS {100*covS/n:5.1f}%")
    big=[]; rest=[]
    for r in rows:
        i=r["idx"]; nr=len(data[i]["ctxs_candidate"])
        (big if len(R.parse_selection(rw[i],nr))>a.thr else rest).append(r)
    print(f"\n[{split}] weak selects > {a.thr} passages vs rest")
    stats(big,  f"n_weak > {a.thr}")
    stats(rest, f"n_weak <= {a.thr}")

for sp in ("val","test"): report(sp)
print(f"\nread: on the big-selection subgroup, SW,SS >> WW,WS and covS >> covW  ->  the rule \u201cweak select-all -> use strong retrieval\u201d is sound;")
print("      cells ~equal -> select-all is noise, not signal. (Rule value is bounded by the subgroup's size: ~4% of instances.)")
