#!/usr/bin/env python3
# shared_census.py v1.0 — CPU census for the shared-passages experiment (24 Sep 2026)
# Per instance: weak-selected set, strong-selected set, their INTERSECTION (shared).
# Reports (val by default): shared-size distribution, % empty intersections,
# jaccard strata, and GOLD-COVERAGE (>=1 selected passage contains a gold answer,
# whole-word) for weak / strong / shared / union — stratified by overlap tercile.
# If shared gold-coverage ~= strong gold-coverage on LOW-overlap instances, the
# unique passages are decorative before any GPU run is spent.
import json, re, string, argparse, statistics as st, collections
import os
os.chdir('/mnt/hdd/jonathan/slr/stage-level-routing/lab_package')
import run_labels4 as R

def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def contains(passage, golds):
    np_ = norm(passage)
    for g in golds:
        ng = norm(g)
        if ng and re.search(r"(?<!\w)"+re.escape(ng)+r"(?!\w)", np_): return True
    return False

ap = argparse.ArgumentParser()
ap.add_argument("--data", default="/mnt/hdd/jonathan/data/tts_test.json")
ap.add_argument("--retr_weak", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_weak.jsonl")
ap.add_argument("--retr_strong", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_strong.jsonl")
ap.add_argument("--splits", default="splits.json")
ap.add_argument("--split", default="val")
a = ap.parse_args()

data = json.load(open(a.data))
def rmap(p):
    m = {}
    for l in open(p):
        r = json.loads(l); m[int(str(r["k"]).split("|")[0])] = r["v"]
    return m
rw, rs = rmap(a.retr_weak), rmap(a.retr_strong)
sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
ids = [i for i in sp[a.split] if i in rw and i in rs]
print(f"[census] split={a.split} n={len(ids)}")

rows = []
for i in ids:
    inst = data[i]; n_refs = len(inst["ctxs_candidate"])
    golds = inst["answers"] if isinstance(inst["answers"], list) else [inst["answers"]]
    W = set(R.parse_selection(rw[i], n_refs)); S = set(R.parse_selection(rs[i], n_refs))
    I = W & S; U = W | S
    jac = len(I) / len(U) if U else 1.0
    texts = inst["ctxs_candidate"]
    def ptext(j):
        # parse_selection returns 1-based reference ids -> passage index j-1; guard range
        k = j - 1
        if k < 0 or k >= len(texts): return ""
        t = texts[k]
        return t.get("text", str(t)) if isinstance(t, dict) else str(t)
    def cov(sel): return any(contains(ptext(j), golds) for j in sel)
    rows.append(dict(i=i, nW=len(W), nS=len(S), nI=len(I), jac=jac,
                     covW=cov(W), covS=cov(S), covI=cov(I), covU=cov(U)))

nI = [r["nI"] for r in rows]
print(f"\n[intersection size] median {st.median(nI)}  mean {sum(nI)/len(nI):.2f}  EMPTY: {sum(1 for x in nI if x==0)} ({100*sum(1 for x in nI if x==0)/len(rows):.1f}%)")
print(f"[selection sizes]   weak median {st.median(r['nW'] for r in rows)}  strong median {st.median(r['nS'] for r in rows)}")
def pct(f): return 100*sum(1 for r in rows if r[f])/len(rows)
print(f"\n[GOLD-COVERAGE, all {a.split}]  weak {pct('covW'):.1f}%  strong {pct('covS'):.1f}%  SHARED {pct('covI'):.1f}%  union {pct('covU'):.1f}%")
terc = sorted(r["jac"] for r in rows); t1, t2 = terc[len(terc)//3], terc[2*len(terc)//3]
print(f"\n[stratified by overlap jaccard: terciles at {t1:.2f} / {t2:.2f}]")
print(f"{'stratum':16s} {'n':>5s} {'empty%':>7s} {'covW':>6s} {'covS':>6s} {'covSHARED':>9s} {'covUNION':>8s}")
for name, lo, hi in [("LOW overlap", -1, t1), ("MID overlap", t1, t2), ("HIGH overlap", t2, 2)]:
    g = [r for r in rows if lo < r["jac"] <= hi]
    if not g: continue
    e = 100*sum(1 for r in g if r["nI"]==0)/len(g)
    def p(f): return 100*sum(1 for r in g if r[f])/len(g)
    print(f"{name:16s} {len(g):5d} {e:7.1f} {p('covW'):6.1f} {p('covS'):6.1f} {p('covI'):9.1f} {p('covU'):8.1f}")
print("\nread: SHARED ~= STRONG coverage on the LOW stratum -> unique passages decorative (no Ret-side signal to route on);")
print("      SHARED << STRONG there -> strong's unique picks are load-bearing -> Ret signal exists; GPU cells then quantify it.")
