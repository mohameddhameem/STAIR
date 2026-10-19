#!/usr/bin/env python3
# score_cluster72b.py v1.0 — lenient re-score of the surviving cluster files (23 Sep 2026)
# Scores EVERY answer under the project's calibrated lenient metric (whole-word
# containment + yes/no guard + any-gold). Region: idx 0..499 = first-500 of
# tts_test.json. Prints per file: n, fused, first-sample, majority-vote,
# oracle-over-samples; for verify40_scores: the VERIFIED pick (argmax p_yes) and
# oracle-over-uniques. Model attribution deliberately NOT asserted here — file
# names only; we pin file->model mapping together afterward.
import json, re, string, collections, os

DIR  = "/mnt/hdd/jonathan/runs/cluster_72b"
DATA = "/mnt/hdd/jonathan/data/tts_test.json"

def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())
def em(pred, golds):
    np_ = norm(pred or "")
    for g in golds:
        ng = norm(g)
        if (np_ in ("yes","no","noanswer") or ng in ("yes","no","noanswer")) and np_ != ng: continue
        if ng and re.search(r"(?<!\w)"+re.escape(ng)+r"(?!\w)", np_): return 1
    return 0

data = json.load(open(DATA))
G = {i: (data[i]["answers"] if isinstance(data[i]["answers"], list) else [data[i]["answers"]]) for i in range(len(data))}

def vote(cands):
    c = collections.Counter(norm(x) for x in cands if x)
    if not c: return ""
    top = c.most_common(1)[0][0]
    for x in cands:
        if norm(x) == top: return x
    return cands[0]

print(f"{'file':22s} {'n':>4s} {'fused':>6s} {'first':>6s} {'vote':>6s} {'oracle':>6s}  (lenient %, first-500 region)")
for f in ["q8_r7","q8_r72","q70_weak","q70_strong","rep1","rep40","repw1","repw40"]:
    p = os.path.join(DIR, f + ".jsonl")
    if not os.path.exists(p): print(f"{f:22s} MISSING"); continue
    n=fu=fi=vo=orc=0
    kc = collections.Counter()
    for l in open(p):
        r = json.loads(l); i = r["idx"]; g = G[i]; cands = r.get("cands") or []
        kc[len(cands)] += 1
        n += 1
        fu += em(r.get("fused",""), g)
        fi += em(cands[0] if cands else "", g)
        vo += em(vote(cands), g)
        orc += 1 if any(em(c, g) for c in cands) else 0
    print(f"{f:22s} {n:4d} {100*fu/n:6.2f} {100*fi/n:6.2f} {100*vo/n:6.2f} {100*orc/n:6.2f}   samples/row: {dict(kc)}")

for f in ["fusion_qa","fusion_k40"]:
    p = os.path.join(DIR, f + ".jsonl")
    if not os.path.exists(p): continue
    print(f"\n[{f}] per source (our lenient re-score | their stored em_* means for reference)")
    by = collections.defaultdict(lambda: [0,0,0,0,0, 0,0,0,0])
    for l in open(p):
        r = json.loads(l); g = G[r["idx"]]; c = r.get("cands") or []
        b = by[r["src"]]
        b[0]+=1; b[1]+=em(r.get("fused",""),g); b[2]+=em(c[0] if c else "",g); b[3]+=em(vote(c),g); b[4]+=1 if any(em(x,g) for x in c) else 0
        b[5]+=r.get("em_fused",0); b[6]+=r.get("em_first",0); b[7]+=r.get("em_vote",0); b[8]+=r.get("em_best",0)
    for src,b in sorted(by.items()):
        print(f"  {src:7s} n={b[0]:4d}  fused {100*b[1]/b[0]:6.2f}  first {100*b[2]/b[0]:6.2f}  vote {100*b[3]/b[0]:6.2f}  oracle {100*b[4]/b[0]:6.2f}   | theirs: fused {100*b[5]/b[0]:.1f} first {100*b[6]/b[0]:.1f} vote {100*b[7]/b[0]:.1f} best {100*b[8]/b[0]:.1f}")

p = os.path.join(DIR, "verify40_scores.jsonl")
if os.path.exists(p):
    print("\n[verify40_scores] VERIFIED pick = argmax p_yes over unique answers")
    by = collections.defaultdict(lambda: [0,0,0])
    for l in open(p):
        r = json.loads(l); g = G[r["idx"]]
        u, py = r.get("uniq") or [], r.get("p_yes") or []
        b = by[r["src"]]
        b[0]+=1
        if u and py: b[1]+=em(u[max(range(len(py)), key=lambda k: py[k])], g)
        b[2]+=1 if any(em(x,g) for x in u) else 0
    for src,b in sorted(by.items()):
        print(f"  {src:7s} n={b[0]:4d}  verified-pick {100*b[1]/b[0]:6.2f}  oracle-over-uniq {100*b[2]/b[0]:6.2f}")
print("\nReference points (same region caveats apply): AgentTTS published 0.74 lenient; our July mixed-grid SS 57.0 lenient; slide-recorded 72B Qwen grid best-fixed 60.2 EM / oracle 82.0 lenient.")
