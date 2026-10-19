import os, re, json, string
from collections import defaultdict
import run_labels as R

OUT='/mnt/hdd/jonathan/runs/full_run'
C=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,SS=0,1
data=json.load(open('/mnt/hdd/jonathan/data/tts_test.json'))
te=json.load(open('splits.json'))["splits"]["test"]

def norm(s):
    s=''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())

em=defaultdict(dict)
for l in open(f'{OUT}/combo_results_full.jsonl'):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in C] for i,d in em.items() if len(d)==4}

retr={}
for w in ('weak','strong'):
    d={}
    for l in open(f'{OUT}/retr_{w}.jsonl'):
        r=json.loads(l); d[int(str(r['k']).split('|')[0])]=r['v']
    retr[w]=d
qa={}
for w in ('weak','strong'):
    d={}
    for l in open(f'{OUT}/qa_{w}.jsonl'):
        r=json.loads(l); k=str(r['k'])
        d[k]=r['v']
    qa[w]=d

def where(i, arm):
    """arm 'ww' or 'ss' -> (answer, 'copied' | 'in pool, not selected' | 'not in pool')"""
    rs='weak'
    ans = qa['weak' if arm=='ww' else 'strong'].get(f"{i}|weak")
    if ans is None: return None, None
    inst=dict(data[i]); pool=R.trim_pool(inst)
    picks=[p for p in set(R.parse_selection(retr[rs][i], len(pool))) if 1<=p<=len(pool)]
    sel = norm(" ".join(pool[p-1] for p in picks))
    allp = norm(" ".join(pool))
    a=norm(ans)
    if not a: return ans, "empty"
    if a in sel:  return ans, "copied"
    if a in allp: return ans, "in pool, not selected"
    return ans, "not in pool"

groups=[("HEADROOM  ww solves, ss fails", [i for i in te if i in em and em[i][WW]==1 and em[i][SS]==0]),
        ("CONTROL   both solve",          [i for i in te if i in em and em[i][WW]==1 and em[i][SS]==1]),
        ("HOPELESS  neither solves",      [i for i in te if i in em and em[i][WW]==0 and em[i][SS]==0])]

for name,g in groups:
    print(f"\n=== {name}   n={len(g)} ===")
    for arm,label in (('ww','weak-weak answer  '),('ss','strong-strong answer')):
        cnt=defaultdict(int)
        for i in g:
            try: _,w = where(i,arm)
            except Exception: w=None
            if w: cnt[w]+=1
        t=sum(cnt.values()) or 1
        parts=" | ".join(f"{k} {100*v/t:.0f}%" for k,v in
                         sorted(cnt.items(), key=lambda x:-x[1]))
        print(f"  {label}: {parts}")

print("\nprediction if the confabulation story holds:")
print("  HEADROOM -> weak answer mostly 'copied', strong answer mostly 'not in pool'")
print("  CONTROL  -> both mostly 'copied'")
