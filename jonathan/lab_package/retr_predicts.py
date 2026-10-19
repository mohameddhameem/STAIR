import json, os, re, string
from collections import defaultdict
import run_labels as R
RUN='/mnt/hdd/jonathan/runs/full_run'
DATA='/mnt/hdd/jonathan/data/tts_test.json'
data=json.load(open(DATA))
em=defaultdict(dict)
for l in open(f'{RUN}/combo_results_full.jsonl'):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:d for i,d in em.items() if len(d)==4}
retr={}
for w in ('weak','strong'):
    d={}
    for l in open(f'{RUN}/retr_{w}.jsonl'):
        r=json.loads(l); d[int(str(r['k']).split('|')[0])]=r['v']
    retr[w]=d
rows=[]
for i,d in em.items():
    t=retr['strong'].get(i)
    if not t or 'OOM' in t: continue
    inst=dict(data[i]); inst['ctxs_candidate']=R.trim_pool(inst); pool=inst['ctxs_candidate']
    picks=[p for p in R.parse_selection(t,len(pool)) if 1<=p<=len(pool)]
    gold=set(R.gold_passages(inst)) if hasattr(R,'gold_passages') else set()
    if not gold: gold={c for c in pool if R.title_of(c) in set(inst.get('supporting_titles',[]))}
    got={pool[p-1] for p in picks}
    tp=len(got&gold)
    f1=0.0 if tp==0 else 2*(tp/len(got))*(tp/len(gold))/((tp/len(got))+(tp/len(gold)))
    rows.append({"idx":i,"ret_f1":f1,"n_sel":len(picks),
                 "weak_ok":d['strong-weak'],"strong_ok":d['strong-strong']})
print(f"n={len(rows)}\n")
def band(v):
    return "0.0" if v==0 else ("0-0.5" if v<0.5 else ("0.5-0.99" if v<0.99 else "1.0"))
g=defaultdict(list)
for r in rows: g[band(r['ret_f1'])].append(r)
print(f"{'Ret_F1':10s} {'n':>5s} {'weak QA ok':>11s} {'strong QA ok':>13s} {'disagree':>9s}")
print("-"*54)
for k in ("0.0","0-0.5","0.5-0.99","1.0"):
    v=g.get(k,[])
    if not v: continue
    w=sum(x['weak_ok'] for x in v)/len(v)
    s=sum(x['strong_ok'] for x in v)/len(v)
    dis=sum(1 for x in v if x['weak_ok']!=x['strong_ok'])/len(v)
    print(f"{k:10s} {len(v):5d} {100*w:10.1f}% {100*s:12.1f}% {100*dis:8.1f}%")
def auc(pos,neg):
    if not pos or not neg: return float('nan')
    xs=sorted([(p,1) for p in pos]+[(p,0) for p in neg]); r={}; i=0
    while i<len(xs):
        j=i
        while j<len(xs) and xs[j][0]==xs[i][0]: j+=1
        for k2 in range(i,j): r[k2]=(i+j+1)/2
        i=j
    s=sum(r[k2] for k2,(p,y) in enumerate(xs) if y==1)
    return (s-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
print()
print(f"AUC, Ret_F1 predicting weak-QA success        {auc([r['ret_f1'] for r in rows if r['weak_ok']],[r['ret_f1'] for r in rows if not r['weak_ok']]):.3f}")
dis=[r for r in rows if r['weak_ok']!=r['strong_ok']]
print(f"AUC, Ret_F1 predicting WHICH model wins       {auc([r['ret_f1'] for r in dis if r['weak_ok']],[r['ret_f1'] for r in dis if not r['weak_ok']]):.3f}  (n={len(dis)})")
