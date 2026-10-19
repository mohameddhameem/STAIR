import os, re, json, math, argparse, string
import numpy as np
from collections import Counter, defaultdict
import run_labels as R
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

COMBOS = ["weak-weak","weak-strong","strong-weak","strong-strong"]
COST = {"weak-weak":2.0,"weak-strong":5.7,"strong-weak":5.7,"strong-strong":9.4}
WW, SS = 0, 3
TOK = lambda s: re.findall(r"[a-z0-9]+", s.lower())

def norm(s):
    s = ''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())

def bm25(qt, docs, k1=1.5, b=0.75):
    N=len(docs); avgdl=sum(len(d) for d in docs)/N
    df=Counter()
    for d in docs: df.update(set(d))
    out=[]
    for d in docs:
        tf=Counter(d); sc=0.0
        for t in qt:
            if t not in tf: continue
            idf=math.log(1+(N-df[t]+0.5)/(df[t]+0.5))
            sc+=idf*tf[t]*(k1+1)/(tf[t]+k1*(1-b+b*len(d)/avgdl))
        out.append(sc)
    return np.array(out)

ap=argparse.ArgumentParser()
ap.add_argument('--out', default='/mnt/hdd/jonathan/runs/full_run')
ap.add_argument('--data', default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--splits', default='splits.json')
ap.add_argument('--cache', default='cascade_feats.npz')
a=ap.parse_args()

data=json.load(open(a.data)); sp=json.load(open(a.splits))["splits"]
tr,te=sp["train"],sp["test"]
em=defaultdict(dict)
for l in open(os.path.join(a.out,'combo_results_full.jsonl')):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in COMBOS] for i,d in em.items() if len(d)==4}

if os.path.exists(a.cache):
    z=np.load(a.cache); X=z['X']; idxs=list(z['idxs']); print("cached", X.shape)
else:
    print("building features ...")
    retr={}
    for l in open(os.path.join(a.out,'retr_weak.jsonl')):
        r=json.loads(l); retr[str(r['k']).split('|')[0]]=r['v']
    qaw={}
    for l in open(os.path.join(a.out,'qa_weak.jsonl')):
        r=json.loads(l); k=str(r['k'])
        if k.endswith('|weak'): qaw[k.split('|')[0]]=r['v']
    rows=[]; idxs=[]
    for i in sorted(em):
        k=str(i)
        if k not in retr or k not in qaw: continue
        inst=dict(data[i]); pool=R.trim_pool(inst)
        qt=TOK(inst['question']); txt=retr[k]; ans=qaw[k]
        picks=sorted(set(R.parse_selection(txt,len(pool))))
        s=bm25(qt,[TOK(c) for c in pool]); ss_=np.sort(s)[::-1]
        top3=set(int(x)+1 for x in np.argsort(s)[::-1][:3])
        seltxt=" ".join(pool[p-1] for p in picks if 1<=p<=len(pool))
        rows.append([len(picks),
                     (min(picks) if picks else len(pool)+1),
                     (float(np.mean(picks)) if picks else len(pool)),
                     (max(picks)-min(picks) if len(picks)>1 else 0),
                     len(top3 & set(picks)),
                     int('Not applicable' in txt), len(txt),
                     len(ans), len(TOK(ans)),
                     int(bool(ans) and norm(ans) in norm(seltxt)),
                     int(norm(ans) in ('yes','no')),
                     ss_[0], ss_[0]-ss_[1], float(s.std()), len(qt)])
        idxs.append(i)
    X=np.array(rows,dtype=float); np.savez(a.cache,X=X,idxs=np.array(idxs))
    print("built", X.shape)

pos={v:k for k,v in enumerate(idxs)}
tr=[i for i in tr if i in pos]; te=[i for i in te if i in pos]
Xtr=X[[pos[i] for i in tr]]; Xte=X[[pos[i] for i in te]]
em_ww=np.mean([em[i][WW] for i in te])*100
em_ss=np.mean([em[i][SS] for i in te])*100
orc=np.mean([max(em[i]) for i in te])*100
print(f"\ntrain {len(tr)} test {len(te)}")
print(f"reference  always-ww {em_ww:.1f}%  always-ss {em_ss:.1f}%  oracle {orc:.1f}%")

y=lambda ix: np.array([int(em[i][SS]==1 and em[i][WW]==0) for i in ix])
ytr,yte=y(tr),y(te)
print(f"escalation-helps rate: train {ytr.mean():.3f}  test {yte.mean():.3f}")
for nm,mdl in (("logistic",make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=0.1))),
               ("gradboost",HGB(max_iter=300,learning_rate=0.06))):
    m=mdl.fit(Xtr,ytr); p=m.predict_proba(Xte)[:,1]
    print(f"\n[{nm}]  AUC = {roc_auc_score(yte,p):.3f}")
    print(f"  {'thr':>5s} {'escal':>7s} {'EM':>7s} {'cost':>7s}   vs always-ss")
    for t in (0.15,0.25,0.35,0.50,0.65):
        e=p>t
        emv=np.mean([em[i][SS] if k else em[i][WW] for i,k in zip(te,e)])*100
        c=np.mean([11.4 if k else 2.0 for k in e])/9.4*100
        print(f"  {t:5.2f} {100*e.mean():6.1f}% {emv:6.1f}% {c:6.1f}%   {emv-em_ss:+5.1f} EM  {c-100:+6.1f} cost")

NAMES=["n_selected","min_rank","mean_rank","rank_spread","bm25_top3_overlap","degenerate",
       "retr_len","ans_chars","ans_tokens","answer_grounded","is_yes_no","bm25_top1","bm25_gap","bm25_std","q_len"]
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=0.1)).fit(Xtr,ytr)
co=lr.named_steps['logisticregression'].coef_[0]
print("\nstrongest signals:")
for k in np.argsort(np.abs(co))[::-1][:8]:
    print(f"  {NAMES[k]:20s} {co[k]:+.3f}")
