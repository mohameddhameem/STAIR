import os, re, json, string, argparse
import numpy as np
from collections import defaultdict
import run_labels as R
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,SS=0,3
CW,CS=2.0,9.4
TOK=lambda s: re.findall(r"[a-z0-9]+", s.lower())
def norm(s):
    s=''.join(c for c in s.lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
ap=argparse.ArgumentParser()
ap.add_argument('--out',default='/mnt/hdd/jonathan/runs/full_run_qwen3')
ap.add_argument('--data',default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--splits',default='splits.json')
ap.add_argument('--cache',default='verify_feats.npz')
a=ap.parse_args()
data=json.load(open(a.data)); sp=json.load(open(a.splits))["splits"]
em=defaultdict(dict)
for l in open(os.path.join(a.out,'combo_results_full.jsonl')):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in COMBOS] for i,d in em.items() if len(d)==4}
retr={}
for w in ('weak','strong'):
    d={}
    for l in open(os.path.join(a.out,f'retr_{w}.jsonl')):
        r=json.loads(l); d[int(str(r['k']).split('|')[0])]=r['v']
    retr[w]=d
qa={}
for w in ('weak','strong'):
    d={}
    for l in open(os.path.join(a.out,f'qa_{w}.jsonl')):
        r=json.loads(l); d[str(r['k'])]=r['v']
    qa[w]=d
aw_={i:qa['weak'].get(f"{i}|weak") for i in em}
as_={i:qa['strong'].get(f"{i}|strong") for i in em}
dis=[i for i in em if aw_.get(i) is not None and as_.get(i) is not None
     and norm(aw_[i])!=norm(as_[i])]
print(f"disagreements: {len(dis)} of {len(em)} ({100*len(dis)/len(em):.1f}%)")
if os.path.exists(a.cache):
    z=np.load(a.cache); X=z['X']; idxs=list(z['idxs']); print(f"cached {X.shape}")
else:
    print("building features ...")
    rows=[]; idxs=[]
    for n,i in enumerate(dis):
        inst=dict(data[i]); pool=R.trim_pool(inst)
        A,B=aw_[i],as_[i]; nw,ns=norm(A),norm(B)
        sel={}
        for w in ('weak','strong'):
            pk=[p for p in R.parse_selection(retr[w][i],len(pool)) if 1<=p<=len(pool)]
            sel[w]=(pk, norm(" ".join(pool[p-1] for p in pk)))
        allp=norm(" ".join(pool))
        def where(x,w):
            if x and x in sel[w][1]: return 2
            if x and x in allp: return 1
            return 0
        rows.append([where(nw,'weak'),where(ns,'strong'),where(nw,'strong'),where(ns,'weak'),
                     len(TOK(A)),len(TOK(B)),len(TOK(A))-len(TOK(B)),len(A),len(B),
                     int(nw in ns or ns in nw),int(nw in('yes','no')),int(ns in('yes','no')),
                     len(sel['weak'][0]),len(sel['strong'][0]),len(TOK(inst['question']))])
        idxs.append(i)
        if (n+1)%500==0: print(f"  {n+1}/{len(dis)} ...",flush=True)
    X=np.array(rows,dtype=float); np.savez(a.cache,X=X,idxs=np.array(idxs))
    print(f"built {X.shape}")
pos={v:k for k,v in enumerate(idxs)}
tr=[i for i in sp["train"] if i in pos]; te=[i for i in sp["test"] if i in pos]
Xtr,Xte=X[[pos[i] for i in tr]],X[[pos[i] for i in te]]
ytr=np.array([em[i][WW] for i in tr]); yte=np.array([em[i][WW] for i in te])
print(f"\ntrain {len(tr)} test {len(te)}  weak-is-right: train {ytr.mean():.3f} test {yte.mean():.3f}")
allte=[i for i in sp["test"] if i in em]
ss=np.mean([em[i][SS] for i in allte])*100
orc=np.mean([max(em[i][WW],em[i][SS]) for i in allte])*100
print(f"\nfull test split ({len(allte)}):")
print(f"  always-strong-strong  {ss:.1f}% EM at 100% cost")
print(f"  perfect verification  {orc:.1f}% EM at {100*(CW+CS)/CS:.0f}% cost ({orc-ss:+.1f} EM)")
print(f"\n{'model':22s} {'CV AUC':>16s} {'test AUC':>9s}")
best=None
for nm,mk in (("logistic",lambda: make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=0.1))),
              ("gradient boosting",lambda: HGB(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,
                                               min_samples_leaf=30,l2_regularization=1.0))):
    cv=cross_val_score(mk(),Xtr,ytr,cv=5,scoring='roc_auc')
    p=mk().fit(Xtr,ytr).predict_proba(Xte)[:,1]; au=roc_auc_score(yte,p)
    print(f"{nm:22s} {cv.mean():.3f} +/- {cv.std():.3f} {au:9.3f}")
    if best is None or au>best[1]: best=(nm,au,p)
nm,au,p=best
print(f"\npolicy using {nm} (take the WEAK answer above threshold):")
print(f"  {'thresh':>7s} {'take weak':>10s} {'EM':>7s} {'vs always-ss':>13s} {'% headroom':>12s}")
base={i:em[i][SS] for i in allte}
for t in (0.30,0.40,0.50,0.60,0.70):
    pick={i:(em[i][WW] if q>=t else em[i][SS]) for i,q in zip(te,p)}
    e=np.mean([pick.get(i,base[i]) for i in allte])*100
    print(f"  {t:7.2f} {100*np.mean(p>=t):9.1f}% {e:6.1f}% {e-ss:+12.1f} "
          f"{100*(e-ss)/max(1e-9,orc-ss):11.1f}%")
NAMES=["ww_grounded","ss_grounded","ww_in_ss_passages","ss_in_ww_passages","ww_len_tok",
       "ss_len_tok","len_diff","ww_chars","ss_chars","one_contains_other","ww_yesno",
       "ss_yesno","ww_n_selected","ss_n_selected","q_len"]
lr=make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=0.1)).fit(Xtr,ytr)
co=lr.named_steps['logisticregression'].coef_[0]
print("\nstrongest signals (positive favours the WEAK answer):")
for k in np.argsort(np.abs(co))[::-1][:8]:
    print(f"  {NAMES[k]:22s} {co[k]:+.3f}")
