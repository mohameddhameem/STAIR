import json, re, string, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
OUT='/mnt/hdd/jonathan/runs/full_run'
C=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,SS=0,3
def norm(s):
    s=''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
em=defaultdict(dict)
for l in open(f'{OUT}/combo_results_full.jsonl'):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in C] for i,d in em.items() if len(d)==4}
qa={}
for w in ('weak','strong'):
    d={}
    for l in open(f'{OUT}/qa_{w}.jsonl'):
        r=json.loads(l); d[str(r['k'])]=r['v']
    qa[w]=d
ag={}
for i in em:
    x=qa['weak'].get(f"{i}|weak"); y=qa['strong'].get(f"{i}|strong")
    if x is not None and y is not None: ag[i]=int(norm(x)==norm(y))
cb={}
for fn in ('closedbook.jsonl','closedbook_train.jsonl'):
    try:
        for l in open(fn):
            r=json.loads(l); cb[r['idx']]=[r['cb_correct'], r['agrees_with_open']]
    except FileNotFoundError: print(f"  (missing {fn})")
print(f"closed-book coverage: {len(cb)} instances")
sp=json.load(open('splits.json'))["splits"]
z=np.load('cascade_feats.npz'); X=z['X']; ix=list(z['idxs']); pos={v:k for k,v in enumerate(ix)}
def build(g, use_cb):
    g=[i for i in g if i in pos and i in ag and (not use_cb or i in cb)]
    M=X[[pos[i] for i in g]]
    if use_cb: M=np.hstack([M, np.array([cb[i] for i in cb and g], dtype=float)])
    return g, M, np.array([ag[i] for i in g])
mk=lambda: HGB(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,
               min_samples_leaf=50,l2_regularization=1.0)
for tag,use_cb in (("cascade only",False),("cascade + CLOSED-BOOK",True)):
    tr,Mtr,ytr=build(sp["train"],use_cb); te,Mte,yte=build(sp["test"],use_cb)
    if len(tr)<500: print(f"\n{tag}: only {len(tr)} train rows — skipped"); continue
    cv=cross_val_score(mk(),Mtr,ytr,cv=5,scoring='roc_auc')
    m=mk().fit(Mtr,ytr); p=m.predict_proba(Mte)[:,1]
    ss=np.mean([em[i][SS] for i in te])*100
    print(f"\n{tag}  ({Mtr.shape[1]} features, {len(tr)} train / {len(te)} test)")
    print(f"  predicting AGREEMENT:  CV AUC {cv.mean():.3f} +/- {cv.std():.3f}   test AUC {roc_auc_score(yte,p):.3f}")
    print(f"  {'route-to-ww':>12s} {'EM':>7s} {'cost':>7s} {'dEM':>6s} {'dcost':>7s}")
    for q in (0.10,0.20,0.30,0.40):
        t=np.quantile(p,1-q); u=p>=t
        E=np.mean([em[i][WW] if k else em[i][SS] for i,k in zip(te,u)])*100
        Cc=np.mean([2.0 if k else 9.4 for k in u])/9.4*100
        print(f"  {100*u.mean():11.0f}% {E:6.1f}% {Cc:6.1f}% {E-ss:+5.1f} {Cc-100:+6.1f}")
