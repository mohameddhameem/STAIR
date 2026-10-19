import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score

OUT='/mnt/hdd/jonathan/runs/full_run'
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]
WW,WS,SW,SS=0,1,2,3
em=defaultdict(dict)
for l in open(f'{OUT}/combo_results_full.jsonl'):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in COMBOS] for i,d in em.items() if len(d)==4}
sp=json.load(open('splits.json'))["splits"]
z=np.load('cascade_feats.npz'); X=z['X']; idxs=list(z['idxs'])
pos={v:k for k,v in enumerate(idxs)}
tr=[i for i in sp["train"] if i in pos]
Xtr=X[[pos[i] for i in tr]]

T={"escalation helps":lambda i:int(em[i][SS]==1 and em[i][WW]==0),
   "weak-weak solves":lambda i:int(em[i][WW]==1)}
rng=np.random.default_rng(0)
big=lambda: HGB(max_iter=800,learning_rate=0.2,max_leaf_nodes=255,
                min_samples_leaf=1,l2_regularization=0.0)

print(f"\n{'target':22s} {'labels':10s} {'TRAIN auc':>10s}   interpretation")
print("-"*72)
for n,f in T.items():
    y=np.array([f(i) for i in tr])
    for tag,yy in (("real",y),("SHUFFLED",rng.permutation(y))):
        a=roc_auc_score(yy,big().fit(Xtr,yy).predict_proba(Xtr)[:,1])
        print(f"{n:22s} {tag:10s} {a:10.3f}")
print("\nif SHUFFLED also reaches ~1.000, memorisation proves nothing about signal.")

print("\nhonest estimate — 5-fold CV on TRAIN only, regularised model:")
for n,f in T.items():
    y=np.array([f(i) for i in tr])
    for mn,m in (("logistic",make_pipeline(StandardScaler(),LogisticRegression(max_iter=5000,C=0.1))),
                 ("gbm (reg)",HGB(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,
                                  min_samples_leaf=50,l2_regularization=1.0))):
        s=cross_val_score(m,Xtr,y,cv=5,scoring='roc_auc')
        print(f"  {n:22s} {mn:10s} AUC {s.mean():.3f} +/- {s.std():.3f}")
