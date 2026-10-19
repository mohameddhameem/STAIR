import os, json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
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
if os.path.exists('probe_feats.npz'):
    p=np.load('probe_feats.npz')
    tr0=[i for i in sp["train"] if i in pos]; te0=[i for i in sp["test"] if i in pos]
    print("(pool/embedding cache present but index order differs — using cascade features only)")
tr=[i for i in sp["train"] if i in pos]; te=[i for i in sp["test"] if i in pos]
Xtr=X[[pos[i] for i in tr]]; Xte=X[[pos[i] for i in te]]

TARGETS={
 "escalation helps (ss & !ww)": lambda i: int(em[i][SS]==1 and em[i][WW]==0),
 "strong RETRIEVAL helps (sw & !ww)": lambda i: int(em[i][SW]==1 and em[i][WW]==0),
 "strong QA helps (ws & !ww)": lambda i: int(em[i][WS]==1 and em[i][WW]==0),
 "weak-weak solves": lambda i: int(em[i][WW]==1),
}

print(f"\n{'target':36s} {'rate':>6s} {'TRAIN auc':>10s} {'TEST auc':>9s}  verdict")
print("-"*86)
for name,f in TARGETS.items():
    ytr=np.array([f(i) for i in tr]); yte=np.array([f(i) for i in te])
    # deliberately high capacity, no regularisation — can it MEMORISE?
    m=HGB(max_iter=800, learning_rate=0.2, max_leaf_nodes=255,
          min_samples_leaf=1, l2_regularization=0.0).fit(Xtr,ytr)
    a_tr=roc_auc_score(ytr, m.predict_proba(Xtr)[:,1])
    a_te=roc_auc_score(yte, m.predict_proba(Xte)[:,1])
    if a_tr < 0.75:   v="FEATURES CARRY NOTHING — cannot even fit train"
    elif a_te < 0.60: v="fits train, fails test -> overfitting / weak signal"
    else:             v="GENERALISES — worth pursuing"
    print(f"{name:36s} {ytr.mean():6.3f} {a_tr:10.3f} {a_te:9.3f}  {v}")

# memorisation floor: can it fit just 200 instances?
print("\nmemorisation test (fit 200 train instances, score those same 200):")
sub=tr[:200]; Xs=X[[pos[i] for i in sub]]
for name,f in TARGETS.items():
    ys=np.array([f(i) for i in sub])
    if ys.sum() in (0,len(ys)): continue
    m=HGB(max_iter=800, learning_rate=0.3, max_leaf_nodes=255,
          min_samples_leaf=1, l2_regularization=0.0).fit(Xs,ys)
    print(f"  {name:36s} train-on-itself auc {roc_auc_score(ys,m.predict_proba(Xs)[:,1]):.3f}")
