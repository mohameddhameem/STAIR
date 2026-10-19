import json, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
OUT='/mnt/hdd/jonathan/runs/full_run'
C=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,SS=0,3
em=defaultdict(dict)
for l in open(f'{OUT}/combo_results_full.jsonl'):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in C] for i,d in em.items() if len(d)==4}
sp=json.load(open('splits.json'))["splits"]
z=np.load('cascade_feats.npz'); X=z['X']; ix=list(z['idxs']); pos={v:k for k,v in enumerate(ix)}
tr=[i for i in sp["train"] if i in pos]; te=[i for i in sp["test"] if i in pos]
y=lambda g: np.array([em[i][WW] for i in g])
m=HGB(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=50,
      l2_regularization=1.0).fit(X[[pos[i] for i in tr]], y(tr))
p=m.predict_proba(X[[pos[i] for i in te]])[:,1]
ss=np.mean([em[i][SS] for i in te])*100
ww=np.mean([em[i][WW] for i in te])*100
print(f"\nalways-ss {ss:.1f}% EM @ 100% cost   |   always-ww {ww:.1f}% @ 21.3%")
print(f"\n{'route-to-ww':>12s} {'EM':>7s} {'cost':>7s} {'ΔEM':>7s} {'Δcost':>7s}")
for q in (0.05,0.10,0.15,0.20,0.30,0.40,0.50):
    t=np.quantile(p,1-q); use_ww=p>=t
    e=np.mean([em[i][WW] if k else em[i][SS] for i,k in zip(te,use_ww)])*100
    c=np.mean([2.0 if k else 9.4 for k in use_ww])/9.4*100
    print(f"{100*use_ww.mean():11.0f}% {e:6.1f}% {c:6.1f}% {e-ss:+6.1f} {c-100:+6.1f}")
