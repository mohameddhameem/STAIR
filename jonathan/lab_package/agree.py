import json, re, string, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
OUT='/mnt/hdd/jonathan/runs/armC_n500'
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
sp=json.load(open('splits.json'))["splits"]
ag={}
for i in em:
    a=qa['weak'].get(f"{i}|weak"); b=qa['strong'].get(f"{i}|strong")
    if a is not None and b is not None: ag[i]=int(norm(a)==norm(b))
te=[i for i in sp["test"] if i in ag]; tr=[i for i in sp["train"] if i in ag]
A=np.array([ag[i] for i in te])
print(f"n={len(te)}   ww and ss give the SAME answer on {100*A.mean():.1f}%")
for grp,mask in (("AGREE",A==1),("DISAGREE",A==0)):
    s=[i for i,m in zip(te,mask) if m]
    print(f"  {grp:9s} n={len(s):5d}  ww EM {100*np.mean([em[i][WW] for i in s]):5.1f}%"
          f"  ss EM {100*np.mean([em[i][SS] for i in s]):5.1f}%")
ss=np.mean([em[i][SS] for i in te])*100
c=np.mean([2.0 if ag[i] else 9.4 for i in te])/9.4*100
e=np.mean([em[i][WW] if ag[i] else em[i][SS] for i in te])*100
print(f"\nCEILING with a perfect agreement predictor: EM {e:.1f}% at {c:.1f}% cost")
print(f"                       (always-ss is EM {ss:.1f}% at 100%)")
z=np.load('cascade_feats_armC.npz'); X=z['X']; ix=list(z['idxs']); pos={v:k for k,v in enumerate(ix)}
cb={}
for l in open('closedbook.jsonl'):
    r=json.loads(l); cb[r['idx']]=[r['cb_correct'], r['agrees_with_open']]
def feats(g, use_cb):
    g=[i for i in g if i in pos and (not use_cb or i in cb)]
    M=X[[pos[i] for i in g]]
    if use_cb: M=np.hstack([M, np.array([cb[i] for i in g])])
    return g, M
for tag,use_cb in (("cascade only",False),("cascade + closed-book",True)):
    g,M=feats(tr,use_cb); y=np.array([ag[i] for i in g])
    if len(g)<200: print(f"{tag}: only {len(g)} train rows with features — skipped"); continue
    m=HGB(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,min_samples_leaf=50,l2_regularization=1.0)
    s=cross_val_score(m,M,y,cv=5,scoring='roc_auc')
    gt,Mt=feats(te,use_cb); yt=np.array([ag[i] for i in gt])
    p=m.fit(M,y).predict_proba(Mt)[:,1]
    print(f"\n{tag}: predicting AGREEMENT  CV AUC {s.mean():.3f}+/-{s.std():.3f}  test AUC {roc_auc_score(yt,p):.3f}")
    ssg=np.mean([em[i][SS] for i in gt])*100
    print(f"  {'route-to-ww':>12s} {'EM':>7s} {'cost':>7s} {'dEM':>6s} {'dcost':>7s}")
    for q in (0.10,0.20,0.30,0.40,0.50):
        t=np.quantile(p,1-q); u=p>=t
        E=np.mean([em[i][WW] if k else em[i][SS] for i,k in zip(gt,u)])*100
        Cc=np.mean([2.0 if k else 9.4 for k in u])/9.4*100
        print(f"  {100*u.mean():11.0f}% {E:6.1f}% {Cc:6.1f}% {E-ssg:+5.1f} {Cc-100:+6.1f}")
