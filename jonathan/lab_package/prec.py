import json, re, string, numpy as np
from collections import defaultdict
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
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
ag={i:int(norm(qa['weak'][f"{i}|weak"])==norm(qa['strong'][f"{i}|strong"]))
    for i in em if f"{i}|weak" in qa['weak'] and f"{i}|strong" in qa['strong']}
cb={}
for fn in ('closedbook.jsonl','closedbook_train.jsonl'):
    for l in open(fn):
        r=json.loads(l); cb[r['idx']]=[r['cb_correct'], r['agrees_with_open']]
sp=json.load(open('splits.json'))["splits"]
z=np.load('cascade_feats.npz'); X=z['X']; ix=list(z['idxs']); pos={v:k for k,v in enumerate(ix)}
def build(g,use_cb):
    g=[i for i in g if i in pos and i in ag and i in cb]
    M=X[[pos[i] for i in g]]
    if use_cb: M=np.hstack([M,np.array([cb[i] for i in g],dtype=float)])
    return g,M,np.array([ag[i] for i in g])
mk=lambda: HGB(max_iter=200,learning_rate=0.05,max_leaf_nodes=15,
               min_samples_leaf=50,l2_regularization=1.0)
print(f"base agreement rate on test: ", end="")
for tag,use_cb in (("cascade",False),("cascade+CB",True)):
    tr,Mtr,ytr=build(sp["train"],use_cb); te,Mte,yte=build(sp["test"],use_cb)
    if tag=="cascade": print(f"{yte.mean():.3f}\n")
    p=mk().fit(Mtr,ytr).predict_proba(Mte)[:,1]
    print(f"{tag}  (test AUC on agreement, top-k precision)")
    print(f"  {'top k%':>7s} {'n':>5s} {'precision':>10s} {'max p':>7s} {'min p':>7s}")
    for q in (0.02,0.05,0.10,0.20,0.30):
        k=int(len(p)*q); o=np.argsort(p)[::-1][:k]
        print(f"  {100*q:6.0f}% {k:5d} {yte[o].mean():10.3f} {p[o].max():7.3f} {p[o].min():7.3f}")
    print(f"  absolute thresholds:")
    for t in (0.6,0.7,0.8,0.9):
        m=p>=t
        if m.sum()==0: print(f"    p>={t}: none"); continue
        E=np.mean([em[i][WW] if k else em[i][SS] for i,k in zip(te,m)])*100
        Cc=np.mean([2.0 if k else 9.4 for k in m])/9.4*100
        ss=np.mean([em[i][SS] for i in te])*100
        print(f"    p>={t}: {100*m.mean():5.1f}% routed  precision {yte[m].mean():.3f}  "
              f"EM {E:.1f}% ({E-ss:+.1f})  cost {Cc:.1f}%")
    print()
