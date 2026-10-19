import json, numpy as np
from collections import defaultdict
from sklearn.metrics import roc_auc_score
OUT='/mnt/hdd/jonathan/runs/full_run'
C=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,WS,SW,SS=0,1,2,3
em=defaultdict(dict)
for l in open(f'{OUT}/combo_results_full.jsonl'):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in C] for i,d in em.items() if len(d)==4}
cb={}                                   # later lines win, so --all overrides the diagnostic
for l in open('closedbook.jsonl'):
    r=json.loads(l); cb[r['idx']]=(r['cb_correct'], r['agrees_with_open'])
te=[i for i in json.load(open('splits.json'))["splits"]["test"] if i in cb and i in em]
print(f"n = {len(te)}")
correct=np.array([cb[i][0] for i in te]); agree=np.array([cb[i][1] for i in te])
print(f"cb_correct {correct.mean():.3f}   agrees {agree.mean():.3f}\n")
T={"ww solves":lambda i: em[i][WW],
   "ss solves":lambda i: em[i][SS],
   "escalation helps (ss & !ww)":lambda i: int(em[i][SS]==1 and em[i][WW]==0),
   "WEAK WINS (ww & !ss)":lambda i: int(em[i][WW]==1 and em[i][SS]==0),
   "anything solves":lambda i: int(max(em[i])==1)}
print(f"{'target':30s} {'rate':>6s} {'AUC cb_correct':>15s} {'AUC agrees':>11s}")
print("-"*66)
for n,f in T.items():
    y=np.array([f(i) for i in te])
    if y.sum() in (0,len(y)): continue
    print(f"{n:30s} {y.mean():6.3f} {roc_auc_score(y,correct):15.3f} {roc_auc_score(y,agree):11.3f}")
