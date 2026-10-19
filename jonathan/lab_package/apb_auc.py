import json,sys
from collections import Counter
f=sys.argv[1] if len(sys.argv)>1 else 'apb_verify_hotpotqa.jsonl'
rows=[json.loads(l) for l in open(f)]
print(f"scored {len(rows)} steps   labels {dict(Counter(r['label'] for r in rows))}\n")
def auc(pos,neg):
    if not pos or not neg: return float('nan')
    xs=sorted([(p,1) for p in pos]+[(p,0) for p in neg])
    r={}; i=0
    while i<len(xs):
        j=i
        while j<len(xs) and xs[j][0]==xs[i][0]: j+=1
        rk=(i+j+1)/2
        for k in range(i,j): r[k]=rk
        i=j
    s=sum(r[k] for k,(p,y) in enumerate(xs) if y==1)
    return (s-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))
pos=[r['p_yes'] for r in rows if r['label']==1]
neg=[r['p_yes'] for r in rows if r['label']==-1]
neu=[r['p_yes'] for r in rows if r['label']==0]
print(f"{'group':22s} {'n':>5s} {'mean P(Yes)':>12s}")
print("-"*42)
for lab,v in (("correct (+1)",pos),("erroneous (-1)",neg),("neutral (0)",neu)):
    if v: print(f"{lab:22s} {len(v):5d} {sum(v)/len(v):11.3f}")
print()
print(f"  AUC, correct vs erroneous          {auc(pos,neg):.3f}")
if neu: print(f"  AUC, correct vs (erroneous+neutral) {auc(pos,neg+neu):.3f}")
print()
best=(0,0)
for t in [i/100 for i in range(1,100)]:
    acc=(sum(1 for p in pos if p>=t)+sum(1 for p in neg if p<t))/(len(pos)+len(neg))
    if acc>best[1]: best=(t,acc)
print(f"  best threshold {best[0]:.2f} -> accuracy {100*best[1]:.1f}%")
print(f"  majority-class baseline           {100*max(len(pos),len(neg))/(len(pos)+len(neg)):.1f}%")
print(f"\n  for reference: our verifier reached AUC 0.802 choosing BETWEEN MODELS on HotpotQA answers")
