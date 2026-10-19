import json, re, string, sys
from collections import Counter
F=sys.argv[1] if len(sys.argv)>1 else 'qwen3_k20.jsonl'
DATA=sys.argv[2] if len(sys.argv)>2 else '/mnt/hdd/jonathan/data/tts_test.json'
data=json.load(open(DATA))
def base(s): return ''.join(c for c in s.lower() if c not in set(string.punctuation))
def strip_think(s):
    s=re.sub(r'<think>.*?</think>','',s,flags=re.S)
    return re.sub(r'</?think>','',s).strip()
def norm(s): return ' '.join(re.sub(r'\b(a|an|the)\b',' ',base(strip_think(s))).split())
rows={}
for l in open(F):
    r=json.loads(l)
    if r['src']=='strong': rows[r['idx']]=[norm(c) for c in r['cands']]
ids=sorted(rows); n=len(ids)
K=min(len(rows[i]) for i in ids)
G={i:{norm(x) for x in data[i]['answers']} for i in ids}
print(f"{F}: {n} instances, k up to {K}\n")
def vote(i,k): return Counter(rows[i][:k]).most_common(1)[0][0]
U={k:100*sum(int(vote(i,k) in G[i]) for i in ids)/n for k in range(1,K+1)}
print("UNIFORM"); 
for k in (1,2,3,5,10,K):
    if k<=K: print(f"  k={k:<3d} {U[k]:5.1f}%   {100*U[k]/U[K]:5.1f}% of k={K}")
print("\n  smallest k reaching:")
for t in (0.90,0.95,0.98,1.00):
    kk=next((j for j in range(1,K+1) if U[j]>=t*U[K]),None)
    print(f"     {int(t*100)}%  ->  k={kk}  ({U[kk]:.1f}%)")
KS={}
for i in ids:
    ks=None
    for k in range(K,0,-1):
        if vote(i,k) in G[i]: ks=k
        else: break
    KS[i]=ks
solv=[i for i in ids if KS[i]]
d=Counter(KS[i] for i in solv)
print(f"\nORACLE k*  solved {len(solv)}/{n} ({100*len(solv)/n:.1f}%)")
print(f"  k*=1: {d.get(1,0)}   k*=2-5: {sum(v for k,v in d.items() if 2<=k<=5)}"
      f"   k*=6+: {sum(v for k,v in d.items() if k>=6)}")
mk=(sum(KS[i] for i in solv)+K*(n-len(solv)))/n
need=next((j for j in range(1,K+1) if U[j]>=100*len(solv)/n-1e-9),K)
print(f"  oracle mean k {mk:.2f}   fixed k needed for same acc: {need}"
      f"   saving {100*(1-mk/need):.1f}%")
AG3={i: Counter(rows[i][:3]).most_common(1)[0][1]/3 for i in ids}
def auc(p,q):
    if not p or not q: return float('nan')
    xs=sorted([(x,1) for x in p]+[(x,0) for x in q]); r={}; i=0
    while i<len(xs):
        j=i
        while j<len(xs) and xs[j][0]==xs[i][0]: j+=1
        for k in range(i,j): r[k]=(i+j+1)/2
        i=j
    s=sum(r[k] for k,(x,y) in enumerate(xs) if y==1)
    return (s-len(p)*(len(p)+1)/2)/(len(p)*len(q))
easy=[i for i in solv if KS[i]<=3]; hard=[i for i in ids if i not in easy]
print(f"\nagreement at k=3 predicts 'k<=3 suffices': AUC "
      f"{auc([AG3[i] for i in easy],[AG3[i] for i in hard]):.3f}")
