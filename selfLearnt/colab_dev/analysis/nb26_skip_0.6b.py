# CPU re-analysis run on 13 Oct 2026 from saved notebook outputs (no GPU). Paths at the top/arguments point to the
# working copies used then; point them at the notebook 26/27 run folders to re-run. Results are in docs/Experiment_Plan_and_Log.md (Set 9).
import json,sys,numpy as np
from sklearn.metrics import roc_auc_score
D=sys.argv[1]
k=lambda s: tuple(sorted(s))
rd=lambda f:[json.loads(l) for l in open(f)]
R={n:{r['qid']:r for r in rd(f'{D}/step/retrieval_{n}.jsonl')} for n in ('r05','r06','r08')}
Q={(r['qid'],k(r['sel'])):r for r in rd(f'{D}/step/answers_q32.jsonl')}
Lm={(r['qid'],k(r['sel'])):r for r in rd(f'{D}/step/answers_l8.jsonl')}
ref={r['qid']:k(r['sel']) for r in rd(f'{D}/step/ref_shortlists.jsonl')}
rq7=[r for r in rd(f'{D}/results/retrieval.jsonl') if r['Retrieval'].startswith('Qwen 7B')][0]['Retrieval TFLOPs per question']*1e12
qids=list(R['r05'])
fixed_c=np.array([rq7+Q[(q,ref[q])]['flops'] for q in qids]); fixed_y=np.array([Q[(q,ref[q])]['lenient'] for q in qids])
lp=np.array([R['r05'][q]['mean_logprob'] for q in qids])
def outcome(q, esc, mode, A):
    s5=k(R['r05'][q]['sel']); f=R['r05'][q]['flops']
    if not esc: sel=s5
    else:
        f+=R['r08'][q]['flops']; s8=k(R['r08'][q]['sel'])
        sel = s8 if mode=='replace' else k(set(s5)|set(s8))
    a=A[(q,sel)]
    return a['lenient'], f+a['flops'], float(set(R['r05'][q]['sel'])<=set(sel) and True), sel
def evalp(esc, mode, A):
    o=[outcome(q,e,mode,A) for q,e in zip(qids,esc)]
    return np.array([x[0] for x in o]), np.array([x[1] for x in o])
def boot(a,b,B=4000):
    rng=np.random.default_rng(0); n=len(a); d=[a[w].mean()-b[w].mean() for w in (rng.integers(0,n,n) for _ in range(B))]
    return np.percentile(d,2.5)*100, np.percentile(d,97.5)*100
print(f"0.5B confidence separates bad shortlists: AUC {roc_auc_score([R['r05'][q]['found_both'] for q in qids], lp):.2f}")
for ans,A,lab in (('q32',Q,'Qwen 32B'),('l8',Lm,'Llama 8B')):
  y8,c8=evalp([True]*len(qids),'replace',A); c8-= np.array([R['r05'][q]['flops'] for q in qids])  # 0.8B alone
  y5,c5=evalp([False]*len(qids),'replace',A)
  print(f"\n== {lab} answers.  0.8B alone: {100*y8.mean():.1f}% at {100*(c8.sum()/fixed_c.sum()-1):+.1f}%;  0.5B alone: {100*y5.mean():.1f}% at {100*(c5.sum()/fixed_c.sum()-1):+.1f}%")
  print(" share escalated | 0.5B->0.8B replace: acc, compute vs fixed | 0.5B->0.8B union: acc, compute | random gate (replace) acc")
  rng=np.random.default_rng(1)
  for sh in (0.1,0.2,0.3,0.4,0.5):
    t=np.quantile(lp,sh); esc=lp<t
    yr,cr=evalp(esc,'replace',A); yu,cu=evalp(esc,'union',A)
    rnd=[evalp(rng.permutation(esc),'replace',A)[0].mean() for _ in range(50)]
    print(f"   {int(sh*100):>3}%        | {100*yr.mean():5.1f}%  {100*(cr.sum()/fixed_c.sum()-1):+6.1f}%   | {100*yu.mean():5.1f}%  {100*(cu.sum()/fixed_c.sum()-1):+6.1f}% | {100*np.mean(rnd):5.1f}%")
  # cross-validated: share chosen in folds (cheapest within 0.5 pt of best training acc), replace mode
  accs=[];comps=[];shs=[];rnds=[]
  for rep in range(5):
    rng=np.random.default_rng(rep); perm=rng.permutation(len(qids)); yv=np.zeros(len(qids)); cv=np.zeros(len(qids)); ev=np.zeros(len(qids),bool)
    for f in range(5):
      te=perm[f::5]; tr=np.setdiff1d(perm,te)
      cands=[]
      for sh in np.arange(0,1.0001,0.025):
        t=np.quantile(lp[tr],sh) if 0<sh<1 else (-np.inf if sh==0 else np.inf)
        yy,cc=evalp(lp<t,'replace',A); cands.append((yy[tr].mean(),cc[tr].mean(),t))
      best=max(c[0] for c in cands); t=min([c for c in cands if c[0]>=best-0.005],key=lambda c:c[1])[2]
      yy,cc=evalp(lp<t,'replace',A); yv[te]=yy[te]; cv[te]=cc[te]; ev[te]=(lp<t)[te]
    accs.append(yv); comps.append(cv); shs.append(ev.mean())
  yv=np.mean(accs,0); cv=np.mean(comps,0)
  lo,hi=boot(yv,y8); lo2,hi2=boot(yv,fixed_y)
  print(f" cross-validated (5x5): {100*yv.mean():.1f}% at {100*(cv.sum()/fixed_c.sum()-1):+.1f}%, escalated {100*np.mean(shs):.0f}%;  vs 0.8B alone {100*(yv.mean()-y8.mean()):+.1f} ({lo:+.1f} to {hi:+.1f}); vs fixed {100*(yv.mean()-fixed_y.mean()):+.1f} ({lo2:+.1f} to {hi2:+.1f}); compute vs 0.8B alone {100*(cv.sum()/c8.sum()-1):+.1f}%")
