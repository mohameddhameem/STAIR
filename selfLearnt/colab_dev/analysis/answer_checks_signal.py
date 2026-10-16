# CPU re-analysis run on 13 Oct 2026 from saved notebook outputs (no GPU). Paths at the top/arguments point to the
# working copies used then; point them at the notebook 26/27 run folders to re-run. Results are in docs/Experiment_Plan_and_Log.md (Set 9).
import json,os,sys,re,numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code"))
import ssw as S
from sklearn.metrics import roc_auc_score as AUC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
k=lambda s: tuple(sorted(s))
rd=lambda f:[json.loads(l) for l in open(f)]
L={(r['qid'],k(r['sel'])):r for r in rd('run/step/answers_l8.jsonl')}
Qw={(r['qid'],k(r['sel'])):r for r in rd('run/step/answers_q32.jsonl')}
P="/content/drive/MyDrive/0_EngD_assignments/0_Small_Big_Model_Switching/data/hotpotqa_q"
qs={}
for f in ('label/route_train.json','label/calib.json'):
    for q in json.load(open(f)): qs[q['qid']]=q
for d in ('dev100_test100_c20_s42_first','fresh200_c20_s43'):
    for s in ('dev','test'):
        for q in json.load(open(f'{P}/{d}/{s}.json')): qs[q['qid']]=q
sel_tr={r['qid']:k(r['sel']) for r in rd('run/step/route_train_shortlists.jsonl')}
sel_ca={r['qid']:k(r['sel']) for r in rd('run/step/calib_route.jsonl')}
sel_te={r['qid']:k(r['sel']) for r in rd('run/step/test_route.jsonl')}
N=S.normalize_answer
def feats(q,sel):
    a=L[(q['qid'],sel)]; pred=a['prediction'] or ''; p=N(pred)
    psg=" ".join(q['ctxs_candidate'][i-1] for i in sel if 1<=i<=len(q['ctxs_candidate']))
    np_=N(psg); yn=p in ('yes','no')
    toks=p.split()
    in_psg=float(bool(p) and (' '+p+' ') in (' '+np_+' '))
    frac=np.mean([t in np_.split() for t in toks]) if toks else 0.0
    in_q=float(bool(p) and p in N(q['question']))
    text=a.get('text') or ''
    return [a.get('mean_logprob') or -1, a.get('stated_conf') or 0.5, float(a.get('hit_cap',False)), a['out_tokens'],
            float(yn), len(toks), in_psg, frac, in_q, float(not p), float('comparison'==q.get('type')), float(bool(re.search(r"\b(not|no information|cannot|unknown|does not)\b", text.lower())))], a['lenient'], Qw.get((q['qid'],sel),{}).get('lenient')
def build(selmap):
    X=[];y=[];yq=[]
    for qid,sel in selmap.items():
        if (qid,sel) not in L: continue
        f,l,qq=feats(qs[qid],sel); X.append(f); y.append(l); yq.append(qq)
    return np.array(X,float),np.array(y,float),yq
Xt,yt,_=build(sel_tr); Xc,yc,qc=build(sel_ca); Xe,ye,qe=build(sel_te)
names=['token prob','stated conf','hit length cap','answer length (tokens)','yes/no answer','answer words','answer found in passages','share of answer words in passages','answer copies the question','empty answer','comparison question','says not found / cannot']
print(f"train {len(yt)}  calib {len(yc)}  test {len(ye)}")
for j,n in enumerate(names):
    try: print(f"  {n:<36} calib AUC {AUC(yc,Xc[:,j]):.2f}  test AUC {AUC(ye,Xe[:,j]):.2f}")
    except ValueError: pass
for cols,lab in (([0,1],'token prob + stated conf (what we had)'),(list(range(len(names))),'all answer checks combined')):
    sc=StandardScaler().fit(Xt[:,cols]); m=LogisticRegression(max_iter=3000).fit(sc.transform(Xt[:,cols]),yt)
    pc=m.predict_proba(sc.transform(Xc[:,cols]))[:,1]; pe=m.predict_proba(sc.transform(Xe[:,cols]))[:,1]
    print(f"  ==> {lab:<40} calib AUC {AUC(yc,pc):.2f}  test AUC {AUC(ye,pe):.2f}")
    if len(cols)>2:
        # switching: Llama, low score -> Qwen (replace); test, shares
        qe_=np.array([x if x is not None else 0 for x in qe],float)
        for sh in (0.2,0.3,0.4):
            t=np.quantile(pc,sh); sw=pe<t
            acc=np.where(sw,qe_,ye).mean()
            rng=np.random.default_rng(0); rnd=np.mean([np.where(rng.permutation(sw),qe_,ye).mean() for _ in range(200)])
            print(f"      switch {100*sw.mean():.0f}% of test to Qwen: {100*acc:.1f}% (random same share {100*rnd:.1f}%; Llama alone {100*ye.mean():.1f}%, Qwen alone {100*qe_.mean():.1f}%)")

print("\n=== readable versions ===")
cols=list(range(len(names)))
sc=StandardScaler().fit(Xt[:,cols]); m=LogisticRegression(max_iter=3000).fit(sc.transform(Xt[:,cols]),yt)
w=m.coef_[0]/sc.scale_
print("logistic weights (per raw unit), largest effect first:")
eff=np.abs(m.coef_[0])
for j in np.argsort(-eff)[:6]: print(f"   {names[j]:<36} weight {m.coef_[0][j]:+.2f} (per std)")
from sklearn.tree import DecisionTreeClassifier, export_text
for depth in (2,3):
    t=DecisionTreeClassifier(max_depth=depth,min_samples_leaf=60,random_state=0).fit(Xt,yt)
    pc=t.predict_proba(Xc)[:,1]; pe=t.predict_proba(Xe)[:,1]
    print(f"\ntree depth {depth}: calib AUC {AUC(yc,pc):.2f} test AUC {AUC(ye,pe):.2f}")
    print(export_text(t,feature_names=names,decimals=3,show_weights=False))
    leaves=t.apply(Xe); 
    for lf in np.unique(leaves):
        msk=leaves==lf; print(f"   leaf {lf}: test n={msk.sum()}, Llama right {100*ye[msk].mean():.0f}%, Qwen right {100*np.mean([x or 0 for x,mm in zip(qe,msk) if mm]):.0f}%")
