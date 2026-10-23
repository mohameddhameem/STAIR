"""Cheap pre-retrieval routing and answer gain models; grouped train-only CV.

No gold, outcome, supporting-title or dataset difficulty fields enter features.
Cached model outcomes are supervised targets only. Test replay follows a disk lock.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[4]
sys.path.insert(0,str(HERE))
if __name__=='__main__':sys.modules['architecture']=sys.modules[__name__]
from suite import Data
from cascade_four_gpu import normalize, write_json

TOK=re.compile(r'\w+')
STOP=set('a an the of in on and or to is was are were did does do which what who when where how with by for'.split())

def words(text): return set(TOK.findall(text.lower()))-STOP

def document_stats(question, passages):
    q=words(question); scores=[]; title=[]
    for p in passages:
        scores.append(len(q & words(p))/max(1,len(q)))
        title.append(len(q & words(p.split(':',1)[0]))/max(1,len(q)))
    a=np.asarray(scores or [0.]); t=np.asarray(title or [0.])
    features=[len(passages),sum(map(len,passages)),len(q),len(question),
        a.max(),a.mean(),a.std(),float((a>0).mean()),t.max(),t.mean(),
        float((t>0).sum()),float(np.sort(a)[-min(5,len(a)):].mean())]
    features += [int(x in question.lower()) for x in ['same','both',' or ','first','before','after','how many','year','who','which','where','when']]
    return features, scores

def pre_features(inst):
    # Only explicitly allowed pre-retrieval fields; never copy source metadata.
    q=inst['question']; pool=[p[:600] for p in inst['ctxs_candidate'][:100]]
    nums,scores=document_stats(q,pool)
    order=sorted(range(len(pool)),key=lambda j:(-scores[j],j))[:8]
    titles='\n'.join(p.split(':',1)[0] for p in pool)
    text=(q+'\n')*3+'Candidate titles:\n'+titles+'\nLexical matches:\n'+'\n'.join(pool[j] for j in order)
    return {'text':text,'num':nums}

def evidence_only(g,branch):
    # Remove copied query before computing evidence overlap.
    text=g['evidence_'+branch]
    return text.split('Selected references:',1)[-1]

def answer_stats(q,ctx,ans):
    a=normalize(ans); aw=words(ans); ew=words(ctx); qw=words(q)
    return [len(ans),len(aw),len(aw&ew)/max(1,len(aw)),
        len(aw&qw)/max(1,len(aw)),int(bool(a) and a in normalize(ctx)),
        sum(c.isdigit() for c in ans),int(a in ['yes','no']),
        int(any(x in ans.lower() for x in ['unknown','cannot','insufficient','not enough'])),
        int('\n' in ans),int('?' in ans)]

def qa_features(g,branch,weak,confidence):
    q=g['question'];ctx=evidence_only(g,branch)
    nums,_=document_stats(q,[p for p in re.split(r'\[\d+\]',ctx) if p.strip()])
    nums += [int(branch=='S')]+answer_stats(q,ctx,weak)+list(confidence)
    return {'text':(q+'\n')*3+ctx+'\nWeak answer: '+weak,'num':nums}

def chooser_features(g,weak,strong,confidence):
    r=qa_features(g,'S',weak,confidence);ctx=evidence_only(g,'S')
    w=words(weak);s=words(strong)
    r['num'] += answer_stats(g['question'],ctx,strong)+[
        int(normalize(weak)==normalize(strong)),len(w&s)/max(1,len(w|s)),len(strong)-len(weak)]
    r['text']+='\nStrong answer: '+strong
    return r

class Features(BaseEstimator,TransformerMixin):
    def fit(self,rows,y=None):
        self.vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=5000,
            sublinear_tf=True,dtype=np.float32)
        xt=self.vec.fit_transform([r['text'] for r in rows])
        self.scaler=StandardScaler().fit([r['num'] for r in rows])
        self.svd=TruncatedSVD(n_components=min(48,xt.shape[1]-1),random_state=17).fit(xt)
        return self
    def transform(self,rows):
        x=self.vec.transform([r['text'] for r in rows]);n=self.scaler.transform([r['num'] for r in rows]).astype(np.float32)
        return hstack([x,csr_matrix(n)],format='csr'),np.column_stack([n,self.svd.transform(x)])

Features.__module__='architecture'

CONFIGS=['ridge10','ridge100','trees','hist']
def estimator(name,multi):
    if name.startswith('ridge'):return Ridge(alpha=float(name[5:]),solver='lsqr')
    if name=='trees':return ExtraTreesRegressor(n_estimators=180,max_depth=12,min_samples_leaf=12,n_jobs=2,random_state=17)
    m=HistGradientBoostingRegressor(max_iter=120,max_leaf_nodes=15,l2_regularization=5.,random_state=17)
    return MultiOutputRegressor(m,n_jobs=1) if multi else m

def fit_stage(name,rows,y,groups,out):
    errors={k:[] for k in CONFIGS}
    for fold,(tr,va) in enumerate(GroupKFold(3).split(rows['train'],y,groups)):
        f=Features().fit([rows['train'][j] for j in tr])
        xt=f.transform([rows['train'][j] for j in tr]);xv=f.transform([rows['train'][j] for j in va])
        for k in CONFIGS:
            m=estimator(k,y.ndim==2);idx=0 if k.startswith('ridge') else 1
            m.fit(xt[idx],y[tr]);p=m.predict(xv[idx])
            errors[k].append(float(np.mean((p-y[va])**2)))
        print(name,'fold',fold,errors,flush=True)
    chosen=min(CONFIGS,key=lambda k:np.mean(errors[k]))
    f=Features().fit(rows['train']);xt=f.transform(rows['train']);idx=0 if chosen.startswith('ridge') else 1
    m=estimator(chosen,y.ndim==2);m.fit(xt[idx],y)
    bundle={'features':f,'model':m,'matrix_index':idx,'selected':chosen,'grouped_cv_mse':errors}
    joblib.dump(bundle,out/(name+'.joblib'))
    # Test features/outcomes are deliberately deferred until all validation locks.
    pred=m.predict(f.transform(rows['val'])[idx])
    return bundle,pred

def policy_options(pre,qa,chooser,prices):
    """Small predeclared operating-point grids; no 2D exact threshold mining."""
    direct,review,fixed,post=prices
    lambdas=[0.,.05,.15,.4,.8,1.6,3.2]
    n=len(pre);opts=[]
    for j in range(4):opts.append(('direct4',f'constant_{j}',np.full(n,j),direct))
    for lam in lambdas:
        cells=np.argmax(np.clip(pre[:,:4],0,1)-lam*direct[None,:]/direct[3],axis=1)
        opts.append(('direct4',f'lambda_{lam}',cells,direct))
    gainw=qa[:,0];gains=qa[:,1];retgain=pre[:,4]
    for lr in lambdas:
        strongret=retgain>lr*(direct[3]-direct[1])/direct[3]
        for lq in lambdas:
            gain=np.where(strongret,gains,gainw)
            extra=np.where(strongret,review[3]-review[2],review[1]-review[0])
            cells=2*strongret.astype(int)+(gain>lq*extra/direct[3]).astype(int)
            opts.append(('pre_ret_qa',f'ret_{lr}_qa_{lq}',cells,review))
    opts.append(('pre_ret_qa','direct_SS_fallback',np.full(n,3),direct))
    # A fixed-S retrieval run reviews weak QA, then upgrades only if helpful.
    for lam in lambdas:
        cells=2+(gains>lam*(fixed[3]-fixed[2])/direct[3]).astype(int)
        opts.append(('fixed_S_qa',f'lambda_{lam}',cells,fixed))
    opts.append(('fixed_S_qa','direct_SS_fallback',np.full(n,3),direct))
    # Strong and weak QA are both paid for, regardless of retained answer.
    for threshold in [-.2,-.1,0.,.1,.2]:
        cells=np.where(chooser>threshold,2,3)
        opts.append(('post_answer_chooser',f'threshold_{threshold}',cells,post))
    opts.append(('post_answer_chooser','direct_SS_fallback',np.full(n,3),direct))
    return opts

def lock_options(options,em,sscost):
    locks={}
    for family in ['direct4','pre_ret_qa','fixed_S_qa','post_answer_chooser']:
        rec=[]
        for fam,key,cells,price in options:
            if fam!=family:continue
            rec.append({'family':fam,'key':key,'val_accuracy':float(em[np.arange(len(em)),cells].mean()),
                        'val_cost':float(price[cells].mean())})
        for objective in ['max_accuracy','ss_budget','cheap95']:
            eligible=rec if objective=='max_accuracy' else [r for r in rec if
                r['val_cost']<=sscost+1e-8] if objective=='ss_budget' else [r for r in rec if
                r['val_accuracy']>=.95*em[:,3].mean()-1e-12]
            key=(lambda r:(r['val_cost'],-r['val_accuracy'])) if objective=='cheap95' else (lambda r:(-r['val_accuracy'],r['val_cost']))
            locks[family+'_'+objective]=min(eligible,key=key)
    return locks

def metrics(em,strict,cells,price,sscost):
    n=len(cells);correct=em[np.arange(n),cells]
    delta=correct-em[:,3];rng=np.random.default_rng(17)
    boot=np.array([delta[rng.integers(0,n,n)].mean() for _ in range(3000)])
    return {'n':n,'accuracy':float(correct.mean()),'strict_accuracy':float(strict[np.arange(n),cells].mean()),
        'cost_pct_SS':float(100*price[cells].mean()/sscost),'mean_cost':float(price[cells].mean()),
        'delta_SS_pp':float(100*delta.mean()),'paired_interval_pp':(100*np.quantile(boot,[.025,.975])).tolist(),
        'picks':dict(Counter(np.asarray(['WW','WS','SW','SS'])[cells]))}

def main(out):
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output directory required')
    out.mkdir(parents=True)
    write_json(out/'status.json',{'phase':'preparing_features'})
    base=ROOT/'runs/nonllm_cascade_four_gpu_full_v1'
    data=Data(base,ROOT/'runs/nonllm_router_improvements_v2/samples')
    source=json.load(gzip.open(ROOT/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt'))
    strong={r['k']:r['v'] for r in __import__('cascade').load(base/'labels/full_run_p100/qa_strong.jsonl')}
    c=json.load(open(base/'token_costs.json'))['gate_eval_constants']
    rw,rs,ww,ws,sw,ss=[c['COST_'+k] for k in ['RET_W','RET_S','QA_W_GIVEN_W','QA_W_GIVEN_S','QA_S_GIVEN_W','QA_S_GIVEN_S']]
    direct=np.array([rw+ww,rw+sw,rs+ws,rs+ss]);review=direct+np.array([0,ww,0,ws])
    fixed=np.array([np.nan,np.nan,rs+ws,rs+ws+ss]);post=np.array([np.nan,np.nan,rs+ws+ss,rs+ws+ss])
    prices=(direct,review,fixed,post)
    manifest={'base':str(base),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'features':'Pre-router: question and capped candidate pool only; QA: paid selected evidence, weak answer, greedy confidence; chooser: additionally paid strong answer',
        'training':'3-fold grouped CV on train only selects model by MSE; then refit all train; no LLM weight changes',
        'models':CONFIGS,'costs':{'direct':direct.tolist(),'review':review.tolist(),'fixed_S':[fixed[2],fixed[3]],'post':[post[2],post[3]]},
        'selection':'Small predeclared lambda/threshold grids on validation; direct SS fallback; lock all 12 policies before test replay',
        'test':'Previously examined, descriptive; paired intervals unadjusted for multiple comparisons',
        'cost_scope':'Same mean parameter-weighted token ruler as prior runs; paid weak calls included; CPU router overhead excluded'}
    write_json(out/'manifest.json',manifest)
    def rows(split):
        ids=data.ids[split];p=[];q=[];ch=[]
        for i in ids:
            g=data.grid[i];p.append(pre_features(source[i]))
            for b in ['W','S']:
                ans=data.answer(i,b);conf=data.samples[i,b,-1]['confidence']
                q.append(qa_features(g,b,ans,conf))
                if b=='S':ch.append(chooser_features(g,ans,strong[f'{i}|strong'],conf))
        return p,q,ch
    train=rows('train');val=rows('val');emt=np.array([data.grid[i]['em_lenient'] for i in data.ids['train']],dtype=float)
    targets=[np.column_stack([emt,emt[:,3]-emt[:,1]]),
        np.column_stack([emt[:,1]-emt[:,0],emt[:,3]-emt[:,2]]).reshape(-1),emt[:,2]-emt[:,3]]
    bundles=[];pred=[]
    for j,name in enumerate(['pre','qa','chooser']):
        write_json(out/'status.json',{'phase':'grouped_cv_and_fit','stage':name})
        groups=np.repeat(data.ids['train'],2) if j==1 else np.array(data.ids['train'])
        b,p=fit_stage(name,{'train':train[j],'val':val[j]},targets[j],groups,out)
        bundles.append(b);pred.append(p)
    pred[1]=pred[1].reshape(-1,2)
    emv=np.array([data.grid[i]['em_lenient'] for i in data.ids['val']])
    locks=lock_options(policy_options(*pred,prices),emv,direct[3])
    write_json(out/'validation_lock.json',locks)
    write_json(out/'status.json',{'phase':'locked_test_replay'})
    test=rows('test');tp=[]
    for j,b in enumerate(bundles):tp.append(b['model'].predict(b['features'].transform(test[j])[b['matrix_index']]))
    tp[1]=tp[1].reshape(-1,2)
    options={(fam,key):(cells,price) for fam,key,cells,price in policy_options(*tp,prices)}
    ids=data.ids['test'];em=np.array([data.grid[i]['em_lenient'] for i in ids]);strict=np.array([data.grid[i]['em_strict'] for i in ids])
    results={}
    for name,lock in locks.items():
        cells,price=options[lock['family'],lock['key']]
        results[name]={'lock':lock,'test':metrics(em,strict,cells,price,direct[3])}
        np.savez(out/(name+'.npz'),ids=ids,cells=cells,correct=em[np.arange(len(ids)),cells],prices=price[cells])
    # Global selection also uses validation exclusively.
    global_selection={}
    for obj in ['max_accuracy','ss_budget','cheap95']:
        pool=[k for k in locks if k.endswith('_'+obj)]
        key=(lambda k:(locks[k]['val_cost'],-locks[k]['val_accuracy'])) if obj=='cheap95' else (lambda k:(-locks[k]['val_accuracy'],locks[k]['val_cost']))
        global_selection[obj]=min(pool,key=key)
    write_json(out/'results.json',{'models':[{k:v for k,v in b.items() if k in ['selected','grouped_cv_mse']} for b in bundles],
        'SS_accuracy':float(em[:,3].mean()),'global_validation_selection':global_selection,'results':results})
    report=['# Non-LLM routing architecture experiments','',
        'Grouped three-fold training CV selects model families. All operating points and global selections use validation only. Test is previously examined; results are descriptive.',
        'Costs include every paid model call, using the fixed mean parameter-weighted token ruler; CPU router overhead is excluded. No LLM weights were trained.', '',
        '| Policy | Val accuracy % | Val cost % SS | Test accuracy % | Strict % | Cost % SS | Delta SS pp |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name,r in results.items():
        l=r['lock'];t=r['test'];report.append(f"| {name} | {100*l['val_accuracy']:.2f} | {100*l['val_cost']/direct[3]:.2f} | {100*t['accuracy']:.2f} | {100*t['strict_accuracy']:.2f} | {t['cost_pct_SS']:.2f} | {t['delta_SS_pp']:+.2f} |")
    report+=['','Global validation selections:']+[f'- {k}: {v}' for k,v in global_selection.items()]
    report+=['','Models selected by train CV: '+', '.join(b['selected'] for b in bundles),
        '',f'Direct SS test accuracy: {100*em[:,3].mean():.2f}%.',
        'Direct SS fallback executes no weak calls and remains an explicitly deployable candidate. Post-answer choice pays both QA calls even when retaining the weak answer.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n')
    write_json(out/'status.json',{'phase':'complete','report':str(out/'REPORT.md')})
    print('\n'.join(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out)
