"""Predefined non-LLM ablations; choose everything on validation, then replay test."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'STAIR/nonLLM'))
from cascade import load, view, costs, pweak, compose
sys.path.insert(0, str(ROOT / 'STAIR/nonLLM/deployment'))
from cascade_four_gpu import normalize, score, write_json


def choose(em, pr, qw, qs, price):
    """Exact distinct-threshold search without materializing a million-row frontier."""
    n = len(em); best = None; bar = .95 * em[:, 3].mean()
    for tr in np.r_[np.unique(pr), np.inf]:
        r = pr < tr; q = np.where(r, qs, qw); order = np.argsort(-q); s = q[order]
        aw = em[np.arange(n), 2*r.astype(int)][order]
        ass = em[np.arange(n), 2*r.astype(int)+1][order]
        base = np.where(r, price[2], price[0])
        extra = np.where(r, price[3]-price[2], price[1]-price[0])[order]
        ks = np.r_[0, np.flatnonzero(np.r_[s[:-1] != s[1:], True])+1]
        acc = (ass.sum()+np.r_[0, np.cumsum(aw-ass)][ks])/n
        cost = (base.sum()+extra.sum()-np.r_[0, np.cumsum(extra)][ks])/n
        good = np.flatnonzero(acc >= bar-1e-12)
        if not len(good): continue
        j = min(good, key=lambda j: (cost[j], -acc[j]))
        rec = {'cost': float(cost[j]), 'accuracy': float(acc[j]), 'theta_ret': float(tr),
               'theta_qa': float(s[ks[j]-1]) if ks[j] else float('inf')}
        if best is None or (rec['cost'], -rec['accuracy']) < (best['cost'], -best['accuracy']): best = rec
    if best is None: raise ValueError('SS validation bar unreachable')
    return best


def rich(g, stage, branch, answer, conf=None, agreement=None):
    text, nums = view(g, stage, branch, answer if stage == 'qa' else None)
    ctx = g['evidence_W' if stage == 'ret' else 'evidence_'+branch]
    qw = set(normalize(g['question']).split()); ew = set(normalize(ctx).split())
    a = normalize(answer or ''); aw = set(a.split())
    nums += [int(branch == 'S'), len(ew), len(qw & ew)/max(1, len(qw)),
             int(' or ' in g['question'].lower()), int('both' in qw)]
    if stage == 'qa':
        nums += [int(a in normalize(ctx)), len(qw & aw)/max(1, len(aw)),
                 sum(c.isdigit() for c in a), int(any(s in (answer or '').lower() for s in ['unknown', 'cannot', 'not enough', 'insufficient'])),
                 int('\n' in (answer or '')), int('?' in (answer or ''))]
        nums += conf if conf is not None else [0.]*7
        nums += agreement if agreement is not None else [0.]*4
    return text, nums


class Data:
    def __init__(self, base, samples=None, include_test_samples=False):
        self.base = base
        self.grid = {g['idx']: g for g in load(base/'labels/full_run_p100/router_data_lenient.jsonl')}
        self.retlabels = {r['idx']: r['ret_target'] for r in load(base/'labels/ret_labels/ret_labels.jsonl')}
        self.qalabels = {(r['idx'], r['ret_choice']): r['qa_target'] for r in load(base/'labels/qa_labels/qa_labels.jsonl')}
        self.weak = {r['k']: r['v'] for r in load(base/'labels/full_run_p100/qa_weak.jsonl')}
        self.prices, self.direct_ss, self.scope = costs(base/'token_costs.json')
        self.ids = {s: sorted(i for i,g in self.grid.items() if g['split']==s) for s in ['train','val','test']}
        self.gold = None
        self.samples = {}
        if samples:
            files = list(samples.glob('trainval_[0-3].jsonl'))
            if include_test_samples: files += list(samples.glob('test_[0-3].jsonl'))
            for f in files:
                for r in load(f):
                    key = (r['idx'],r['branch'],r['sample'])
                    if key in self.samples: raise ValueError(f'Duplicate sample {key}')
                    self.samples[key] = r

    def answer(self, i, b): return self.weak[f"{i}|{'weak' if b=='W' else 'strong'}"]

    def dataset(self, stage, split, augmented=False, consensus=False):
        if augmented or consensus:
            if self.gold is None:
                import gzip
                with gzip.open(ROOT/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f: self.gold=json.load(f)
        rows=[]
        for i in self.ids[split]:
            g=self.grid[i]
            for b in (['W'] if stage=='ret' else ['W','S']):
                if consensus and stage=='qa':
                    records=[self.samples[i,b,k] for k in [-1,0,1,2,3]]
                    keys=[normalize(r['text']) for r in records]; counts=Counter(keys)
                    winner=max(counts,key=lambda s:(counts[s],-keys.index(s)))
                    index=keys.index(winner); r=records[index]
                    agreement=[counts[winner]/5,len(counts)/5,sum(k==keys[0] for k in keys)/5,
                               -sum((v/5)*np.log(v/5) for v in counts.values())]
                    variants=[(r['text'],r['confidence'],agreement,-2)]
                elif augmented and stage=='qa' and split=='train':
                    variants=[(self.samples[i,b,k]['text'],self.samples[i,b,k]['confidence'],None,k) for k in [-1,0,1,2,3]]
                else:
                    r=self.samples.get((i,b,-1),{})
                    variants=[(self.answer(i,b),r.get('confidence'),None,-1)]
                for ans, conf, agreement, k in variants:
                    text,num=rich(g,stage,b,ans,conf,agreement)
                    if stage=='ret':
                        weak=g['em_lenient'][1]; strong=g['em_lenient'][3]
                        original=self.retlabels[i]
                        outcome='W' if any(g['em_lenient'][:2]) or not any(g['em_lenient'][2:]) else 'S'
                    else:
                        j=0 if b=='W' else 2; weak=g['em_lenient'][j]; strong=g['em_lenient'][j+1]
                        original=self.qalabels[i,b]; outcome=original
                        if k!=-1:
                            weak=score(ans,self.gold[i]['answers'])[1]
                            original=outcome='W' if weak or not strong else 'S'
                    rows.append({'idx':i,'branch':b,'text':text,'num':num,'original':original,'outcome':outcome,
                                 'weak':weak,'strong':strong,'weight':1/len(variants)})
        return rows


def matrices(data, stage, augmented=False, consensus=False):
    rows={s:data.dataset(stage,s,augmented,consensus) for s in ['train','val','test'] if not(consensus and s=='test')}
    vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=2500,sublinear_tf=True,dtype=np.float32)
    vec.fit([r['text'] for r in rows['train']])
    scaler=StandardScaler().fit(np.array([r['num'] for r in rows['train']]))
    dense={s:scaler.transform(np.array([r['num'] for r in rs])).astype(np.float32) for s,rs in rows.items()}
    mixed={s:hstack([vec.transform([r['text'] for r in rs]),csr_matrix(dense[s])],format='csr') for s,rs in rows.items()}
    return rows,dense,mixed,vec,scaler


def scores_by_question(rows,p,ids,stage):
    d={(r['idx'],r['branch']):float(v) for r,v in zip(rows,p)}
    if stage=='ret': return np.array([d[i,'W'] for i in ids])
    return np.array([d[i,'W'] for i in ids]),np.array([d[i,'S'] for i in ids])


def evaluate(data,name,pred,out,price=None,grid_override=None,lock=None):
    price=data.prices if price is None else price
    grid=data.grid if grid_override is None else grid_override
    ids=data.ids['val']; em=np.array([grid[i]['em_lenient'] for i in ids])
    pr,qw,qs=pred['val']
    lock=choose(em,pr,qw,qs,price) if lock is None else lock
    v,vc=compose([grid[i] for i in ids],pr,qw,qs,lock,price,data.direct_ss)
    assert np.isclose(v['lenient_accuracy'],lock['accuracy']) and np.isclose(v['mean_cost'],lock['cost'])
    folder=out/name;folder.mkdir(parents=True,exist_ok=True)
    write_json(folder/'lock.json',lock)  # Before touching test outcomes.
    result={'name':name,'lock':lock,'val':v,'cost_scope':data.scope}
    if 'test' in pred:
        ids=data.ids['test'];pr,qw,qs=pred['test']
        test,tc=compose([grid[i] for i in ids],pr,qw,qs,lock,price,data.direct_ss)
        result['test']=test
        emt=np.array([grid[i]['em_lenient'] for i in ids])
        np.savez(folder/'test_predictions.npz',ids=ids,cells=tc,correct=emt[np.arange(len(ids)),tc])
    write_json(folder/'results.json',result)
    print(name,json.dumps(result),flush=True)
    return result


def classical(data,out,prefix='',augmented=False,consensus=False):
    mat={s:matrices(data,s,augmented,consensus) for s in ['ret','qa']}
    results=[]
    configs=[('rich_logreg','class','original'),('rich_trees','class','original'),
             ('rich_hist','class','original'),('outcome_logreg','class','outcome'),
             ('outcome_hist','class','outcome'),('gain_trees','gain','outcome'),
             ('cost_sensitive_logreg','cost','outcome')]
    if augmented: configs=[('augmentation_logreg','class','outcome'),('augmentation_hist','class','outcome')]
    if consensus: configs=[('consensus_logreg','class','outcome'),('consensus_hist','class','outcome')]
    for name,kind,target in configs:
        pred={s:[] for s in mat['ret'][0]}; metrics={}; saved={}
        for stage in ['ret','qa']:
            rows,dense,mixed,vec,scaler=mat[stage]
            weights=np.array([r['weight'] for r in rows['train']])
            if kind=='gain': y=np.array([r['strong']-r['weak'] for r in rows['train']])
            elif kind=='cost':
                delta=np.array([r['strong']-r['weak'] for r in rows['train']])
                extra=np.array([data.prices[3]-data.prices[1] if stage=='ret' else
                    data.prices[1]-data.prices[0] if r['branch']=='W' else data.prices[3]-data.prices[2] for r in rows['train']])
                advantage=delta-.4*extra/data.direct_ss
                y=np.where(advantage>0,'S','W');weights*=np.abs(advantage)+1e-6
            else:y=np.array([r[target] for r in rows['train']])
            trials=[]
            for hp in ([.1,1.,10.] if 'logreg' in name else [3,8]):
                if kind=='gain': model=ExtraTreesRegressor(n_estimators=160,max_depth=hp,min_samples_leaf=8,n_jobs=2,random_state=17)
                elif 'logreg' in name:model=LogisticRegression(C=hp,class_weight=None if kind=='cost' else 'balanced',max_iter=1200,solver='liblinear',random_state=17)
                elif 'hist' in name:model=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=2**hp if hp==3 else 31,l2_regularization=1.,random_state=17)
                else:model=ExtraTreesClassifier(n_estimators=160,max_depth=hp,min_samples_leaf=8,class_weight='balanced',n_jobs=2,random_state=17)
                xs=mixed if 'logreg' in name else dense
                model.fit(xs['train'],y,sample_weight=weights)
                p=-model.predict(xs['val']) if kind=='gain' else pweak(model,xs['val'])
                vy=np.array([r[target]=='W' for r in rows['val']])
                auc=float(roc_auc_score(vy,p)) if len(set(vy))>1 else .5
                trials.append((auc,model,hp))
            auc,model,hp=max(trials,key=lambda r:r[0]);metrics[stage]={'val_auc':auc,'parameter':hp}
            saved[stage]={'model':model,'vectorizer':vec,'scaler':scaler,'kind':kind,'numeric_only':'logreg' not in name}
            for split in pred:
                p=-model.predict(xs[split]) if kind=='gain' else pweak(model,xs[split])
                shaped=scores_by_question(rows[split],p,data.ids[split],stage)
                pred[split].extend([shaped] if stage=='ret' else list(shaped))
        if consensus:
            override,price=consensus_grid(data)
            result=evaluate(data,prefix+name,pred,out,price,override)
        else:result=evaluate(data,prefix+name,pred,out)
        result['stage_metrics']=metrics;write_json(out/(prefix+name)/'results.json',result)
        joblib.dump(saved,out/(prefix+name)/'models.joblib')
        results.append(result)
    return results


def consensus_grid(data,include_test=False):
    import copy
    grid={i:copy.deepcopy(g) for i,g in data.grid.items()}
    if data.gold is None:
        import gzip
        with gzip.open(ROOT/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:data.gold=json.load(f)
    totals={'W':[],'S':[]}
    for split in (['train','val','test'] if include_test else ['train','val']):
        for i in data.ids[split]:
            for b,j in [('W',0),('S',2)]:
                records=[data.samples[i,b,k] for k in [-1,0,1,2,3]]
                keys=[normalize(r['text']) for r in records];cnt=Counter(keys)
                winner=max(cnt,key=lambda s:(cnt[s],-keys.index(s)));r=records[keys.index(winner)]
                strict,loose=score(r['text'],data.gold[i]['answers'])
                grid[i]['em_lenient'][j]=loose;grid[i]['em_strict'][j]=strict
                if split=='val':totals[b].append(sum(r['n_in']+r['n_out'] for r in records[1:])*1.5)
    # Keep original cost ruler fixed. Add four sampled weak calls by validation mean.
    costs_file=json.load(open(data.base/'token_costs.json'))
    unit=min(v['params_times_tokens'] for v in costs_file['measured'].values())
    price=data.prices.copy();price[:2]+=np.mean(totals['W'])/unit;price[2:]+=np.mean(totals['S'])/unit
    return grid,price


def frozen_baseline(data,out):
    pred={}
    for split in ['val','test']:
        ids=data.ids[split];ps=[]
        for stage in ['ret','qa']:
            bundle=joblib.load(data.base/f'router/{stage}.joblib');vec=bundle['vectorizer'];model=bundle['model']
            rows=[]
            for i in ids:
                for b in (['W'] if stage=='ret' else ['W','S']):
                    text,num=view(data.grid[i],stage,b,data.answer(i,b) if stage=='qa' else None)
                    rows.append((i,b,text,num))
            x=np.array([r[3] for r in rows]) if bundle['family']=='random_forest' else vec.transform([r[2] for r in rows])
            p=pweak(model,x);d={(r[0],r[1]):v for r,v in zip(rows,p)}
            ps.extend([np.array([d[i,'W'] for i in ids])] if stage=='ret' else
                      [np.array([d[i,'W'] for i in ids]),np.array([d[i,'S'] for i in ids])])
        pred[split]=ps
    old=json.load(open(data.base/'router/theta_lock.json'));lock=old['families'][old['selected_family']]
    return evaluate(data,'frozen_baseline',pred,out,lock=lock)


def policy_methods(data,out,prefix=''):
    import torch
    torch.set_num_threads(2)
    mat={s:matrices(data,s) for s in ['ret','qa']}
    # Identical features and warm start for exact-reward and GRPO comparisons.
    x={s:{sp:torch.tensor(m.toarray(),dtype=torch.float32) for sp,m in mat[s][2].items()} for s in mat}
    ids=data.ids['train'];retrows=mat['ret'][0]['train'];qarows=mat['qa'][0]['train']
    assert [r['idx'] for r in retrows]==ids
    assert [(r['idx'],r['branch']) for r in qarows]==[(i,b) for i in ids for b in ['W','S']]
    yr=torch.tensor([r['outcome']=='S' for r in retrows],dtype=torch.float32)
    yq=torch.tensor([r['outcome']=='S' for r in qarows],dtype=torch.float32)
    em=torch.tensor([data.grid[i]['em_lenient'] for i in ids],dtype=torch.float32)
    cost=torch.tensor(data.prices/data.direct_ss,dtype=torch.float32)
    seeds=[17,29,43];results=[]
    for lam in [0.,.15,.4,.8]:
        predictions={m:{sp:[] for sp in ['val','test']} for m in ['exact_reward','grpo']}
        for seednum in seeds:
            torch.manual_seed(seednum)
            r=torch.nn.Linear(x['ret']['train'].shape[1],1);q=torch.nn.Linear(x['qa']['train'].shape[1],1)
            opt=torch.optim.Adam(list(r.parameters())+list(q.parameters()),lr=.01,weight_decay=.001)
            gen=torch.Generator().manual_seed(seednum)
            for ep in range(10):
                for batch in torch.randperm(len(ids),generator=gen).split(256):
                    qb=(batch[:,None]*2+torch.arange(2)[None]).flatten()
                    lr=torch.nn.functional.binary_cross_entropy_with_logits(r(x['ret']['train'][batch]).flatten(),yr[batch])
                    lq=torch.nn.functional.binary_cross_entropy_with_logits(q(x['qa']['train'][qb]).flatten(),yq[qb])
                    opt.zero_grad();(lr+lq).backward();opt.step()
            initialr={k:v.clone() for k,v in r.state_dict().items()};initialq={k:v.clone() for k,v in q.state_dict().items()}
            for method in ['exact_reward','grpo']:
                r.load_state_dict(initialr);q.load_state_dict(initialq)
                opt=torch.optim.Adam(list(r.parameters())+list(q.parameters()),lr=.002,weight_decay=.001)
                torch.manual_seed(seednum)
                gen=torch.Generator().manual_seed(seednum+100)
                for ep in range(30):
                    for batch in torch.randperm(len(ids),generator=gen).split(256):
                        qb=(batch[:,None]*2+torch.arange(2)[None]).flatten()
                        rp=r(x['ret']['train'][batch]).flatten().sigmoid().clamp(1e-5,1-1e-5)
                        qp=q(x['qa']['train'][qb]).reshape(-1,2).sigmoid().clamp(1e-5,1-1e-5)
                        probs=torch.stack([(1-rp)*(1-qp[:,0]),(1-rp)*qp[:,0],rp*(1-qp[:,1]),rp*qp[:,1]],1)
                        rewards=em[batch]-lam*cost[None]
                        entropy=-(probs*probs.clamp_min(1e-8).log()).sum(1).mean()
                        if method=='exact_reward':loss=-(probs*rewards).sum(1).mean()-.002*entropy
                        else:
                            old=probs.detach();actions=torch.multinomial(old,8,replacement=True)
                            reward=rewards.gather(1,actions)
                            adv=(reward-reward.mean(1,keepdim=True))/reward.std(1,keepdim=True,unbiased=False).clamp_min(1e-6)
                            oldp=old.gather(1,actions)
                            # Two PPO updates on the same sampled group; old policy fixed.
                            for update in range(2):
                                if update:
                                    rp=r(x['ret']['train'][batch]).flatten().sigmoid().clamp(1e-5,1-1e-5)
                                    qp=q(x['qa']['train'][qb]).reshape(-1,2).sigmoid().clamp(1e-5,1-1e-5)
                                    probs=torch.stack([(1-rp)*(1-qp[:,0]),(1-rp)*qp[:,0],rp*(1-qp[:,1]),rp*qp[:,1]],1)
                                ratio=probs.gather(1,actions)/oldp
                                loss=-torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean()
                                # Trust-region KL to the rollout policy; no LLM reference model.
                                kl=(old*(old.clamp_min(1e-8).log()-probs.clamp_min(1e-8).log())).sum(1).mean()
                                opt.zero_grad();(loss+.01*kl).backward();opt.step()
                            continue
                        opt.zero_grad();loss.backward();opt.step()
                folder=out/f'{prefix}{method}_lambda{lam}';folder.mkdir(parents=True,exist_ok=True)
                torch.save({'ret':r.state_dict(),'qa':q.state_dict(),'lambda':lam,'seed':seednum},folder/f'seed_{seednum}.pt')
                for split in ['val','test']:
                    with torch.no_grad():
                        pr=(1-r(x['ret'][split]).flatten().sigmoid()).numpy()
                        pq=(1-q(x['qa'][split]).reshape(-1,2).sigmoid()).numpy()
                    predictions[method][split].append((pr,pq[:,0],pq[:,1]))
        for method in predictions:
            pred={sp:[np.mean([ps[j] for ps in runs],axis=0) for j in range(3)] for sp,runs in predictions[method].items()}
            result=evaluate(data,f'{prefix}{method}_lambda{lam}',pred,out)
            result['training']={'seeds':seeds,'ensemble':'mean action probabilities','lambda':lam,'groups':8,
                                'nonLLM':'linear policies over TF-IDF and numeric features'}
            write_json(out/result['name']/'results.json',result);results.append(result)
        joblib.dump({s:{'vectorizer':mat[s][3],'scaler':mat[s][4]} for s in mat},out/f'{prefix}policy_preprocessing.joblib')
    return results


def fixed_mixture_accuracy(cost,prices,accuracies):
    candidates=[a for p,a in zip(prices,accuracies) if p<=cost+1e-12]
    for i in range(len(prices)):
        for j in range(i+1,len(prices)):
            if prices[i]==prices[j]:continue
            t=(cost-prices[i])/(prices[j]-prices[i])
            if 0<=t<=1:candidates.append((1-t)*accuracies[i]+t*accuracies[j])
    return max(candidates) if candidates else float('nan')


def report(out,data):
    results=[json.load(open(p)) for p in out.glob('*/results.json')]
    tested=[r for r in results if 'test' in r]
    best=min(results,key=lambda r:(r['val']['mean_cost'],-r['val']['lenient_accuracy'],r['name']))
    lines=['# Non-LLM router improvement experiments','',
           'All hyperparameters and thresholds use validation only. Existing test is descriptive and has been examined previously.',
           'Costs are parameter-weighted measured token proxies; CPU router overhead is excluded.',
           'GRPO here is a two-stage finite-action adaptation on linear non-LLM policies, not an LLM training run.',
           '',f"Cheapest validation-feasible method: {best['name']}.",'',
           '| Method | Val accuracy % | Val cost % direct SS | Test lenient % | Test strict % | Test cost % direct SS | Gain over cost-matched fixed mixture (pp) |',
           '|---|---:|---:|---:|---:|---:|---:|']
    fixed=np.array([data.grid[i]['em_lenient'] for i in data.ids['test']]).mean(0)
    for r in sorted(results,key=lambda r:r['name']):
        v=r['val'];t=r.get('test')
        gain=100*(t['lenient_accuracy']-fixed_mixture_accuracy(t['mean_cost'],data.prices,fixed)) if t else None
        tail=f"{100*t['lenient_accuracy']:.2f} | {100*t['strict_accuracy']:.2f} | {t['cost_pct_of_direct_SS']:.2f} | {gain:+.2f}" if t else 'pending | pending | pending | pending'
        lines.append(f"| {r['name']} | {100*v['lenient_accuracy']:.2f} | {v['cost_pct_of_direct_SS']:.2f} | {tail} |")
    base=next((r for r in tested if r['name']=='frozen_baseline'),None)
    if base:
        baseline=np.load(out/'frozen_baseline/test_predictions.npz')['correct'];rng=np.random.default_rng(707)
        bootstrap=rng.integers(0,len(baseline),size=(2000,len(baseline)))
        lines += ['','Paired question-bootstrap intervals against the frozen baseline (descriptive, no multiple-comparison correction):','']
        for r in tested:
            if r['name']=='frozen_baseline':continue
            y=np.load(out/r['name']/'test_predictions.npz')['correct'];delta=y-baseline
            lo,hi=np.quantile(delta[bootstrap].mean(1),[.025,.975])*100
            lines.append(f"- {r['name']}: accuracy difference {100*delta.mean():+.2f} pp, 95% interval [{lo:+.2f}, {hi:+.2f}].")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    write_json(out/'summary.json',{'selected_on_val':best['name'],'results':results})


def finish_consensus(data,out):
    override,price=consensus_grid(data,include_test=True)
    for name in ['sampled_consensus_logreg','sampled_consensus_hist']:
        saved=joblib.load(out/name/'models.joblib');pred={'val':[],'test':[]}
        for stage in ['ret','qa']:
            bundle=saved[stage]
            for split in pred:
                rows=data.dataset(stage,split,consensus=True)
                num=bundle['scaler'].transform(np.array([r['num'] for r in rows])).astype(np.float32)
                x=num if bundle['numeric_only'] else hstack([bundle['vectorizer'].transform([r['text'] for r in rows]),csr_matrix(num)],format='csr')
                p=pweak(bundle['model'],x);shaped=scores_by_question(rows,p,data.ids[split],stage)
                pred[split].extend([shaped] if stage=='ret' else list(shaped))
        lock=json.load(open(out/name/'lock.json'))
        evaluate(data,name,pred,out,price,override,lock)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--samples',type=Path);p.add_argument('--phase',choices=['cached','sampled','finish'],required=True)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    d=Data(a.base,a.samples if a.phase!='cached' else None,include_test_samples=a.phase=='finish')
    if a.phase=='cached':
        frozen_baseline(d,a.out);classical(d,a.out);policy_methods(d,a.out)
    elif a.phase=='sampled':
        classical(d,a.out,prefix='confidence_');classical(d,a.out,prefix='sampled_',augmented=True)
        classical(d,a.out,prefix='sampled_',consensus=True)
    else:finish_consensus(d,a.out)
    report(a.out,d)
