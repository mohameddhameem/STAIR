"""Patient-disjoint non-LLM replay, aligned with Jonathan's Training 1 / 3.

Strict A1 recipe: fresh classifier, sampled reward-minus-group-mean (no std),
entropy .03, no KL/PPO, AdamW1e-5, clip1, accum8, seed17, 40epochs.
Additional seeds0/1/2 are repeats, not an ensemble or a model-selection search.
"""
import argparse,ast,hashlib,json,random,re,shutil,subprocess,time
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

ARCHES={'linear':[],'mlp64':[64],'mlp128_64':[128,64]}
SEEDS=[17,0,1,2]
LR=1e-5;BETA=.03;LAMBDA=.1;ACCUM=8;GROUP=8;EPOCHS=40

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def table(out,setting):return json.loads((out/f'reference/results/medagentbench/training{setting}_table.json').read_text())
def branch(t,setting):
    cost=np.array([t['cw'],t['cw' if setting==1 else 'cw2']+t['cs']],dtype=np.float64)
    q=np.array([t['qw'],t['qs']],dtype=np.float64);x=cost/t['cs']
    return q,cost,q-LAMBDA*x-np.maximum(0,x-1)
def numeric(text):
    low=text.lower();words=low.split();rounds=re.findall(r'(\d+)\s+round',low)
    return [len(text),len(words),len(set(words)),text.count('\n'),
            sum(c.isdigit() for c in text)/max(1,len(text)),
            int('finished after' in low),int('not finish' in low),
            float(rounds[-1]) if rounds else 0.,low.count('http'),text.count('{')]
def transform(bundle,texts):
    return hstack([bundle['vectorizer'].transform(texts),csr_matrix(bundle['scaler'].transform(np.asarray([numeric(t) for t in texts],np.float32)))],format='csr',dtype=np.float32).toarray()

def verify_source(out):
    manifest=json.loads((out/'download_manifest.json').read_text());ref=out/'reference'
    for name,sha in manifest['source_hashes'].items():assert digest(ref/name)==sha
    assert digest(ref/'test_data_v2.json')==manifest['official_dataset_sha256']
    spl=ref/'common/stair_common/splits/medagentbench'
    result=subprocess.run([__import__('sys').executable,str(ref/'common/tools/medagentbench_patient_split.py'),'--check','--data',str(ref/'test_data_v2.json'),'--splits',str(spl)],capture_output=True,text=True,check=True)
    (out/'patient_split_check.txt').write_text(result.stdout)
    ids={k:(spl/f'{k}.txt').read_text().split() for k in ['calib','calib_sweep','test']}
    groups={'train':set(ids['calib'])-set(ids['calib_sweep']),'val':set(ids['calib_sweep']),'test':set(ids['test'])}
    tasks={r['id']:r for r in json.loads((ref/'test_data_v2.json').read_text())}
    patients={s:{tasks[tid]['eval_MRN'] for tid in tids if tasks[tid].get('eval_MRN') is not None} for s,tids in groups.items()}
    for a,b in [('train','val'),('train','test'),('val','test')]:assert not patients[a]&patients[b]
    common=json.loads((ref/'common/stair_common/splits/MANIFEST.json').read_text())
    assert digest(ref/'test_data_v2.json')==common['sources']['medagentbench']['sha256']
    for key in ids:assert digest(spl/f'{key}.txt')==common['parts'][f'medagentbench/{key}']['sha256']
    earlier={t['tid']:t for t in table(out,1)}
    for setting in [1,3]:
        rows=table(out,setting);assert len(rows)==300 and len({t['tid'] for t in rows})==300
        for t in rows:
            assert t['tid'] in groups[t['split']]
            assert t['view'].startswith(tasks[t['tid']]['instruction'])
            assert t['qw'] in [0,1] and t['qs'] in [0,1] and t['cw']>=0 and t['cs']>0
            if setting==3:
                assert 0<=t['cw2']<=t['cw']+.02
                for key in ['qw','qs','cw','cs','idx','split']:assert t[key]==earlier[t['tid']][key]
        # Execute only the author's pure reward/table builder, no model imports.
        path=ref/f'lab_package/medagentbench/training{setting}_trainer.py'
        tree=ast.parse(path.read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['hinge','build_rows']]
        env={'json':json,'TABLE':str(ref/f'results/medagentbench/training{setting}_table.json'),'LAMBDA':LAMBDA,'USER_TAIL':''}
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),env)
        lookup={t['tid']:t for t in rows}
        for part,rr in env['build_rows']().items():
            for r in rr:
                q,c,reward=branch(lookup[r['tid']],setting)
                np.testing.assert_allclose(reward,[r['rp']['W'],r['rp']['S']],rtol=0,atol=1e-12)
                np.testing.assert_allclose(c,[r['cost']['W'],r['cost']['S']],rtol=0,atol=1e-12)
    return {'source_commit':manifest['jonathan_commit'],'split_sizes':{s:len(v) for s,v in groups.items()},
            'patients_per_split':{s:len(v) for s,v in patients.items()},'all_three_patient_boundaries_disjoint':True,
            'split_files_match_manifest':True,'author_reward_builder_parity_rows':300,
            'two_replay_table_hashes_verified':True,'task_instruction_prefix_matches_official_dataset':True}

def prepare(out):
    assert not (out/'protocol.json').exists(),'Use a new run directory'
    verification=verify_source(out);(out/'data').mkdir()
    dims={}
    for setting in [1,3]:
        rows=table(out,setting);parts={s:[r for r in rows if r['split']==s] for s in ['train','val']}
        text=[t['view'] for t in parts['train']]
        vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,max_features=6000,sublinear_tf=True,dtype=np.float32).fit(text)
        scaler=StandardScaler().fit(np.asarray([numeric(t) for t in text],np.float32))
        bundle={'vectorizer':vectorizer,'scaler':scaler};joblib.dump(bundle,out/'data'/f'features_t{setting}.joblib')
        for part in parts:
            rr=parts[part];x=transform(bundle,[t['view'] for t in rr]);dims[str(setting)]=x.shape[1]
            qc=[branch(t,setting) for t in rr]
            np.savez(out/'data'/f't{setting}_{part}.npz',x=x,q=np.stack([a[0] for a in qc]),cost=np.stack([a[1] for a in qc]),reward=np.stack([a[2] for a in qc]))
    protocol={'source':json.loads((out/'download_manifest.json').read_text()),'dimensions':dims,'architectures':ARCHES,'seeds':SEEDS,
      'primary_seed':17,'training_settings':[1,3],'split':'90 train / 60 validation / 150 test, patients30/19/49; no overlap at all three boundaries',
      'inputs':'Only the exact view field: task instruction/context + weak attempt summary. T3 uses round2 view. No tid, idx, patient identifier field, correctness, risk score, cost or strong output is appended as feature. Patient references already in the same view remain as shown to LLM.',
      'representation':'Train-only TF-IDF unigram/bigram max6000 plus ten standardized deterministic view statistics; linear or ReLU MLP. No pretrained encoder or executor calls.',
      'alignment':'Same replay outcomes, observable view, split, cost equations, reward, A1 update, optimizer settings and fixed operating points as Jonathan T1/T3. Router representation/parameterization differ by design; non-LLM models have no pretrained knowledge.',
      'reward':'Q-.1*(C/cs)-max(0,C/cs-1), per-instance strong cost cs. T1 C=[cw,cw+cs]; T3 C=[cw,cw2+cs]. Benchmark pass/fail, not lenient QA.',
      'training':'Fresh model, no SFT; GRPO A1 reward minus group mean, NO standard deviation normalization; G8; entropy.03; AdamW lr1e-5, default weight_decay.01; accumulation8, per-row loss/8; clip1;40epochs; drop last2 row gradients per epoch exactly as author loop; Python random.shuffle and action draws.',
      'selection':'Check every10 optimizer updates; greedy delivered-path validation mean reward rounded4 decimals, pDeliver>0.5, strict improvement; no epoch0 candidate, same source code. Keep best of 40epochs. No test feedback.',
      'operating_points':'Prespecified .50 and .20; additionally select threshold maximizing validation mean delivered reward over0:.01:1, cost tie-break. All thresholds locked before per-router single test inference. Risk gate tau>=.8 evaluated from same stored probabilities, no training risk term.',
      'cost':'Measured normalized parameter-FLOP units supplied in table. Report sum policy costs / sum(cw+cs) on same test150 for both T1 andT3. This is weak-first always-escalate baseline, not direct strong cs. Decision-model and risk-scorer compute excluded, consistent with replay board; CPU decision latency separately reported.',
      'repeatability':'Seed17 matches campaign. Extra0/1/2 are independent repeats, mean/range reported, no probability ensemble or best-test seed selection.',
      'scope':'Two-action AI-only Training1 andTraining3; not the newer human-handoff Training5. Reuses pass/fail results from benchmark grader; no fresh live-environment validation.',
      'test_disclosure':'Reference test numbers were public and known; local test predictions computed once after all model/threshold locks. This is a paired replay experiment, not a newly secret benchmark.',
      'source_table_hashes':{str(s):digest(out/f'reference/results/medagentbench/training{s}_table.json') for s in [1,3]}}
    write(out/'protocol.json',protocol);write(out/'verification.json',verification)
    (out/'code').mkdir();shutil.copy2(__file__,out/'code'/Path(__file__).name)
    write(out/'status.json',{'phase':'prepared','fits':24})
    print('Prepared',dims,verification,flush=True)

def network(dims,arch):
    import torch
    layers=[];last=dims
    for width in ARCHES[arch]:layers += [torch.nn.Linear(last,width),torch.nn.ReLU()];last=width
    layers.append(torch.nn.Linear(last,2));return torch.nn.Sequential(*layers)

def a1_loss(logp,rewards,draws):
    # Draws [G] use the author's Python RNG, detached from the policy.
    r=rewards[draws];adv=r-r.mean()
    entropy=-(logp.exp()*logp).sum()
    return -(adv.detach()*logp[draws]).mean()-BETA*entropy

def train(out,setting,arch,seed):
    import torch
    torch.set_num_threads(1);random.seed(seed);torch.manual_seed(seed)
    folder=out/'training'/f't{setting}'/arch/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    data={s:dict(np.load(out/'data'/f't{setting}_{s}.npz')) for s in ['train','val']}
    x={s:torch.from_numpy(data[s]['x']) for s in data}
    reward={s:torch.tensor(data[s]['reward'],dtype=torch.float32) for s in data}
    net=network(x['train'].shape[1],arch)
    optimizer=torch.optim.AdamW(net.parameters(),lr=LR)
    def val():
        net.eval()
        with torch.no_grad():p=net(x['val']).softmax(-1)[:,0].numpy()
        a=(p<=.5).astype(int);ix=np.arange(len(a))
        net.train()
        return {'val_meanR_delivered':round(float(data['val']['reward'][ix,a].mean()),4),
                'val_acc':round(float(data['val']['q'][ix,a].mean()),4),
                'val_mean_cost':round(float(data['val']['cost'][ix,a].mean()),1),
                'mean_pDeliver':round(float(p.mean()),4)}
    start=time.monotonic();initial=val();best=-float('inf');step=0;curve=[];order=list(range(90));losses=[]
    for epoch in range(EPOCHS):
        random.shuffle(order);optimizer.zero_grad()
        for k,i in enumerate(order):
            lp=net(x['train'][i:i+1]).log_softmax(-1)[0]
            p=float(lp[0].detach().exp());draws=torch.tensor([0 if random.random()<p else 1 for _ in range(GROUP)])
            loss=a1_loss(lp,reward['train'][i],draws)/ACCUM
            assert torch.isfinite(loss)
            loss.backward();losses.append(float(loss.detach())*ACCUM)
            if (k+1)%ACCUM==0:
                torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step();optimizer.zero_grad();step+=1
                if step%10==0:
                    record={'epoch':epoch,'step':step,**val(),'train_surrogate_loss':float(np.mean(losses)),'seconds':time.monotonic()-start};losses=[];curve.append(record)
                    if record['val_meanR_delivered']>best:
                        best=record['val_meanR_delivered'];best_step=step;best_epoch=epoch
                        joblib.dump({'state':{k:v.detach().numpy().copy() for k,v in net.state_dict().items()},'setting':setting,'architecture':arch,'seed':seed,'epoch':epoch,'step':step},folder/'best.joblib')
                    write(folder/'curve.json',curve)
                    print(setting,arch,seed,epoch,step,best,flush=True)
    assert step==440 and len(curve)==44
    z=joblib.load(folder/'best.joblib');net.load_state_dict({k:torch.tensor(v) for k,v in z['state'].items()});net.eval()
    with torch.no_grad():pred=net(x['val']).softmax(-1)[:,0].numpy()
    np.save(folder/'val_pDeliver.npy',pred)
    write(folder/'complete.json',{'setting':setting,'architecture':arch,'seed':seed,'best_step':best_step,'best_epoch_zero_based':best_epoch,'last_step':step,'epochs':EPOCHS,'best_val_reward':best,'fresh_start_eval':initial,'seconds':time.monotonic()-start})

def choices(rows,setting,p,theta,gate=None):
    a=(np.asarray(p)<theta).astype(int)
    if gate is not None:a[np.asarray(gate,dtype=bool)]=1
    q=np.stack([branch(t,setting)[0] for t in rows]);c=np.stack([branch(t,setting)[1] for t in rows]);r=np.stack([branch(t,setting)[2] for t in rows]);ix=np.arange(len(rows))
    return a,q[ix,a],c[ix,a],r[ix,a]

def lock(out):
    assert not (out/'test_scores.csv').exists(),'Test inference already performed'
    locked=[];states=[]
    for setting in [1,3]:
        rows=[t for t in table(out,setting) if t['split']=='val']
        for arch in ARCHES:
            for seed in SEEDS:
                folder=out/'training'/f't{setting}'/arch/f'seed{seed}'
                assert (folder/'complete.json').exists();p=np.load(folder/'val_pDeliver.npy')
                candidates=[]
                for th in np.linspace(0,1,101):
                    a,q,c,r=choices(rows,setting,p,th)
                    candidates.append({'threshold':float(th),'val_reward':float(r.mean()),'val_accuracy':float(q.mean()),'val_cost':float(c.mean())})
                best=min(candidates,key=lambda v:(-v['val_reward'],v['val_cost'],v['threshold']))
                locked.append({'setting':setting,'architecture':arch,'seed':seed,'val_reward_threshold':best,'fixed_thresholds':[.5,.2]})
                states.append({'path':str((folder/'best.joblib').relative_to(out)),'sha256':digest(folder/'best.joblib')})
    write(out/'validation_lock.json',{'models':locked,'model_hashes':states,'all_24_fits_complete_before_test':True,'selection':'validation reward, then lower cost; no test score feedback'})

def metric(rows,setting,p,theta,taus,gate=False):
    high=np.array([taus[t['tid']]>=.8 for t in rows]);a,q,c,r=choices(rows,setting,p,theta,high if gate else None)
    denom=sum(t['cw']+t['cs'] for t in rows)
    return {'n':len(rows),'correct':int(q.sum()),'accuracy':float(q.mean()),'mean_cost':float(c.mean()),'total_cost':float(c.sum()),'cost_pct_always_escalate':100*float(c.sum())/denom,'mean_reward':float(r.mean()),
            'deliver':int((a==0).sum()),'escalate':int(a.sum()),'high_n':int(high.sum()),'high_accuracy':float(q[high].mean()),'high_delivered':int(((a==0)&high).sum()),'high_wrong_delivered':int(((a==0)&high&(q==0)).sum())},(a,q,c,r)

def evaluate(out):
    import torch
    torch.set_num_threads(1);assert not (out/'test_scores.csv').exists(),'Refusing a second test inference'
    p=json.loads((out/'protocol.json').read_text());locked=json.loads((out/'validation_lock.json').read_text());tau=json.loads((out/'reference/results/medagentbench/tau_scores_v2.json').read_text());taus={r['tid']:r['tau'] for r in tau['scores']}
    raw=[];result=[];details=[];latency={}
    for setting in [1,3]:
        rows=[r for r in table(out,setting) if r['split']=='test'];assert len(rows)==150
        bundle=joblib.load(out/'data'/f'features_t{setting}.joblib')
        for item in [v for v in locked['models'] if v['setting']==setting]:
            arch=item['architecture'];seed=item['seed'];folder=out/'training'/f't{setting}'/arch/f'seed{seed}'
            z=joblib.load(folder/'best.joblib');net=network(p['dimensions'][str(setting)],arch);net.load_state_dict({k:torch.tensor(v) for k,v in z['state'].items()});net.eval()
            # One pass over test per trained router, including feature extraction.
            t=time.perf_counter();x=torch.from_numpy(transform(bundle,[r['view'] for r in rows]))
            with torch.no_grad():prob=net(x).softmax(-1)[:,0].numpy()
            ms=(time.perf_counter()-t)*1000/len(rows);latency[(setting,arch,seed)]=ms
            raw.extend({'setting':setting,'architecture':arch,'seed':seed,'tid':r['tid'],'idx':r['idx'],'pDeliver':float(v)} for r,v in zip(rows,prob))
            for point,th in [('OP_A_fixed_050',.5),('OP_B_fixed_020',.2),('val_reward_selected',item['val_reward_threshold']['threshold'])]:
                for gate in [False,True]:
                    metrics,(a,q,c,reward)=metric(rows,setting,prob,th,taus,gate)
                    value={'setting':setting,'architecture':arch,'seed':seed,'point':point,'threshold':th,'risk_gate':gate,'ms_feature_and_batch_decision_per_task':ms,**metrics};result.append(value)
                    details.extend({'setting':setting,'architecture':arch,'seed':seed,'point':point,'risk_gate':gate,'tid':r['tid'],'action':int(aa),'correct':int(qq),'cost':float(cc),'reward':float(rr)} for r,aa,qq,cc,rr in zip(rows,a,q,c,reward))
    # Reference LLM outcomes are evaluated only after all non-LLM locks/inference.
    reference=[]
    for setting in [1,3]:
        rows=[r for r in table(out,setting) if r['split']=='test'];scores={r['tid']:r['pDeliver'] for r in map(json.loads,(out/f'reference/results/medagentbench/training{setting}_test_scores.jsonl').read_text().splitlines())}
        prob=np.array([scores[r['tid']] for r in rows])
        for point,th in [('OP_A_fixed_050',.5),('OP_B_fixed_020',.2)]:
            for gate in [False,True]:
                metrics,_=metric(rows,setting,prob,th,taus,gate);reference.append({'setting':setting,'point':point,'threshold':th,'risk_gate':gate,**metrics})
        for name,prob in [('always_weak',np.ones(150)),('always_escalate',np.zeros(150))]:
            metrics,_=metric(rows,setting,prob,.5,taus);reference.append({'setting':setting,'point':name,'risk_gate':False,**metrics})
        oracle=[]
        for row in rows:
            q,c,_=branch(row,setting)
            action=np.where(q.astype(bool),c,np.inf).argmin() if q.any() else c.argmin()
            oracle.append(float(action==0))
        metrics,_=metric(rows,setting,np.array(oracle),.5,taus);reference.append({'setting':setting,'point':'cheapest_correct_oracle','risk_gate':False,**metrics})
    assert round(next(r for r in reference if r['setting']==1 and r['point']=='OP_A_fixed_050' and not r['risk_gate'])['accuracy']*100,2)==61.33
    assert round(next(r for r in reference if r['setting']==3 and r['point']=='OP_A_fixed_050' and not r['risk_gate'])['accuracy']*100,2)==63.33
    pd.DataFrame(raw).to_csv(out/'test_scores.csv',index=False)
    pd.DataFrame(details).to_csv(out/'per_task.csv.gz',index=False)
    write(out/'results.json',result);pd.DataFrame(result).to_csv(out/'results.csv',index=False);write(out/'llm_reference.json',reference)
    summarise(out)

def summarise(out):
    result=json.loads((out/'results.json').read_text());refs=json.loads((out/'llm_reference.json').read_text());detail=pd.read_csv(out/'per_task.csv.gz');raw=pd.read_csv(out/'test_scores.csv')
    assert len(raw)==24*150 and not raw.duplicated(['setting','architecture','seed','tid']).any()
    assert len(detail)==24*150*6
    # Per-task cost/quality reconstructed from canonical tables, not summaries.
    lookup={s:{r['tid']:r for r in table(out,s)} for s in [1,3]}
    for r in detail.itertuples():
        q,c,reward=branch(lookup[r.setting][r.tid],r.setting)
        assert r.correct==q[r.action] and abs(r.cost-c[r.action])<1e-7 and abs(r.reward-reward[r.action])<1e-7
    for r in result:
        chosen=detail[(detail.setting==r['setting'])&(detail.architecture==r['architecture'])&(detail.seed==r['seed'])&(detail.point==r['point'])&(detail.risk_gate==r['risk_gate'])]
        assert len(chosen)==150 and chosen.correct.sum()==r['correct'] and abs(chosen.cost.sum()-r['total_cost'])<1e-6
    repeat=[]
    for setting in [1,3]:
        for arch in ARCHES:
            for point in ['OP_A_fixed_050','OP_B_fixed_020','val_reward_selected']:
                rr=[r for r in result if r['setting']==setting and r['architecture']==arch and r['seed'] in [0,1,2] and r['point']==point and not r['risk_gate']]
                repeat.append({'setting':setting,'architecture':arch,'point':point,'seeds':[0,1,2],
                    'accuracy_mean':float(np.mean([r['accuracy'] for r in rr])),'accuracy_min':min(r['accuracy'] for r in rr),'accuracy_max':max(r['accuracy'] for r in rr),
                    'cost_pct_mean':float(np.mean([r['cost_pct_always_escalate'] for r in rr]))})
    write(out/'seed_repeats.json',repeat)
    verification=json.loads((out/'verification.json').read_text());verification.update({'completed_grpo_fits':24,'test_inference_scores':len(raw),'audited_per_task_policy_rows':len(detail),'paired_reference_accuracy_reproduced':True,'per_task_cost_correctness_and_aggregates_verified':True,'no_risk_scores_or_grader_fields_as_features':True})
    write(out/'verification.json',verification)
    report=['# MedAgentBench non-LLM replay: Jonathan-aligned Training 1 and 3','',
        'Same committed replay views, outcomes, 90/60/150 patient-disjoint split, benchmark pass/fail grading, reward/cost accounting, and fresh GRPO A1 mechanics. Primary seed17; seeds0/1/2 are independent repeats. No executor calls or training, FHIR server, raw episode generation, or SFT warm start.',
        '', '## What is aligned','',
        'Weak Qwen2.5-7B vs strong Qwen2.5-72B-AWQ; one deliver/escalate decision. Training1 sees completed weak attempt; Training3 sees round2 view, deliver pays full weak cost and escalate pays round2 weak cost + full strong cost. Both normalize report cost by the same full-episode always-escalate sum(cw+cs). This is not HotpotQA Table5/929 or direct-strong-only accounting.',
        '', 'Fresh classifier, G8; advantage=reward-group mean, no std normalization, no KL/PPO; entropy.03; AdamW1e-5, accum8, clip1,40epochs; best rounded validation delivered-path reward every10 optimizer updates. Features fitted only on train90; only view content is used, with TF-IDF and numerical text statistics. Parameterization and pretrained knowledge differ from the LLM router by design.',
        '', '## Primary seed17, fixed theta0.50, no risk gate','',
        '| Setting | Router | Accuracy | Cost / full always-escalate | Deliver /150 |','|---|---|---:|---:|---:|']
    for setting in [1,3]:
        for name,point in [('Always weak','always_weak'),('Always escalate','always_escalate'),('Oracle','cheapest_correct_oracle'),('Jonathan LLM','OP_A_fixed_050')]:
            r=next(r for r in refs if r['setting']==setting and r['point']==point and not r['risk_gate']);report.append(f"| T{setting} | {name} | {r['accuracy']*100:.2f}% | {r['cost_pct_always_escalate']:.2f}% | {r['deliver']} |")
        for arch in ARCHES:
            r=next(r for r in result if r['setting']==setting and r['architecture']==arch and r['seed']==17 and r['point']=='OP_A_fixed_050' and not r['risk_gate']);report.append(f"| T{setting} | {arch} | {r['accuracy']*100:.2f}% | {r['cost_pct_always_escalate']:.2f}% | {r['deliver']} |")
    report+=['','## Primary seed17, threshold selected on validation reward','',
       '| Setting | Router | Theta | Test accuracy | Cost / full always-escalate |','|---|---|---:|---:|---:|']
    for r in result:
        if r['seed']==17 and r['point']=='val_reward_selected' and not r['risk_gate']:
            report.append(f"| T{r['setting']} | {r['architecture']} | {r['threshold']:.2f} | {r['accuracy']*100:.2f}% | {r['cost_pct_always_escalate']:.2f}% |")
    report+=['','## Interpretation and limits','',
       'Fixed theta0.50 matches the reference decision rule, but classifiers do not share LLM probability calibration. The additional validation-reward threshold is chosen only on calib_sweep, then applied to test. All models/thresholds were locked before a single test inference per model. Gate and OP-B views reuse the same probabilities. The source campaign test results were already public: this is a paired replay, not a freshly secret test.',
       '', 'Decision-model FLOPs and frozen risk-score generation are excluded from the supplied cost proxy, as on the replay board. Batch CPU feature+decision timing is measured separately; it is not streaming per-request latency. No clinical performance or safety claim follows from this small replay. Uncertainty should be clustered by patient rather than task.',
       '', 'Scope is Training1 andTraining3, two AI actions. The newer Training5 human-handoff policies require different action space, failure penalties and human-cost assumptions and are not pooled here. Full seed/point/risk-gate results are in results.csv; independent0/1/2 mean/range in seed_repeats.json.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n');write(out/'status.json',{'phase':'complete','fits':24,'test_n':150})
    print('\n'.join(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',choices=['prepare','train','lock','evaluate'],required=True);ap.add_argument('--setting',type=int,choices=[1,3]);ap.add_argument('--architecture',choices=list(ARCHES));ap.add_argument('--seed',type=int,default=17);a=ap.parse_args()
    if a.phase=='prepare':prepare(a.out)
    elif a.phase=='train':train(a.out,a.setting,a.architecture,a.seed)
    elif a.phase=='lock':lock(a.out)
    else:evaluate(a.out)
