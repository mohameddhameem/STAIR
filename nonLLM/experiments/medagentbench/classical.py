"""Supervised cost-sensitive classical routers on the frozen MedAgentBench replay.

Labels are argmax of the same action reward; sample weight is action regret.
This is a new training method, not a claim that RF/SVM were trained with GRPO.
"""
import argparse,json,shutil,time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC,SVC
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from classical_features import NumericTextStats
import replay

FAMILIES=['random_forest','extra_trees','decision_tree','linear_svm','rbf_svm','logistic_regression','multinomial_nb']
SEEDS=[17,0,1,2]

def target(reward):
    reward=np.asarray(reward);y=reward.argmax(1)
    gap=np.abs(reward[:,0]-reward[:,1]);weight=np.maximum(gap,1e-8)
    return y,weight/weight.mean()

def candidates(family,seed):
    if family in ['random_forest','extra_trees']:
        cls=RandomForestClassifier if family=='random_forest' else ExtraTreesClassifier
        return [({'depth':depth,'leaf':leaf},cls(n_estimators=300,max_depth=depth,min_samples_leaf=leaf,max_features='sqrt',random_state=seed,n_jobs=1)) for depth,leaf in [(3,1),(6,1),(None,1),(None,3)]]
    if family=='decision_tree':return [({'depth':d},DecisionTreeClassifier(max_depth=d,min_samples_leaf=2,random_state=seed)) for d in [2,4,6,None]]
    if family=='linear_svm':return [({'C':c},LinearSVC(C=c,dual='auto',max_iter=10000,random_state=seed)) for c in [.1,1.,10.]]
    if family=='rbf_svm':return [({'C':c,'gamma':g},SVC(C=c,gamma=g,kernel='rbf',probability=False,random_state=seed)) for c in [1.,10.] for g in ['scale',.1]]
    if family=='logistic_regression':return [({'C':c},LogisticRegression(C=c,max_iter=3000,random_state=seed)) for c in [.1,1.,10.]]
    return [({'alpha':a},MultinomialNB(alpha=a)) for a in [.1,1.,10.]]

def make_pipeline(source,setting,model,family):
    # clone removes all fitted state. Calibration-fold vocabularies and scalers
    # are therefore fitted only on that fold's training patients.
    original=joblib.load(source/'data'/f'features_t{setting}.joblib')
    vectorizer=clone(original['vectorizer'])
    feature=vectorizer if family=='multinomial_nb' else FeatureUnion([
        ('text',vectorizer),('numeric',Pipeline([('extract',NumericTextStats()),('scale',StandardScaler())]))])
    return Pipeline([('features',feature),('model',model)])

def fit_candidate(pipe,family,text,y,weight,groups,seed):
    if 'svm' not in family:
        pipe.fit(text,y,model__sample_weight=weight)
        return {'pipeline':pipe,'calibrator':None},[]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(np.zeros(len(y)),y,groups))
    oof=np.full(len(y),np.nan);audits=[]
    for fold,(fit,held) in enumerate(folds):
        assert not set(groups[fit])&set(groups[held])
        assert len(np.unique(y[fit]))==2
        fold_weight=weight[fit]/weight[fit].mean()
        estimator=clone(pipe);estimator.fit([text[i] for i in fit],y[fit],model__sample_weight=fold_weight)
        oof[held]=estimator.decision_function([text[i] for i in held])
        audits.append({'fold':fold,'fit_indices':fit.tolist(),'held_indices':held.tolist(),
                       'fit_patients':len(set(groups[fit])),'held_patients':len(set(groups[held])),
                       'patient_overlap':0,'feature_fitting':'entire feature pipeline refit on fold training views only'})
    assert np.isfinite(oof).all()
    # Group-held-out logistic/Platt-style calibration, never SVC(probability=True)
    # whose internal folds could mix records belonging to the same patient.
    calibrator=LogisticRegression(C=1.,max_iter=2000,random_state=seed)
    calibrator.fit(oof[:,None],y,sample_weight=weight)
    pipe.fit(text,y,model__sample_weight=weight)
    return {'pipeline':pipe,'calibrator':calibrator},audits

def predict(bundle,text):
    if bundle['calibrator'] is not None:
        margin=bundle['pipeline'].decision_function(text)
        model=bundle['calibrator'];prob=model.predict_proba(margin[:,None])
    else:model=bundle['pipeline'];prob=model.predict_proba(text)
    classes=model.classes_;return prob[:,int(np.flatnonzero(classes==0)[0])]

def prepare(source,out):
    assert not out.exists(),'Use a new output directory';out.mkdir(parents=True)
    assert json.loads((source/'status.json').read_text())['phase']=='complete'
    integrity=replay.verify_source(source)
    p={'source_run':str(source.resolve()),'source_protocol_sha256':replay.digest(source/'protocol.json'),
       'source_table_hashes':{str(s):replay.digest(source/f'reference/results/medagentbench/training{s}_table.json') for s in [1,3]},
       'families':FAMILIES,'seeds':SEEDS,'primary_seed':17,'split':'90 train / 60 validation / 150 test, same patient-disjoint frozen split',
       'input':'Only exact view field, same as LLM/GRPO. TF-IDF+10 deterministic text statistics; NB text-only to ensure nonnegative features.',
       'training':'Supervised cost-sensitive action imitation, not GRPO. y=argmax action reward; ties W. Weight=abs(RW-RS), normalized by train mean. Reward exactly same as replay/LLM recipe.',
       'svm_calibration':'3-fold patient-grouped out-of-fold margins; entire feature pipeline refitted inside each fold. Logistic calibration C1 fitted on OOF margins with train-only regret weights, then base refit on90 train. No outer-val/test calibration and no SVC probability=True.',
       'selection':'Per family/seed select hyperparameters by fixed theta.50 validation mean reward, then lower cost, then earlier candidate. Lock fixed theta.50/.20 and validation-reward-selected threshold0:.01:1 before all test inference.',
       'evaluation':'Actual trained classifier reads test view once per selected model, then same frozen weak/strong outcomes/costs are replayed. Both no risk gate and same frozen tau>=.8 forced-escalation gate.',
       'comparison':'Identical task executor outputs, observable inputs, split, objective, cost units and readout; different router training algorithm and architecture. No claim of same GRPO training.',
       'scope':'Training1 complete weak attempt, Training3 round2; no human Training5 and no fresh executor episodes.',
       'test_disclosure':'Previously inspected public test, exploratory/descriptive repeated-method comparison; no new secret holdout or test-based model selection.',
       'cost':'sum selected measured costs / sum full cw+cs for same150 tasks. Router CPU work and cached tau generation excluded consistently; timing recorded separately.'}
    replay.write(out/'protocol.json',p);replay.write(out/'verification.json',integrity)
    (out/'code').mkdir()
    for name in ['classical.py','classical_features.py','replay.py']:shutil.copy2(Path(__file__).with_name(name),out/'code'/name)
    replay.write(out/'status.json',{'phase':'prepared','selected_models':56})

def train(source,out,setting,family,seed):
    folder=out/'training'/f't{setting}'/family/f'seed{seed}';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'complete.json').exists():return
    parts={s:[r for r in replay.table(source,setting) if r['split']==s] for s in ['train','val']}
    text={s:[r['view'] for r in parts[s]] for s in parts}
    rewards=np.stack([replay.branch(t,setting)[2] for t in parts['train']]);y,weight=target(rewards)
    data={r['id']:r for r in json.loads((source/'reference/test_data_v2.json').read_text())}
    groups=np.array([data[r['tid']].get('eval_MRN') or 'no_patient:'+r['tid'] for r in parts['train']])
    records=[];best=None;start=time.monotonic()
    for index,(param,model) in enumerate(candidates(family,seed)):
        pipe=make_pipeline(source,setting,model,family)
        bundle,audits=fit_candidate(pipe,family,text['train'],y,weight,groups,seed)
        pv=predict(bundle,text['val']);a,q,c,r=replay.choices(parts['val'],setting,pv,.5)
        record={'candidate':index,'parameter':param,'val_reward_fixed050':float(r.mean()),'val_accuracy_fixed050':float(q.mean()),'val_cost_fixed050':float(c.mean()),'seconds':time.monotonic()-start}
        records.append(record);key=(record['val_reward_fixed050'],-record['val_cost_fixed050'],-index)
        if best is None or key>best[0]:
            best=(key,record);joblib.dump({**bundle,'family':family,'seed':seed,'setting':setting,'parameter':param},folder/'best.joblib')
            np.save(folder/'val_pDeliver.npy',pv);replay.write(folder/'calibration_folds.json',audits)
        print(setting,family,seed,param,round(record['val_reward_fixed050'],5),flush=True)
    replay.write(folder/'candidates.json',records)
    replay.write(folder/'complete.json',{'setting':setting,'family':family,'seed':seed,'selected':best[1],'candidate_count':len(records),'train_rows':90,'val_rows':60,'routing_label_counts':np.bincount(y,minlength=2).tolist(),'sample_weight_min':float(weight.min()),'sample_weight_max':float(weight.max()),'seconds':time.monotonic()-start})

def lock(source,out):
    assert not (out/'test_scores.csv').exists();locked=[];hashes=[]
    for setting in [1,3]:
        rows=[r for r in replay.table(source,setting) if r['split']=='val']
        for family in FAMILIES:
            for seed in SEEDS:
                folder=out/'training'/f't{setting}'/family/f'seed{seed}'
                done=json.loads((folder/'complete.json').read_text());p=np.load(folder/'val_pDeliver.npy');thresholds=[]
                for th in np.linspace(0,1,101):
                    a,q,c,r=replay.choices(rows,setting,p,th)
                    thresholds.append({'threshold':float(th),'val_reward':float(r.mean()),'val_accuracy':float(q.mean()),'val_cost':float(c.mean())})
                selected=min(thresholds,key=lambda v:(-v['val_reward'],v['val_cost'],v['threshold']))
                locked.append({'setting':setting,'family':family,'seed':seed,'parameter':done['selected']['parameter'],'fixed_validation':done['selected'],'selected_threshold':selected})
                hashes.append({'path':str((folder/'best.joblib').relative_to(out)),'sha256':replay.digest(folder/'best.joblib')})
    global_choice=[]
    for setting in [1,3]:
        for seed in SEEDS:
            available=[v for v in locked if v['setting']==setting and v['seed']==seed]
            for point,field in [('OP_A_fixed_050','fixed_validation'),('val_reward_selected','selected_threshold')]:
                if field=='fixed_validation':key=lambda v:(-v[field]['val_reward_fixed050'],v[field]['val_cost_fixed050'],v['family'])
                else:key=lambda v:(-v[field]['val_reward'],v[field]['val_cost'],v['family'])
                chosen=min(available,key=key);global_choice.append({'setting':setting,'seed':seed,'point':point,'family':chosen['family']})
    replay.write(out/'validation_lock.json',{'models':locked,'model_hashes':hashes,'global_family_choices':global_choice,'all_56_models_complete_before_test':True})

def evaluate(source,out):
    assert not (out/'test_scores.csv').exists(),'Refusing second test inference'
    locks=json.loads((out/'validation_lock.json').read_text());raw=[];details=[];results=[]
    tau={r['tid']:r['tau'] for r in json.loads((source/'reference/results/medagentbench/tau_scores_v2.json').read_text())['scores']}
    for setting in [1,3]:
        rows=[r for r in replay.table(source,setting) if r['split']=='test'];text=[r['view'] for r in rows]
        for item in [r for r in locks['models'] if r['setting']==setting]:
            family=item['family'];seed=item['seed'];folder=out/'training'/f't{setting}'/family/f'seed{seed}'
            bundle=joblib.load(folder/'best.joblib');start=time.perf_counter();p=predict(bundle,text);ms=(time.perf_counter()-start)*1000/len(rows)
            raw.extend({'setting':setting,'family':family,'seed':seed,'tid':r['tid'],'pDeliver':float(pp)} for r,pp in zip(rows,p))
            for point,th in [('OP_A_fixed_050',.5),('OP_B_fixed_020',.2),('val_reward_selected',item['selected_threshold']['threshold'])]:
                for gate in [False,True]:
                    metric,(a,q,c,reward)=replay.metric(rows,setting,p,th,tau,gate)
                    results.append({'setting':setting,'family':family,'seed':seed,'point':point,'threshold':th,'risk_gate':gate,'ms_features_and_batch_prediction_per_task':ms,**metric})
                    details.extend({'setting':setting,'family':family,'seed':seed,'point':point,'risk_gate':gate,'tid':t['tid'],'action':int(aa),'correct':int(qq),'cost':float(cc),'reward':float(rr)} for t,aa,qq,cc,rr in zip(rows,a,q,c,reward))
    assert len(raw)==56*150
    pd.DataFrame(raw).to_csv(out/'test_scores.csv',index=False);pd.DataFrame(details).to_csv(out/'per_task.csv.gz',index=False)
    replay.write(out/'results.json',results);pd.DataFrame(results).to_csv(out/'results.csv',index=False)
    summarise(source,out)

def summarise(source,out):
    result=json.loads((out/'results.json').read_text());locks=json.loads((out/'validation_lock.json').read_text());detail=pd.read_csv(out/'per_task.csv.gz')
    assert len(result)==56*6 and len(detail)==56*6*150
    lookup={s:{t['tid']:t for t in replay.table(source,s)} for s in [1,3]}
    for r in detail.itertuples():
        q,c,reward=replay.branch(lookup[r.setting][r.tid],r.setting)
        assert r.correct==q[r.action] and abs(r.cost-c[r.action])<1e-7 and abs(r.reward-reward[r.action])<1e-7
    candidates=0;fold_count=0
    for item in locks['model_hashes']:assert replay.digest(out/item['path'])==item['sha256']
    for f in sorted((out/'training').glob('t*/*/seed*')):
        done=json.loads((f/'complete.json').read_text());rr=json.loads((f/'candidates.json').read_text());best=min(rr,key=lambda r:(-r['val_reward_fixed050'],r['val_cost_fixed050'],r['candidate']))
        assert best['candidate']==done['selected']['candidate'];candidates+=len(rr)
        for fold in json.loads((f/'calibration_folds.json').read_text()):assert fold['patient_overlap']==0;fold_count+=1
    repeats=[]
    for s in [1,3]:
        for family in FAMILIES:
            for point in ['OP_A_fixed_050','val_reward_selected']:
                for gate in [False,True]:
                    rr=[r for r in result if r['setting']==s and r['family']==family and r['seed'] in [0,1,2] and r['point']==point and r['risk_gate']==gate]
                    repeats.append({'setting':s,'family':family,'point':point,'risk_gate':gate,'seeds':[0,1,2],
                        'accuracy_mean':float(np.mean([r['accuracy'] for r in rr])),'accuracy_min':min(r['accuracy'] for r in rr),'accuracy_max':max(r['accuracy'] for r in rr),'cost_pct_mean':float(np.mean([r['cost_pct_always_escalate'] for r in rr]))})
    replay.write(out/'seed_repeats.json',repeats)
    verification=json.loads((out/'verification.json').read_text());verification.update({'selected_classifiers':56,'hyperparameter_candidate_fits':candidates,'selected_svm_patient_folds':fold_count,'test_inference_scores':56*150,'audited_policy_rows':len(detail),'all_per_task_rewards_and_costs_match_reference':True,'all_model_hashes_match_pretest_lock':True,'all_hyperparameters_selected_on_validation_only':True})
    replay.write(out/'verification.json',verification)
    report=['# Classical non-LLM MedAgentBench routers','',
       'Same Jonathan Training1/3 replay, view fields, reward/cost equations, and patient-disjoint90/60/150 split as the previous GRPO experiment. These methods use supervised reward-optimal action imitation, not GRPO: y=argmax(RW,RS), sample weight=absolute reward gap normalized by train mean. Actual classifiers run on test views once; executor outcomes are replayed. No new medical episodes or task-model training.',
       '', '## Model selection and calibration','',
       'TF-IDF+view statistics, fitted on90 training tasks only; NB text-only. SVMs use patient-grouped3-fold OOF margins with fold-local feature fitting, followed by train-only logistic calibration. Hyperparameters are selected by validation mean reward at theta.50, cost tie-break. A second threshold is selected on validation reward; every choice/model hash is locked before test. Seed17 is primary;0/1/2 repeats are not an ensemble. These action probabilities are not calibrated probabilities of weak-answer correctness.',
       '', '## Primary seed17, fixed theta0.50, no risk gate','',
       '| Router | T1 accuracy / cost | T3 accuracy / cost |','|---|---:|---:|',
       '| Jonathan LLM |61.33% /54.11% |63.33% /81.47% |']
    for family in FAMILIES:
        rr=[next(r for r in result if r['setting']==s and r['family']==family and r['seed']==17 and r['point']=='OP_A_fixed_050' and not r['risk_gate']) for s in [1,3]]
        report.append('| '+family+' | '+' | '.join(f"{r['accuracy']*100:.2f}% / {r['cost_pct_always_escalate']:.2f}%" for r in rr)+' |')
    report+=['','## Primary seed17, validation-selected threshold and common risk gate','',
       '| Router | T1 accuracy / cost | T3 accuracy / cost |','|---|---:|---:|',
       '| Jonathan LLM, own fixed theta.50 + same gate |61.33% /80.99% |63.33% /81.47% |']
    for family in FAMILIES:
        rr=[next(r for r in result if r['setting']==s and r['family']==family and r['seed']==17 and r['point']=='val_reward_selected' and r['risk_gate']) for s in [1,3]]
        report.append('| '+family+' | '+' | '.join(f"{r['accuracy']*100:.2f}% / {r['cost_pct_always_escalate']:.2f}%" for r in rr)+' |')
    report+=['','## Validation-selected family (not best-test family)','',
       '| Setting | Point | Family chosen on validation | Test accuracy | Cost % |','|---|---|---|---:|---:|']
    for selected in locks['global_family_choices']:
        if selected['seed']!=17:continue
        r=next(r for r in result if all(r[k]==selected[k] for k in ['setting','family','seed','point']) and not r['risk_gate'])
        report.append(f"|T{r['setting']}|{r['point']}|{r['family']}|{r['accuracy']*100:.2f}%|{r['cost_pct_always_escalate']:.2f}%|")
    report+=['','## Limits','',
       'Reported costs are measured replay FLOP units normalized by sum(full weak cost+strong cost) on the same150 tasks, including sunk weak work. They exclude router compute and the frozen risk scorer, consistently with the LLM board; CPU feature+batch prediction time is separate. These are not HotpotQA Table5 prices or direct-strong-only costs.',
       '', 'Ungated savings may deliver high-risk weak failures. Risk-gate results use identical frozen tau>=.8 forced escalation. Dataset small and task-type homogeneous; test already inspected in prior research, so these are exploratory paired-replay findings, not a new sealed test. Do not select method or seed based on test. Classifier probabilities model reward-optimal action, not actual weak success. Complete per-seed/point/gate rows are in results.csv and means/ranges in seed_repeats.json.']
    (out/'REPORT.md').write_text('\n'.join(report)+'\n');replay.write(out/'status.json',{'phase':'complete','selected_models':56,'test_n':150})
    print('\n'.join(report),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--phase',choices=['prepare','train','lock','evaluate'],required=True);ap.add_argument('--setting',type=int,choices=[1,3]);ap.add_argument('--family',choices=FAMILIES);ap.add_argument('--seed',type=int,default=17);a=ap.parse_args()
    if a.phase=='prepare':prepare(a.source,a.out)
    elif a.phase=='train':train(a.source,a.out,a.setting,a.family,a.seed)
    elif a.phase=='lock':lock(a.source,a.out)
    else:evaluate(a.source,a.out)
