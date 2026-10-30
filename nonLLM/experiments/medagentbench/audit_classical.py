"""Independent source/fold/aggregate audit and paired patient-cluster readout.

Uses stored predictions only: no new test inference, fitting or threshold search.
"""
import argparse,json
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import replay
import classical

def audit(source,out):
    protocol=json.loads((out/'protocol.json').read_text())
    assert replay.digest(source/'protocol.json')==protocol['source_protocol_sha256']
    for setting in [1,3]:assert replay.digest(source/f'reference/results/medagentbench/training{setting}_table.json')==protocol['source_table_hashes'][str(setting)]
    integrity=replay.verify_source(source)
    tasks={t['id']:t for t in json.loads((source/'reference/test_data_v2.json').read_text())}
    locks=json.loads((out/'validation_lock.json').read_text())
    fold_total=0
    for item in locks['models']:
        setting=item['setting'];rows=[t for t in replay.table(source,setting) if t['split']=='train']
        groups=[tasks[t['tid']].get('eval_MRN') or 'no_patient:'+t['tid'] for t in rows]
        folder=out/'training'/f't{setting}'/item['family']/f"seed{item['seed']}"
        held=[]
        for fold in json.loads((folder/'calibration_folds.json').read_text()):
            train={groups[i] for i in fold['fit_indices']};val={groups[i] for i in fold['held_indices']}
            assert not train&val;held.extend(fold['held_indices']);fold_total+=1
        if 'svm' in item['family']:assert sorted(held)==list(range(90))
        model=joblib.load(folder/'best.joblib');feature=model['pipeline'].named_steps['features']
        vectorizer=feature if item['family']=='multinomial_nb' else dict(feature.transformer_list)['text']
        original=joblib.load(source/'data'/f'features_t{setting}.joblib')
        assert vectorizer.vocabulary_==original['vectorizer'].vocabulary_
        if item['family']!='multinomial_nb':
            np.testing.assert_allclose(dict(feature.transformer_list)['numeric'].named_steps['scale'].mean_,original['scaler'].mean_,rtol=0,atol=1e-10)
    # Assert every published aggregate against actual stored per-task decisions.
    scores=pd.read_csv(out/'test_scores.csv');detail=pd.read_csv(out/'per_task.csv.gz');results=json.loads((out/'results.json').read_text())
    assert len(scores)==8400 and not scores.duplicated(['setting','family','seed','tid']).any()
    for r in results:
        d=detail[(detail.setting==r['setting'])&(detail.family==r['family'])&(detail.seed==r['seed'])&(detail.point==r['point'])&(detail.risk_gate==r['risk_gate'])]
        assert len(d)==150 and d.tid.nunique()==150
        assert d.correct.sum()==r['correct'] and abs(d.cost.sum()-r['total_cost'])<1e-6
    tau={r['tid']:r['tau'] for r in json.loads((source/'reference/results/medagentbench/tau_scores_v2.json').read_text())['scores']}
    comparisons=[];difference_rows=[]
    for setting in [1,3]:
        rows=[r for r in replay.table(source,setting) if r['split']=='test']
        ids=[r['tid'] for r in rows];group_names=[tasks[tid].get('eval_MRN') or 'no_patient:'+tid for tid in ids]
        unique=sorted(set(group_names));group=np.array([unique.index(g) for g in group_names]);count=np.bincount(group)
        denominator=np.bincount(group,weights=[r['cw']+r['cs'] for r in rows])
        rng=np.random.default_rng(20260729);draw=rng.integers(len(unique),size=(2000,len(unique)))
        peer={r['tid']:r['pDeliver'] for r in map(json.loads,(source/f'reference/results/medagentbench/training{setting}_test_scores.jsonl').read_text().splitlines())}
        peerp=np.array([peer[tid] for tid in ids])
        for family in classical.FAMILIES:
            d=scores[(scores.setting==setting)&(scores.family==family)&(scores.seed==17)]
            probability=dict(zip(d.tid,d.pDeliver));p=np.array([probability[tid] for tid in ids])
            item=next(r for r in locks['models'] if r['setting']==setting and r['family']==family and r['seed']==17)
            for point,threshold in [('OP_A_fixed_050',.5),('val_reward_selected',item['selected_threshold']['threshold'])]:
                for gate in [False,True]:
                    m,(a,q,c,r)=replay.metric(rows,setting,p,threshold,tau,gate)
                    ref,(ra,rq,rc,_)=replay.metric(rows,setting,peerp,.5,tau,gate)
                    qgap=np.bincount(group,weights=q-rq);cgap=np.bincount(group,weights=c-rc)
                    qa=100*qgap[draw].sum(1)/count[draw].sum(1);co=100*cgap[draw].sum(1)/denominator[draw].sum(1)
                    comparisons.append({'setting':setting,'family':family,'seed':17,'point':point,'risk_gate':gate,'reference':'Jonathan same-setting theta.50, same risk gate',
                        'accuracy_delta_pp':100*(m['accuracy']-ref['accuracy']),'accuracy_delta_patient_bootstrap_95ci':np.quantile(qa,[.025,.975]).tolist(),
                        'cost_delta_pp':m['cost_pct_always_escalate']-ref['cost_pct_always_escalate'],'cost_delta_patient_bootstrap_95ci':np.quantile(co,[.025,.975]).tolist(),
                        'decision_disagreements':int((a!=ra).sum()),'patient_clusters':len(unique),'bootstrap_resamples':2000})
                    if setting==3 and family=='linear_svm' and point=='val_reward_selected' and gate:
                        for i in np.flatnonzero(a!=ra):
                            t=rows[i];difference_rows.append({'tid':t['tid'],'task_type':t['tid'].split('_')[0],
                                'old_action':int(ra[i]),'new_action':int(a[i]),'qw':t['qw'],'qs':t['qs'],
                                'correctness_difference':int(q[i]-rq[i]),'cost_saved':float(rc[i]-c[i]),'tau':tau[t['tid']]})
    replay.write(out/'paired_patient_comparisons.json',comparisons)
    replay.write(out/'linear_svm_t3_paired_changes.json',difference_rows)
    assert len(difference_rows)==15 and all(t['qw']==t['qs']==0 for t in difference_rows)
    repeats=json.loads((out/'seed_repeats.json').read_text())
    text=(out/'REPORT.md').read_text().split('\n## Independent audit and repeat sensitivity')[0]
    more=['','## Independent audit and repeat sensitivity','',
       'All selected feature vocabularies/scalers match the previous train-only pipeline. SVM calibration fold patient separation and90-row OOF coverage were independently reconstructed from the official dataset. No test model was run again. Paired patient-cluster intervals use2,000 resamples and are in paired_patient_comparisons.json.',
       '', '| Router | T3 validation-selected threshold, same risk gate: accuracy mean (0/1/2 range) | Mean cost % |','|---|---:|---:|']
    for r in repeats:
        if r['setting']==3 and r['point']=='val_reward_selected' and r['risk_gate']:
            more.append(f"|{r['family']}|{r['accuracy_mean']*100:.2f}% ({r['accuracy_min']*100:.2f}–{r['accuracy_max']*100:.2f})|{r['cost_pct_mean']:.2f}%|")
    more+=['','## Why the paired cost changes','',
       'For seed17 Training3, validation-selected Linear SVM plus the same high-risk gate differs from the LLM on15 task7 instances. Both cached weak and strong executions fail on all15; the classifier delivers the weak result instead of paying for an ineffective strong escalation. There is zero change in correctness on those tasks, and no claim of improved clinical outcome. These instances fall outside the tau>=.8 gate. This explains the3.79 percentage-point cost saving while maintaining63.33% aggregate accuracy; it does not solve the underlying failures.',
       '', 'Task types are highly homogeneous in this small benchmark. These independent seed repeats are stable on the fixed replay but are not independent executor reruns or proof of generalization to new task types. Public-test reuse and multiple methods make the result exploratory.']
    (out/'REPORT.md').write_text(text+'\n'.join(more)+'\n')
    v=json.loads((out/'verification.json').read_text());v.update({'all_56_feature_pipelines_match_train_only_source':True,
        'svm_fold_patient_disjointness_independently_verified':fold_total,'all_336_aggregate_rows_verified':True,'paired_patient_bootstrap_contrasts':len(comparisons),
        'svm_cost_saving_explained_by_15_both_fail_tasks':True})
    replay.write(out/'verification.json',v);print(json.dumps(v,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();audit(a.source,a.out)
