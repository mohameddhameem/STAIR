"""Add validation-budget operating points without retraining or changing v1 results."""
import argparse,itertools,json
from pathlib import Path
import numpy as np
import pandas as pd
import aligned_model_slide as models
import jonathan_aligned as aligned

def evaluate(source,run,out):
    out.mkdir(parents=True,exist_ok=True)
    p=json.loads((source/'protocol.json').read_text())
    val=np.load(source/'data/grid_val.npz')['correct'].reshape(-1,4)
    locks=[]
    for arch in ['linear',*models.ARCHES]:
        for name,method in models.METHODS.items():
            pred=models.predict(source,run,arch,method,'val');best={}
            for threshold in itertools.product(np.linspace(0,1,21),repeat=3):
                actions=aligned.route(pred,threshold)
                n=int(val[np.arange(len(val)),actions].sum());cost=float(aligned.PRICE[actions].mean());key=(n,-cost)
                for objective in ['max_accuracy','ss_budget']:
                    if objective=='ss_budget' and cost>aligned.SS:continue
                    if objective not in best or key>best[objective][0]:best[objective]=(key,threshold)
            for objective,(key,threshold) in best.items():
                locks.append({'architecture':arch,'training':name,'method':method,'objective':objective,
                              'threshold':list(threshold),'validation_correct':key[0],'validation_cost':-key[1]})
    assert len(locks)==18
    aligned.write(out/'validation_lock.json',{'selections':locks,'rule':'max validation accuracy, cost tie-break; ss_budget additionally requires mean validation Table5 cost<=929',
                    'disclosure':'Previously inspected test; new budget thresholds selected solely on validation; no training changes.'})
    test=np.load(source/'data/grid_test.npz')['correct'].reshape(-1,4);rows=[];detail=[];predictions={}
    for lock in locks:
        key=(lock['architecture'],lock['method'])
        if key not in predictions:predictions[key]=models.predict(source,run,*key,'test')
        actions=aligned.route(predictions[key],lock['threshold']);v=test[np.arange(len(test)),actions]
        counts=np.bincount(actions,minlength=4).tolist()
        rows.append({**lock,'test_n':len(test),'test_correct':int(v.sum()),'accuracy_pct':100*int(v.sum())/len(test),
                     'cost':float(aligned.PRICE[actions].mean()),'cost_pct_ss':100*float(aligned.PRICE[actions].mean())/aligned.SS,
                     'route_counts':dict(zip(['WW','WS','SW','SS'],counts))})
        detail.extend({'architecture':key[0],'method':key[1],'objective':lock['objective'],'id':qid,'route':int(a),'lenient':int(c),'cost':float(aligned.PRICE[a])} for qid,a,c in zip(p['ids']['test'],actions,v))
    # Independent row-level check: call costs, original standard scorer grid,
    # old accuracy-first results, and previously audited linear budget results.
    ix={qid:i for i,qid in enumerate(p['ids']['test'])}
    for r in detail:
        assert r['lenient']==test[ix[r['id']],r['route']]
        retrieval,qa=divmod(r['route'],2)
        assert r['cost']==77+retrieval*814+3+qa*115
    old=json.loads((run/'results.json').read_text());oldlinear=json.loads((source/'results.json').read_text())
    for r in rows:
        if r['objective']=='max_accuracy':
            prior=next(v for v in old if v['architecture']==r['architecture'] and v['method']==r['method'])
            assert r['test_correct']==prior['test_correct'] and abs(r['cost']-prior['cost'])<1e-9
        elif r['architecture']=='linear':
            method={'sft':'sft','exact':'sft_reward_both','grpo':'sft_grpo_stage'}[r['method']
            ]
            prior=next(v for v in oldlinear if v['method']==method and v['objective']=='ss_budget')
            assert r['test_correct']==prior['correct'] and abs(r['cost']-prior['cost'])<1e-9
    aligned.write(out/'results.json',rows);pd.DataFrame(rows).to_csv(out/'results.csv',index=False)
    pd.DataFrame(detail).to_csv(out/'per_question.csv.gz',index=False)
    aligned.write(out/'verification.json',{'operating_points':18,'audited_predictions':len(detail),'accuracy_first_matches_v1':True,'linear_budget_matches_prior_audit':True,'weak_call_costs_verified':True})
    aligned.write(out/'status.json',{'phase':'complete'})
    print(pd.DataFrame(rows)[['architecture','training','objective','accuracy_pct','cost_pct_ss']].to_string(index=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();evaluate(a.source,a.run,a.out)
