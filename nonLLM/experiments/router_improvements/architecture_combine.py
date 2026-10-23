"""Combine pre-routing and paid post-answer choice; lock on validation first.

Follow-up to architecture.py. Uses existing train-CV-selected models without
refitting or selecting families from observed test results. Exploratory replay.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import sys

import joblib
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from architecture import ROOT,Data,pre_features,chooser_features,metrics,write_json
from cascade import load

LAMBDAS=[0.,.05,.15,.4,.8,1.6,3.2]
THRESHOLDS=[-.2,-.1,0.,.1,.2]

def combined_cells(initial,chooser_predictions,threshold):
    cells=initial.copy();mask=initial==3
    cells[mask]=np.where(chooser_predictions>threshold,2,3)
    return cells

def main(a):
    if a.out.exists() and any(a.out.iterdir()):raise ValueError('Fresh output required')
    a.out.mkdir(parents=True)
    base=ROOT/'runs/nonllm_cascade_four_gpu_full_v1'
    data=Data(base,ROOT/'runs/nonllm_router_improvements_v2/samples')
    source=json.load(gzip.open(ROOT/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt'))
    strong={r['k']:r['v'] for r in load(base/'labels/full_run_p100/qa_strong.jsonl')}
    pre=joblib.load(a.parent/'pre.joblib');chooser=joblib.load(a.parent/'chooser.joblib')
    manifest=json.load(open(a.parent/'manifest.json'))
    direct=np.array(manifest['costs']['direct']);extra=manifest['costs']['post'][0]-direct[3]
    write_json(a.out/'manifest.json',{'parent':str(a.parent),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'lambdas':LAMBDAS,'thresholds':THRESHOLDS,'selection':'Validation only; inherited train-only grouped-CV models; fixed small grid',
        'cost':'Direct route calls plus weak QA on every initially selected SS, even if weak answer is finally retained',
        'test':'Exploratory follow-up on previously examined test split, unadjusted intervals'})
    def options(split):
        ids=data.ids[split]
        for i in ids:
            assert source[i]['_id']==data.grid[i]['_id']
            assert source[i]['question']==data.grid[i]['question']
        rows=[pre_features(source[i]) for i in ids]
        q=pre['model'].predict(pre['features'].transform(rows)[pre['matrix_index']])[:,:4]
        result={}
        for lam in LAMBDAS:
            initial=np.argmax(np.clip(q,0,1)-lam*direct[None,:]/direct[3],axis=1)
            # Only initially-SS questions execute both QA calls and use their features.
            chosen_ids=[i for i,j in zip(ids,initial) if j==3]
            rows=[chooser_features(data.grid[i],data.answer(i,'S'),strong[f'{i}|strong'],
                data.samples[i,'S',-1]['confidence']) for i in chosen_ids]
            gains=chooser['model'].predict(chooser['features'].transform(rows)[chooser['matrix_index']]) if rows else np.array([])
            paid=direct[initial]+(initial==3)*extra
            for threshold in THRESHOLDS:
                key=f'lambda_{lam}_threshold_{threshold}'
                result[key]=(combined_cells(initial,gains,threshold),paid)
            # An uncombined route may be better; include it in every objective.
            result[f'lambda_{lam}_no_chooser']=(initial,direct[initial])
        result['direct_SS']=(np.full(len(ids),3),np.full(len(ids),direct[3]))
        return result
    val=options('val');em=np.array([data.grid[i]['em_lenient'] for i in data.ids['val']]);records=[]
    for key,(cells,paid) in val.items():records.append({'key':key,'val_accuracy':float(em[np.arange(len(em)),cells].mean()),'val_cost':float(paid.mean())})
    locks={}
    for obj in ['max_accuracy','ss_budget','cheap95']:
        eligible=records if obj=='max_accuracy' else [r for r in records if r['val_cost']<=direct[3]+1e-8] if obj=='ss_budget' else [r for r in records if r['val_accuracy']>=.95*em[:,3].mean()-1e-12]
        key=(lambda r:(r['val_cost'],-r['val_accuracy'])) if obj=='cheap95' else (lambda r:(-r['val_accuracy'],r['val_cost']))
        locks[obj]=min(eligible,key=key)
    write_json(a.out/'validation_lock.json',locks)
    test=options('test');ids=data.ids['test'];em=np.array([data.grid[i]['em_lenient'] for i in ids]);strict=np.array([data.grid[i]['em_strict'] for i in ids]);results={}
    for obj,lock in locks.items():
        cells,paid=test[lock['key']]
        # Metrics accepts cell-indexed costs; here cost depends on the initial route.
        r=metrics(em,strict,cells,np.zeros(4),direct[3])
        r['mean_cost']=float(paid.mean());r['cost_pct_SS']=float(100*paid.mean()/direct[3])
        results[obj]={'lock':lock,'test':r}
        np.savez(a.out/(obj+'.npz'),ids=ids,cells=cells,correct=em[np.arange(len(ids)),cells],prices=paid)
    write_json(a.out/'results.json',{'SS_accuracy':float(em[:,3].mean()),'results':results})
    report=['# Combined pre-routing and answer selection','',
        'Inherited train-CV models. Operating points locked on validation before test replay. Exploratory follow-up on a previously examined test set.',
        'Strong and weak QA costs are both charged on every initially selected SS route, irrespective of the retained answer.','',
        '| Objective | Validation configuration | Val accuracy % | Val cost % SS | Test accuracy % | Strict % | Test cost % SS | Delta SS pp | Paired interval pp |',
        '|---|---|---:|---:|---:|---:|---:|---:|---|']
    for obj,r in results.items():
        l=r['lock'];t=r['test'];report.append(f"| {obj} | {l['key']} | {100*l['val_accuracy']:.2f} | {100*l['val_cost']/direct[3]:.2f} | {100*t['accuracy']:.2f} | {100*t['strict_accuracy']:.2f} | {t['cost_pct_SS']:.2f} | {t['delta_SS_pp']:+.2f} | {t['paired_interval_pp']} |")
    (a.out/'REPORT.md').write_text('\n'.join(report)+'\n')
    write_json(a.out/'status.json',{'phase':'complete'})
    print('\n'.join(report))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--out',type=Path,required=True);main(p.parse_args())
