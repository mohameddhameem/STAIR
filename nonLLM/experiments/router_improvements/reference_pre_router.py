"""Train the direct pre-router on authoritative reference outcomes, not a new grader."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from architecture import pre_features,fit_stage,metrics
from cascade_four_gpu import write_json

ROOT=Path(__file__).resolve().parents[4]
PRICES=np.array([2.,5.7,5.7,9.4])

def options(pred):
    for j in range(4):yield f'constant_{j}',np.full(len(pred),j)
    for lam in [0.,.05,.15,.4,.8,1.6,3.2]:
        yield f'lambda_{lam}',np.argmax(np.clip(pred[:,:4],0,1)-lam*PRICES[None]/PRICES[3],axis=1)

def main(out):
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output directory required')
    out.mkdir(parents=True)
    path=ROOT/'Labels/slr_labels_configA/full_run_p100/router_data_lenient.jsonl'
    grid={r['idx']:r for r in map(json.loads,path.open())}
    ids={s:sorted(i for i,g in grid.items() if g['split']==s) for s in ['train','val','test']}
    assert sum(grid[i]['em_lenient'][3] for i in ids['val'])==586
    assert sum(grid[i]['em_lenient'][3] for i in ids['test'])==793
    def features(split):
        rows=[]
        for i in ids[split]:
            g=grid[i]
            # Use the original archived pool and ordering, not the regenerated pool.
            passages=[p.strip() for p in re.split(r'(?:^|\n)\[\d+\] ',g['pool_text'])[1:]]
            assert len(passages)==g['n_pool']==100
            rows.append(pre_features({'question':g['question'],'ctxs_candidate':passages}))
        return rows
    manifest={'metric':'Provided original em_lenient outcomes; no regrading',
        'baseline_val':586/1110,'baseline_test':793/1481,
        'grid_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'inputs':'Question and original archived candidate pool only; no regenerated answer/evidence/cache inputs',
        'costs':'Original flat model-size proxy, WW/WS/SW/SS = 2/5.7/5.7/9.4; excludes router CPU overhead',
        'selection':'3-fold question-grouped training CV selects the regressor; small validation cost-penalty grid selects operating point; locked before test predictions',
        'test':'Previously examined split; descriptive',
        'scope':'Direct pre-retrieval router only; no claim that missing raw-answer methods have been aligned'}
    write_json(out/'manifest.json',manifest);write_json(out/'status.json',{'phase':'training'})
    em=np.array([grid[i]['em_lenient'] for i in ids['train']],dtype=float)
    targets=np.column_stack([em,em[:,3]-em[:,1]])
    bundle,pred=fit_stage('pre',{'train':features('train'),'val':features('val')},targets,np.array(ids['train']),out)
    emv=np.array([grid[i]['em_lenient'] for i in ids['val']]);records=[]
    for key,cells in options(pred):records.append({'key':key,'val_accuracy':float(emv[np.arange(len(emv)),cells].mean()),'val_cost':float(PRICES[cells].mean())})
    locks={}
    for obj in ['max_accuracy','ss_budget','cheap95']:
        eligible=records if obj=='max_accuracy' else [r for r in records if r['val_cost']<=PRICES[3]+1e-8] if obj=='ss_budget' else [r for r in records if r['val_accuracy']>=.95*emv[:,3].mean()-1e-12]
        key=(lambda r:(r['val_cost'],-r['val_accuracy'])) if obj=='cheap95' else (lambda r:(-r['val_accuracy'],r['val_cost']))
        locks[obj]=min(eligible,key=key)
    write_json(out/'validation_lock.json',locks)
    p=bundle['model'].predict(bundle['features'].transform(features('test'))[bundle['matrix_index']])
    candidates=dict(options(p));em=np.array([grid[i]['em_lenient'] for i in ids['test']]);strict=np.array([grid[i]['em_strict'] for i in ids['test']])
    results={}
    for obj,lock in locks.items():
        cells=candidates[lock['key']];r=metrics(em,strict,cells,PRICES,PRICES[3])
        results[obj]={'lock':lock,'test':r};np.savez(out/(obj+'.npz'),ids=ids['test'],cells=cells,correct=em[np.arange(len(em)),cells])
    write_json(out/'results.json',{'metric':'original em_lenient','SS_val':586/1110,'SS_test':793/1481,'results':results})
    lines=['# Direct pre-router on the original reference grid','',
        'Authoritative supplied em_lenient outcomes; no new scoring rule or regenerated answers.',
        'Reference SS: validation 52.79% (586/1110), test 53.54% (793/1481).',
        'Costs use the original flat proxy, not measured tokens. Test is descriptive.','',
        '| Objective | Validation accuracy % | Test lenient % | Cost % SS | Delta test SS pp |',
        '|---|---:|---:|---:|---:|']
    for obj,r in results.items():
        t=r['test'];lines.append(f"| {obj} | {100*r['lock']['val_accuracy']:.2f} | {100*t['accuracy']:.2f} | {t['cost_pct_SS']:.2f} | {t['delta_SS_pp']:+.2f} |")
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n');write_json(out/'status.json',{'phase':'complete'});print('\n'.join(lines))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
