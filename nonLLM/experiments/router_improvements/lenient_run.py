"""Finish all non-LLM families with one review-validated lenient scorer.

Original outcomes remain authoritative in their separate reference cohort.
Regenerated Instruct answers are regraded consistently, never joined to
the original answers' outcome labels. No previous model weights are reused.
"""
import argparse
import csv
import gzip
import json
from pathlib import Path
import shutil
import subprocess
import sys

from corrected_run import ROOT,HERE,OLD,PRIOR,MORE,ENV,write,sha,load,rows,symlink,state,launch,join
from reference_lenient import score,lenient_match


def validate_reviews():
    base=ROOT/'Labels/slr_labels_configA'
    grid={g['idx']:g for g in load(base/'full_run_p100/router_data_lenient.jsonl')}
    with gzip.open(ROOT/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:source=json.load(f)
    seen={};truncated=0
    for path,b,w,s,limit in [('chooser_ceiling/chooser_review.csv','ret_branch','weak_qa_answer','strong_qa_answer',300),
            ('qa_labels/qa_label_review.csv','ret_choice','weak_answer','strong_answer',200)]:
        for r in csv.DictReader((base/path).open(encoding='utf-8-sig')):
            i=int(r['idx']);branch=r[b]
            for who,col in [(0,w),(1,s)]:
                if len(r[col])>=limit:truncated+=1;continue
                key=(i,branch,who)
                if key in seen:assert seen[key]==r[col]
                seen[key]=r[col]
                expected=grid[i]['em_lenient'][(0 if branch=='W' else 2)+who]
                assert lenient_match(r[col],source[i]['answers'])==expected,(key,r[col],expected)
    assert len(seen)==4267
    return {'review_records':len(seen),'mismatches':0,'excluded_truncated_records':truncated,
        'coverage':'Selected original train/validation review answers, not a complete independent scorer certification',
        'original_SS_val':586/1110,'original_SS_test':793/1481}


def prepare(out):
    import numpy as np
    base=out/'base';code=out/'code';code.mkdir(parents=True)
    validation=validate_reviews()
    grid=load(OLD/'labels/full_run_p100/router_data_lenient.jsonl')
    with gzip.open(ROOT/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:source=json.load(f)
    answers={who:{r['k']:r['v'] for r in load(OLD/f'labels/full_run_p100/qa_{who}.jsonl')} for who in ['weak','strong']}
    ret_labels={r['idx']:r for r in load(OLD/'labels/ret_labels/ret_labels.jsonl')}
    ret=[];qa=[];checked=0
    for g in grid:
        i=g['idx'];inst=source[i];assert g['_id']==inst['_id']
        judged=[score(answers[who][f'{i}|{branch}'],inst['answers']) for branch in ['weak','strong'] for who in ['weak','strong']]
        strict=[x[0] for x in judged];lenient=[x[1] for x in judged]
        assert strict==g['em_strict'];checked+=4
        g['em_strict']=strict;g['em_lenient']=lenient
        g['evaluation_metric']='review-validated reference-lenient reconstruction'
        pool=[p[:600] for p in inst['ctxs_candidate'][:100]]
        g['pool_text']='\n'.join(f'[{j+1}] {p}' for j,p in enumerate(pool))
        g['ret_label']='W' if any(lenient[:2]) else ('S' if any(lenient[2:]) else None)
        ret.append(ret_labels[i])
        for branch,j in [('W',0),('S',2)]:
            label='W' if lenient[j] else ('S' if lenient[j+1] else 'X')
            g['qa_label_'+branch]=None if label=='X' else label
            qa.append({'idx':i,'split':g['split'],'ret_choice':branch,'qa_label':label,'qa_target':'W' if label=='X' else label})
    rows(base/'labels/full_run_p100/router_data_lenient.jsonl',grid)
    rows(base/'labels/ret_labels/ret_labels.jsonl',ret)
    rows(base/'labels/qa_labels/qa_labels.jsonl',qa)
    for kind in ['retr_weak','retr_strong','qa_weak','qa_strong']:
        for prefix in ['', 'meta_']:
            name=prefix+kind+'.jsonl';symlink(OLD/'labels/full_run_p100'/name,base/'labels/full_run_p100'/name)
    shutil.copy2(OLD/'token_costs.json',base/'token_costs.json')
    symlink(PRIOR/'samples',out/'initial/samples');symlink(code,out/'initial/code')
    symlink(MORE/'more_samples',out/'extended/more_samples')
    protocol={'primary_metric':'Review-validated reference-lenient reconstruction; forward whole-word containment',
        'scorer_sha256':sha(HERE/'reference_lenient.py'),'review_validation':validation,
        'rules':['Lowercase; remove Unicode punctuation; remove English articles; collapse whitespace',
            'Exact normalized equality or gold whole-word sequence contained in prediction',
            'For gold yes/no, reject a prediction containing both yes and no',
            'Do not accept prediction-contained-in-gold reverse matching; no blanket quantifier-prefix rejection'],
        'limitation':'Original mint_labels.py is unavailable. Agreement on all 4267 reviewed answers supports this reconstruction but does not certify unobserved original cases.',
        'cohorts':'Original em_lenient reference grid remains separate; this run uses regenerated Instruct answers with freshly matching lenient labels',
        'generation':'Qwen2.5-1.5B/7B Instruct FP16, existing greedy and random weak samples; Qwen weights unchanged; no Base generations',
        'generation_manifest':json.load(open(OLD/'manifest.json')),
        'strict_diagnostic':'Official normalized answer EM retained only as a secondary diagnostic, never used for targets, rewards, model selection or thresholds',
        'labels':'Original-style and gain/policy targets use lenient outcomes; Config A retrieval coverage labels unchanged; all QA and sampled rewards rebuilt with the same scorer',
        'features':'Target-free extended-policy feature arrays reused from identical Instruct observations; no trained router weights or old reward/target arrays reused',
        'cost':'One six-price mean parameter-weighted token ruler; direct routes skip unused calls; cascades and consensus pay for all calls; CPU overhead excluded',
        'test':'Previously examined; exploratory fixed-split results, not confirmatory evidence',
        'greedy_QA_regraded':checked,
        'SS':{sp:{'n':len(rs:=[g for g in grid if g['split']==sp]),'correct':sum(g['em_lenient'][3] for g in rs),'accuracy':sum(g['em_lenient'][3] for g in rs)/len(rs),
                  'strict_accuracy':sum(g['em_strict'][3] for g in rs)/len(rs)} for sp in ['train','val','test']}}
    write(base/'manifest.json',protocol);write(out/'protocol.json',protocol)
    c=json.load(open(base/'token_costs.json'))['gate_eval_constants']
    rw,rs,ww,ws,sw,ss=[c['COST_'+k] for k in ['RET_W','RET_S','QA_W_GIVEN_W','QA_W_GIVEN_S','QA_S_GIVEN_W','QA_S_GIVEN_S']]
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name not in ['corrected_run.py','lenient_run.py']}
    files['baselines.py']=(ROOT/'STAIR/nonLLM/baselines.py').read_text().replace('COSTS = np.array([2.0, 5.7, 5.7, 9.4])',f'COSTS = np.array({[rw+ww,rw+sw,rs+ws,rs+ss]!r})').replace('stored pipeline FLOP units','mean parameter-weighted token units')
    files['cascade.py']=(ROOT/'STAIR/nonLLM/cascade.py').read_text()
    helper=(ROOT/'STAIR/nonLLM/deployment/cascade_four_gpu.py').read_text()
    start=helper.index('def score(pred, gold):');end=helper.index('\n\ndef finalize',start)
    helper=helper[:start]+'''def score(pred, gold):
    """Secondary strict EM and primary review-validated reference lenient."""
    from reference_lenient import score as reference_score
    return reference_score(pred,gold)
'''+helper[end:]
    files['cascade_four_gpu.py']=helper
    for name,text in files.items():
        text=text.replace("str(ROOT / 'STAIR/nonLLM')","str(Path(__file__).resolve().parent)")
        text=text.replace("str(ROOT / 'STAIR/nonLLM/deployment')","str(Path(__file__).resolve().parent)")
        text=text.replace("ROOT/'runs/nonllm_cascade_four_gpu_full_v1'",f"Path({str(base)!r})")
        text=text.replace("ROOT/'runs/nonllm_router_improvements_v2'",f"Path({str(out/'initial')!r})")
        (code/name).write_text('# Isolated review-validated lenient rerun; see ../protocol.json.\n'+text)
    samples={}
    for folder in [PRIOR/'samples',MORE/'more_samples']:
        for path in folder.glob('trainval_[0-3].jsonl'):
            for r in load(path):
                key=(r['idx'],r['branch'],r['sample']);assert key not in samples;samples[key]=r
    train_ids=sorted(g['idx'] for g in grid if g['split']=='train');byid={g['idx']:g for g in grid}
    for mode,k in [('base',1),('confidence',1),('aug5',5),('aug9',9)]:
        dest=out/'extended/data'/mode;dest.mkdir(parents=True);old=MORE/'data'/mode
        for path in old.iterdir():
            if path.name.startswith(('ret_train','ret_val','ret_test','qa_train','qa_val','qa_test')) or path.name.endswith('_features.joblib') or path.name=='ret_index.npy':symlink(path,dest/path.name)
        reward=[];targets=[]
        for i in train_ids:
            for sample in ([-1] if k==1 else [-1,*range(k-1)]):
                r=list(byid[i]['em_lenient'])
                for branch,j in [('W',0),('S',2)]:
                    if sample!=-1:r[j]=lenient_match(samples[i,branch,sample]['text'],source[i]['answers'])
                    targets.append(int(not r[j] and bool(r[j+1])))
                reward.append(r)
        assert np.asarray(reward).shape==np.load(old/'reward.npy').shape
        np.testing.assert_array_equal(np.load(old/'ret_index.npy'),np.repeat(np.arange(len(train_ids)),k))
        np.save(dest/'reward.npy',np.array(reward,dtype=np.float32));np.save(dest/'qa_target.npy',np.array(targets,dtype=np.float32))
        np.save(dest/'ret_target.npy',np.array([int(not any(byid[i]['em_lenient'][:2]) and any(byid[i]['em_lenient'][2:])) for i in train_ids],dtype=np.float32))
        for sp in ['val','test']:
            ids=sorted(g['idx'] for g in grid if g['split']==sp)
            np.save(dest/f'em_{sp}.npy',np.array([byid[i]['em_lenient'] for i in ids]))
        write(dest/'ready.json',{'metric':protocol['primary_metric'],'variants':k,'all_targets_and_rewards_rebuilt':True,'train_questions':len(train_ids)})
    write(out/'source_hashes.json',{p.name:sha(p) for p in code.glob('*.py')})
    print('Prepared review-validated lenient labels and rewards; SS baselines:',protocol['SS'],flush=True)


def report(out):
    protocol=json.load(open(out/'protocol.json'));result=[]
    def add(name,t):
        result.append({'method':name,'accuracy':t.get('lenient_accuracy',t.get('accuracy')),
            'strict_accuracy':t['strict_accuracy'],'cost_pct_SS':t.get('cost_pct_of_direct_SS',t.get('cost_pct_SS',t.get('cost_pct_of_SS')))})
    for name,t in json.load(open(out/'original/results.json'))['composed'].items():add('Original / '+name,t)
    for folder,name in [('base/router','October full cascade'),('pilot','October no-answer ablation')]:add(name,json.load(open(out/folder/'results.json'))['test'])
    initial=json.load(open(out/'initial/results/summary.json'))
    for r in initial['results']:add('Initial / '+r['name'],r['test'])
    extended=json.load(open(out/'extended/results.json'))
    for r in extended['results']:add('Extended / '+r['name'],r['test'])
    for folder,label in [('architecture','Architecture'),('combined','Combined')]:
        for name,r in json.load(open(out/folder/'results.json'))['results'].items():add(label+' / '+name,r['test'])
    assert len(result)==73
    write(out/'results.json',{'primary_metric':protocol['primary_metric'],'SS':protocol['SS'],'results':result,
        'original_reference_SS':protocol['review_validation'],'original_scorer_source_recovered':False})
    lines=['# Non-LLM routers: review-validated lenient rerun','',
        '**Primary metric: review-validated reference-lenient reconstruction, not exact match.**',
        'All 73 method/operating-point records were retrained and selected using lenient targets/rewards/validation accuracy. Strict EM is diagnostic only.',
        'The scorer agrees with all 4267 available original answer-review judgments. The original scoring source is unavailable; full equivalence on unobserved cases is not certified.',
        'Regenerated Qwen2.5 Instruct outputs and matching labels form one cohort. Original reference outcomes remain separate: SS validation 52.79%, test 53.54%.',
        'Token-price costs include sunk weak calls and consensus answers; CPU router overhead is excluded. Previously examined test split is exploratory.','',
        f"Regenerated cohort SS: validation **{100*protocol['SS']['val']['accuracy']:.2f}%**, test **{100*protocol['SS']['test']['accuracy']:.2f}%**.",'',
        '| Method | Test lenient % | Strict EM diagnostic % | Cost % SS |','|---|---:|---:|---:|']
    lines += [f"| {r['method']} | {100*r['accuracy']:.2f} | {100*r['strict_accuracy']:.2f} | {r['cost_pct_SS']:.2f} |" for r in result]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    p=out/'initial/results/REPORT.md';p.write_text(p.read_text().replace('Gain over cost-matched fixed mixture','Gain over cost-matched cascade-policy mixture'))


def main(out):
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output directory required')
    out.mkdir(parents=True);jobs=[]
    try:
        state(out,'validating_reference_and_rebuilding_lenient_labels');prepare(out)
        py=sys.executable;code=out/'code';base=out/'base'
        cascade=[py,'-u',str(code/'cascade.py'),'--labels-root',str(base/'labels'),'--token-costs',str(base/'token_costs.json')]
        jobs=[launch(out,'original',[py,'-u',str(code/'baselines.py'),'--data',str(base/'labels/full_run_p100/router_data_lenient.jsonl'),
            '--splits',str(ROOT/'extracted/Dataset/splits.json'),'--out',str(out/'original')]),
            launch(out,'cascade',cascade+['--qa-weak',str(base/'labels/full_run_p100/qa_weak.jsonl'),'--out',str(base/'router')]),
            launch(out,'pilot',cascade+['--without-weak-answer','--out',str(out/'pilot')]),
            launch(out,'architecture',[py,'-u',str(code/'architecture.py'),'--out',str(out/'architecture')])]
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if len(free.splitlines())!=4 or min(map(int,free.splitlines()))<2500:raise RuntimeError('Four GPUs with >=2500 MiB free required')
        jobs += [launch(out,f'extended_gpu_{i}',[py,'-u',str(code/'extended_policy.py'),'--out',str(out/'extended'),'--phase','train','--shard',str(i)],{'CUDA_VISIBLE_DEVICES':str(i)}) for i in range(4)]
        join(out,jobs,'training_classifiers_architecture_and_144_extended_streams');jobs=[]
        join(out,[launch(out,'extended_evaluate',[py,'-u',str(code/'extended_policy.py'),'--out',str(out/'extended'),'--phase','evaluate']),
            launch(out,'combined',[py,'-u',str(code/'architecture_combine.py'),'--parent',str(out/'architecture'),'--out',str(out/'combined')]),
            launch(out,'suite_cached',[py,'-u',str(code/'suite.py'),'--base',str(base),'--out',str(out/'initial/results'),'--phase','cached'])],
            'validation_selection_and_initial_suite')
        for phase in ['sampled','finish']:
            join(out,[launch(out,'suite_'+phase,[py,'-u',str(code/'suite.py'),'--base',str(base),'--out',str(out/'initial/results'),
                '--samples',str(out/'initial/samples'),'--phase',phase])],'initial_suite_'+phase)
        report(out);state(out,'complete',report=str(out/'REPORT.md'))
    except BaseException as error:
        for n,p,f in jobs:
            if p.poll() is None:p.terminate()
            f.close()
        state(out,'failed',error=repr(error));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args().out.resolve())
