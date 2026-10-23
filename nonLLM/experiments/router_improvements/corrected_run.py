"""Replay every completed non-LLM family with official normalized answer EM.

Creates isolated, inspectable source snapshots. Existing Instruct generations
are read-only inputs. Historical lenient experiments are never overwritten.
"""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import string
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
OLD=ROOT/'runs/nonllm_cascade_four_gpu_full_v1'
PRIOR=ROOT/'runs/nonllm_router_improvements_v2'
MORE=ROOT/'runs/nonllm_grpo_extended_v1'
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')

def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def load(path):
    with path.open() as f:return [json.loads(l) for l in f]
def rows(path,items):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w') as f:
        for r in items:f.write(json.dumps(r,ensure_ascii=False)+'\n')
def symlink(src,dest):
    dest.parent.mkdir(parents=True,exist_ok=True);dest.symlink_to(src.resolve())
def state(out,phase,**kw):write(out/'status.json',{'phase':phase,**kw})

def official_metric():
    path=ROOT/'tools/hotpotqa_reference/hotpot_evaluate_v1.py'
    tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_answer','exact_match_score']]
    assert len(nodes)==2
    scope={'re':re,'string':string};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),scope)
    return scope['exact_match_score']

def prepare(out):
    import numpy as np
    base=out/'base';code=out/'code'
    code.mkdir(parents=True)
    original=load(OLD/'labels/full_run_p100/router_data_lenient.jsonl')
    with gzip.open(ROOT/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:source=json.load(f)
    answers={who:{r['k']:r['v'] for r in load(OLD/f'labels/full_run_p100/qa_{who}.jsonl')} for who in ['weak','strong']}
    em=official_metric()
    def score(text,gold):
        if isinstance(gold,str):gold=[gold]
        return int(any(em(text,g) for g in gold))
    ret_old={r['idx']:r for r in load(OLD/'labels/ret_labels/ret_labels.jsonl')}
    retrows=[];qarows=[];checked=0
    for g in original:
        i=g['idx'];inst=source[i];assert g['_id']==inst['_id']
        correct=[score(answers[who][f'{i}|{branch}'],inst['answers']) for branch in ['weak','strong'] for who in ['weak','strong']]
        assert correct==g['em_strict'];checked+=4
        # Legacy loaders retain em_lenient as an explicitly documented alias of EM.
        g['em_lenient']=list(correct);g['em_strict']=list(correct)
        g['evaluation_metric']='official HotpotQA normalized answer EM'
        pool=[p[:600] for p in inst['ctxs_candidate'][:100]]
        g['pool_text']='\n'.join(f'[{j+1}] {p}' for j,p in enumerate(pool))
        g['ret_label']='W' if any(correct[:2]) else ('S' if any(correct[2:]) else None)
        # Config-A's original target is evidence coverage and does not depend on QA grading.
        retrows.append(ret_old[i])
        for branch,j in [('W',0),('S',2)]:
            label='W' if correct[j] else ('S' if correct[j+1] else 'X')
            g['qa_label_'+branch]=None if label=='X' else label
            qarows.append({'idx':i,'split':g['split'],'ret_choice':branch,'qa_label':label,'qa_target':'W' if label=='X' else label})
    rows(base/'labels/full_run_p100/router_data_lenient.jsonl',original)
    rows(base/'labels/ret_labels/ret_labels.jsonl',retrows)
    rows(base/'labels/qa_labels/qa_labels.jsonl',qarows)
    for kind in ['retr_weak','retr_strong','qa_weak','qa_strong']:
        for prefix in ['', 'meta_']:
            name=prefix+kind+'.jsonl';symlink(OLD/'labels/full_run_p100'/name,base/'labels/full_run_p100'/name)
    shutil.copy2(OLD/'token_costs.json',base/'token_costs.json')
    symlink(PRIOR/'samples',out/'initial/samples')
    symlink(code,out/'initial/code')
    symlink(MORE/'more_samples',out/'extended/more_samples')
    metric_info={'primary_metric':'official HotpotQA normalized answer EM, max over supplied aliases',
        'official_metric_sha256':sha(ROOT/'tools/hotpotqa_reference/hotpot_evaluate_v1.py'),
        'verified_cached_QA_answers':checked,'strict_mismatches':0,
        'compatibility':'em_lenient and lenient_accuracy names in legacy code are aliases of strict EM in this isolated run; no containment matching is used for any reward or label',
        'reference':'Original label-grid validation lenient SS is 586/1110=52.79%; it is a different metric and is not the EM baseline',
        'AgentTTS_limit':'AgentTTS delegates to missing evaluation_2wiki.py; exact equivalence to that implementation is not certified',
        'labels':'Original-style: correctness outcomes, unresolved skipped; Config A retrieval: unchanged coverage labels; all QA, gain and policy rewards: official EM',
        'generation':'Reuse verified FP16 Qwen2.5-1.5B/7B Instruct generations and weak answer samples; no Base generations',
        'cost':'Same six mean parameter-weighted token prices for every family; direct routes skip unused calls; cascades include sunk weak calls; CPU overhead excluded',
        'test':'Previously examined split; exploratory, not an independent confirmatory holdout',
        'generation_manifest':json.load(open(OLD/'manifest.json')),
        'SS':{s:{'n':len(z:=[g for g in original if g['split']==s]),'correct':sum(g['em_strict'][3] for g in z),'accuracy':sum(g['em_strict'][3] for g in z)/len(z)} for s in ['train','val','test']}}
    write(base/'manifest.json',metric_info);write(out/'protocol.json',metric_info)
    costs=json.load(open(base/'token_costs.json'))['gate_eval_constants']
    rw,rs,ww,ws,sw,ss=[costs['COST_'+k] for k in ['RET_W','RET_S','QA_W_GIVEN_W','QA_W_GIVEN_S','QA_S_GIVEN_W','QA_S_GIVEN_S']]
    files={p.name:p.read_text() for p in HERE.glob('*.py') if p.name!='corrected_run.py'}
    files['baselines.py']=(ROOT/'STAIR/nonLLM/baselines.py').read_text().replace('COSTS = np.array([2.0, 5.7, 5.7, 9.4])',f'COSTS = np.array({[rw+ww,rw+sw,rs+ws,rs+ss]!r})').replace('stored pipeline FLOP units','mean parameter-weighted token units')
    files['cascade.py']=(ROOT/'STAIR/nonLLM/cascade.py').read_text()
    helper=(ROOT/'STAIR/nonLLM/deployment/cascade_four_gpu.py').read_text()
    start=helper.index('def score(pred, gold):');end=helper.index('\n\ndef finalize',start)
    helper=helper[:start]+'''def score(pred, gold):
    """Official normalized EM; both legacy tuple slots intentionally equal EM."""
    if isinstance(gold,str):gold=[gold]
    value=int(any(normalize(pred)==normalize(g) for g in gold))
    return value,value
'''+helper[end:]
    files['cascade_four_gpu.py']=helper
    for name,text in files.items():
        text=text.replace("str(ROOT / 'STAIR/nonLLM')","str(Path(__file__).resolve().parent)")
        text=text.replace("str(ROOT / 'STAIR/nonLLM/deployment')","str(Path(__file__).resolve().parent)")
        text=text.replace("ROOT/'runs/nonllm_cascade_four_gpu_full_v1'",f"Path({str(base)!r})")
        text=text.replace("ROOT/'runs/nonllm_router_improvements_v2'",f"Path({str(out/'initial')!r})")
        # The extended preparation is rebuilt below from verified target-free feature arrays.
        (code/name).write_text('# Isolated official-EM rerun; see ../protocol.json.\n'+text)
    # Verify sampled rewards too; no reuse of previous lenient training targets.
    sample_records={}
    for folder,pattern in [(PRIOR/'samples','trainval_[0-3].jsonl'),(MORE/'more_samples','trainval_[0-3].jsonl')]:
        for path in folder.glob(pattern):
            for r in load(path):
                key=(r['idx'],r['branch'],r['sample']);assert key not in sample_records;sample_records[key]=r
    train_ids=sorted(g['idx'] for g in original if g['split']=='train');grid={g['idx']:g for g in original}
    for mode,k in [('base',1),('confidence',1),('aug5',5),('aug9',9)]:
        dest=out/'extended/data'/mode;dest.mkdir(parents=True)
        old=MORE/'data'/mode
        for path in old.iterdir():
            if path.name.startswith(('ret_train','ret_val','ret_test','qa_train','qa_val','qa_test')) or path.name.endswith('_features.joblib') or path.name=='ret_index.npy':
                symlink(path,dest/path.name)
        rewards=[];targets=[]
        for i in train_ids:
            for sample in ([-1] if k==1 else [-1,*range(k-1)]):
                reward=list(grid[i]['em_strict'])
                for branch,j in [('W',0),('S',2)]:
                    if sample!=-1:reward[j]=score(sample_records[i,branch,sample]['text'],source[i]['answers'])
                    targets.append(int(not reward[j] and bool(reward[j+1])))
                rewards.append(reward)
        assert np.asarray(rewards).shape==np.load(old/'reward.npy').shape
        assert np.asarray(targets).shape==np.load(old/'qa_target.npy').shape
        np.save(dest/'reward.npy',np.array(rewards,dtype=np.float32));np.save(dest/'qa_target.npy',np.array(targets,dtype=np.float32))
        np.save(dest/'ret_target.npy',np.array([int(not any(grid[i]['em_strict'][:2]) and any(grid[i]['em_strict'][2:])) for i in train_ids],dtype=np.float32))
        for split in ['val','test']:
            ids=sorted(g['idx'] for g in original if g['split']==split)
            np.save(dest/f'em_{split}.npy',np.array([grid[i]['em_strict'] for i in ids]))
        write(dest/'ready.json',{'metric':'official EM','variants':k,'feature_reuse':'Unsupervised features from the same immutable Instruct observations; all labels/rewards rebuilt','train_questions':len(train_ids)})
    write(out/'source_hashes.json',{p.name:sha(p) for p in code.glob('*.py')})
    print('prepared official EM; SS baselines',metric_info['SS'],flush=True)

def launch(out,name,cmd,extra_env=None):
    f=(out/(name+'.log')).open('a')
    p=subprocess.Popen(cmd,env=dict(ENV,**(extra_env or {})),stdout=f,stderr=subprocess.STDOUT)
    return name,p,f

def join(out,jobs,phase):
    state(out,phase,workers={n:p.pid for n,p,f in jobs})
    try:
        while any(p.poll() is None for n,p,f in jobs):
            for n,p,f in jobs:
                if p.poll() not in (None,0):raise RuntimeError(f'{n} failed; inspect {out/(n+".log")}')
            time.sleep(3)
        for n,p,f in jobs:
            if p.returncode:raise RuntimeError(f'{n} failed')
    finally:
        for n,p,f in jobs:
            if p.poll() is None:p.terminate()
        for n,p,f in jobs:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
            f.close()

def report(out):
    protocol=json.load(open(out/'protocol.json'));rows_out=[]
    base=json.load(open(out/'original/results.json'))
    for name,r in base['composed'].items():rows_out.append({'method':'Original / '+name,'accuracy':r['strict_accuracy'],'cost_pct_SS':r['cost_pct_of_SS']})
    for folder,label in [('base/router','October full cascade'),('pilot','October no-answer ablation')]:
        r=json.load(open(out/folder/'results.json'))['test'];rows_out.append({'method':label,'accuracy':r['strict_accuracy'],'cost_pct_SS':r['cost_pct_of_direct_SS']})
    r=json.load(open(out/'initial/results/summary.json'))
    for x in r['results']:
        if 'test' in x:rows_out.append({'method':'Initial / '+x['name'],'accuracy':x['test']['strict_accuracy'],'cost_pct_SS':x['test']['cost_pct_of_direct_SS']})
    r=json.load(open(out/'extended/results.json'))
    for x in r['results']:rows_out.append({'method':'Extended / '+x['name'],'accuracy':x['test']['strict_accuracy'],'cost_pct_SS':x['test']['cost_pct_of_direct_SS']})
    for folder,label in [('architecture','Architecture'),('combined','Combined')]:
        r=json.load(open(out/folder/'results.json'))
        for name,x in r['results'].items():rows_out.append({'method':label+' / '+name,'accuracy':x['test']['strict_accuracy'],'cost_pct_SS':x['test']['cost_pct_SS']})
    write(out/'results.json',{'primary_metric':'official normalized answer EM','SS':protocol['SS'],'results':rows_out})
    lines=['# Corrected non-LLM router experiments','',
        '**Primary metric: official HotpotQA normalized answer exact match. All labels, sampled-answer rewards and operating points were rebuilt.**',
        'Qwen2.5-1.5B/7B Instruct FP16 outputs are reused; no Base generations. No LLM weights were trained.',
        'All cost percentages use the same direct-SS token ruler; sunk weak calls and consensus samples are included; CPU overhead excluded.',
        'Existing test is exploratory. Original reference 52.79% is validation lenient matching, not an EM target.',
        '',f"Direct SS validation EM: **{100*protocol['SS']['val']['accuracy']:.2f}%**; test EM: **{100*protocol['SS']['test']['accuracy']:.2f}%**.",'',
        '| Method | Test EM % | Cost % SS |','|---|---:|---:|']
    lines += [f"| {r['method']} | {100*r['accuracy']:.2f} | {r['cost_pct_SS']:.2f} |" for r in rows_out]
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n')
    # Keep compatibility JSON keys, but remove misleading metric names from human reports.
    for p in out.rglob('REPORT.md'):
        text=p.read_text().replace('lenient','exact-match').replace('Lenient','Exact-match')
        if p!=out/'REPORT.md':text='> Corrected run: both legacy accuracy columns represent official normalized EM. See root protocol.json.\n\n'+text
        p.write_text(text)

def main(out):
    if out.exists() and any(out.iterdir()):raise ValueError('Fresh output directory required')
    out.mkdir(parents=True);jobs=[]
    try:
        state(out,'preparing_official_EM');prepare(out)
        py=sys.executable;code=out/'code';base=out/'base'
        cascade=[py,'-u',str(code/'cascade.py'),'--labels-root',str(base/'labels'),'--token-costs',str(base/'token_costs.json')]
        jobs=[launch(out,'original',[py,'-u',str(code/'baselines.py'),'--data',str(base/'labels/full_run_p100/router_data_lenient.jsonl'),
            '--splits',str(ROOT/'extracted/Dataset/splits.json'),'--out',str(out/'original')]),
            launch(out,'cascade',cascade+['--qa-weak',str(base/'labels/full_run_p100/qa_weak.jsonl'),'--out',str(base/'router')]),
            launch(out,'pilot',cascade+['--without-weak-answer','--out',str(out/'pilot')]),
            launch(out,'architecture',[py,'-u',str(code/'architecture.py'),'--out',str(out/'architecture')])]
        # GPU streams run independently of CPU classifiers, using already-available features.
        free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
        if len(free.splitlines())!=4 or min(map(int,free.splitlines()))<2500:raise RuntimeError('Four GPUs with >=2500 MiB free required')
        jobs += [launch(out,f'extended_gpu_{i}',[py,'-u',str(code/'extended_policy.py'),'--out',str(out/'extended'),'--phase','train','--shard',str(i)],{'CUDA_VISIBLE_DEVICES':str(i)}) for i in range(4)]
        join(out,jobs,'training_original_cascade_architecture_and_extended');jobs=[]
        join(out,[launch(out,'extended_evaluate',[py,'-u',str(code/'extended_policy.py'),'--out',str(out/'extended'),'--phase','evaluate']),
            launch(out,'combined',[py,'-u',str(code/'architecture_combine.py'),'--parent',str(out/'architecture'),'--out',str(out/'combined')]),
            launch(out,'suite_cached',[py,'-u',str(code/'suite.py'),'--base',str(base),'--out',str(out/'initial/results'),'--phase','cached'])],
            'evaluating_extended_and_training_initial_suite')
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
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();main(a.out.resolve())
