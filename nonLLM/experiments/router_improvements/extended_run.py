"""Four-GPU sample expansion, balanced policy training, validation-only selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent
ROOT=Path(__file__).resolve().parents[4]
ENV=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')


def status(out,phase,**kw):
    p=out/'status.tmp';p.write_text(json.dumps({'phase':phase,**kw},indent=2));p.replace(out/'status.json')


def pool(out,phase,commands):
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if len(free.splitlines())!=4 or min(map(int,free.splitlines()))<6500:
        raise RuntimeError('Four available GPUs with >=6500 MiB free required')
    children=[];logs=[]
    try:
        for i,cmd in enumerate(commands):
            logs.append((out/f'{phase}_gpu_{i}.log').open('a'))
            children.append(subprocess.Popen(cmd,env=dict(ENV,CUDA_VISIBLE_DEVICES=str(i)),stdout=logs[-1],stderr=subprocess.STDOUT))
        status(out,phase,worker_pids=[p.pid for p in children])
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError(f'{phase} worker failed')
            time.sleep(5)
        if any(p.returncode for p in children):raise RuntimeError(f'{phase} worker failed')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in logs:f.close()


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    manifest={'base':str(ROOT/'runs/nonllm_cascade_four_gpu_full_v1'),
        'base_manifest':json.load(open(ROOT/'runs/nonllm_cascade_four_gpu_full_v1/manifest.json')),
        'prior_samples':str(ROOT/'runs/nonllm_router_improvements_v2/samples'),
        'more_samples':{'indices':[4,5,6,7],'split':'train only','count':38512,'temperature':.7,'top_p':.95,'top_k':0,'max_new_tokens':128},
        'policy':{'modes':['base','confidence','aug5','aug9'],'heads':'linear TF-IDF + numeric',
                  'epochs':[30,60,120],'warm_epochs':10,'groups':[8,16,32],'seeds':[17,29,43],
                  'lambdas':[0.,.15,.4],'methods':['exact','grpo'],'entropy_regularizer':.002,
                  'clip':.2,'rollout_KL':.01,'PPO_updates_per_group':2,'batch_questions':256,
                  'warm_lr':.01,'policy_lr':.002,'gradient_norm_clip':1.,
                  'update_matching':'All variants of a question in each minibatch; same optimizer-update counts across modes',
                  'numeric_fix':'Stable joint log probabilities and sigmoid(-logit) weak scores avoid subtraction saturation'},
        'selection':{'split':'validation only','checkpoint_candidates':[30,60,120],
                     'objectives':['cheap95','max_accuracy','ss_budget'],'test':'Previously examined; descriptive comparisons'},
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    try:
        commands=[[sys.executable,'-u',str(HERE/'collect.py'),'--base',str(ROOT/'runs/nonllm_cascade_four_gpu_full_v1'),
            '--out',str(a.out/'more_samples'),'--phase','trainval','--shard',str(i),
            '--sample-start','4','--sample-end','8','--skip-greedy','--sampling-splits','train'] for i in range(4)]
        pool(a.out,'sampling_extra4',commands)
        count=0
        for f in (a.out/'more_samples').glob('trainval_[0-3].jsonl'):
            with f.open() as stream:count+=sum(1 for _ in stream)
        if count!=38512:raise ValueError(f'Expected 38512 extra samples, got {count}')
        for mode in ['base','confidence','aug5','aug9']:
            status(a.out,'preparing_features',mode=mode)
            with (a.out/'prepare.log').open('a') as f:
                subprocess.run([sys.executable,'-u',str(HERE/'extended_policy.py'),'--out',str(a.out),'--phase','prepare','--mode',mode],
                    env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
        commands=[[sys.executable,'-u',str(HERE/'extended_policy.py'),'--out',str(a.out),'--phase','train','--shard',str(i)] for i in range(4)]
        pool(a.out,'training_policies',commands)
        status(a.out,'validation_selection_and_test_replay')
        with (a.out/'evaluate.log').open('a') as f:
            subprocess.run([sys.executable,'-u',str(HERE/'extended_policy.py'),'--out',str(a.out),'--phase','evaluate'],
                env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)
        status(a.out,'complete',report=str(a.out/'REPORT.md'))
    except BaseException as e:
        status(a.out,'failed',error=repr(e));raise


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);main(p.parse_args())
