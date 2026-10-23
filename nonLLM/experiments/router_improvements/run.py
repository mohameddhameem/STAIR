"""Autonomous local experiment chain; records progress and stops on worker errors."""
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


def state(out,phase,**kwargs):
    tmp=out/'status.tmp';tmp.write_text(json.dumps({'phase':phase,**kwargs},indent=2));tmp.replace(out/'status.json')


def gpu_phase(a,phase):
    statuses=[a.out/f'samples/{phase}_{i}.status.json' for i in range(4)]
    if all(p.exists() and json.loads(p.read_text())['phase']=='complete' for p in statuses):return
    free=subprocess.check_output(['nvidia-smi','--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True)
    if len(free.splitlines())!=4 or min(int(s) for s in free.splitlines())<6500:
        raise RuntimeError('Four GPUs with at least 6500 MiB free each required; no automatic contention retry')
    children=[];logs=[]
    try:
        for i in range(4):
            logs.append((a.out/f'{phase}_gpu_{i}.log').open('a'))
            children.append(subprocess.Popen([sys.executable,'-u',str(HERE/'collect.py'),'--base',str(a.base),
                '--out',str(a.out/'samples'),'--shard',str(i),'--phase',phase],
                env=dict(ENV,CUDA_VISIBLE_DEVICES=str(i)),stdout=logs[-1],stderr=subprocess.STDOUT))
        state(a.out,'collect_'+phase,worker_pids=[p.pid for p in children])
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None,0) for p in children):raise RuntimeError(f'{phase} GPU worker failed')
            time.sleep(5)
        if any(p.returncode for p in children):raise RuntimeError(f'{phase} GPU worker failed')
    finally:
        for p in children:
            if p.poll() is None:p.terminate()
        for p in children:
            try:p.wait(timeout=15)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        for f in logs:f.close()


def cpu(a,phase):
    state(a.out,'fit_'+phase)
    with (a.out/f'{phase}_cpu.log').open('a') as f:
        subprocess.run([sys.executable,'-u',str(HERE/'suite.py'),'--base',str(a.base),'--out',str(a.out/'results'),
            '--phase',phase,'--samples',str(a.out/'samples')],env=ENV,stdout=f,stderr=subprocess.STDOUT,check=True)


def main(a):
    a.out.mkdir(parents=True,exist_ok=True)
    manifest={'base':str(a.base),'source_manifest':json.load(open(a.base/'manifest.json')),
        'protocol':{'new_weak_samples':4,'temperature':.7,'top_p':.95,'top_k':0,'max_new_tokens':128,
                    'seeds':'SHA256(idx|branch|sample)','strong_answerer':'existing greedy cache',
                    'splits':'unchanged question IDs; train only for fitting; val only for selection',
                    'test':'Previously examined, descriptive fixed-policy comparisons',
                    'GRPO':'finite-action group-relative PPO adaptation, 8 trajectories, 3 seeds, linear TF-IDF policies',
                    'lambdas':[0.,.15,.4,.8],'consensus':'one greedy + four sampled weak calls, all costs charged'},
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in HERE.glob('*.py')}}
    (a.out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    cached=None;cachedlog=None
    try:
        # A concurrently started cached suite can finish while GPU data is collected.
        if not a.cached_pid:
            cachedlog=(a.out/'cached_cpu.log').open('a')
            cached=subprocess.Popen([sys.executable,'-u',str(HERE/'suite.py'),'--base',str(a.base),
                '--out',str(a.out/'results'),'--phase','cached'],env=ENV,stdout=cachedlog,stderr=subprocess.STDOUT)
            a.cached_pid=cached.pid
        gpu_phase(a,'trainval')
        if a.cached_pid:
            state(a.out,'waiting_cached_suite',pid=a.cached_pid)
            while True:
                try:os.kill(a.cached_pid,0)
                except ProcessLookupError:break
                status=Path(f'/proc/{a.cached_pid}/stat')
                if status.exists() and status.read_text().split()[2]=='Z':break
                time.sleep(5)
            if not (a.out/'results/summary.json').exists():raise RuntimeError('Cached suite failed; inspect cached CPU log')
            if cached and cached.wait()!=0:raise RuntimeError('Cached suite failed')
        cpu(a,'sampled')  # Locks consensus thresholds without any test samples.
        gpu_phase(a,'test')
        cpu(a,'finish')
        state(a.out,'complete',report=str(a.out/'results/REPORT.md'))
    except BaseException as e:
        if cached and cached.poll() is None:
            cached.terminate()
            try:cached.wait(timeout=15)
            except subprocess.TimeoutExpired:cached.kill();cached.wait()
        state(a.out,'failed',error=repr(e));raise
    finally:
        if cachedlog:cachedlog.close()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--cached-pid',type=int);main(p.parse_args())
