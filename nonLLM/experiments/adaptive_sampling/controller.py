"""Four independent GPU replicas, staged W/S jobs, exclusive-use watchdog."""
import json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path
ROOT=Path('/data1/zmcheng/cs707_sampling40')
PYTHON='/home/zmcheng/miniconda3/envs/sglang518/bin/python'
children=[]
def spawn(args,name,env=None):
    log=open(ROOT/'logs'/f'{name}.log','a');p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True);children.append(p);return p

def stop(ps):
    for p in ps:
        try:os.killpg(p.pid,signal.SIGTERM)
        except ProcessLookupError:pass
    deadline=time.time()+30
    while time.time()<deadline and any(p.poll() is None for p in ps):time.sleep(1)
    for p in ps:
        try:os.killpg(p.pid,signal.SIGKILL)
        except ProcessLookupError:pass

def safety():
    # Only our process groups may occupy GPUs. Never terminate external work.
    groups={p.pid for p in children}
    rows=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    for row in rows.splitlines():
        try:pid=int(row.strip());pg=os.getpgid(pid)
        except (ValueError,ProcessLookupError):continue
        if pg not in groups:raise RuntimeError(f'External GPU process {pid}; stopped our own workload')

def healthy(port):
    try:return urllib.request.urlopen(f'http://127.0.0.1:{port}/health',timeout=3).status==200
    except Exception:return False

def run():
    safety()
    for stage,model in [('retrieval','W'),('retrieval','S'),('answers','W'),('answers','S')]:
        if all((ROOT/'runs'/f'{stage}_{model}_{g}.done').exists() for g in range(4)):continue
        size='1.5B' if model=='W' else '7B';servers=[]
        for g in range(4):
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=str(g);env['SGLANG_CACHE_DIR']=str(ROOT/'runtime_cache');env['OMP_NUM_THREADS']='8';env['PATH']='/home/zmcheng/miniconda3/envs/sglang518/bin:/usr/local/cuda-13.0/bin:'+env.get('PATH','')
            servers.append(spawn([PYTHON,'-m','sglang.launch_server','--model-path',str(ROOT/'models'/f'Qwen2.5-{size}-GGUF'/f'Qwen2.5-{size}.Q4_K_M.gguf'),'--tokenizer-path',str(ROOT/'tokenizers'/f'Qwen2.5-{size}'),'--load-format','gguf','--quantization','gguf','--host','127.0.0.1','--port',str(30080+g),'--context-length','131072','--mem-fraction-static','0.82','--max-running-requests','128','--chunked-prefill-size','8192'],f'server_{stage}_{model}_{g}',env))
        deadline=time.time()+1200
        while not all(healthy(30080+g) for g in range(4)):
            safety()
            if any(p.poll() is not None for p in servers):raise RuntimeError('Model server failed; inspect logs')
            if time.time()>deadline:raise RuntimeError('Server readiness timed out')
            time.sleep(5)
        workers=[spawn([PYTHON,str(ROOT/'code/worker.py'),'--root',str(ROOT),'--stage',stage,'--model',model,'--shard',str(g),'--port',str(30080+g),'--concurrency','4'],f'worker_{stage}_{model}_{g}') for g in range(4)]
        while any(p.poll() is None for p in workers):
            safety()
            if any(p.poll() is not None for p in servers):raise RuntimeError('Model server died')
            if any(p.poll() not in [None,0] for p in workers):raise RuntimeError('Worker failed; inspect logs')
            time.sleep(3)
        if any(p.returncode!=0 for p in workers):raise RuntimeError('Worker failed')
        stop(servers);print(json.dumps({'completed_stage':stage,'model':model}),flush=True)
    subprocess.run([PYTHON,str(ROOT/'code/analyze.py'),'--root',str(ROOT)],check=True);print('ALL COMPLETE',flush=True)
if __name__=='__main__':
    try:run()
    except Exception as e:
        (ROOT/'runs/FAILED.txt').write_text(str(e));print(str(e),flush=True);sys.exit(1)
    finally:stop(children)
