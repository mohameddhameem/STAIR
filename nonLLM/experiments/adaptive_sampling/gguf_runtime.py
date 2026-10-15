"""Isolated GGUF replicas; never adopts or terminates external GPU work."""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request

PYTHON = '/home/zmcheng/miniconda3/envs/sglang518/bin/python'


def healthy(port):
    try:
        return urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).status == 200
    except Exception:
        return False


def launch(root, model, gpu, port):
    root = Path(root)
    used = int(subprocess.check_output(['nvidia-smi', '--id', str(gpu), '--query-gpu=memory.used',
                                       '--format=csv,noheader,nounits'], text=True).strip())
    if used > 1024:
        raise RuntimeError(f'GPU {gpu} already uses {used} MiB; refusing to overlap another task')
    profile = json.loads((root / 'profile.json').read_text())
    filename = ('qwen2.5-1.5b-instruct-q4_k_m.gguf' if model == 'W' else
                'qwen2.5-7b-instruct-q4_k_m-00001-of-00002.gguf')
    args = [str(root / 'bin/llama-server'), '--model', str(root / 'models' / filename),
            '--host', '127.0.0.1', '--port', str(port), '--n-gpu-layers', '99',
            '--ctx-size', str(profile['kv_pool_tokens']), '--parallel', str(profile['slots']),
            '--kv-unified', '--kv-unified-per-slot', str(profile['context_limit']),
            '--override-kv', f'qwen2.context_length=int:{profile["context_limit"]}',
            '--rope-scaling', 'yarn', '--rope-scale', '4', '--yarn-orig-ctx', '32768',
            '--rope-freq-base', '1000000', '--flash-attn', 'on', '--cache-type-k', 'f16',
            '--cache-type-v', 'f16', '--batch-size', '4096', '--ubatch-size', '512',
            '--threads', '8', '--threads-batch', '8', '--threads-http', '16',
            '--no-context-shift', '--metrics', '--cache-ram', '4096', '--fit', 'off']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='8')
    env['LD_LIBRARY_PATH'] = str(root / 'lib') + ':' + str(root / 'bin') + ':' + env.get('LD_LIBRARY_PATH', '')
    log = root / 'logs' / f'server_{model}_gpu{gpu}_{time.time_ns()}.log'
    with log.open('w') as out:
        process = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
    return process, {'pid': process.pid, 'gpu': gpu, 'port': port, 'model': model,
                     'log': str(log), 'command': args}


def stop(processes):
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
    deadline = time.time() + 30
    while time.time() < deadline and any(p.poll() is None for p in processes):
        time.sleep(.5)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass


def exclusive(processes):
    groups = {p.pid for p in processes if p.poll() is None}
    lines = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                     '--format=csv,noheader,nounits'], text=True)
    for line in lines.splitlines():
        try:
            pid = int(line.strip())
            group = os.getpgid(pid)
        except (ValueError, ProcessLookupError):
            continue
        if group not in groups:
            raise RuntimeError(f'External GPU process {pid}; stop only our own processes')


def wait_ready(processes, ports):
    deadline = time.time() + 600
    while not all(healthy(p) for p in ports):
        exclusive(processes)
        if any(p.poll() is not None for p in processes):
            raise RuntimeError('A GGUF replica exited during loading; inspect server logs')
        if time.time() > deadline:
            raise RuntimeError('GGUF readiness timeout')
        time.sleep(2)
