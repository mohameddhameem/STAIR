"""Compare fusion concurrency on training examples without writing run caches."""
import argparse
import asyncio
import gzip
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import time

import aiohttp
from transformers import AutoTokenizer

from gguf_client import Client
from gguf_runtime import PYTHON, exclusive, launch, stop, wait_ready
from gguf_worker import Admission
from protocol import load_protocol


async def benchmark(a):
    root = Path(a.root)
    profile = json.loads((root / 'profile.json').read_text())
    train = json.loads((root / 'assets/splits.json').read_text())['splits']['train']
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    prompt = load_protocol(root / 'upstream')
    tok = AutoTokenizer.from_pretrained(root / 'tokenizers' / a.model)
    client = Client(root, a.model, a.port, tok)
    indices = [0, 12]
    assert set(indices).issubset(train)
    jobs = []
    for i in indices:
        db = sqlite3.connect(f'file:{root}/runs/answers_W_{i % 4}.sqlite?mode=ro', uri=True)
        row = db.execute('SELECT payload FROM results WHERE key=?', (f'{i}:qa:WW',)).fetchone()
        db.close()
        answers = [x['text'] for x in json.loads(row[0])['outputs']]
        for k in [40, 36, 32, 28, 24, 20, 16, 12, 8, 4, 2, 1]:
            jobs.append((i, k, prompt('fusion', data[i], answers=answers[:k])))
    lanes = asyncio.Semaphore(a.lanes)
    admission = Admission(int(profile['kv_pool_tokens'] * .75), profile['slots'])
    rows = []
    start = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800),
                                     connector=aiohttp.TCPConnector(force_close=True)) as session:
        async def one(i, k, messages):
            async with lanes:
                reservation = len(client.encode(messages)) + 512 + 256
                await admission.acquire(reservation, 1)
                try:
                    value = await client.generate(session, f'benchmark:{i}:fuse:{a.model}:{k}', messages, .3)
                    rows.append({'index': i, 'budget': k, 'value': value})
                finally:
                    await admission.release(reservation, 1)
        # As in the worker, only two questions are in flight together.
        await asyncio.gather(*(one(*job) for job in jobs))
    seconds = time.time() - start
    result = {'model': a.model, 'lanes': a.lanes, 'questions': indices,
              'requests': len(rows), 'seconds': seconds,
              'requests_per_second': len(rows) / seconds,
              'output_tokens': sum(x['value']['outputs'][0]['meta_info']['completion_tokens'] for x in rows),
              'rows': rows}
    Path(a.output).write_text(json.dumps(result))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}), flush=True)


def orchestrate(a):
    root, out = Path(a.root), Path(a.output)
    out.mkdir(parents=True, exist_ok=True)
    samples, summary = [], []
    for model in a.models:
        servers, workers, logs = [], [], []
        try:
            for gpu in range(len(a.lane_options)):
                process, _ = launch(root, model, gpu, 30480 + gpu)
                servers.append(process)
            wait_ready(servers, list(range(30480, 30480 + len(servers))))
            launched = time.time()
            for gpu, lanes in enumerate(a.lane_options):
                path = out / f'{model}_lanes{lanes}.json'
                log = (out / f'{model}_lanes{lanes}.log').open('w')
                logs.append(log)
                env = dict(os.environ, RAYON_NUM_THREADS='4', TOKENIZERS_PARALLELISM='true', OMP_NUM_THREADS='4')
                worker = subprocess.Popen([PYTHON, __file__, '--root', str(root), '--model', model,
                                           '--port', str(30480 + gpu), '--lanes', str(lanes),
                                           '--output', str(path)], stdout=log, stderr=subprocess.STDOUT,
                                          env=env, start_new_session=True)
                workers.append(worker)
            while any(w.poll() is None for w in workers):
                exclusive(servers)
                if any(p.poll() is not None for p in servers):
                    raise RuntimeError('Benchmark server exited')
                if any(w.poll() not in (None, 0) for w in workers):
                    raise RuntimeError('Benchmark worker failed; inspect logs')
                if time.time() - launched > a.timeout:
                    for worker, lanes in zip(workers, a.lane_options):
                        if worker.poll() is None:
                            (out / f'{model}_lanes{lanes}_incomplete.json').write_text(json.dumps({
                                'model': model, 'lanes': lanes, 'status': 'incomplete_at_timeout',
                                'wall_seconds': time.time() - launched}))
                    stop(workers)
                    break
                line = subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used',
                                                '--format=csv,noheader,nounits'], text=True)
                samples.append({'time': time.time(), 'model': model, 'gpus': line})
                time.sleep(1)
            for lanes in a.lane_options:
                path = out / f'{model}_lanes{lanes}.json'
                if not path.exists():
                    path = out / f'{model}_lanes{lanes}_incomplete.json'
                value = json.loads(path.read_text())
                summary.append({k: v for k, v in value.items() if k != 'rows'})
            print(json.dumps(summary[-len(a.lane_options):]), flush=True)
        finally:
            stop(workers)
            stop(servers)
            for log in logs:
                log.close()
            (out / 'utilization.json').write_text(json.dumps(samples))
        time.sleep(2)
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', choices=['W', 'S'])
    parser.add_argument('--port', type=int)
    parser.add_argument('--lanes', type=int)
    parser.add_argument('--lane-options', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--models', choices=['W', 'S'], nargs='+', default=['W', 'S'])
    parser.add_argument('--timeout', type=float, default=90)
    args = parser.parse_args()
    if not (1 <= len(args.lane_options) <= 4 and
            len(set(args.lane_options)) == len(args.lane_options) and
            all(1 <= n <= 64 for n in args.lane_options) and args.timeout > 0):
        parser.error('Choose one to four distinct lane counts in 1..64 and a positive timeout')
    if args.model:
        asyncio.run(benchmark(args))
    else:
        orchestrate(args)
