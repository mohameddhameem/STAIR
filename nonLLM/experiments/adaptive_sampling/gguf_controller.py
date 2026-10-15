"""Four-GPU serverA controller, fixed temperatures, resumable isolated caches."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sqlite3
import sys
import time

from gguf_runtime import PYTHON, exclusive, launch, stop, wait_ready


def main(root, gpus):
    root = Path(root)
    if not (root / 'gates/PASSED').exists():
        raise RuntimeError('Training integration gate has not passed')
    if not json.loads((root / 'gates/all_context.json').read_text())['passed']:
        raise RuntimeError('Full-corpus context bound has not passed')
    if not json.loads((root / 'gates/worker_smoke.json').read_text())['passed']:
        raise RuntimeError('The actual resumable worker smoke has not passed')
    files = ['profile.json', 'assets/splits.json', 'assets/AgentTTS-HotpotQA.json.gz']
    files += ['code/' + n for n in ['protocol.py', 'gguf_client.py', 'gguf_extract.py', 'answer_parser.py', 'grounded_answer.py', 'gguf_worker.py', 'gguf_runtime.py', 'gguf_gpu_queue.py']]
    locked = {name: hashlib.file_digest((root / name).open('rb'), 'sha256').hexdigest() for name in files}
    lock = root / 'inference_lock.json'
    if lock.exists() and json.loads(lock.read_text()) != locked:
        raise RuntimeError('Inference files changed; do not mix caches across protocols')
    lock.write_text(json.dumps(locked, indent=2))
    processes, servers = [], []
    current = None

    def state(**kwargs):
        path = root / 'state.json'
        value = {'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'controller_pid': os.getpid(),
                 'host': 'serverA', 'uses_serverB': False, 'physical_gpus': gpus, **kwargs}
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
        print(json.dumps(value), flush=True)

    def child(args, log_name):
        env = dict(os.environ, RAYON_NUM_THREADS='4', TOKENIZERS_PARALLELISM='true', OMP_NUM_THREADS='4')
        with (root / 'logs' / log_name).open('a') as out:
            p = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, env=env, start_new_session=True)
        processes.append(p)
        return p

    def terminate(signum, frame):
        raise RuntimeError(f'Controller received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        exclusive([])
        (root / 'runs/FAILED.txt').unlink(missing_ok=True)
        for stage, model in [('retrieval', 'W'), ('retrieval', 'S'), ('answers', 'W'),
                             ('answers', 'S'), ('extraction', 'S')]:
            if all((root / 'runs' / f'{stage}_{model}_{g}.done').exists() for g in range(4)):
                continue
            if stage == 'answers' and not (root / 'gates/qa_context.json').exists():
                state(stage='qa_context_preflight', model=model)
                subprocess.run([PYTHON, str(root / 'code/gguf_preflight.py'), '--root', str(root), '--retrievals'], check=True)
            if model != current:
                stop(servers)
                servers = []
                # Wait for CUDA teardown before admitting the replacement replicas.
                deadline = time.time() + 60
                while True:
                    used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                                    '--format=csv,noheader,nounits'], text=True)
                    if all(int(used.splitlines()[g].strip()) < 1024 for g in gpus):
                        break
                    if time.time() > deadline:
                        raise RuntimeError('GPUs are still occupied; refusing to overlap a launch')
                    exclusive(processes)
                    time.sleep(1)
                metadata = []
                for rank, gpu in enumerate(gpus):
                    process, info = launch(root, model, gpu, 30380 + rank)
                    processes.append(process)
                    servers.append(process)
                    metadata.append(info)
                state(stage='loading', next_stage=stage, model=model, servers=metadata)
                wait_ready(servers, [30380 + rank for rank in range(len(gpus))])
                current = model
            # Keep all replicas involved in each logical shard. A resumed run
            # can have most cached answers in one shard after reduced-GPU work;
            # assigning a whole shard to each GPU would then leave a card idle.
            workers = [child([PYTHON, str(root / 'code/gguf_gpu_queue.py'), '--root', str(root),
                              '--stage', stage, '--model', model, '--rank', str(rank),
                              '--ranks', str(len(gpus)), '--port', str(30380 + rank)],
                             f'worker_{stage}_{model}_gpu{gpu}.log') for rank, gpu in enumerate(gpus)]
            state(stage=stage, model=model, workers=[p.pid for p in workers], servers=[p.pid for p in servers])
            while any(p.poll() is None for p in workers):
                exclusive(processes)
                if any(p.poll() is not None for p in servers):
                    raise RuntimeError('A GGUF model server exited')
                if any(p.poll() not in [None, 0] for p in workers):
                    raise RuntimeError(f'{stage}/{model} worker failed; inspect logs')
                time.sleep(3)
            if any(p.returncode != 0 for p in workers):
                raise RuntimeError(f'{stage}/{model} worker failed')
            for shard in range(4):
                path = root / 'runs' / f'{stage}_{model}_{shard}.sqlite'
                db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
                count = db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
                db.close()
                expected = len(range(shard, 7405, 4)) * {'retrieval': 1, 'answers': 82, 'extraction': 160}[stage]
                if count != expected:
                    raise RuntimeError(f'Incomplete logical shard {stage}/{model}/{shard}: {count}/{expected}')
                (root / 'runs' / f'{stage}_{model}_{shard}.done').write_text('complete\n')
            print(json.dumps({'completed_stage': stage, 'model': model}), flush=True)
        stop(servers)
        state(stage='analysis')
        subprocess.run([PYTHON, str(root / 'code/gguf_analyze.py'), '--root', str(root)], check=True)
        if not (root / 'analysis/report.json').exists():
            raise RuntimeError('Final report is incomplete')
        (root / 'COMPLETE').write_text('All 7,405 questions, candidates, prefix fusions and scoring complete\n')
        state(stage='complete')
    except Exception as error:
        (root / 'runs/FAILED.txt').write_text(str(error))
        state(stage='failed', error=str(error))
        raise
    finally:
        stop(processes)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1, 2, 3])
    args = parser.parse_args()
    try:
        if not args.gpus or len(set(args.gpus)) != len(args.gpus) or any(g not in range(4) for g in args.gpus):
            raise RuntimeError('Invalid physical GPU selection')
        main(args.root, args.gpus)
    except Exception as error:
        print(str(error), file=sys.stderr, flush=True)
        sys.exit(1)
