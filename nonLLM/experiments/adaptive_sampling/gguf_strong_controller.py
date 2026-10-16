"""Finish all Strong answers, frozen v2 extraction and Strong-only analysis."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sqlite3
import subprocess
import time

from gguf_runtime import PYTHON, exclusive, launch, stop, wait_ready


def count(path, fusions=False):
    if not path.exists():
        return 0
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True, timeout=60)
    query = "SELECT COUNT(*) FROM results" + (" WHERE key LIKE '%:fuse:%'" if fusions else '')
    value = db.execute(query).fetchone()[0]
    db.close()
    return value


def prepare(root):
    lock = json.loads((root / 'inference_lock.json').read_text())
    assert all(hashlib.file_digest((root / name).open('rb'), 'sha256').hexdigest() == sha
               for name, sha in lock.items()), 'Frozen inference files changed'
    pilot = root / 'priority_strong_12/evaluation_v2'
    assert (pilot / 'COMPLETE').exists(), 'Training v2 reader validation not complete'
    profile = json.loads((pilot / 'profile.json').read_text())
    assert profile['evaluation']['wrapper_sha256'] == hashlib.sha256(
        (root / 'code/gguf_pilot_reextract.py').read_bytes()).hexdigest()
    assert profile['evaluation']['original_profile_sha256'] == hashlib.sha256(
        (root / 'profile.json').read_bytes()).hexdigest()
    reader = root / 'evaluation_v2'
    reader.mkdir(exist_ok=True)
    if (reader / 'profile.json').exists():
        assert (reader / 'profile.json').read_bytes() == (pilot / 'profile.json').read_bytes()
    else:
        shutil.copy2(pilot / 'profile.json', reader / 'profile.json')
    for name in ['assets', 'tokenizers', 'upstream', 'code', 'models', 'bin', 'lib']:
        if not (reader / name).exists():
            (reader / name).symlink_to(root / name, target_is_directory=True)
    for name in ['runs', 'logs']:
        (reader / name).mkdir(exist_ok=True)
    for model in ['W', 'S']:
        for shard in range(4):
            name = f'answers_{model}_{shard}.sqlite'
            path = reader / 'runs' / name
            if not path.is_symlink() and not path.exists():
                path.symlink_to(root / 'runs' / name)
    # Copy, rather than hard-link, the frozen reference extraction caches.
    for shard in range(4):
        source = pilot / 'runs' / f'extraction_S_{shard}.sqlite'
        target = reader / 'runs' / source.name
        if source.exists() and not target.exists():
            src = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
            dst = sqlite3.connect(target)
            src.backup(dst)
            dst.close()
            src.close()
    # Interleave logical shards so every physical rank shares their cached work.
    indices = [start + shard + 4 * rank for start in range(0, 7405, 16)
               for shard in range(4) for rank in range(4) if start + shard + 4 * rank < 7405]
    assert len(indices) == len(set(indices)) == 7405 and set(indices) == set(range(7405))
    (reader / 'config.json').write_text(json.dumps({
        'indices': indices, 'scope': 'All 7405 questions; no gold used for extraction.',
        'evaluation': profile['evaluation'], 'generation_root': str(root)}, indent=2))
    plan = {'reader_root': str(reader), 'pairs': ['WS', 'SS'], 'physical_gpus': [0, 1, 2, 3],
            'stages': ['answers_S', 'extraction_v2_available_W_and_all_S', 'analysis_WS_SS'],
            'weak_generation_paused': True, 'temperature_sweep': False,
            'evaluation': profile['evaluation'],
            'weak_answer_counts_at_start': [count(root / 'runs' / f'answers_W_{g}.sqlite') for g in range(4)],
            'reader_expected': sum(count(root / 'runs' / f'answers_W_{g}.sqlite', True) for g in range(4)) + 7405 * 80}
    path = root / 'strong_workflow.json'
    if path.exists():
        previous = json.loads(path.read_text())
        assert previous['weak_answer_counts_at_start'] == plan['weak_answer_counts_at_start'], 'Weak sources changed during Strong-only resume'
    path.write_text(json.dumps(plan, indent=2))
    return reader, plan


def main(root, validate=False):
    root = Path(root)
    reader, plan = prepare(root)
    if validate:
        print(json.dumps(plan, indent=2))
        return
    owned, servers, workers = [], [], []

    def state(stage, **extra):
        value = {'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'controller_pid': os.getpid(),
                 'host': 'serverA', 'uses_serverB': False, 'physical_gpus': [0, 1, 2, 3],
                 'model': 'S', 'stage': stage, 'reader_root': str(reader),
                 'weak_generation_paused': True, 'reader_expected': plan['reader_expected'], **extra}
        path = root / 'state.strong.tmp'
        path.write_text(json.dumps(value, indent=2))
        path.replace(root / 'state.json')
        print(json.dumps(value), flush=True)

    def child(args, log):
        env = dict(os.environ, RAYON_NUM_THREADS='4', TOKENIZERS_PARALLELISM='true', OMP_NUM_THREADS='4')
        with (root / 'logs' / log).open('a') as output:
            proc = subprocess.Popen(args, stdin=subprocess.DEVNULL, stdout=output,
                                    stderr=subprocess.STDOUT, env=env, start_new_session=True)
        owned.append(proc)
        return proc

    def wait(workers):
        while any(p.poll() is None for p in workers):
            exclusive(owned)
            if any(p.poll() is not None for p in servers) or any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError('Strong workflow process failed; inspect logs')
            time.sleep(3)
        if any(p.returncode != 0 for p in workers):
            raise RuntimeError('Strong workflow worker failed')

    def terminate(signum, frame):
        raise RuntimeError(f'Strong workflow received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        exclusive([])
        (root / 'runs/FAILED.txt').unlink(missing_ok=True)
        for gpu in range(4):
            proc, _ = launch(root, 'S', gpu, 30680 + gpu)
            owned.append(proc)
            servers.append(proc)
        state('strong_loading', servers=[p.pid for p in servers])
        wait_ready(servers, list(range(30680, 30684)))
        workers = [child([PYTHON, str(root / 'code/gguf_gpu_queue.py'), '--root', str(root),
                          '--stage', 'answers', '--model', 'S', '--rank', str(rank), '--ranks', '4',
                          '--port', str(30680 + rank)], f'strong_answers_gpu{rank}.log') for rank in range(4)]
        state('answers', workers=[p.pid for p in workers], servers=[p.pid for p in servers], workflow='strong_first')
        wait(workers)
        for shard in range(4):
            expected = len(range(shard, 7405, 4)) * 82
            assert count(root / 'runs' / f'answers_S_{shard}.sqlite') == expected
            (root / 'runs' / f'answers_S_{shard}.done').write_text('complete\n')
        workers = [child([PYTHON, str(root / 'code/gguf_pilot_reextract.py'), '--root', str(reader),
                          '--worker', '--rank', str(rank)], f'strong_reader_v2_gpu{rank}.log') for rank in range(4)]
        state('strong_extraction_v2', workers=[p.pid for p in workers], servers=[p.pid for p in servers])
        wait(workers)
        for shard in range(4):
            expected = sum(count(root / 'runs' / f'answers_{m}_{shard}.sqlite', True) for m in ['W', 'S'])
            assert count(reader / 'runs' / f'extraction_S_{shard}.sqlite') == expected
            (reader / 'runs' / f'extraction_S_{shard}.done').write_text('All available fixed source fusions extracted\n')
        stop(servers)
        state('strong_analysis')
        analysis = child([PYTHON, str(root / 'code/gguf_analyze.py'), '--root', str(root),
                          '--output', 'analysis_strong_v2', '--pairs', 'WS', 'SS',
                          '--reader-root', str(reader)], 'strong_analysis.log')
        if analysis.wait() != 0:
            raise RuntimeError('Strong analysis failed')
        report = json.loads((root / 'analysis_strong_v2/report.json').read_text())
        assert report['complete_questions'] == 7405 and report['pairs'] == ['WS', 'SS']
        (root / 'STRONG_COMPLETE').write_text('All Strong candidates, prefixes, frozen v2 extraction and WS/SS analysis complete. Weak generation remains incomplete.\n')
        state('strong_complete', report=str(root / 'analysis_strong_v2/report.json'))
    except Exception as exc:
        (root / 'runs/FAILED.txt').write_text(str(exc))
        state('strong_failed', error=str(exc))
        raise
    finally:
        stop(owned)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    main(args.root, args.validate)
