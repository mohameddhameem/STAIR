"""Prioritize 500 comparable questions, export tables, then resume full Strong."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

from gguf_priority import report
from gguf_runtime import PYTHON, exclusive, launch, stop, wait_ready


def prefix_count(db, index):
    return db.execute('SELECT COUNT(*) FROM results WHERE key>=? AND key<?',
                      (f'{index}:', f'{index};')).fetchone()[0]


def prepare(root, cohort):
    config = json.loads((cohort / 'config.json').read_text())
    ids = config['indices']
    train = set(json.loads((root / 'assets/splits.json').read_text())['splits']['train'])
    assert len(ids) == len(set(ids)) == 500 and set(ids) <= train
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    assert len({data[i]['_id'] for i in ids}) == 500
    assert len({' '.join(data[i]['question'].lower().split()) for i in ids}) == 500
    lock = json.loads((root / 'inference_lock.json').read_text())
    assert all(hashlib.file_digest((root / p).open('rb'), 'sha256').hexdigest() == sha for p, sha in lock.items())
    for shard in sorted({i % 4 for i in ids}):
        db = sqlite3.connect(f'file:{root}/runs/answers_W_{shard}.sqlite?mode=ro', uri=True)
        assert all(prefix_count(db, i) == 82 for i in ids if i % 4 == shard)
        db.close()
    global_reader = root / 'evaluation_v2'
    profile = json.loads((global_reader / 'profile.json').read_text())
    assert profile['evaluation']['wrapper_sha256'] == hashlib.sha256((root / 'code/gguf_pilot_reextract.py').read_bytes()).hexdigest()
    assert profile['evaluation']['original_profile_sha256'] == hashlib.sha256((root / 'profile.json').read_bytes()).hexdigest()
    reader = cohort / 'evaluation_v2'
    reader.mkdir(exist_ok=True)
    if (reader / 'profile.json').exists():
        assert (reader / 'profile.json').read_bytes() == (global_reader / 'profile.json').read_bytes()
    else:
        (reader / 'profile.json').write_bytes((global_reader / 'profile.json').read_bytes())
    for name in ['assets', 'tokenizers', 'upstream', 'code', 'models', 'bin', 'lib']:
        if not (reader / name).exists():
            (reader / name).symlink_to(root / name, target_is_directory=True)
    for name in ['runs', 'logs']:
        (reader / name).mkdir(exist_ok=True)
    (reader / 'config.json').write_text(json.dumps(config, indent=2))
    for shard in sorted({i % 4 for i in ids}):
        for model in ['W', 'S']:
            path = reader / 'runs' / f'answers_{model}_{shard}.sqlite'
            if not path.is_symlink() and not path.exists():
                path.symlink_to(root / 'runs' / path.name)
        source = global_reader / 'runs' / f'extraction_S_{shard}.sqlite'
        target = reader / 'runs' / source.name
        if source.exists() and not target.exists():
            src = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
            dst = sqlite3.connect(target)
            src.backup(dst)
            # Keep this cohort's accounting independent of unrelated cached IDs.
            remove = [(key,) for key, in dst.execute('SELECT key FROM results') if int(key.split(':')[0]) not in set(ids)]
            dst.executemany('DELETE FROM results WHERE key=?', remove)
            dst.commit()
            dst.close()
            src.close()
    return config, reader


def worker(a):
    root, cohort = Path(a.root), Path(a.cohort)
    ids = json.loads((cohort / 'config.json').read_text())['indices'][a.rank::4]
    for shard in range(4):
        subset = [i for i in ids if i % 4 == shard]
        if not subset:
            continue
        path = cohort / f'indices_{a.rank}_{shard}.json'
        path.write_text(json.dumps(subset))
        subprocess.run([PYTHON, str(root / 'code/gguf_worker.py'), '--root', str(root),
                        '--stage', 'answers', '--model', 'S', '--shard', str(shard),
                        '--port', str(30680 + a.rank), '--indices-file', str(path)], check=True)


def export_tables(result, cohort):
    n = len(result['config']['indices'])
    summary = {'questions': n, 'scope': result['qualification'],
               'definition': 'Earliest strictly correct extracted answer, gold-assisted. Counts overlap across pairs; not true optimal budgets. Alias and extraction errors can affect EM.', 'pairs': {}}
    lines = ['# Matched 500-question reference', '', result['config']['scope'], '',
             'W=1.5B, S=7B. First letter: retrieval; second: answer/fusion.', '',
             'Strict extracted EM. Every question has all four pairs and k=1..40.', '',
             '| Pair | Any successful k | First k=1 | 2-5 | 6-10 | 11-20 | 21-40 |',
             '|---|---:|---:|---:|---:|---:|---:|']
    rows = []
    for pair, grid in result['grid'].items():
        first = []
        for item in result['examples']:
            ks = item['pairs'][pair]['correct_ks']
            if ks:
                first.append(min(ks))
            rows.append([item['index'], pair, min(ks) if ks else '', len(ks), json.dumps(ks)])
        bins = [sum(k == 1 for k in first)] + [sum(lo <= k <= hi for k in first) for lo, hi in [(2, 5), (6, 10), (11, 20), (21, 40)]]
        best = max(v['extracted_em'] for v in grid.values())
        summary['pairs'][pair] = {'ever_correct_questions': len(first), 'first_success_k_histogram': dict(sorted(Counter(first).items())),
                                   'first_success_bins': dict(zip(['1', '2-5', '6-10', '11-20', '21-40'], bins)),
                                   'best_fixed_ks': [int(k) for k, v in grid.items() if v['extracted_em'] == best],
                                   'best_fixed_hits': round(best * n)}
        lines.append('| ' + ' | '.join(map(str, [pair, len(first), *bins])) + ' |')
    lines += ['', 'Counts overlap across pairs. This training availability-based cohort is not a uniform sample or held-out benchmark.', '',
              '| Pair | Best shared k on this cohort | Strict hits | Retrospective budget oracle hits |', '|---|---|---:|---:|']
    for pair, value in summary['pairs'].items():
        lines.append(f'| {pair} | {value["best_fixed_ks"]} | {value["best_fixed_hits"]} | {value["ever_correct_questions"]} |')
    lines += ['', 'Gold-assisted oracle selection is descriptive, not a deployable or cost-constrained policy. Strict EM can miss aliases; long-budget successes need manual interpretation.', '']
    (cohort / 'summary.json').write_text(json.dumps(summary, indent=2))
    (cohort / 'table.md').write_text('\n'.join(lines))
    with (cohort / 'per_question.csv').open('w') as out:
        csv.writer(out).writerows([['index', 'pair', 'first_success_k', 'correct_budget_count', 'correct_ks'], *rows])
    with (cohort / 'budget_grid.csv').open('w') as out:
        writer = csv.writer(out)
        writer.writerow(['pair', 'k', 'em_hits', 'em', 'f1', 'extraction_coverage'])
        for pair, grid in result['grid'].items():
            for k, v in grid.items():
                writer.writerow([pair, k, round(v['extracted_em'] * n), v['extracted_em'], v['extracted_f1'], v['extraction_coverage']])


def merge_reader(root, reader):
    for source in (reader / 'runs').glob('extraction_S_*.sqlite'):
        target = root / 'evaluation_v2/runs' / source.name
        src = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
        dst = sqlite3.connect(target, timeout=60)
        try:
            if not dst.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='results'").fetchone():
                src.backup(dst)
            else:
                assert src.execute("SELECT value FROM metadata WHERE key='profile'").fetchone() == dst.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
                for key, payload in src.execute('SELECT key,payload FROM results'):
                    old = dst.execute('SELECT payload FROM results WHERE key=?', (key,)).fetchone()
                    assert old is None or old[0] == payload, f'Extraction cache conflict: {key}'
                    if old is None:
                        dst.execute('INSERT INTO results VALUES (?,?)', (key, payload))
                dst.commit()
        except Exception:
            dst.rollback()
            raise
        finally:
            src.close()
            dst.close()


def main(a):
    root, cohort = Path(a.root), Path(a.cohort)
    config, reader = prepare(root, cohort)
    if a.validate:
        print(json.dumps({'questions': len(config['indices']), 'reader_root': str(reader), 'Strong_already_complete_at_selection': config['Strong_complete_at_selection']}))
        return
    owned, servers = [], []

    def state(stage, **extra):
        value = {'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'controller_pid': os.getpid(), 'host': 'serverA',
                 'physical_gpus': [0, 1, 2, 3], 'uses_serverB': False, 'model': 'S', 'workflow': 'cohort_500',
                 'stage': stage, 'cohort': str(cohort), 'reader_root': str(reader), 'reader_expected': 80000,
                 'reader_scope': '500 training questions, all four pairs', 'weak_generation_paused': True, **extra}
        (cohort / 'state.json').write_text(json.dumps(value, indent=2))
        temp = root / 'state.cohort.tmp'
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(root / 'state.json')
        print(json.dumps(value), flush=True)

    def child(args, log):
        with (cohort / log).open('a') as out:
            p = subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                 env=dict(os.environ, RAYON_NUM_THREADS='4', TOKENIZERS_PARALLELISM='true', OMP_NUM_THREADS='4'), start_new_session=True)
        owned.append(p)
        return p

    def wait(workers):
        while any(p.poll() is None for p in workers):
            exclusive(owned)
            if any(p.poll() is not None for p in servers) or any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError('Cohort process failed; inspect cohort logs')
            time.sleep(3)
        assert all(p.returncode == 0 for p in workers)

    def terminate(signum, frame):
        raise RuntimeError(f'Cohort controller received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    success = False
    try:
        exclusive([])
        (root / 'runs/FAILED.txt').unlink(missing_ok=True)
        for gpu in range(4):
            p, _ = launch(root, 'S', gpu, 30680 + gpu)
            servers.append(p)
            owned.append(p)
        state('cohort500_loading', servers=[p.pid for p in servers])
        wait_ready(servers, list(range(30680, 30684)))
        workers = [child([PYTHON, __file__, '--root', str(root), '--cohort', str(cohort), '--worker', '--rank', str(rank)], f'answers_rank{rank}.log') for rank in range(4)]
        state('cohort500_answers', workers=[p.pid for p in workers], servers=[p.pid for p in servers])
        wait(workers)
        workers = [child([PYTHON, str(root / 'code/gguf_pilot_reextract.py'), '--root', str(reader), '--worker', '--rank', str(rank)], f'reader_rank{rank}.log') for rank in range(4)]
        state('cohort500_extraction', workers=[p.pid for p in workers], servers=[p.pid for p in servers])
        wait(workers)
        state('cohort500_report')
        result = report(reader, reader)
        assert len(result['examples']) == 500
        export_tables(result, cohort)
        merge_reader(root, reader)
        (cohort / 'COMPLETE').write_text('500 unique questions, four pairs, forty budgets, frozen v2 extraction and comparison tables complete\n')
        state('cohort500_complete', table=str(cohort / 'table.md'))
        success = True
    except Exception as exc:
        (cohort / 'FAILED.txt').write_text(str(exc))
        (root / 'runs/FAILED.txt').write_text(str(exc))
        state('cohort500_failed', error=str(exc))
        raise
    finally:
        stop(owned)
    if success:
        try:
            deadline = time.time() + 60
            while True:
                exclusive([])
                used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'], text=True)
                if all(int(value) < 1024 for value in used.splitlines()):
                    break
                if time.time() > deadline:
                    raise RuntimeError('GPU memory still occupied after cohort cleanup')
                time.sleep(1)
            command = [PYTHON, str(root / 'code/gguf_strong_controller.py'), '--root', str(root)]
            proc = child(command, 'full_strong_resumed.log')
            (root / 'controller.pid').write_text(str(proc.pid) + '\n')
            (root / 'strong_controller.pid').write_text(str(proc.pid) + '\n')
            (cohort / 'full_strong_resumed.json').write_text(json.dumps({'pid': proc.pid, 'command': command, 'time': time.strftime('%Y-%m-%d %H:%M:%S')}, indent=2))
        except Exception as exc:
            (cohort / 'RESUME_FAILED.txt').write_text(str(exc))
            state('cohort500_resume_failed', error=str(exc), table=str(cohort / 'table.md'))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--cohort', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rank', type=int)
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    worker(args) if args.worker else main(args)
