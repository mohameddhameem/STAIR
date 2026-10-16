"""Run a resumable training-only Strong pilot, then restore the full controller."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import signal
import sqlite3
import subprocess
import time

from gguf_analyze import official
from gguf_runtime import PYTHON, exclusive, launch, stop, wait_ready


def read(db, key):
    row = db.execute('SELECT payload FROM results WHERE key=?', (key,)).fetchone()
    return json.loads(row[0]) if row else None


def validate(root, pilot):
    config = json.loads((pilot / 'config.json').read_text())
    ids = config['indices']
    train = set(json.loads((root / 'assets/splits.json').read_text())['splits']['train'])
    assert len(ids) == len(set(ids)) == 12 and set(ids) <= train
    lock = json.loads((root / 'inference_lock.json').read_text())
    assert all(hashlib.file_digest((root / p).open('rb'), 'sha256').hexdigest() == sha
               for p, sha in lock.items()), 'Frozen inference files changed'
    for g in range(4):
        db = sqlite3.connect(f'file:{root}/runs/answers_W_{g}.sqlite?mode=ro', uri=True)
        for i in ids:
            if i % 4 != g:
                continue
            assert db.execute('SELECT COUNT(*) FROM results WHERE key LIKE ?',
                              (f'{i}:%',)).fetchone()[0] == 82
        db.close()
    return config


def worker(a):
    root, pilot = Path(a.root), Path(a.pilot)
    ids = json.loads((pilot / 'config.json').read_text())['indices']
    selected = ids[a.rank::4]
    for shard in range(4):
        subset = [i for i in selected if i % 4 == shard]
        if not subset:
            continue
        path = pilot / f'indices_rank{a.rank}_shard{shard}.json'
        path.write_text(json.dumps(subset))
        subprocess.run([PYTHON, str(root / 'code/gguf_worker.py'), '--root', str(root),
                        '--stage', a.stage, '--model', 'S', '--shard', str(shard),
                        '--port', str(30580 + a.rank), '--indices-file', str(path)], check=True)


def report(root, pilot):
    config = json.loads((pilot / 'config.json').read_text())
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    metric = official(root)
    dbs = {}
    for stage, model in [('answers', 'W'), ('answers', 'S'), ('extraction', 'S')]:
        for shard in sorted({i % 4 for i in config['indices']}):
            dbs[stage, model, shard] = sqlite3.connect(
                f'file:{root}/runs/{stage}_{model}_{shard}.sqlite?mode=ro', uri=True)
    grid, examples, quality, scored = {}, [], Counter(), {}
    try:
        for i in config['indices']:
            item = {'index': i, 'question': data[i]['question'], 'gold_answers': data[i]['answers'], 'pairs': {}}
            for pair in ['WW', 'WS', 'SW', 'SS']:
                ans = dbs['answers', pair[1], i % 4]
                reader = dbs['extraction', 'S', i % 4]
                qa = read(ans, f'{i}:qa:{pair}')
                assert qa is not None and len(qa['outputs']) == 40
                if pair[1] == 'S':
                    quality['qa_batches'] += 1
                    quality['qa_unique_texts_total'] += len(set(x['text'] for x in qa['outputs']))
                    quality['qa_unique_seed_batches'] += int(len(set(x['seed'] for x in qa['outputs'])) == 40)
                shown, correct = {}, []
                for k in range(1, 41):
                    key = f'{i}:fuse:{pair}:{k}'
                    fused, ex = read(ans, key), read(reader, key)
                    assert fused is not None and ex is not None, key
                    text = fused['outputs'][0]['text']
                    assert hashlib.sha256(text.encode()).hexdigest() == ex['source_sha256']
                    answer = ex['extraction']['answer'] or ''
                    gold = data[i]['answers']
                    raw_em = max(metric['exact_match_score'](text, g) for g in gold)
                    em = max(metric['exact_match_score'](answer, g) for g in gold)
                    f1 = max(metric['f1_score'](answer, g)[0] for g in gold)
                    scored[i, pair, k] = float(em)
                    if em:
                        correct.append(k)
                    value = {'raw_em': float(raw_em), 'extracted_em': float(em),
                             'extracted_f1': float(f1), 'extraction_coverage': float(bool(answer))}
                    grid.setdefault((pair, k), []).append(value)
                    if pair[1] == 'S':
                        output = fused['outputs'][0]
                        quality['fusion_outputs'] += 1
                        quality['empty_fusions'] += int(not text.strip())
                        quality['input_truncated_fusions'] += int(bool(output['llama'].get('truncated')))
                        quality['fusion_output_limit'] += int(output['llama'].get('stop_type') == 'limit')
                    if k in [1, 5, 10, 20, 40]:
                        shown[str(k)] = {**value, 'extracted_answer': answer,
                                         'extraction_method': ex['extraction']['method'],
                                         'fusion_text': text}
                item['pairs'][pair] = {'budgets': shown, 'correct_ks': correct}
            examples.append(item)
        means = {pair: {str(k): {name: sum(v[name] for v in grid[pair, k]) / len(config['indices'])
                                 for name in grid[pair, k][0]} for k in range(1, 41)}
                 for pair in ['WW', 'WS', 'SW', 'SS']}
        observations = {}
        for pair in means:
            observations[pair] = {
                'k1_correct_k40_wrong': sum(scored[i, pair, 1] == 1 and scored[i, pair, 40] == 0 for i in config['indices']),
                'k40_correct_k1_wrong': sum(scored[i, pair, 40] == 1 and scored[i, pair, 1] == 0 for i in config['indices']),
                'retrospective_budget_oracle_em': sum(max(scored[i, pair, k] for k in range(1, 41))
                                                    for i in config['indices']) / len(config['indices'])}
        result = {'config': config, 'completed': time.strftime('%Y-%m-%d %H:%M:%S'),
                  'qualification': f'{len(config["indices"])} training questions selected without using correctness. Preliminary matched comparison only; not held-out accuracy, not an AgentTTS reproduction score. Oracle uses gold retrospectively and cannot be used at inference. No model/prompt/temperature changes.',
                  'grid': means, 'budget_observations': observations,
                  'strong_output_checks': dict(quality), 'examples': examples}
        (pilot / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        return result
    finally:
        for db in dbs.values():
            db.close()


def supervise(a):
    root, pilot = Path(a.root), Path(a.pilot)
    config = validate(root, pilot)
    servers, workers = [], []
    error = None

    def state(stage, **extra):
        value = {'updated': time.strftime('%Y-%m-%d %H:%M:%S'), 'controller_pid': os.getpid(),
                 'host': 'serverA', 'uses_serverB': False, 'physical_gpus': [0, 1, 2, 3],
                 'stage': stage, 'model': 'S', 'pilot': str(pilot),
                 'resume_main_after_pilot': True, **extra}
        (pilot / 'state.json').write_text(json.dumps(value, indent=2))
        temporary = root / 'state.priority.tmp'
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(root / 'state.json')
        print(json.dumps(value), flush=True)

    def terminate(signum, frame):
        raise RuntimeError(f'Priority supervisor received signal {signum}')

    signal.signal(signal.SIGTERM, terminate)
    signal.signal(signal.SIGINT, terminate)
    try:
        exclusive([])
        (root / 'runs/FAILED.txt').unlink(missing_ok=True)
        for gpu in range(4):
            p, _ = launch(root, 'S', gpu, 30580 + gpu)
            servers.append(p)
        state('priority_strong_loading', servers=[p.pid for p in servers])
        wait_ready(servers, list(range(30580, 30584)))
        for stage in ['answers', 'extraction']:
            workers = []
            for rank in range(4):
                env = dict(os.environ, RAYON_NUM_THREADS='4', TOKENIZERS_PARALLELISM='true', OMP_NUM_THREADS='4')
                with (pilot / f'{stage}_rank{rank}.log').open('a') as out:
                    p = subprocess.Popen([PYTHON, __file__, '--root', str(root), '--pilot', str(pilot),
                                          '--worker', '--stage', stage, '--rank', str(rank)],
                                         stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         env=env, start_new_session=True)
                workers.append(p)
            state('priority_strong_' + stage, workers=[p.pid for p in workers], servers=[p.pid for p in servers])
            while any(p.poll() is None for p in workers):
                exclusive(servers + workers)
                if any(p.poll() is not None for p in servers):
                    raise RuntimeError('Priority Strong server exited')
                if any(p.poll() not in [None, 0] for p in workers):
                    raise RuntimeError('Priority worker failed; inspect pilot logs')
                time.sleep(2)
            if any(p.returncode != 0 for p in workers):
                raise RuntimeError('Priority worker failed')
        result = report(root, pilot)
        (pilot / 'COMPLETE').write_text('Strong candidates, all prefixes, matched answer extraction and report complete\n')
        state('priority_strong_complete', report=str(pilot / 'report.json'))
    except Exception as exc:
        error = str(exc)
        (pilot / 'FAILED.txt').write_text(error)
        state('priority_strong_failed', error=error)
    finally:
        stop(workers)
        stop(servers)
        deadline = time.time() + 60
        while time.time() < deadline:
            used = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                                            '--format=csv,noheader,nounits'], text=True)
            if all(int(x) < 1024 for x in used.splitlines()):
                break
            time.sleep(1)
        try:
            exclusive([])
            command = [PYTHON, str(root / 'code/gguf_controller.py'), '--root', str(root), '--gpus', '0', '1', '2', '3']
            with (root / 'logs/controller.log').open('a') as log:
                main = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=log,
                                        stderr=subprocess.STDOUT, cwd=root, start_new_session=True)
            (root / 'controller.pid').write_text(str(main.pid) + '\n')
            (pilot / 'main_resumed.json').write_text(json.dumps({'pid': main.pid, 'command': command,
                                                               'time': time.strftime('%Y-%m-%d %H:%M:%S'),
                                                               'pilot_error': error}, indent=2))
        except Exception as exc:
            (pilot / 'RESUME_FAILED.txt').write_text(str(exc))
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--pilot', required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--report', action='store_true')
    parser.add_argument('--stage', choices=['answers', 'extraction'])
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    if args.validate:
        print(json.dumps(validate(Path(args.root), Path(args.pilot)), indent=2))
    elif args.report:
        result = report(Path(args.root), Path(args.pilot))
        pilot = Path(args.pilot)
        failure = pilot / 'FAILED.txt'
        if failure.exists():
            failure.replace(pilot / 'FAILED_before_report_fix.txt')
        (pilot / 'COMPLETE').write_text('Strong candidates, all prefixes, matched answer extraction and report complete\n')
        (pilot / 'report_rebuilt.json').write_text(json.dumps({
            'time': result['completed'], 'model_inference_rerun': False,
            'fix': 'Open only shards represented by the selected questions; absent unused shards are valid.'}, indent=2))
        state_path = pilot / 'state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        state.pop('error', None)
        state.update(stage='priority_strong_complete', updated=result['completed'], report=str(pilot / 'report.json'))
        state_path.write_text(json.dumps(state, indent=2))
        print(json.dumps({'completed': result['completed'], 'strong_output_checks': result['strong_output_checks']}))
    elif args.worker:
        worker(args)
    else:
        supervise(args)
