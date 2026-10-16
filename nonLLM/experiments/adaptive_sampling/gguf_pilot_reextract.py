"""Versioned pilot evaluation: narrative replies always use the grounded reader."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from gguf_runtime import PYTHON, exclusive, launch, stop, wait_ready


def worker(a):
    import gguf_extract
    from gguf_worker import main
    original = gguf_extract.extract_answer

    def short_only(text):
        # Apply the same rule to every reply, without consulting gold answers.
        # Longer prose requires the existing gold-blind grounded span reader.
        if len(text.strip().split()) > 6:
            return None
        return original(text)

    gguf_extract.extract_answer = short_only
    ids = json.loads((Path(a.root) / 'config.json').read_text())['indices'][a.rank::4]
    for shard in range(4):
        subset = [i for i in ids if i % 4 == shard]
        if not subset:
            continue
        path = Path(a.root) / f'indices_{a.rank}_{shard}.json'
        path.write_text(json.dumps(subset))
        asyncio.run(main(SimpleNamespace(root=a.root, stage='extraction', model='S',
                                         shard=shard, port=30680 + a.rank,
                                         indices_file=str(path), limit=0)))


def run(a):
    root, original = Path(a.root), Path(a.source)
    root.mkdir(exist_ok=True)
    for name in ['assets', 'tokenizers', 'upstream', 'code', 'models', 'bin', 'lib']:
        path = root / name
        if not path.exists():
            path.symlink_to(original / name, target_is_directory=True)
    for name in ['runs', 'logs']:
        (root / name).mkdir(exist_ok=True)
    config = json.loads((original / 'priority_strong_12/config.json').read_text())
    (root / 'config.json').write_text(json.dumps(config, indent=2))
    profile = json.loads((original / 'profile.json').read_text())
    profile['evaluation'] = {'version': 2, 'rule': 'Replies over six whitespace-separated words always use the existing gold-independent grounded reader.',
                             'wrapper_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                             'original_profile_sha256': hashlib.sha256((original / 'profile.json').read_bytes()).hexdigest()}
    (root / 'profile.json').write_text(json.dumps(profile, indent=2))
    for model in ['W', 'S']:
        for shard in sorted({i % 4 for i in config['indices']}):
            name = f'answers_{model}_{shard}.sqlite'
            path = root / 'runs' / name
            if not path.exists():
                path.symlink_to(original / 'runs' / name)
    servers, workers = [], []
    try:
        exclusive([])
        for gpu in range(4):
            p, _ = launch(root, 'S', gpu, 30680 + gpu)
            servers.append(p)
        wait_ready(servers, list(range(30680, 30684)))
        for rank in range(4):
            env = dict(os.environ, RAYON_NUM_THREADS='4', TOKENIZERS_PARALLELISM='true', OMP_NUM_THREADS='4')
            with (root / 'logs' / f'worker_{rank}.log').open('a') as out:
                p = subprocess.Popen([PYTHON, __file__, '--root', str(root), '--worker', '--rank', str(rank)],
                                     stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     env=env, start_new_session=True)
            workers.append(p)
        while any(p.poll() is None for p in workers):
            exclusive(servers + workers)
            if any(p.poll() is not None for p in servers) or any(p.poll() not in (None, 0) for p in workers):
                raise RuntimeError('Pilot evaluation v2 process failed')
            time.sleep(2)
        assert all(p.returncode == 0 for p in workers)
        from gguf_priority import report
        result = report(root, root)
        result['evaluation_version'] = profile['evaluation']
        result['qualification'] += ' Evaluation v2 only changes deterministic extraction admission; narrative replies use the same grounded reader. Original inference and extraction caches are untouched. Alias equivalence is not added to strict EM.'
        (root / 'report.json').write_text(json.dumps(result, ensure_ascii=False, indent=2))
        (root / 'COMPLETE').write_text('Re-extraction and matched report complete\n')
    except Exception as exc:
        (root / 'FAILED.txt').write_text(str(exc))
        raise
    finally:
        stop(workers)
        stop(servers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--source')
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    worker(args) if args.worker else run(args)
