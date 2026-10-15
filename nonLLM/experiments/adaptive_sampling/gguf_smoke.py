"""Exercise the actual resumable worker on one predetermined training question."""
import argparse
import json
from pathlib import Path
import sqlite3
import subprocess
from gguf_runtime import PYTHON


def main(root):
    root = Path(root)
    train = json.loads((root / 'assets/splits.json').read_text())['splits']['train']
    assert train[0] == 0
    for stage, model, port in [('retrieval', 'W', 30280), ('retrieval', 'S', 30281),
                               ('answers', 'W', 30280), ('answers', 'S', 30281), ('extraction', 'S', 30281)]:
        print('WORKER SMOKE', stage, model, flush=True)
        subprocess.run([PYTHON, str(root / 'code/gguf_worker.py'), '--root', str(root),
                        '--stage', stage, '--model', model, '--shard', '0', '--port', str(port),
                        '--limit', '1'], check=True)
    checks = {}
    for stage, model, expected in [('retrieval', 'W', 1), ('retrieval', 'S', 1),
                                   ('answers', 'W', 82), ('answers', 'S', 82), ('extraction', 'S', 160)]:
        db = sqlite3.connect(root / 'runs' / f'{stage}_{model}_0.sqlite')
        count = db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
        assert count == expected, (stage, model, count)
        checks[stage + '_' + model] = {'rows': count, 'expected': expected}
        db.close()
    db = sqlite3.connect(root / 'runs/extraction_S_0.sqlite')
    ss = [json.loads(payload)['extraction']['answer'] for _, payload in
          db.execute("SELECT key,payload FROM results WHERE key LIKE '0:fuse:SS:%'")]
    db.close()
    # Known first training question asks whether two Americans share nationality.
    assert 'yes' in ss, 'No correct asserted SS answer in 40 budgets; inspect serving'
    (root / 'gates/worker_smoke.json').write_text(json.dumps({
        'passed': True, 'training_index': 0, 'checks': checks,
        'ss_yes_budgets': ss.count('yes'), 'reusable_as_main_cache': True,
        'note': 'Uses frozen main settings; each QA batch generated once, all 40 prefixes fused.'}, indent=2))
    print('FULL WORKER SMOKE PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    main(parser.parse_args().root)
