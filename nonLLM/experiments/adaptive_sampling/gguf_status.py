"""Read-only progress for the isolated serverA GGUF Instruct experiment."""
import argparse
import json
from pathlib import Path
import sqlite3

parser = argparse.ArgumentParser()
parser.add_argument('--root', required=True)
args = parser.parse_args()
root = Path(args.root)
state = root / 'state.json'
out = {'state': json.loads(state.read_text()) if state.exists() else None, 'stages': {}}
for stage, model, factor in [('retrieval', 'W', 1), ('retrieval', 'S', 1),
                             ('answers', 'W', 82), ('answers', 'S', 82), ('extraction', 'S', 160)]:
    count = 0
    for shard in range(4):
        path = root / 'runs' / f'{stage}_{model}_{shard}.sqlite'
        if path.exists():
            db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
            count += db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
            db.close()
    out['stages'][stage + '_' + model] = {'saved_requests': count, 'expected': 7405 * factor}
plan_path = root / 'strong_workflow.json'
if plan_path.exists():
    plan = json.loads(plan_path.read_text())
    active = out['state'] or {}
    reader = Path(active.get('reader_root', plan['reader_root']))
    count = 0
    for shard in range(4):
        path = reader / 'runs' / f'extraction_S_{shard}.sqlite'
        if path.exists():
            db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
            count += db.execute('SELECT COUNT(*) FROM results').fetchone()[0]
            db.close()
    out['stages']['extraction_v2'] = {'saved_requests': count, 'expected': active.get('reader_expected', plan['reader_expected']),
                                     'scope': active.get('reader_scope', 'All Strong fusions and already available Weak fusions'),
                                     'reader_root': str(reader)}
cohort_path = root / 'cohort_500'
if (cohort_path / 'config.json').exists():
    ids = json.loads((cohort_path / 'config.json').read_text())['indices']
    progress = {'selected_questions': len(ids), 'complete_Weak': 0, 'complete_Strong': 0,
                'Strong_requests': 0, 'Strong_expected_requests': len(ids) * 82,
                'extraction_v2_records': 0, 'extraction_expected': len(ids) * 160,
                'table_ready': (cohort_path / 'COMPLETE').exists(), 'path': str(cohort_path)}
    for stage, model, folder in [('answers', 'W', root), ('answers', 'S', root),
                                  ('extraction', 'S', cohort_path / 'evaluation_v2')]:
        for shard in sorted({i % 4 for i in ids}):
            path = folder / 'runs' / f'{stage}_{model}_{shard}.sqlite'
            if not path.exists():
                continue
            db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
            for i in ids:
                if i % 4 != shard:
                    continue
                n = db.execute('SELECT COUNT(*) FROM results WHERE key>=? AND key<?', (f'{i}:', f'{i};')).fetchone()[0]
                if stage == 'answers':
                    progress['complete_' + ('Weak' if model == 'W' else 'Strong')] += int(n == 82)
                    if model == 'S':
                        progress['Strong_requests'] += n
                else:
                    progress['extraction_v2_records'] += n
            db.close()
    out['cohort_500'] = progress
if (root / 'runs/FAILED.txt').exists():
    out['failure'] = (root / 'runs/FAILED.txt').read_text()
print(json.dumps(out, indent=2))
