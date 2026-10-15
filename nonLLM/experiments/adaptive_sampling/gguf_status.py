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
if (root / 'runs/FAILED.txt').exists():
    out['failure'] = (root / 'runs/FAILED.txt').read_text()
print(json.dumps(out, indent=2))
