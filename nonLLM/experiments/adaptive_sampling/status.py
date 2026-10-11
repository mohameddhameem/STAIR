"""Read-only status of resumable serverA experiment."""
import json,sqlite3
from pathlib import Path
root=Path('/data1/zmcheng/cs707_sampling40');out={'expected_questions':7405,'stages':{}}
for stage in ['retrieval','answers']:
 for m in ['W','S']:
  count=0;completed_shards=0
  for g in range(4):
   path=root/'runs'/f'{stage}_{m}_{g}.sqlite'
   if path.exists():
    db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
    count+=db.execute('SELECT COUNT(*) FROM results').fetchone()[0];db.close()
   completed_shards+=int((root/'runs'/f'{stage}_{m}_{g}.done').exists())
  out['stages'][f'{stage}_{m}']={'saved_requests':count,'expected_requests':7405*(1 if stage=='retrieval' else 82),'completed_shards':completed_shards}
failed=root/'runs/FAILED.txt'
if failed.exists():out['failed']=failed.read_text()
print(json.dumps(out,indent=2))
