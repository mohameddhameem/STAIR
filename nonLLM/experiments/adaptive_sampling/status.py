"""Read-only status of resumable serverA experiment."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',default='/data1/zmcheng/cs707_sampling40');args=p.parse_args();root=Path(args.root);out={'expected_questions':7405,'stages':{}}
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
for role in ['A','B']:
 path=root/'code'/f'tasks_{role}.json'
 if path.exists():out['assigned_questions']=len(json.load(open(path)));out['assignment_role']=role
out['strong_assignment_done_shards']=sum(int((root/'runs'/f'answers_S_{g}.assigned.done').exists()) for g in range(4))
failed=root/'runs/FAILED.txt'
if failed.exists():out['failed']=failed.read_text()
print(json.dumps(out,indent=2))
