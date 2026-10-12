"""Export append-only helper results from a consistent SQLite read snapshot."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--after',required=True);a=p.parse_args();root=Path(a.root);after=json.loads(a.after);last=list(after);count=0
with open(root/'runs/transfer_delta.jsonl','w') as f:
 for g in range(4):
  path=root/'runs'/f'answers_S_{g}.sqlite'
  if not path.exists():continue
  db=sqlite3.connect(f'file:{path}?mode=ro',uri=True)
  if not db.execute("SELECT name FROM sqlite_master WHERE name='results'").fetchone():db.close();continue
  for rid,key,payload in db.execute('SELECT rowid,key,payload FROM results WHERE rowid>? ORDER BY rowid LIMIT 20000',(after[g],)):
   if int(key.split(':')[0])<3860:raise RuntimeError('Helper emitted unassigned question')
   f.write(json.dumps({'shard':g,'rowid':rid,'key':key,'payload':payload})+'\n');last[g]=rid;count+=1
  db.close()
meta={'after':last,'rows':count,'helper_done':all((root/'runs'/f'answers_S_{g}.assigned.done').exists() for g in range(4))}
(root/'runs/transfer_meta.json').write_text(json.dumps(meta));print(json.dumps(meta))
