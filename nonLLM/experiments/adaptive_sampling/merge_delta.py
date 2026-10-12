"""Merge disjoint helper rows into serverA, rejecting conflicting existing outputs."""
import argparse,json,sqlite3
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--delta',required=True);a=p.parse_args();root=Path(a.root)
dbs=[sqlite3.connect(root/'runs'/f'answers_S_{g}.sqlite',timeout=60) for g in range(4)];count=0
try:
 for line in open(a.delta):
  x=json.loads(line);i=int(x['key'].split(':')[0]);g=x['shard']
  if i<3860 or i%4!=g:raise RuntimeError('Partition mismatch')
  db=dbs[g];old=db.execute('SELECT payload FROM results WHERE key=?',(x['key'],)).fetchone()
  if old and old[0]!=x['payload']:raise RuntimeError(f'Conflicting existing result {x["key"]}')
  if not old:db.execute('INSERT INTO results VALUES (?,?)',(x['key'],x['payload']));count+=1
  if count%256==0:
   for d in dbs:d.commit()
 for d in dbs:d.commit()
finally:
 for d in dbs:d.close()
print(json.dumps({'merged':count}))
