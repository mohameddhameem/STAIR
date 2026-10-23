"""Verify the published corrected cohort without downloading models or running inference."""
import csv,gzip,hashlib,json,math
from pathlib import Path
import pandas as pd
from stair_common.schema import validate
ROOT=Path(__file__).resolve().parent

def main():
 folder=ROOT/'menu_ws_gguf_v2'
 df=pd.read_csv(folder/'results.csv.gz');validate(df)
 assert len(df)==29000
 assert df.groupby(['policy','seed']).size().eq(500).all()
 assert df.groupby(['policy','seed']).id.nunique().eq(500).all()
 summary=json.loads((folder/'results.json').read_text())['summaries'];assert len(summary)==58
 for row in summary:
  chosen=df[(df.policy==row['policy'])&(df.seed==row['seed'])]
  assert math.isclose(chosen.lenient.mean(),row['accuracy'],abs_tol=1e-12)
  assert math.isclose(chosen.units.mean(),row['units'],abs_tol=1e-8)
 fixed={row['policy']:row['accuracy'] for row in summary if row['policy'].startswith('gguf-fixed-')}
 assert fixed=={'gguf-fixed-qwen7-llama3':.54,'gguf-fixed-qwen7-llama70':.622,'gguf-fixed-qwen72-llama3':.658,'gguf-fixed-qwen72-llama70':.718}
 lock=json.loads((folder/'validation_lock.json').read_text());assert len(lock['selections'])==54
 assert all(row['selected']['type']=='fixed' and row['selected']['r']==1 and row['selected']['a']==1 for row in lock['selections'])
 protocol=json.loads((folder/'protocol.json').read_text())
 assert hashlib.sha256((folder/'evaluation_snapshot/lenient_match_agenttts.py').read_bytes()).hexdigest()==protocol['ruler_sha256']
 assert set(protocol['models'])=={'qwen7','qwen72','llama3','llama70'}
 assert json.loads((folder/'status.json').read_text())['phase']=='complete'
 for part,ids in protocol['splits'].items():
  frozen=(folder/'evaluation_snapshot/splits/hotpotqa'/f'{part}.txt').read_text().splitlines();assert frozen==ids
 inventory=json.loads((ROOT/'artifact_inventory.json').read_text())
 for artifact in inventory['files']:
  path=ROOT.parents[1]/artifact['published']
  assert path.is_file(),path
  assert hashlib.sha256(path.read_bytes()).hexdigest()==artifact['published_sha256'],path
 rows=list(csv.DictReader((ROOT/'all_results.csv').open()));assert len(rows)==144
 for row in rows:
  assert 0<=float(row['accuracy_pct'])<=100
  assert float(row['cost_pct_ss'])>=0
  assert (ROOT/row['source']).is_file()
 print('Verified: 29000 schema-valid test rows, 58 summaries, 54 fixed-SS selections, frozen ruler/splits, artifact hashes and144 cross-cohort index rows.')

if __name__=='__main__':main()
