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
 rows=list(csv.DictReader((ROOT/'all_results.csv').open()));expected_index=144
 if (ROOT/'menu_ws_grpo_v1/results.json').exists():
  gd=pd.read_csv(ROOT/'menu_ws_grpo_v1/results.csv.gz');validate(gd);gs=json.loads((ROOT/'menu_ws_grpo_v1/results.json').read_text())['summaries'];expected_index+=len(gs)
  assert len(gd)==10500 and len(gs)==21
  for r in gs:
   chosen=gd[gd.policy==r['policy']];assert len(chosen)==500 and chosen.id.nunique()==500
   assert math.isclose(chosen.lenient.mean(),r['accuracy'],abs_tol=1e-12)
   assert math.isclose(chosen.units.mean(),r['units'],abs_tol=1e-8)
  gr=gd[gd.policy=='grpo-v2-grpo-max_accuracy-learned_only'];assert gr.lenient.sum()==359 and gr.ret_choice.eq(1).all() and gr.qa_choice.eq(1).all()
  ex=gd[gd.policy=='grpo-v2-exact-max_accuracy-learned_only'];assert ex.lenient.sum()==357 and ex.ret_choice.eq(0).sum()==96
  assert json.loads((ROOT/'menu_ws_grpo_v1/status.json').read_text())['phase']=='complete'
 if (ROOT/'menu_ws_alternatives_v1/results.json').exists():
  ad=pd.read_csv(ROOT/'menu_ws_alternatives_v1/results.csv.gz');validate(ad);asummary=json.loads((ROOT/'menu_ws_alternatives_v1/results.json').read_text())['summaries'];expected_index+=len(asummary)
  assert len(ad)==38000 and len(asummary)==76
  for r in asummary:
   chosen=ad[ad.policy==r['policy']];assert len(chosen)==500 and chosen.id.nunique()==500
   assert math.isclose(chosen.lenient.mean(),r['accuracy'],abs_tol=1e-12) and math.isclose(chosen.units.mean(),r['units'],abs_tol=1e-8)
  assert ad[ad.policy=='alternative-control-qwen72-llama70'].lenient.sum()==359
  assert ad[ad.policy=='alternative-outcome_trees-max_accuracy-learned_only'].lenient.sum()==362
  assert ad[ad.policy=='alternative-expected_baseline-max_accuracy-learned_only'].lenient.sum()==360
  assert ad[ad.policy=='alternative-exact_augmented-max_accuracy-learned_only'].lenient.sum()==359
  assert ad[ad.policy=='alternative-global-max_accuracy-learned_only'].lenient.sum()==356
  assert json.loads((ROOT/'menu_ws_alternatives_v1/status.json').read_text())['phase']=='complete'
 assert len(rows)==expected_index
 for row in rows:
  assert 0<=float(row['accuracy_pct'])<=100
  assert float(row['cost_pct_ss'])>=0
  assert (ROOT/row['source']).is_file()
 print(f'Verified classical, GRPO and available alternative test records, summaries, route choices, frozen ruler/splits, artifact hashes and {expected_index} cross-cohort index rows.')

if __name__=='__main__':main()
