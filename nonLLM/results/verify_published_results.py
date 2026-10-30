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
 if (ROOT/'menu_ws_next_v1/results.json').exists():
  nd=pd.read_csv(ROOT/'menu_ws_next_v1/results.csv.gz');validate(nd);ns=json.loads((ROOT/'menu_ws_next_v1/results.json').read_text())['summaries'];expected_index+=len(ns)
  assert len(nd)==57000 and len(ns)==114
  for r in ns:
   chosen=nd[nd.policy==r['policy']];assert len(chosen)==500 and chosen.id.nunique()==500
   assert math.isclose(chosen.lenient.mean(),r['accuracy'],abs_tol=1e-12) and math.isclose(chosen.units.mean(),r['units'],abs_tol=1e-8)
  controls={(r,a):nd[nd.policy==f'next-control-{rt}-{at}'].set_index('id') for r,rt in enumerate(['qwen7','qwen72']) for a,at in enumerate(['llama3','llama70'])}
  lookup={qid:([float(controls[(r,0)].loc[qid,'units_retrieve']) for r in range(2)],[[float(controls[(r,a)].loc[qid,'units_answer']) for a in range(2)] for r in range(2)]) for qid in controls[(0,0)].index}
  for row in nd.itertuples():
   ret,qa=lookup[row.id];r,a=row.ret_choice,row.qa_choice;cascade=row.execution=='weak_first_cascade'
   ru=ret[0]+(ret[1] if r else 0) if cascade else ret[r]
   au=qa[r][0]+(qa[r][1] if a else 0) if cascade else qa[r][a]
   assert math.isclose(row.units_retrieve,ru,abs_tol=1e-8) and math.isclose(row.units_answer,au,abs_tol=1e-8)
  folds=json.loads((ROOT/'menu_ws_next_v1/fold_audits.json').read_text())['outer'];all_held=[]
  for fold in folds:
   train,held=set(fold['train']),set(fold['heldout']);assert not train&held and train|held==set(range(5125));all_held+=fold['heldout']
   for inner in fold['inner_folds']:assert not held&(set(inner['train'])|set(inner['heldout']))
  assert sorted(all_held)==list(range(5125))
  assert nd[nd.policy=='next-control-qwen72-llama70'].lenient.sum()==359
  assert nd[nd.policy=='next-stack_joint-max_accuracy-learned_only'].lenient.sum()==360
  audit=json.loads((ROOT/'menu_ws_next_v1/verification.json').read_text());assert audit['strict_nested_group_isolation'] and audit['cascade_paid_call_cost_parity'] and audit['all_calibration_criteria_met']
  assert not audit['risk_certified'] and len(audit['risk_test_above_nominal'])>0
  assert audit['cascade_all_strong_cost_pct_ss']>100
  assert json.loads((ROOT/'menu_ws_next_v1/status.json').read_text())['phase']=='complete'
 from verify_recent_results import verify_recent
 expected_index+=verify_recent(ROOT)
 assert len(rows)==expected_index
 for row in rows:
  assert 0<=float(row['accuracy_pct'])<=100
  assert float(row['cost_pct_ss'])>=0
  assert (ROOT/row['source']).is_file()
 print(f'Verified all published classical, GRPO, alternative and next-study records, summaries, route choices, frozen ruler/splits, artifact hashes and {expected_index} cross-cohort index rows.')

if __name__=='__main__':main()
