"""Publish compact experiment artifacts from the sibling workspace, without weights/caches."""
import csv,gzip,hashlib,io,json,re,shutil,subprocess
from pathlib import Path
from datetime import datetime,timezone,timedelta
REPO=Path(__file__).resolve().parents[2]
WORK=REPO.parent
DEST=Path(__file__).resolve().parent
COMMON_COMMIT='c02b09367718249af962140e605a91ddd392baa0'
SUMMARY_NAMES={'oof_candidates.csv','risk_summary.csv','fold_audits.json','proposal_lock.json','final_meta_crossfit.json','method_summary.csv','training_summary.json','warm_start_verification.json','summary.json','generation_summary.json','token_costs.json','config.json','REPORT.md','METHODS.md','COMPARISON.md','README.md','results.json','comparison.json','verification.json','protocol.json','manifest.json','status.json','source_hashes.json','all_families_test.json','theta_lock.json','validation_lock.json','fit_records.json','training_progress.json','migration.json','concurrency_runtime.json','port_fix.json','RESULTS_INVALIDATED.json','INVALID_RESULTS.md','token_cost_audit.json','frontier_gains.json'}
EXCLUDE={'cache','samples','training','routers','checkpoints','data','code','__pycache__','more_samples'}

def scrub(text):
 text=text.replace(str(WORK),'<WORKSPACE>').replace(str(Path.home()),'<HOME>')
 text=re.sub(r'/ssd\d+/[^/\s\"\']+/cs701(?:/\.env)?','<PYTHON_ENV>',text)
 text=re.sub(r'/home/[^/\s\"\']+','<HOME>',text)
 return text

def copy_text(source,target):
 target.parent.mkdir(parents=True,exist_ok=True)
 target.write_text(scrub(source.read_text()))
 return {'source':str(source.relative_to(WORK)),'published':str(target.relative_to(REPO)),'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'published_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}

def main():
 inventory=[];files=[];archive=DEST/'archive_20261023'
 for folder in sorted((WORK/'runs').iterdir()):
  if not folder.is_dir():continue
  exported=[]
  for source in sorted(folder.rglob('*')):
   if not source.is_file() or source.name not in SUMMARY_NAMES:continue
   rel=source.relative_to(folder)
   if set(rel.parts[:-1])&EXCLUDE or source.stat().st_size>2_000_000:continue
   target=archive/folder.name/rel;files.append(copy_text(source,target));exported.append(str(target.relative_to(DEST)))
  if exported:
   state={}
   try:state=json.loads((folder/'status.json').read_text())
   except (FileNotFoundError,json.JSONDecodeError):pass
   inventory.append({'run':folder.name,'source_phase':state.get('phase',state.get('status','not recorded')),'published_artifacts':exported,'classification':'cohort-specific, not pooled; see report and results/README.md'})
 # Current corrected cohort gets a stable, prominent path.
 current=WORK/'runs/nonllm_common_ws_gguf_v2';primary=DEST/'menu_ws_gguf_v2'
 for source in current.iterdir():
  if source.is_file() and source.name in SUMMARY_NAMES:files.append(copy_text(source,primary/source.name))
 with (current/'results.csv').open('rb') as raw,(primary/'results.csv.gz').open('wb') as stream:
  with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:shutil.copyfileobj(raw,compressed)
 for name in ['nonllm_ws_template_audit_v1','nonllm_ws_concurrency_audit_v1','nonllm_ws_concurrency_audit_v2']:
  source=WORK/'runs'/name
  for item in source.iterdir():
   if item.name in ['REPORT.md','summary.json']:files.append(copy_text(item,DEST/'audits'/name/item.name))
 assets=DEST/'historical_slides'
 for name in ['method_1_direct.png','method_2_cascade.png','nonllm_router_two_methods.pdf','slide_data.json']:
  source=WORK/'reports/router_method_slides'/name;assets.mkdir(exist_ok=True)
  if source.suffix=='.json':files.append(copy_text(source,assets/name))
  else:shutil.copy2(source,assets/name)
 (assets/'README.md').write_text('# Historical method slides\n\nThese slides describe the older reconstructed-lenient cohort (SS test52.94%), including its GRPO experiments. They do not describe the corrected four-model500-question cohort or the new common0.3 SLR golden set. See ../README.md for current results.\n')
 # Freeze the original shared implementation rather than whichever main is checked out later.
 frozen=primary/'evaluation_snapshot';frozen.mkdir(exist_ok=True)
 names=['score.py','lenient_match_agenttts.py','cost.py','schema.py','splits.py']+[f'splits/hotpotqa/{name}.txt' for name in ['val','test','train_extra']]
 for name in names:
  source=f'common/stair_common/{name}'
  raw=subprocess.check_output(['git','-C',str(REPO),'show',f'{COMMON_COMMIT}:{source}'])
  target=frozen/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_text(scrub(raw.decode()))
  files.append({'source':f'git:{COMMON_COMMIT}:{source}','published':str(target.relative_to(REPO)),'source_sha256':hashlib.sha256(raw).hexdigest(),'published_sha256':hashlib.sha256(target.read_bytes()).hexdigest()})
 (frozen/'SOURCE.md').write_text(f'# Evaluation snapshot\n\nOriginal common implementation at `{COMMON_COMMIT}`. The finished cohort uses common0.2 test500 and token-derived units. Later common0.3 slr-* golden sets and Table5 prices must not be retroactively substituted. This is an audit snapshot, not a separately installed package.\n')
 # Include metric curves and test-policy arrays, never model/optimizer checkpoints.
 for name in ['nonllm_common_ws_alternatives_v1','nonllm_common_ws_grpo_v1','nonllm_lenient_longer_v1','nonllm_grpo_extended_v1','nonllm_reference_lenient_v1','nonllm_architecture_v1','nonllm_architecture_combined_v1']:
  folder=WORK/'runs'/name
  for source in folder.glob('training/*/curve.jsonl'):
   target=archive/name/source.relative_to(folder);target=target.with_suffix('.jsonl.gz');target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('wb') as stream:
    with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:compressed.write(scrub(source.read_text()).encode())
  for source in folder.glob('*.npz'):
   if 'test' in source.name or name in ['nonllm_architecture_v1','nonllm_architecture_combined_v1']:
    target=archive/name/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  for source in folder.glob('*.png'):
   target=archive/name/source.name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
 manifest=json.loads((WORK/'models/agenttts_gguf/manifest.json').read_text())
 manifest['models']=[m for m in manifest['models'] if m['role']!='medium'];manifest['subset']='Four-model weak/strong menu; GGUF exploratory'
 (primary/'model_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 # A compact cross-cohort index; numbers are never pooled across cohorts.
 rows=[]
 current_data=json.loads((current/'results.json').read_text());ss=next(x for x in current_data['summaries'] if x['policy']=='gguf-fixed-qwen72-llama70')['units']
 for r in current_data['summaries']:rows.append({'cohort':'four-model-GGUF-common02-test500','method':r['policy'],'seed':r['seed'],'n':500,'accuracy_pct':100*r['accuracy'],'cost_pct_ss':100*r['units']/ss,'units':r['units'],'status':'corrected; descriptive; not common03 golden','source':'menu_ws_gguf_v2/results.json'})
 old=json.loads((WORK/'runs/nonllm_reference_lenient_v1/results.json').read_text())
 rows.append({'cohort':'historical-reconstructed-lenient-test1481','method':'SS','n':1481,'accuracy_pct':100*old['SS']['test']['accuracy'],'cost_pct_ss':100,'source':'archive_20261023/nonllm_reference_lenient_v1/results.json'})
 for r in old['results']:rows.append({'cohort':'historical-reconstructed-lenient-test1481','method':r['method'],'n':1481,'accuracy_pct':100*r['accuracy'],'cost_pct_ss':r['cost_pct_SS'],'status':'historical scorer; not author-script common02/03','source':'archive_20261023/nonllm_reference_lenient_v1/results.json'})
 longer=json.loads((WORK/'runs/nonllm_lenient_longer_v1/results.json').read_text())
 for r in longer['results']:rows.append({'cohort':'historical-GRPO-300epoch-test1481','method':r['name'],'n':r['test']['n'],'accuracy_pct':100*r['test']['lenient_accuracy'],'cost_pct_ss':r['test']['cost_pct_of_direct_SS'],'status':'historical scorer; val-selected checkpoint; descriptive','source':'archive_20261023/nonllm_lenient_longer_v1/results.json'})
 # New corrected-cache GRPO is a separate cohort, not a rewrite of historical results.
 grpo_run=WORK/'runs/nonllm_common_ws_grpo_v1'
 if (grpo_run/'results.json').exists():
  published=DEST/'menu_ws_grpo_v1'
  for source in grpo_run.iterdir():
   if source.is_file() and source.name in SUMMARY_NAMES:files.append(copy_text(source,published/source.name))
  with (grpo_run/'results.csv').open('rb') as raw,(published/'results.csv.gz').open('wb') as stream:
   with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:shutil.copyfileobj(raw,compressed)
  if (grpo_run/'learning_curves.png').exists():shutil.copy2(grpo_run/'learning_curves.png',published/'learning_curves.png')
  data=json.loads((grpo_run/'results.json').read_text())
  for r in data['summaries']:rows.append({'cohort':'four-model-GGUF-GRPO-common02-test500','method':r['policy'],'seed':0,'n':500,'accuracy_pct':100*r['accuracy'],'cost_pct_ss':100*r['units']/data['ss']['units'],'units':r['units'],'status':'three-seed ensemble for learned policies; descriptive; not common03 golden','source':'menu_ws_grpo_v1/results.json'})
 alternative_run=WORK/'runs/nonllm_common_ws_alternatives_v1'
 if (alternative_run/'results.json').exists():
  published=DEST/'menu_ws_alternatives_v1'
  for source in alternative_run.iterdir():
   if source.is_file() and source.name in SUMMARY_NAMES:files.append(copy_text(source,published/source.name))
  with (alternative_run/'results.csv').open('rb') as raw,(published/'results.csv.gz').open('wb') as stream:
   with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:shutil.copyfileobj(raw,compressed)
  for name in ['accuracy_cost.png','validation_candidates.csv.gz']:
   if (alternative_run/name).exists():shutil.copy2(alternative_run/name,published/name)
  data=json.loads((alternative_run/'results.json').read_text())
  for r in data['summaries']:rows.append({'cohort':'four-model-GGUF-alternatives-common02-test500','method':r['policy'],'seed':0,'n':500,'accuracy_pct':100*r['accuracy'],'cost_pct_ss':100*r['units']/data['ss']['units'],'units':r['units'],'status':'per-family and global validation choices separate; exploratory repeated test','source':'menu_ws_alternatives_v1/results.json'})
 next_run=WORK/'runs/nonllm_common_ws_next_v1'
 if (next_run/'results.json').exists():
  published=DEST/'menu_ws_next_v1'
  for source in next_run.iterdir():
   if source.is_file() and source.name in SUMMARY_NAMES:files.append(copy_text(source,published/source.name))
  with (next_run/'results.csv').open('rb') as raw,(published/'results.csv.gz').open('wb') as stream:
   with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:shutil.copyfileobj(raw,compressed)
  if (next_run/'accuracy_cost.png').exists():shutil.copy2(next_run/'accuracy_cost.png',published/'accuracy_cost.png')
  data=json.loads((next_run/'results.json').read_text())
  for r in data['summaries']:rows.append({'cohort':'four-model-GGUF-nestedOOF-risk-cascade-test500','method':r['policy'],'seed':0,'n':500,'accuracy_pct':100*r['accuracy'],'cost_pct_ss':100*r['units']/data['ss']['units'],'units':r['units'],'status':'fit5125/calibration1280; OOF and val choices separate; exploratory risk gates','source':'menu_ws_next_v1/results.json'})
 with (DEST/'all_results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=['cohort','method','seed','n','accuracy_pct','cost_pct_ss','units','status','source'],lineterminator='\n');writer.writeheader();writer.writerows(rows)
 (DEST/'artifact_inventory.json').write_text(json.dumps({'exported_on':datetime.now(timezone(timedelta(hours=8))).date().isoformat(),'local_data_and_weights_included':False,'runs':inventory,'files':files},indent=2)+'\n')
 print(f'Exported {len(files)} text artifacts from {len(inventory)} run directories; {len(rows)} indexed result rows.')

if __name__=='__main__':main()
