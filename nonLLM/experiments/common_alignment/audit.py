"""Read-only migration audit. Does not retrain, relabel saved runs or publish results."""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

REPO=Path(__file__).resolve().parents[3]

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def readrows(path):return [json.loads(line) for line in path.open()]

def main(workspace,out):
 from stair_common import load_models
 from stair_common.splits import load_split
 from stair_common.score import lenient,RULER,RULER_SHA256
 from stair_common.cost import units
 repo=REPO
 author=subprocess.check_output(['git','show','origin/jonathan:jonathan/final/lenient_match_agenttts.py'],cwd=repo)
 assert hashlib.sha256(author).hexdigest()==sha(RULER)==RULER_SHA256
 spec=importlib.util.spec_from_file_location('historical_lenient',repo/'nonLLM/experiments/router_improvements/reference_lenient.py')
 old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
 grid=readrows(workspace/'runs/nonllm_reference_lenient_v1/base/labels/full_run_p100/router_data_lenient.jsonl')
 byid={g['_id']:g for g in grid};byidx={g['idx']:g for g in grid}
 sets={part:load_split('hotpotqa',part) for part in ['train','train_extra','val','test','nanojev_v1_seen']}
 assert len(sets['train_extra'])==6405 and len(sets['val'])==len(sets['test'])==500
 assert not set(sets['train_extra'])&set(sets['val']+sets['test'])
 assert set(sets['train_extra']+sets['val']+sets['test'])==set(byid)
 original_dev=json.loads((workspace/'tools/AgentTTS/data/HotpotQA/hotpotqa-dev.json').read_text())
 assert sets['val']==[q['_id'] for q in original_dev[:500]]
 assert sets['test']==[q['_id'] for q in original_dev[500:1000]]
 intersections={part:dict(Counter(byid[i]['split'] for i in sets[part])) for part in ['val','test','train_extra']}
 with gzip.open(workspace/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:questions=json.load(f)
 answers={who:{r['k']:r['v'] for r in readrows(workspace/f'runs/nonllm_reference_lenient_v1/base/labels/full_run_p100/qa_{who}.jsonl')} for who in ['weak','strong']}
 changed=Counter();direct_ss={};reviewed=0
 for g in grid:
  i=g['idx'];golds=questions[i]['answers'];assert questions[i]['_id']==g['_id']
  grades=[]
  for branch in ['weak','strong']:
   for who in ['weak','strong']:
    pred=answers[who][f'{i}|{branch}'];j=len(grades)
    grade=lenient(pred,golds);grades.append(grade);reviewed+=1
    if grade!=g['em_lenient'][j]:changed[['WW','WS','SW','SS'][j]]+=1
  direct_ss[g['_id']]=grades[3]
 diagnostic={}
 for name,ids in [('historical_test',[g['_id'] for g in grid if g['split']=='test']),('common_val',sets['val']),('common_test',sets['test'])]:
  diagnostic[name]={'n':len(ids),'historical_scorer_SS_correct':sum(byid[i]['em_lenient'][3] for i in ids),'common_scorer_SS_correct':sum(direct_ss[i] for i in ids),'scope':'Diagnostic on old Qwen1.5B/7B executor caches; not menu-A/B pooled results.'}
 models=load_models();menus={}
 for menu in ['A','B','C']:
  menus[menu]={stage:[{'name':name,**entry} for name,entry in models[menu][stage].items()] for stage in ['retriever','answerer','planner']}
 samples={'params_b':3,'prompt_tokens':600,'samples':40,'generated_tokens_per_sample':16}
 costs={'shared_prompt_once_units':units(3,600,40*16),'repeated_prompt_40_times_units':40*units(3,600,16),'scope':'Illustration only, excluding fusion and routing; no measured historical cost relabeling.'}
 probes=[{'answer':a,'gold':g,'historical':int(old.lenient_match(a,g)),'common':lenient(a,g)} for a,g in [('1936',['7 January 1936']),('Yes\nNo',['yes']),('more than 20',['20']),('Armenian-American',['Armenian–American'])]]
 obj={'date':'2026-10-18','main_commit':subprocess.check_output(['git','rev-parse','origin/main'],cwd=repo,text=True).strip(),'local_merge_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),'common_version':'0.2.0','policy_status':'Merged main implementation; common/README.md still labels the setup a proposal until team agreement. No Teams/Word attachment was available beyond the user-pasted discussion.','standard_scorer':{'path':'common/stair_common/lenient_match_agenttts.py','sha256':RULER_SHA256,'byte_identical_to_jonathan_branch':True,'published_74_6_claim':'Claim in upstream header; not independently reproduced in this synchronization.'},'splits':{'hotpotqa_counts':{p:len(v) for p,v in sets.items()},'common_ids_overlap_historical_partitions':intersections,'old_training_examples_in_common_test':intersections['test'].get('train',0),'nanojev_seen_test':len(set(sets['test'])&set(sets['nanojev_v1_seen'])),'nanojev_seen_val':len(set(sets['val'])&set(sets['nanojev_v1_seen'])),'source_order_assertions_passed':True},'scorer_drift':{'cached_answers_audited':reviewed,'changed_grades_per_cell':dict(changed),'changed_total':sum(changed.values()),'probes':probes},'diagnostic_only_old_executor_SS':diagnostic,'menus':menus,'cost_shared_prompt_example':{'input':samples,**costs},'historical_results_pooled_eligible':False,'reasons':['Historical split differs; some common test questions were used in router training.','Historical reconstructed lenient scorer differs from now-available author standard.','Historical Qwen1.5B/7B pair and GGUF copies do not equal the pinned pooled menus.','Historical mean token cost percentages differ from per-question AgentTTS units and shared-prompt accounting.','Router seed/checkpoint/history/output layouts need common adapters.']}
 out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(obj,indent=2))
 print(json.dumps({k:obj[k] for k in ['splits','scorer_drift','cost_shared_prompt_example']},indent=2))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=REPO.parent);p.add_argument('--out',type=Path,default=REPO/'nonLLM/experiments/common_alignment/audit_20261018.json');a=p.parse_args();main(a.workspace.resolve(),a.out)
