"""Freeze the common menu-A non-LLM protocol and question manifests; no inference."""
import gzip,hashlib,json
from pathlib import Path
from stair_common import load_models
from stair_common.splits import load_split
from stair_common.score import RULER_SHA256

REPO=Path(__file__).resolve().parents[3]
WORKSPACE=REPO.parent
OUT=WORKSPACE/'runs/nonllm_common_menuA_v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 OUT.mkdir(exist_ok=True)
 menus=load_models();menu=menus['A']
 data=WORKSPACE/'extracted/Dataset/AgentTTS-HotpotQA.json.gz'
 with gzip.open(data,'rt') as f:source=json.load(f)
 byid={r['_id']:r for r in source};parts={s:load_split('hotpotqa',s) for s in ['train_extra','val','test']}
 assert len(byid)==7405 and len(set(sum(parts.values(),[])))==7405
 (OUT/'question_ids').mkdir(exist_ok=True)
 for part,ids in parts.items():
  assert set(ids)<=set(byid)
  (OUT/'question_ids'/f'{part}.txt').write_text('\n'.join(ids)+'\n')
 models={s:menu[s] for s in ['retriever','answerer']}
 settings={'id':'nonllm-common-hotpotqa-menuA-k1-direct-v1','status':'prepared_not_trained','menu':'A','dataset':'hotpotqa','question_source':{'file':'extracted/Dataset/AgentTTS-HotpotQA.json.gz','sha256':sha(data),'pool_status':'Existing supplied candidate pool/order frozen for a proposed run; team-common currently freezes IDs, not this candidate pool or prompt contract. Match the comparison partner before claiming a router-only comparison.','no_gold_features':True},'splits':{'train_part':'train_extra','train':6405,'val':500,'test':500,'seed':20260729,'id_hashes':{s:sha(OUT/'question_ids'/f'{s}.txt') for s in parts}},'models':models,'executors':{'fine_tune':False,'backend':'vllm','dtype':'From common models.toml: bf16, AWQ only70B/72B','sampling_count':1,'fuse':False,'proposed_sampling':{'temperature':0.0,'max_new_tokens':512,'top_p':1.0},'sampling_status':'Proposed deterministic K1 router-control protocol, not an upstream common mandate or original AgentTTS sampling recipe. Must match LLM-router comparator/cache.','evidence_policy':'Use only actual selected retrieval output; never gold evidence. Pool cap, passage char limit and context budget must be frozen with the shared generation/cache contract; no silent truncation.'},'router':{'architecture':'Direct two-stage, three-way model selection at each stage','retrieval_inputs':['question','candidate pool'], 'answering_inputs':['question','selected retrieval evidence','retrieval model identity'],'forbidden_inputs':['gold answers','gold supporting annotations','unexecuted weak/strong answers','per-question observed correctness at deployment'],'families':['majority','logistic_regression','linear_svm','multinomial_nb','random_forest','extra_trees'],'features':'Training-only TF-IDF vocabulary and numerical text features; trees use appropriate numeric features. QA training uses all3 branches grouped by question.','labels':'Proposed: cheapest correct QA model per branch, cheapest retrieval+QA success branch for retrieval; mask all-incorrect groups rather than invent a correct target. Train only on train_extra outcomes.','seeds':[0,1,2],'selection':'Training CV within question groups for fit settings; validation for operating points; freeze before test. No artificial epoch count for non-iterative classifiers. Iterative additions must log common train/val loss and keep best val epoch.','grpo':'Separate later extension, not part of this supervised K1 baseline.'},'generation_plan':{'retrieval_calls_per_question':3,'qa_calls_per_question':9,'total_offline_calls':7405*12,'n_routes':9,'note':'Full offline9-cell grid supports labels, fixed-route controls and exact replay. Deployment charges only executed calls, not all offline counterfactual cells. Must re-generate with common pinned executors; old Qwen1.5B/7B labels are not substitutes.'},'controls':['all9 fixed retrieval-answering pairs','strongest: Qwen72B-AWQ + Llama70B-AWQ, K1/no fusion','accuracy-cost frontier / cheapest validation >=95% of current strongest validation accuracy'],'scoring':{'import':'stair_common.score.score','ruler_sha256':RULER_SHA256,'headline':'lenient','secondary':['em','f1']},'cost':{'import':'stair_common.cost.units','basis':'Actual per-question prompt and generated tokens, nominal params from common menu; include every executed model call. Shared prompt once for sampling extensions.','report':['absolute AgentTTS units','relative cost versus strongest run on the same questions','ms_router'],'router_overhead':'Time feature extraction and decision; no fictional LLM size for a classical router. FLOP proxy excludes CPU router work, which must be disclosed.'},'outputs':{'results':'stair_common.schema.COLUMNS, schema.validate','loss_history':'stair_common.training.HISTORY_COLUMNS for iterative models','test_status':'Historical test content previously examined; new routers must not fit it. This project cannot recover a never-observed confirmation set.'},'comparison':{'pure_router':'Share exact executor caches, prompts, candidate pools, decoding, copies, split and cost/score scripts with LLM-router comparator.','full_system':'If generation/retrieval differs, label as whole-system comparison, not an isolated router comparison.','other_track_results':'No aligned external results imported yet.'}}
 (OUT/'protocol.json').write_text(json.dumps(settings,indent=2))
 print(json.dumps({'status':settings['status'],'split_counts':{s:len(v) for s,v in parts.items()},'menu':'A','routes':9,'offline_calls':settings['generation_plan']['total_offline_calls'],'sampling':'proposed K1/greedy/no fusion; requires matched comparison contract'},indent=2))

if __name__=='__main__':main()
