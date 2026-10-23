"""Four-GPU, six-GGUF menu-A grid; isolated from pooled bf16/AWQ experiments."""
import argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
import gzip,hashlib,json,os,re,shutil,socket,subprocess,sys,time,threading
from pathlib import Path
import requests

REPO=Path(__file__).resolve().parents[3];WORK=REPO.parent
MODELS=[('qwen7','retrieval',0,[0]),('qwen32','retrieval',1,[1]),('qwen72','retrieval',2,[2,3]),
        ('llama3','answering',0,[0]),('llama8','answering',1,[1]),('llama70','answering',2,[2,3])]
BINARY=WORK/'tools/llama.cpp/build/bin/llama-server'
STOP=threading.Event()

def join_futures(futures):
 for future in as_completed(futures):
  try:future.result()
  except Exception:
   STOP.set();raise

def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(obj,indent=2));tmp.replace(path)
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path):
 if not path.exists():return {}
 result={}
 raw=path.read_text()
 lines=raw.splitlines()
 if raw and not raw.endswith('\n'):lines=lines[:-1] # concurrent/incomplete final write is not a committed record
 for line in lines:
  r=json.loads(line);assert r['key'] not in result,(path,r['key']);result[r['key']]=r
 return result

def model_cache(out,tag):
 records={}
 paths=[out/'cache'/f'{tag}.jsonl',*sorted((out/'cache').glob(f'{tag}.replica*.jsonl'))]
 for path in paths:
  for key,r in load(path).items():
   assert key not in records,('Duplicate model cache key',tag,key)
   records[key]=r
 return records

def prepare(out):
 from stair_common.splits import load_split
 from stair_common.score import RULER_SHA256
 manifest=json.loads((WORK/'models/agenttts_gguf/manifest.json').read_text())
 verification=json.loads((WORK/'models/agenttts_gguf/verification.json').read_text())
 assert verification['all_sha256_match'] and verification['models']==6
 mapping={}
 for tag,stage,size,gpus in MODELS:
  match=next(m for m in manifest['models'] if m['stage']==stage and m['role']==['weak','medium','strong'][size])
  entry={**match,'tag':tag,'index':size,'gpus':gpus,'params_b':{'qwen7':7,'qwen32':32,'qwen72':72,'llama3':3,'llama8':8,'llama70':70}[tag]}
  entry['path']=str(WORK/'models/agenttts_gguf'/match['repo'].split('/')[-1]/match['files'][0]['filename'])
  for f in match['files']:assert (Path(entry['path']).parent/f['filename']).stat().st_size==f['size']
  mapping[tag]=entry
 datafile=WORK/'extracted/Dataset/AgentTTS-HotpotQA.json.gz'
 with gzip.open(datafile,'rt') as f:data=json.load(f)
 ids={s:load_split('hotpotqa',s) for s in ['train_extra','val','test']}
 assert set(sum(ids.values(),[]))=={q['_id'] for q in data}
 settings={'run':'nonllm-menuA-GGUF-local-v1','comparison_scope':'GGUF exploratory; NOT eligible for pooled bf16/AWQ headline tables without team adopting the identical GGUF copies','models':mapping,'data_sha256':digest(datafile),'splits':ids,'ruler_sha256':RULER_SHA256,'generation':{'K':1,'temperature':0.,'top_p':1.,'repeat_penalty':1.,'max_tokens':512,'context':16384,'passage_cap_chars':600,'candidate_count':100,'pool_order':'supplied; no new gold insertion','fuse':False,'kv_k':'q8_0','kv_v':'q8_0','backend':'llama.cpp','backend_version':subprocess.check_output([str(BINARY),'--version'],text=True,stderr=subprocess.STDOUT).strip()},'router':'direct two-stage, three choices per stage; no weak output before selection','sampling_seed':0,'router_seeds':[0,1,2],'execution_schedule':'Model-by-model on up to4 free GPUs: up to4 single-GPU replicas for7B/32B/3B/8B; up to2 two-GPU replicas for72B/70B; never occupy external work','label_policy':'Cheapest correct answerer per retrieval branch and cheapest successful route for retrieval; all-incorrect labels masked','control':'all9 fixed routes and strongest Qwen72+Llama70','shared_executor_contract':'This run freezes its own prompts/pools/decoding; pure router comparisons require matching these caches, not merely matching common IDs.'}
 out.mkdir(parents=True,exist_ok=True)
 if (out/'protocol.json').exists():assert json.loads((out/'protocol.json').read_text())==settings,'Configuration changed; use a fresh run'
 else:write(out/'protocol.json',settings)
 (out/'code').mkdir(exist_ok=True)
 for name in ['run.py','train.py']:
  p=Path(__file__).with_name(name)
  if p.exists():
   dst=out/'code'/name
   if dst.exists():assert digest(dst)==digest(p),'Code changed; create a new run or explicitly migrate'
   else:shutil.copy2(p,dst)
 return settings,{q['_id']:q for q in data}

def pool(q):return [str(p)[:600] for p in q['ctxs_candidate'][:100]]
def ret_messages(q):
 refs='\n'.join(f'[{i+1}] {p}' for i,p in enumerate(pool(q)))
 user=('You have been provided with a query and a set of references. Your task is to select references that comprehensively cover the information needed to answer the query. '
       "Your response should include the original query and a list of selected references in the following format: 'Query: [query]\\nSelected references: [reference number] [reference details]'. "
       f"Here is the query and references:\nQuery: {q['question']}\nReferences:\n{refs}.")
 return [{'role':'system','content':'You are a helpful assistant that return the beneficial references to the query.'},{'role':'user','content':user}]
def parse_ids(text,n):
 # No selection fallback: retain an empty selection as an actual execution outcome.
 body=re.split(r'selected\s+references\s*:',text,flags=re.I)[-1]
 return list(dict.fromkeys(int(x)-1 for x in re.findall(r'\[(\d+)\]',body) if 1<=int(x)<=n))
def evidence(q,rec):return '\n'.join(f'[{j+1}] {pool(q)[i]}' for j,i in enumerate(rec['selected']))
def qa_messages(q,rec):
 refs=evidence(q,rec)
 user=('You will be provided with a set of references related to the query. Your task is to answer the query using the provided references. '
       'If the references are insufficient, supplement your response with your own knowledge. '
       f"Here is the query and references:\nQuery: {q['question']}\nReferences:\n{refs}.")
 return [{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':user}]

def worker(out,tag,pilot=False,replica=0,replicas=1,gpus=None):
 settings=json.loads((out/'protocol.json').read_text());m=dict(settings['models'][tag])
 if gpus is not None:m['gpus']=gpus
 with gzip.open(WORK/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:data={q['_id']:q for q in json.load(f)}
 ids=settings['splits']['train_extra'][:2] if pilot else sum(settings['splits'].values(),[])
 stage=m['stage'];path=out/'cache'/(f'{tag}.jsonl' if pilot else f'{tag}.replica{replica}.jsonl');records=model_cache(out,tag)
 retrieval={r:model_cache(out,r) for r in ['qwen7','qwen32','qwen72']} if stage=='answering' else None
 jobs=[(i,r) for i in ids for r in (['qwen7','qwen32','qwen72'] if retrieval is not None else [None])]
 jobs=jobs[replica::replicas]
 jobs=[(i,r) for i,r in jobs if (i if r is None else i+'|'+r) not in records]
 if not jobs:return
 port=19300+list(settings['models']).index(tag)*10+replica
 args=[str(BINARY),'-m',m['path'],'--host','127.0.0.1','--port',str(port),'-ngl','999','--fit','off','--ctx-size','16384','--parallel','1','--flash-attn','on','--cache-type-k','q8_0','--cache-type-v','q8_0','--batch-size','256','--ubatch-size','64','--threads','8','--threads-batch','8','--no-context-shift','--metrics']
 if len(m['gpus'])==2:args+=['--tensor-split','1,1','--split-mode','layer']
 used=subprocess.check_output(['nvidia-smi','--id',','.join(map(str,m['gpus'])),'--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 if any(int(x.strip())>=1024 for x in used.splitlines()):raise RuntimeError(f'GPU allocation for {tag} became occupied; refusing overlap')
 env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(map(str,m['gpus'])),OMP_NUM_THREADS='8')
 (out/'logs').mkdir(exist_ok=True);log=(out/'logs'/f'{tag}_replica{replica}_{time.time_ns()}.log').open('w')
 probe=socket.socket()
 try:probe.bind(('127.0.0.1',port))
 finally:probe.close()
 server=subprocess.Popen(args,env=env,stdout=log,stderr=subprocess.STDOUT)
 write(out/'workers'/f'{tag}_replica{replica}.json',{'phase':'loading','model':tag,'gpus':m['gpus'],'pid':server.pid,'command':args,'pilot':pilot})
 try:
  deadline=time.monotonic()+600
  while True:
   if server.poll() is not None:raise RuntimeError(f'{tag} server exited; inspect logs')
   try:
    if requests.get(f'http://127.0.0.1:{port}/health',timeout=2).status_code==200:break
   except requests.RequestException:pass
   if time.monotonic()>deadline:raise TimeoutError(f'{tag} loading timed out')
   time.sleep(2)
  path.parent.mkdir(exist_ok=True)
  if path.exists():
   raw=path.read_bytes()
   if raw and not raw.endswith(b'\n'):
    cut=raw.rfind(b'\n')+1
    path.with_suffix(f'.interrupted-tail-{time.time_ns()}.bin').write_bytes(raw[cut:])
    with path.open('r+b') as repair:repair.truncate(cut)
  with path.open('a') as stream:
   for j,(qid,branch) in enumerate(jobs):
    if STOP.is_set() or (out/'STOP').exists():raise RuntimeError('Run stopped; committed cache records retained')
    q=data[qid];key=qid if branch is None else qid+'|'+branch
    messages=ret_messages(q) if branch is None else qa_messages(q,retrieval[branch][qid])
    # llama.cpp renders the actual chat template; reject oversize input before generation.
    template=requests.post(f'http://127.0.0.1:{port}/apply-template',json={'messages':messages,'add_generation_prompt':True},timeout=30);template.raise_for_status()
    rendered=template.json()['prompt']
    tokenized=requests.post(f'http://127.0.0.1:{port}/tokenize',json={'content':rendered,'add_special':True},timeout=30);tokenized.raise_for_status()
    n_prompt=len(tokenized.json()['tokens'])
    if n_prompt+512>16384:raise ValueError(f'{tag}/{key}: {n_prompt}+512 exceeds context; no truncation allowed')
    t=time.monotonic()
    response=requests.post(f'http://127.0.0.1:{port}/v1/chat/completions',json={'model':tag,'messages':messages,'temperature':0.,'top_p':1.,'top_k':0,'repeat_penalty':1.,'max_tokens':512,'seed':0,'cache_prompt':False},timeout=900)
    response.raise_for_status();obj=response.json();choice=obj['choices'][0];usage=obj['usage'];answer=choice['message']['content'] or ''
    assert usage['prompt_tokens']+usage['completion_tokens']<=16384,(tag,key,usage)
    rec={'key':key,'id':qid,'model':tag,'branch':branch,'text':answer,'prompt_tokens':usage['prompt_tokens'],'generated_tokens':usage['completion_tokens'],'finish_reason':choice['finish_reason'],'seconds':time.monotonic()-t,'preflight_prompt_tokens':n_prompt,'physical_gpus':m['gpus'],'replica':replica}
    if branch is None:rec['selected']=parse_ids(answer,len(pool(q)))
    stream.write(json.dumps(rec,ensure_ascii=False)+'\n');stream.flush();os.fsync(stream.fileno())
    write(out/'workers'/f'{tag}_replica{replica}.json',{'phase':'generating','model':tag,'gpus':m['gpus'],'pilot':pilot,'done_this_launch':j+1,'todo_this_launch':len(jobs),'cached_total':len(records)+j+1,'last_id':qid,'last_prompt_tokens':usage['prompt_tokens'],'last_generated_tokens':usage['completion_tokens'],'last_seconds':rec['seconds']})
  write(out/'workers'/f'{tag}_replica{replica}.json',{'phase':'complete','model':tag,'pilot':pilot,'cached_total':len(records)+len(jobs)})
 finally:
  if server.poll() is None:
   server.terminate()
   try:server.wait(timeout=30)
   except subprocess.TimeoutExpired:server.kill();server.wait()
  log.close()

def run_phase(out,stage,pilot):
 tasks=[m[0] for m in MODELS if m[1]==stage]
 if pilot:
  with ThreadPoolExecutor(max_workers=3) as ex:
   results=[ex.submit(worker,out,tag,True) for tag in tasks]
   join_futures(results)
 else:
  for tag in tasks:
   needed=2 if tag in ['qwen72','llama70'] else 1
   while True:
    used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
    free=[i for i,x in enumerate(used.splitlines()) if int(x.strip())<1024]
    if len(free)>=needed:break
    write(out/'resource_wait.json',{'model':tag,'phase':'waiting_for_free_GPUs','required':needed,'free':free,'updated':time.time()})
    time.sleep(15)
   groups=[free[j:j+needed] for j in range(0,len(free)-needed+1,needed)]
   write(out/'active_model.json',{'stage':stage,'model':tag,'replicas':len(groups),'gpu_groups':groups})
   with ThreadPoolExecutor(max_workers=len(groups)) as ex:
    results=[ex.submit(worker,out,tag,False,j,len(groups),gpus) for j,gpus in enumerate(groups)]
    join_futures(results)

def main(out,pilot_only=False,import_pilot=None):
 settings,data=prepare(out)
 if import_pilot and not (out/'pilot.json').exists():
  previous=json.loads((import_pilot/'protocol.json').read_text());passed=json.loads((import_pilot/'pilot.json').read_text())
  assert passed['status']=='passed'
  for key in ['generation','data_sha256','ruler_sha256','splits']:assert previous[key]==settings[key],key
  for tag in settings['models']:
   for key in ['repo','revision','files']:assert previous['models'][tag][key]==settings['models'][tag][key]
   (out/'cache').mkdir(exist_ok=True);shutil.copy2(import_pilot/'cache'/f'{tag}.jsonl',out/'cache'/f'{tag}.jsonl')
  passed['imported_from']=str(import_pilot);passed['source_protocol_sha256']=digest(import_pilot/'protocol.json')
  write(out/'pilot.json',passed)
 if pilot_only:
  gpu=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
  assert all(int(x.strip())<1024 for x in gpu.splitlines()),'Pilot requires all4 GPUs free; never adopt external work'
 if not (out/'pilot.json').exists():
  for stage in ['retrieval','answering']:
   write(out/'status.json',{'phase':'pilot_'+stage,'full_generation_started':False,'training_started':False})
   run_phase(out,stage,True)
  ids=settings['splits']['train_extra'][:2];summary={}
  for tag,stage,_,_ in MODELS:
   recs=model_cache(out,tag);expected=2 if stage=='retrieval' else 6
   assert len(recs)>=expected
   summary[tag]={'calls':len(recs),'max_prompt_tokens':max(r['prompt_tokens'] for r in recs.values()),'mean_seconds':sum(r['seconds'] for r in recs.values())/len(recs)}
   if stage=='retrieval':assert any(r['selected'] for r in recs.values()),f'{tag} pilot selected no references'
  write(out/'pilot.json',{'status':'passed','training_ids_only':ids,'models':summary})
 if pilot_only:return
 for stage in ['retrieval','answering']:
  write(out/'status.json',{'phase':stage,'full_generation_started':True,'training_started':False,'scope':settings['comparison_scope']})
  run_phase(out,stage,False)
 write(out/'status.json',{'phase':'training','full_generation_started':True,'training_started':True})
 for name in ['run.py','train.py']:assert digest(Path(__file__).with_name(name))==digest(out/'code'/name),'Code changed during generation; explicit migration required'
 subprocess.run([sys.executable,str(Path(__file__).with_name('train.py')),'--out',str(out)],check=True)
 write(out/'status.json',{'phase':'complete','full_generation_started':True,'training_started':True,'scope':settings['comparison_scope']})

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=WORK/'runs/nonllm_common_menuA_gguf_v1');p.add_argument('--pilot-only',action='store_true');p.add_argument('--import-pilot',type=Path);a=p.parse_args()
 try:main(a.out.resolve(),a.pilot_only,a.import_pilot)
 except Exception as e:
  write(a.out/'status.json',{'phase':'failed','error':str(e),'error_type':type(e).__name__});raise
