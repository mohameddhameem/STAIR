"""Small training-only pilot. Public AgentTTS generation prompts remain unchanged."""
import argparse,ast,asyncio,gzip,json,re,string,time
from collections import Counter,defaultdict
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer
from protocol import load_protocol
from grounded_answer import extract
R=Path('/data1/zmcheng/cs707_sampling40');MODELS=R/'diagnostics/models'
async def main(a):
 O=R/'diagnostics/results'/a.label;O.mkdir(exist_ok=True);PORT={'W':30186 if a.long_context else 30184,'S':30187 if a.long_context else 30185};capacity=131072 if a.long_context else 32768
 tok={m:AutoTokenizer.from_pretrained(MODELS/f'Qwen2.5-{size}-Instruct-HF') for m,size in [('W','1.5B'),('S','7B')]};generation={m:json.load(open(MODELS/f'Qwen2.5-{size}-Instruct-HF/generation_config.json')) for m,size in [('W','1.5B'),('S','7B')]};prompt=load_protocol(R/'upstream');data=json.load(gzip.open(R/'assets/AgentTTS-HotpotQA.json.gz','rt'));train=json.load(open(R/'assets/splits.json'))['splits']['train']
 env={'re':re,'string':string,'Counter':Counter};tree=ast.parse((R/'assets/hotpot_evaluate_v1.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_answer','exact_match_score','f1_score']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official>','exec'),env)
 rows=[];genpath=O/'instruct_repair_pilot_generations.jsonl';manifest=O/'profile.json';profile={'model_generation_configs':generation,'context_limit':capacity,'candidates':a.candidates,'budgets':a.budgets,'questions':a.questions,'hf_repetition_penalty':a.hf_repetition_penalty}
 if manifest.exists() and json.loads(manifest.read_text())!=profile:raise RuntimeError('Pilot profile changed; select a fresh --label')
 if genpath.exists() and not manifest.exists():raise RuntimeError('Unversioned diagnostic cache; select a fresh --label')
 manifest.write_text(json.dumps(profile,indent=2));cached={json.loads(l)['key']:json.loads(l)['value'] for l in genpath.read_text().splitlines()} if genpath.exists() else {}
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=900)) as session:
  for port in sorted(set([*PORT.values(),30185])):
   deadline=time.time()+300
   while True:
    try:
     async with session.get(f'http://127.0.0.1:{port}/health') as health:
      if health.status==200:break
    except aiohttp.ClientError:pass
    if time.time()>deadline:raise RuntimeError(f'Serving readiness timeout on port {port}')
    await asyncio.sleep(2)
  checks=[]
  for question,response,expected in [('What is the capital of France?','The answer is Berlin.','Berlin'),('What is 2 + 2?','The answer is 5.','5'),('What is the capital of France?','Berlin is the capital. Paris is not the answer.','Berlin'),('What is 2 + 2?','The sum given in this response is five.','five'),('Who wrote this book?','I cannot determine the author.',None),('What is the capital of France?','What is the capital of France?',None),('Are they of the same nationality?','No, they are not of the same nationality.','no'),('Are J and K of the same nationality?','Both J and K are British.','yes'),('Are J and K of the same nationality?','J is British, while K is German.','no'),('Is 2 + 2 equal to 4?','Is 2 + 2 equal to 4?',None)]:
   result=await extract(session,'http://127.0.0.1:30185',tok['S'],question,response);ok=result['answer']==expected or (expected is not None and result['answer'] is None);checks.append({'question':question,'response':response,'expected':expected,'result':result,'passed':ok})
  (O/'extractor_quality_gate.json').write_text(json.dumps(checks,indent=2));print('EXTRACTOR CHECKS',[(r['expected'],r['result']['answer'],r['passed']) for r in checks],flush=True)
  if not all(c['passed'] for c in checks) or sum(c['result']['answer'] is not None for c in checks)<4:raise RuntimeError('Extractor changed a response or invented an answer; stop pilot')
  async def generate(key,m,messages,temp,n=1):
   if key in cached:return cached[key]
   text=tok[m].apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tok[m].encode(text,add_special_tokens=False)
   if len(ids)+512>capacity:raise RuntimeError(f'Pilot context limit: {key} {len(ids)}')
   start=time.time();body={'input_ids':ids,'sampling_params':{'temperature':temp,'top_k':generation[m].get('top_k',50),'top_p':generation[m].get('top_p',1),'max_new_tokens':512,'n':n,'repetition_penalty':generation[m].get('repetition_penalty',1),'stop_token_ids':generation[m]['eos_token_id']}}
   if a.hf_repetition_penalty:
    from hf_repetition import apply_to_request
    apply_to_request(body,generation[m].get('repetition_penalty',1))
   async with session.post(f'http://127.0.0.1:{PORT[m]}/generate',json=body) as res:
    raw=await res.json()
    if res.status!=200:raise RuntimeError(str(raw))
   out=raw if isinstance(raw,list) else [raw]
   if len(out)!=n:raise RuntimeError(f'Expected {n} outputs for {key}, received {len(out)}')
   value={'outputs':out,'seconds':time.time()-start,'input_tokens':len(ids)};cached[key]=value
   with open(genpath,'a') as f:f.write(json.dumps({'key':key,'value':value})+'\n')
   return value
  eligible=[];skipped=[]
  for i in train[:16]:
   msg=prompt('fusion',data[i],answers=['x']*a.candidates);count=len(tok['S'].encode(tok['S'].apply_chat_template(msg,tokenize=False,add_generation_prompt=True),add_special_tokens=False))
   if count+a.candidates*512+512<=capacity:eligible.append(i)
   else:skipped.append(i)
   if len(eligible)==a.questions:break
  sem=asyncio.Semaphore(2)
  if len(eligible)!=a.questions:raise RuntimeError(f'Only {len(eligible)} training questions fit the requested diagnostic profile')
  async def question(i):
   async with sem:
     e=data[i];ret={}
     for m in ['W','S']:ret[m]=(await generate(f'{i}:ret:{m}',m,prompt('retrieval',e),.9))['outputs'][0]['text']
     for rm,am in [('W','W'),('W','S'),('S','W'),('S','S')]:
      pair=rm+am;qa=await generate(f'{i}:qa:{pair}',am,prompt('qa',e,retrieval=ret[rm]),.7,a.candidates);answers=[o['text'] for o in qa['outputs']]
      for k in sorted(a.budgets,reverse=True):
       fused=await generate(f'{i}:fuse:{pair}:{k}',am,prompt('fusion',e,answers=answers[:k]),.3);text=fused['outputs'][0]['text'];extracted=await extract(session,'http://127.0.0.1:30185',tok['S'],e['question'],text);pred=extracted['answer'] or ''
       row={'index':i,'pair':pair,'k':k,'question':e['question'],'gold':e['answers'],'raw_text':text,'extraction':extracted,'raw_em':float(max(env['exact_match_score'](text,g) for g in e['answers'])),'raw_f1':float(max(env['f1_score'](text,g)[0] for g in e['answers'])),'extracted_em':float(max(env['exact_match_score'](pred,g) for g in e['answers'])),'extracted_f1':float(max(env['f1_score'](pred,g)[0] for g in e['answers']))}
       rows.append(row);(O/'instruct_repair_pilot_predictions.json').write_text(json.dumps(rows,indent=2));print('PILOT',i,pair,k,repr(pred),row['extracted_em'],flush=True)
  await asyncio.gather(*(question(i) for i in eligible))
  report={'training_indices':eligible,'skipped_for_diagnostic_context_limit':skipped,'context_limit':capacity,'long_context_yarn':a.long_context,'max_generation_input_tokens':max(v['input_tokens'] for v in cached.values()),'questions':len(eligible),'rows':len(rows),'generation_prompts':'pinned public AgentTTS unchanged','generation_sampling':'retrieval .9; QA .7; fusion .3; max512; other parameters inherited from official generation_config as in upstream model.generate','model_generation_configs':generation,'qa_candidates':a.candidates,'budgets':a.budgets,'postprocessor':'deterministic parser then fixed Qwen2.5-7B-Instruct, greedy JSON answer/evidence extraction with quoted-evidence grounding; no gold input; separately recorded inference cost','raw_em':sum(r['raw_em'] for r in rows)/len(rows),'extracted_em':sum(r['extracted_em'] for r in rows)/len(rows),'extracted_f1':sum(r['extracted_f1'] for r in rows)/len(rows),'extraction_coverage':sum(r['extraction']['answer'] is not None for r in rows)/len(rows),'extractor_prompt_tokens':sum(r['extraction']['usage'].get('prompt_tokens',0) for r in rows),'extractor_completion_tokens':sum(r['extraction']['usage'].get('completion_tokens',0) for r in rows),'extraction_seconds':sum(r['extraction']['seconds'] for r in rows),'caution':'Small training-only debugging pilot, not a held-out performance estimate or proof of adaptive-budget benefit'}
  report['repetition_penalty_scope']='prompt_and_output' if a.hf_repetition_penalty else 'native_output_only'
  (O/'instruct_repair_pilot_report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--label',default='source_defaults');parser.add_argument('--long-context',action='store_true');parser.add_argument('--questions',type=int,default=8);parser.add_argument('--candidates',type=int,default=4);parser.add_argument('--budgets',type=int,nargs='+',default=[1,4]);parser.add_argument('--hf-repetition-penalty',action='store_true');a=parser.parse_args()
 if min(a.budgets)<1 or max(a.budgets)>a.candidates:parser.error('budgets must be within 1..candidates')
 asyncio.run(main(a))
