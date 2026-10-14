"""Training-only generation controls; never modifies the completed experiment."""
import ast,asyncio,gzip,json,re,sqlite3,string,time
from collections import Counter
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer
from protocol import load_protocol
R=Path('/data1/zmcheng/cs707_sampling40');OUT=R/'diagnostics/results';OUT.mkdir(exist_ok=True)
async def main():
 tok=AutoTokenizer.from_pretrained(R/'tokenizers/Qwen2.5-7B');prompt=load_protocol(R/'upstream');data=json.load(gzip.open(R/'assets/AgentTTS-HotpotQA.json.gz','rt'));train=json.load(open(R/'assets/splits.json'))['splits']['train'];cases=[]
 def add(name,text,answer,**meta):
  if isinstance(text,list):text=tok.apply_chat_template(text,tokenize=False,add_generation_prompt=True)
  cases.append({'name':name,'prompt':text,'input_ids':tok.encode(text,add_special_tokens=False),'gold':answer,**meta})
 add('capital_completion','The capital of France is',['Paris'],kind='raw_control')
 add('capital_chat',[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':'What is the capital of France? Answer with the city name only.'}],['Paris'],kind='chat_control')
 add('arithmetic_chat',[{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':'What is 2 + 2? Answer with the number only.'}],['4'],kind='chat_control')
 for i in train[:12]:
  e=data[i];goldids=[j+1 for j,t in enumerate(e['ctxs_candidate']) if t.split(':',1)[0] in e['supporting_titles']]
  if len(goldids)!=len(e['supporting_titles']):continue
  selected='Query: '+e['question']+'\nSelected references: '+', '.join(f'[{j}]' for j in goldids)
  msgs=prompt('qa',e,retrieval=selected);add(f'q{i}_public_chat',msgs,e['answers'],kind='public_qa_gold_evidence',index=i)
  modified=[dict(m) for m in msgs];modified[-1]['content']+='\nReturn only the short answer, without explanation.';add(f'q{i}_short_chat',modified,e['answers'],kind='short_contract_qa_gold_evidence',index=i)
  add(f'q{i}_plain_answer_prefix',msgs[-1]['content']+'\nAnswer:',e['answers'],kind='base_plain_qa_gold_evidence',index=i)
  if i in train[:3]:
   for contract in ['public','short']:
    msgs=prompt('fusion',e,answers=[e['answers'][0]]*3)
    if contract=='short':msgs[-1]['content']+='\nReturn only the short answer, without explanation.'
    add(f'q{i}_{contract}_fusion_gold_candidates',msgs,e['answers'],kind=contract+'_fusion_gold_candidates',index=i)
 (OUT/'cases.json').write_text(json.dumps(cases,indent=2));print('CASES',len(cases),flush=True)
 env={'re':re,'string':string,'Counter':Counter};tree=ast.parse((R/'assets/hotpot_evaluate_v1.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_answer','exact_match_score','f1_score']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official>','exec'),env)
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as s:
  async def backend(name,port,sg=False,extra_stop=False):
   path=OUT/f'{name}.jsonl';done={json.loads(l)['case'] for l in path.read_text().splitlines()} if path.exists() else set()
   with open(path,'a') as f:
    for c in cases:
     if c['name'] in done:continue
     if len(c['input_ids'])+128>32768:continue
     if sg:
      body={'input_ids':c['input_ids'],'sampling_params':{'temperature':0,'top_k':50,'top_p':1,'max_new_tokens':128,'repetition_penalty':1}}
      if extra_stop:body['sampling_params']['stop_token_ids']=[151645]
      url='/generate'
     else:body={'prompt':c['input_ids'],'temperature':0,'top_k':50,'top_p':1,'min_p':0,'repeat_penalty':1,'n_predict':128,'cache_prompt':True,'return_tokens':True,'seed':42};url='/completion'
     start=time.time()
     async with s.post(f'http://127.0.0.1:{port}'+url,json=body) as resp:
      raw=await resp.json()
      if resp.status!=200:raise RuntimeError(f'{name} {c["name"]}: {raw}')
     text=raw['text'] if sg else raw['content'];ntext=env['normalize_answer'](text)
     row={'backend':name,'case':c['name'],'kind':c['kind'],'index':c.get('index'),'gold':c['gold'],'text':text,'seconds':time.time()-start,'input_tokens':len(c['input_ids']),'em':float(max(env['exact_match_score'](text,g) for g in c['gold'])),'f1':float(max(env['f1_score'](text,g)[0] for g in c['gold'])),'gold_coverage_diagnostic':any(re.search(r'(?<!\w)'+re.escape(env['normalize_answer'](g))+r'(?!\w)',ntext) is not None for g in c['gold']),'raw':raw}
     f.write(json.dumps(row)+'\n');f.flush();print(name,c['name'],repr(text[:120]),flush=True)
  await asyncio.gather(backend('llama_base7',30180),backend('llama_base15',30183),backend('llama_instruct15',30181))
  # SGLang is a separate cross-backend control with identical HF token IDs.
  while True:
   try:
    async with s.get('http://127.0.0.1:30182/health') as resp:
     if resp.status==200:break
   except aiohttp.ClientError:pass
   await asyncio.sleep(3)
  await backend('sglang_base7',30182,True);await backend('sglang_base7_chatstop',30182,True,True)
if __name__=='__main__':asyncio.run(main())
