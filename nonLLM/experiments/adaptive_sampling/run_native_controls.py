"""Official safetensors controls, same saved token IDs as the GGUF controls."""
import ast,asyncio,json,re,string,time
from collections import Counter
from pathlib import Path
import aiohttp
from answer_parser import extract_answer
R=Path('/data1/zmcheng/cs707_sampling40');O=R/'diagnostics/results'
async def main():
 cases=json.load(open(O/'cases.json'));env={'re':re,'string':string,'Counter':Counter};tree=ast.parse((R/'assets/hotpot_evaluate_v1.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_answer','exact_match_score','f1_score']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official>','exec'),env)
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
  async def backend(name,port):
   deadline=time.time()+900
   while True:
    try:
     async with session.get(f'http://127.0.0.1:{port}/health') as r:
      if r.status==200:break
    except aiohttp.ClientError:pass
    if time.time()>deadline:raise RuntimeError('Model readiness timeout')
    await asyncio.sleep(3)
   path=O/f'{name}.jsonl';done={json.loads(l)['case'] for l in path.read_text().splitlines()} if path.exists() else set()
   with open(path,'a') as f:
    for c in cases:
     if c['name'] in done or len(c['input_ids'])+128>32768:continue
     start=time.time()
     body={'input_ids':c['input_ids'],'sampling_params':{'temperature':0,'top_k':50,'top_p':1,'max_new_tokens':128,'repetition_penalty':1}}
     async with session.post(f'http://127.0.0.1:{port}/generate',json=body) as resp:
      raw=await resp.json()
      if resp.status!=200:raise RuntimeError(str(raw))
     text=raw['text'];pred=extract_answer(text);row={'backend':name,'case':c['name'],'kind':c['kind'],'index':c.get('index'),'gold':c['gold'],'text':text,'seconds':time.time()-start,'input_tokens':len(c['input_ids']),'em':float(max(env['exact_match_score'](text,g) for g in c['gold'])),'f1':float(max(env['f1_score'](text,g)[0] for g in c['gold'])),'extracted_answer':pred,'extracted_em':float(max(env['exact_match_score'](pred or '',g) for g in c['gold'])),'raw':raw};f.write(json.dumps(row)+'\n');f.flush();print(name,c['name'],repr(text[:120]),flush=True)
  await asyncio.gather(backend('sglang_instruct15_hf',30184),backend('sglang_instruct7_hf',30185))
if __name__=='__main__':asyncio.run(main())
