"""Synthetic >32k input canary under the unchanged public QA prompt."""
import argparse,asyncio,gzip,json,time
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer
from protocol import load_protocol
R=Path('/data1/zmcheng/cs707_sampling40');OUT=R/'diagnostics/results'
async def main(label):
 tok=AutoTokenizer.from_pretrained(R/'diagnostics/models/Qwen2.5-7B-Instruct-HF');p=load_protocol(R/'upstream')
 data=json.load(gzip.open(R/'assets/AgentTTS-HotpotQA.json.gz','rt'));train=json.load(open(R/'assets/splits.json'))['splits']['train'];pieces=[];seen=set();chars=0
 for i in train:
  for ref in data[i]['ctxs_candidate']:
   if ref not in seen:pieces.append(ref.replace('[','(').replace(']',')'));seen.add(ref);chars+=len(ref)
  if chars>=240000:break
 filler=' '.join(pieces)
 ref='Project Bluebird: The verification code assigned to project Bluebird is K7P3. '+filler
 e={'question':'What is the verification code assigned to project Bluebird?','ctxs_candidate':[ref]};messages=p('qa',e,retrieval='Query: control\nSelected references: [1]');text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True);ids=tok.encode(text,add_special_tokens=False);assert len(ids)>32768
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1200)) as s:
  async def one(name,port):
   deadline=time.time()+900
   while True:
    try:
     async with s.get(f'http://127.0.0.1:{port}/health') as r:
      if r.status==200:break
    except aiohttp.ClientError:pass
    if time.time()>deadline:raise RuntimeError('Long-context server readiness timeout')
    await asyncio.sleep(3)
   start=time.time()
   async with s.post(f'http://127.0.0.1:{port}/generate',json={'input_ids':ids,'sampling_params':{'temperature':0,'max_new_tokens':128,'top_k':50,'top_p':1,'repetition_penalty':1}}) as r:
    raw=await r.json()
    if r.status!=200:raise RuntimeError(str(raw))
   result={'model':name,'input_tokens':len(ids),'seconds':time.time()-start,'output':raw['text'],'needle_recovered':'K7P3' in raw['text'],'prompt_tokens_received':raw['meta_info']['prompt_tokens'],'raw':raw};(OUT/f'long_context_{label}_{name}.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='raw'}),flush=True)
   if not result['needle_recovered'] or result['prompt_tokens_received']!=len(ids):raise RuntimeError('Long-context validation failed')
  results=await asyncio.gather(one('instruct15',30186),one('instruct7',30187),return_exceptions=True)
  for r in results:
   if isinstance(r,Exception):print('VALIDATION ERROR',str(r),flush=True)
  if any(isinstance(r,Exception) for r in results):raise SystemExit('Long-context gate failed; do not start dataset jobs')
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--label',default='realistic');asyncio.run(main(parser.parse_args().label))
