"""Run a small synthetic serving canary before spending on dataset inference."""
import argparse,asyncio,json
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer
CHECKS=[('What is the capital of France? Answer with the city name only.','paris'),('What is 2 + 2? Answer with the number only.','4'),('Reply with exactly the word READY and nothing else.','ready')]
async def main(a):
 tok=AutoTokenizer.from_pretrained(a.tokenizer);rows=[]
 async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=120)) as s:
  for question,expected in CHECKS:
   text=tok.apply_chat_template([{'role':'system','content':'You are a helpful assistant.'},{'role':'user','content':question}],tokenize=False,add_generation_prompt=True)
   body={'input_ids':tok.encode(text,add_special_tokens=False),'sampling_params':{'temperature':0,'max_new_tokens':32,'top_k':50,'top_p':1,'repetition_penalty':1}}
   async with s.post(a.endpoint+'/generate',json=body) as resp:
    raw=await resp.json()
    if resp.status!=200:raise RuntimeError(str(raw))
   output=raw['text'].strip();passed=output.strip(' .!\n\t').lower()==expected;rows.append({'question':question,'expected':expected,'output':output,'passed':passed})
 report={'endpoint':a.endpoint,'tokenizer':a.tokenizer,'passed':all(x['passed'] for x in rows),'checks':rows};Path(a.output).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
 if not report['passed']:raise SystemExit('Serving quality canary failed; do not start dataset jobs')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--endpoint',required=True);p.add_argument('--tokenizer',required=True);p.add_argument('--output',required=True);asyncio.run(main(p.parse_args()))
