"""Resumable native SGLang sampling and prefix-fusion experiment."""
import argparse, asyncio, gzip, hashlib, json, sqlite3, time
from pathlib import Path
import aiohttp
from transformers import AutoTokenizer
from protocol import load_protocol

def seed(key): return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4],'big')%2147483647

async def main(a):
    root=Path(a.root); data=json.load(gzip.open(root/'assets/AgentTTS-HotpotQA.json.gz','rt'))
    prompt=load_protocol(root/'upstream'); tok=AutoTokenizer.from_pretrained(root/'tokenizers'/f'Qwen2.5-{ "1.5B" if a.model=="W" else "7B"}')
    db=sqlite3.connect(root/'runs'/f'{a.stage}_{a.model}_{a.shard}.sqlite'); db.execute('PRAGMA journal_mode=WAL'); db.execute('CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY, payload TEXT NOT NULL)'); db.commit()
    def get(key):
        row=db.execute('SELECT payload FROM results WHERE key=?',(key,)).fetchone();return json.loads(row[0]) if row else None
    def save(key,value): db.execute('INSERT OR REPLACE INTO results VALUES (?,?)',(key,json.dumps(value)));db.commit()
    def ret(i,m):
        other=sqlite3.connect(root/'runs'/f'retrieval_{m}_{i%4}.sqlite'); row=other.execute('SELECT payload FROM results WHERE key=?',(f'{i}:ret:{m}',)).fetchone();other.close()
        if not row:raise RuntimeError(f'Missing retrieval {i} {m}')
        return json.loads(row[0])['outputs'][0]['text']
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as session:
        async def generate(key,messages,temp,n=1):
            cached=get(key)
            if cached is not None:return cached
            text=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            ids=tok.encode(text,add_special_tokens=False)
            # Match original HF right-truncation at tokenizer.model_max_length.
            truncated=max(0,len(ids)-tok.model_max_length); ids=ids[:tok.model_max_length]
            if len(ids)+512>131072:raise RuntimeError(f'Context exceeds deployed capacity: {key} {len(ids)}')
            body={'input_ids':ids,'sampling_params':{'temperature':temp,'top_k':50,'top_p':1.0,'repetition_penalty':1.0,'max_new_tokens':512,'n':n,'sampling_seed':seed(key)},'return_logprob':False}
            start=time.time()
            for attempt in range(4):
                try:
                    async with session.post(f'http://127.0.0.1:{a.port}/generate',json=body) as response:
                        raw=await response.json()
                        if response.status!=200:raise RuntimeError(str(raw)[:800])
                    outputs=raw if isinstance(raw,list) else [raw]
                    if len(outputs)!=n:raise RuntimeError(f'Expected {n} outputs, got {len(outputs)}')
                    value={'outputs':outputs,'seconds':time.time()-start,'input_tokens':len(ids),'truncated_tokens':truncated,'seed':seed(key)};save(key,value);return value
                except Exception:
                    if attempt==3:raise
                    await asyncio.sleep(3*(attempt+1))
        sem=asyncio.Semaphore(a.concurrency)
        async def question(i):
            async with sem:
                example=data[i]
                if a.stage=='retrieval':await generate(f'{i}:ret:{a.model}',prompt('retrieval',example),.9)
                else:
                    for rm in ['W','S']:
                        pair=rm+a.model
                        candidates=await generate(f'{i}:qa:{pair}',prompt('qa',example,retrieval=ret(i,rm)),.7,40)
                        answers=[x['text'] for x in candidates['outputs']]
                        # Sequential prefixes maximize shared prefix cache; questions run concurrently.
                        for k in range(1,41):await generate(f'{i}:fuse:{pair}:{k}',prompt('fusion',example,answers=answers[:k]),.3)
                print(json.dumps({'stage':a.stage,'model':a.model,'shard':a.shard,'index':i,'completed':db.execute('SELECT COUNT(*) FROM results').fetchone()[0]}),flush=True)
        indices=[i for i in range(len(data)) if i%4==a.shard]
        if a.limit:indices=indices[:a.limit]
        # Bounded task creation avoids holding millions of futures.
        for offset in range(0,len(indices),a.concurrency):await asyncio.gather(*(question(i) for i in indices[offset:offset+a.concurrency]))
    (root/'runs'/f'{a.stage}_{a.model}_{a.shard}.done').write_text('complete\n');db.close()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--stage',choices=['retrieval','answers'],required=True);p.add_argument('--model',choices=['W','S'],required=True);p.add_argument('--shard',type=int,required=True);p.add_argument('--port',type=int,required=True);p.add_argument('--concurrency',type=int,default=4);p.add_argument('--limit',type=int,default=0);asyncio.run(main(p.parse_args()))
