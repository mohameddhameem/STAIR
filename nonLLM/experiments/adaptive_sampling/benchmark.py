import asyncio,aiohttp,gzip,json,sys,time
from pathlib import Path
from transformers import AutoTokenizer
from protocol import load_protocol
R=Path('/data1/zmcheng/cs707_sampling40')
async def main():
    tok=AutoTokenizer.from_pretrained(R/'tokenizers/Qwen2.5-7B');p=load_protocol(R/'upstream');e=json.load(gzip.open(R/'assets/AgentTTS-HotpotQA.json.gz','rt'))[0]
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800)) as s:
        async def call(kind,n,answers=None):
            msgs=p(kind,e,retrieval='Query: q\nSelected references: [1], [3]',answers=answers)
            ids=tok.encode(tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True),add_special_tokens=False)
            start=time.time()
            async with s.post('http://127.0.0.1:30080/generate',json={'input_ids':ids,'sampling_params':{'temperature':.7 if kind=='qa' else .3,'top_k':50,'top_p':1.,'max_new_tokens':512,'n':n,'sampling_seed':12345}}) as resp:
                result=await resp.json()
                if resp.status!=200:raise RuntimeError(str(result))
            outputs=result if isinstance(result,list) else [result]
            print(json.dumps({'kind':kind,'n':n,'input_tokens':len(ids),'seconds':time.time()-start,'outputs':len(outputs),'completion_tokens':sum(x['meta_info']['completion_tokens'] for x in outputs)}),flush=True)
            return outputs
        qa=await call('qa',40);answers=[x['text'] for x in qa]
        start=time.time();await call('fusion',1,answers)
        await asyncio.gather(*(call('fusion',1,answers[:k]) for k in range(1,40)))
        print(json.dumps({'prefix_fusions':40,'seconds':time.time()-start}),flush=True)
        (R/'runs/benchmark.success').write_text('40 candidates and all 40 prefix fusions verified\n')
if __name__=='__main__':asyncio.run(main())
