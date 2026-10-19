import os,json,re,argparse,urllib.request,urllib.error
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
P=argparse.ArgumentParser()
P.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct')
P.add_argument('--tag',default='weak')
P.add_argument('--n',type=int,default=5)
P.add_argument('--rounds',type=int,default=8)
P.add_argument('--data',default='/mnt/hdd/jonathan/data/MedAgentBench/data/medagentbench')
P.add_argument('--base',default='http://localhost:8080/fhir/')
P.add_argument('--q4',action='store_true')
a=P.parse_args()
PROMPT = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""
d=json.load(open(f"{a.data}/test_data_v2.json"))
funcs=json.load(open(f"{a.data}/funcs_v1.json"))
FSTR=json.dumps(funcs)
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
kw=dict(device_map="auto",attn_implementation="sdpa")
if a.q4:
    from transformers import BitsAndBytesConfig
    kw["quantization_config"]=BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_compute_dtype=torch.float16)
else: kw["torch_dtype"]=torch.float16
m=AutoModelForCausalLM.from_pretrained(a.model,**kw).eval()
def gen(msgs):
    t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    enc=tok(t,return_tensors="pt",truncation=True,max_length=16384).to(m.device)
    with torch.no_grad():
        o=m.generate(**enc,max_new_tokens=400,do_sample=False,pad_token_id=tok.pad_token_id)
    return tok.decode(o[0][enc.input_ids.shape[1]:],skip_special_tokens=True).strip()
def http_get(u):
    try:
        with urllib.request.urlopen(u,timeout=30) as r:
            b=r.read(); return 200,b[:6000].decode(errors='ignore'),len(b)
    except urllib.error.HTTPError as e:
        b=e.read(); return e.code,b[:2000].decode(errors='ignore'),len(b)
    except Exception as ex: return 0,str(ex)[:300],0
out=open(f"medq2_{a.tag}.jsonl","w")
tasks=sorted({x['id'].split('_')[0] for x in d}, key=lambda s:int(s[4:]))
print(f"{a.model} | {a.n}/task | max {a.rounds} rounds",flush=True)
for t in tasks:
    for case in [x for x in d if x['id'].startswith(t+'_')][:a.n]:
        msgs=[{"role":"user","content":PROMPT.format(api_base=a.base,functions=FSTR,
                 context=case.get('context',''),question=case['instruction'])}]
        log=[]; fin=None; tot=0
        for rd in range(a.rounds):
            r=gen(msgs); msgs.append({"role":"assistant","content":r})
            mf=re.search(r'FINISH\((.*)\)',r,re.S)
            if mf: fin=mf.group(1)[:300]; log.append({"rd":rd,"act":"FINISH"}); break
            mg=re.match(r'\s*GET\s+(\S+)',r)
            mp=re.match(r'\s*POST\s+(\S+)',r)
            if mg:
                code,body,n=http_get(mg.group(1).strip('`"\''))
                tot+=n; log.append({"rd":rd,"act":"GET","http":code,"bytes":n,"url":mg.group(1).strip(chr(96)+chr(34)+chr(39))})
                msgs.append({"role":"user","content":body})
            elif mp:
                ok=False
                try: json.loads(r.split("\n",1)[1]); ok=True
                except Exception: pass
                log.append({"rd":rd,"act":"POST","json_ok":ok})
                msgs.append({"role":"user","content":"POST request accepted" if ok else "Invalid JSON payload"})
            else:
                log.append({"rd":rd,"act":"MALFORMED","txt":r[:120]})
                msgs.append({"role":"user","content":"Invalid format. Reply with GET, POST or FINISH only."})
        out.write(json.dumps({"id":case['id'],"task":t,"rounds":len(log),"msgs":msgs[1:],
                              "finished":fin is not None,"answer":fin,"bytes":tot,"log":log})+"\n")
    out.flush(); print(f"  {t} done",flush=True)
out.close(); print("DONE")
