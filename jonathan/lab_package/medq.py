import os,json,re,argparse,urllib.request
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
a=argparse.ArgumentParser()
a.add_argument('--model',default='Qwen/Qwen2.5-7B-Instruct')
a.add_argument('--tag',default='weak')
a.add_argument('--n',type=int,default=10)
a.add_argument('--data',default='/mnt/hdd/jonathan/data/MedAgentBench/data/medagentbench')
a.add_argument('--base',default='http://localhost:8080/fhir')
a=a.parse_args()
d=json.load(open(f"{a.data}/test_data_v2.json"))
funcs=json.load(open(f"{a.data}/funcs_v1.json"))
def _spec(f):
    fn=f.get("function",f)
    ps=fn.get("parameters",{}).get("properties",{})
    pl=", ".join(sorted(ps)) if ps else "(none documented)"
    return f"- {fn['name'].replace(chr(123)+'api_base'+chr(125), a.base)}\n    {fn.get('description','')[:200]}\n    parameters: {pl}"
tools="\n".join(_spec(f) for f in funcs)
SYS=("You are a clinical assistant with access to a FHIR API. "
     "Reply with ONE line only: the exact HTTP request to make, e.g.\n"
     "GET {base}/Observation?patient=S123&code=K&_count=1\n"
     "No explanation.")
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
    device_map="auto",attn_implementation="sdpa").eval()
def ask(instr):
    msgs=[{"role":"system","content":SYS.format(base=a.base)},
          {"role":"user","content":f"Available API endpoints:\n{tools}\n\nTask: {instr}\n\nThe request:"}]
    t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    enc=tok(t,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
    with torch.no_grad():
        o=m.generate(**enc,max_new_tokens=120,do_sample=False,pad_token_id=tok.pad_token_id)
    return tok.decode(o[0][enc.input_ids.shape[1]:],skip_special_tokens=True).strip()
def fetch(url):
    try:
        with urllib.request.urlopen(url,timeout=30) as r: return len(r.read()),200
    except urllib.error.HTTPError as e: return 0,e.code
    except Exception: return 0,0
out=open(f"medq_{a.tag}.jsonl","w")
tasks=sorted({x['id'].split('_')[0] for x in d}, key=lambda s:int(s[4:]))
print(f"model {a.model}  |  {a.n} instances per task")
for t in tasks:
    xs=[x for x in d if x['id'].startswith(t+'_')][:a.n]
    for x in xs:
        raw=ask(x['instruction'])
        mm=re.search(r'(GET|POST)\s+(\S+)',raw)
        verb,url=(mm.group(1),mm.group(2)) if mm else ("","")
        url=url.strip('`"\'')
        url=url.replace("{api_base}",a.base).replace("{base}",a.base)
        if url.startswith("/"): url=a.base+url
        n,code=(fetch(url) if verb=="GET" and url.startswith("http") else (0,-1))
        out.write(json.dumps({"id":x['id'],"task":t,"raw":raw[:300],
                              "verb":verb,"url":url[:300],"bytes":n,"http":code})+"\n")
    out.flush(); print(f"  {t} done",flush=True)
out.close(); print("DONE")
