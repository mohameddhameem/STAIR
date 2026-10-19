import os, re, json, string, argparse
import numpy as np
from collections import defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, run_labels as R

COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,SS=0,3
SYS_CB="You are a helpful assistant. Answer the question as briefly as possible."
USR_CB=("Answer the following question from your own knowledge. "
        "Reply with the answer only \u2014 no explanation.\n\nQuestion: {q}\nAnswer:")

def norm(s):
    s=''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
def em_any(p,gs): return int(any(norm(p)==norm(g) for g in gs))

ap=argparse.ArgumentParser()
ap.add_argument('--out',default='/mnt/hdd/jonathan/runs/full_run')
ap.add_argument('--data',default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--splits',default='splits.json')
ap.add_argument('--n',type=int,default=150)
ap.add_argument('--all',action='store_true')
ap.add_argument('--save',default='closedbook.jsonl')
a=ap.parse_args()

data=json.load(open(a.data)); te=json.load(open(a.splits))["splits"][os.environ.get("SLR_SPLIT","test")]
em=defaultdict(dict)
for l in open(os.path.join(a.out,'combo_results_full.jsonl')):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in COMBOS] for i,d in em.items() if len(d)==4}
open_ans={}
for l in open(os.path.join(a.out,'qa_strong.jsonl')):
    r=json.loads(l); k=str(r['k'])
    if k.endswith('|strong'): open_ans[int(k.split('|')[0])]=r['v']

if a.all:
    groups=[("test split",[i for i in te if i in em and i in open_ans])]
else:
    sel=lambda w,s: [i for i in te if i in em and i in open_ans
                     and em[i][WW]==w and em[i][SS]==s][:a.n]
    groups=[("HEADROOM  weak wins, strong fails",sel(1,0)),
            ("CONTROL   both solve",sel(1,1)),
            ("HOPELESS  neither solves",sel(0,0))]

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
print(f"loading {R.STRONG_MODEL} (4-bit) ...")
tok=AutoTokenizer.from_pretrained(R.STRONG_MODEL)
model=AutoModelForCausalLM.from_pretrained(R.STRONG_MODEL,device_map="cuda",
    attn_implementation="sdpa",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True))
model.eval()

def closed_book(q):
    msgs=[{"role":"system","content":SYS_CB},{"role":"user","content":USR_CB.format(q=q)}]
    try: txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ids=tok(txt,return_tensors="pt",truncation=True,max_length=1024).to(model.device)
    with torch.no_grad():
        o=model.generate(**ids,max_new_tokens=32,do_sample=False,pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][ids.input_ids.shape[1]:],skip_special_tokens=True).strip()

done=set()  # RESUME
if os.path.exists(a.save):
    for _l in open(a.save):
        try: done.add(json.loads(_l)['idx'])
        except Exception: pass
    print(f'resuming, {len(done)} already recorded')
groups=[(n,[i for i in g if i not in done]) for n,g in groups]
sink=open(a.save,'a')
print(f"\n{'group':38s} {'n':>4s} {'agree':>7s} {'CB right':>9s}")
print("-"*64)
for name,group in groups:
    ag=cr=0
    for n,i in enumerate(group):
        cb=closed_book(data[i]['question']); ob=open_ans[i]
        x=int(norm(cb)==norm(ob)); c=em_any(cb,data[i]['answers'])
        ag+=x; cr+=c
        sink.write(json.dumps({"idx":i,"closed_book":cb[:120],
                               "agrees_with_open":x,"cb_correct":c})+"\n")
        if (n+1)%50==0: print(f"    ... {name[:20]} {n+1}/{len(group)}",flush=True)
    sink.flush(); t=max(1,len(group))
    print(f"{name:38s} {len(group):4d} {100*ag/t:6.1f}% {100*cr/t:8.1f}%")
print("\nagree = same answer with and without passages, i.e. the evidence changed nothing.")
