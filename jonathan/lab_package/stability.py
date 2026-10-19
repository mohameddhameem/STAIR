import os, json, torch, numpy as np
from collections import defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
import run_labels as R

OUT='/mnt/hdd/jonathan/runs/full_run'
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]
WW,SS=0,3
N_SAMPLES=3; TEMP=0.7

data=json.load(open('/mnt/hdd/jonathan/data/tts_test.json'))
sp=json.load(open('splits.json'))["splits"]; te=sp["test"]
em=defaultdict(dict)
for l in open(f'{OUT}/combo_results_full.jsonl'):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in COMBOS] for i,d in em.items() if len(d)==4}

# the instances that ARE the headroom: weak solves, strong does not
gap=[i for i in te if i in em and em[i][WW]==1 and em[i][SS]==0][:120]
ctl=[i for i in te if i in em and em[i][SS]==1 and em[i][WW]==1][:120]
print(f"headroom instances (ww solves, ss fails): {len(gap)}")
print(f"control (both solve):                     {len(ctl)}")

retr={}
for l in open(f'{OUT}/retr_strong.jsonl'):
    r=json.loads(l); retr[str(r['k']).split('|')[0]]=r['v']

m,tok=R.load_model('strong','4bit') if hasattr(R,'load_model') else (None,None)
if m is None:
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    tok=AutoTokenizer.from_pretrained(R.STRONG_MODEL)
    m=AutoModelForCausalLM.from_pretrained(R.STRONG_MODEL, device_map="cuda",
        attn_implementation="sdpa",
        quantization_config=BitsAndBytesConfig(load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"))
m.eval()

def sample_em(i):
    inst=dict(data[i]); pool=R.trim_pool(inst)
    picks=R.parse_selection(retr[str(i)], len(pool))
    blocks=[pool[p-1] for p in picks if 1<=p<=len(pool)]
    msgs=[{"role":"system","content":R.SYS_QA},
          {"role":"user","content":R.qa_prompt(inst, blocks)}]
    txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    ids=tok(txt,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
    outs=[]
    with torch.no_grad():
        for _ in range(N_SAMPLES):
            o=m.generate(**ids,max_new_tokens=128,do_sample=True,temperature=TEMP,
                         top_p=0.95,pad_token_id=tok.eos_token_id)
            a=tok.decode(o[0][ids.input_ids.shape[1]:],skip_special_tokens=True)
            outs.append(max(R.em_official(a,g) for g in inst['answers']))
    return outs

for name,group in (("HEADROOM (ww wins)",gap),("CONTROL (both win)",ctl)):
    flips=0; tot=0
    for n,i in enumerate(group):
        if str(i) not in retr: continue
        r=sample_em(i); tot+=1
        if len(set(r))>1: flips+=1
        if (n+1)%40==0: print(f"  {name}: {n+1}/{len(group)} ...", flush=True)
    print(f"\n{name}: {flips}/{tot} instances flip across {N_SAMPLES} samples at T={TEMP} "
          f"({100*flips/max(1,tot):.1f}%)")
