import os, re, json, string, argparse
from collections import Counter, defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch, run_labels as R
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]
VSYS="You are a careful evaluator. Judge only from the references provided."
VTPL=("References:\n{refs}\n\nQuestion: {q}\n\nProposed answer: {a}\n\n"
      "Is this proposed answer correct and supported by the references above? "
      "Reply with exactly one word: Yes or No.")
def norm(s):
    s=''.join(c for c in s.lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
ap=argparse.ArgumentParser()
ap.add_argument('--out',default='/mnt/hdd/jonathan/runs/armC_n500')
ap.add_argument('--data',default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--qa-model',default='unsloth/Llama-3.2-3B-Instruct')
ap.add_argument('--verifier',default='Qwen/Qwen3-8B')
ap.add_argument('--k',type=int,default=5); ap.add_argument('--temp',type=float,default=0.9)
ap.add_argument('--max-refs',type=int,default=10); ap.add_argument('--n',type=int,default=0)
ap.add_argument('--cands',default='vsel_cands.jsonl'); ap.add_argument('--scores',default='vsel_scores.jsonl')
a=ap.parse_args()
data=json.load(open(a.data))
retr={}
for w in ('weak','strong'):
    d={}
    for l in open(os.path.join(a.out,f'retr_{w}.jsonl')):
        r=json.loads(l); d[int(str(r['k']).split('|')[0])]=r['v']
    retr[w]=d
_pk=len(R.trim_pool(dict(data[sorted(retr['weak'])[0]])))
print(f'POOL SIZE = {_pk}  (must be 100 for arm C — set SLR_POOL_K=100)')
assert _pk>=100, f'WRONG POOL: {_pk}. Rerun with SLR_POOL_K=100'
ids=sorted(retr['weak'])
if a.n: ids=ids[:a.n]
def refs_for(inst,i):
    pool=inst['ctxs_candidate']; picks=[]
    for w in ('weak','strong'):
        picks+=[p for p in R.parse_selection(retr[w][i],len(pool)) if 1<=p<=len(pool)]
    picks=sorted(set(picks))[:a.max_refs]
    return "\n".join(f"[{k+1}] {pool[p-1]}" for k,p in enumerate(picks)) or "(none)"
done=set()
if os.path.exists(a.cands):
    for l in open(a.cands):
        try: done.add(json.loads(l)['key'])
        except Exception: pass
todo=[(i,w) for i in ids for w in ('weak','strong') if f"{i}|{w}" not in done]
if todo:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[1/3] sampling {a.k}x with {a.qa_model} — {len(todo)} to do")
    tok=AutoTokenizer.from_pretrained(a.qa_model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.qa_model,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    sink=open(a.cands,'a')
    for c,(i,w) in enumerate(todo):
        inst=dict(data[i]); inst['ctxs_candidate']=R.trim_pool(inst); pool=inst['ctxs_candidate']
        picks=[p for p in R.parse_selection(retr[w][i],len(pool)) if 1<=p<=len(pool)]
        msgs=[{"role":"system","content":"You are a helpful assistant."},
              {"role":"user","content":R.qa_prompt(inst,R.build_block(inst,picks))}]
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=tok(t,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
        with torch.no_grad():
            o=m.generate(**enc,max_new_tokens=64,do_sample=True,temperature=a.temp,
                         top_p=0.95,num_return_sequences=a.k,pad_token_id=tok.pad_token_id)
        ni=enc.input_ids.shape[1]
        cc=[tok.decode(x[ni:],skip_special_tokens=True).strip() for x in o]
        sink.write(json.dumps({"key":f"{i}|{w}","idx":i,"src":w,"cands":cc})+"\n")
        if (c+1)%100==0: sink.flush(); print(f"    {c+1}/{len(todo)}",flush=True)
    sink.close(); del m; torch.cuda.empty_cache()
cands={}
for l in open(a.cands):
    r=json.loads(l); cands[r['key']]=r
scored=set()
if os.path.exists(a.scores):
    for l in open(a.scores):
        try: scored.add(json.loads(l)['key'])
        except Exception: pass
todo=[k for k in cands if k not in scored and int(k.split('|')[0]) in ids]
if todo:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"[2/3] verifying with {a.verifier} — {len(todo)} x {a.k}")
    vtok=AutoTokenizer.from_pretrained(a.verifier)
    if vtok.pad_token is None: vtok.pad_token=vtok.eos_token
    vm=AutoModelForCausalLM.from_pretrained(a.verifier,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    idY=vtok.encode("Yes",add_special_tokens=False)[0]
    idN=vtok.encode("No",add_special_tokens=False)[0]
    def p_yes(q,refs,ans):
        msgs=[{"role":"system","content":VSYS},
              {"role":"user","content":VTPL.format(refs=refs,q=q,a=ans)}]
        try: t=vtok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: t=vtok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=vtok(t,return_tensors="pt",truncation=True,max_length=8192).to(vm.device)
        with torch.no_grad():
            o=vm.generate(**enc,max_new_tokens=1,do_sample=False,output_scores=True,
                          return_dict_in_generate=True,pad_token_id=vtok.pad_token_id)
        lg=o.scores[0][0]
        return float(torch.softmax(torch.tensor([lg[idY],lg[idN]]).float(),0)[0])
    sink=open(a.scores,'a')
    for c,key in enumerate(todo):
        i=int(key.split('|')[0])
        inst=dict(data[i]); inst['ctxs_candidate']=R.trim_pool(inst)
        ps=[p_yes(inst['question'],refs_for(inst,i),x) for x in cands[key]['cands']]
        sink.write(json.dumps({"key":key,"p_yes":ps})+"\n")
        if (c+1)%50==0: sink.flush(); print(f"    {c+1}/{len(todo)}",flush=True)
    sink.close()
sc={}
for l in open(a.scores):
    r=json.loads(l); sc[r['key']]=r['p_yes']
em=defaultdict(dict)
for l in open(os.path.join(a.out,'combo_results_full.jsonl')):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:d for i,d in em.items() if len(d)==4}
g=[i for i in ids if i in em and f"{i}|weak" in sc and f"{i}|strong" in sc]; n=len(g)
print(f"\n[3/3] scored {n} instances\n")
def rule(i,w,how):
    key=f"{i}|{w}"; c=cands[key]['cands']; p=sc[key]
    gold=[norm(x) for x in data[i]['answers']]
    if how=='first': return int(norm(c[0]) in gold)
    if how=='best': return int(any(norm(x) in gold for x in c))
    if how=='vote': return int(Counter([norm(x) for x in c]).most_common(1)[0][0] in gold)
    if how=='verify': return int(norm(c[int(np.argmax(p))]) in gold)
print(f"{'selection rule':22s} {'@ weak retr':>12s} {'@ strong retr':>14s}")
print("-"*52)
print(f"{'1 greedy (baseline)':22s} {30.8:11.1f}% {38.0:13.1f}%")
for lbl,how in (("1 sample",'first'),("majority vote",'vote'),
                ("VERIFIER-SELECTED",'verify'),("best-of-k (ceiling)",'best')):
    v=[100*sum(rule(i,w,how) for i in g)/n for w in ('weak','strong')]
    print(f"{lbl:22s} {v[0]:11.1f}% {v[1]:13.1f}%")
ww=100*sum(rule(i,'weak','verify') for i in g)/n
sw=100*sum(rule(i,'strong','verify') for i in g)/n
ws=100*sum(em[i]['weak-strong'] for i in g)/n
ss=100*sum(em[i]['strong-strong'] for i in g)/n
o_old=100*sum(max(em[i][c] for c in COMBOS) for i in g)/n
o_new=100*sum(max(rule(i,'weak','verify'),em[i]['weak-strong'],
                  rule(i,'strong','verify'),em[i]['strong-strong']) for i in g)/n
print(f"\n{'':16s} {'before':>10s} {'after':>10s}")
print(f"{'best-fixed':16s} {51.2:9.1f}% {max(ww,ws,sw,ss):9.1f}%")
print(f"{'oracle':16s} {o_old:9.1f}% {o_new:9.1f}%")
print(f"\n  AgentTTS reported 0.74 on HotpotQA")
