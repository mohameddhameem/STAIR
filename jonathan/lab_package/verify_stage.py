import os, re, json, string, argparse
from collections import defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch, run_labels as R
from sklearn.metrics import roc_auc_score
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,WS,SW,SS=0,1,2,3
COST={"weak-weak":2.0,"weak-strong":5.7,"strong-weak":5.7,"strong-strong":9.4}
SYS="You are a careful evaluator. Judge only from the references provided."
TPL=("References:\n{refs}\n\nQuestion: {q}\n\nProposed answer: {a}\n\n"
     "Is this proposed answer correct and supported by the references above? "
     "Reply with exactly one word: Yes or No.")
def norm(s):
    s=''.join(c for c in s.lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
ap=argparse.ArgumentParser()
ap.add_argument('--out',default='/mnt/hdd/jonathan/runs/full_run_qwen3')
ap.add_argument('--data',default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--splits',default='splits.json')
ap.add_argument('--model',default='Qwen/Qwen3-8B')
ap.add_argument('--max-refs',type=int,default=8)
ap.add_argument('--save',default='verify_qa_perretr.jsonl')
ap.add_argument('--retr-scores',default='verifretr_full_run_qwen3.jsonl')
a=ap.parse_args()
data=json.load(open(a.data)); te=json.load(open(a.splits))["splits"][os.environ.get("SLR_SPLIT","test")]
em=defaultdict(dict)
for l in open(os.path.join(a.out,'combo_results_full.jsonl')):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:[d[c] for c in COMBOS] for i,d in em.items() if len(d)==4}
retr={}
for w in ('weak','strong'):
    d={}
    for l in open(os.path.join(a.out,f'retr_{w}.jsonl')):
        r=json.loads(l); d[int(str(r['k']).split('|')[0])]=r['v']
    retr[w]=d
qa={}
for w in ('weak','strong'):
    d={}
    for l in open(os.path.join(a.out,f'qa_{w}.jsonl')):
        r=json.loads(l); d[str(r['k'])]=r['v']
    qa[w]=d
ids=[i for i in te if i in em and i in retr['weak'] and i in retr['strong']]
done={}
if os.path.exists(a.save):
    for l in open(a.save):
        try: r=json.loads(l); done[r['idx']]=r
        except Exception: pass
    print(f"resuming, {len(done)} judged")
todo=[i for i in ids if i not in done]
print(f"instances to judge: {len(todo)} (4 calls each)")
if todo:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading {a.model} ...")
    tok=AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    idY=tok.encode("Yes",add_special_tokens=False)[0]
    idN=tok.encode("No",add_special_tokens=False)[0]
    def p_yes(q,refs,ans):
        msgs=[{"role":"system","content":SYS},{"role":"user","content":TPL.format(refs=refs,q=q,a=ans)}]
        try: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=tok(t,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
        with torch.no_grad():
            o=m.generate(**enc,max_new_tokens=1,do_sample=False,output_scores=True,
                         return_dict_in_generate=True,pad_token_id=tok.pad_token_id)
        lg=o.scores[0][0]
        return float(torch.softmax(torch.tensor([lg[idY],lg[idN]]).float(),0)[0])
    sink=open(a.save,'a')
    for n,i in enumerate(todo):
        inst=dict(data[i]); inst['ctxs_candidate']=R.trim_pool(inst); pool=inst['ctxs_candidate']
        rec={"idx":i}
        for rs in ('weak','strong'):
            picks=[p for p in R.parse_selection(retr[rs][i],len(pool)) if 1<=p<=len(pool)][:a.max_refs]
            refs="\n".join(f"[{k+1}] {pool[p-1]}" for k,p in enumerate(picks)) or "(none)"
            for qs in ('weak','strong'):
                ans=qa[qs].get(f"{i}|{rs}")
                rec[f"{rs}_{qs}"]=p_yes(inst['question'],refs,ans) if ans is not None else None
        sink.write(json.dumps(rec)+"\n")
        if (n+1)%100==0: sink.flush(); print(f"  {n+1}/{len(todo)} ...",flush=True)
    sink.close()
S={}
for l in open(a.save):
    r=json.loads(l); S[r['idx']]=r
RS={}
if os.path.exists(a.retr_scores):
    for l in open(a.retr_scores):
        r=json.loads(l); RS[r['idx']]=r['margin']
g=[i for i in ids if i in S and all(S[i].get(k) is not None for k in
   ('weak_weak','weak_strong','strong_weak','strong_strong'))]
print(f"\nscored {len(g)} instances\n")
allte=[i for i in te if i in em]
def report(name,pick):
    e=100*np.mean([em[i][pick[i]] if i in pick else em[i][SS] for i in allte])
    c=100*np.mean([COST[COMBOS[pick[i]]] if i in pick else COST["strong-strong"] for i in allte])/COST["strong-strong"]
    print(f"  {name:38s} EM {e:5.1f}%   cost {c:5.1f}%")
    return e
bfix=max(np.mean([em[i][c] for i in allte]) for c in range(4))*100
orc=np.mean([max(em[i]) for i in allte])*100
print(f"  {'always strong-strong (best fixed)':38s} EM {bfix:5.1f}%   cost 100.0%")
# QA verifier only, retriever held strong
report("QA verifier @ strong retrieval", {i:(SW if S[i]['strong_weak']>S[i]['strong_strong'] else SS) for i in g})
# retrieval verifier only, QA held strong
if RS: report("retrieval verifier @ strong QA", {i:(WS if RS.get(i,-9)>0 else SS) for i in g if i in RS})
# BOTH stages
if RS:
    both={}
    for i in g:
        if i not in RS: continue
        rs = 'weak' if RS[i]>0 else 'strong'
        qs = 'weak' if S[i][f'{rs}_weak']>S[i][f'{rs}_strong'] else 'strong'
        both[i]=COMBOS.index(f"{rs}-{qs}")
    report("BOTH stages verified", both)
print(f"  {'oracle (upper bound)':38s} EM {orc:5.1f}%   cost   \u2014")
print(f"\n  headroom available: +{orc-bfix:.1f} EM")
