import os, re, json, string, argparse
from collections import defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import numpy as np, torch, run_labels as R
from sklearn.metrics import roc_auc_score
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,WS,SW,SS=0,1,2,3
SYS="You are a careful evaluator. Judge only from the references provided."
TPL=("References:\n{refs}\n\nQuestion: {q}\n\n"
     "Do these references contain the information needed to answer the question? "
     "Reply with exactly one word: Yes or No.")
ap=argparse.ArgumentParser()
ap.add_argument('--out',default='/mnt/hdd/jonathan/runs/full_run_qwen3')
ap.add_argument('--data',default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--splits',default='splits.json')
ap.add_argument('--model',default='Qwen/Qwen3-8B')
ap.add_argument('--max-refs',type=int,default=8)
ap.add_argument('--n',type=int,default=0); ap.add_argument('--save',default=None)
a=ap.parse_args()
tag=os.path.basename(a.out.rstrip('/')); save=a.save or f"verifretr_{tag}.jsonl"
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
ids=[]; picks={}
for i in te:
    if i not in em or i not in retr['weak'] or i not in retr['strong']: continue
    inst=dict(data[i]); inst['ctxs_candidate']=R.trim_pool(inst); pool=inst['ctxs_candidate']
    pw=tuple(sorted(set(p for p in R.parse_selection(retr['weak'][i],len(pool)) if 1<=p<=len(pool))))
    ps=tuple(sorted(set(p for p in R.parse_selection(retr['strong'][i],len(pool)) if 1<=p<=len(pool))))
    if pw!=ps: ids.append(i); picks[i]=(pw,ps)
if a.n: ids=ids[:a.n]
print(f"pool size {len(R.trim_pool(dict(data[te[0]])))}")
print(f"instances where the retrievers DIFFER: {len(ids)}")
done={}
if os.path.exists(save):
    for l in open(save):
        try: r=json.loads(l); done[r['idx']]=r
        except Exception: pass
    print(f"  resuming, {len(done)} judged")
todo=[i for i in ids if i not in done]
if todo:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading {a.model} ...")
    tok=AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    idY=tok.encode("Yes",add_special_tokens=False)[0]
    idN=tok.encode("No",add_special_tokens=False)[0]
    def p_yes(q,refs):
        msgs=[{"role":"system","content":SYS},{"role":"user","content":TPL.format(refs=refs,q=q)}]
        try: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=tok(t,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
        with torch.no_grad():
            o=m.generate(**enc,max_new_tokens=1,do_sample=False,output_scores=True,
                         return_dict_in_generate=True,pad_token_id=tok.pad_token_id)
        lg=o.scores[0][0]
        return float(torch.softmax(torch.tensor([lg[idY],lg[idN]]).float(),0)[0])
    sink=open(save,'a')
    for n,i in enumerate(todo):
        inst=dict(data[i]); inst['ctxs_candidate']=R.trim_pool(inst); pool=inst['ctxs_candidate']
        pw,ps=picks[i]
        def refs_of(sel):
            sel=list(sel)[:a.max_refs]
            return "\n".join(f"[{k+1}] {pool[p-1]}" for k,p in enumerate(sel)) or "(none)"
        x=p_yes(inst['question'],refs_of(pw)); y2=p_yes(inst['question'],refs_of(ps))
        sink.write(json.dumps({"idx":i,"p_weak":x,"p_strong":y2,"margin":x-y2,
                               "n_weak":len(pw),"n_strong":len(ps)})+"\n")
        if (n+1)%100==0: sink.flush(); print(f"  {n+1}/{len(todo)} ...",flush=True)
    sink.close()
res={}
for l in open(save):
    r=json.loads(l); res[r['idx']]=r
g=[i for i in ids if i in res]
mar=np.array([res[i]['margin'] for i in g])
pw=np.array([res[i]['p_weak'] for i in g]); ps=np.array([res[i]['p_strong'] for i in g])
best_w=np.array([max(em[i][WW],em[i][WS]) for i in g])
best_s=np.array([max(em[i][SW],em[i][SS]) for i in g])
y=(best_w>best_s).astype(int)
print(f"\njudged {len(g)}   weak retriever strictly better on {y.mean():.3f}")
print(f"mean P(yes): weak selection {pw.mean():.3f}   strong selection {ps.mean():.3f}")
if 0<y.sum()<len(y):
    print(f"\nAUC on margin  {roc_auc_score(y,mar):.3f}")
    print("  (a-priori retrieval-choice probes sat at 0.51-0.57)")
allte=[i for i in te if i in em]
bfix=max(np.mean([em[i][c] for i in allte]) for c in range(4))*100
orc=np.mean([max(em[i]) for i in allte])*100
print(f"\nfull test split: best-fixed {bfix:.1f}%  oracle {orc:.1f}%  (+{orc-bfix:.1f})")
print(f"\n  {'margin >':>9s} {'take weak retr':>15s} {'EM':>7s} {'vs best-fixed':>14s}")
base={i:em[i][SS] for i in allte}
for t in (-0.10,0.0,0.10,0.20,0.30):
    pick={i:(em[i][WS] if q>t else em[i][SS]) for i,q in zip(g,mar)}
    e=np.mean([pick.get(i,base[i]) for i in allte])*100
    print(f"  {t:9.2f} {100*np.mean(mar>t):14.1f}% {e:6.1f}% {e-bfix:+13.1f}")
