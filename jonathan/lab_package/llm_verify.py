import os, re, json, string, argparse
import numpy as np
from collections import defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, run_labels as R
from sklearn.metrics import roc_auc_score
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,SS=0,3
CW,CS=2.0,9.4
SYS="You are a careful evaluator. Judge only from the references provided."
TPL="""Question: {q}

References:
{refs}

Candidate A: {a}
Candidate B: {b}

Which candidate answer is better supported by the references above? Reply with exactly one letter, A or B."""
def norm(s):
    s=''.join(c for c in s.lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
ap=argparse.ArgumentParser()
ap.add_argument('--out',default='/mnt/hdd/jonathan/runs/full_run_qwen3')
ap.add_argument('--data',default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--splits',default='splits.json')
ap.add_argument('--model',default='Qwen/Qwen3-1.7B')
ap.add_argument('--max-refs',type=int,default=10)
ap.add_argument('--n',type=int,default=0)
ap.add_argument('--save',default=None)
a=ap.parse_args(); save=a.save or f"llmverify_{a.model.split('/')[-1]}.jsonl"
data=json.load(open(a.data)); te=json.load(open(a.splits))["splits"]["test"]
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
dis=[i for i in te if i in em and f"{i}|weak" in qa['weak'] and f"{i}|strong" in qa['strong']
     and norm(qa['weak'][f"{i}|weak"])!=norm(qa['strong'][f"{i}|strong"])]
if a.n: dis=dis[:a.n]
print(f"disagreements to judge: {len(dis)}")
done={}
if os.path.exists(save):
    for l in open(save):
        try: r=json.loads(l); done[r['idx']]=r
        except Exception: pass
    print(f"  resuming, {len(done)} judged")
todo=[i for i in dis if i not in done]
if todo:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading {a.model} ...")
    tok=AutoTokenizer.from_pretrained(a.model)
    m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    idA=tok.encode("A",add_special_tokens=False)[0]; idB=tok.encode("B",add_special_tokens=False)[0]
    def judge(q,refs,x,y):
        msgs=[{"role":"system","content":SYS},
              {"role":"user","content":TPL.format(q=q,refs=refs,a=x,b=y)}]
        try: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=tok(t,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
        with torch.no_grad():
            o=m.generate(**enc,max_new_tokens=1,do_sample=False,output_scores=True,
                         return_dict_in_generate=True,pad_token_id=tok.eos_token_id)
        lg=o.scores[0][0]
        return float(torch.softmax(torch.tensor([lg[idA],lg[idB]]).float(),0)[0])
    sink=open(save,'a')
    for n,i in enumerate(todo):
        inst=dict(data[i]); pool=R.trim_pool(inst)
        picks=[]
        for w in ('weak','strong'):
            picks+=[p for p in R.parse_selection(retr[w][i],len(pool)) if 1<=p<=len(pool)]
        picks=sorted(set(picks))[:a.max_refs]
        refs="\n".join(f"[{k+1}] {pool[p-1]}" for k,p in enumerate(picks)) or "(none)"
        A=qa['weak'][f"{i}|weak"]; B=qa['strong'][f"{i}|strong"]
        p1=judge(inst['question'],refs,A,B); p2=judge(inst['question'],refs,B,A)
        sink.write(json.dumps({"idx":i,"p_weak":0.5*(p1+(1-p2)),
                               "p_weak_as_A":p1,"p_weak_as_B":1-p2,"n_refs":len(picks)})+"\n")
        if (n+1)%50==0: sink.flush(); print(f"  {n+1}/{len(todo)} ...",flush=True)
    sink.close()
res={}
for l in open(save):
    r=json.loads(l); res[r['idx']]=r
g=[i for i in dis if i in res]
p=np.array([res[i]['p_weak'] for i in g]); y=np.array([em[i][WW] for i in g])
pa=np.array([res[i]['p_weak_as_A'] for i in g]); pb=np.array([res[i]['p_weak_as_B'] for i in g])
print(f"\njudged {len(g)}   weak-is-right {y.mean():.3f}")
print(f"position bias: P(weak) first {pa.mean():.3f} second {pb.mean():.3f}  gap {abs(pa.mean()-pb.mean()):.3f}")
print(f"  orders agree: {100*np.mean((pa>0.5)==(pb>0.5)):.1f}%")
print(f"\nAUC debiased {roc_auc_score(y,p):.3f}  |  first {roc_auc_score(y,pa):.3f}  |  second {roc_auc_score(y,pb):.3f}")
print(f"  (hand-crafted features gave 0.766)")
allte=[i for i in te if i in em]
ss=np.mean([em[i][SS] for i in allte])*100
orc=np.mean([max(em[i][WW],em[i][SS]) for i in allte])*100
print(f"\nalways-strong {ss:.1f}%  |  perfect verification {orc:.1f}% ({orc-ss:+.1f} EM at {100*(CW+CS)/CS:.0f}% cost)")
print(f"\n  {'thresh':>7s} {'take weak':>10s} {'EM':>7s} {'vs always-ss':>13s} {'% headroom':>12s}")
base={i:em[i][SS] for i in allte}
for t in (0.40,0.50,0.60,0.70,0.80):
    pick={i:(em[i][WW] if q>=t else em[i][SS]) for i,q in zip(g,p)}
    e=np.mean([pick.get(i,base[i]) for i in allte])*100
    print(f"  {t:7.2f} {100*np.mean(p>=t):9.1f}% {e:6.1f}% {e-ss:+12.1f} {100*(e-ss)/max(1e-9,orc-ss):11.1f}%")
