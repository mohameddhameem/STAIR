import os, re, json, string, argparse
import numpy as np
from collections import defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, run_labels as R
from sklearn.metrics import roc_auc_score
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]; WW,SS=0,3
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
ap.add_argument('--max-refs',type=int,default=10)
ap.add_argument('--n',type=int,default=0)
ap.add_argument('--save',default=None)
a=ap.parse_args(); save=a.save or f"verify2_{a.model.split('/')[-1]}.jsonl"
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
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    idY=tok.encode("Yes",add_special_tokens=False)[0]
    idN=tok.encode("No", add_special_tokens=False)[0]
    def p_yes(q,refs,ans):
        msgs=[{"role":"system","content":SYS},
              {"role":"user","content":TPL.format(refs=refs,q=q,a=ans)}]
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
        picks=[]
        for w in ('weak','strong'):
            picks+=[p for p in R.parse_selection(retr[w][i],len(pool)) if 1<=p<=len(pool)]
        picks=sorted(set(picks))[:a.max_refs]
        refs="\n".join(f"[{k+1}] {pool[p-1]}" for k,p in enumerate(picks)) or "(none)"
        A=qa['weak'][f"{i}|weak"]; B=qa['strong'][f"{i}|strong"]
        pw=p_yes(inst['question'],refs,A); ps=p_yes(inst['question'],refs,B)
        sink.write(json.dumps({"idx":i,"p_weak":pw,"p_strong":ps,"margin":pw-ps})+"\n")
        if (n+1)%100==0: sink.flush(); print(f"  {n+1}/{len(todo)} ...",flush=True)
    sink.close()
res={}
for l in open(save):
    r=json.loads(l); res[r['idx']]=r
g=[i for i in dis if i in res]
mar=np.array([res[i]['margin'] for i in g]); y=np.array([em[i][WW] for i in g])
pw=np.array([res[i]['p_weak'] for i in g]); ps=np.array([res[i]['p_strong'] for i in g])
print(f"\njudged {len(g)}   weak-is-right {y.mean():.3f}")
print(f"mean P(yes): weak answer {pw.mean():.3f}  strong answer {ps.mean():.3f}")
print(f"\nAUC on margin  {roc_auc_score(y,mar):.3f}")
print(f"AUC on P(weak) {roc_auc_score(y,pw):.3f}")
print("  (hand-crafted features gave 0.766; A/B judge gave 0.540)")
allte=[i for i in te if i in em]
ss=np.mean([em[i][SS] for i in allte])*100
orc=np.mean([max(em[i][WW],em[i][SS]) for i in allte])*100
print(f"\nalways-strong {ss:.1f}%   perfect verification {orc:.1f}% ({orc-ss:+.1f})")
base={i:em[i][SS] for i in allte}
print(f"\n  {'margin >':>9s} {'take weak':>10s} {'EM':>7s} {'vs always-ss':>13s} {'% headroom':>12s}")
for t in (0.0,0.1,0.2,0.3,0.5):
    pick={i:(em[i][WW] if q>t else em[i][SS]) for i,q in zip(g,mar)}
    e=np.mean([pick.get(i,base[i]) for i in allte])*100
    print(f"  {t:9.2f} {100*np.mean(mar>t):9.1f}% {e:6.1f}% {e-ss:+12.1f} {100*(e-ss)/max(1e-9,orc-ss):11.1f}%")
