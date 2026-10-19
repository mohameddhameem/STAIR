import os,json,argparse,random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, torch.nn as nn
from collections import defaultdict
from transformers import AutoTokenizer, AutoModel
P=argparse.ArgumentParser()
P.add_argument('--run',default='/mnt/hdd/jonathan/runs/full_run')
P.add_argument('--splits',default='splits.json')
P.add_argument('--model',default='Qwen/Qwen3-1.7B')
P.add_argument('--epochs',type=int,default=2)
P.add_argument('--bs',type=int,default=4)
P.add_argument('--G',type=int,default=8)
P.add_argument('--lr',type=float,default=1e-5)
P.add_argument('--alpha',type=float,default=0.9)
P.add_argument('--ent_coef',type=float,default=0.02)
P.add_argument('--maxlen',type=int,default=1024)
P.add_argument('--save',default='grpo_lab.jsonl')
a=P.parse_args()
CW,CS=3.0,115.0
B=a.alpha*CS
em=defaultdict(dict)
for l in open(f"{a.run}/combo_results_full.jsonl"):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:d for i,d in em.items() if len(d)==4}
Q={i:{0:d['strong-weak'],1:d['strong-strong']} for i,d in em.items()}
DIS={i for i,d in em.items() if d['strong-weak']!=d['strong-strong']}
retr={}
for l in open(f"{a.run}/retr_strong.jsonl"):
    r=json.loads(l); retr[int(str(r['k']).split('|')[0])]=r['v']
S=json.load(open(a.splits))['splits']
have=lambda i: i in Q and i in retr and retr[i] and 'OOM' not in retr[i]
tr=[i for i in S['train'] if have(i)]; va=[i for i in S['val'] if have(i)]; te=[i for i in S['test'] if have(i)]
print(f"train {len(tr)} | val {len(va)} | test {len(te)} | disagreements {len(DIS&set(Q))}",flush=True)
def rew(i,act): return Q[i][act] - max(0.0,(CW if act==0 else CS)/B - 1.0)
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
enc=AutoModel.from_pretrained(a.model,torch_dtype=torch.bfloat16).cuda()
enc.gradient_checkpointing_enable()
head=nn.Linear(enc.config.hidden_size,2).cuda().to(torch.bfloat16)
opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()),lr=a.lr)
def logits(ids):
    xs=[retr[i][:4000] for i in ids]
    b=tok(xs,return_tensors='pt',truncation=True,max_length=a.maxlen,padding=True).to('cuda')
    h=enc(**b).last_hidden_state
    m=b['attention_mask'].unsqueeze(-1).to(h.dtype)
    return head((h*m).sum(1)/m.sum(1))
def auc(p,q):
    if not p or not q: return float('nan')
    xs=sorted([(x,1) for x in p]+[(x,0) for x in q]); r={}; i=0
    while i<len(xs):
        j=i
        while j<len(xs) and xs[j][0]==xs[i][0]: j+=1
        for k in range(i,j): r[k]=(i+j+1)/2
        i=j
    s=sum(r[k] for k,(x,y) in enumerate(xs) if y==1)
    return (s-len(p)*(len(p)+1)/2)/(len(p)*len(q))
def evaluate(ids,tag):
    enc.eval(); sc={}
    with torch.no_grad():
        for s in range(0,len(ids),a.bs):
            ch=ids[s:s+a.bs]
            p=torch.softmax(logits(ch).float(),-1)[:,0]
            for i,v in zip(ch,p.tolist()): sc[i]=v
    enc.train()
    pw=sum(1 for i in ids if sc[i]>=0.5)/len(ids)
    acc=sum(Q[i][0 if sc[i]>=0.5 else 1] for i in ids)/len(ids)
    cost=sum(CW if sc[i]>=0.5 else CS for i in ids)/len(ids)
    d=[i for i in ids if i in DIS]
    A=auc([sc[i] for i in d if Q[i][0]==1],[sc[i] for i in d if Q[i][0]==0])
    print(f"  {tag}: routed weak {100*pw:5.1f}% | EM {100*acc:5.1f}% | cost {cost:6.1f} "
          f"| AUC(disagree) {A:.3f} n={len(d)}",flush=True)
    return sc
print(f"  always-weak   EM {100*sum(Q[i][0] for i in te)/len(te):.1f}% cost {CW}")
print(f"  always-strong EM {100*sum(Q[i][1] for i in te)/len(te):.1f}% cost {CS}",flush=True)
random.seed(0)
for ep in range(a.epochs):
    random.shuffle(tr); tot=0.0; nb=0; skip=0
    for s in range(0,len(tr),a.bs):
        ch=tr[s:s+a.bs]
        lg=logits(ch); logp=torch.log_softmax(lg.float(),-1)
        loss=0.0; used=0
        for bi,i in enumerate(ch):
            probs=logp[bi].exp().detach()
            # forced exploration: half the group sampled uniformly, so the
            # group always contains both actions and the advantage is defined
            nh=a.G//2
            acts=torch.cat([torch.multinomial(probs,a.G-nh,replacement=True),
                            torch.randint(0,2,(nh,),device=probs.device)])
            rs=torch.tensor([rew(i,int(x)) for x in acts],device='cuda',dtype=torch.float32)
            sd=rs.std()
            if sd < 1e-4: skip+=1; continue
            adv=(rs-rs.mean())/sd
            ent=-(logp[bi].exp()*logp[bi]).sum()
            loss=loss-(adv*logp[bi][acts]).mean()-a.ent_coef*ent; used+=1
        if used==0: continue
        loss=loss/used
        loss.backward(); opt.step(); opt.zero_grad()
        tot+=float(loss); nb+=1
        if nb%200==0: print(f"ep{ep} step{nb} loss {float(loss):.4f} skipped {skip}",flush=True)
    print(f"epoch {ep} mean loss {tot/max(1,nb):.4f} | degenerate groups skipped {skip}",flush=True)
    evaluate(va,f"val ep{ep}")
sc=evaluate(te,"TEST")
with open(a.save,'w') as f:
    for i,v in sc.items():
        f.write(json.dumps({"idx":i,"p_weak":v,"q_weak":Q[i][0],"q_strong":Q[i][1],"disagree":i in DIS})+"\n")
print("wrote",a.save)
