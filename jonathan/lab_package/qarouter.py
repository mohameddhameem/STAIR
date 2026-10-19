import os,json,argparse,random,math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, torch.nn as nn
from transformers import AutoTokenizer, AutoModel
P=argparse.ArgumentParser()
P.add_argument('--run',default='/mnt/hdd/jonathan/runs/full_run')
P.add_argument('--splits',default='splits.json')
P.add_argument('--model',default='Qwen/Qwen3-1.7B')
P.add_argument('--epochs',type=int,default=2)
P.add_argument('--bs',type=int,default=8)
P.add_argument('--lr',type=float,default=1e-5)
P.add_argument('--maxlen',type=int,default=1024)
P.add_argument('--out',default='qarouter_scores.jsonl')
a=P.parse_args()

# ---- labels: for each instance, did WEAK QA succeed where STRONG did not, etc.
em={}
for l in open(f"{a.run}/combo_results_full.jsonl"):
    r=json.loads(l); em.setdefault(r['idx'],{})[r['combo']]=r['em_official']
# QA-stage decision, holding retrieval fixed at STRONG
lab={i:d for i,d in em.items() if len(d)==4}
# y = 1 means the WEAK QA model suffices (strong-weak is right)
Y={i:(1 if d['strong-weak']==1 else 0) for i,d in lab.items()}
DIS={i for i,d in lab.items() if d['strong-weak']!=d['strong-strong']}
print(f"{len(lab)} instances | weak-QA correct {sum(Y.values())} "
      f"| disagreements {len(DIS)} ({100*len(DIS)/len(lab):.1f}%)")

retr={}
for l in open(f"{a.run}/retr_strong.jsonl"):
    r=json.loads(l); retr[int(str(r['k']).split('|')[0])]=r['v']

S=json.load(open(a.splits))['splits']
tr=[i for i in S['train'] if i in lab and i in retr
    and retr[i] and 'OOM' not in retr[i]]   # 17Aug: no empty-input label noise in train
va=[i for i in S['val']   if i in lab and i in retr]
te=[i for i in S['test']  if i in lab and i in retr]
print(f"train {len(tr)} | val {len(va)} | test {len(te)}")

def text(i):
    t=retr[i]
    return t[:4000] if t and 'OOM' not in t else ""

tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
enc=AutoModel.from_pretrained(a.model,torch_dtype=torch.bfloat16).cuda()
enc.gradient_checkpointing_enable()
head=nn.Linear(enc.config.hidden_size,2).cuda().to(torch.bfloat16)
opt=torch.optim.AdamW(list(enc.parameters())+list(head.parameters()),lr=a.lr)
lossf=nn.CrossEntropyLoss()

def batch(ids):
    xs=[text(i) for i in ids]
    b=tok(xs,return_tensors='pt',truncation=True,max_length=a.maxlen,padding=True).to('cuda')
    h=enc(**b).last_hidden_state
    mask=b['attention_mask'].unsqueeze(-1).to(h.dtype)
    pooled=(h*mask).sum(1)/mask.sum(1)
    return head(pooled)

def auc(pos,neg):
    if not pos or not neg: return float('nan')
    xs=sorted([(p,1) for p in pos]+[(p,0) for p in neg])
    r={}; i=0
    while i<len(xs):
        j=i
        while j<len(xs) and xs[j][0]==xs[i][0]: j+=1
        for k in range(i,j): r[k]=(i+j+1)/2
        i=j
    s=sum(r[k] for k,(p,y) in enumerate(xs) if y==1)
    return (s-len(pos)*(len(pos)+1)/2)/(len(pos)*len(neg))

def evaluate(ids,tag):
    enc.eval(); sc={}
    with torch.no_grad():
        for s in range(0,len(ids),a.bs):
            ch=ids[s:s+a.bs]
            p=torch.softmax(batch(ch).float(),-1)[:,1]
            for i,v in zip(ch,p.tolist()): sc[i]=v
    enc.train()
    pos=[sc[i] for i in ids if Y[i]==1]; neg=[sc[i] for i in ids if Y[i]==0]
    A=auc(pos,neg)
    d=[i for i in ids if i in DIS]
    Ad=auc([sc[i] for i in d if Y[i]==1],[sc[i] for i in d if Y[i]==0])
    print(f"  {tag}: AUC {A:.3f} | AUC on disagreements only {Ad:.3f} (n={len(d)})",flush=True)
    return sc,A

def _pnorm():
    with torch.no_grad():
        return sum(p.float().norm().item() for p in enc.parameters())
print(f'param-norm BEFORE training: {_pnorm():.4f}', flush=True)
random.seed(0)
for ep in range(a.epochs):
    random.shuffle(tr); tot=0
    for s in range(0,len(tr),a.bs):
        ch=tr[s:s+a.bs]
        logits=batch(ch)
        y=torch.tensor([Y[i] for i in ch]).cuda()
        loss=lossf(logits.float(),y)
        loss.backward(); opt.step(); opt.zero_grad(); tot+=loss.item()
        if (s//a.bs)%100==0: print(f"ep{ep} step{s//a.bs} loss {loss.item():.4f}",flush=True)
    print(f"epoch {ep} mean loss {tot/(len(tr)/a.bs):.4f}",flush=True)
    evaluate(va,f"val ep{ep}")

print(f'param-norm AFTER training:  {_pnorm():.4f}', flush=True)
sc,_=evaluate(te,"TEST")
with open(a.out,'w') as f:
    for i,v in sc.items():
        f.write(json.dumps({"idx":i,"p_weak":v,"y":Y[i],"disagree":i in DIS})+"\n")
print("wrote",a.out)
