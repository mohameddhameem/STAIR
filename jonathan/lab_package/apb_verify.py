import os,json,argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
a=argparse.ArgumentParser()
a.add_argument('--split',default='hotpotqa')
a.add_argument('--root',default='/mnt/hdd/jonathan/data/AgentProcessBench/data/AgentProcessBench')
a.add_argument('--model',default='Qwen/Qwen3-8B')
a.add_argument('--cap',type=int,default=600)
a.add_argument('--save',default=None)
a=a.parse_args()
save=a.save or f"apb_verify_{a.split}.jsonl"
SYS="You are a careful evaluator of an AI agent's reasoning steps."
TPL=("Question: {q}\n\nSteps the agent has taken so far:\n{hist}\n\n"
     "Next step by the agent:\n{step}\n\n"
     "Is this next step correct and useful for answering the question? "
     "Reply with exactly one word: Yes or No.")
def fmt(m,cap):
    if m.get('tool_calls'):
        out=[]
        for tc in m['tool_calls']:
            try: out.append(json.dumps(json.loads(tc['function']['arguments'])))
            except Exception: out.append(str(tc['function'].get('arguments'))[:200])
        return "SEARCH: "+" ; ".join(out)[:cap]
    if m['role']=='tool': return "RESULT: "+(m.get('content') or '')[:cap]
    return (m.get('content') or '')[:cap]
recs=[json.loads(l) for l in open(f"{a.root}/{a.split}.jsonl")]
items=[]
for ri,r in enumerate(recs):
    for k,lab in r['step_labels'].items():
        i=int(k)
        if i>=len(r['messages']): continue
        hist="\n".join(f"  {fmt(m,a.cap)}" for m in r['messages'][2:i] if m['role'] in ('assistant','tool')) or "  (none)"
        items.append({"rec":ri,"idx":i,"label":lab,
                      "q":r['question'],"hist":hist,"step":fmt(r['messages'][i],a.cap)})
done=set()
if os.path.exists(save):
    for l in open(save):
        try:
            x=json.loads(l); done.add((x['rec'],x['idx']))
        except Exception: pass
todo=[x for x in items if (x['rec'],x['idx']) not in done]
print(f"{len(items)} labelled steps, {len(todo)} to score",flush=True)
tok=AutoTokenizer.from_pretrained(a.model)
if tok.pad_token is None: tok.pad_token=tok.eos_token
m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
    device_map="auto",attn_implementation="sdpa").eval()
idY=tok.encode("Yes",add_special_tokens=False)[0]
idN=tok.encode("No",add_special_tokens=False)[0]
s=open(save,'a')
for n,x in enumerate(todo):
    msgs=[{"role":"system","content":SYS},
          {"role":"user","content":TPL.format(q=x['q'],hist=x['hist'],step=x['step'])}]
    try: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
    except TypeError: t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
    enc=tok(t,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
    with torch.no_grad():
        o=m.generate(**enc,max_new_tokens=1,do_sample=False,output_scores=True,
                     return_dict_in_generate=True,pad_token_id=tok.pad_token_id)
    lg=o.scores[0][0]
    p=float(torch.softmax(torch.tensor([lg[idY],lg[idN]]).float(),0)[0])
    s.write(json.dumps({"rec":x['rec'],"idx":x['idx'],"label":x['label'],"p_yes":p})+"\n")
    if (n+1)%100==0: s.flush(); print(f"  {n+1}/{len(todo)}",flush=True)
s.close(); print("DONE")
