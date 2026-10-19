import os, re, json, string, argparse
from collections import defaultdict
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, run_labels as R
COMBOS=["weak-weak","weak-strong","strong-weak","strong-strong"]
FUSE=("You have been provided with a set of responses from various open-source models to "
      "the latest user query, which is {q}. Your task is to synthesize these responses into "
      "a single, high-quality response. It is crucial to critically evaluate the information "
      "provided in these responses, recognizing that some of it may be biased or incorrect. "
      "Your response should not simply replicate the given answers but should offer a refined, "
      "accurate, and comprehensive reply to the instruction. Once again, the query is: {q}\n"
      "Responses from models:\n{cands}\n\n"
      "Answer with ONLY the exact answer phrase - no explanation, no full sentence.")
def norm(s):
    s=''.join(c for c in s.lower() if c not in set(string.punctuation))
    return ' '.join(re.sub(r'\b(a|an|the)\b',' ',s).split())
ap=argparse.ArgumentParser()
ap.add_argument('--out',default='/mnt/hdd/jonathan/runs/armC_n500')
ap.add_argument('--data',default='/mnt/hdd/jonathan/data/tts_test.json')
ap.add_argument('--model',default='unsloth/Llama-3.2-3B-Instruct')
ap.add_argument('--k',type=int,default=5); ap.add_argument('--temp',type=float,default=0.9)
ap.add_argument('--n',type=int,default=500); ap.add_argument('--save',default='fusion_qa.jsonl')
a=ap.parse_args()
data=json.load(open(a.data))
retr={}
for w in ('weak','strong'):
    d={}
    for l in open(os.path.join(a.out,f'retr_{w}.jsonl')):
        r=json.loads(l); d[int(str(r['k']).split('|')[0])]=r['v']
    retr[w]=d
done=set()
if os.path.exists(a.save):
    for l in open(a.save):
        try: done.add(json.loads(l)['key'])
        except Exception: pass
    print(f"resuming, {len(done)} done")
ids=[i for i in sorted(retr['weak']) if i<a.n]
todo=[(i,w) for i in ids for w in ('weak','strong') if f"{i}|{w}" not in done and i in retr[w]]
if todo:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    print(f"loading {a.model} ... {len(todo)} calls")
    tok=AutoTokenizer.from_pretrained(a.model)
    if tok.pad_token is None: tok.pad_token=tok.eos_token
    m=AutoModelForCausalLM.from_pretrained(a.model,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    def gen(sp,up,k=1,temp=None,maxnew=64):
        msgs=[{"role":"system","content":sp},{"role":"user","content":up}]
        t=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=tok(t,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
        kw=dict(max_new_tokens=maxnew,pad_token_id=tok.pad_token_id)
        if temp: kw.update(do_sample=True,temperature=temp,top_p=0.95,num_return_sequences=k)
        else:    kw.update(do_sample=False)
        with torch.no_grad(): o=m.generate(**enc,**kw)
        ni=enc.input_ids.shape[1]
        return [tok.decode(x[ni:],skip_special_tokens=True).strip() for x in o]
    sink=open(a.save,'a')
    for c,(i,w) in enumerate(todo):
        inst=dict(data[i]); inst['ctxs_candidate']=R.trim_pool(inst); pool=inst['ctxs_candidate']
        picks=[p for p in R.parse_selection(retr[w][i],len(pool)) if 1<=p<=len(pool)]
        block=R.build_block(inst,picks)
        cands=gen("You are a helpful assistant.",R.qa_prompt(inst,block),k=a.k,temp=a.temp)
        fused=gen("You are a helpful assistant.",
                  FUSE.format(q=inst['question'],cands="\n".join(f"- {x}" for x in cands)))[0]
        gold=[norm(x) for x in inst['answers']]
        sink.write(json.dumps({"key":f"{i}|{w}","idx":i,"src":w,"fused":fused[:80],"cands":[x[:60] for x in cands],
            "em_vote":int(__import__("collections").Counter([norm(x) for x in cands]).most_common(1)[0][0] in gold),
            "em_fused":int(norm(fused) in gold),
            "em_best":int(any(norm(x) in gold for x in cands)),
            "em_first":int(norm(cands[0]) in gold)})+"\n")
        if (c+1)%100==0: sink.flush(); print(f"  {c+1}/{len(todo)} ...",flush=True)
    sink.close()
rows=defaultdict(dict)
for l in open(a.save):
    r=json.loads(l); rows[r['idx']][r['src']]=r
em=defaultdict(dict)
for l in open(os.path.join(a.out,'combo_results_full.jsonl')):
    r=json.loads(l); em[r['idx']][r['combo']]=r['em_official']
em={i:d for i,d in em.items() if len(d)==4}
g=[i for i in rows if i in em and len(rows[i])==2]; n=len(g)
print(f"\nscored {n} instances\n")
old={c:100*sum(em[i][c] for i in g)/n for c in COMBOS}
new=dict(old)
new['weak-weak']=100*sum(rows[i]['weak']['em_fused'] for i in g)/n
new['strong-weak']=100*sum(rows[i]['strong']['em_fused'] for i in g)/n
print(f"{'configuration':16s} {'1 greedy':>10s} {'5+fusion':>10s} {'delta':>8s}")
print("-"*48)
for c in COMBOS:
    d=new[c]-old[c]; tag="" if abs(d)<1e-9 else f"{d:+7.1f}"
    print(f"{c:16s} {old[c]:9.1f}% {new[c]:9.1f}% {tag:>8s}")
print(f"  majority vote:            {100*sum(rows[i]['strong']['em_vote'] for i in g)/n:.1f}%")
print(f"\n  best-of-{a.k} on strong retrieval: {100*sum(rows[i]['strong']['em_best'] for i in g)/n:.1f}%  (fusion ceiling)")
o_old=100*sum(max(em[i][c] for c in COMBOS) for i in g)/n
o_new=100*sum(max((rows[i]['weak']['em_fused'] if c=='weak-weak' else
                   rows[i]['strong']['em_fused'] if c=='strong-weak' else em[i][c])
                  for c in COMBOS) for i in g)/n
bf_old,bf_new=max(old.values()),max(new.values())
print(f"\n{'':16s} {'before':>10s} {'after':>10s}")
print(f"{'best-fixed':16s} {bf_old:9.1f}% {bf_new:9.1f}%")
print(f"{'oracle':16s} {o_old:9.1f}% {o_new:9.1f}%")
print(f"{'headroom':16s} {o_old-bf_old:+9.1f} {o_new-bf_new:+9.1f}")
print(f"\n  AgentTTS reported 0.74 (72B retrieval + this QA method)")
