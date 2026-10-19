import os, json, time
os.environ.setdefault("HF_HOME","/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
import torch, run_labels as R
from transformers import AutoTokenizer, AutoModelForCausalLM
MODEL=os.environ.get("SLR_TERSE_MODEL","Qwen/Qwen3-1.7B")
N=int(os.environ.get("SLR_TERSE_N",60)); SAVE="terse_check.jsonl"
TERSE=("\n\nIMPORTANT: reply with ONLY the reference numbers, in the form "
       "'Selected references: [3] [17]'. Do NOT quote or repeat any reference text.")
data=json.load(open('/mnt/hdd/jonathan/data/tts_test.json'))
ids=json.load(open('splits.json'))["splits"]["test"][:N]
done=set()
if os.path.exists(SAVE):
    for l in open(SAVE):
        try: done.add(json.loads(l)['idx'])
        except Exception: pass
    print(f"resuming, {len(done)} recorded")
todo=[i for i in ids if i not in done]
if todo:
    print(f"loading {MODEL} (fp16) ... {len(todo)} to run")
    tok=AutoTokenizer.from_pretrained(MODEL)
    m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float16,
        device_map="cuda",attn_implementation="sdpa").eval()
    def gen(prompt,cap=2048):
        msgs=[{"role":"system","content":R.SYS_RETR},{"role":"user","content":prompt}]
        try: txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        except TypeError: txt=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        enc=tok(txt,return_tensors="pt",truncation=True,max_length=8192).to(m.device)
        n_in=enc.input_ids.shape[1]; t0=time.time()
        with torch.no_grad():
            o=m.generate(**enc,max_new_tokens=cap,do_sample=False,pad_token_id=tok.eos_token_id)
        return (tok.decode(o[0][n_in:],skip_special_tokens=True),
                int(o.shape[1])-n_in, time.time()-t0, n_in)
    sink=open(SAVE,'a')
    for n,i in enumerate(todo):
        inst=dict(data[i])
        inst['ctxs_candidate']=R.trim_pool(inst)      # <-- THE FIX
        pool=inst['ctxs_candidate']
        gold=set(inst.get('supporting_titles',[])); bp=R.retr_prompt(inst)
        row={"idx":i,"need":len(gold)}
        for tag,prompt in (("baseline",bp),("terse",bp+TERSE)):
            txt,nout,secs,n_in=gen(prompt)
            picks=[p for p in R.parse_selection(txt,len(pool)) if 1<=p<=len(pool)]
            got={R.title_of(pool[p-1]) for p in picks}
            row[tag]={"n_in":n_in,"n_out":nout,"secs":secs,"n_sel":len(picks),
                      "hit":len(got&gold),"wf":int('Selected references' in txt)}
        sink.write(json.dumps(row)+"\n"); sink.flush()
        if (n+1)%10==0: print(f"  {n+1}/{len(todo)} ...",flush=True)
    sink.close()
rows=[json.loads(l) for l in open(SAVE)]
print(f"\nmean prompt tokens: {sum(r['baseline']['n_in'] for r in rows)/len(rows):.0f}  (should be ~4100, not ~13800)")
print(f"\n{'condition':10s} {'mean_out':>9s} {'mean_s':>7s} {'med_sel':>8s} {'gold_recall':>12s} {'well-formed':>12s}")
print("-"*64)
for tag in ("baseline","terse"):
    s=sorted(r[tag]['n_sel'] for r in rows)
    print(f"{tag:10s} {sum(r[tag]['n_out'] for r in rows)/len(rows):9.1f} "
          f"{sum(r[tag]['secs'] for r in rows)/len(rows):7.2f} {s[len(s)//2]:8d} "
          f"{100*sum(r[tag]['hit'] for r in rows)/max(1,sum(r['need'] for r in rows)):11.1f}% "
          f"{100*sum(r[tag]['wf'] for r in rows)/len(rows):11.1f}%")
ob=sum(r['baseline']['n_out'] for r in rows)/len(rows); ot=sum(r['terse']['n_out'] for r in rows)/len(rows)
rb=sum(r['baseline']['hit'] for r in rows)/max(1,sum(r['need'] for r in rows))
rt=sum(r['terse']['hit'] for r in rows)/max(1,sum(r['need'] for r in rows))
print(f"\ntokens {ob/ot:.2f}x shorter   gold recall {100*(rt-rb):+.1f} pts")
