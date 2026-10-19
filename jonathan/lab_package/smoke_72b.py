#!/usr/bin/env python3
"""smoke_72b.py — can the lab card run their retrieval stage at full length?

Loads Qwen2.5-72B-Instruct-AWQ (or NF4 with --nf4), builds THEIR retrieval
prompt verbatim (their concatenation, missing newline included) on 3 real
Route-A pools, generates at their settings (T 0.9, 512 tokens, sampled), and
reports: load time, peak GPU memory, prompt tokens, seconds per instance, and
a projection for 500 instances.

Prereq: build_routeA_pools.py has written routeA_test500.json.
Run: [LAB SSH, venv, GPU FREE — check ps first]
  python3 smoke_72b.py            # AWQ attempt
  python3 smoke_72b.py --nf4     # fallback if AWQ OOMs
"""
import os, json, time, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

ap = argparse.ArgumentParser()
ap.add_argument("--nf4", action="store_true", help="load base 72B in bnb NF4 instead of AWQ")
ap.add_argument("--data", default="/mnt/hdd/jonathan/data/routeA_test500.json")
ap.add_argument("--n", type=int, default=3)
a = ap.parse_args()

MODEL = ("Qwen/Qwen2.5-72B-Instruct" if a.nf4 else "Qwen/Qwen2.5-72B-Instruct-AWQ")
SYSM = "You are a helpful assistant that return the beneficial references to the query."
USER = ("You have been provided with a query and a set of references. Your task is to "
        "select references that comprehensively cover the information needed to answer "
        "the query. Your response should include the original query and a list of "
        "selected references in the following format: 'Query: [query]\nSelected "
        "references: [reference number] [reference details]'.Here is the query and "
        "references:\n {qr}.")

def their_block(ex):
    # their exact concatenation, archon_hotpotqa.py:572-574 — NO newline before "References:"
    q = "Query: " + ex["question"] + "References: "
    for k, ref in enumerate(ex["ctxs_candidate"]):
        q += f"\n[{k+1}] " + ref
    return q

data = json.load(open(a.data))[: a.n]
print(f"loading {MODEL} ...", flush=True)
t0 = time.time()
kw = dict(torch_dtype="auto", device_map="cuda", attn_implementation="sdpa")
if a.nf4:
    kw["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, **kw).eval()
print(f"loaded in {time.time()-t0:.0f}s | "
      f"alloc {torch.cuda.memory_allocated()/2**30:.1f} GiB", flush=True)

times = []
for n, ex in enumerate(data):
    msgs = [{"role": "system", "content": SYSM},
            {"role": "user", "content": USER.format(qr=their_block(ex))}]
    txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(txt, return_tensors="pt", truncation=True).to(model.device)
    ntok = enc.input_ids.shape[1]
    torch.cuda.reset_peak_memory_stats()
    t1 = time.time()
    with torch.no_grad():
        out = model.generate(**enc, max_new_tokens=512, do_sample=True,
                             temperature=0.9, pad_token_id=tok.eos_token_id)
    dt = time.time() - t1
    times.append(dt)
    peak = torch.cuda.max_memory_allocated() / 2**30
    gen = tok.decode(out[0][ntok:], skip_special_tokens=True)
    print(f"[{n}] prompt {ntok} tok | {dt:.1f}s | peak {peak:.1f} GiB | "
          f"out head: {gen[:90]!r}", flush=True)

avg = sum(times) / len(times)
print(f"\nmean {avg:.1f}s/instance -> 500 instances ~= {500*avg/3600:.1f} h "
      f"(retrieval stage only)")
print("Decision: peak under ~44 GiB -> AWQ viable, proceed to the full arm "
      "runner. OOM or >45 GiB -> rerun with --nf4 and record the deviation "
      "in the ledger (same NF4 family as their answering models).")
