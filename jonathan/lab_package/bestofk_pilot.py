#!/usr/bin/env python3
# bestofk_pilot.py v1.0 — best-of-K weak-answer sampling pilot (23 Sep 2026)
# Lever: test-time scaling of the WEAK answerer. For each pilot instance (val only),
# generate K answers on the SAME strong-evidence block the grid used:
#   sample 0: greedy (do_sample=False) — must reproduce the grid's stored answer
#             (printed sanity check against qa_weak.jsonl)
#   samples 1..K-1: do_sample=True, temperature T
# Prompts are BYTE-IDENTICAL to the pipeline by construction: this script imports
# run_labels4 and reuses its qa_prompt / parse_selection / build_block / load_model /
# SYS ("You are a helpful assistant.") / QA_OUT=128 / MAXLEN — only the sampling
# kwargs differ, and only for samples >= 1.
# Output (resume-safe, run_labels4.Store format): k="idx|s{j}", v=answer text.
# Scoring/verification happens in a separate step — this script only generates.
import os, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import run_labels4 as R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hdd/jonathan/data/tts_test.json")
    ap.add_argument("--retr", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_strong.jsonl")
    ap.add_argument("--qa_weak", default="/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--n", type=int, default=500, help="pilot size (first N of the split)")
    ap.add_argument("--k", type=int, default=4, help="total samples incl. greedy #0")
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--out", default="/mnt/hdd/jonathan/runs/verifier_scores/bestofk_pilot.jsonl")
    a = ap.parse_args()

    data = json.load(open(a.data))
    retr = {}
    for l in open(a.retr):
        r = json.loads(l); retr[int(str(r["k"]).split("|")[0])] = r["v"]
    grid_weak = {}
    for l in open(a.qa_weak):
        r = json.loads(l); i, br = r["k"].split("|")
        if br == "strong": grid_weak[int(i)] = (r.get("v") or "").strip()
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    ids = [i for i in sp[a.split] if i in retr][: a.n]
    print(f"[cfg] split={a.split} n={len(ids)} k={a.k} temp={a.temp} out={a.out}", flush=True)

    m, tok = R.load_model("weak", stage="qa")
    SYS = "You are a helpful assistant."

    def gen(user, sample):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids_ = tok(text, return_tensors="pt", truncation=True, max_length=R.MAXLEN).to(m.device)
        kw = dict(max_new_tokens=R.QA_OUT, pad_token_id=tok.eos_token_id)
        if sample: kw.update(do_sample=True, temperature=a.temp)
        else:      kw.update(do_sample=False)
        with torch.no_grad():
            out = m.generate(**ids_, **kw)
        return tok.decode(out[0][ids_.input_ids.shape[1]:], skip_special_tokens=True)

    store = R.Store(a.out)
    t0 = time.time(); match = 0; checked = 0; done = 0
    for i in ids:
        inst = data[i]
        sel = R.parse_selection(retr[i], len(inst["ctxs_candidate"]))
        block = R.build_block(inst, sel)
        user = R.qa_prompt(inst, block)
        for j in range(a.k):
            key = f"{i}|s{j}"
            if key in store: continue
            store.put(key, gen(user, sample=(j >= 1)))
        g = store[f"{i}|s0"].strip()
        if i in grid_weak:
            checked += 1; match += (g == grid_weak[i])
        done += 1
        if done % 25 == 0:
            store.flush()
            print(f"  ... {done}/{len(ids)}  greedy==grid: {match}/{checked}  ({time.time()-t0:.0f}s)", flush=True)
    store.flush()
    print(f"[done] {done} instances x {a.k} samples -> {a.out}", flush=True)
    print(f"[sanity] greedy sample reproduces the grid answer on {match}/{checked} "
          f"({100*match/max(checked,1):.1f}%) — expect near 100%; large deviation = prompt mismatch, stop and investigate.", flush=True)

if __name__ == "__main__":
    main()
