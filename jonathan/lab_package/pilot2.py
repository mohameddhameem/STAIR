#!/usr/bin/env python3
# pilot2.py v1.0 — the two remaining escalation pilots (23 Sep 2026), VAL-ONLY
# Pool for both: the champion's ESCALATED instances (greedy weak p_supported < 1.0).
# --mode union    : strong answerer, greedy, evidence = UNION of weak+strong selections
#                   (weak-only gold passages return; attacks retrieval-bound failures).
#                   Output k="idx|un".
# --mode strongbok: strong answerer, 3 temperature samples (temp 1.0) on the STANDARD
#                   strong evidence (greedy already exists = the grid's SS answer).
#                   Output k="idx|t1..t3". Verification of these happens separately.
# Prompts byte-identical to the pipeline (imports run_labels4: qa_prompt, parse_selection,
# build_block, load_model, QA_OUT, MAXLEN, same system string); only the evidence set
# (union mode) or sampling kwargs (strongbok mode) differ.
import os, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import run_labels4 as R

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["union", "strongbok"])
    ap.add_argument("--data", default="/mnt/hdd/jonathan/data/tts_test.json")
    ap.add_argument("--retr_strong", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_strong.jsonl")
    ap.add_argument("--retr_weak", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_weak.jsonl")
    ap.add_argument("--ver", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--keep_gate", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--temp", type=float, default=1.0)
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or f"/mnt/hdd/jonathan/runs/verifier_scores/pilot_{a.mode}.jsonl"

    data = json.load(open(a.data))
    def rmap(p):
        m = {}
        for l in open(p):
            r = json.loads(l); m[int(str(r["k"]).split("|")[0])] = r["v"]
        return m
    rs, rw = rmap(a.retr_strong), rmap(a.retr_weak)
    ver = {json.loads(l)["idx"]: json.loads(l)["p_supported"] for l in open(a.ver)}
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    ids = [i for i in sp[a.split] if i in rs and i in rw and ver.get(i, 1.0) < a.keep_gate][: a.n]
    print(f"[cfg] mode={a.mode} escalated-pool n={len(ids)} out={out}", flush=True)

    m, tok = R.load_model("strong", dtype=a.dtype, stage="qa")
    SYS = "You are a helpful assistant."

    def gen(user, sample):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids_ = tok(text, return_tensors="pt", truncation=True, max_length=R.MAXLEN).to(m.device)
        kw = dict(max_new_tokens=R.QA_OUT, pad_token_id=tok.eos_token_id)
        kw.update(dict(do_sample=True, temperature=a.temp) if sample else dict(do_sample=False))
        with torch.no_grad():
            out_ = m.generate(**ids_, **kw)
        return tok.decode(out_[0][ids_.input_ids.shape[1]:], skip_special_tokens=True)

    store = R.Store(out)
    t0 = time.time(); done = 0
    for i in ids:
        inst = data[i]
        n_refs = len(inst["ctxs_candidate"])
        if a.mode == "union":
            sel = sorted(set(R.parse_selection(rs[i], n_refs)) | set(R.parse_selection(rw[i], n_refs)))
            user = R.qa_prompt(inst, R.build_block(inst, sel))
            key = f"{i}|un"
            if key not in store: store.put(key, gen(user, sample=False))
        else:
            sel = R.parse_selection(rs[i], n_refs)
            user = R.qa_prompt(inst, R.build_block(inst, sel))
            for j in (1, 2, 3):
                key = f"{i}|t{j}"
                if key not in store: store.put(key, gen(user, sample=True))
        done += 1
        if done % 25 == 0:
            store.flush(); print(f"  ... {done}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
    store.flush()
    print(f"[done] mode={a.mode} {done} instances -> {out}", flush=True)

if __name__ == "__main__":
    main()
