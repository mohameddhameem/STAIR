#!/usr/bin/env python3
# fuser_pilot.py v1.0 — AgentTTS's answer-creation method on OUR candidates (24 Sep 2026)
# Applies the stock Archon Fuser (their 0.74 config's combiner) to our 4 weak samples
# per instance: prompt text VERBATIM from the Archon library (pulled 23 Sep), fuser
# temperature 0.3, max_tokens 1024, system string "You are a helpful assistant who
# fuses multiple answers" — all their settings, none assumed.
# Pool: the champion's ESCALATED val instances (p_supported < 1.0), first --n (500),
# identical to every pilot this week. Candidates per instance = greedy weak answer
# (grid) + the 3 stored best-of-K samples.
# --fuser weak  -> Qwen2.5-1.5B fuses (their same-model design, faithful mirror)
# --fuser strong-> Qwen2.5-7B fuses (the "bigger fuser" upgrade probe)
import os, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import run_labels4 as R

def fuser_prompt(query, cands):
    # VERBATIM structure from archon/completions/components/prompts.py (no-critique branch)
    p = (f"You have been provided with a set of responses from various open-source models to the latest user query, which is {query}."
         f"            Your task is to synthesize these responses into a single, high-quality response.             "
         f"It is crucial to critically evaluate the information provided in these responses, recognizing that some of it may be biased or incorrect.             "
         f"Your response should not simply replicate the given answers but should offer a refined, accurate, and comprehensive reply to the instruction.             "
         f"Ensure your response is well-structured, coherent, and adheres to the highest standards of accuracy and reliability.\n")
    p += f"Once again, the query is: {query}\n"
    p += "Responses from models:"
    for i, c in enumerate(cands):
        p += f"\n{i+1}. {c}"
    return p

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuser", required=True, choices=["weak", "strong"], help="which model fuses: weak=1.5B (their same-model design), strong=7B")
    ap.add_argument("--data", default="/mnt/hdd/jonathan/data/tts_test.json")
    ap.add_argument("--qa_weak", default="/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl")
    ap.add_argument("--samples", default="/mnt/hdd/jonathan/runs/verifier_scores/bestofk_pilot.jsonl", help="the stored best-of-K weak candidates: s0=greedy, s1..s3=samples (4 rows/instance)")
    ap.add_argument("--ver", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--keep_gate", type=float, default=1.0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--temp", type=float, default=0.3, help="their fuser temperature")
    ap.add_argument("--max_new", type=int, default=1024, help="their fuser max_tokens")
    ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or f"/mnt/hdd/jonathan/runs/verifier_scores/fuser_{a.fuser}.jsonl"

    data = json.load(open(a.data))
    greedy = {}
    for l in open(a.qa_weak):
        r = json.loads(l); i, br = r["k"].split("|")
        if br == "strong": greedy[int(i)] = (r.get("v") or "").strip() or "(no answer)"
    samples = {}
    for l in open(a.samples):
        r = json.loads(l); i = int(str(r["k"]).split("|")[0])
        samples.setdefault(i, []).append((r.get("v") or "").strip() or "(no answer)")
    ver = {json.loads(l)["idx"]: json.loads(l)["p_supported"] for l in open(a.ver)}
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    ids = [i for i in sp[a.split] if i in samples and len(samples[i]) >= 4 and ver.get(i, 1.0) < a.keep_gate][: a.n]
    print(f"[cfg] fuser={a.fuser} pool n={len(ids)} temp={a.temp} max_new={a.max_new} out={out}", flush=True)
    from collections import Counter
    print(f"[census] candidates per instance: {dict(Counter(len(samples[i][:4]) for i in ids))}  (expect 4; s0=greedy included)", flush=True)

    m, tok = R.load_model(a.fuser, dtype=a.dtype, stage="qa")
    SYS = "You are a helpful assistant who fuses multiple answers"

    def gen(user):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids_ = tok(text, return_tensors="pt", truncation=True, max_length=R.MAXLEN).to(m.device)
        with torch.no_grad():
            o = m.generate(**ids_, max_new_tokens=a.max_new, do_sample=True, temperature=a.temp, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0][ids_.input_ids.shape[1]:], skip_special_tokens=True)

    store = R.Store(out)
    t0 = time.time(); done = 0
    for i in ids:
        key = f"{i}|fu"
        if key in store: done += 1; continue
        q = data[i]["question"] if "question" in data[i] else data[i].get("query", "")
        cands = samples[i][:4]
        store.put(key, gen(fuser_prompt(q, cands)))
        done += 1
        if done % 25 == 0:
            store.flush(); print(f"  ... {done}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
    store.flush()
    print(f"[done] fuser={a.fuser} {done} fused answers -> {out}", flush=True)

if __name__ == "__main__":
    main()
