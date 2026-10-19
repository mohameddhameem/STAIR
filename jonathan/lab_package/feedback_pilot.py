#!/usr/bin/env python3
# feedback_pilot.py v1.0 — feedback escalation pilot (23 Sep 2026)
# HYPOTHESIS: on escalated instances, showing the STRONG answerer the failed weak
# answer ("this failed verification") steers it around the trap (decoy, wrong format)
# and lifts the escalated cell ABOVE the plain-SS rate. Escalated mass = ~75% of
# instances, so +1.5 on the cell ~= +1.1 composed.
# PILOT: VAL ONLY. Instances = the champion's ESCALATED pool (greedy weak answer's
# p_supported < 1.0 — the val-selected keep gate), capped at --n (default 500).
# For each: strong answerer, greedy decode, standard qa_prompt + a FEEDBACK block
# appended (approved wording). Prompts otherwise byte-identical to the pipeline:
# imports run_labels4 (qa_prompt/parse_selection/build_block/load_model/QA_OUT/MAXLEN),
# same system string. Output Store rows: k="idx|fb", v=answer.
# The analysis afterward compares em(feedback answer) vs em(plain SS answer) on the
# same instances (metric = calibrated whole-word lenient) — cell vs cell.
import os, json, time, argparse
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch
import run_labels4 as R

FEEDBACK = ('\n\nA first attempt answered: "{wa}"\n'
            'This answer failed verification against the references \u2014 it may be incorrect or unsupported.\n'
            'Avoid repeating its mistake. Answer the query correctly using the references.')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hdd/jonathan/data/tts_test.json")
    ap.add_argument("--retr", default="/mnt/hdd/jonathan/runs/full_run_p100/retr_strong.jsonl")
    ap.add_argument("--qa_weak", default="/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl")
    ap.add_argument("--ver", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--split", default="val")
    ap.add_argument("--keep_gate", type=float, default=1.0, help="champion's val-selected keep threshold; escalated = p_supported < this")
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--dtype", default="fp16", help="strong model dtype (manifest: fp16)")
    ap.add_argument("--out", default="/mnt/hdd/jonathan/runs/verifier_scores/feedback_pilot.jsonl")
    a = ap.parse_args()

    data = json.load(open(a.data))
    retr = {}
    for l in open(a.retr):
        r = json.loads(l); retr[int(str(r["k"]).split("|")[0])] = r["v"]
    wa = {}
    for l in open(a.qa_weak):
        r = json.loads(l); i, br = r["k"].split("|")
        if br == "strong": wa[int(i)] = (r.get("v") or "").strip() or "(no answer)"
    ver = {json.loads(l)["idx"]: json.loads(l)["p_supported"] for l in open(a.ver)}
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    ids = [i for i in sp[a.split] if i in retr and i in wa and ver.get(i, 1.0) < a.keep_gate][: a.n]
    print(f"[cfg] split={a.split} escalated-pool n={len(ids)} keep_gate={a.keep_gate} dtype={a.dtype}", flush=True)

    m, tok = R.load_model("strong", dtype=a.dtype, stage="qa")
    SYS = "You are a helpful assistant."

    def gen(user):
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user}]
        try:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        except TypeError:
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        ids_ = tok(text, return_tensors="pt", truncation=True, max_length=R.MAXLEN).to(m.device)
        with torch.no_grad():
            out = m.generate(**ids_, max_new_tokens=R.QA_OUT, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids_.input_ids.shape[1]:], skip_special_tokens=True)

    store = R.Store(a.out)
    t0 = time.time(); done = 0
    for i in ids:
        key = f"{i}|fb"
        if key in store: done += 1; continue
        inst = data[i]
        sel = R.parse_selection(retr[i], len(inst["ctxs_candidate"]))
        block = R.build_block(inst, sel)
        user = R.qa_prompt(inst, block) + FEEDBACK.format(wa=wa[i])
        store.put(key, gen(user))
        done += 1
        if done % 25 == 0:
            store.flush(); print(f"  ... {done}/{len(ids)}  ({time.time()-t0:.0f}s)", flush=True)
    store.flush()
    print(f"[done] {done} feedback answers -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
