#!/usr/bin/env python3
# verifier_compare.py v1.0 — comparative verification (23 Sep 2026)
# For each val+test instance: show the 7B checker the question, the STRONG evidence,
# and BOTH produced answers (the weak answerer's and the strong answerer's), and ask
# WHICH is better supported. Records P(weak answer is the better one) from A/B logits.
# Fixes two-sided's failure mode: absolute scores saturate near 1.0 and cannot RANK;
# a forced comparison can. Answer order is swapped per instance (idx parity) to cancel
# position bias; the recorded probability always refers to the WEAK answer.
# Output rows: {idx, split, p_weak_better}. ~2,591 prompts, ~1 hr GPU.
import json, argparse, os

PROMPT = (
    "You are comparing two proposed answers against evidence.\n\n"
    "Question: {q}\n\n"
    "Evidence:\n{ev}\n\n"
    "Answer A: {a}\n"
    "Answer B: {b}\n\n"
    "Which answer is correct and better supported by the evidence? "
    "Reply with exactly one letter: A or B."
)

def jl(p): return [json.loads(l) for l in open(p)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
    ap.add_argument("--weak_answers", default="/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl")
    ap.add_argument("--strong_answers", default="/mnt/hdd/jonathan/runs/full_run_p100/qa_strong.jsonl")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--only_splits", default="val,test")
    ap.add_argument("--out", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_compare.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    grid = jl(a.grid)
    def amap(path):
        m = {}
        for r in jl(path):
            i, br = r["k"].split("|")
            if br == "strong": m[int(i)] = (r.get("v") or "").strip() or "(no answer)"
        return m
    wa = amap(a.weak_answers); sa = amap(a.strong_answers)
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    wanted = [s.strip() for s in a.only_splits.split(",")]
    ids = [i for s in wanted for i in sp[s]]
    if a.limit: ids = ids[:a.limit]
    where = {i: s for s in wanted for i in sp[s]}
    print(f"[cfg] model={a.model} n={len(ids)} out={a.out}", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    A_id = tok.encode("A", add_special_tokens=False); B_id = tok.encode("B", add_special_tokens=False)
    assert len(A_id) == 1 and len(B_id) == 1
    A_id, B_id = A_id[0], B_id[0]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = 0
    with open(a.out, "w") as f, torch.no_grad():
        for i in ids:
            r = grid[i]
            weak_first = (i % 2 == 0)          # swap order by parity to cancel position bias
            fa, fb = (wa.get(i, "(no answer)"), sa.get(i, "(no answer)")) if weak_first else (sa.get(i, "(no answer)"), wa.get(i, "(no answer)"))
            uc = PROMPT.format(q=r["question"], ev=r["evidence_S"], a=fa, b=fb)
            msgs = [{"role": "user", "content": uc}]
            t = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                    return_tensors="pt").to(model.device)
            logits = model(**t).logits[0, -1]
            pA = torch.softmax(torch.tensor([logits[A_id].item(), logits[B_id].item()]), dim=0)[0].item()
            p_weak = pA if weak_first else (1.0 - pA)
            f.write(json.dumps({"idx": i, "split": where[i], "p_weak_better": round(p_weak, 6)}) + "\n")
            done += 1
            if done % 200 == 0: print(f"  ... {done}/{len(ids)}", flush=True)
    print(f"[done] wrote {done} rows -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
