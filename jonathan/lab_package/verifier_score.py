#!/usr/bin/env python3
# verifier_score.py v1.1 — adds free-form --branch (s1/s2/s3), --evidence, --only_scored (23 Sep 2026)
# For every val+test instance: show the 7B scorer (question, strong evidence, the weak
# answerer's produced answer) and record P(supported) from the Yes/No logits.
# These scores are DUAL-USE: (a) calibrated gate for the cascade re-sim;
# (b) graded training labels for the router (experiment #3) — the campaign's new teacher.
# Read-only inputs; writes one jsonl. GPU ~30-90 min for 2,591 prompts.
import json, argparse, os

PROMPT = (
    "You are checking an answer against evidence.\n\n"
    "Question: {q}\n\n"
    "Evidence:\n{ev}\n\n"
    "Proposed answer: {ans}\n\n"
    "Is the proposed answer correct and supported by the evidence? "
    "Reply with exactly one word: Yes or No."
)

def load_jsonl(p):
    return [json.loads(l) for l in open(p)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
    ap.add_argument("--answers", default="/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl")
    ap.add_argument("--branch", default="strong",
                    help="answer-key suffix to verify: strong / weak / s1 / s2 / s3 ...")
    ap.add_argument("--evidence", default="strong", choices=["strong","weak"],
                    help="which evidence the answers are checked against (samples use strong)")
    ap.add_argument("--only_scored", action="store_true",
                    help="score only instances present in the answers file (pilot subsets)")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--only_splits", default="val,test")
    ap.add_argument("--out", default="/mnt/hdd/jonathan/runs/verifier_scores/verifier_scores.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--evidence_chars", type=int, default=-1, help="-1 = full evidence")
    ap.add_argument("--limit", type=int, default=0, help="smoke: score only first N")
    a = ap.parse_args()

    grid = load_jsonl(a.grid)
    amap = {}
    for r in load_jsonl(a.answers):
        i, br = r["k"].split("|")
        if br == a.branch: amap[int(i)] = r.get("v") or ""
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    wanted = [s.strip() for s in a.only_splits.split(",")]
    ids = [i for s in wanted for i in sp[s]]
    if a.only_scored: ids = [i for i in ids if i in amap]
    if a.limit: ids = ids[:a.limit]
    print(f"[cfg] model={a.model} branch={a.branch} n={len(ids)} out={a.out}", flush=True)

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    yes_id = tok.encode("Yes", add_special_tokens=False); no_id = tok.encode("No", add_special_tokens=False)
    assert len(yes_id) == 1 and len(no_id) == 1, f"multi-token Yes/No: {yes_id}/{no_id}"
    yes_id, no_id = yes_id[0], no_id[0]

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    where = {i: s for s in wanted for i in sp[s]}
    done = 0
    with open(a.out, "w") as f, torch.no_grad():
        for i in ids:
            r = grid[i]
            ev_key = "evidence_S" if a.evidence == "strong" else "evidence_W"
            ev = r[ev_key]
            if a.evidence_chars > 0: ev = ev[:a.evidence_chars]
            uc = PROMPT.format(q=r["question"], ev=ev, ans=(amap.get(i, "") or "(no answer)"))
            msgs = [{"role": "user", "content": uc}]
            t = tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True),
                    return_tensors="pt").to(model.device)
            logits = model(**t).logits[0, -1]
            p = torch.softmax(torch.tensor([logits[yes_id].item(), logits[no_id].item()]), dim=0)[0].item()
            f.write(json.dumps({"idx": i, "split": where[i], "p_supported": round(p, 6)}) + "\n")
            done += 1
            if done % 200 == 0: print(f"  ... {done}/{len(ids)}", flush=True)
    print(f"[done] wrote {done} rows -> {a.out}", flush=True)

if __name__ == "__main__":
    main()
