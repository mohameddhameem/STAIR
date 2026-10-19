#!/usr/bin/env python3
"""show_prompt.py — render the EXACT prompts sft_router2.py sends, on real
instances, for validation before the qretr launch. CPU-only; safe beside a
running GPU job. Run from lab_package (venv active):
    python3 show_prompt.py
"""
import json

# Templates copied verbatim from sft_router2.py — if you edit them there,
# re-copy here (or better: tell Claude and both get updated together).
SYS_Q = ("You route a two-stage question-answering pipeline. Each stage runs either a "
         "weak (cheap) or a strong (expensive) model. Choose the configuration that "
         "answers the question correctly at the lowest cost.")
INSTR_Q = ("A = weak retrieval, weak QA\nB = weak retrieval, strong QA\n"
           "C = strong retrieval, weak QA\nD = strong retrieval, strong QA\n\n"
           "Question: {q}\n\nAnswer with one letter only.")
SYS_R = ("You decide which answering model a question-answering pipeline should use. "
         "Retrieval has already been done by the strong model; its selected references "
         "are shown. Choose the cheapest answering model that will still answer "
         "correctly.")
INSTR_R = ("A = weak (cheap) answering model\nB = strong (expensive) answering model\n\n"
           "Question: {q}\n\nRetrieved references (strong retriever):\n{r}\n\n"
           "Answer with one letter only.")

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

rows = {json.loads(l)["idx"]: json.loads(l) for l in open("router_data.jsonl")}
retr = {}
for l in open("/mnt/hdd/jonathan/runs/full_run/retr_strong.jsonl"):
    r = json.loads(l)
    retr[int(str(r["k"]).split("|")[0])] = (r["v"] or "")[:1500]
tr = json.load(open("splits.json"))["splits"]["train"]
C = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]

shown = 0
for i in tr:
    if i not in rows or i not in retr or not retr[i] or "OOM" in retr[i]:
        continue
    q = rows[i]["question"]; R = rows[i]["R"]
    arg4 = C[max(range(4), key=lambda k: R[k])]
    arg2 = C[2] if R[2] >= R[3] else C[3]
    print("=" * 78)
    print(f"idx {i}")
    print(f"R = {[round(x,3) for x in R]}")
    print(f"target (mode q, 4-way argmax-R)     : {arg4}")
    print(f"target (mode qretr, sw-vs-ss argmax): {arg2}")
    for mode, sysm, instr in (("q", SYS_Q, INSTR_Q.format(q=q)),
                              ("qretr", SYS_R, INSTR_R.format(q=q, r=retr[i]))):
        msgs = [{"role": "system", "content": sysm},
                {"role": "user", "content": instr}]
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        n = len(tok.encode(txt))
        print(f"\n----- mode {mode} | {n} tokens "
              f"{'(OK, < maxlen 1024)' if n <= 1024 else '(!! EXCEEDS maxlen — will truncate)'} -----")
        if mode == "q":
            print(txt)
        else:
            print(txt[:1100])
            print("   ...... [middle elided for readability] ......")
            print(txt[-260:])
    shown += 1
    if shown >= 3:
        break

print("\nCHECKLIST:")
print(" 1. Every prompt ends with '<|im_start|>assistant\\n' and NOTHING after")
print(" 2. mode q ~90-130 tokens; mode qretr well under 1024 (no truncation)")
print(" 3. qretr retrieval text is a sane selection block, not junk/OOM")
print(" 4. letter mapping matches: q A/B/C/D = ww/ws/sw/ss; qretr A/B = weak/strong QA")
print(" 5. printed targets are ones you'd defend by hand from the R vector")
