#!/usr/bin/env python3
"""
probe_ret_coverage.py — v1.0 2026-10-02
The coverage teacher's LEARNABILITY PROBE (pulled forward from the training
experiment's first deliverable, per 2 Oct decision). Answers one question:
can (question + weak's selections) predict the coverage label at all?
Comparable to overlap's 0.605 (same inputs, same model; only the teacher differs).

MACHINERY (each block names its settled rule):
  P1 prompts  : SYS_CASCADE + INSTR_RET2, byte-identical to the FROZEN slide text;
                calls end at the opened assistant turn (add_generation_prompt=True)
  P2 reading  : logits z_W/z_S at the answer slot -> softmax -> P(W); the model
                never generates a letter (meter, not switch)
  P3 loss     : class-weighted CE vs ret_target (the post-mapping 2-class letter)
  P4 protocol : train on TRAIN split only; probe = label-match + AUC on VAL,
                every EVAL_EVERY steps; best + last adapter saved; test untouched

PINNED KNOBS (await Jonathan's sign-off before launch):
  LORA r=16 alpha=32 dropout=0.05 on q/k/v/o  ·  lr=1e-4 AdamW  ·  EPOCHS=2
  BATCH=8 via grad-accum (per-step batch 1)   ·  MAXLEN=8192 tokens
  CHAR_CAP=600 chars per selected passage (keeps select-all prompts inside MAXLEN;
  the SAME cap must be used by gate_eval later — byte-identical contract)

usage:
  python probe_ret_coverage.py --dry-run     # prints 3 assembled prompts, no GPU
  python probe_ret_coverage.py               # the probe run (~1-2 h shared GPU)
"""
import argparse, json, os, time
import sys
sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")

LABELS = "/mnt/hdd/jonathan/runs/ret_labels/ret_labels.jsonl"
DATA   = "/mnt/hdd/jonathan/data/tts_test.json"
RETR_W = "/mnt/hdd/jonathan/runs/full_run_p100/retr_weak.jsonl"
OUTDIR = "/mnt/hdd/jonathan/runs/probe_ret_coverage_16k"
MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"

MAXLEN, CHAR_CAP, LR, EPOCHS, ACCUM, EVAL_EVERY, SEED = 16384, 600, 1e-4, 2, 8, 50, 17

# ---- P1: the frozen prompt bodies (byte-identical to slide 17; do not edit) ----
SYS_CASCADE = ("You route a question-answering pipeline that runs the weak model first "
 "and escalates only when needed. Each stage runs either Qwen2.5-1.5B (weak, cheap) or "
 "Qwen2.5-7B (strong, expensive). From the weak model's output shown below, choose the "
 "cheapest option that still lets the pipeline answer correctly.")
INSTR_RET2 = ("Stage: RETRIEVAL REVIEW — the weak retriever has already selected references "
 "from the 100-passage pool.\n"
 "W = keep the weak selections (no further retrieval spend)\n"
 "S = escalate: run strong retrieval to reselect\n"
 "Question: {q}\n"
 "Weak retriever selected {n_sel} of 100 passages: {selections}\n"
 "Answer with one letter only.")

def retr_map(path):
    m = {}
    for l in open(path):
        r = json.loads(l); m[int(str(r["k"]).split("|")[0])] = r["v"]
    return m

def build_user(q, sel_ids, pool):
    parts = [f"[{k}] {pool[k-1][:CHAR_CAP]}" for k in sel_ids]
    return INSTR_RET2.format(q=q, n_sel=len(sel_ids), selections=" ".join(parts))

def assemble(tok, user):
    msgs = [{"role": "system", "content": SYS_CASCADE}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
    return ids[:, -MAXLEN:]          # truncate from the LEFT if over (keeps options+question tail)

def load_rows():
    import run_labels4 as R
    data = json.load(open(DATA)); rw = retr_map(RETR_W)
    rows = {"train": [], "val": []}
    for l in open(LABELS):
        r = json.loads(l)
        if r["split"] not in rows: continue
        i = r["idx"]; inst = data[i]
        sel = R.parse_selection(rw[i], len(inst["ctxs_candidate"]))
        rows[r["split"]].append({"idx": i, "user": build_user(inst["question"], sel, inst["ctxs_candidate"]),
                                 "target": r["ret_target"]})
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    rows = load_rows()
    print(f"train={len(rows['train'])}  val={len(rows['val'])}")
    if a.dry_run:
        for r in rows["train"][:3]:
            print("="*80); print("TARGET:", r["target"]); print(r["user"][:1500])
        return

    import torch, random
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    random.seed(SEED); torch.manual_seed(SEED)
    tok = AutoTokenizer.from_pretrained(MODEL)
    idW = tok.encode("W", add_special_tokens=False); idS = tok.encode("S", add_special_tokens=False)
    assert len(idW) == 1 and len(idS) == 1, "Design Note 1 violated: W/S must be single tokens"
    idW, idS = idW[0], idS[0]
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM"))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()     # needed when combining LoRA with checkpointing
    model.config.use_cache = False         # checkpointing and KV-cache do not mix
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    man = json.load(open("/mnt/hdd/jonathan/runs/ret_labels/manifest.json"))
    cw = man["class_weights_post_mapping"]           # P3: weights from the label manifest
    os.makedirs(OUTDIR, exist_ok=True)
    curve = open(os.path.join(OUTDIR, "probe_curve.jsonl"), "a")

    def logits_WS(user):
        ids = assemble(tok, user).to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return out[idW], out[idS]                    # P2: read, never generate

    def evaluate():
        model.eval(); hit = 0; ps, ys = [], []
        with torch.no_grad():
            for r in rows["val"]:
                zW, zS = logits_WS(r["user"])
                pW = torch.softmax(torch.stack([zW, zS]), 0)[0].item()
                pred = "W" if pW >= 0.5 else "S"
                hit += (pred == r["target"]); ps.append(pW); ys.append(1 if r["target"]=="W" else 0)
        # AUC (rank-based, no sklearn)
        pairs = sorted(zip(ps, ys)); rank = {id(p): i+1 for i, p in enumerate(pairs)}
        pos = [i+1 for i,(p,y) in enumerate(pairs) if y==1]; nP=len(pos); nN=len(pairs)-nP
        auc = (sum(pos) - nP*(nP+1)/2) / (nP*nN) if nP and nN else float("nan")
        model.train(); return hit/len(rows["val"]), auc

    best = 0.0; step = 0; t0 = time.time()
    order = rows["train"][:]
    for ep in range(EPOCHS):
        random.shuffle(order)
        opt.zero_grad()
        for k, r in enumerate(order):
            zW, zS = logits_WS(r["user"])
            logp = torch.log_softmax(torch.stack([zW, zS]), 0)
            w = cw[r["target"]]
            loss = -w * (logp[0] if r["target"]=="W" else logp[1]) / ACCUM   # P3
            loss.backward()
            if (k+1) % ACCUM == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % EVAL_EVERY == 0:
                    acc, auc = evaluate()
                    line = {"step": step, "epoch": ep, "val_match": round(acc,4),
                            "val_auc": round(auc,4), "min": round((time.time()-t0)/60,1)}
                    print(json.dumps(line), flush=True); curve.write(json.dumps(line)+"\n"); curve.flush()
                    if acc > best:
                        best = acc; model.save_pretrained(os.path.join(OUTDIR, "best_adapter"))
    acc, auc = evaluate()
    print(json.dumps({"FINAL": True, "val_match": round(acc,4), "val_auc": round(auc,4),
                      "best_val_match": round(best,4)}), flush=True)
    model.save_pretrained(os.path.join(OUTDIR, "last_adapter"))

if __name__ == "__main__":
    main()
