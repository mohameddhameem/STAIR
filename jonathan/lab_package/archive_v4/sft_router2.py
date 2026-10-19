#!/usr/bin/env python3
"""sft_router2.py — SFT + cross-entropy router, clean rebuild (17 Aug 2026).

Design decisions baked in from the audit:
  * fp16 base + fp32 LoRA adapters (no bf16-AdamW underflow path)
  * left padding + logits read at out[:, -1]  (the padding bug is impossible here)
  * ties (weak-strong vs strong-weak, equal R) DROPPED from SFT targets by
    default — they teach an arbitrary class (--keep_ties restores argmax order)
  * two prompt modes:
      q      question only                -> 4-way joint decision (A/B/C/D)
      qretr  question + STRONG retrieval  -> 2-way QA-stage decision (A/B =
             strong-weak / strong-strong), conditioned on stage-1 output
  * manifest printed and saved; seed fixed; eval reports headroom vs
    train-selected best-fixed, oracle, and disagreement-AUC with its n.

Usage (from lab_package, venv active):
  python3 sft_router2.py --mode q     --out runs/sft2_q
  python3 sft_router2.py --mode qretr --out runs/sft2_qretr
  add --epochs 3 --lr 1e-4 --batch 16 as needed
"""
import os, json, argparse, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
import torch
import torch.nn.functional as F
from collections import Counter

COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
LETTERS = ["A", "B", "C", "D"]

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

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["q", "qretr"], default="q")
ap.add_argument("--data", default="router_data.jsonl")
ap.add_argument("--splits", default="splits.json")
ap.add_argument("--retr", default="/mnt/hdd/jonathan/runs/full_run/retr_strong.jsonl")
ap.add_argument("--router_model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--maxlen", type=int, default=1024)
ap.add_argument("--retr_chars", type=int, default=1500)
ap.add_argument("--keep_ties", action="store_true")
ap.add_argument("--seed", type=int, default=20260817)
a = ap.parse_args()

rows = {json.loads(l)["idx"]: json.loads(l) for l in open(a.data)}
sp = json.load(open(a.splits))["splits"]

retr = {}
if a.mode == "qretr":
    for l in open(a.retr):
        r = json.loads(l)
        retr[int(str(r["k"]).split("|")[0])] = (r["v"] or "")[: a.retr_chars]

if a.mode == "q":
    act_map = [0, 1, 2, 3]                       # A..D -> combos
else:
    act_map = [2, 3]                              # A,B -> strong-weak, strong-strong
n_act = len(act_map)
letters = LETTERS[:n_act]

def usable(i):
    return i in rows and (a.mode == "q" or (i in retr and retr[i] and "OOM" not in retr[i]))

def target(i):
    """SFT target over act_map by argmax-R; None if tied (unless --keep_ties)."""
    R = [rows[i]["R"][k] for k in act_map]
    m = max(R); winners = [j for j, v in enumerate(R) if abs(v - m) < 1e-9]
    if len(winners) > 1 and not a.keep_ties:
        return None
    return winners[0]

tr_all = [i for i in sp["train"] if usable(i)]
va = [i for i in sp["val"] if usable(i)]
te = [i for i in sp["test"] if usable(i)]
tr = [i for i in tr_all if target(i) is not None]
manifest = {"script": "sft_router2.py", "mode": a.mode, "n_actions": n_act,
            "epochs": a.epochs, "lr": a.lr, "batch": a.batch, "lora_r": a.lora_r,
            "maxlen": a.maxlen, "retr_chars": a.retr_chars, "keep_ties": a.keep_ties,
            "seed": a.seed, "router_model": a.router_model,
            "train_usable": len(tr_all), "train_after_tie_drop": len(tr),
            "val": len(va), "test": len(te),
            "currency": "our FLOP units (ww2.0/ws5.7/sw5.7/ss9.4)"}
print(json.dumps(manifest, indent=2), flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

tok = AutoTokenizer.from_pretrained(a.router_model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
lab_ids = []
for ch in letters:
    e = tok.encode(ch, add_special_tokens=False)
    assert len(e) == 1, f"label {ch!r} not a single token"
    lab_ids.append(e[0])

model = AutoModelForCausalLM.from_pretrained(
    a.router_model, torch_dtype=torch.float16, device_map="cuda",
    attn_implementation="sdpa")
model = get_peft_model(model, LoraConfig(
    r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], task_type="CAUSAL_LM"))
for p in model.parameters():                     # LoRA adapters -> fp32
    if p.requires_grad:
        p.data = p.data.float()
model.print_trainable_parameters()

def prompt_for(i):
    q = rows[i]["question"]
    if a.mode == "q":
        msgs = [{"role": "system", "content": SYS_Q},
                {"role": "user", "content": INSTR_Q.format(q=q)}]
    else:
        msgs = [{"role": "system", "content": SYS_R},
                {"role": "user", "content": INSTR_R.format(q=q, r=retr[i])}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def action_logits(ids):
    texts = [prompt_for(i) for i in ids]
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=a.maxlen).to("cuda")
    out = model(**enc).logits
    sel = out[:, -1]                              # left padding: last position, always
    return sel[:, lab_ids].float()

def evaluate(ids, tag):
    model.eval(); pick = {}; probs = {}
    with torch.no_grad():
        for b0 in range(0, len(ids), 32):
            b = ids[b0:b0 + 32]
            lg = action_logits(b)
            pr = torch.softmax(lg, -1)
            for i, c, p0 in zip(b, lg.argmax(-1).tolist(), pr[:, 0].tolist()):
                pick[i] = act_map[c]; probs[i] = p0
    model.train()
    n = len(ids)
    em = sum(rows[i]["em"][pick[i]] for i in ids) / n
    cost = sum(COST[COMBOS[pick[i]]] for i in ids) / n / COST["strong-strong"]
    Rm = sum(rows[i]["R"][pick[i]] for i in ids) / n
    mix = Counter(COMBOS[pick[i]] for i in ids)
    fixed = {COMBOS[k]: sum(rows[i]["em"][k] for i in ids) / n for k in act_map}
    bf = max(act_map, key=lambda k: sum(rows[i]["R"][k] for i in tr_all) / max(1, len(tr_all)))
    orc = sum(max(rows[i]["R"][k] for k in act_map) for i in ids) / n
    bfR = sum(rows[i]["R"][bf] for i in ids) / n
    head = 100 * (Rm - bfR) / (orc - bfR) if orc > bfR + 1e-9 else float("nan")
    d = [i for i in ids if rows[i]["em"][act_map[0]] != rows[i]["em"][act_map[-1]]]
    pos = [probs[i] for i in d if rows[i]["em"][act_map[0]] == 1]
    neg = [probs[i] for i in d if rows[i]["em"][act_map[0]] == 0]
    auc = float("nan")
    if pos and neg:
        xs = sorted([(v, 1) for v in pos] + [(v, 0) for v in neg]); rk = {}; j = 0
        while j < len(xs):
            k2 = j
            while k2 < len(xs) and xs[k2][0] == xs[j][0]: k2 += 1
            for t in range(j, k2): rk[t] = (j + k2 + 1) / 2
            j = k2
        s = sum(rk[t] for t, (v, yy) in enumerate(xs) if yy == 1)
        auc = (s - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    print(f"[{tag}] EM {100*em:.1f}%  cost {100*cost:.1f}%  R {Rm:.4f}  "
          f"headroom {head:.1f}%  picks {dict(mix)}", flush=True)
    print(f"[{tag}] anchors: " + "  ".join(f"{c} EM {100*v:.1f}%" for c, v in fixed.items())
          + f"  | best-fixed(train,R)={COMBOS[bf]}", flush=True)
    print(f"[{tag}] AUC(first-action) on disagreements {auc:.3f}  n={len(d)} "
          f"({len(pos)} pos / {len(neg)} neg)", flush=True)
    return {"em": em, "cost": cost, "R": Rm, "headroom_pct": head,
            "picks": dict(mix), "auc_disagree": auc, "n_disagree": len(d)}

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
rnd = random.Random(a.seed)
step = 0
for ep in range(a.epochs):
    order = list(tr); rnd.shuffle(order)
    model.train()
    for b0 in range(0, len(order), a.batch):
        b = order[b0:b0 + a.batch]
        logits = action_logits(b)
        y = torch.tensor([target(i) for i in b], device="cuda")
        loss = F.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step(); opt.zero_grad(); step += 1
        if step % 25 == 0:
            print(f"ep{ep} step{step} loss {loss.item():.4f}", flush=True)
    evaluate(va, f"val ep{ep}")

res = evaluate(te, "TEST")
os.makedirs(a.out, exist_ok=True)
model.save_pretrained(a.out)
json.dump({"manifest": manifest, "test": res},
          open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"saved -> {a.out}", flush=True)
