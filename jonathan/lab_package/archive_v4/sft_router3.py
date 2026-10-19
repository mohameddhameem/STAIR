#!/usr/bin/env python3
"""sft_router3.py — SFT + CE router with stabilized optimization (17 Aug, v3).

v2 -> v3, motivated by the v2 loss graphs (no sustained descent, late spikes,
val degrading epoch-over-epoch, best AUC at the earliest checkpoint):
  * linear WARMUP over the first --warmup_frac of optimizer steps
  * GRADIENT ACCUMULATION (--accum, default 4 -> effective batch 64)
  * MID-EPOCH VAL PROBES every --eval_every optimizer steps on a fixed seeded
    subsample (--val_probe_n) of val: probe loss + disagreement-AUC
  * BEST-CHECKPOINT KEEPING by probe AUC; adapter saved to <out>/best; final
    TEST is evaluated on BOTH the last state and the best state
Everything else inherited from v2: fp16 base + fp32 LoRA, left padding with
out[:, -1], ties dropped, manifest, seed, train-selected best-fixed, AUC with n.

Usage (from lab_package, venv active):
  python3 sft_router3.py --mode qretr --out runs/sft3_qretr
  python3 sft_router3.py --mode q     --out runs/sft3_q
"""
import os, json, argparse, random, copy
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
ap.add_argument("--mode", choices=["q", "qretr"], default="qretr")
ap.add_argument("--data", default="router_data.jsonl")
ap.add_argument("--splits", default="splits.json")
ap.add_argument("--retr", default="/mnt/hdd/jonathan/runs/full_run/retr_strong.jsonl")
ap.add_argument("--router_model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--accum", type=int, default=8)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--warmup_frac", type=float, default=0.05)
ap.add_argument("--eval_every", type=int, default=25)   # optimizer steps
ap.add_argument("--val_probe_n", type=int, default=400)
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

act_map = [0, 1, 2, 3] if a.mode == "q" else [2, 3]
n_act = len(act_map); letters = LETTERS[:n_act]

def usable(i):
    return i in rows and (a.mode == "q" or (i in retr and retr[i] and "OOM" not in retr[i]))

def target(i):
    R = [rows[i]["R"][k] for k in act_map]
    m = max(R); w = [j for j, v in enumerate(R) if abs(v - m) < 1e-9]
    if len(w) > 1 and not a.keep_ties:
        return None
    return w[0]

tr_all = [i for i in sp["train"] if usable(i)]
va = [i for i in sp["val"] if usable(i)]
te = [i for i in sp["test"] if usable(i)]
tr = [i for i in tr_all if target(i) is not None]
opt_steps_per_ep = (len(tr) + a.batch * a.accum - 1) // (a.batch * a.accum)
total_opt_steps = opt_steps_per_ep * a.epochs
warmup_steps = max(1, int(total_opt_steps * a.warmup_frac))
probe_ids = random.Random(a.seed).sample(va, min(a.val_probe_n, len(va)))

manifest = {"script": "sft_router3.py (v3.1 memory-safe)", "mode": a.mode, "n_actions": n_act,
            "epochs": a.epochs, "batch": a.batch, "accum": a.accum,
            "eff_batch": a.batch * a.accum, "lr": a.lr,
            "warmup_steps": warmup_steps, "total_opt_steps": total_opt_steps,
            "eval_every": a.eval_every, "val_probe_n": len(probe_ids),
            "lora_r": a.lora_r, "maxlen": a.maxlen, "retr_chars": a.retr_chars,
            "keep_ties": a.keep_ties, "seed": a.seed,
            "router_model": a.router_model,
            "train_usable": len(tr_all), "train_after_tie_drop": len(tr),
            "val": len(va), "test": len(te),
            "currency": "our FLOP units (ww2.0/ws5.7/sw5.7/ss9.4)"}
print(json.dumps(manifest, indent=2), flush=True)

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict

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
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
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
    try:
        out = model(**enc, logits_to_keep=1).logits      # [B, 1, V] — final position only
    except TypeError:
        try:
            out = model(**enc, num_logits_to_keep=1).logits
        except TypeError:
            out = model(**enc).logits                     # full [B, T, V] fallback
    return out[:, -1][:, lab_ids].float()

def auc_of(scores_pos, scores_neg):
    if not scores_pos or not scores_neg:
        return float("nan")
    xs = sorted([(v, 1) for v in scores_pos] + [(v, 0) for v in scores_neg])
    rk = {}; j = 0
    while j < len(xs):
        k2 = j
        while k2 < len(xs) and xs[k2][0] == xs[j][0]: k2 += 1
        for t in range(j, k2): rk[t] = (j + k2 + 1) / 2
        j = k2
    s = sum(rk[t] for t, (v, y) in enumerate(xs) if y == 1)
    return (s - len(scores_pos) * (len(scores_pos) + 1) / 2) / (len(scores_pos) * len(scores_neg))

@torch.no_grad()
def probe(ids):
    """Fast mid-training probe: mean loss + disagreement-AUC on a subsample."""
    model.eval()
    losses, p0 = [], {}
    for b0 in range(0, len(ids), 32):
        b = ids[b0:b0 + 32]
        lg = action_logits(b)
        ys = torch.tensor([target(i) if target(i) is not None else 0 for i in b], device="cuda")
        losses.append(F.cross_entropy(lg, ys).item())
        pr = torch.softmax(lg, -1)
        for i, v in zip(b, pr[:, 0].tolist()):
            p0[i] = v
    model.train()
    d = [i for i in ids if rows[i]["em"][act_map[0]] != rows[i]["em"][act_map[-1]]]
    pos = [p0[i] for i in d if rows[i]["em"][act_map[0]] == 1]
    neg = [p0[i] for i in d if rows[i]["em"][act_map[0]] == 0]
    return sum(losses) / len(losses), auc_of(pos, neg), len(d), len(pos)

@torch.no_grad()
def evaluate(ids, tag):
    model.eval(); pick = {}; probs = {}
    for b0 in range(0, len(ids), 32):
        b = ids[b0:b0 + 32]
        lg = action_logits(b)
        pr = torch.softmax(lg, -1)
        for i, c, v in zip(b, lg.argmax(-1).tolist(), pr[:, 0].tolist()):
            pick[i] = act_map[c]; probs[i] = v
    model.train()
    n = len(ids)
    em = sum(rows[i]["em"][pick[i]] for i in ids) / n
    cost = sum(COST[COMBOS[pick[i]]] for i in ids) / n / COST["strong-strong"]
    Rm = sum(rows[i]["R"][pick[i]] for i in ids) / n
    mix = Counter(COMBOS[pick[i]] for i in ids)
    bf = max(act_map, key=lambda k: sum(rows[i]["R"][k] for i in tr_all) / max(1, len(tr_all)))
    orc = sum(max(rows[i]["R"][k] for k in act_map) for i in ids) / n
    bfR = sum(rows[i]["R"][bf] for i in ids) / n
    head = 100 * (Rm - bfR) / (orc - bfR) if orc > bfR + 1e-9 else float("nan")
    d = [i for i in ids if rows[i]["em"][act_map[0]] != rows[i]["em"][act_map[-1]]]
    pos = [probs[i] for i in d if rows[i]["em"][act_map[0]] == 1]
    neg = [probs[i] for i in d if rows[i]["em"][act_map[0]] == 0]
    auc = auc_of(pos, neg)
    print(f"[{tag}] EM {100*em:.1f}%  cost {100*cost:.1f}%  R {Rm:.4f}  "
          f"headroom {head:.1f}%  picks {dict(mix)}", flush=True)
    print(f"[{tag}] AUC(first-action) on disagreements {auc:.3f}  n={len(d)} "
          f"({len(pos)} pos / {len(neg)} neg)", flush=True)
    return {"em": em, "cost": cost, "R": Rm, "headroom_pct": head,
            "picks": dict(mix), "auc_disagree": auc, "n_disagree": len(d)}

opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
def lr_at(step):
    return a.lr * min(1.0, step / warmup_steps)

rnd = random.Random(a.seed)
best = {"auc": -1.0, "step": -1}
os.makedirs(a.out, exist_ok=True)
opt_step = 0
for ep in range(a.epochs):
    order = list(tr); rnd.shuffle(order)
    model.train()
    micro = 0
    opt.zero_grad()
    for b0 in range(0, len(order), a.batch):
        b = order[b0:b0 + a.batch]
        logits = action_logits(b)
        y = torch.tensor([target(i) for i in b], device="cuda")
        loss = F.cross_entropy(logits, y) / a.accum
        loss.backward()
        micro += 1
        if micro % a.accum == 0 or b0 + a.batch >= len(order):
            for g in opt.param_groups:
                g["lr"] = lr_at(opt_step + 1)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad(); opt_step += 1
            if opt_step % 5 == 0:
                print(f"ep{ep} opt_step {opt_step}/{total_opt_steps} "
                      f"loss {loss.item()*a.accum:.4f} lr {lr_at(opt_step):.2e}", flush=True)
            if opt_step % a.eval_every == 0:
                pl, pauc, nd, npos = probe(probe_ids)
                mark = ""
                if pauc == pauc and pauc > best["auc"]:
                    best = {"auc": pauc, "step": opt_step}
                    torch.save(get_peft_model_state_dict(model),
                               os.path.join(a.out, "best_adapter.pt"))
                    mark = "  <-- new best (adapter saved)"
                print(f"  [probe @ {opt_step}] loss {pl:.4f}  AUC-dis {pauc:.3f} "
                      f"(n={nd}, {npos} pos){mark}", flush=True)
    evaluate(va, f"val ep{ep} (last state)")

print(f"\nbest probe AUC {best['auc']:.3f} at opt_step {best['step']}", flush=True)
res_last = evaluate(te, "TEST last-state")
if best["step"] > 0:
    set_peft_model_state_dict(model, torch.load(os.path.join(a.out, "best_adapter.pt")))
    res_best = evaluate(te, f"TEST best-checkpoint (opt_step {best['step']})")
else:
    res_best = None
model.save_pretrained(a.out)
json.dump({"manifest": manifest, "best_probe": best,
           "test_last": res_last, "test_best": res_best},
          open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"saved -> {a.out}", flush=True)
