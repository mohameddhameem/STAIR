#!/usr/bin/env python3
# =============================================================================
# sft_router4.py — VERSION: sft_router4 v1.0 2026-08-18
#
# Evolves sft_router3.py (v3.1) to the slide 4-12 training design:
#   * mode qpool (default): question + the EXACT grid pool (from
#     build_router_data4.py), four-way over A/B/C/D answer letters (legend in the prompt)
#   * CLASS-WEIGHTED cross-entropy (inverse-frequency over the active train
#     labels; disable with --no_class_weights for the ablation)
#   * --scenario B (default): train on ALL usable train instances, unsolved
#     labelled WW (ASSUMPTION-b). --scenario A: solved instances only
#     (slide 4's ablation; B-vs-A separates difficulty from model choice)
#   * eval: LENIENT primary — router lenient vs strong-strong lenient vs
#     oracle lenient, with headroom capture % = (router-SS)/(oracle-SS);
#     strict EM and R reported as reference columns
#   * answer codes WW/WS/SW/SS: single-token head if each code is one
#     tokenizer token, otherwise automatic sequence-scoring fallback;
#     the manifest records which path ran
# Inherited unchanged from v3.1: warmup, grad accumulation, mid-epoch probes,
# best-checkpoint keeping, fp16 base + fp32 LoRA, left padding, seeds,
# manifest, memory guards. result.json keeps v3's structure (kpi_report.py
# compatible) with new fields ADDED.
#
# ASSUMPTION-a: --maxlen 4096 (the grid's own truncation cap, so long pools
#               truncate at the same horizon the graded pipeline had)
# ASSUMPTION-b: Scenario B labels unsolved instances WW
# ASSUMPTION-c: training cost currency = our FLOP units (2.0/5.7/5.7/9.4),
#               the units the labels were minted in
#
# Usage (lab, venv active; data file from build_router_data4.py):
#   python3 sft_router4.py --router_model Qwen/Qwen2.5-1.5B-Instruct \
#       --data /mnt/hdd/jonathan/runs/full_run/router_data4.jsonl \
#       --out runs/sft4_qpool_1p5b
#   python3 sft_router4.py --dry_run ...   # data+manifest+prompt only, no GPU
# =============================================================================
import os, json, argparse, random
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
from collections import Counter

VERSION = "sft_router4 v1.2 2026-08-18"
COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
CODES = ["A", "B", "C", "D"]  # legend in prompt + manifest
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7,
        "strong-strong": 9.4}                                   # ASSUMPTION-c
SS = 3  # index of strong-strong

SYS_POOL = ("You route a two-stage question-answering pipeline. Stage 1 "
            "(retrieval) selects references from a candidate pool; Stage 2 "
            "(answering) writes the answer using the selected references. "
            "Each stage runs either Qwen2.5-1.5B (weak, cheap) or Qwen2.5-7B "
            "(strong, expensive). Choose the cheapest configuration that "
            "will still answer the question correctly.")
INSTR_POOL = ("A = weak retrieval, weak answering\n"
              "B = weak retrieval, strong answering\n"
              "C = strong retrieval, weak answering\n"
              "D = strong retrieval, strong answering\n\n"
              "Question: {q}\n\n"
              "Candidate pool ({k} passages, as given to the retrieval "
              "stage):\n{pool}\n\n"
              "Answer with one letter only.")

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="router_data4.jsonl from build_router_data4.py")
ap.add_argument("--splits", default="splits.json")
ap.add_argument("--router_model", default="Qwen/Qwen2.5-1.5B-Instruct")
ap.add_argument("--out", required=True)
ap.add_argument("--scenario", choices=["A", "B"], default="B")
ap.add_argument("--no_class_weights", action="store_true")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--batch", type=int, default=2)
ap.add_argument("--accum", type=int, default=32)
ap.add_argument("--lr", type=float, default=2e-5)
ap.add_argument("--warmup_frac", type=float, default=0.05)
ap.add_argument("--eval_every", type=int, default=25)
ap.add_argument("--val_probe_n", type=int, default=400)
ap.add_argument("--lora_r", type=int, default=16)
ap.add_argument("--maxlen", type=int, default=4096)             # ASSUMPTION-a
ap.add_argument("--keep_ties", action="store_true")
ap.add_argument("--seed", type=int, default=20260818)
ap.add_argument("--dry_run", action="store_true",
                help="print manifest, class weights, one rendered prompt; no GPU/model")
a = ap.parse_args()
print(VERSION, flush=True)

rows = {}
for l in open(a.data):
    r = json.loads(l)
    rows[r["idx"]] = r
sp = json.load(open(a.splits))["splits"]

def label_idx(i):
    return COMBOS.index(rows[i]["label"])

def usable(i, scenario):
    if i not in rows:
        return False
    r = rows[i]
    if scenario == "A" and not r["solved"]:
        return False
    if r["solved"] and len(r["ties"]) > 1 and not a.keep_ties:
        return False        # tie handling inherited from v3 (G4 footnote)
    return True

tr = [i for i in sp["train"] if usable(i, a.scenario)]
# EVAL on the FULL frozen splits: ties are a training-label question, not an
# evaluation one — metrics come from the per-cell EM arrays (v1.2 fix)
va = [i for i in sp["val"] if i in rows]
te = [i for i in sp["test"] if i in rows]

# class weights: inverse frequency over the ACTIVE train labels
cnt = Counter(label_idx(i) for i in tr)
if a.no_class_weights:
    w = [1.0] * 4
else:
    n, k = sum(cnt.values()), 4
    w = [n / (k * max(1, cnt.get(c, 0))) for c in range(4)]

opt_steps_per_ep = (len(tr) + a.batch * a.accum - 1) // (a.batch * a.accum)
total_opt_steps = opt_steps_per_ep * a.epochs
warmup_steps = max(1, int(total_opt_steps * a.warmup_frac))
probe_ids = random.Random(a.seed).sample(va, min(a.val_probe_n, len(va)))

manifest = {"script": f"sft_router4.py ({VERSION})", "mode": "qpool",
            "scenario": a.scenario, "n_actions": 4, "codes": CODES,
            "code_legend": dict(zip(CODES, COMBOS)),
            "class_weights": [round(x, 3) for x in w],
            "class_weights_disabled": a.no_class_weights,
            "train_label_dist": {CODES[c]: cnt.get(c, 0) for c in range(4)},
            "epochs": a.epochs, "batch": a.batch, "accum": a.accum,
            "eff_batch": a.batch * a.accum, "lr": a.lr,
            "warmup_steps": warmup_steps, "total_opt_steps": total_opt_steps,
            "eval_every": a.eval_every, "val_probe_n": len(probe_ids),
            "lora_r": a.lora_r, "maxlen": a.maxlen,
            "keep_ties": a.keep_ties, "seed": a.seed,
            "router_model": a.router_model,
            "train": len(tr), "val": len(va), "test": len(te),
            "metric_primary": "lenient (whole-word containment, v1 rules — "
                              "pending quantifier-guard reconciliation)",
            "currency": "our FLOP units (ww2.0/ws5.7/sw5.7/ss9.4)"}
print(json.dumps(manifest, indent=2), flush=True)

def user_content(i):
    r = rows[i]
    return INSTR_POOL.format(q=r["question"], k=r["n_pool"], pool=r["pool_text"])

if a.dry_run:
    i0 = tr[0]
    print("\n===== DRY RUN: system prompt =====\n" + SYS_POOL)
    print("\n===== DRY RUN: rendered user content (idx %d, label %s) =====" %
          (i0, rows[i0]["label"]))
    print(user_content(i0)[:2500])
    print("\n... [truncated for display] ..." if len(user_content(i0)) > 2500 else "")
    print("\ndry run complete — no GPU touched.")
    raise SystemExit(0)

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                  set_peft_model_state_dict)

tok = AutoTokenizer.from_pretrained(a.router_model)
tok.padding_side = "left"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

# ---- answer-code head: single-token if possible, else sequence scoring ------
code_tok = [tok.encode(c, add_special_tokens=False) for c in CODES]
SINGLE = all(len(t) == 1 for t in code_tok)
manifest["label_head"] = "single_token" if SINGLE else "seq_score"
print(f"label head: {manifest['label_head']}  "
      f"(code token lengths {[len(t) for t in code_tok]})", flush=True)
if SINGLE:
    lab_ids = [t[0] for t in code_tok]

model = AutoModelForCausalLM.from_pretrained(
    a.router_model, torch_dtype=torch.float16, device_map="cuda",
    attn_implementation="sdpa")
model = get_peft_model(model, LoraConfig(
    r=a.lora_r, lora_alpha=2 * a.lora_r, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    task_type="CAUSAL_LM"))
for p in model.parameters():
    if p.requires_grad:
        p.data = p.data.float()
model.config.use_cache = False
model.gradient_checkpointing_enable()
model.enable_input_require_grads()
model.print_trainable_parameters()

W = None  # class-weight tensor, created lazily on cuda

def prompt_for(i):
    msgs = [{"role": "system", "content": SYS_POOL},
            {"role": "user", "content": user_content(i)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

def action_scores(ids):
    """[B, 4] scores over WW/WS/SW/SS."""
    texts = [prompt_for(i) for i in ids]
    if SINGLE:
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.maxlen).to("cuda")
        try:
            out = model(**enc, logits_to_keep=1).logits
        except TypeError:
            try:
                out = model(**enc, num_logits_to_keep=1).logits
            except TypeError:
                out = model(**enc).logits
        return out[:, -1][:, lab_ids].float()
    # seq-score fallback: log-likelihood of each code appended to the prompt
    scores = torch.zeros(len(ids), 4, device="cuda")
    for c in range(4):
        texts_c = [t + CODES[c] for t in texts]
        enc = tok(texts_c, return_tensors="pt", padding=True, truncation=True,
                  max_length=a.maxlen + 4).to("cuda")
        out = model(**enc).logits.float()
        L = len(code_tok[c])
        lp = torch.log_softmax(out[:, -L - 1:-1], dim=-1)
        tgt = torch.tensor(code_tok[c], device="cuda")
        scores[:, c] = lp.gather(-1, tgt.view(1, L, 1).expand(len(ids), L, 1)
                                 ).squeeze(-1).sum(-1)
    return scores

def cap(v, ss, orc):
    return 100 * (v - ss) / (orc - ss) if orc > ss + 1e-9 else float("nan")

def split_stats(ids, metric):
    ss = sum(rows[i][metric][SS] for i in ids) / len(ids)
    orc = sum(max(rows[i][metric]) for i in ids) / len(ids)
    return ss, orc

@torch.no_grad()
def probe(ids):
    model.eval()
    losses, picks = [], {}
    for b0 in range(0, len(ids), 16):
        b = ids[b0:b0 + 16]
        sc = action_scores(b)
        ys = torch.tensor([label_idx(i) for i in b], device="cuda")
        losses.append(F.cross_entropy(sc, ys, weight=W).item())
        for i, c in zip(b, sc.argmax(-1).tolist()):
            picks[i] = c
    model.train()
    lenient = sum(rows[i]["em_lenient"][picks[i]] for i in ids) / len(ids)
    ss_l, orc_l = split_stats(ids, "em_lenient")
    capture = cap(lenient, ss_l, orc_l)
    ent = Counter(picks.values())
    return sum(losses) / len(losses), capture, dict(
        (CODES[c], ent.get(c, 0)) for c in range(4))

@torch.no_grad()
def evaluate(ids, tag):
    model.eval(); pick = {}
    for b0 in range(0, len(ids), 16):
        b = ids[b0:b0 + 16]
        sc = action_scores(b)
        for i, c in zip(b, sc.argmax(-1).tolist()):
            pick[i] = c
    model.train()
    n = len(ids)
    res = {"picks": dict(Counter(CODES[pick[i]] for i in ids))}
    for name, metric in (("lenient", "em_lenient"), ("strict", "em_strict")):
        v = sum(rows[i][metric][pick[i]] for i in ids) / n
        ss, orc = split_stats(ids, metric)
        res[name] = {"router": round(100 * v, 2), "strong_strong": round(100 * ss, 2),
                     "oracle": round(100 * orc, 2), "capture_pct": round(cap(v, ss, orc), 1)}
    Rv = sum(rows[i]["R"][pick[i]] for i in ids) / n
    ssR, orcR = split_stats(ids, "R")
    res["R"] = {"router": round(Rv, 4), "strong_strong": round(ssR, 4),
                "oracle": round(orcR, 4), "capture_pct": round(cap(Rv, ssR, orcR), 1)}
    res["cost_pct_of_ss"] = round(
        100 * sum(COST[COMBOS[pick[i]]] for i in ids) / n / COST["strong-strong"], 1)
    print(f"[{tag}] LENIENT router {res['lenient']['router']}  "
          f"SS {res['lenient']['strong_strong']}  oracle {res['lenient']['oracle']}  "
          f"capture {res['lenient']['capture_pct']}%", flush=True)
    print(f"[{tag}] strict  router {res['strict']['router']}  "
          f"SS {res['strict']['strong_strong']}  oracle {res['strict']['oracle']}  "
          f"capture {res['strict']['capture_pct']}%  "
          f"cost {res['cost_pct_of_ss']}% of SS  picks {res['picks']}", flush=True)
    return res

W = torch.tensor(w, device="cuda", dtype=torch.float32)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
def lr_at(step):
    return a.lr * min(1.0, step / warmup_steps)

rnd = random.Random(a.seed)
best = {"capture": float("-inf"), "step": -1}
os.makedirs(a.out, exist_ok=True)
opt_step = 0
for ep in range(a.epochs):
    order = list(tr); rnd.shuffle(order)
    model.train(); micro = 0; opt.zero_grad()
    for b0 in range(0, len(order), a.batch):
        b = order[b0:b0 + a.batch]
        sc = action_scores(b)
        y = torch.tensor([label_idx(i) for i in b], device="cuda")
        loss = F.cross_entropy(sc, y, weight=W) / a.accum
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
                      f"loss {loss.item()*a.accum:.4f} lr {lr_at(opt_step):.2e}",
                      flush=True)
            if opt_step % a.eval_every == 0:
                pl, pcap, ppicks = probe(probe_ids)
                mark = ""
                if pcap == pcap and pcap > best["capture"]:
                    best = {"capture": pcap, "step": opt_step}
                    torch.save(get_peft_model_state_dict(model),
                               os.path.join(a.out, "best_adapter.pt"))
                    mark = "  <-- new best (adapter saved)"
                print(f"  [probe @ {opt_step}] loss {pl:.4f}  "
                      f"lenient-capture {pcap:.1f}%  picks {ppicks}{mark}",
                      flush=True)
    evaluate(va, f"val ep{ep} (last state)")

print(f"\nbest probe lenient-capture {best['capture']:.1f}% "
      f"at opt_step {best['step']}", flush=True)
res_last = evaluate(te, "TEST last-state")
if best["step"] > 0:
    set_peft_model_state_dict(
        model, torch.load(os.path.join(a.out, "best_adapter.pt")))
    res_best = evaluate(te, f"TEST best-checkpoint (opt_step {best['step']})")
else:
    res_best = None
model.save_pretrained(a.out)
json.dump({"manifest": manifest, "best_probe": best,
           "test_last": res_last, "test_best": res_best},
          open(os.path.join(a.out, "result.json"), "w"), indent=2)
print(f"saved -> {a.out}", flush=True)
