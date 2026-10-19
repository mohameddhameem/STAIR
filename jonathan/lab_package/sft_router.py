#!/usr/bin/env python3
# =============================================================================
# sft_router.py — VERSION: sft_router v1.0 2026-08-31  (stage-level design)
#
# ONE shared LoRA router, prompted per stage, trained in SEQUENTIAL phases
# (Prof. Pan, 22 Aug; deck slides 4, 6, 8):
#   Phase 1 RETRIEVAL : question + full candidate pool            -> A/B (W/S retriever)
#   Phase 2 QA        : question + retrieval choice + retrieved refs -> A/B (W/S answerer)
# Phase 2 continues from Phase 1's final state (D8). Probes every --eval_every
# optimizer steps in BOTH phases score BOTH stages (retrieval acc, QA acc on
# fixed validation probe sets; chance = ln 2 on the loss), so forgetting during
# Phase 2 is visible. Best checkpoint = highest mean(ret_acc, qa_acc).
#
# Data: router_data.jsonl from build_router_data.py (one row per instance:
#   pool_text, evidence_W/S, em_strict/em_lenient/R per cell, ret_label,
#   qa_label_W, qa_label_S). Training examples are expanded here:
#   retrieval example  <- ret_label  (skipped if None = unresolved, Option 1)
#   QA examples        <- qa_label_W and qa_label_S (each skipped if None)
#
# Evaluation (composed, stored-grid lookup, no model runs): retrieval decision
# -> QA decision on THAT retriever's references -> cell -> em/R/cost. Reported
# on the FULL frozen test split (deployment view) and the SOLVED subset
# (discriminative view), plus per-stage accuracies on labeled test rows.
#
# Flags = the decisions on slide 8 / plan D (defaults = the decided values):
#   --phase_mode sequential|mixed     D5 (mixed = both stages in every batch: forgetting variant)
#   --qa_choices both|labeled         D3 (labeled = QA example only for the labeled retrieval choice)
#   --scenario A|B                    D2 (A = drop unresolved [default]; B = label them W, retired rule)
#   --ret_input pool|question         variant: question-only retrieval router
#   --phase1_ckpt last|best           D8 (last = continue from Phase 1's final state [default])
#   --evidence_chars 0                D4 (0 = full retrieved references, bounded by the grid's output cap)
#   --no_class_weights                ablation
#   --maxlen 32768                    AgentTTS's effective input limit; NO silent truncation:
#                                     over-length prompts abort the run (answer-slot bug guard)
#   --smoke N                         pipeline check: N train examples per phase, one probe
#
# Inherited from sft_router4 v1.2 (validated 18 Aug): warmup, grad accumulation,
# left padding, fp16 base + fp32 LoRA (r16 on q/k/v/o), gradient checkpointing,
# answer-slot logits at the final position, best-adapter keeping, manifest.
# Structural fix: argparse lives inside main() (v4 parsed at import time).
#
# Usage (lab, venv active):
#   python3 sft_router.py --data /mnt/hdd/jonathan/runs/full_run_p100/router_data.jsonl \
#       --splits splits.json --out /mnt/hdd/jonathan/runs/sft_v5_1p5b
#   python3 sft_router.py ... --dry_run      # data, manifest, prompts, token lengths; no GPU
#   python3 sft_router.py ... --smoke 64     # end-to-end on a few examples
# =============================================================================
import os, json, argparse, random, math
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")
from collections import Counter

VERSION = "sft_router v1.0 2026-08-31"
COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
SS = 3
LETTERS = ["A", "B"]            # A = weak, B = strong — same for both stages
CLASSES = ["W", "S"]
STAGES = ["ret", "qa"]
CHOICE_WORD = {"W": "weak", "S": "strong"}

# ---- the three prompt constants (deck slide 6, verbatim) --------------------
SYS_STAGE = ("You route one stage of a multi-stage question-answering pipeline. "
             "Each stage runs either Qwen2.5-1.5B (weak, cheap) or Qwen2.5-7B "
             "(strong, expensive). Choose the cheapest model for the stage "
             "described below that will still let the pipeline answer the "
             "question correctly.")
INSTR_RET = ("Stage: RETRIEVAL — selects the references relevant to the question "
             "from the candidate pool.\n"
             "A = weak retrieval (Qwen2.5-1.5B)\n"
             "B = strong retrieval (Qwen2.5-7B)\n"
             "Question: {q}\n"
             "Candidate pool ({k} passages, as given to the retrieval stage):\n{pool}\n"
             "Answer with one letter only.")
INSTR_RET_QONLY = ("Stage: RETRIEVAL — selects the references relevant to the question "
                   "from the candidate pool (pool not shown).\n"
                   "A = weak retrieval (Qwen2.5-1.5B)\n"
                   "B = strong retrieval (Qwen2.5-7B)\n"
                   "Question: {q}\n"
                   "Answer with one letter only.")
INSTR_QA = ("Stage: ANSWERING — writes the answer using the retrieved references below.\n"
            "A = weak answering (Qwen2.5-1.5B)\n"
            "B = strong answering (Qwen2.5-7B)\n"
            "Question: {q}\n"
            "Retrieved references (selected by the {ret_choice} retriever):\n{evidence}\n"
            "Answer with one letter only.")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="router_data.jsonl from build_router_data.py")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--router_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phase_mode", choices=["sequential", "mixed"], default="sequential")
    ap.add_argument("--phases", default="ret,qa", help="which phases to run, in order (sequential mode)")
    ap.add_argument("--qa_choices", choices=["both", "labeled"], default="both")
    ap.add_argument("--scenario", choices=["A", "B"], default="A")
    ap.add_argument("--ret_input", choices=["pool", "question"], default="pool")
    ap.add_argument("--phase1_ckpt", choices=["last", "best"], default="last")
    ap.add_argument("--evidence_chars", type=int, default=0, help="0 = full retrieved references")
    ap.add_argument("--no_class_weights", action="store_true")
    ap.add_argument("--epochs", type=int, default=2, help="per phase")
    ap.add_argument("--batch", type=int, default=2, help="micro-batch (prompts per forward)")
    ap.add_argument("--accum", type=int, default=32, help="micro-batches per optimizer step (2x32 = 64 effective)")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--eval_every", type=int, default=25)
    ap.add_argument("--val_probe_n", type=int, default=400, help="probe examples PER STAGE")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--maxlen", type=int, default=32768)
    ap.add_argument("--seed", type=int, default=20260818)
    ap.add_argument("--smoke", type=int, default=0, help="N train examples per phase; one probe; then eval")
    ap.add_argument("--dry_run", action="store_true")
    return ap.parse_args()


# ---- data expansion: instances -> stage examples ---------------------------
def load_rows(path):
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["idx"]] = r
    return rows

def make_examples(rows, ids, a):
    """Return list of examples {idx, stage, ret_choice, label}. Unresolved rows
    yield nothing under Scenario A; under Scenario B they are labelled W."""
    ex = []
    for i in ids:
        r = rows[i]
        rl = r["ret_label"]
        if rl is None and a.scenario == "B": rl = "W"
        if rl is not None:
            ex.append({"idx": i, "stage": "ret", "ret_choice": None, "label": rl})
        choices = ["W", "S"] if a.qa_choices == "both" else ([r["ret_label"]] if r["ret_label"] else [])
        for ch in choices:
            ql = r[f"qa_label_{ch}"]
            if ql is None and a.scenario == "B": ql = "W"
            if ql is not None:
                ex.append({"idx": i, "stage": "qa", "ret_choice": ch, "label": ql})
    return ex

def class_weights(examples, stage, disabled):
    cnt = Counter(e["label"] for e in examples if e["stage"] == stage)
    if disabled: return {"W": 1.0, "S": 1.0}, cnt
    n = sum(cnt.values())
    return {c: n / (2 * max(1, cnt.get(c, 0))) for c in CLASSES}, cnt


def main():
    a = build_args()
    print(VERSION, flush=True)
    rows = load_rows(a.data)
    sp = json.load(open(a.splits))["splits"]
    tr_ids = [i for i in sp["train"] if i in rows]
    va_ids = [i for i in sp["val"] if i in rows]
    te_ids = [i for i in sp["test"] if i in rows]

    train_ex = make_examples(rows, tr_ids, a)
    val_ex = make_examples(rows, va_ids, a)
    rnd = random.Random(a.seed)
    probe = {}
    for st in STAGES:
        pool_st = [e for e in val_ex if e["stage"] == st]
        probe[st] = rnd.sample(pool_st, min(a.val_probe_n, len(pool_st)))

    phases = a.phases.split(",") if a.phase_mode == "sequential" else ["mixed"]
    by_stage = {st: [e for e in train_ex if e["stage"] == st] for st in STAGES}
    if a.smoke:
        for st in STAGES:
            by_stage[st] = by_stage[st][:a.smoke]
        train_ex = by_stage["ret"] + by_stage["qa"]
    W, cnt = {}, {}
    for st in STAGES:
        W[st], cnt[st] = class_weights(train_ex, st, a.no_class_weights)

    def phase_examples(ph):
        return train_ex if ph == "mixed" else by_stage[ph]
    eff = a.batch * a.accum
    steps_per_phase = {ph: a.epochs * ((len(phase_examples(ph)) + eff - 1) // eff) for ph in phases}
    total_opt_steps = sum(steps_per_phase.values())

    manifest = {"script": f"sft_router.py ({VERSION})", "design": "stage-level, one shared router",
                "phase_mode": a.phase_mode, "phases": phases, "qa_choices": a.qa_choices,
                "scenario": a.scenario, "ret_input": a.ret_input, "phase1_ckpt": a.phase1_ckpt,
                "evidence_chars": a.evidence_chars, "letters": LETTERS,
                "letter_legend": {"A": "weak", "B": "strong"},
                "class_weights": {st: {c: round(W[st][c], 3) for c in CLASSES} for st in STAGES},
                "class_weights_disabled": a.no_class_weights,
                "train_label_dist": {st: dict(cnt[st]) for st in STAGES},
                "train_examples": {st: len(by_stage[st]) for st in STAGES},
                "probe_n": {st: len(probe[st]) for st in STAGES},
                "epochs_per_phase": a.epochs, "batch": a.batch, "accum": a.accum, "eff_batch": eff,
                "steps_per_phase": steps_per_phase, "total_opt_steps": total_opt_steps,
                "lr": a.lr, "warmup_frac": a.warmup_frac, "eval_every": a.eval_every,
                "lora_r": a.lora_r, "maxlen": a.maxlen, "seed": a.seed, "smoke": a.smoke,
                "router_model": a.router_model,
                "instances": {"train": len(tr_ids), "val": len(va_ids), "test": len(te_ids)},
                "decoding": "router reads A/B logits at the final position; argmax",
                "metric_primary": "lenient (whole-word containment, v1 rules — pending quantifier-guard reconciliation)",
                "currency": "our FLOP units (ww2.0/ws5.7/sw5.7/ss9.4)"}
    print(json.dumps(manifest, indent=2), flush=True)

    # ---- prompt rendering ----------------------------------------------------
    def evidence(r, ch):
        ev = r[f"evidence_{ch}"]
        return ev if a.evidence_chars <= 0 else ev[:a.evidence_chars]

    def user_content(e):
        r = rows[e["idx"]]
        if e["stage"] == "ret":
            if a.ret_input == "question":
                return INSTR_RET_QONLY.format(q=r["question"])
            return INSTR_RET.format(q=r["question"], k=r["n_pool"], pool=r["pool_text"])
        ch = e["ret_choice"]
        return INSTR_QA.format(q=r["question"], ret_choice=CHOICE_WORD[ch], evidence=evidence(r, ch))

    if a.dry_run:
        print("\n===== DRY RUN: SYS_STAGE =====\n" + SYS_STAGE)
        for st in STAGES:
            e0 = next(e for e in train_ex if e["stage"] == st)
            print(f"\n===== DRY RUN: {st.upper()} user content (idx {e0['idx']}, "
                  f"ret_choice {e0['ret_choice']}, label {e0['label']}) =====")
            uc = user_content(e0)
            print(uc[:2500] + ("\n... [truncated for display] ..." if len(uc) > 2500 else ""))
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(a.router_model)
            for st in STAGES:
                L = []
                for e in [x for x in train_ex if x["stage"] == st][:300]:
                    msgs = [{"role": "system", "content": SYS_STAGE}, {"role": "user", "content": user_content(e)}]
                    L.append(len(tok(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)).input_ids))
                L.sort()
                print(f"\n{st}: prompt tokens over {len(L)} examples — min {L[0]} median {L[len(L)//2]} "
                      f"p90 {L[int(len(L)*.9)]} max {L[-1]}  over maxlen({a.maxlen}): {sum(x > a.maxlen for x in L)}")
        except Exception as ex:
            print(f"(token-length check skipped: {ex})")
        print("\ndry run complete — no GPU touched.")
        return

    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import (LoraConfig, get_peft_model, get_peft_model_state_dict,
                      set_peft_model_state_dict)

    tok = AutoTokenizer.from_pretrained(a.router_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # ---- answer letters must be single, distinct tokens (hard fail) ----------
    let_tok = [tok.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(t) == 1 for t in let_tok) and len(set(t[0] for t in let_tok)) == 2, \
        f"answer letters are not single distinct tokens: {list(zip(LETTERS, let_tok))}"
    lab_ids = [t[0] for t in let_tok]
    manifest["letter_token_ids"] = lab_ids

    def prompt_for(e):
        msgs = [{"role": "system", "content": SYS_STAGE}, {"role": "user", "content": user_content(e)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # ---- NO silent truncation: check every prompt we will ever feed ----------
    def check_lengths(exs, tag):
        over, mx = 0, 0
        for e in exs:
            n = len(tok(prompt_for(e)).input_ids)
            mx = max(mx, n); over += n > a.maxlen
        print(f"[length check] {tag}: n={len(exs)} max_tokens={mx} over_maxlen={over}", flush=True)
        assert over == 0, f"{over} {tag} prompts exceed --maxlen {a.maxlen}: raise maxlen (truncation would move the answer slot)"
        return mx
    manifest["max_prompt_tokens"] = {
        "train_ret": check_lengths(by_stage["ret"], "train ret"),
        "train_qa": check_lengths(by_stage["qa"], "train qa"),
        "probe_ret": check_lengths(probe["ret"], "probe ret"),
        "probe_qa": check_lengths(probe["qa"], "probe qa")}

    model = AutoModelForCausalLM.from_pretrained(
        a.router_model, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa")
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

    def scores(exs):
        """[B, 2] logits over A/B at the final position (the assistant tag)."""
        texts = [prompt_for(e) for e in exs]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=False).to("cuda")
        try:
            out = model(**enc, logits_to_keep=1).logits
        except TypeError:
            try:
                out = model(**enc, num_logits_to_keep=1).logits
            except TypeError:
                out = model(**enc).logits
        return out[:, -1][:, lab_ids].float()

    def weighted_ce(sc, exs):
        y = torch.tensor([CLASSES.index(e["label"]) for e in exs], device="cuda")
        w = torch.tensor([W[e["stage"]][e["label"]] for e in exs], device="cuda", dtype=torch.float32)
        return (F.cross_entropy(sc, y, reduction="none") * w).sum() / w.sum()

    @torch.no_grad()
    def probe_stage(st):
        model.eval()
        losses, correct = [], 0
        exs = probe[st]
        bs = 1 if st == "ret" else 8
        for b0 in range(0, len(exs), bs):
            b = exs[b0:b0 + bs]
            sc = scores(b)
            losses.append(weighted_ce(sc, b).item())
            pred = sc.argmax(-1).tolist()
            correct += sum(CLASSES[p] == e["label"] for p, e in zip(pred, b))
        model.train()
        return (sum(losses) / max(1, len(losses))), correct / max(1, len(exs))

    def run_probe(phase, step):
        rl, racc = probe_stage("ret"); ql, qacc = probe_stage("qa")
        rec = {"phase": phase, "opt_step": step, "ret_loss": round(rl, 4), "ret_acc": round(racc, 4),
               "qa_loss": round(ql, 4), "qa_acc": round(qacc, 4), "mean_acc": round((racc + qacc) / 2, 4)}
        return rec

    # ---- composed evaluation via stored-grid lookup --------------------------
    def cap(v, ss, orc):
        return 100 * (v - ss) / (orc - ss) if orc > ss + 1e-9 else float("nan")

    @torch.no_grad()
    def decide(exs, bs=8):
        model.eval(); out = {}
        for b0 in range(0, len(exs), bs):
            b = exs[b0:b0 + bs]
            for e, p in zip(b, scores(b).argmax(-1).tolist()):
                out[(e["idx"], e["stage"], e["ret_choice"])] = CLASSES[p]
        model.train()
        return out

    def evaluate(ids, tag):
        ret_dec = decide([{"idx": i, "stage": "ret", "ret_choice": None, "label": "W"} for i in ids], bs=1)
        qa_ex = [{"idx": i, "stage": "qa", "ret_choice": ret_dec[(i, "ret", None)], "label": "W"} for i in ids]
        qa_dec = decide(qa_ex)
        cell = {}
        for i in ids:
            rc = ret_dec[(i, "ret", None)]; qc = qa_dec[(i, "qa", rc)]
            cell[i] = COMBOS.index(f"{CHOICE_WORD[rc]}-{CHOICE_WORD[qc]}")
        res = {"picks": dict(Counter(COMBOS[cell[i]] for i in ids)),
               "ret_picks": dict(Counter(ret_dec[(i, "ret", None)] for i in ids))}
        def block(sub):
            n = len(sub); b = {"n": n}
            for name, metric in (("lenient", "em_lenient"), ("strict", "em_strict")):
                v = sum(rows[i][metric][cell[i]] for i in sub) / n
                ss = sum(rows[i][metric][SS] for i in sub) / n
                orc = sum(max(rows[i][metric]) for i in sub) / n
                b[name] = {"router": round(100 * v, 2), "strong_strong": round(100 * ss, 2),
                           "oracle": round(100 * orc, 2), "capture_pct": round(cap(v, ss, orc), 1)}
            Rv = sum(rows[i]["R"][cell[i]] for i in sub) / n
            ssR = sum(rows[i]["R"][SS] for i in sub) / n
            orcR = sum(max(rows[i]["R"]) for i in sub) / n
            b["R"] = {"router": round(Rv, 4), "strong_strong": round(ssR, 4),
                      "oracle": round(orcR, 4), "capture_pct": round(cap(Rv, ssR, orcR), 1)}
            b["cost_pct_of_ss"] = round(100 * sum(COST[COMBOS[cell[i]]] for i in sub) / n / COST["strong-strong"], 1)
            return b
        res["full"] = block(ids)
        solved = [i for i in ids if any(rows[i]["em_strict"])]
        res["solved_subset"] = block(solved) if solved else None
        # per-stage accuracy on labeled test rows (the discriminative numbers)
        lab_r = [i for i in ids if rows[i]["ret_label"]]
        res["ret_acc_labeled"] = round(sum(ret_dec[(i, "ret", None)] == rows[i]["ret_label"] for i in lab_r) / max(1, len(lab_r)), 4)
        qa_hits = qa_n = 0
        for i in ids:
            rc = ret_dec[(i, "ret", None)]; lab = rows[i][f"qa_label_{rc}"]
            if lab: qa_n += 1; qa_hits += qa_dec[(i, "qa", rc)] == lab
        res["qa_acc_labeled_on_chosen_retriever"] = round(qa_hits / max(1, qa_n), 4)
        f = res["full"]
        print(f"[{tag}] FULL  lenient router {f['lenient']['router']} SS {f['lenient']['strong_strong']} "
              f"oracle {f['lenient']['oracle']} capture {f['lenient']['capture_pct']}% | strict router "
              f"{f['strict']['router']} | cost {f['cost_pct_of_ss']}% of SS | picks {res['picks']}", flush=True)
        if res["solved_subset"]:
            s = res["solved_subset"]
            print(f"[{tag}] SOLVED lenient router {s['lenient']['router']} SS {s['lenient']['strong_strong']} "
                  f"oracle {s['lenient']['oracle']} capture {s['lenient']['capture_pct']}% (n={s['n']})", flush=True)
        print(f"[{tag}] per-stage acc on labeled rows: retrieval {res['ret_acc_labeled']}  "
              f"QA(on chosen retriever) {res['qa_acc_labeled_on_chosen_retriever']}", flush=True)
        return res

    # ---- training: sequential phases (or one mixed phase) --------------------
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    os.makedirs(a.out, exist_ok=True)
    history, best = [], {"mean_acc": -1.0, "opt_step": -1, "phase": None}
    phase_best = {}
    opt_step = 0
    rnd = random.Random(a.seed + 1)
    for ph in phases:
        exs_all = phase_examples(ph)
        n_steps = steps_per_phase[ph]
        warm = max(1, int(n_steps * a.warmup_frac))
        ph_step0 = opt_step
        if ph == "qa" and a.phase1_ckpt == "best" and "ret" in phase_best:
            set_peft_model_state_dict(model, torch.load(os.path.join(a.out, "phase_ret_best.pt")))
            print("[phase qa] starting from Phase 1 BEST checkpoint (--phase1_ckpt best)", flush=True)
        print(f"\n===== PHASE {ph}: {len(exs_all)} examples, {n_steps} opt steps, warmup {warm} =====", flush=True)
        for ep in range(a.epochs):
            order = list(exs_all); rnd.shuffle(order)
            model.train(); micro = 0; opt.zero_grad()
            for b0 in range(0, len(order), a.batch):
                b = order[b0:b0 + a.batch]
                loss = weighted_ce(scores(b), b) / a.accum
                loss.backward(); micro += 1
                if micro % a.accum == 0 or b0 + a.batch >= len(order):
                    for g in opt.param_groups:
                        g["lr"] = a.lr * min(1.0, (opt_step - ph_step0 + 1) / warm)
                    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                    opt.step(); opt.zero_grad(); opt_step += 1
                    if opt_step % 5 == 0:
                        print(f"[{ph} ep{ep}] opt_step {opt_step}/{total_opt_steps} loss {loss.item()*a.accum:.4f}", flush=True)
                    if opt_step % a.eval_every == 0 or (a.smoke and micro % a.accum == 0):
                        rec = run_probe(ph, opt_step); history.append(rec)
                        mark = ""
                        if rec["mean_acc"] > best["mean_acc"]:
                            best = {"mean_acc": rec["mean_acc"], "opt_step": opt_step, "phase": ph}
                            torch.save(get_peft_model_state_dict(model), os.path.join(a.out, "best_adapter.pt"))
                            mark += "  <-- new BEST (mean acc; adapter saved)"
                        st_key = "ret_acc" if ph == "ret" else "qa_acc"
                        if ph in STAGES and rec[st_key] > phase_best.get(ph, -1):
                            phase_best[ph] = rec[st_key]
                            torch.save(get_peft_model_state_dict(model), os.path.join(a.out, f"phase_{ph}_best.pt"))
                        print(f"  [probe @ {opt_step}] ret loss {rec['ret_loss']} acc {rec['ret_acc']} | "
                              f"qa loss {rec['qa_loss']} acc {rec['qa_acc']} | chance loss ln2=0.693{mark}", flush=True)
                        if a.smoke: break
                if a.smoke and micro >= a.accum: break
            if a.smoke: break
        torch.save(get_peft_model_state_dict(model), os.path.join(a.out, f"phase_{ph}_last.pt"))

    print(f"\nbest probe mean acc {best['mean_acc']} at opt_step {best['opt_step']} (phase {best['phase']})", flush=True)
    res_last = evaluate(te_ids, "TEST last-state")
    res_best = None
    if best["opt_step"] > 0:
        set_peft_model_state_dict(model, torch.load(os.path.join(a.out, "best_adapter.pt")))
        res_best = evaluate(te_ids, f"TEST best-checkpoint (opt_step {best['opt_step']})")
    model.save_pretrained(a.out)
    json.dump({"manifest": manifest, "probe_history": history, "best_probe": best,
               "phase_best_acc": phase_best, "test_last": res_last, "test_best": res_best},
              open(os.path.join(a.out, "result.json"), "w"), indent=2)
    print(f"saved -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
