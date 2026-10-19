#!/usr/bin/env python3
# =============================================================================
# overfit_probe.py — VERSION: overfit_probe v1.3 2026-09-10 (v1.2 + logits_to_keep=1 — last-position logits only; math identical)
#
# PROF. PAN'S CODE-VALIDATION EXPERIMENT: can the SAME training mechanism
# (Qwen2.5-1.5B + LoRA + CE on the label token) MEMORIZE 10 instances?
#   * 10 instances, retrieval stage only, Label-Lenient targets
#   * batch = 1: ONE instance per optimizer step (300 steps = 30 epochs x 10)
#   * feed order: random.Random(42), shuffled without replacement per epoch —
#     BYTE-IDENTICAL to the published feed_schedule.xlsx
#   * every step prints one auditable line (six checks, deck slide 18)
#   * after each epoch: frozen-weights sweep over all 10 (accuracy n/10)
#   * after step 300: held-out act two (memorize vs generalize picture)
#
# OUTPUTS (the published target filenames):
#   <out>/overfit_probe.log   — written by tee/nohup redirection (see USAGE)
#   <out>/probe_steps.csv     — Excel mirror of every step line + sweep row
#   <out>/probe_graphs.png    — panels (a) accuracy (b) mean loss (c) per-instance
#
# DECLARED DESIGN CHOICES (pre-registered):
#   * loss = cross-entropy over the TWO label tokens (softmax over logits of
#     "A" and "B", then -ln p(label)) — this makes the printed p(W)/p(S) and
#     the printed loss one consistent system (checks 4-5) and is the
#     decision-relevant CE; the letter-logit readout convention of the project.
#   * NO class weights (10 instances fed equally; weighted path was exercised
#     by Runs 1-2), NO gradient accumulation, tie (p=0.5) predicts W.
#   * lr default 1e-4 (gentler than Run 1's batch-32 rate; batch-1 is noisy).
#
# PROMPTS AND LORA: VERBATIM from the lab's sft_router.py (verified 10 Sep):
#   SYS_STAGE + INSTR_RET copied byte-for-byte; pool text comes from the
#   jsonl's own pool_text/n_pool fields (the SAME bytes training fed);
#   LoRA r=16, alpha=2r=32, dropout=0.05, q/k/v/o_proj; trainable params
#   cast to float32 (sft_router.py L284-286). REVIEW-1/2: CLOSED.
#
# USAGE (lab, ~30-40 min on the card):
#   cd /mnt/hdd/jonathan/slr/stage-level-routing/lab_package
#   nohup python3 overfit_probe.py \
#       --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#       --out  /mnt/hdd/jonathan/runs/overfit_probe \
#       > /mnt/hdd/jonathan/runs/overfit_probe/overfit_probe.log 2>&1 &
# =============================================================================
import os, json, math, random, argparse, csv

os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")

VERSION = "overfit_probe v1.3 2026-09-10 (v1.2 + logits_to_keep=1 — last-position logits only; math identical)"
PROBE_IDX = [5, 6, 9, 10, 12, 13, 16, 17, 22, 23]   # first 10 labeled TRAIN instances (7 W / 3 S)

SYS_STAGE = ("You route one stage of a multi-stage question-answering pipeline. "
             "Each stage runs either Qwen2.5-1.5B (weak, cheap) or Qwen2.5-7B "
             "(strong, expensive). Choose the cheapest model for the stage "
             "described below that will still let the pipeline answer the "
             "question correctly.")
INSTR_RET = ("Stage: RETRIEVAL \u2014 selects the references relevant to the question "
             "from the candidate pool.\n"
             "A = weak retrieval (Qwen2.5-1.5B)\n"
             "B = strong retrieval (Qwen2.5-7B)\n"
             "Question: {q}\n"
             "Candidate pool ({k} passages, as given to the retrieval stage):\n{pool}\n"
             "Answer with one letter only.")


def build_schedule(epochs, seed=42):
    """BYTE-IDENTICAL to the generator that produced feed_schedule.xlsx."""
    rng = random.Random(seed)
    sched = []
    for ep in range(epochs):
        order = PROBE_IDX[:]
        rng.shuffle(order)
        for i in order:
            sched.append((len(sched) + 1, ep + 1, i))
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="router_data_lenient.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=24576)
    ap.add_argument("--use_idx", default="", help="comma list overriding the 10 (subset runs)")
    ap.add_argument("--single_idx", type=int, default=-1, help="train on ONE instance only")
    ap.add_argument("--dump_prompt", type=int, default=0, help="write step N's rendered prompt to a file and continue")
    ap.add_argument("--holdout_n", type=int, default=100, help="held-out labeled train instances for act two")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(VERSION, flush=True)

    global PROBE_IDX
    if a.single_idx >= 0:
        PROBE_IDX = [a.single_idx]
    elif a.use_idx:
        PROBE_IDX = [int(x) for x in a.use_idx.split(",")]
    print(f"instances: {PROBE_IDX}", flush=True)

    # ---- data: labels + pools -------------------------------------------------
    rows = {}
    for l in open(a.data):
        r = json.loads(l)
        rows[r["idx"]] = r
    for i in PROBE_IDX:
        assert rows[i]["ret_label"] in ("W", "S"), f"idx {i}: no retrieval label"
    labels = {i: rows[i]["ret_label"] for i in PROBE_IDX}
    print("labels:", labels, flush=True)

    for i in PROBE_IDX:
        for f in ("question", "pool_text", "n_pool"):
            assert f in rows[i], f"idx {i}: jsonl missing field {f}"

    # ---- held-out set: next labeled TRAIN instances not in the probe ----------
    holdout = [i for i, r in sorted(rows.items()) if r.get("split") == "train"
               and r["ret_label"] in ("W", "S") and i not in PROBE_IDX][: a.holdout_n]

    # ---- model ----------------------------------------------------------------
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    LAB = [tok.encode(c, add_special_tokens=False)[0] for c in ("A", "B")]  # A=W, B=S
    assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in ("A", "B")), "labels must be single tokens"

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
                                                 device_map="cuda", attn_implementation="sdpa")
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,          # = v5: alpha=2r, verified
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                      task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()                       # sft_router.py L284-286, verbatim
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()                 # memory: recompute activations in backward
    model.enable_input_require_grads()                    # needed for checkpointing + LoRA
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr)

    def render(i):
        r = rows[i]
        uc = INSTR_RET.format(q=r["question"], k=r["n_pool"], pool=r["pool_text"])
        return tok.apply_chat_template(
            [{"role": "system", "content": SYS_STAGE}, {"role": "user", "content": uc}],
            tokenize=False, add_generation_prompt=True)

    # length check (v5 discipline: no silent truncation)
    lens = [len(tok.encode(render(i))) for i in PROBE_IDX]
    print(f"[length check] {len(PROBE_IDX)} prompts, max_tokens={max(lens)} over_maxlen={sum(l > a.max_len for l in lens)}", flush=True)
    assert max(lens) <= a.max_len, "prompt exceeds max_len — raise --max_len"

    def forward_probs(i, grad):
        """(p_W, p_S, loss_tensor): softmax over the two label logits; loss = -ln p(label)."""
        enc = tok(render(i), return_tensors="pt", truncation=True, max_length=a.max_len).to("cuda")
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            try:
                out = model(**enc, logits_to_keep=1)              # last-position logits only (memory)
            except TypeError:
                out = model(**enc, num_logits_to_keep=1)          # older transformers naming
            logits2 = out.logits[0, -1][LAB]                      # [logit_A, logit_B]
            logp = torch.log_softmax(logits2.float(), dim=-1)    # renormalized over the pair
            y = 0 if labels_all[i] == "W" else 1
            loss = -logp[y]
        return math.exp(logp[0].item()), math.exp(logp[1].item()), loss

    labels_all = dict(labels)
    for i in holdout:
        labels_all[i] = rows[i]["ret_label"]

    def fmt(i, pW, pS, loss):
        lab = labels_all[i]
        pred = "W" if pW >= pS else "S"                           # tie -> W (declared)
        ok = "ok" if pred == lab else "WRONG"
        return lab, pred, ok, (f"label {lab} | p(W)={pW:.4f}  p(S)={pS:.4f} | "
                               f"predicted {pred} | loss {loss:.4f} | {ok}")

    sched = build_schedule(a.epochs)
    csvf = open(os.path.join(a.out, "probe_steps.csv"), "w", newline="")
    cw = csv.writer(csvf)
    cw.writerow(["kind", "step", "epoch", "idx", "label", "p_W", "p_S", "predicted", "loss", "correct"])

    sweeps = []   # (epoch, acc, mean_loss, {idx: p_correct})
    for step, epoch, i in sched:
        if a.dump_prompt == step:
            open(os.path.join(a.out, f"prompt_step{step}_idx{i}.txt"), "w").write(render(i))
            print(f"[dump] wrote prompt of step {step} (idx {i})", flush=True)
        model.train()
        pW, pS, loss = forward_probs(i, grad=True)
        lab, pred, ok, tail = fmt(i, pW, pS, loss.item())
        print(f"step {step} | epoch {epoch} | idx {i} | {tail}", flush=True)
        cw.writerow(["step", step, epoch, i, lab, f"{pW:.4f}", f"{pS:.4f}", pred, f"{loss.item():.4f}", int(ok == 'ok')])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % len(PROBE_IDX) == 0:                            # end of epoch -> sweep
            model.eval()
            print(f"=== sweep after epoch {epoch} (step {step}) ===", flush=True)
            pcs, losses, ncorr = {}, [], 0
            for j in sorted(PROBE_IDX):
                pW, pS, l = forward_probs(j, grad=False)
                lab, pred, ok, _ = fmt(j, pW, pS, l.item())
                pc = pW if lab == "W" else pS
                pcs[j] = pc; losses.append(l.item()); ncorr += (ok == "ok")
                print(f"  idx {j} | label {lab} | p({lab})={pc:.4f} | pred {pred} | loss {l.item():.4f} | {ok}", flush=True)
                cw.writerow(["sweep", step, epoch, j, lab, f"{pW:.4f}", f"{pS:.4f}", pred, f"{l.item():.4f}", int(ok == 'ok')])
            acc = ncorr / len(PROBE_IDX); ml = sum(losses) / len(losses)
            print(f"  accuracy {ncorr}/{len(PROBE_IDX)} = {acc:.2f} | mean loss {ml:.4f}", flush=True)
            sweeps.append((epoch, acc, ml, pcs))
            csvf.flush()

    # ---- act two: held-out ----------------------------------------------------
    if holdout:
        model.eval()
        print(f"=== ACT TWO: held-out {len(holdout)} labeled TRAIN instances (never fed) ===", flush=True)
        ncorr = 0; maj = sum(1 for j in holdout if labels_all[j] == "W")
        for j in holdout:
            pW, pS, l = forward_probs(j, grad=False)
            lab, pred, ok, _ = fmt(j, pW, pS, l.item())
            ncorr += (ok == "ok")
        print(f"held-out accuracy {ncorr}/{len(holdout)} = {ncorr/len(holdout):.3f} | "
              f"always-W baseline {maj}/{len(holdout)} = {maj/len(holdout):.3f}", flush=True)

    csvf.close()

    # ---- graphs ---------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E = [s[0] for s in sweeps]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    axes[0].step(E, [s[1] for s in sweeps], where="post", marker="o", ms=4, lw=2, color="#1F3A5F")
    axes[0].axhline(1.0, ls="--", lw=1, color="green"); axes[0].axhline(len([i for i in PROBE_IDX if labels[i]=='W'])/len(PROBE_IDX), ls="--", lw=1, color="grey")
    axes[0].set_ylim(0.5, 1.05); axes[0].set_title("(a) sweep accuracy — one point per epoch", fontsize=10)
    axes[0].set_xlabel("epoch (= %d optimizer steps)" % len(PROBE_IDX)); axes[0].set_ylabel("accuracy = correct / %d" % len(PROBE_IDX))
    axes[1].plot(E, [s[2] for s in sweeps], marker="o", ms=4, lw=2, color="#8B1A1A")
    axes[1].axhline(0.693, ls="--", lw=1, color="grey"); axes[1].axhline(0.05, ls="--", lw=1, color="green")
    axes[1].set_title("(b) sweep mean loss — mean of -ln p(correct)", fontsize=10)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    for i in PROBE_IDX:
        c = "#2C8FA3" if labels[i] == "W" else "#D2691E"
        axes[2].plot(E, [s[3][i] for s in sweeps], lw=1.4, color=c, alpha=0.9)
    axes[2].axhline(0.5, ls="--", lw=1, color="grey"); axes[2].set_ylim(0, 1.02)
    axes[2].set_title("(c) per-instance p(correct label) — blue=W, orange=S", fontsize=10)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("p(correct)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "probe_graphs.png"), bbox_inches="tight")
    print(f"saved -> {a.out}/probe_steps.csv, probe_graphs.png", flush=True)


if __name__ == "__main__":
    main()
