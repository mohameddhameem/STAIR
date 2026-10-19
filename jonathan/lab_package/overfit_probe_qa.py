#!/usr/bin/env python3
# =============================================================================
# overfit_probe_qa.py — VERSION: overfit_probe_qa v1.0 2026-09-11
#
# PHASE-2 TWIN of overfit_probe v1.3: same harness, same six-checks contract,
# now exercising the QA-stage code path (INSTR_QA prompt branch: question +
# retrieval choice + that retriever's evidence -> which ANSWERER, A/B).
#
# 10 QA examples (an example = instance x retrieval scenario), deterministic:
# the first 7 W-labeled and first 3 S-labeled QA examples in file order
# over TRAIN instances (scenario order W,S within an instance):
#   (5,S,W) (5,W,W) (6,S,W) (6,W,W) (9,S,W) (10,S,W) (10,W,W)
#   (13,S,S) (13,W,S) (22,S,S)
# The script ASSERTS each jsonl label equals the expected label above and
# aborts on any mismatch (local derivation vs lab file must agree).
#
# Same declared choices as the retrieval probe: batch 1 (one example per
# optimizer step), no class weights, lr 1e-4, tie -> W, loss = CE over the
# two label tokens = -ln p(label), seed-42 shuffle per epoch (published in
# feed_schedule_qa.xlsx), gradient checkpointing + logits_to_keep=1.
#
# OUTPUTS: <out>/overfit_probe_qa.log (via nohup), probe_qa_steps.csv,
#          probe_qa_graphs.png  ·  ACT TWO: next 100 unseen QA examples.
#
# USAGE (lab, ~5-10 min — QA prompts are short):
#   cd /mnt/hdd/jonathan/slr/stage-level-routing/lab_package
#   mkdir -p /mnt/hdd/jonathan/runs/overfit_probe_qa
#   nohup python3 overfit_probe_qa.py \
#       --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#       --out  /mnt/hdd/jonathan/runs/overfit_probe_qa \
#       > /mnt/hdd/jonathan/runs/overfit_probe_qa/overfit_probe_qa.log 2>&1 &
# =============================================================================
import os, json, math, random, argparse, csv

os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")

VERSION = "overfit_probe_qa v1.0 2026-09-11"
QA_EX = [(5, "S", "W"), (5, "W", "W"), (6, "S", "W"), (6, "W", "W"), (9, "S", "W"),
         (10, "S", "W"), (10, "W", "W"), (13, "S", "S"), (13, "W", "S"), (22, "S", "S")]

CHOICE_WORD = {"W": "weak", "S": "strong"}
SYS_STAGE = ("You route one stage of a multi-stage question-answering pipeline. "
             "Each stage runs either Qwen2.5-1.5B (weak, cheap) or Qwen2.5-7B "
             "(strong, expensive). Choose the cheapest model for the stage "
             "described below that will still let the pipeline answer the "
             "question correctly.")
INSTR_QA = ("Stage: ANSWERING \u2014 writes the answer using the retrieved references below.\n"
            "A = weak answering (Qwen2.5-1.5B)\n"
            "B = strong answering (Qwen2.5-7B)\n"
            "Question: {q}\n"
            "Retrieved references (selected by the {ret_choice} retriever):\n{evidence}\n"
            "Answer with one letter only.")

QA_KEYS = {"W": ["qa_given_W", "label_QA_given_W", "qa_label_W", "qa_W"],
           "S": ["qa_given_S", "label_QA_given_S", "qa_label_S", "qa_S"]}


def qa_label(row, scen):
    for k in QA_KEYS[scen]:
        if k in row:
            return row[k]
    raise KeyError(f"no QA label field for scenario {scen}; row keys: {sorted(row.keys())[:20]}")


def build_schedule(epochs, keys, seed=42):
    rng = random.Random(seed)
    sched = []
    for ep in range(epochs):
        order = keys[:]
        rng.shuffle(order)
        for k in order:
            sched.append((len(sched) + 1, ep + 1, k))
    return sched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max_len", type=int, default=8192)
    ap.add_argument("--holdout_n", type=int, default=100)
    ap.add_argument("--dump_prompt", type=int, default=0)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(VERSION, flush=True)

    rows = {}
    for l in open(a.data):
        r = json.loads(l)
        rows[r["idx"]] = r

    # verify the hardcoded selection against the lab file — abort on any mismatch
    labels = {}
    for i, scen, expect in QA_EX:
        got = qa_label(rows[i], scen)
        assert got == expect, f"LABEL MISMATCH idx {i} scen {scen}: jsonl says {got!r}, expected {expect!r} — STOP"
        labels[(i, scen)] = got
    keys = [(i, s) for i, s, _ in QA_EX]
    print("examples:", [(i, s, labels[(i, s)]) for i, s in keys], flush=True)

    # held-out: next QA examples in file order, either scenario, not in the probe
    holdout = []
    for i in sorted(rows):
        r = rows[i]
        if r.get("split") != "train":
            continue
        for scen in ("W", "S"):
            try:
                lab = qa_label(r, scen)
            except KeyError:
                lab = None
            if lab in ("W", "S") and (i, scen) not in labels:
                holdout.append((i, scen, lab))
        if len(holdout) >= a.holdout_n:
            break
    holdout = holdout[: a.holdout_n]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    LAB = [tok.encode(c, add_special_tokens=False)[0] for c in ("A", "B")]

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.float16,
                                                 device_map="cuda", attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                                             task_type="CAUSAL_LM"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    model.print_trainable_parameters()
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=a.lr)

    def render(i, scen):
        r = rows[i]
        uc = INSTR_QA.format(q=r["question"], ret_choice=CHOICE_WORD[scen], evidence=r[f"evidence_{scen}"])
        return tok.apply_chat_template(
            [{"role": "system", "content": SYS_STAGE}, {"role": "user", "content": uc}],
            tokenize=False, add_generation_prompt=True)

    lens = [len(tok.encode(render(i, s))) for i, s in keys]
    print(f"[length check] {len(keys)} prompts, max_tokens={max(lens)} over_maxlen={sum(l > a.max_len for l in lens)}", flush=True)
    assert max(lens) <= a.max_len

    all_labels = dict(labels)
    for i, s, lab in holdout:
        all_labels[(i, s)] = lab

    def forward_probs(key, grad):
        i, scen = key
        enc = tok(render(i, scen), return_tensors="pt", truncation=True, max_length=a.max_len).to("cuda")
        ctx = torch.enable_grad() if grad else torch.no_grad()
        with ctx:
            try:
                out = model(**enc, logits_to_keep=1)
            except TypeError:
                out = model(**enc, num_logits_to_keep=1)
            logits2 = out.logits[0, -1][LAB]
            logp = torch.log_softmax(logits2.float(), dim=-1)
            y = 0 if all_labels[key] == "W" else 1
            loss = -logp[y]
        return math.exp(logp[0].item()), math.exp(logp[1].item()), loss

    def fmt(key, pW, pS, loss):
        lab = all_labels[key]
        pred = "W" if pW >= pS else "S"
        ok = "ok" if pred == lab else "WRONG"
        return lab, pred, ok, (f"label {lab} | p(W)={pW:.4f}  p(S)={pS:.4f} | "
                               f"predicted {pred} | loss {loss:.4f} | {ok}")

    sched = build_schedule(a.epochs, keys)
    csvf = open(os.path.join(a.out, "probe_qa_steps.csv"), "w", newline="")
    cw = csv.writer(csvf)
    cw.writerow(["kind", "step", "epoch", "idx", "scen", "label", "p_W", "p_S", "predicted", "loss", "correct"])

    sweeps = []
    for step, epoch, key in sched:
        i, scen = key
        if a.dump_prompt == step:
            open(os.path.join(a.out, f"prompt_step{step}_idx{i}{scen}.txt"), "w").write(render(i, scen))
        model.train()
        pW, pS, loss = forward_probs(key, grad=True)
        lab, pred, ok, tail = fmt(key, pW, pS, loss.item())
        print(f"step {step} | epoch {epoch} | idx {i}/{scen} | {tail}", flush=True)
        cw.writerow(["step", step, epoch, i, scen, lab, f"{pW:.4f}", f"{pS:.4f}", pred, f"{loss.item():.4f}", int(ok == 'ok')])
        opt.zero_grad(); loss.backward(); opt.step()

        if step % len(keys) == 0:
            model.eval()
            print(f"=== sweep after epoch {epoch} (step {step}) ===", flush=True)
            pcs, losses, nc = {}, [], 0
            for k2 in keys:
                pW, pS, l = forward_probs(k2, grad=False)
                lab, pred, ok, _ = fmt(k2, pW, pS, l.item())
                pc = pW if lab == "W" else pS
                pcs[k2] = pc; losses.append(l.item()); nc += (ok == "ok")
                print(f"  idx {k2[0]}/{k2[1]} | label {lab} | p({lab})={pc:.4f} | pred {pred} | loss {l.item():.4f} | {ok}", flush=True)
                cw.writerow(["sweep", step, epoch, k2[0], k2[1], lab, f"{pW:.4f}", f"{pS:.4f}", pred, f"{l.item():.4f}", int(ok == 'ok')])
            print(f"  accuracy {nc}/{len(keys)} = {nc/len(keys):.2f} | mean loss {sum(losses)/len(losses):.4f}", flush=True)
            sweeps.append((epoch, nc / len(keys), sum(losses) / len(losses), pcs))
            csvf.flush()

    if holdout:
        model.eval()
        print(f"=== ACT TWO: held-out {len(holdout)} QA examples (never fed) ===", flush=True)
        nc = 0; maj = sum(1 for _, _, lab in holdout if lab == "W")
        for i, s, lab in holdout:
            pW, pS, l = forward_probs((i, s), grad=False)
            _, pred, ok, _ = fmt((i, s), pW, pS, l.item())
            nc += (ok == "ok")
        print(f"held-out accuracy {nc}/{len(holdout)} = {nc/len(holdout):.3f} | "
              f"always-W baseline {maj}/{len(holdout)} = {maj/len(holdout):.3f}", flush=True)
    csvf.close()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    E = [s[0] for s in sweeps]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), dpi=150)
    axes[0].step(E, [s[1] for s in sweeps], where="post", marker="o", ms=4, lw=2, color="#1F3A5F")
    axes[0].axhline(1.0, ls="--", lw=1, color="green"); axes[0].axhline(0.7, ls="--", lw=1, color="grey")
    axes[0].set_ylim(0.5, 1.05); axes[0].set_title("(a) QA sweep accuracy", fontsize=10)
    axes[0].set_xlabel("epoch (= 10 optimizer steps)"); axes[0].set_ylabel("accuracy = correct / 10")
    axes[1].plot(E, [s[2] for s in sweeps], marker="o", ms=4, lw=2, color="#8B1A1A")
    axes[1].axhline(0.693, ls="--", lw=1, color="grey"); axes[1].axhline(0.05, ls="--", lw=1, color="green")
    axes[1].set_title("(b) QA sweep mean loss", fontsize=10); axes[1].set_xlabel("epoch"); axes[1].set_ylabel("loss")
    for k in keys:
        lab = labels[k]
        c = "#2C8FA3" if lab == "W" else "#D2691E"
        axes[2].plot(E, [s[3][k] for s in sweeps], lw=1.4, color=c, alpha=0.9)
    axes[2].axhline(0.5, ls="--", lw=1, color="grey"); axes[2].set_ylim(0, 1.02)
    axes[2].set_title("(c) per-example p(correct label) — blue=W, orange=S", fontsize=10)
    axes[2].set_xlabel("epoch"); axes[2].set_ylabel("p(correct)")
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "probe_qa_graphs.png"), bbox_inches="tight")
    print(f"saved -> {a.out}/probe_qa_steps.csv, probe_qa_graphs.png", flush=True)


if __name__ == "__main__":
    main()
