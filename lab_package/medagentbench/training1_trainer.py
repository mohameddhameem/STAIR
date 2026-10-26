#!/usr/bin/env python3
# =============================================================================
# training1_trainer.py — VERSION: v1.0 2026-10-26
# "Training 1 — risk-free training, risk tested as a guardrail at readout."
#
# MedAgentBench variant of the audited grpo_trainer.py (v1.2, HotpotQA):
# SAME mechanics kept line-for-line in spirit —
#   * GRPO with AMENDMENT A1 advantage = R' - group mean (NO /std),
#   * entropy bonus BETA, LR 1e-5, grad clip 1.0, accum 8, seed 17,
#   * non-finite-loss skip, checkpoint by THE OBJECTIVE (val mean-R'),
#   * manifest discipline, smoke mode first.
# STRUCTURAL CHANGES (and nothing else):
#   1) ONE router, ONE decision per instance: W = DELIVER the weak attempt,
#      S = ESCALATE to the strong model (weak-first cascade; the weak attempt
#      already exists for every instance).
#   2) Outcomes REPLAYED from the Training 1 table (both paths measured for
#      all 300 instances — training is execution-free):
#        R'(W) = qw - LAMBDA*(cw/cs)        - max(0, cw/cs - 1)
#        R'(S) = qs - LAMBDA*((cw+cs)/cs)   - max(0, (cw+cs)/cs - 1)
#      pass/fail correctness; per-instance C_SS_i = cs (measured strong cost);
#      weak-first costs: deliver pays cw, escalate pays cw + cs.
#   3) Router prompt = the validated MedAgentBench prompt (registered 26 Oct):
#      task-definition wrapper + the HotpotQA champion's objective sentence
#      VERBATIM; no principles. Self-check below refuses to run on drift.
#   4) Fresh LoRA only (no MedAgentBench warm start exists); adapter SHAPE
#      copied from the HotpotQA SFT ret adapter's own adapter_config.json —
#      same lineage rule as the audited --from-scratch path.
# NO risk anywhere in this file: tau and the guardrail live in the readout
# tally, never in training.
#
# Usage (lab, slr-env):
#   python3 training1_trainer.py --smoke
#   python3 training1_trainer.py --baseline 0.1890     # always-escalate val bar
# =============================================================================
import os, sys, json, time, random, argparse, hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

VERSION = "training1_trainer v1.0 2026-10-26 (MedAgentBench deliver/escalate, replayed, A1)"
TABLE   = "/mnt/hdd/jonathan/runs/mab/training1_table.json"
MODEL   = "Qwen/Qwen2.5-1.5B-Instruct"
LORA_SHAPE_FROM = "/mnt/hdd/jonathan/runs/config_b/sft_router5_ret/best_adapter"
WEAK_NAME, STRONG_NAME = "Qwen2.5-7B-Instruct", "Qwen2.5-72B-Instruct"
LR, ACCUM, SEED, BETA, LAMBDA = 1e-5, 8, 17, 0.03, 0.10
EVAL_EVERY = 10              # optimizer steps (90 rows / accum 8 ~ 11 steps/epoch)

# ---- the registered router prompt (validated by Jonathan, 26 Oct) -----------
SYS = ("You route a medical record-system task. The weak model {weak_name} has "
       "already attempted it; choose whether to deliver its result or escalate "
       "to {strong_name}. Your objective: the task should be completed correctly "
       "while spending as little compute as possible - correctness first, cost "
       "second. Based on the information shown below, choose one option."
       ).format(weak_name=WEAK_NAME, strong_name=STRONG_NAME)
USER_TAIL = ("\n\nOptions:\nW) Deliver the weak model's result\n"
             "S) Escalate to the strong model\nAnswer with a single letter: W or S.")

def _prompt_selfcheck():
    assert ("Your objective: the task should be completed correctly while "
            "spending as little compute as possible - correctness first, cost "
            "second.") in SYS, "objective sentence drifted"
    assert "Based on the information shown below, choose one option." in SYS
    for banned in ("Routing principles", "only when needed", "(1)", "(2)", "(3)",
                   "cheap)", "expensive)"):
        assert banned not in SYS, f"banned (principle-flavored) phrase: {banned}"
    print("[training1 prompt] self-check OK", flush=True)
_prompt_selfcheck()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hinge(x): return max(0.0, x - 1.0)


def build_rows():
    """One row per instance: the router input (task text + weak-attempt view,
    from the registered table — never the grader's verdict) + the two replayed
    branch rewards. Correctness is consumed HERE, into R' — it never appears
    in any router input."""
    table = json.load(open(TABLE))
    rows = {"train": [], "val": []}
    for t in table:
        if t["split"] not in rows:
            continue                      # test stays sealed
        xw = t["cw"] / t["cs"]
        xs = (t["cw"] + t["cs"]) / t["cs"]
        rows[t["split"]].append({
            "idx": t["idx"], "tid": t["tid"],
            "user": t["view"] + USER_TAIL,
            "rp": {"W": t["qw"] - LAMBDA * xw - hinge(xw),
                   "S": t["qs"] - LAMBDA * xs - hinge(xs)},
            "acc": {"W": t["qw"], "S": t["qs"]},
            "cost": {"W": t["cw"], "S": t["cw"] + t["cs"]},
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/mnt/hdd/jonathan/runs/mab/training1_router")
    ap.add_argument("--G", type=int, default=8, help="rollouts per instance")
    ap.add_argument("--epochs", type=int, default=40,
                    help="small train set (90) -> more epochs than HotpotQA's 6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--baseline", type=float, default=None,
                    help="kill-criterion bar: always-escalate val mean-R' (+0.1890)")
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args()
    if a.smoke:
        a.outdir = a.outdir.rstrip("/") + "_smoke"
    os.makedirs(a.outdir, exist_ok=True)

    random.seed(a.seed); torch.manual_seed(a.seed)
    rows = build_rows()
    if a.smoke:
        rows["train"] = rows["train"][:20]
        rows["val"] = rows["val"][:30]
        a.G, eval_every, max_steps = 4, 2, 6
    else:
        eval_every, max_steps = EVAL_EVERY, None
    print(f"{VERSION}  G={a.G}  train={len(rows['train'])}  val={len(rows['val'])}  "
          f"epochs={a.epochs}  warm_start=FRESH (LoRA shape from {LORA_SHAPE_FROM})",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    def one_id(s):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"letter {s!r} is not a single token: {ids}"
        return ids[0]
    idW, idS = one_id("W"), one_id("S")

    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                device_map="cuda")
    cfg = PeftConfig.from_pretrained(LORA_SHAPE_FROM)   # SAME LoRA shape as SFT
    cfg.inference_mode = False
    model = get_peft_model(base, LoraConfig(**{k: v for k, v in cfg.to_dict().items()
                                               if k in LoraConfig.__dataclass_fields__}))
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"fresh router: trainable params = {n_train:,} "
          f"(must match the HotpotQA fresh runs' count)", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)

    def logprobs_WS(user):
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": user}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return torch.log_softmax(torch.stack([out[idW], out[idS]]), 0)

    # ---- evaluation: greedy decision, delivered-path R' ----------------------
    def evaluate(vrows):
        model.eval()
        tot_r, tot_acc, tot_cost, pws = 0.0, 0, 0.0, []
        with torch.no_grad():
            for r in vrows:
                lp = logprobs_WS(r["user"])
                pW = float(lp[0].exp()); pws.append(pW)
                c = "W" if pW > 0.5 else "S"
                tot_r += r["rp"][c]; tot_acc += r["acc"][c]; tot_cost += r["cost"][c]
        model.train()
        n = len(vrows)
        return {"val_meanR_delivered": round(tot_r / n, 4),
                "val_acc": round(tot_acc / n, 4),
                "val_mean_cost": round(tot_cost / n, 1),
                "mean_pDeliver": round(sum(pws) / len(pws), 4)}

    ev0 = evaluate(rows["val"])
    print(json.dumps({"step": 0, "FRESH_START_EVAL": True, **ev0,
                      "baseline": a.baseline}), flush=True)

    curve = open(os.path.join(a.outdir, "curve.jsonl"), "a")
    best_obj, step, t0, stop = -1e9, 0, time.time(), False
    order = rows["train"][:]
    for ep in range(a.epochs):
        if stop:
            break
        random.shuffle(order); opt.zero_grad()
        for k, r in enumerate(order):
            # ---- one GRPO group: G rollouts over the 2-action space ----------
            lp = logprobs_WS(r["user"])
            pW = lp[0].exp()
            draws = []
            for _ in range(a.G):
                c = "W" if random.random() < float(pW.detach()) else "S"
                draws.append((c, r["rp"][c]))
            rewards = [d[1] for d in draws]
            mu = sum(rewards) / a.G
            adv = [x - mu for x in rewards]          # AMENDMENT A1: no /std
            pg = 0.0
            for (c, _), A in zip(draws, adv):
                if A == 0.0:
                    continue
                pg = pg + A * lp[0 if c == "W" else 1]
            ent = -(lp.exp() * lp).sum()
            loss = (-(pg / a.G) - BETA * ent) / ACCUM
            if not torch.isfinite(loss):
                print(f"skipped non-finite loss (epoch {ep}, row {k})", flush=True)
                continue
            loss.backward()
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % eval_every == 0:
                    ev = evaluate(rows["val"])
                    line = {"step": step, "epoch": ep, **ev, "baseline": a.baseline,
                            "beats_baseline": (None if a.baseline is None
                                               else ev["val_meanR_delivered"] > a.baseline),
                            "min": round((time.time() - t0) / 60, 1)}
                    print(json.dumps(line), flush=True)
                    curve.write(json.dumps(line) + "\n"); curve.flush()
                    if ev["val_meanR_delivered"] > best_obj:
                        best_obj = ev["val_meanR_delivered"]
                        model.save_pretrained(os.path.join(a.outdir, "best_adapter"))
                if max_steps and step >= max_steps:
                    stop = True
                    break

    model.save_pretrained(os.path.join(a.outdir, "last_adapter"))

    # ---- final: reload BEST, score val, write per-instance scores ------------
    del model; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, os.path.join(a.outdir, "best_adapter")).eval()
    final = evaluate(rows["val"])
    with torch.no_grad(), \
         open(os.path.join(a.outdir, "val_scores.jsonl"), "w") as fv:
        for r in rows["val"]:
            pW = float(logprobs_WS(r["user"])[0].exp())
            fv.write(json.dumps({"idx": r["idx"], "tid": r["tid"],
                                 "pDeliver": round(pW, 6)}) + "\n")

    manifest = {"stage": "training1_risk_free_deliver_escalate", "version": VERSION,
                "objective": (f"GRPO (A1): G={a.G} sampled rollouts/instance over the 2-action "
                              f"space (W=deliver weak attempt, S=escalate), outcomes replayed "
                              f"from the Training 1 table (pass/fail correctness), "
                              f"R' = Q - {LAMBDA}*(C/C_SS_i) - hinge, per-instance C_SS_i, "
                              f"weak-first costs, MEAN-BASELINE advantage, + {BETA}*entropy. "
                              f"NO risk term anywhere in training."),
                "base_model": MODEL, "warm_start": f"FRESH (LoRA shape from {LORA_SHAPE_FROM})",
                "table_file": TABLE, "table_sha256": sha256(TABLE),
                "router_prompt_sys": SYS,
                "lr": LR, "epochs": a.epochs, "accum": ACCUM, "seed": a.seed,
                "lambda": LAMBDA, "beta": BETA, "G": a.G, "smoke": a.smoke,
                "kill_baseline": a.baseline,
                "final_fullval": final, "fresh_start_eval": ev0}
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)


if __name__ == "__main__":
    main()
