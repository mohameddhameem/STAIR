#!/usr/bin/env python3
# =============================================================================
# training5_trainer.py — VERSION: v1.0 2026-10-28
# "Training 5 — the three-way router: deliver / escalate / hand to a person."
# DELTA over the validated training2_trainer.py v1.0 (everything else identical):
#   * THIRD ACTION H = hand the task to a person, AVAILABLE ONLY ON HIGH-RISK
#     INSTANCES (tau >= TAU_GATE). On the other instances the router keeps its
#     two options (H is masked out of the softmax), and a failure there is
#     accepted. The ROUTER decides H from the task text + weak attempt only; no
#     outcome information is ever in its input.
#   * Replayed reward of H — v1.2 (28 Oct 23:40, Jonathan): NO human price; the
#     hand-off is judged purely on whether the strong model would fail:
#       R'(H) = 1[qw == 0 and qs == 0] - 1[qs == 1] - LAMBDA * C_weak/C_SS_i
#     +1 when the AI would have failed (a needed hand-off), -1 when the strong
#     model would have succeeded (an unnecessary hand-off wastes a correct answer
#     and a person's time); weak-first, so H also pays the spent C_weak; no hinge,
#     no risk penalty. The router therefore hands over iff it judges failure more
#     likely than not (break-even at P = 0.5).
#     History: v1.0 priced a person at k*C_SS_i (k in {6, 8, 9}; exploratory sweep,
#     k = 9 separated the tasks); v1.1 rewarded only needed hand-offs with no
#     penalty for unnecessary ones (break-even ~5%), so the router handed over the
#     whole tier — rejected 23:40 on the validation arithmetic.
#   * RISK TERM — v1.3 (28 Oct 23:43, Jonathan): the confidence proxy of Training 2
#     (lambda_t * max(0, tau - c) on deliver) is replaced by a RISK-WEIGHTED
#     FAILURE penalty on every AI action at tau >= TAU_GATE:
#       - lambda_t * tau_i * 1[the chosen AI action (W or S) failed]
#     H is never an AI failure and carries no risk penalty. Effect: an AI failure
#     on a high-risk instance costs 1 + lambda_t*tau instead of 1, so the hand-off
#     break-even moves from P(fail) = 0.5 (lambda_t = 0) to 1/(2 + lambda_t*tau):
#     0.36 at lambda_t = 1, 0.28 at lambda_t = 2 — a governance dial, never
#     touching low-risk work.
#   * EXPLORATION WARM-UP (v1.3c, 29 Oct 00:10): --explore-epochs N. From a fresh
#     LoRA the v1.3 reward never discovered H (hand-offs 0 at every evaluation,
#     epochs 0-39, both lambda_t): H starts at ~1 % probability, so it is sampled
#     rarely, and the first -1.05 on a task-5 draw suppresses the shared H logit
#     before the +0.96 on task-9 rows can build it. For the first N epochs the G
#     rollouts are drawn UNIFORMLY over the allowed actions (W/S/H on the high
#     tier, W/S elsewhere) instead of from the policy; from epoch N on, from the
#     policy as usual. Same fresh adapter, same reward, same objective; only the
#     sampling distribution of the first N epochs changes. Recorded in the manifest.
#     W and S rewards unchanged from Training 1/2; the risk term (--lambda-t,
#     deliver branch, tau >= TAU_GATE) is kept: --lambda-t 0 = risk-free run
#     (5a), --lambda-t 1 = risk-aware run (5b).
#   * evaluate() adds the human-load metrics on the high tier: hand-offs, their
#     precision/recall against the both-fail set, AI-subset accuracy, human cost.
#   * Prompt: ONE added option line ("H) Hand the task to a person"); the
#     objective sentence is verbatim; no principles. Self-check guards both.
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
#   python3 training5_trainer.py --lambda-t 0 --smoke
#   python3 training5_trainer.py --lambda-t 0 --explore-epochs 10 --baseline 0.1890   # 5a (break-even P(fail) = 0.50)
#   python3 training5_trainer.py --lambda-t 1 --explore-epochs 10 --baseline 0.1890   # 5b (risk-averse, break-even 0.36)
# =============================================================================
import os, sys, json, time, random, argparse, hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

VERSION = "training5_trainer v1.3c 2026-10-29 (three-way router W/S/H on the high tier; H = +1 if both fail / -1 if strong succeeds; risk = lambda_t*tau on any AI failure at tau>=gate; delta on training2_trainer v1.0)"
TABLE   = "/mnt/hdd/jonathan/runs/mab/training1_table.json"
TAU     = "/mnt/hdd/jonathan/runs/mab/tau_scores_v2.json"
TAU_GATE = 0.80
Q_H = 1.0            # a person completes the task correctly (used for accuracy-with-H)
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
USER_TAIL_2 = ("\n\nOptions:\nW) Deliver the weak model's result\n"
               "S) Escalate to the strong model\nAnswer with a single letter: W or S.")
USER_TAIL_3 = ("\n\nOptions:\nW) Deliver the weak model's result\n"
               "S) Escalate to the strong model\n"
               "H) Hand the task to a person\nAnswer with a single letter: W, S or H.")

def _prompt_selfcheck():
    assert ("Your objective: the task should be completed correctly while "
            "spending as little compute as possible - correctness first, cost "
            "second.") in SYS, "objective sentence drifted"
    assert "Based on the information shown below, choose one option." in SYS
    for banned in ("Routing principles", "only when needed", "(1)", "(2)", "(3)",
                   "cheap)", "expensive)"):
        assert banned not in SYS, f"banned (principle-flavored) phrase: {banned}"
    assert USER_TAIL_3.count("\n") == USER_TAIL_2.count("\n") + 1, "H adds exactly one option line"
    assert "H) Hand the task to a person" in USER_TAIL_3
    print("[training5 prompt] self-check OK", flush=True)
_prompt_selfcheck()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hinge(x): return max(0.0, x - 1.0)


def risk_penalty(tau, failed, lambda_t):
    """v1.3: risk-weighted failure, any AI action (W or S), high tier only:
    lambda_t * tau if the chosen AI action failed, else 0."""
    if tau < TAU_GATE or not failed:
        return 0.0
    return lambda_t * tau


def build_rows():
    """One row per instance: the router input (task text + weak-attempt view,
    from the registered table — never the grader's verdict) + the two replayed
    branch rewards. Correctness is consumed HERE, into R' — it never appears
    in any router input."""
    table = json.load(open(TABLE))
    tauj = json.load(open(TAU))
    taus = {s["tid"]: s["tau"] for s in tauj["scores"]}
    rows = {"train": [], "val": []}
    for t in table:
        if t["split"] not in rows:
            continue                      # test stays sealed
        xw = t["cw"] / t["cs"]
        xs = (t["cw"] + t["cs"]) / t["cs"]
        high = taus[t["tid"]] >= TAU_GATE
        r = {"idx": t["idx"], "tid": t["tid"], "high": high,
             "user": t["view"] + (USER_TAIL_3 if high else USER_TAIL_2),
             "rp": {"W": t["qw"] - LAMBDA * xw - hinge(xw),
                    "S": t["qs"] - LAMBDA * xs - hinge(xs)},
             "acc": {"W": t["qw"], "S": t["qs"]},
             "cost": {"W": t["cw"], "S": t["cw"] + t["cs"]},
             "both_fail": (t["qw"] == 0 and t["qs"] == 0),
             "tau": taus[t["tid"]]}
        if high:                          # H exists only on the high tier
            r["rp"]["H"] = (1.0 if r["both_fail"] else 0.0) - (1.0 if t["qs"] else 0.0) - LAMBDA * xw   # v1.2
            r["acc"]["H"] = Q_H
            r["cost"]["H"] = t["cw"]      # AI cost (the weak attempt already ran)
        rows[t["split"]].append(r)
    for split, rs in rows.items():
        hi = [r for r in rs if r["high"]]
        print(f"[training5] {split}: n={len(rs)}, high-tier (H available)={len(hi)}, "
              f"both-fail on high tier={sum(r['both_fail'] for r in hi)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="default: runs/mab/training5v13_router_lt<lambda_t>")
    ap.add_argument("--lambda-t", type=float, default=0.0, dest="lambda_t",
                    help="risk-weighted failure penalty lambda_t*tau on any AI failure at tau>=gate: 0 = plain (5a), 1 = risk-averse (5b)")

    ap.add_argument("--G", type=int, default=8, help="rollouts per instance")
    ap.add_argument("--epochs", type=int, default=40,
                    help="small train set (90) -> more epochs than HotpotQA's 6")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--baseline", type=float, default=None,
                    help="kill-criterion bar: always-escalate val mean-R' (+0.1890)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--explore-epochs", type=int, default=0, dest="explore_epochs",
                    help="v1.3c: epochs during which rollouts are sampled uniformly over the allowed actions (default 0 = policy sampling throughout)")
    a = ap.parse_args()
    if a.outdir is None:
        a.outdir = f"/mnt/hdd/jonathan/runs/mab/training5v13c_router_lt{a.lambda_t:g}"
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
    print(f"{VERSION}  lambda_t={a.lambda_t}  G={a.G}  train={len(rows['train'])}  val={len(rows['val'])}  "
          f"epochs={a.epochs}  warm_start=FRESH (LoRA shape from {LORA_SHAPE_FROM})",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    def one_id(s):
        ids = tok.encode(s, add_special_tokens=False)
        assert len(ids) == 1, f"letter {s!r} is not a single token: {ids}"
        return ids[0]
    idW, idS, idH = one_id("W"), one_id("S"), one_id("H")

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

    ACT = ["W", "S", "H"]
    def logprobs(r):
        """log-probabilities over the allowed actions: [W, S] on low-risk rows,
        [W, S, H] on high-risk rows (H is masked out of the softmax elsewhere)."""
        msgs = [{"role": "system", "content": SYS},
                {"role": "user", "content": r["user"]}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True,
                                      return_tensors="pt")
        if not torch.is_tensor(ids):
            ids = ids["input_ids"]
        ids = ids.to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        logits = [out[idW], out[idS]] + ([out[idH]] if r["high"] else [])
        return torch.log_softmax(torch.stack(logits), 0)

    # ---- evaluation: greedy decision, delivered-path R' ----------------------
    def evaluate(vrows):
        """Greedy decision = argmax over the allowed actions. Accuracy counts H as
        correct (Q_H = 1); ai_acc is the accuracy of what the AI handled; human
        metrics are on the high tier, where H exists."""
        model.eval()
        tot_r, tot_acc, tot_cost, pws = 0.0, 0, 0.0, []
        ai_n = ai_acc = 0
        hi = {"n": 0, "W": 0, "S": 0, "H": 0, "H_on_both_fail": 0, "both_fail": 0,
              "acc": 0, "ai_cost": 0.0, "fail_no_human": 0}
        with torch.no_grad():
            for r in vrows:
                lp = logprobs(r)
                pW = float(lp[0].exp()); pws.append(pW)
                c = ACT[int(lp.argmax())]
                pen = risk_penalty(r["tau"], r["acc"][c] == 0, a.lambda_t) if c != "H" else 0.0
                tot_r += r["rp"][c] - pen
                tot_acc += r["acc"][c]; tot_cost += r["cost"][c]
                if c != "H":
                    ai_n += 1; ai_acc += r["acc"][c]
                if r["high"]:
                    hi["n"] += 1; hi[c] += 1; hi["acc"] += r["acc"][c]
                    hi["both_fail"] += r["both_fail"]
                    hi["H_on_both_fail"] += (c == "H" and r["both_fail"])
                    hi["ai_cost"] += r["cost"][c]
                    if c != "H":
                        hi["fail_no_human"] += (1 - r["acc"][c])
        model.train()
        n = len(vrows)
        H = hi["H"]
        return {"val_meanR_delivered": round(tot_r / n, 4),
                "val_acc": round(tot_acc / n, 4),
                "val_ai_acc": (round(ai_acc / ai_n, 4) if ai_n else None),
                "val_mean_ai_cost": round(tot_cost / n, 1),
                "mean_pDeliver": round(sum(pws) / len(pws), 4),
                "high_tier": {"n": hi["n"], "deliver": hi["W"], "escalate": hi["S"], "to_person": H,
                              "human_load": (round(H / hi["n"], 4) if hi["n"] else None),
                              "H_precision": (round(hi["H_on_both_fail"] / H, 4) if H else None),
                              "H_recall": (round(hi["H_on_both_fail"] / hi["both_fail"], 4) if hi["both_fail"] else None),
                              "acc": (round(hi["acc"] / hi["n"], 4) if hi["n"] else None),
                              "to_person_needed": hi["H_on_both_fail"], "to_person_unnecessary": H - hi["H_on_both_fail"],
                              "fail_with_no_human": hi["fail_no_human"],
                              "mean_ai_cost": (round(hi["ai_cost"] / hi["n"], 1) if hi["n"] else None)}}

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
            # ---- one GRPO group: G rollouts over the allowed actions ---------
            lp = logprobs(r)
            probs = lp.exp().detach().tolist()      # sampling distribution (no grad)
            if ep < a.explore_epochs:               # v1.3c warm-up: uniform over allowed actions
                probs = [1.0 / len(probs)] * len(probs)
            draws = []
            for _ in range(a.G):
                c = random.choices(ACT[:len(probs)], weights=probs)[0]
                rew = r["rp"][c]
                if c != "H":
                    rew -= risk_penalty(r["tau"], r["acc"][c] == 0, a.lambda_t)
                draws.append((c, rew))
            rewards = [d[1] for d in draws]
            mu = sum(rewards) / a.G
            adv = [x - mu for x in rewards]          # AMENDMENT A1: no /std
            pg = 0.0
            for (c, _), A in zip(draws, adv):
                if A == 0.0:
                    continue
                pg = pg + A * lp[ACT.index(c)]
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
            pr = logprobs(r).exp().tolist()
            fv.write(json.dumps({"idx": r["idx"], "tid": r["tid"], "high": r["high"],
                                 "pDeliver": round(pr[0], 6), "pEscalate": round(pr[1], 6),
                                 "pHuman": (round(pr[2], 6) if r["high"] else None)}) + "\n")

    manifest = {"stage": "training5_three_way_router", "version": VERSION,
                "lambda_t": a.lambda_t, "H_reward": "1[both models fail] - 1[strong succeeds] - lambda*C_weak/C_SS_i (v1.2+, no human price)", "risk_term": "lambda_t*tau*1[AI action failed], W and S, tau>=gate (v1.3)", "Q_H": Q_H,
                "tau_file": TAU, "tau_sha256": sha256(TAU), "tau_gate": TAU_GATE,
                "H_available_on": "tau >= tau_gate only",
                "objective": (f"GRPO (A1): G={a.G} sampled rollouts/instance over the allowed actions "
                              f"(W=deliver weak attempt, S=escalate, H=hand to a person), outcomes replayed, "
                              f"three actions on tau >= {TAU_GATE} (W/S/H), two elsewhere (W/S); "
                              f"R'(W), R'(S) as Training 1; R'(H) = 1[qw==0 and qs==0] - 1[qs==1] - {LAMBDA}*C_weak/C_SS_i (no price, no hinge, no risk term); "
                              f"risk term {a.lambda_t}*tau*1[AI action failed] on W and S, "
                              f"dormant below tau {TAU_GATE}; tau frozen (rubric v2). "
                              f"MEAN-BASELINE advantage, + {BETA}*entropy."),
                "base_model": MODEL, "warm_start": f"FRESH (LoRA shape from {LORA_SHAPE_FROM})",
                "table_file": TABLE, "table_sha256": sha256(TABLE),
                "router_prompt_sys": SYS,
                "lr": LR, "epochs": a.epochs, "explore_epochs": a.explore_epochs, "accum": ACCUM, "seed": a.seed,
                "lambda": LAMBDA, "beta": BETA, "G": a.G, "smoke": a.smoke,
                "kill_baseline": a.baseline,
                "final_fullval": final, "fresh_start_eval": ev0}
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)


if __name__ == "__main__":
    main()
