#!/usr/bin/env python3
# =============================================================================
# training6_trainer.py — VERSION: v1.0 2026-11-08 (DRAFT for Jonathan's line-by-line review; NOT yet run)
# "Training 6 — the priced reward: human cost and failure cost on the high-risk tier."
# DELTA over the validated training5_trainer.py v1.3c (everything else identical, line for line):
#   * TIER BRANCH ONLY (tau >= TAU_GATE). Below the gate nothing changes: W/S, R'(W)/R'(S) as Training 1.
#   * PRICES: per task type, rho_i = C_F,i / C_H,i read from tier_prices_v1_computed.json (built by
#     tier_prices_build.py from the registered inputs; one setting chosen with --price lower|central|operating).
#     C_H = clinician time x rate; C_F = rework + p_slip * p_event * cost of an incident.
#   * REWARD ON THE TIER (slides 69/70, agreed 7 Nov; symbols C_F, C_H, lambda_t):
#       R'(W) = qw - LAMBDA*xw - hinge(xw) - LAMBDA_T * rho_i * 1[qw == 0]
#       R'(S) = qs - LAMBDA*xs - hinge(xs) - LAMBDA_T * rho_i * 1[qs == 0]
#       R'(H) = Q_H - LAMBDA*xw          - LAMBDA_T * 1          (Q_H = 1: a person completes the task;
#                                                                  the weak attempt already ran; one hand-off = one unit)
#     i.e. a failure by a model costs rho_i hand-offs; a hand-off costs exactly one. Break-even: hand off when
#     P(fail) > 1 / (1 + rho_i)  (0.7 % task 5 / 1.1 % task 9 at the operating price; 40 % at the floor).
#   * REMOVED from v1.3c on the tier: the +1/-1 "needed / unnecessary hand-off" H reward (it used the strong
#     model's replayed outcome) and the tau-weighted failure term lambda_t*tau. The H reward is now OUTCOME-FREE;
#     all outcome information enters through the W/S failure penalty, as on every other instance.
#   * LAMBDA_T = 1 by construction (one hand-off = one unit of task value; V = C_H declared). Not swept.
#   * Everything else — prompt (one added option line), fresh LoRA, GRPO A1 (advantage = R' - group mean, no /std),
#     entropy BETA, LR, accum, seed, --explore-epochs warm-up, checkpoint by val mean-R', manifest — UNCHANGED.
#   * evaluate() adds, on the tier: priced human cost (SGD) and priced failure cost (SGD) of the greedy policy,
#     and the count of unseen failures (fail_with_no_human) — reporting only, never in the router input.
#
# Usage (lab, slr-env):
#   python3 tier_prices_build.py --inputs tier_prices_v1.json --out /mnt/hdd/jonathan/runs/mab/tier_prices_v1_computed.json
#   python3 training6_trainer.py --price operating --smoke
#   python3 training6_trainer.py --price operating --explore-epochs 10 --baseline 0.2076     # the registered run
#   python3 training6_trainer.py --price central   --explore-epochs 10 --baseline 0.2076     # sensitivity
#   python3 training6_trainer.py --price lower     --explore-epochs 10 --baseline 0.2076     # sensitivity (floor)
# =============================================================================
import os, sys, json, time, random, argparse, hashlib
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig

VERSION = "training6_trainer v1.1 2026-11-08 (priced reward on the high tier: model failure costs rho_i = C_F/C_H hand-offs, a hand-off costs 1; H reward outcome-free; v1.1 adds --rho-cap: the penalty that reaches the GRPO advantage is min(rho_i, cap); evaluation, manifest and test read keep the full rho_i; no cap = v1.0 behaviour)"
TABLE   = "/mnt/hdd/jonathan/runs/mab/training1_table.json"
TAU     = "/mnt/hdd/jonathan/runs/mab/tau_scores_v2.json"
TAU_GATE = 0.80
PRICES  = "/mnt/hdd/jonathan/runs/mab/tier_prices_v1_computed.json"   # from tier_prices_build.py
LAMBDA_T = 1.0       # one hand-off = one unit of task value (V = C_H); fixed, not swept
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
    print("[training6 prompt] self-check OK", flush=True)
_prompt_selfcheck()


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hinge(x): return max(0.0, x - 1.0)


def failure_penalty(high, failed, rho):
    """Training 6: priced failure, any AI action (W or S), high tier only:
    LAMBDA_T * rho_i if the chosen AI action failed, else 0. rho_i = C_F,i / C_H,i."""
    if not high or not failed:
        return 0.0
    return LAMBDA_T * rho


def load_prices(price_setting):
    """rho per task type for the chosen setting, from the computed prices file (sha recorded in the manifest)."""
    P = json.load(open(PRICES))
    assert P["tau_gate"] == TAU_GATE, (P["tau_gate"], TAU_GATE)
    rho = {task: v["settings"][price_setting]["rho"] for task, v in P["task_types"].items()}
    C_H = {task: v["C_H"] for task, v in P["task_types"].items()}
    C_F = {task: v["settings"][price_setting]["C_F"] for task, v in P["task_types"].items()}
    print(f"[training6 prices] setting={price_setting}  rho={rho}  C_H={C_H}  C_F={C_F}  "
          f"break-even P(fail)={ {t: round(1/(1+r), 4) for t, r in rho.items()} }", flush=True)
    return rho, C_H, C_F


def build_rows(rho, C_H, C_F):
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
        task = t["tid"].split("_")[0]     # task type, e.g. "task5"
        r["task"] = task
        if high:                          # H exists only on the high tier
            assert task in rho, f"no price for {task} (tier task types must be priced)"
            r["rho"] = rho[task]; r["C_H"] = C_H[task]; r["C_F"] = C_F[task]
            r["rp"]["H"] = Q_H - LAMBDA * xw - LAMBDA_T * 1.0      # Training 6: outcome-free; one hand-off = one unit
            r["acc"]["H"] = Q_H
            r["cost"]["H"] = t["cw"]      # AI cost (the weak attempt already ran)
        else:
            r["rho"] = 0.0
        rows[t["split"]].append(r)
    for split, rs in rows.items():
        hi = [r for r in rs if r["high"]]
        print(f"[training6] {split}: n={len(rs)}, high-tier (H available)={len(hi)}, "
              f"both-fail on high tier={sum(r['both_fail'] for r in hi)}", flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=None,
                    help="default: runs/mab/training6_priced_<price>")
    ap.add_argument("--price", choices=["lower", "central", "operating"], required=True,
                    help="price setting from tier_prices_v1_computed.json: operating = the registered run; central / lower = sensitivity")
    ap.add_argument("--rho-cap", type=float, default=None, dest="rho_cap",
                    help="Training 6b: cap on rho_i in the TRAINING penalty only (gradient); full rho_i is kept for evaluation, "
                         "the manifest and the test read. Default None = no cap (v1.0 behaviour). Registered value: 15.")

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
        a.outdir = f"/mnt/hdd/jonathan/runs/mab/training6_priced_{a.price}" + (f"_cap{a.rho_cap:g}" if a.rho_cap is not None else "")
    if a.smoke:
        a.outdir = a.outdir.rstrip("/") + "_smoke"
    os.makedirs(a.outdir, exist_ok=True)

    random.seed(a.seed); torch.manual_seed(a.seed)
    rho, C_H, C_F = load_prices(a.price)
    rows = build_rows(rho, C_H, C_F)
    if a.smoke:
        rows["train"] = rows["train"][:20]
        rows["val"] = rows["val"][:30]
        a.G, eval_every, max_steps = 4, 2, 6
    else:
        eval_every, max_steps = EVAL_EVERY, None
    print(f"{VERSION}  price={a.price}  lambda_t={LAMBDA_T}  G={a.G}  train={len(rows['train'])}  val={len(rows['val'])}  "
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
              "acc": 0, "ai_cost": 0.0, "fail_no_human": 0, "human_cost_sgd": 0.0, "failure_cost_sgd": 0.0}
        with torch.no_grad():
            for r in vrows:
                lp = logprobs(r)
                pW = float(lp[0].exp()); pws.append(pW)
                c = ACT[int(lp.argmax())]
                pen = failure_penalty(r["high"], r["acc"][c] == 0, r["rho"]) if c != "H" else 0.0
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
                        hi["failure_cost_sgd"] += r["C_F"] * (1 - r["acc"][c])   # priced cost of this failure (reporting)
                    else:
                        hi["human_cost_sgd"] += r["C_H"]                        # priced cost of this hand-off (reporting)
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
                              "human_cost_sgd": round(hi["human_cost_sgd"], 2), "failure_cost_sgd": round(hi["failure_cost_sgd"], 2),
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
                    rho_grad = r["rho"] if a.rho_cap is None else min(r["rho"], a.rho_cap)   # v1.1: cap in the gradient only
                    rew -= failure_penalty(r["high"], r["acc"][c] == 0, rho_grad)
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

    manifest = {"stage": "training6_priced_router", "version": VERSION,
                "price_setting": a.price, "prices_file": PRICES, "prices_sha256": sha256(PRICES),
                "rho_by_task": rho, "C_H_by_task": C_H, "C_F_by_task": C_F, "lambda_t": LAMBDA_T,
                "rho_cap_in_gradient": a.rho_cap,
                "H_reward": "Q_H - lambda*C_weak/C_SS_i - lambda_t*1 (outcome-free; one hand-off = one unit)",
                "failure_term": "lambda_t*rho_i*1[AI action failed], W and S, tau>=gate; rho_i = C_F,i/C_H,i", "Q_H": Q_H,
                "tau_file": TAU, "tau_sha256": sha256(TAU), "tau_gate": TAU_GATE,
                "H_available_on": "tau >= tau_gate only",
                "objective": (f"GRPO (A1): G={a.G} sampled rollouts/instance over the allowed actions "
                              f"(W=deliver weak attempt, S=escalate, H=hand to a person), outcomes replayed, "
                              f"three actions on tau >= {TAU_GATE} (W/S/H), two elsewhere (W/S); "
                              f"R'(W), R'(S) as Training 1 minus {LAMBDA_T}*rho_i on a failure (tau >= {TAU_GATE} only); "
                              f"R'(H) = {Q_H} - {LAMBDA}*C_weak/C_SS_i - {LAMBDA_T}; prices {a.price}; "
                              f"nothing changes below tau {TAU_GATE}; tau frozen (rubric v2). "
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
