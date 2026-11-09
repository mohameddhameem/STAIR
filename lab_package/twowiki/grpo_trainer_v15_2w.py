#!/usr/bin/env python3
# =============================================================================
# grpo_trainer.py — VERSION: v1.1 2026-10-21 (AMENDMENT A1, approved by Jonathan)
#
# AMENDMENT A1 (21 Oct, after the registered v1.0 run FAILED the kill
# criterion — best 0.6316 vs bar 0.6892, with over-cheapening: ret P(W)
# 0.61->0.90, lenient 71.3%->64%):
#   advantage = R' - group mean   (the /std normalization is REMOVED).
#   Diagnosis: dividing by the group std erases the reward's calibrated
#   magnitudes — on the 52% of questions where all four F1 tie, a ~0.04
#   cost-only gap became a full-strength "go cheap" gradient on half the
#   dataset, overwhelming the (up to 0.45, but equally normalized) quality
#   signal. The certified expected-R' trainers push proportionally to the RAW
#   reward gap; the mean-baseline advantage restores that property while
#   keeping GRPO's sampling. Everything else is unchanged from v1.0.
#   Output dir default changed so the failed registered run stays intact.
#
# GRPO training of BOTH routers (Ret + QA1) against the registered reward,
# per the pre-registered plan amended 20 Oct. Sibling of the certified
# configb_reward_train_ret.py / _qa.py: same model loading, same prompt
# contract (router_prompts), same logits_WS extraction, same guards, same
# manifest discipline. THE ONLY STRUCTURAL CHANGE IS THE LOSS:
#
#   certified reward trainers:  loss = -( pi_W*R'_W + pi_S*R'_S ) - BETA*H
#                               (exact expectation — possible only because the
#                                grid makes both branch rewards available)
#   this trainer (GRPO):        sample G rollouts; each samples a decision at
#                               BOTH stages from the routers' current
#                               probabilities; the chosen path's outcome is
#                               REPLAYED from the certified record; one R' per
#                               rollout scores the DELIVERED answer;
#                               advantage = (R' - group mean)/(group std);
#                               loss = -(1/G) * sum_g A_g * [ log pi_ret(rc_g)
#                                      + log pi_qa(qc_g | rc_g) ]  - BETA*H
#
# No labels anywhere: gold answers appear only inside R', after the episode.
# Decisions are SAMPLED (exploration replaces the label). One update covers
# both adapters (together in time and signal, separate weights).
#
# Imports of certified implementations (nothing re-implemented):
#   tokenF1, PATH, C_SS, LAMBDA  <- configb_reward_train_ret.py
#   prompts, assemble, letter_ids <- router_prompts.py (byte-identical inputs;
#       assemble() called with DEFAULT model names, exactly as the certified
#       Config B trainers called it — byte-identity with the certified
#       adapters' training inputs takes precedence)
#   trim_pool, parse_selection, build_block <- run_labels4.py
#   LEN <- lenient_match_agenttts.py (reporting only, never in the signal)
#
# Guards (all from the certified ladder): drift check before training (refuses
# to train if router_prompts drifted from the probe), warm start from the
# SFT (label) adapters, entropy bonus BETA, LR 1e-5, grad clip 1.0, non-finite
# loss skipped, checkpoint by THE OBJECTIVE (delivered-path val mean-R',
# greedy decisions — the kill criterion's currency).
#
# Modes:
#   --smoke         ~50 train questions, G=4, eval every 5 steps, stops at 20
#                   steps, writes to <outdir>_smoke — run this FIRST.
#   --from-scratch  fresh LoRA adapters (config copied from the SFT adapters'
#                   own adapter_config.json — no hand-written LoraConfig), the
#                   one guarded from-scratch run the register allows.
#   --baseline X    the kill-criterion baseline (both-reward pair delivered
#                   val mean-R', from grpo_feasibility_analysis v1.1); printed
#                   against every eval so the 3-epoch kill rule is visible.
#
# Usage (lab, slr-env):
#   python3 grpo_trainer.py --smoke
#   python3 grpo_trainer.py --baseline <v1.1 number>
# =============================================================================
import os, sys, json, time, random, argparse, hashlib, subprocess

os.environ["SLR_POOL_K"] = "110"   # 2W: 2Wiki pools are 100-104; labels ran untrimmed (K=110) -> trim_pool is a no-op
os.environ.setdefault("SLR_SHUFFLE", "off")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import torch
import run_labels4 as R
import router_prompts as RP
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig, get_peft_model, LoraConfig
from configb_reward_train_ret import tokenF1, PATH, C_SS, LAMBDA
from lenient_match_agenttts import LEN     # reporting only

VERSION = "grpo_trainer v1.5-2W 2026-11-09 (v1.5 byte-identical except the lines marked # 2W)"
DATA   = "/mnt/hdd/jonathan/runs/twowiki/grpo_grid/tts_2wiki_trainval.json"   # 2W: train5000 + val400, staged
GRID   = "/mnt/hdd/jonathan/runs/twowiki/grpo_grid"                           # 2W: staged stores + ret_labels
RET_ROWS = os.path.join(GRID, "ret_labels", "ret_labels.jsonl")   # rows + splits only
MODEL  = "Qwen/Qwen2.5-1.5B-Instruct"
WARM_RET = "/mnt/hdd/jonathan/runs/config_b/sft_router5_ret/best_adapter"   # 2W: --from-scratch copies ONLY its LoRA config
WARM_QA  = "/mnt/hdd/jonathan/runs/config_b/sft_router5_qa/best_adapter"    # 2W: same (config only, no weights)
LR, EPOCHS, ACCUM, EVAL_EVERY, SEED = 1e-5, 3, 8, 100, 17
BETA = 0.03                     # entropy bonus, as in the certified ladder
CURVE_EVAL_MAX = 1110

PATHKEY = {("W", "W"): "WW", ("W", "S"): "WS", ("S", "W"): "SW", ("S", "S"): "SS"}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_store(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line); d[r["k"]] = r["v"]
        except Exception:
            pass
    return d


def build_rows(data):
    """One row per question: the two router inputs (QA input per retrieval
    branch) + the four replayed path rewards. Inputs byte-identical to the
    certified trainers' build_rows. Gold is consumed HERE, into R' — it never
    appears in any router input."""
    lab = [json.loads(l) for l in open(RET_ROWS)]
    retr = {"W": load_store(os.path.join(GRID, "retr_weak.jsonl")),
            "S": load_store(os.path.join(GRID, "retr_strong.jsonl"))}
    qa_w = load_store(os.path.join(GRID, "qa_weak.jsonl"))
    qa_s = load_store(os.path.join(GRID, "qa_strong.jsonl"))
    rows = {"train": [], "val": []}
    for r in lab:
        if r["split"] not in rows:
            continue
        i = r["idx"]; inst = data[i]
        pool = R.trim_pool(inst)
        golds = inst["answers"]

        # Ret router input: question + the WEAK retriever's selections (never the pool)
        sel_w = R.parse_selection(retr["W"][str(i)], len(pool))
        ret_user = RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel_w),
                       selections=RP.format_selections(sel_w, pool))

        # QA router input, one per retrieval branch: question + that branch's
        # evidence + the weak answer produced ON that branch
        qa_user, rprime = {}, {}
        for rc, sfx in (("W", "weak"), ("S", "strong")):
            sel = R.parse_selection(retr[rc][str(i)], len(pool))
            qa_user[rc] = RP.INSTR_QA1.format(q=inst["question"],
                              evidence=RP.format_evidence(R.build_block(inst, sel)),
                              weak_answer=str(qa_w[f"{i}|{sfx}"]))
            for qc, store in (("W", qa_w), ("S", qa_s)):
                delivered = str(store[f"{i}|{sfx}"])
                rprime[rc + qc] = tokenF1(delivered, golds) - LAMBDA * PATH[(rc, qc)] / C_SS
                # lenient verdict for reporting only
                rprime[rc + qc + "_len"] = LEN(delivered, golds)
        rows[r["split"]].append({"idx": i, "ret_user": ret_user,
                                 "qa_user": qa_user, "rp": rprime})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/mnt/hdd/jonathan/runs/twowiki/grpo_2w_a1_fresh_ep6_objective")   # 2W
    ap.add_argument("--G", type=int, default=8, help="rollouts per question")
    ap.add_argument("--epochs", type=int, default=EPOCHS, help="training epochs; default 3 unchanged")
    ap.add_argument("--prompts", choices=["standard","guided","objective"], default="standard", help="router system prompt variant; standard = byte-identical to all previous runs")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--from-scratch", action="store_true", dest="from_scratch")
    ap.add_argument("--baseline", type=float, default=None,
                    help="kill-criterion baseline: both-reward pair delivered val mean-R'")
    ap.add_argument("--seed", type=int, default=SEED,
                    help="training seed (default 17, the registered run's; vary for the seed-robustness run)")
    a = ap.parse_args()
    if a.smoke:
        a.outdir = a.outdir.rstrip("/") + "_smoke"
    if os.path.exists(os.path.join(a.outdir, "manifest.json")) and not a.smoke:   # 2W: overwrite guard
        raise SystemExit(f"REFUSING: {a.outdir}/manifest.json exists")
    if not a.from_scratch:                                                        # 2W: this file is the FRESH run only
        raise SystemExit("REFUSING: 2W trainer runs --from-scratch only (the HotpotQA-initialised run is separate)")
    os.makedirs(a.outdir, exist_ok=True)

    # ---- guard 1: drift check (identical to the certified trainers) ----------
    chk = subprocess.run([sys.executable, os.path.join(HERE, "router_prompts.py"),
                          os.path.join(HERE, "probe_ret_coverage.py")],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        print("DRIFT CHECK FAILED -- refusing to train.\n", chk.stdout, chk.stderr)
        sys.exit(1)
    print("drift check OK", flush=True)
    if a.prompts == "guided":
        chk2 = subprocess.run([sys.executable, os.path.join(HERE, "router_prompts_guided.py")], capture_output=True, text=True)
        if chk2.returncode != 0:
            print("GUIDED PROMPT CHECK FAILED -- refusing to train.\n", chk2.stdout, chk2.stderr); sys.exit(1)
        import router_prompts_guided
        router_prompts_guided.activate()
        print("guided prompts ACTIVE:", router_prompts_guided.GUIDANCE, flush=True)
    if a.prompts == "objective":
        chk3 = subprocess.run([sys.executable, os.path.join(HERE, "router_prompts_objective.py")], capture_output=True, text=True)
        if chk3.returncode != 0:
            print("OBJECTIVE PROMPT CHECK FAILED -- refusing to train.\n", chk3.stdout, chk3.stderr); sys.exit(1)
        import router_prompts_objective
        router_prompts_objective.activate()
        print("objective prompt ACTIVE:", router_prompts_objective.OBJECTIVE_TEMPLATE, flush=True)

    random.seed(a.seed); torch.manual_seed(a.seed)
    data = json.load(open(DATA))
    rows = build_rows(data)
    if a.smoke:
        rows["train"] = rows["train"][:50]
        rows["val"] = rows["val"][:100]
        a.G, eval_every, max_steps = 4, 5, 20
    else:
        eval_every, max_steps = EVAL_EVERY, None
    print(f"{VERSION}  G={a.G}  train={len(rows['train'])}  val={len(rows['val'])}  "
          f"warm_start={'FRESH (from-scratch, guarded)' if a.from_scratch else 'SFT (label) adapters'}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)

    def load_router(warm):
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                    device_map="cuda")
        if a.from_scratch:
            cfg = PeftConfig.from_pretrained(warm)      # SAME LoRA shape as SFT
            cfg.inference_mode = False
            m = get_peft_model(base, LoraConfig(**{k: v for k, v in cfg.to_dict().items()
                                                   if k in LoraConfig.__dataclass_fields__}))
        else:
            m = PeftModel.from_pretrained(base, warm, is_trainable=True)
        m.gradient_checkpointing_enable()
        m.enable_input_require_grads()
        m.config.use_cache = False
        n_train = sum(p.numel() for p in m.parameters() if p.requires_grad)
        print(f"loaded router from {warm}: trainable params = {n_train:,} "
              f"(from-scratch and warm-start runs must show the SAME count)", flush=True)
        return m

    ret_model = load_router(WARM_RET)
    qa_model  = load_router(WARM_QA)
    params = [p for m in (ret_model, qa_model) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)

    def logprobs_WS(model, user):
        """log pi over (W, S) at the opened assistant turn — the certified
        extraction: two letter logits, log_softmax over exactly those two."""
        ids = RP.assemble(tok, user).to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return torch.log_softmax(torch.stack([out[idW], out[idS]]), 0)

    # ---- evaluation: greedy decisions at BOTH stages, delivered-path R' ------
    curve_val = rows["val"] if len(rows["val"]) <= CURVE_EVAL_MAX \
                else random.sample(rows["val"], CURVE_EVAL_MAX)

    def evaluate(vrows):
        ret_model.eval(); qa_model.eval()
        tot_r, tot_len, tot_cost, pws = 0.0, 0, 0.0, []
        with torch.no_grad():
            for r in vrows:
                lp = logprobs_WS(ret_model, r["ret_user"])
                pW = float(lp[0].exp()); pws.append(pW)
                rc = "W" if pW > 0.5 else "S"
                lq = logprobs_WS(qa_model, r["qa_user"][rc])
                qc = "W" if float(lq[0].exp()) > 0.5 else "S"
                tot_r += r["rp"][rc + qc]
                tot_len += r["rp"][rc + qc + "_len"]
                tot_cost += PATH[(rc, qc)]
        ret_model.train(); qa_model.train()
        n = len(vrows)
        return {"val_meanR_delivered": round(tot_r / n, 4),
                "val_lenient_acc": round(tot_len / n, 4),
                "val_mean_cost": round(tot_cost / n, 1),
                "mean_ret_pW": round(sum(pws) / len(pws), 4)}

    ev0 = evaluate(curve_val)
    print(json.dumps({"step": 0, "WARM_START_EVAL": True, **ev0,
                      "baseline": a.baseline}), flush=True)

    curve = open(os.path.join(a.outdir, "curve.jsonl"), "a")
    best_obj, step, t0, stop = -1e9, 0, time.time(), False
    order = rows["train"][:]
    for ep in range(a.epochs):
        if stop:
            break
        random.shuffle(order); opt.zero_grad()
        for k, r in enumerate(order):
            # ---- one GRPO group: G rollouts over the 2x2 action space --------
            lp_ret = logprobs_WS(ret_model, r["ret_user"])
            pW_ret = lp_ret[0].exp()
            lp_qa = {}
            # sample decisions (exploration replaces the label)
            draws = []
            for _ in range(a.G):
                rc = "W" if random.random() < float(pW_ret.detach()) else "S"
                if rc not in lp_qa:
                    lp_qa[rc] = logprobs_WS(qa_model, r["qa_user"][rc])
                qc = "W" if random.random() < float(lp_qa[rc][0].exp().detach()) else "S"
                draws.append((rc, qc, r["rp"][rc + qc]))
            rewards = [d[2] for d in draws]
            mu = sum(rewards) / a.G
            # AMENDMENT A1: mean-baseline advantage, NO std normalization —
            # keeps the reward's calibrated magnitudes (a 0.45 quality gap
            # pushes ~10x harder than a 0.04 cost gap, as in the certified
            # expected-R' trainers). An all-tie group still teaches nothing
            # (every advantage is exactly 0).
            adv = [x - mu for x in rewards]
            # ---- policy-gradient loss over the sampled decisions -------------
            pg = 0.0
            for (rc, qc, _), A in zip(draws, adv):
                if A == 0.0:
                    continue
                pg = pg + A * (lp_ret[0 if rc == "W" else 1]
                               + lp_qa[rc][0 if qc == "W" else 1])
            # entropy bonus on the distributions actually used
            ent = -(lp_ret.exp() * lp_ret).sum()
            for rc in lp_qa:
                ent = ent + (-(lp_qa[rc].exp() * lp_qa[rc]).sum()) / max(1, len(lp_qa))
            loss = (-(pg / a.G) - BETA * ent) / ACCUM
            if not torch.isfinite(loss):
                print(f"skipped non-finite loss (epoch {ep}, row {k})", flush=True)
                continue
            loss.backward()
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % eval_every == 0:
                    ev = evaluate(curve_val)
                    line = {"step": step, "epoch": ep, **ev,
                            "baseline": a.baseline,
                            "beats_baseline": (None if a.baseline is None
                                               else ev["val_meanR_delivered"] > a.baseline),
                            "min": round((time.time() - t0) / 60, 1)}
                    print(json.dumps(line), flush=True)
                    curve.write(json.dumps(line) + "\n"); curve.flush()
                    if ev["val_meanR_delivered"] > best_obj:   # checkpoint by THE OBJECTIVE
                        best_obj = ev["val_meanR_delivered"]
                        ret_model.save_pretrained(os.path.join(a.outdir, "ret_best_adapter"))
                        qa_model.save_pretrained(os.path.join(a.outdir, "qa_best_adapter"))
                if max_steps and step >= max_steps:
                    stop = True
                    break
        # epoch-end kill-criterion line (the 3-epoch rule is applied by Jonathan)
        ev = evaluate(curve_val)
        print(json.dumps({"EPOCH_END": ep, **ev, "baseline": a.baseline,
                          "beats_baseline": (None if a.baseline is None
                                             else ev["val_meanR_delivered"] > a.baseline)}),
              flush=True)

    ret_model.save_pretrained(os.path.join(a.outdir, "ret_last_adapter"))
    qa_model.save_pretrained(os.path.join(a.outdir, "qa_last_adapter"))

    # ---- final: reload BEST, score full validation, write STORE inputs -------
    del ret_model, qa_model; torch.cuda.empty_cache()
    def reload(sub):
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                    device_map="cuda")
        return PeftModel.from_pretrained(base, os.path.join(a.outdir, sub)).eval()
    ret_model, qa_model = reload("ret_best_adapter"), reload("qa_best_adapter")
    final = evaluate(rows["val"])
    with torch.no_grad(), \
         open(os.path.join(a.outdir, "ret_val_scores.jsonl"), "w") as fr, \
         open(os.path.join(a.outdir, "qa_val_scores.jsonl"), "w") as fq:
        for r in rows["val"]:
            pW = float(logprobs_WS(ret_model, r["ret_user"])[0].exp())
            fr.write(json.dumps({"idx": r["idx"], "pW": round(pW, 6)}) + "\n")
            for rc in ("W", "S"):
                q = float(logprobs_WS(qa_model, r["qa_user"][rc])[0].exp())
                fq.write(json.dumps({"idx": r["idx"], "ret_choice": rc,
                                     "pW": round(q, 6)}) + "\n")

    manifest = {"stage": "grpo_ret_qa_mean_baseline_advantage_A1", "version": VERSION,
                "objective": (f"GRPO (A1): G={a.G} sampled rollouts/question, decisions sampled at both "
                              f"stages, outcomes replayed from the certified record, one R' per "
                              f"delivered answer (tokenF1 - {LAMBDA}*C_path/C_SS), MEAN-BASELINE "
                              f"advantage (no std normalization — amendment A1, 21 Oct), one update "
                              f"across both adapters; + {BETA}*entropy"),
                "base_model": MODEL,
                "warm_start": ("FRESH (config copied from SFT adapters)" if a.from_scratch
                               else {"ret": WARM_RET, "qa": WARM_QA}),
                "rows_file": RET_ROWS, "rows_sha256": sha256(RET_ROWS),
                "char_cap": RP.CHAR_CAP, "maxlen": RP.MAXLEN,
                "lr": LR, "epochs": a.epochs, "prompts": a.prompts, "accum": ACCUM, "seed": a.seed,
                "lambda": LAMBDA, "beta": BETA, "c_ss": round(C_SS, 4), "G": a.G,
                "smoke": a.smoke, "kill_baseline": a.baseline,
                "final_fullval": final,
                "warm_start_eval": ev0}
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)


if __name__ == "__main__":
    main()
