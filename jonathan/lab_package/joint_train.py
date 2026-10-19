# joint_train.py -- EXPERIMENT J (pre-registered 13 Oct): JOINT training of BOTH
# Config A routers on the EXACT expected reward over the four cached paths.
#
#   E[R'] = sum over r in {W,S}, a in {W,S} of
#             pi_ret(r) * pi_qa(a | branch r) * R'(r, a)
#   loss  = -( E[R'] + BETA * [H(pi_ret) + H(pi_qa|W) + H(pi_qa|S)] )
#
# R'(r,a) = tokenF1(cached answer of path r,a) - LAMBDA * C_path(r,a) / C_SS --
# the registered reward, unchanged. Every term is cached: the expectation is
# EXACT, no sampling anywhere in Run J.
# INHERITED GUARDS: warm start from the LABEL-trained adapters of both stages;
# entropy BETA=0.03; LR 1e-5; grad clip 1.0 + non-finite-loss skip; checkpoint
# by validation mean E[R'].
# KILL CRITERION: if validation mean E[R'] has not beaten the stabilized
# both-reward pair's joint E[R'] (printed as KILL BAR) within 3 epochs -> stop.
# CERTIFIED VIEWS BY IMPORT, NEVER BY COPY: reward_train_ret.build_rows and
# reward_train_qa.build_rows supply the prompts.
# USAGE:
#   python joint_train.py --check     # CPU-only: rewards, anchors, kill bar
#   python joint_train.py             # full run
import os, sys, json, time, random, argparse, hashlib, subprocess
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch

os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import reward_train_ret as RT
import reward_train_qa as QT
import router_prompts as RP

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LR, EPOCHS, ACCUM, EVAL_EVERY, SEED = 1e-5, 3, 8, 100, 23
BETA = 0.03
CURVE_EVAL_MAX = 1110
WARM_RET = "/mnt/hdd/jonathan/runs/sft_router5_ret/best_adapter"
WARM_QA = "/mnt/hdd/jonathan/runs/sft_router5_qa/best_adapter"
REF_RET = "/mnt/hdd/jonathan/runs/reward_train_ret2/val_scores.jsonl"
REF_QA = "/mnt/hdd/jonathan/runs/reward_train_qa4/val_scores.jsonl"
PATHS, C_SS, LAMBDA = RT.PATH, RT.C_SS, RT.LAMBDA

def build_joint_rows(data):
    ret_rows = RT.build_rows(data)
    qa_rows = QT.build_rows(data)
    probe = qa_rows["train"][0]
    for k in ("idx", "user", "ret_choice"):
        assert k in probe, f"reward_train_qa rows lack key {k!r}; found {sorted(probe)} -- stopping"
    qa_user = {}
    for sp in ("train", "val"):
        for r in qa_rows[sp]:
            qa_user[(r["idx"], r["ret_choice"])] = r["user"]
    qa_w = RT.load_store(os.path.join(RT.GRID, "qa_weak.jsonl"))
    qa_s = RT.load_store(os.path.join(RT.GRID, "qa_strong.jsonl"))
    rows = {"train": [], "val": []}
    for sp in ("train", "val"):
        for r in ret_rows[sp]:
            i = r["idx"]; golds = data[i]["answers"]
            assert (i, "W") in qa_user and (i, "S") in qa_user, f"missing qa prompt for idx {i}"
            R = {}
            for rc, sfx in (("W", "weak"), ("S", "strong")):
                R[(rc, "W")] = RT.tokenF1(str(qa_w[f"{i}|{sfx}"]), golds) - LAMBDA * PATHS[(rc, "W")] / C_SS
                R[(rc, "S")] = RT.tokenF1(str(qa_s[f"{i}|{sfx}"]), golds) - LAMBDA * PATHS[(rc, "S")] / C_SS
            rows[sp].append({"idx": i, "ret_user": r["user"],
                             "qaW_user": qa_user[(i, "W")], "qaS_user": qa_user[(i, "S")],
                             "R": R, "ret_target": r["target"]})
    return rows

def anchors(rows):
    n = len(rows)
    aWW = sum(r["R"][("W", "W")] for r in rows) / n
    aSS = sum(r["R"][("S", "S")] for r in rows) / n
    best = sum(max(r["R"].values()) for r in rows) / n
    return aWW, aSS, best

def kill_bar(rows):
    retp = {r["idx"]: r["pW"] for r in map(json.loads, open(REF_RET))}
    qap = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(REF_QA))}
    tot = 0.0
    for r in rows:
        i = r["idx"]; pr = retp[i]; qW = qap[(i, "W")]; qS = qap[(i, "S")]
        tot += (pr * (qW * r["R"][("W", "W")] + (1 - qW) * r["R"][("W", "S")])
                + (1 - pr) * (qS * r["R"][("S", "W")] + (1 - qS) * r["R"][("S", "S")]))
    return tot / len(rows)

def auc_of(ps, ys):
    pairs = sorted(zip(ps, ys))
    pos = [i + 1 for i, (p, y) in enumerate(pairs) if y == 1]
    nP, nN = len(pos), len(pairs) - len(pos)
    return (sum(pos) - nP * (nP + 1) / 2) / (nP * nN) if nP and nN else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/mnt/hdd/jonathan/runs/joint_train_j")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    chk = subprocess.run([sys.executable, os.path.join(HERE, "router_prompts.py"),
                          os.path.join(HERE, "probe_ret_coverage.py")],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        print("DRIFT CHECK FAILED -- refusing to run.\n", chk.stdout, chk.stderr); sys.exit(1)
    print("drift check OK", flush=True)

    random.seed(SEED); torch.manual_seed(SEED)
    data = json.load(open(RT.DATA))
    rows = build_joint_rows(data)
    aWW, aSS, best = anchors(rows["val"])
    bar = kill_bar(rows["val"])
    print(f"train={len(rows['train'])}  val={len(rows['val'])}")
    print(f"val anchors (joint E[R']): always-WW {aWW:.4f}  always-SS {aSS:.4f}  best-path {best:.4f}")
    print(f"KILL BAR (stabilized 12 Oct pair, joint E[R']): {bar:.4f}", flush=True)

    if a.check:
        print("\n--check: reward table for 3 sample validation rows")
        for r in rows["val"][:3]:
            print(f"  idx {r['idx']}: " + "  ".join(
                f"{rc}{ac}={r['R'][(rc, ac)]:+.4f}" for rc in "WS" for ac in "WS")
                + f"  ret_target={r['ret_target']}")
        print("--check complete; nothing trained, nothing written."); return

    os.makedirs(a.outdir, exist_ok=True)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU before load: {free/2**30:.1f} GiB free of {total/2**30:.1f}", flush=True)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)

    def load_router(warm):
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16, device_map="cuda")
        m = PeftModel.from_pretrained(m, warm, is_trainable=True)
        m.gradient_checkpointing_enable(); m.enable_input_require_grads()
        m.config.use_cache = False
        return m
    retM = load_router(WARM_RET)
    qaM = load_router(WARM_QA)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU after load: {free/2**30:.1f} GiB free", flush=True)
    params = [p for m in (retM, qaM) for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)

    def dist(model, user):
        ids = RP.assemble(tok, user).to("cuda")
        z = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return torch.softmax(torch.stack([z[idW], z[idS]]), 0)

    def joint_exp(r):
        pr = dist(retM, r["ret_user"])
        qW = dist(qaM, r["qaW_user"])
        qS = dist(qaM, r["qaS_user"])
        exp = (pr[0] * (qW[0] * r["R"][("W", "W")] + qW[1] * r["R"][("W", "S")])
               + pr[1] * (qS[0] * r["R"][("S", "W")] + qS[1] * r["R"][("S", "S")]))
        ent = sum(-(d[0] * torch.log(d[0] + 1e-8) + d[1] * torch.log(d[1] + 1e-8))
                  for d in (pr, qW, qS))
        return exp, ent, float(pr[0]), float(qW[0]), float(qS[0])

    curve_val = rows["val"] if len(rows["val"]) <= CURVE_EVAL_MAX \
        else random.sample(rows["val"], CURVE_EVAL_MAX)

    def evaluate(vrows):
        retM.eval(); qaM.eval()
        exp_t, prs, ys = 0.0, [], []
        with torch.no_grad():
            for r in vrows:
                e, _, pr0, _, _ = joint_exp(r)
                exp_t += float(e); prs.append(pr0)
                ys.append(1 if r["ret_target"] == "W" else 0)
        retM.train(); qaM.train()
        return exp_t / len(vrows), auc_of(prs, ys), prs

    curve = open(os.path.join(a.outdir, "curve.jsonl"), "a")
    best_obj, step, skipped, t0 = -1e9, 0, 0, time.time()
    order = rows["train"][:]
    for ep in range(EPOCHS):
        random.shuffle(order); opt.zero_grad()
        for k, r in enumerate(order):
            R = r["R"]
            with torch.no_grad():                   # cheap value pass, builds no graphs
                pr0d = float(dist(retM, r["ret_user"])[0])
                qW0d = float(dist(qaM, r["qaW_user"])[0])
                qS0d = float(dist(qaM, r["qaS_user"])[0])
            Ad = qW0d * R[("W", "W")] + (1 - qW0d) * R[("W", "S")]
            Bd = qS0d * R[("S", "W")] + (1 - qS0d) * R[("S", "S")]
            bad = False
            # staged backward: one graph alive at a time; by the product rule
            # the three staged gradients sum to EXACTLY the joint gradient
            for which, model, user, c0, c1, w in (
                    ("ret", retM, r["ret_user"], Ad, Bd, 1.0),
                    ("qaW", qaM, r["qaW_user"], R[("W", "W")], R[("W", "S")], pr0d),
                    ("qaS", qaM, r["qaS_user"], R[("S", "W")], R[("S", "S")], 1 - pr0d)):
                d = dist(model, user)
                ent = -(d[0] * torch.log(d[0] + 1e-8) + d[1] * torch.log(d[1] + 1e-8))
                part = -(w * (d[0] * c0 + d[1] * c1) + BETA * ent) / ACCUM
                if not torch.isfinite(part):
                    skipped += 1; bad = True
                    print(f"skipping non-finite {which} loss at epoch {ep} row {k}", flush=True)
                    break
                part.backward()
            if bad:
                continue
            if (k + 1) % ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step(); opt.zero_grad(); step += 1
                if step % EVAL_EVERY == 0:
                    meanR, aucv, prs = evaluate(curve_val)
                    line = {"step": step, "epoch": ep, "val_meanR": round(meanR, 4),
                            "vs_kill_bar": round(meanR - bar, 4),
                            "mean_pW_ret": round(sum(prs) / len(prs), 4),
                            "ret_auc_vs_labels": round(aucv, 4), "skipped": skipped,
                            "min": round((time.time() - t0) / 60, 1)}
                    print(json.dumps(line), flush=True)
                    curve.write(json.dumps(line) + "\n"); curve.flush()
                    if meanR > best_obj:
                        best_obj = meanR
                        retM.save_pretrained(os.path.join(a.outdir, "ret_adapter"))
                        qaM.save_pretrained(os.path.join(a.outdir, "qa_adapter"))

    retM.save_pretrained(os.path.join(a.outdir, "ret_adapter_last"))
    qaM.save_pretrained(os.path.join(a.outdir, "qa_adapter_last"))

    del retM, qaM; torch.cuda.empty_cache()
    from transformers import AutoModelForCausalLM as AM
    retM = PeftModel.from_pretrained(AM.from_pretrained(MODEL, dtype=torch.float16,
            device_map="cuda"), os.path.join(a.outdir, "ret_adapter")).eval()
    qaM = PeftModel.from_pretrained(AM.from_pretrained(MODEL, dtype=torch.float16,
            device_map="cuda"), os.path.join(a.outdir, "qa_adapter")).eval()
    exp_t = 0.0
    with torch.no_grad(), \
         open(os.path.join(a.outdir, "val_scores_ret.jsonl"), "w") as fr, \
         open(os.path.join(a.outdir, "val_scores_qa.jsonl"), "w") as fq:
        for r in rows["val"]:
            e, _, pr0, qW0, qS0 = joint_exp(r)
            exp_t += float(e)
            fr.write(json.dumps({"idx": r["idx"], "pW": round(pr0, 6)}) + "\n")
            fq.write(json.dumps({"idx": r["idx"], "ret_choice": "W", "pW": round(qW0, 6)}) + "\n")
            fq.write(json.dumps({"idx": r["idx"], "ret_choice": "S", "pW": round(qS0, 6)}) + "\n")
    manifest = {"experiment": "J_joint_exact", "objective":
                f"maximize joint E[R'] over 4 cached paths + {BETA}*entropy(all 3 dists); "
                f"R' = tokenF1 - {LAMBDA}*C_path/C_SS",
                "base_model": MODEL, "warm_ret": WARM_RET, "warm_qa": WARM_QA,
                "lr": LR, "epochs": EPOCHS, "accum": ACCUM, "seed": SEED,
                "beta": BETA, "lambda": LAMBDA, "c_ss": round(C_SS, 4),
                "kill_bar_stabilized_pair": round(bar, 4),
                "val_anchors": {"always_WW": round(aWW, 4), "always_SS": round(aSS, 4),
                                "best_path": round(best, 4)},
                "skipped_nonfinite": skipped,
                "best_val_joint_meanR_fullval": round(exp_t / len(rows["val"]), 4)}
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)

if __name__ == "__main__":
    main()
