# reward_train_qa.py -- THE FIRST OPTIMIZATION AGAINST THE REGISTERED REWARD.
# Trains the ANSWERING router to maximize EXPECTED R' (not to imitate labels):
#
#   loss per row = -( pi_W * R'(deliver weak) + pi_S * R'(escalate) )
#   R'(choice)   = tokenF1(that answer)  -  LAMBDA * C_path / C_SS
#
# K=1 exact form of the GRPO idea: with two actions and cached outcomes the
# expectation is computed exactly -- no sampling, no variance. pi_W/pi_S come
# from the router's own letter logits (softmax), so the gradient teaches
# CALIBRATION against the reward: it pushes probability toward the higher-R'
# action with force |R'_W - R'_S| * pi_W * pi_S -- the pi_W*pi_S factor damps
# updates once the router is confident (unlike cross-entropy, which keeps
# pushing -- the overconfidence spiral of the v1/v2 weighting runs).
#
# Scope (agreed 11 Oct): answering router ONLY; the retrieval router is NOT
# touched (frozen = Wednesday's adapter stays as-is); it joins in the next
# experiment. Training covers BOTH retrieval branches per instance (two rows,
# like the label file), so the result composes with any retrieval threshold.
# Rewards use gold at training time only (F1 vs gold), never as router input.
# Registered parameters: LAMBDA = 0.10, B = C_SS (hinge silent at K=1).
import os, sys, json, re, string, time, random, argparse, hashlib, subprocess
from collections import Counter
import torch
os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/hdd/jonathan/data/tts_test.json"
GRID = "/mnt/hdd/jonathan/runs/config_b"
LABELS = "/mnt/hdd/jonathan/runs/config_b/qa_labels/qa_labels.jsonl"   # rows + splits only
COSTS = "/mnt/hdd/jonathan/runs/config_b/token_costs/token_costs.json"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LR, EPOCHS, ACCUM, EVAL_EVERY, SEED = 1e-5, 3, 8, 100, 17   # 10x below SFT rate: policy refinement, not from-scratch learning
CURVE_EVAL_MAX = 1110
LAMBDA = 0.10
BETA = 0.03
WARM_START = "/mnt/hdd/jonathan/runs/config_b/sft_router5_qa/best_adapter"  # SFT-then-reward: start from the class-weighted answering adapter     # entropy bonus: keeps pi from saturating before discrimination is learned

import run_labels4 as R
import router_prompts as RP
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

tc = json.load(open(COSTS))["gate_eval_constants"]
C_SS = tc["COST_RET_S"] + tc["COST_QA_S_GIVEN_S"]   # budget = baseline SS without router (814+115=929), registered 15 Oct
PATH = {("W", "W"): tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"],
        ("W", "S"): tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"] + tc["COST_QA_S_GIVEN_W"],
        ("S", "W"): tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"],
        ("S", "S"): C_SS}

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

def base_(s): return ''.join(c for c in s.lower() if c not in set(string.punctuation))
def norm_(s): return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', base_(s)).split())
def tokenF1(pred, golds):
    pt = norm_(pred).split(); best = 0.0
    for g in golds:
        gt = norm_(g).split()
        if not pt or not gt: continue
        common = sum((Counter(pt) & Counter(gt)).values())
        if common:
            p, r = common / len(pt), common / len(gt)
            best = max(best, 2 * p * r / (p + r))
    return best

def load_store(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line); d[r["k"]] = r["v"]
        except Exception: pass
    return d

def build_rows(data):
    lab = [json.loads(l) for l in open(LABELS)]
    retr = {"W": load_store(os.path.join(GRID, "retr_weak.jsonl")),
            "S": load_store(os.path.join(GRID, "retr_strong.jsonl"))}
    qa_w = load_store(os.path.join(GRID, "qa_weak.jsonl"))
    qa_s = load_store(os.path.join(GRID, "qa_strong.jsonl"))
    rows = {"train": [], "val": []}
    for r in lab:
        if r["split"] not in rows: continue
        i = r["idx"]; inst = data[i]
        rc = r["ret_choice"]; sfx = "weak" if rc == "W" else "strong"
        pool = R.trim_pool(inst)
        sel = R.parse_selection(retr[rc][str(i)], len(pool))
        user = RP.INSTR_QA1.format(q=inst["question"],
                   evidence=RP.format_evidence(R.build_block(inst, sel)),
                   weak_answer=str(qa_w[f"{i}|{sfx}"]))
        golds = inst["answers"]
        rW = tokenF1(str(qa_w[f"{i}|{sfx}"]), golds) - LAMBDA * PATH[(rc, "W")] / C_SS
        rS = tokenF1(str(qa_s[f"{i}|{sfx}"]), golds) - LAMBDA * PATH[(rc, "S")] / C_SS
        rows[r["split"]].append({"idx": i, "ret_choice": rc, "user": user,
                                 "rW": rW, "rS": rS,
                                 "target": r["qa_target"]})   # kept for AUC reporting only
    return rows

def auc_of(ps, ys):
    pairs = sorted(zip(ps, ys))
    pos = [i+1 for i, (p, y) in enumerate(pairs) if y == 1]
    nP, nN = len(pos), len(pairs) - len(pos)
    return (sum(pos) - nP*(nP+1)/2) / (nP*nN) if nP and nN else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/mnt/hdd/jonathan/runs/config_b/reward_train_qa")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    chk = subprocess.run([sys.executable, os.path.join(HERE, "router_prompts.py"),
                          os.path.join(HERE, "probe_ret_coverage.py")],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        print("DRIFT CHECK FAILED -- refusing to train.\n", chk.stdout, chk.stderr)
        sys.exit(1)
    print("drift check OK", flush=True)

    random.seed(SEED); torch.manual_seed(SEED)
    data = json.load(open(DATA))
    rows = build_rows(data)
    dR = [abs(r["rW"] - r["rS"]) for r in rows["train"]]
    print(f"objective=expected R' (tokenF1 - {LAMBDA}*C/C_SS)  train={len(rows['train'])}  "
          f"val={len(rows['val'])}  mean|R'_W - R'_S|={sum(dR)/len(dR):.4f}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 device_map="cuda")
    model = PeftModel.from_pretrained(model, WARM_START, is_trainable=True)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)

    def logits_WS(user):
        ids = RP.assemble(tok, user).to("cuda")
        out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
        return out[idW], out[idS]

    curve_val = rows["val"] if len(rows["val"]) <= CURVE_EVAL_MAX \
                else random.sample(rows["val"], CURVE_EVAL_MAX)

    def evaluate(vrows):
        model.eval(); ps, ys, exp = [], [], 0.0
        with torch.no_grad():
            for r in vrows:
                zW, zS = logits_WS(r["user"])
                pW = float(torch.softmax(torch.stack([zW, zS]), 0)[0])
                ps.append(pW); ys.append(1 if r["target"] == "W" else 0)
                exp += pW * r["rW"] + (1 - pW) * r["rS"]
        model.train()
        return auc_of(ps, ys), exp / len(vrows), ps

    curve = open(os.path.join(a.outdir, "curve.jsonl"), "a")
    best_obj, step, t0 = -1e9, 0, time.time()
    order = rows["train"][:]
    for ep in range(EPOCHS):
        random.shuffle(order); opt.zero_grad()
        for k, r in enumerate(order):
            zW, zS = logits_WS(r["user"])
            pW = torch.softmax(torch.stack([zW, zS]), 0)[0]
            exp_reward = pW * r["rW"] + (1 - pW) * r["rS"]
            pS = 1 - pW
            entropy = -(pW * torch.log(pW + 1e-8) + pS * torch.log(pS + 1e-8))
            loss = -(exp_reward + BETA * entropy) / ACCUM
            loss.backward()
            if (k+1) % ACCUM == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % EVAL_EVERY == 0:
                    aucv, meanR, ps_ = evaluate(curve_val)
                    line = {"step": step, "epoch": ep, "val_meanR": round(meanR, 4),
                            "mean_pW": round(sum(ps_)/len(ps_), 4),
                            "pW_gt_half": round(sum(p > 0.5 for p in ps_)/len(ps_), 4),
                            "val_auc_vs_labels": round(aucv, 4),
                            "min": round((time.time()-t0)/60, 1)}
                    print(json.dumps(line), flush=True)
                    curve.write(json.dumps(line) + "\n"); curve.flush()
                    if meanR > best_obj:          # checkpoint by THE OBJECTIVE
                        best_obj = meanR
                        model.save_pretrained(os.path.join(a.outdir, "best_adapter"))

    model.save_pretrained(os.path.join(a.outdir, "last_adapter"))
    del model; torch.cuda.empty_cache()
    basem = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 device_map="cuda")
    model = PeftModel.from_pretrained(basem, os.path.join(a.outdir, "best_adapter")).eval()
    aucv, meanR, ps = evaluate(rows["val"])
    with open(os.path.join(a.outdir, "val_scores.jsonl"), "w") as f:
        for r, p in zip(rows["val"], ps):
            f.write(json.dumps({"idx": r["idx"], "pW": round(p, 6),
                                "ret_choice": r["ret_choice"]}) + "\n")
    manifest = {"stage": "qa_reward_optimized_entropy_warmstart", "objective":
                f"maximize E[R'] + {BETA}*entropy, R' = tokenF1 - {LAMBDA}*C/C_SS (B=C_SS, hinge silent at K=1)",
                "base_model": MODEL, "warm_start": WARM_START, "labels_file": LABELS,
                "labels_sha256": sha256(LABELS),
                "char_cap": RP.CHAR_CAP, "maxlen": RP.MAXLEN,
                "lr": LR, "epochs": EPOCHS, "accum": ACCUM, "seed": SEED,
                "lambda": LAMBDA, "c_ss": round(C_SS, 4),
                "curve_eval_rows": len(curve_val),
                "best_val_meanR_fullval": round(meanR, 4),
                "fullval_auc_vs_labels": round(aucv, 4)}
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)

if __name__ == "__main__":
    main()
