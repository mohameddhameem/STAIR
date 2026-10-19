# sft_router5_rw.py -- REWARD-WEIGHTED SFT for the ANSWERING router.
# Derived from sft_router5.py (the certified trainer); the differences are
# deliberately few and listed here in full:
#   1. qa stage only (retrieval-stage stakes need a downstream assumption --
#      its weighting is a separate, future design).
#   2. labels: qa_labels_rw.jsonl (from add_qa_stakes.py) -- same rows as
#      qa_labels.jsonl plus per-row reward stakes.
#   3. loss: per-example REWARD weight replaces the per-class weight.
#      weights are normalized to mean 1 over the training rows, then floored
#      at W_FLOOR so quality-tie rows keep a small gradient.
#   4. output dir: sft_router5rw_qa (Wednesday's adapters are never touched).
# Everything else -- prompt contract import, drift-check pre-flight, LoRA
# shape, AUC-based checkpointing, full-val rescoring -- is unchanged.
import os, sys, json, time, random, argparse, hashlib, subprocess
import torch
os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/hdd/jonathan/data/tts_test.json"
GRID = "/mnt/hdd/jonathan/runs/full_run_p100"
LABELS = "/mnt/hdd/jonathan/runs/qa_labels/qa_labels_rw.jsonl"
TARGET = "qa_target"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LR, EPOCHS, ACCUM, EVAL_EVERY, SEED = 1e-4, 3, 8, 100, 17
CURVE_EVAL_MAX = 1110
W_FLOOR = 0.05          # minimum normalized weight for quality-tie rows

import run_labels4 as R
import router_prompts as RP
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, PeftModel

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()

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
    rows = {"train": [], "val": []}
    for r in lab:
        if r["split"] not in rows: continue
        i = r["idx"]; inst = data[i]
        pool = R.trim_pool(inst)
        rc = r["ret_choice"]; sfx = "weak" if rc == "W" else "strong"
        sel = R.parse_selection(retr[rc][str(i)], len(pool))
        user = RP.INSTR_QA1.format(q=inst["question"],
                   evidence=RP.format_evidence(R.build_block(inst, sel)),
                   weak_answer=str(qa_w[f"{i}|{sfx}"]))
        rows[r["split"]].append({"idx": i, "ret_choice": rc, "user": user,
                                 "target": r[TARGET], "weight": r["weight"]})
    # normalize weights to mean 1 over TRAIN, then floor
    wm = sum(r["weight"] for r in rows["train"]) / len(rows["train"])
    for split in rows:
        for r in rows[split]:
            r["w"] = max(W_FLOOR, r["weight"] / wm)
    return rows, wm

def auc_of(ps, ys):
    pairs = sorted(zip(ps, ys))
    pos = [i+1 for i, (p, y) in enumerate(pairs) if y == 1]
    nP, nN = len(pos), len(pairs) - len(pos)
    return (sum(pos) - nP*(nP+1)/2) / (nP*nN) if nP and nN else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="/mnt/hdd/jonathan/runs/sft_router5rw_qa")
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
    rows, wmean = build_rows(data)
    hi = sum(1 for r in rows["train"] if r["w"] > 1.0)
    print(f"stage=qa(reward-weighted)  train={len(rows['train'])}  "
          f"val={len(rows['val'])}  raw-weight mean={wmean:.4f}  "
          f"rows above mean weight: {hi}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                 device_map="cuda")
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"], task_type="CAUSAL_LM"))
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
        model.eval(); ps, ys = [], []
        with torch.no_grad():
            for r in vrows:
                zW, zS = logits_WS(r["user"])
                ps.append(float(torch.softmax(torch.stack([zW, zS]), 0)[0]))
                ys.append(1 if r["target"] == "W" else 0)
        model.train()
        match = sum((p >= 0.5) == bool(y) for p, y in zip(ps, ys)) / len(ps)
        return auc_of(ps, ys), match, ps

    curve = open(os.path.join(a.outdir, "curve.jsonl"), "a")
    best_auc, step, t0 = -1.0, 0, time.time()
    order = rows["train"][:]
    for ep in range(EPOCHS):
        random.shuffle(order); opt.zero_grad()
        for k, r in enumerate(order):
            zW, zS = logits_WS(r["user"])
            logp = torch.log_softmax(torch.stack([zW, zS]), 0)
            loss = -r["w"] * (logp[0] if r["target"] == "W" else logp[1]) / ACCUM
            loss.backward()
            if (k+1) % ACCUM == 0:
                opt.step(); opt.zero_grad(); step += 1
                if step % EVAL_EVERY == 0:
                    auc, match, _ = evaluate(curve_val)
                    line = {"step": step, "epoch": ep, "val_auc": round(auc, 4),
                            "val_match": round(match, 4),
                            "min": round((time.time()-t0)/60, 1)}
                    print(json.dumps(line), flush=True)
                    curve.write(json.dumps(line) + "\n"); curve.flush()
                    if auc > best_auc:
                        best_auc = auc
                        model.save_pretrained(os.path.join(a.outdir, "best_adapter"))

    model.save_pretrained(os.path.join(a.outdir, "last_adapter"))
    del model; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, os.path.join(a.outdir, "best_adapter")).eval()
    auc, match, ps = evaluate(rows["val"])
    with open(os.path.join(a.outdir, "val_scores.jsonl"), "w") as f:
        for r, p in zip(rows["val"], ps):
            f.write(json.dumps({"idx": r["idx"], "pW": round(p, 6),
                                "ret_choice": r["ret_choice"]}) + "\n")
    manifest = {"stage": "qa_reward_weighted", "base_model": MODEL,
                "labels_file": LABELS, "labels_sha256": sha256(LABELS),
                "char_cap": RP.CHAR_CAP, "maxlen": RP.MAXLEN,
                "lr": LR, "epochs": EPOCHS, "accum": ACCUM, "seed": SEED,
                "weighting": {"alpha": 0.74, "gamma": 0.25, "w_floor": W_FLOOR,
                              "raw_weight_mean": round(wmean, 6),
                              "note": "per-example |delta R'| replaces class weights"},
                "curve_eval_rows": len(curve_val),
                "best_val_auc_fullval": round(auc, 4),
                "fullval_match_at_0.5": round(match, 4)}
    json.dump(manifest, open(os.path.join(a.outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)

if __name__ == "__main__":
    main()
