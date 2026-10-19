# 10 sft_router5.py -- the production router trainer (SFT + LoRA).
# One training loop, two routers:
#   --stage ret : adapter-Ret, view INSTR_RET2 (question + weak's selections),
#                 labels ret_labels.jsonl -> ret_target
#   --stage qa  : adapter-QA,  view INSTR_QA1 (question + evidence in use +
#                 weak's answer), labels qa_labels.jsonl -> qa_target
#                 (2 rows/instance, one per retrieval choice; teacher-1 lane --
#                 the comparative lane was retired by 7's gate, 7 Oct)
#
# Inherited, certified choices: prompt contract from router_prompts (600 /
# 16384, audit 7 Oct, re-cert AUC 0.766); LoRA r16/a32/qkvo fp16; class
# weights from the stage manifest; checkpoint selection by val AUC, NEVER
# match-at-0.5; gradient checkpointing + last-position-only scores.
# Refuses to train if the prompt drift check fails.
import os, sys, json, csv, time, random, argparse, hashlib, subprocess
import torch
os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = "/mnt/hdd/jonathan/data/tts_test.json"
GRID = "/mnt/hdd/jonathan/runs/config_b"
CFG = {
  "ret": {"labels": "/mnt/hdd/jonathan/runs/config_b/ret_labels/ret_labels.jsonl",
          "manifest": "/mnt/hdd/jonathan/runs/config_b/ret_labels/manifest.json",
          "target": "ret_target"},
  "qa":  {"labels": "/mnt/hdd/jonathan/runs/config_b/qa_labels/qa_labels.jsonl",
          "manifest": "/mnt/hdd/jonathan/runs/config_b/qa_labels/manifest.json",
          "target": "qa_target"},
}
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
LR, EPOCHS, ACCUM, EVAL_EVERY, SEED = 1e-4, 3, 8, 100, 17
CURVE_EVAL_MAX = 1110   # curve evals use at most this many val rows (seeded);
                        # the FINAL eval + val_scores dump always use the full val set

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

def build_rows(stage, data):
    cfg = CFG[stage]
    lab = [json.loads(l) for l in open(cfg["labels"])]
    retr = {"W": load_store(os.path.join(GRID, "retr_weak.jsonl")),
            "S": load_store(os.path.join(GRID, "retr_strong.jsonl"))}
    qa_w = load_store(os.path.join(GRID, "qa_weak.jsonl")) if stage == "qa" else None
    rows = {"train": [], "val": []}
    for r in lab:
        if r["split"] not in rows: continue
        i = r["idx"]; inst = data[i]
        pool = R.trim_pool(inst)
        if stage == "ret":
            sel = R.parse_selection(retr["W"][str(i)], len(pool))
            user = RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel),
                       selections=RP.format_selections(sel, pool))
            rows[r["split"]].append({"idx": i, "user": user, "target": r[cfg["target"]]})
        else:
            rc = r["ret_choice"]; sfx = "weak" if rc == "W" else "strong"
            sel = R.parse_selection(retr[rc][str(i)], len(pool))
            user = RP.INSTR_QA1.format(q=inst["question"],
                       evidence=RP.format_evidence(R.build_block(inst, sel)),
                       weak_answer=str(qa_w[f"{i}|{sfx}"]))
            rows[r["split"]].append({"idx": i, "ret_choice": rc,
                                     "user": user, "target": r[cfg["target"]]})
    return rows, cfg

def auc_of(ps, ys):
    pairs = sorted(zip(ps, ys))
    pos = [i+1 for i, (p, y) in enumerate(pairs) if y == 1]
    nP, nN = len(pos), len(pairs) - len(pos)
    return (sum(pos) - nP*(nP+1)/2) / (nP*nN) if nP and nN else float("nan")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("ret", "qa"), required=True)
    ap.add_argument("--outdir", default=None)
    a = ap.parse_args()
    outdir = a.outdir or f"/mnt/hdd/jonathan/runs/config_b/sft_router5_{a.stage}"
    os.makedirs(outdir, exist_ok=True)

    # pre-flight: the prompt contract must not have drifted (checklist item)
    chk = subprocess.run([sys.executable, os.path.join(HERE, "router_prompts.py"),
                          os.path.join(HERE, "probe_ret_coverage.py")],
                         capture_output=True, text=True)
    if chk.returncode != 0:
        print("DRIFT CHECK FAILED -- refusing to train.\n", chk.stdout, chk.stderr)
        sys.exit(1)
    print("drift check OK", flush=True)

    random.seed(SEED); torch.manual_seed(SEED)
    data = json.load(open(DATA))
    rows, cfg = build_rows(a.stage, data)
    cw = json.load(open(cfg["manifest"]))["class_weights_post_mapping"]
    print(f"stage={a.stage}  train={len(rows['train'])}  val={len(rows['val'])}  "
          f"weights={cw}", flush=True)

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

    curve = open(os.path.join(outdir, "curve.jsonl"), "a")
    best_auc, step, t0 = -1.0, 0, time.time()
    order = rows["train"][:]
    for ep in range(EPOCHS):
        random.shuffle(order); opt.zero_grad()
        for k, r in enumerate(order):
            zW, zS = logits_WS(r["user"])
            logp = torch.log_softmax(torch.stack([zW, zS]), 0)
            loss = -cw[r["target"]] * (logp[0] if r["target"] == "W" else logp[1]) / ACCUM
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
                    if auc > best_auc:        # selection by AUC, never match
                        best_auc = auc
                        model.save_pretrained(os.path.join(outdir, "best_adapter"))

    model.save_pretrained(os.path.join(outdir, "last_adapter"))
    # final: reload the BEST adapter and score the FULL val set (feeds the theta sweep)
    del model; torch.cuda.empty_cache()
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                device_map="cuda")
    model = PeftModel.from_pretrained(base, os.path.join(outdir, "best_adapter")).eval()
    auc, match, ps = evaluate(rows["val"])
    with open(os.path.join(outdir, "val_scores.jsonl"), "w") as f:
        for r, p in zip(rows["val"], ps):
            rec = {"idx": r["idx"], "pW": round(p, 6)}
            if "ret_choice" in r: rec["ret_choice"] = r["ret_choice"]
            f.write(json.dumps(rec) + "\n")
    manifest = {"stage": a.stage, "base_model": MODEL,
                "labels_file": cfg["labels"], "labels_sha256": sha256(cfg["labels"]),
                "char_cap": RP.CHAR_CAP, "maxlen": RP.MAXLEN,
                "lr": LR, "epochs": EPOCHS, "accum": ACCUM, "seed": SEED,
                "class_weights": cw, "curve_eval_rows": len(curve_val),
                "best_val_auc_fullval": round(auc, 4),
                "fullval_match_at_0.5": round(match, 4)}
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=2)
    print(json.dumps({"FINAL": True, **manifest}), flush=True)

if __name__ == "__main__":
    main()
