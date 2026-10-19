# configb_ablation_score_val.py -- scores the FOUR retrieval-router sizes on the
# VALIDATION split (1,110 rows), timing every decision. Test split untouched.
# Prompts, parsing and inference reuse the trainers' certified code via import,
# exactly as configb_score_test.py does; row construction mirrors its
# test_rows_ret with ONE change: split == "val".
# Writes /mnt/hdd/jonathan/runs/config_b/ablation_eval/val_scores_<size>.jsonl
# and ablation_eval/summary.json. Inputs read-only.
import os, json, time, torch
import configb_reward_train_ret as RT
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

B   = "/mnt/hdd/jonathan/runs/config_b"
OUT = os.path.join(B, "ablation_eval")
SIZES = [  # (name, base model, adapter)
 ("0.5B", "Qwen/Qwen2.5-0.5B-Instruct", f"{B}/ablation_ret_05b/best_adapter"),
 ("1.5B", "Qwen/Qwen2.5-1.5B-Instruct", f"{B}/sft_router5_ret/best_adapter"),
 ("3B",   "Qwen/Qwen2.5-3B-Instruct",   f"{B}/ablation_ret_3b/best_adapter"),
 ("7B",   "Qwen/Qwen2.5-7B-Instruct",   f"{B}/ablation_ret_7b/best_adapter"),
]
WARMUP = 10  # first N decisions excluded from timing stats (CUDA warm-up)

def val_rows_ret(data):
    lab = [json.loads(l) for l in open(RT.LABELS)]
    retr_w = RT.load_store(os.path.join(RT.GRID, "retr_weak.jsonl"))
    rows = []
    for r in lab:
        if r["split"] != "val": continue
        i = r["idx"]; inst = data[i]
        pool = RT.R.trim_pool(inst)
        sel = RT.R.parse_selection(retr_w[str(i)], len(pool))
        user = RT.RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel),
                   selections=RT.RP.format_selections(sel, pool))
        rows.append({"idx": i, "user": user, "y": 1 if r["ret_target"] == "W" else 0})
    return rows

def rank_auc(ps, ys):
    pairs = sorted(zip(ps, ys))
    pos = [i+1 for i, (p, y) in enumerate(pairs) if y == 1]
    nP, nN = len(pos), len(pairs) - len(pos)
    return (sum(pos) - nP*(nP+1)/2) / (nP*nN)

def main():
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(RT.DATA))
    rows = val_rows_ret(data)
    print("val rows:", len(rows), "(expect 1110)", flush=True)
    summary = {}
    for name, model_name, adapter in SIZES:
        out_path = os.path.join(OUT, f"val_scores_{name}.jsonl")
        if not os.path.isdir(adapter):
            print("skip (no adapter yet):", name, adapter); continue
        if os.path.exists(out_path):
            print("skip (already scored):", out_path); continue
        tok = AutoTokenizer.from_pretrained(model_name)
        idW, idS = RT.RP.letter_ids(tok)
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")
        model = PeftModel.from_pretrained(base, adapter).eval()
        ps, ys, ms = [], [], []
        with open(out_path, "w") as f, torch.no_grad():
            for k, r in enumerate(rows):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ids = RT.RP.assemble(tok, r["user"]).to("cuda")
                z = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
                pW = float(torch.softmax(torch.stack([z[idW], z[idS]]), 0)[0])
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
                ps.append(pW); ys.append(r["y"])
                if k >= WARMUP: ms.append(dt)
                f.write(json.dumps({"idx": r["idx"], "pW": round(pW, 6), "ms": round(dt, 2)}) + "\n")
                if (k+1) % 200 == 0: print(name, k+1, flush=True)
        ms_sorted = sorted(ms)
        summary[name] = {"model": model_name, "n": len(ps),
            "val_auc": round(rank_auc(ps, ys), 4),
            "match_at_0.5": round(sum((p >= 0.5) == bool(y) for p, y in zip(ps, ys)) / len(ps), 4),
            "ms_mean": round(sum(ms)/len(ms), 1), "ms_median": round(ms_sorted[len(ms)//2], 1)}
        print(name, json.dumps(summary[name]), flush=True)
        del model, base; torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "summary.json"), flush=True)

if __name__ == "__main__":
    main()
