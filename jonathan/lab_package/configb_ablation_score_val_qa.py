# configb_ablation_score_val_qa.py -- scores the FOUR answering-router sizes on
# the VALIDATION split (2,220 rows = 1,110 instances x both retrieval branches),
# timing every decision. Test split untouched.
# Row construction mirrors configb_score_test.py's test_rows_qa with ONE change:
# split == "val". Prompts, parsing and inference are the trainers' certified
# code via import. Writes ablation_eval/val_scores_qa_<size>.jsonl + summary_qa.json.
import os, json, time, torch
import configb_reward_train_qa as QT
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

B   = "/mnt/hdd/jonathan/runs/config_b"
OUT = os.path.join(B, "ablation_eval")
SIZES = [  # (name, base model, adapter)
 ("0.5B", "Qwen/Qwen2.5-0.5B-Instruct", f"{B}/ablation_qa_05b/best_adapter"),
 ("1.5B", "Qwen/Qwen2.5-1.5B-Instruct", f"{B}/sft_router5_qa/best_adapter"),
 ("3B",   "Qwen/Qwen2.5-3B-Instruct",   f"{B}/ablation_qa_3b/best_adapter"),
 ("7B",   "Qwen/Qwen2.5-7B-Instruct",   f"{B}/ablation_qa_7b/best_adapter"),
]
WARMUP = 10

def val_rows_qa(data):
    lab = [json.loads(l) for l in open(QT.LABELS)]
    retr = {"W": QT.load_store(os.path.join(QT.GRID, "retr_weak.jsonl")),
            "S": QT.load_store(os.path.join(QT.GRID, "retr_strong.jsonl"))}
    qa_w = QT.load_store(os.path.join(QT.GRID, "qa_weak.jsonl"))
    rows = []
    for r in lab:
        if r["split"] != "val": continue
        i = r["idx"]; inst = data[i]
        rc = r["ret_choice"]; sfx = "weak" if rc == "W" else "strong"
        pool = QT.R.trim_pool(inst)
        sel = QT.R.parse_selection(retr[rc][str(i)], len(pool))
        user = QT.RP.INSTR_QA1.format(q=inst["question"],
                   evidence=QT.RP.format_evidence(QT.R.build_block(inst, sel)),
                   weak_answer=str(qa_w[f"{i}|{sfx}"]))
        rows.append({"idx": i, "ret_choice": rc, "user": user,
                     "y": 1 if r["qa_target"] == "W" else 0})
    return rows

def rank_auc(ps, ys):
    pairs = sorted(zip(ps, ys))
    pos = [i+1 for i, (p, y) in enumerate(pairs) if y == 1]
    nP, nN = len(pos), len(pairs) - len(pos)
    return (sum(pos) - nP*(nP+1)/2) / (nP*nN)

def main():
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(QT.DATA))
    rows = val_rows_qa(data)
    print("val qa rows:", len(rows), "(expect 2220)", flush=True)
    summary = {}
    for name, model_name, adapter in SIZES:
        out_path = os.path.join(OUT, f"val_scores_qa_{name}.jsonl")
        if not os.path.isdir(adapter):
            print("skip (no adapter yet):", name, adapter); continue
        if os.path.exists(out_path):
            print("skip (already scored):", out_path); continue
        tok = AutoTokenizer.from_pretrained(model_name)
        idW, idS = QT.RP.letter_ids(tok)
        base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float16, device_map="cuda")
        model = PeftModel.from_pretrained(base, adapter).eval()
        ps, ys, ms = [], [], []
        with open(out_path, "w") as f, torch.no_grad():
            for k, r in enumerate(rows):
                torch.cuda.synchronize(); t0 = time.perf_counter()
                ids = QT.RP.assemble(tok, r["user"]).to("cuda")
                z = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
                pW = float(torch.softmax(torch.stack([z[idW], z[idS]]), 0)[0])
                torch.cuda.synchronize(); dt = (time.perf_counter() - t0) * 1000
                ps.append(pW); ys.append(r["y"])
                if k >= WARMUP: ms.append(dt)
                f.write(json.dumps({"idx": r["idx"], "ret_choice": r["ret_choice"],
                                    "pW": round(pW, 6), "ms": round(dt, 2)}) + "\n")
                if (k+1) % 400 == 0: print(name, k+1, flush=True)
        ms_sorted = sorted(ms)
        summary[name] = {"model": model_name, "n": len(ps),
            "val_auc": round(rank_auc(ps, ys), 4),
            "match_at_0.5": round(sum((p >= 0.5) == bool(y) for p, y in zip(ps, ys)) / len(ps), 4),
            "ms_mean": round(sum(ms)/len(ms), 1), "ms_median": round(ms_sorted[len(ms)//2], 1)}
        print(name, json.dumps(summary[name]), flush=True)
        del model, base; torch.cuda.empty_cache()
    json.dump(summary, open(os.path.join(OUT, "summary_qa.json"), "w"), indent=2)
    print("wrote", os.path.join(OUT, "summary_qa.json"), flush=True)

if __name__ == "__main__":
    main()
