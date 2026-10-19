# configb_score_test.py -- scores the four Config B routers on the SEALED TEST split.
# Prompts, parsing, token ids and inference reuse the trainers' own certified code
# via import (RT/QT namespaces). Row construction mirrors their build_rows
# line-for-line with ONE change: split == "test" instead of train/val.
# Writes /mnt/hdd/jonathan/runs/config_b/test_scores/*.jsonl. Inputs read-only.
import os, json, torch
import reward_train_ret2 as RT
import reward_train_qa4 as QT
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

B   = "/mnt/hdd/jonathan/runs"
OUT = os.path.join(B, "test_scores")
ADAPTERS = {
 "ret_label":  ("ret", f"{B}/sft_router5_ret/best_adapter"),
 "ret_reward": ("ret", f"{B}/reward_train_ret2/best_adapter"),
 "qa_label":   ("qa",  f"{B}/sft_router5_qa/best_adapter"),
 "qa_reward":  ("qa",  f"{B}/reward_train_qa4/best_adapter"),
}

def test_rows_ret(data):
    lab = [json.loads(l) for l in open(RT.LABELS)]
    retr_w = RT.load_store(os.path.join(RT.GRID, "retr_weak.jsonl"))
    rows = []
    for r in lab:
        if r["split"] != "test": continue
        i = r["idx"]; inst = data[i]
        pool = RT.R.trim_pool(inst)
        sel = RT.R.parse_selection(retr_w[str(i)], len(pool))
        user = RT.RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel),
                   selections=RT.RP.format_selections(sel, pool))
        rows.append({"idx": i, "user": user})
    return rows

def test_rows_qa(data):
    lab = [json.loads(l) for l in open(QT.LABELS)]
    retr = {"W": QT.load_store(os.path.join(QT.GRID, "retr_weak.jsonl")),
            "S": QT.load_store(os.path.join(QT.GRID, "retr_strong.jsonl"))}
    qa_w = QT.load_store(os.path.join(QT.GRID, "qa_weak.jsonl"))
    rows = []
    for r in lab:
        if r["split"] != "test": continue
        i = r["idx"]; inst = data[i]
        rc = r["ret_choice"]; sfx = "weak" if rc == "W" else "strong"
        pool = QT.R.trim_pool(inst)
        sel = QT.R.parse_selection(retr[rc][str(i)], len(pool))
        user = QT.RP.INSTR_QA1.format(q=inst["question"],
                   evidence=QT.RP.format_evidence(QT.R.build_block(inst, sel)),
                   weak_answer=str(qa_w[f"{i}|{sfx}"]))
        rows.append({"idx": i, "ret_choice": rc, "user": user})
    return rows

def main():
    os.makedirs(OUT, exist_ok=True)
    data = json.load(open(RT.DATA))
    rows = {"ret": test_rows_ret(data), "qa": test_rows_qa(data)}
    print("test rows: ret", len(rows["ret"]), "(expect 1481)  qa", len(rows["qa"]), "(expect 2962)", flush=True)
    tok = AutoTokenizer.from_pretrained(RT.MODEL)
    idW, idS = RT.RP.letter_ids(tok)
    for name, (stage, adapter) in ADAPTERS.items():
        out_path = os.path.join(OUT, f"{name}.jsonl")
        if os.path.exists(out_path):
            print("skip (already scored):", out_path); continue
        base = AutoModelForCausalLM.from_pretrained(RT.MODEL, dtype=torch.float16, device_map="cuda")
        model = PeftModel.from_pretrained(base, adapter).eval()
        with open(out_path, "w") as f, torch.no_grad():
            for k, r in enumerate(rows[stage]):
                ids = RT.RP.assemble(tok, r["user"]).to("cuda")
                z = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
                pW = float(torch.softmax(torch.stack([z[idW], z[idS]]), 0)[0])
                o = {"idx": r["idx"], "pW": round(pW, 6)}
                if stage == "qa": o["ret_choice"] = r["ret_choice"]
                f.write(json.dumps(o) + "\n")
                if (k+1) % 500 == 0: print(name, k+1, flush=True)
        del model, base; torch.cuda.empty_cache()
        print("wrote", out_path, flush=True)

if __name__ == "__main__":
    main()
