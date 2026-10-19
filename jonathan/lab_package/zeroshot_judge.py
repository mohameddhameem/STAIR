# B1 baseline: zero-shot LLM judge on the Ret routing decision.
# Same base model, same prompt contract, same val split as the trained router
# (probe AUC 0.7663 at 16K) -- the ONLY difference is: no training.
import os, json, torch
os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")

MODEL  = os.environ.get("JUDGE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
LABELS = "/mnt/hdd/jonathan/runs/ret_labels/ret_labels.jsonl"
GRID   = "/mnt/hdd/jonathan/runs/full_run_p100"
DATA   = "/mnt/hdd/jonathan/data/tts_test.json"
OUT    = "/mnt/hdd/jonathan/runs/zeroshot_judge"

import run_labels4 as R            # frozen pipeline: trim_pool, parse_selection, Store
import router_prompts as RP        # the certified prompt contract (600 / 16384)
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = [json.loads(l) for l in open(LABELS)]
    val  = [r for r in rows if r.get("split") == "val"]
    data = json.load(open(DATA))
    store = R.Store(os.path.join(GRID, "retr_weak.jsonl"))   # read-only use
    print(f"val rows: {len(val)} | judge: {MODEL}", flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)                            # single-token W / S
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, dtype=torch.float16, device_map="cuda").eval()

    ps, ys, miss = [], [], 0
    with torch.no_grad():
        for k, r in enumerate(val):
            i = r["idx"]; key = str(i)
            if key not in store: miss += 1; continue
            inst = data[i]
            pool = R.trim_pool(inst)
            sel_ids = R.parse_selection(store[key], len(pool))
            user = RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel_ids),
                      selections=RP.format_selections(sel_ids, pool))
            ids = RP.assemble(tok, user).to("cuda")
            out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
            pW  = RP.pair_probs(out, idW, idS)[0]
            ps.append(float(pW))
            ys.append(1 if r["ret_target"] == "W" else 0)    # 2-class target, X->W
            if (k+1) % 200 == 0: print(f"{k+1}/{len(val)}", flush=True)

    pairs = sorted(zip(ps, ys))                              # rank-based AUC, as the probe
    pos = [i+1 for i, (p, y) in enumerate(pairs) if y == 1]
    nP, nN = len(pos), len(pairs) - len(pos)
    auc = (sum(pos) - nP*(nP+1)/2) / (nP*nN)
    match = sum((p >= 0.5) == bool(y) for p, y in zip(ps, ys)) / len(ps)
    res = {"judge": MODEL, "n": len(ps), "missing": miss,
           "zero_shot_auc": round(auc, 4), "match_at_0.5": round(match, 4),
           "trained_router_auc_16k": 0.7663}
    print(json.dumps(res), flush=True)
    json.dump(res, open(os.path.join(OUT, "result.json"), "w"), indent=2)

if __name__ == "__main__":
    main()
