# 12 compose_eval.py -- EVALUATION OF THE RUNTIME (by replay).
# Routers decide live; each decision's outcome and cost are looked up in the
# deterministic grid cache (greedy decoding => replay == runtime).
#
#   --mode val  : compose the trained configuration end-to-end on VALIDATION
#                 with the locked thetas; cross-checks 11's locked row.
#   --mode test : THE ONE SEALED-TEST READ. Guards, pre-declared:
#                 (1) refuses unless theta_lock.json exists;
#                 (2) refuses unless the locked row's val accuracy >= the lock's
#                     "bar" (0.95 x SS val accuracy, measured by 11);
#                 (3) refuses if test_result.json already exists (the vault
#                     is read ONCE); scores the 1,481 test instances with the
#                 best adapters (the only new GPU work), composes once with
#                 theta_bar ONLY, writes test_result.json with the Ruler rungs.
import os, sys, json, argparse
import torch

os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")

GRID_DIR  = "/mnt/hdd/jonathan/runs/full_run_p100"
GRID_FILE = os.path.join(GRID_DIR, "router_data_lenient.jsonl")
DATA      = "/mnt/hdd/jonathan/data/tts_test.json"
RET_DIR   = "/mnt/hdd/jonathan/runs/sft_router5_ret"
QA_DIR    = "/mnt/hdd/jonathan/runs/sft_router5_qa"
LOCK      = "/mnt/hdd/jonathan/runs/gate_eval/theta_lock.json"
OUT       = "/mnt/hdd/jonathan/runs/compose_eval"
MODEL     = "Qwen/Qwen2.5-1.5B-Instruct"
# Test trigger is NOT hardcoded: it is read from theta_lock.json ("bar" =
# 0.95 x SS val accuracy, measured by 11) so there is one source of truth.

import run_labels4 as R
import router_prompts as RP
import gate_eval as GE        # single source for the flat cost family

def load_jsonl(p): return [json.loads(l) for l in open(p)]
def load_store(p):
    d = {}
    for line in open(p):
        try:
            r = json.loads(line); d[r["k"]] = r["v"]
        except Exception: pass
    return d

def compose(idxs, grid, retp, qap, tR, tQ):
    n = acc = cost = 0
    for i in idxs:
        em = grid[i]["em_lenient"]
        escR = retp[i] < tR
        rc = "S" if escR else "W"
        escQ = qap[(i, rc)] < tQ
        ok = em[(2 if escR else 0) + (1 if escQ else 0)]
        # prices conditional on the retrieval choice (GIVEN_S when escalated)
        c = GE.COST_RET_W + (GE.COST_RET_S if escR else 0) \
            + (GE.COST_QA_W_GIVEN_S if escR else GE.COST_QA_W_GIVEN_W)
        if escQ:
            c += GE.COST_QA_S_GIVEN_S if escR else GE.COST_QA_S_GIVEN_W
        n += 1; acc += ok; cost += c
    return 100.0 * acc / n, cost / n, n

def score_split(split, grid_rows, data):
    """Score a split's instances with both best adapters. Resume-safe k/v store."""
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel
    os.makedirs(OUT, exist_ok=True)
    spath = os.path.join(OUT, f"{split}_scores.jsonl")
    done = load_store(spath) if os.path.exists(spath) else {}
    f = open(spath, "a")
    retr = {"W": load_store(os.path.join(GRID_DIR, "retr_weak.jsonl")),
            "S": load_store(os.path.join(GRID_DIR, "retr_strong.jsonl"))}
    qa_w = load_store(os.path.join(GRID_DIR, "qa_weak.jsonl"))
    tok = AutoTokenizer.from_pretrained(MODEL)
    idW, idS = RP.letter_ids(tok)
    def scorer(adapter_dir):
        base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16,
                                                    device_map="cuda")
        return PeftModel.from_pretrained(base, os.path.join(adapter_dir,
                                                            "best_adapter")).eval()
    jobs = []
    for g in grid_rows:
        i = g["idx"]; inst = data[i]; pool = R.trim_pool(inst)
        selW = R.parse_selection(retr["W"][str(i)], len(pool))
        jobs.append((f"ret|{i}", "ret",
            RP.INSTR_RET2.format(q=inst["question"], n_sel=len(selW),
                selections=RP.format_selections(selW, pool))))
        for rc, sfx in (("W", "weak"), ("S", "strong")):
            sel = R.parse_selection(retr[rc][str(i)], len(pool))
            jobs.append((f"qa|{i}|{rc}", "qa",
                RP.INSTR_QA1.format(q=inst["question"],
                    evidence=RP.format_evidence(R.build_block(inst, sel)),
                    weak_answer=str(qa_w[f"{i}|{sfx}"]))))
    for stage, adir in (("ret", RET_DIR), ("qa", QA_DIR)):
        todo = [(k, u) for k, s, u in jobs if s == stage and k not in done]
        if not todo:
            print(f"{stage}: all {split} scores cached"); continue
        model = scorer(adir)
        with torch.no_grad():
            for j, (k, user) in enumerate(todo):
                ids = RP.assemble(tok, user).to("cuda")
                out = model(input_ids=ids, logits_to_keep=1).logits[0, -1]
                pW = RP.pair_probs(out, idW, idS)[0]
                done[k] = float(pW)
                f.write(json.dumps({"k": k, "v": float(pW)}) + "\n")
                if (j+1) % 200 == 0: print(f"{stage} {j+1}/{len(todo)}", flush=True)
        f.flush(); del model; torch.cuda.empty_cache()
    retp = {int(k.split("|")[1]): v for k, v in done.items() if k.startswith("ret|")}
    qap  = {(int(k.split("|")[1]), k.split("|")[2]): v
            for k, v in done.items() if k.startswith("qa|")}
    return retp, qap

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("val", "test"), required=True)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if not os.path.exists(LOCK):
        raise SystemExit("REFUSED: theta_lock.json does not exist -- run gate_eval.py (11) first.")
    lock = json.load(open(LOCK))
    grid_all = load_jsonl(GRID_FILE)
    data = json.load(open(DATA))

    if a.mode == "val":
        grid = {g["idx"]: g for g in grid_all if g["split"] == "val"}
        retp = {r["idx"]: r["pW"] for r in load_jsonl(os.path.join(RET_DIR, "val_scores.jsonl"))}
        qap  = {(r["idx"], r["ret_choice"]): r["pW"]
                for r in load_jsonl(os.path.join(QA_DIR, "val_scores.jsonl"))}
        for name in ("theta_bar", "theta_knee"):
            t = lock.get(name)
            if t is None: continue
            acc, cost, n = compose(sorted(grid), grid, retp, qap,
                                   t["theta_ret"], t["theta_qa"])
            print(f"[val] {name}: accuracy={acc:.2f} cost={cost:.3f} n={n} "
                  f"(lock recorded: {t['val_accuracy']} / {t['cost']})")
        return

    # ---- mode test: the one read ----
    result_path = os.path.join(OUT, "test_result.json")
    if os.path.exists(result_path):
        raise SystemExit("REFUSED: test_result.json exists -- the sealed test was already read.")
    t = lock.get("theta_bar") or lock.get("theta_knee")
    primary = lock.get("primary", "theta_bar")
    if t is None:
        raise SystemExit("REFUSED: no locked theta in theta_lock.json.")
    trigger = lock["bar"]        # 0.95 x SS val accuracy, measured by 11
    if t["val_accuracy"] < trigger:
        raise SystemExit(f"REFUSED: locked val accuracy {t['val_accuracy']} < bar "
                         f"{trigger} (95% of SS val accuracy "
                         f"{lock['ss_val_accuracy']}) -- iterate on validation.")
    grid = {g["idx"]: g for g in grid_all if g["split"] == "test"}
    print(f"sealed-test read: n={len(grid)} with {primary} "
          f"(theta_ret={t['theta_ret']}, theta_qa={t['theta_qa']})")
    retp, qap = score_split("test", [grid[i] for i in sorted(grid)], data)
    acc, cost, n = compose(sorted(grid), grid, retp, qap,
                           t["theta_ret"], t["theta_qa"])
    import numpy as np
    em = np.array([grid[i]["em_lenient"] for i in sorted(grid)])
    rungs = {"WW": round(100*em[:,0].mean(),2), "WS": round(100*em[:,1].mean(),2),
             "SW": round(100*em[:,2].mean(),2), "SS": round(100*em[:,3].mean(),2),
             "oracle_any": round(100*em.max(axis=1).mean(),2)}
    result = {"test_accuracy": round(acc, 2), "test_cost": round(cost, 3), "n": n,
              "theta_used": t, "primary": primary, "ruler_test_rungs": rungs,
              "cost_family": "flat (Ruler slide); re-express via 13 for AgentTTS comparison",
              "read_discipline": "single read; this file's existence blocks any re-read"}
    json.dump(result, open(result_path, "w"), indent=2)
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
