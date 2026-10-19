# validate_compose.py -- review points 2 and 3 for compose_eval.py. CPU-only,
# read-only, needs no trained router (dummy scores; thetas at +/- infinity).
#
# Part A (point 2, the corner check): with thetas forced to +/- infinity every
#   instance lands in one fixed configuration (WW/WS/SW/SS), so compose() must
#   reproduce that grid column's val accuracy EXACTLY, and the fixed cost.
# Part B (point 3): renders one retrieval prompt and one QA prompt through the
#   same assembly calls score_split uses, into a text file for eyeball
#   comparison with the prompt render approved for sft_router5.
import os, json
import numpy as np
import compose_eval as CE

grid_all = CE.load_jsonl(CE.GRID_FILE)
grid = {g["idx"]: g for g in grid_all if g["split"] == "val"}
idxs = sorted(grid)
em = np.array([grid[i]["em_lenient"] for i in idxs])
dummyR = {i: 0.5 for i in idxs}
dummyQ = {(i, rc): 0.5 for i in idxs for rc in ("W", "S")}
INF = float("inf")
G = CE.GE

print(f"Part A: corner check on {len(idxs)} val instances")
# expected costs from gate_eval's measured constants (QA price conditional on
# the retrieval choice: GIVEN_W / GIVEN_S = evidence from weak / strong retrieval)
cases = [("WW never escalate   ", -INF, -INF, 0,
          G.COST_RET_W + G.COST_QA_W_GIVEN_W),
         ("WS escalate QA only ", -INF,  INF, 1,
          G.COST_RET_W + G.COST_QA_W_GIVEN_W + G.COST_QA_S_GIVEN_W),
         ("SW escalate Ret only",  INF, -INF, 2,
          G.COST_RET_W + G.COST_RET_S + G.COST_QA_W_GIVEN_S),
         ("SS escalate both    ",  INF,  INF, 3,
          G.COST_RET_W + G.COST_RET_S + G.COST_QA_W_GIVEN_S + G.COST_QA_S_GIVEN_S)]
fails = 0
for name, tR, tQ, col, want_cost in cases:
    acc, cost, n = CE.compose(idxs, grid, dummyR, dummyQ, tR, tQ)
    want_acc = 100.0 * em[:, col].mean()
    ok = abs(acc - want_acc) < 1e-9 and abs(cost - want_cost) < 1e-9
    fails += 0 if ok else 1
    print(f"[{name}] {'PASS' if ok else 'FAIL'}: "
          f"accuracy {acc:.4f} (expect {want_acc:.4f}) | "
          f"cost {cost:.3f} (expect {want_cost:.3f})")
print("Part A:", "ALL PASS -- outcome mapping proven on every val instance"
      if fails == 0 else f"{fails} FAILURES -- stop and report")

print("\nPart B: prompt render via score_split's assembly path")
data = json.load(open(CE.DATA))
retr = {"W": CE.load_store(os.path.join(CE.GRID_DIR, "retr_weak.jsonl")),
        "S": CE.load_store(os.path.join(CE.GRID_DIR, "retr_strong.jsonl"))}
qa_w = CE.load_store(os.path.join(CE.GRID_DIR, "qa_weak.jsonl"))
i = idxs[0]
inst = data[i]; pool = CE.R.trim_pool(inst)
selW = CE.R.parse_selection(retr["W"][str(i)], len(pool))
ret_prompt = CE.RP.INSTR_RET2.format(
    q=inst["question"], n_sel=len(selW),
    selections=CE.RP.format_selections(selW, pool))
qa_prompt = CE.RP.INSTR_QA1.format(
    q=inst["question"],
    evidence=CE.RP.format_evidence(CE.R.build_block(inst, selW)),
    weak_answer=str(qa_w[f"{i}|weak"]))
out = "/mnt/hdd/jonathan/runs/compose_prompt_render.txt"
with open(out, "w") as f:
    f.write(f"=== val instance idx {i} ===\n\n--- RETRIEVAL prompt ---\n"
            + ret_prompt + "\n\n--- QA prompt (retrieval choice W) ---\n"
            + qa_prompt + "\n")
print(f"wrote {out}")
print("compare with the sft_router5 render you approved: same question style,")
print("same 'Selected references' numbering, same evidence block, one-letter ask.")
