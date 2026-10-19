# configa_test_readout.py -- Config A's sealed-test readout (read-only, CPU).
# For each pair: re-derive the operating point on VALIDATION exactly as the
# 13 Oct standard-ruler re-selection did -- evaluate every theta row of the
# pair's frontier.csv on standard-ruler val accuracy, pick the PEAK accuracy
# row (ties -> cheapest) -- then apply those frozen thetas ONCE to the test
# scores. Costs: Config A's measured token family (weak-first), C_SS = 250.06.
import json, csv

R = "/mnt/hdd/jonathan/runs"
GRID = f"{R}/full_run_p100/router_data_standard.jsonl"
TC = json.load(open(f"{R}/token_costs/token_costs.json"))["gate_eval_constants"]
C_SS = TC["COST_RET_W"] + TC["COST_RET_S"] + TC["COST_QA_W_GIVEN_S"] + TC["COST_QA_S_GIVEN_S"]
PRICE = {0: TC["COST_RET_W"] + TC["COST_QA_W_GIVEN_W"],
         1: TC["COST_RET_W"] + TC["COST_QA_W_GIVEN_W"] + TC["COST_QA_S_GIVEN_W"],
         2: TC["COST_RET_W"] + TC["COST_RET_S"] + TC["COST_QA_W_GIVEN_S"],
         3: C_SS}

rows = [json.loads(l) for l in open(GRID)]
val  = [g for g in rows if g["split"] == "val"]
test = [g for g in rows if g["split"] == "test"]

def load(path):
    return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(path))}

PAIRS = [
 ("label only",       f"{R}/sft_router5_ret/val_scores.jsonl",  f"{R}/sft_router5_qa/val_scores.jsonl",
                      f"{R}/test_scores/ret_label.jsonl",       f"{R}/test_scores/qa_label.jsonl",
                      f"{R}/gate_eval/frontier.csv"),
 ("reward answering", f"{R}/sft_router5_ret/val_scores.jsonl",  f"{R}/reward_train_qa4/val_scores.jsonl",
                      f"{R}/test_scores/ret_label.jsonl",       f"{R}/test_scores/qa_reward.jsonl",
                      f"{R}/gate_eval_rw_qa4/frontier.csv"),
]

def evaluate(rows_, ret, qa, tr, tq):
    acc = cost = 0
    for g in rows_:
        i, em = g["idx"], g["em_standard"]
        if ret[(i, None)] >= tr: c = 0 if qa[(i, "W")] >= tq else 1
        else:                    c = 2 if qa[(i, "S")] >= tq else 3
        acc += em[c]; cost += PRICE[c]
    n = len(rows_)
    return 100*acc/n, 100*(cost/n)/C_SS

ss_t = 100*sum(g["em_standard"][3] for g in test)/len(test)
ss_v = 100*sum(g["em_standard"][3] for g in val)/len(val)
print(f"always-SS: val {ss_v:.2f} | TEST {ss_t:.2f} (both at 100% = {C_SS:.1f} units)")

for name, rv, qv, rt, qt, fcsv in PAIRS:
    ret_v, qa_v = load(rv), load(qv)
    ret_t, qa_t = load(rt), load(qt)
    thetas = {(float(r["theta_ret"]), float(r["theta_qa"])) for r in csv.DictReader(open(fcsv))}
    best = None
    for tr, tq in sorted(thetas):
        a, c = evaluate(val, ret_v, qa_v, tr, tq)
        if best is None or a > best[0] + 1e-9 or (abs(a - best[0]) < 1e-9 and c < best[1]):
            best = (a, c, tr, tq)
    a_v, c_v, tr, tq = best
    a_t, c_t = evaluate(test, ret_t, qa_t, tr, tq)
    print(f"{name}: selected on val {a_v:.2f} at {c_v:.1f}% (theta {tr:.4f}/{tq:.4f})")
    print(f"  SEALED TEST: {a_t:.2f} at {c_t:.1f}%  (vs SS {ss_t:.2f})")
