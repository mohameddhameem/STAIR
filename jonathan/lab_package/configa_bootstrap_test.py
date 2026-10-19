# configa_bootstrap_test.py -- bootstrap verdicts on Config A's sealed test (read-only).
# B=1,000 resamples, seed 20261016. Frozen thetas from the certified readout.
import json, random

R = "/mnt/hdd/jonathan/runs"
TC = json.load(open(f"{R}/token_costs/token_costs.json"))["gate_eval_constants"]
C_SS = TC["COST_RET_W"] + TC["COST_RET_S"] + TC["COST_QA_W_GIVEN_S"] + TC["COST_QA_S_GIVEN_S"]
PRICE = {0: TC["COST_RET_W"] + TC["COST_QA_W_GIVEN_W"],
         1: TC["COST_RET_W"] + TC["COST_QA_W_GIVEN_W"] + TC["COST_QA_S_GIVEN_W"],
         2: TC["COST_RET_W"] + TC["COST_RET_S"] + TC["COST_QA_W_GIVEN_S"], 3: C_SS}
test = [g for g in map(json.loads, open(f"{R}/full_run_p100/router_data_standard.jsonl")) if g["split"] == "test"]
def load(p): return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(p))}
ret_l, qa_l, qa_r = load(f"{R}/test_scores/ret_label.jsonl"), load(f"{R}/test_scores/qa_label.jsonl"), load(f"{R}/test_scores/qa_reward.jsonl")

def cell(g, qa, tr, tq):
    i = g["idx"]
    if ret_l[(i, None)] >= tr: return 0 if qa[(i, "W")] >= tq else 1
    return 2 if qa[(i, "S")] >= tq else 3

rows = []
for g in test:
    em = g["em_standard"]
    cl = cell(g, qa_l, 0.8354, 0.9751)   # label only, certified theta
    cr = cell(g, qa_r, 0.7881, 0.6372)   # reward answering, certified theta
    rows.append((em[3], C_SS, em[cl], PRICE[cl], em[cr], PRICE[cr]))

rng = random.Random(20261016); n = len(rows)
wins = {"label_acc_>=_SS": 0, "label_cheaper_than_SS": 0, "rw_acc_>=_SS": 0, "rw_cheaper_than_SS": 0, "rw_cheaper_than_label": 0}
for _ in range(1000):
    s = [rows[rng.randrange(n)] for _ in range(n)]
    ssA = sum(r[0] for r in s)/n; ssC = sum(r[1] for r in s)/n
    lA = sum(r[2] for r in s)/n; lC = sum(r[3] for r in s)/n
    rA = sum(r[4] for r in s)/n; rC = sum(r[5] for r in s)/n
    wins["label_acc_>=_SS"] += lA >= ssA; wins["label_cheaper_than_SS"] += lC < ssC
    wins["rw_acc_>=_SS"] += rA >= ssA;    wins["rw_cheaper_than_SS"] += rC < ssC
    wins["rw_cheaper_than_label"] += rC < lC
print(f"test n={n}  B=1000  seed=20261016")
for k, v in wins.items(): print(f"  {k}: {v/1000:.3f}")
