# oracle_standard.py -- the oracle row (cheapest succeeding path; unsolved pay the
# cheapest path as a floor) under the STANDARD ruler, token-aware prices.
import json
GRID = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_standard.jsonl"
tc = json.load(open("/mnt/hdd/jonathan/runs/token_costs/token_costs.json"))["gate_eval_constants"]
RW, RS = tc["COST_RET_W"], tc["COST_RET_S"]
QWW, QWS = tc["COST_QA_W_GIVEN_W"], tc["COST_QA_W_GIVEN_S"]
QSW, QSS = tc["COST_QA_S_GIVEN_W"], tc["COST_QA_S_GIVEN_S"]
PATH = [RW+QWW, RW+QWW+QSW, RW+RS+QWS, RW+RS+QWS+QSS]   # WW, WS, SW, SS
for split in ("val", "test"):
    acc = cost = unsolved = n = 0
    for g in map(json.loads, open(GRID)):
        if g["split"] != split: continue
        e = g["em_standard"]; n += 1; acc += max(e)
        succ = [PATH[c] for c in range(4) if e[c]]
        cost += min(succ) if succ else min(PATH)
        unsolved += 0 if succ else 1
    print(f"{split} (n={n}): oracle acc {100*acc/n:.2f}  "
          f"token-aware cost {100*(cost/n)/PATH[3]:.1f}% of SS  unsolved {100*unsolved/n:.1f}%")
