# grpo_fresh_bootstrap.py -- bootstrap verdicts for the label-free champion test read.
# Same recipe as configb_bootstrap_test.py: B=1,000 resamples, seed 20261015,
# frozen thresholds, means over resampled rows, no tuning.
import json, random
B_DIR = "/mnt/hdd/jonathan/runs/config_b"
PRICE = [80, 195, 894, 1009]; C_SS = 929
BOOT, SEED = 1000, 20261015
test = [g for g in map(json.loads, open(f"{B_DIR}/router_data_standard.jsonl")) if g["split"] == "test"]
def load(fn):
    return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(f"{B_DIR}/test_scores/{fn}.jsonl"))}
ret, qa = load("ret_grpo_fresh_objective"), load("qa_grpo_fresh_objective")
def decide(g, tr, tw, ts):
    i = g["idx"]
    if ret[(i, None)] >= tr: return 0 if qa[(i, "W")] >= tw else 1
    return 2 if qa[(i, "S")] >= ts else 3
rows = []
for g in test:
    em = g["em_standard"]
    cg = decide(g, 0.50, 0.50, 0.50)   # greedy, frozen
    cc = decide(g, 0.80, 0.05, 0.05)   # cap<=70% point, frozen 28 Oct
    rows.append((em[3], em[cg], PRICE[cg], em[cc], PRICE[cc]))
rng = random.Random(SEED); n = len(rows)
wins = {"greedy_cheaper_than_SS": 0, "cap_cheaper_than_SS": 0,
        "greedy_acc_>=_SS": 0, "cap_acc_>=_SS": 0}
for b in range(BOOT):
    s = [rows[rng.randrange(n)] for _ in range(n)]
    ssA = sum(r[0] for r in s)/n
    gA  = sum(r[1] for r in s)/n; gC = sum(r[2] for r in s)/n
    cA  = sum(r[3] for r in s)/n; cC = sum(r[4] for r in s)/n
    wins["greedy_cheaper_than_SS"] += gC < C_SS
    wins["cap_cheaper_than_SS"]    += cC < C_SS
    wins["greedy_acc_>=_SS"]       += gA >= ssA
    wins["cap_acc_>=_SS"]          += cA >= ssA
for k, v in wins.items():
    print(f"{k}: {v}/{BOOT}")

