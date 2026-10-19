# configb_bootstrap_test.py -- bootstrap verdicts on the SEALED TEST rows (read-only, CPU).
# B=1,000 resamples, seed 20261015. For each duel: resample the n=1,481 test rows
# with replacement and count how often the claim holds. Same recipe as Config A's
# bootstrap (means over resampled rows; fixed frozen thresholds; no tuning).
import json, random

B_DIR = "/mnt/hdd/jonathan/runs/config_b"
PRICE = [80, 195, 894, 1009]; C_SS = 929
BOOT, SEED = 1000, 20261015

test = [g for g in map(json.loads, open(f"{B_DIR}/router_data_standard.jsonl")) if g["split"] == "test"]
def load(fn):
    return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(f"{B_DIR}/test_scores/{fn}.jsonl"))}
ret_l, qa_l, qa_r = load("ret_label"), load("qa_label"), load("qa_reward")

def decide(g, ret, qa, tr, tw, ts):
    i = g["idx"]
    if ret[(i, None)] >= tr: return 0 if qa[(i, "W")] >= tw else 1
    return 2 if qa[(i, "S")] >= ts else 3

# per-row (accuracy, cost) for each policy
rows = []
for g in test:
    em = g["em_standard"]
    c_lab = decide(g, ret_l, qa_l, 0.80, 0.90, 1.00)   # label-only, parity point
    c_pk  = decide(g, ret_l, qa_l, 0.95, 1.00, 1.00)   # label-only, peak point
    c_rw  = decide(g, ret_l, qa_r, 0.80, 0.55, 0.60)   # reward answering, parity point
    rows.append((em[3], 929,                       # always-SS (its own bill: 929)
                 em[c_lab], PRICE[c_lab], em[c_pk], PRICE[c_pk], em[c_rw], PRICE[c_rw]))

rng = random.Random(SEED)
n = len(rows)
wins = {"rw_cheaper_than_SS":0, "rw_acc_>=_SS":0, "pk_acc_>=_SS":0,
        "rw_acc_>_label":0, "rw_cheaper_than_label":0}
for b in range(BOOT):
    s = [rows[rng.randrange(n)] for _ in range(n)]
    ssA  = sum(r[0] for r in s)/n; ssC  = sum(r[1] for r in s)/n
    labA = sum(r[2] for r in s)/n; labC = sum(r[3] for r in s)/n
    pkA  = sum(r[4] for r in s)/n
    rwA  = sum(r[6] for r in s)/n; rwC  = sum(r[7] for r in s)/n
    wins["rw_cheaper_than_SS"]    += rwC < ssC
    wins["rw_acc_>=_SS"]          += rwA >= ssA
    wins["pk_acc_>=_SS"]          += pkA >= ssA
    wins["rw_acc_>_label"]        += rwA > labA
    wins["rw_cheaper_than_label"] += rwC < labC
print(f"test n={n}  B={BOOT}  seed={SEED}")
for k, v in wins.items():
    print(f"  {k}: {v/BOOT:.3f}")
