# grpo_test_readout.py -- the GRPO sealed-test readout (read-only, CPU).
# VERBATIM COPY of the certified configb_test_readout.py logic; ONLY the RUNS
# table differs. Second read of the sealed test, DECLARED 21 Oct before
# execution: thresholds below are frozen from the 21 Oct validation sweep
# (grpo_pair_sweep.py); no tuning here. Reports both certified protocols.
import json

B = "/mnt/hdd/jonathan/runs/config_b"
GRID  = f"{B}/router_data_standard.jsonl"
TS    = f"{B}/test_scores"
PRICE = [80, 195, 894, 1009]
C_SS  = 929

# pair -> (ret file, qa file, [(point name, t_ret, t_qaW, t_qaS), ...])  all frozen from val, 21 Oct
RUNS = [
 ("GRPO(Reward) label-free champion", "ret_grpo_fresh_objective", "qa_grpo_fresh_objective",
  [("greedy (router's own choice)", 0.50, 0.50, 0.50),
   ("cap<=70% point (frozen 28 Oct)", 0.80, 0.05, 0.05)]),
]

test = [g for g in map(json.loads, open(GRID)) if g["split"] == "test"]
sub  = [g for g in test if g["idx"] < 500]
print(f"test n={len(test)} | AgentTTS-slice rows n={len(sub)}")
for name, rows in (("full test", test), ("105-slice", sub)):
    ss = sum(g["em_standard"][3] for g in rows) / len(rows)
    ww = sum(g["em_standard"][0] for g in rows) / len(rows)
    print(f"  {name}: always-SS {100*ss:.2f} at 100% | always-WW {100*ww:.2f} at {100*80/C_SS:.1f}%")
print("  AgentTTS published (their pipeline, all 500): 74.6 at ~101%")

def load(fn, qa=False):
    d = {}
    for r in map(json.loads, open(f"{TS}/{fn}.jsonl")):
        d[(r["idx"], r.get("ret_choice"))] = r["pW"]
    return d

for pair, rf, qf, points in RUNS:
    ret, qa = load(rf), load(qf, True)
    for pname, tr, tw, ts in points:
        out = []
        for rows, label in ((test, "n=1,481"), (sub, "n=105 ")):
            acc = cost = 0
            for g in rows:
                i, em = g["idx"], g["em_standard"]
                pr = ret[(i, None)]
                if pr >= tr:  c = 0 if qa[(i, "W")] >= tw else 1
                else:         c = 2 if qa[(i, "S")] >= ts else 3
                acc += em[c]; cost += PRICE[c]
            out.append(f"{label}: {100*acc/len(rows):.2f} at {100*(cost/len(rows))/C_SS:.1f}%")
        print(f"{pair} [{pname}, t={tr:.2f}/{tw:.2f}/{ts:.2f}]  " + "  |  ".join(out))
