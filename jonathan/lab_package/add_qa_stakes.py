# add_qa_stakes.py -- enrich the answering-stage labels with REWARD STAKES.
# CPU-only. Reads qa_labels.jsonl + the grid outcomes + the measured costs,
# writes qa_labels_rw.jsonl: every original field kept, plus
#   em_weak_ans / em_strong_ans : did the weak / strong ANSWER succeed on this
#       retrieval branch (grid columns: WW/WS for branch W, SW/SS for branch S)
#   weight : |R'(choose S) - R'(choose W)|   (raw; the trainer normalizes)
#
# R'(choice) = em  -  max(0, C_path / B - 1)  -  GAMMA * C_path / C_SS
#   * first two terms = the registered reward R = Q - max(0, C/B - 1)
#     with B = ALPHA * C_SS  (ALPHA = 0.74, agreed 11 Oct; sweep 0.6-1.0 later)
#   * the GAMMA linear term exists for WEIGHTING ONLY: under budget the hinge
#     is 0, so quality ties would get weight 0 and the router would never
#     learn to prefer the cheap answer when both succeed/fail. GAMMA keeps a
#     small, cost-proportional pull on those ties. R itself (evaluation, GRPO)
#     keeps the hinge form unchanged.
import json, os

GRID = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
LABELS = "/mnt/hdd/jonathan/runs/qa_labels/qa_labels.jsonl"
COSTS = "/mnt/hdd/jonathan/runs/token_costs/token_costs.json"
OUT = "/mnt/hdd/jonathan/runs/qa_labels/qa_labels_rw.jsonl"

ALPHA, GAMMA = 0.74, 0.25

tc = json.load(open(COSTS))["gate_eval_constants"]
C_SS = tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"] + tc["COST_QA_S_GIVEN_S"]
B = ALPHA * C_SS
# path costs by (retrieval choice, answering choice)
PATH = {("W", "W"): tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"],
        ("W", "S"): tc["COST_RET_W"] + tc["COST_QA_W_GIVEN_W"] + tc["COST_QA_S_GIVEN_W"],
        ("S", "W"): tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"],
        ("S", "S"): tc["COST_RET_W"] + tc["COST_RET_S"] + tc["COST_QA_W_GIVEN_S"] + tc["COST_QA_S_GIVEN_S"]}

def Rprime(em, c):
    return em - max(0.0, c / B - 1.0) - GAMMA * c / C_SS

em = {}
for l in open(GRID):
    r = json.loads(l); em[r["idx"]] = r["em_lenient"]   # [WW, WS, SW, SS]

n = bad = 0
stats = {}
with open(OUT, "w") as f:
    for l in open(LABELS):
        r = json.loads(l)
        e = em[r["idx"]]
        rc = r["ret_choice"]
        ew, es = (e[0], e[1]) if rc == "W" else (e[2], e[3])
        # audit: the honest label must agree with the outcome bits
        want = "W" if ew else ("S" if es else "X")
        if r["qa_label"] != want: bad += 1
        w = abs(Rprime(es, PATH[(rc, "S")]) - Rprime(ew, PATH[(rc, "W")]))
        r.update(em_weak_ans=ew, em_strong_ans=es, weight=round(w, 6))
        key = (rc, "decisive" if ew != es else "tie")
        s = stats.setdefault(key, [0, 0.0]); s[0] += 1; s[1] += w
        f.write(json.dumps(r) + "\n"); n += 1

print(f"wrote {OUT}: {n} rows | ALPHA={ALPHA} B={B:.1f} GAMMA={GAMMA} C_SS={C_SS:.1f}")
print(f"audit (qa_label vs outcome bits): {bad} mismatches -- MUST be 0")
print(f"{'branch':>7s} {'kind':>9s} {'rows':>6s} {'mean raw weight':>16s}")
for (rc, kind), (cnt, tot) in sorted(stats.items()):
    print(f"{rc:>7s} {kind:>9s} {cnt:6d} {tot/cnt:16.4f}")
