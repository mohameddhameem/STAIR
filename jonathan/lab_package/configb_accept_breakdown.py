# configb_accept_breakdown.py -- instance-level evidence for the identical-peak
# question. Read-only, CPU-only, runs in seconds.
# For the 1.5B answering router's peak point (thresholds 0.80/0.90/1.00):
# classify every ACCEPTED weak answer by (weak correct?, strong correct?).
# Accuracy difference vs escalating those same instances = n2 - n3.
# For 0.5B/3B/7B at their peak (tw=1.00): count how many instances their
# decisions actually touch.
import json

B = "/mnt/hdd/jonathan/runs/config_b"
val = [g for g in map(json.loads, open(f"{B}/router_data_standard.jsonl")) if g["split"] == "val"]
ret = {r["idx"]: r["pW"] for r in map(json.loads, open(f"{B}/ablation_eval/val_scores_1.5B.jsonl"))}

def qa_scores(name):
    return {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(f"{B}/ablation_eval/val_scores_qa_{name}.jsonl"))}

# ---- 1.5B answering router at its peak: tr=0.80, tw=0.90 -------------------
qa = qa_scores("1.5B")
accepted = [g for g in val if ret[g["idx"]] >= 0.80 and qa[(g["idx"], "W")] >= 0.90]
n1 = sum(1 for g in accepted if g["em_standard"][0] == 1 and g["em_standard"][1] == 1)
n2 = sum(1 for g in accepted if g["em_standard"][0] == 1 and g["em_standard"][1] == 0)
n3 = sum(1 for g in accepted if g["em_standard"][0] == 0 and g["em_standard"][1] == 1)
n4 = sum(1 for g in accepted if g["em_standard"][0] == 0 and g["em_standard"][1] == 0)
print("1.5B answering router, peak point (tr=0.80, tw=0.90):")
print(f"  accepted weak answers: {len(accepted)} (expect 213)")
print(f"  n1 both correct (harmless, saves 115 units each): {n1}")
print(f"  n2 weak right / strong wrong (accuracy GAINED)  : {n2}")
print(f"  n3 weak wrong / strong right (accuracy LOST)    : {n3}")
print(f"  n4 both wrong (harmless)                        : {n4}")
print(f"  net accuracy effect of accepting (n2 - n3)      : {n2 - n3}")

# ---- the other three at their peak (tw = 1.00): what do they touch? --------
print("\nother answering routers at their peak (accept only if pW >= 1.00):")
for name in ["0.5B", "3B", "7B"]:
    q = qa_scores(name)
    accW = [g for g in val if ret[g["idx"]] >= 0.95 and q[(g["idx"], "W")] >= 1.0]
    accS = [g for g in val if ret[g["idx"]] <  0.95 and q[(g["idx"], "S")] >= 1.0]
    note = ""
    if accS:
        d2 = sum(g["em_standard"][2] - g["em_standard"][3] for g in accS)
        note = f" | SW accuracy effect vs SS on those: {d2:+d}"
    print(f"  {name}: weak-branch acceptances {len(accW)}, strong-branch acceptances {len(accS)}{note}")
