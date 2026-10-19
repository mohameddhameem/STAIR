# final_signal_tests.py -- pre-registered 13 Oct (evening): the last two measurements
# before the findings summary.
# TEST A, the draft-answer signal check: the weak answerer's draft is already paid
#   for on every path (the cascade always runs the weak answerer first), so its
#   features are free, inference-legal input for the answering decision. Do they
#   carry ranking signal for the reward-best answering choice?
# TEST B, the unanswerable-questions weighting measurement: what share of the
#   training pull comes from questions NO path can answer (flat quality, pure
#   cost gradient, all pushing "choose weak")?
# PREDICTIONS, stated before first run:
#   A: at least one draft feature reaches AUC >= 0.58 pooled (modest but real --
#      enough for a trained router to combine; individual features understate a
#      learned combination).
#   B: flat-quality rows are ~30% of instances but carry 25-45% of the
#      retrieval-stage training pull -- large enough to justify the weighting rule.
# CPU-only, read-only. Certified ingredients by import.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_train_ret as RT
import run_labels4 as R
import router_prompts as RP

REF_QA = "/mnt/hdd/jonathan/runs/reward_train_qa4/val_scores.jsonl"
STD = os.path.join(RT.GRID, "router_data_standard.jsonl")
LAMBDA, PATHS, C_SS = RT.LAMBDA, RT.PATH, RT.C_SS
N = RT.norm_

def auc(ps, ys):
    # tie-corrected (midrank) AUC -- equal feature values share their average rank
    order = sorted(range(len(ps)), key=lambda k: ps[k])
    ranks = [0.0]*len(ps); i = 0
    while i < len(order):
        j = i
        while j+1 < len(order) and ps[order[j+1]] == ps[order[i]]: j += 1
        mid = (i + j)/2 + 1
        for k in range(i, j+1): ranks[order[k]] = mid
        i = j + 1
    pos = [ranks[k] for k, y in enumerate(ys) if y == 1]
    nP = len(pos); nN = len(ys) - nP
    return (sum(pos) - nP*(nP+1)/2)/(nP*nN) if nP and nN else float("nan")

data = json.load(open(RT.DATA))
qa_w = RT.load_store(os.path.join(RT.GRID, "qa_weak.jsonl"))
qa_s = RT.load_store(os.path.join(RT.GRID, "qa_strong.jsonl"))
retr_w = RT.load_store(os.path.join(RT.GRID, "retr_weak.jsonl"))
retr_s = RT.load_store(os.path.join(RT.GRID, "retr_strong.jsonl"))
em_std = {g["idx"]: g["em_standard"] for g in map(json.loads, open(STD)) if g["split"] == "val"}
qap = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(REF_QA))}
lab = [json.loads(l) for l in open(RT.LABELS)]
val_idx = sorted(r["idx"] for r in lab if r["split"] == "val")

NOANS = ("no answer", "noanswer", "not provided", "not mentioned", "unknown",
         "cannot", "can not", "no information", "does not", "doesnt")

def sel_text(i, store):
    inst = data[i]; pool = R.trim_pool(inst)
    sel = R.parse_selection(store[str(i)], len(pool))
    return RP.format_selections(sel, pool)

recs, flat_ret, live_ret, flat_qa, live_qa, nflat = [], 0.0, 0.0, 0.0, 0.0, 0
for i in val_idx:
    inst = data[i]; golds = inst["answers"]
    F, Rr = {}, {}
    for rc, sfx in (("W","weak"), ("S","strong")):
        for ac, store in (("W", qa_w), ("S", qa_s)):
            f = RT.tokenF1(str(store[f"{i}|{sfx}"]), golds)
            F[(rc,ac)] = f; Rr[(rc,ac)] = f - LAMBDA*PATHS[(rc,ac)]/C_SS
    flat = max(F.values()) == 0.0
    nflat += flat
    ret_stake = abs(max(Rr[("W","W")], Rr[("W","S")]) - max(Rr[("S","W")], Rr[("S","S")]))
    qa_stake = abs(Rr[("W","W")]-Rr[("W","S")]) + abs(Rr[("S","W")]-Rr[("S","S")])
    if flat: flat_ret += ret_stake; flat_qa += qa_stake
    else: live_ret += ret_stake; live_qa += qa_stake
    for b, sfx, rstore in (("W","weak",retr_w), ("S","strong",retr_s)):
        draft = str(qa_w[f"{i}|{sfx}"])
        dn = N(draft); toks = dn.split()
        ev = set(N(sel_text(i, rstore)).split())
        recs.append({
            "y": 1 if Rr[(b,"W")] >= Rr[(b,"S")] else 0,     # deliver-weak is reward-best
            "router": qap[(i,b)],
            "len": float(len(toks)),
            "noans": -float(any(p in dn for p in NOANS)),     # sign: no-answer => escalate
            "ground": (sum(1 for t in toks if t in ev)/len(toks)) if toks else 0.0,
            "draftF1_ref": RT.tokenF1(draft, golds),          # TRAINING-TIME reference only
        })

n = len(recs)
print(f"val rows (instance x branch): {n}")
print("\nTEST A | can the weak answerer's draft rank the reward-best answering choice?")
for name in ("router", "len", "noans", "ground", "draftF1_ref"):
    tag = "  (reference, not inference-legal)" if name == "draftF1_ref" else ""
    print(f"  AUC {name:12s} {auc([r[name] for r in recs], [r['y'] for r in recs]):.4f}{tag}")

nv = len(val_idx)
lenX = sum(1 for i in val_idx if max(em_std[i]) == 0)
print(f"\nTEST B | unanswerable questions' share of the training pull")
print(f"  flat-quality instances (all four F1 = 0): {nflat} = {100*nflat/nv:.1f}% of validation")
print(f"  (lenient-X for reference: {100*lenX/nv:.1f}%)")
print(f"  retrieval-stage pull from flat rows: {100*flat_ret/(flat_ret+live_ret):.1f}%")
print(f"  answering-stage pull from flat rows: {100*flat_qa/(flat_qa+live_qa):.1f}%")
