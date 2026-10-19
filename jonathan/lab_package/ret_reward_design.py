# ret_reward_design.py -- EXPERIMENT 2 DESIGN BY MEASUREMENT (CPU, read-only).
# Candidate reward for the RETRIEVAL router: downstream answer F1 minus the
# lambda-scaled path cost. A retrieval choice has no quality of its own here --
# it earns whatever the answering stage downstream earns. Two downstream
# assumptions bracket reality:
#   A1  frozen answering router decides (Wednesday's adapter, its actual pW)
#   A2  best answering action per branch (oracle upper bound)
import json, re, string
from collections import Counter

LAMBDA = 0.10
DATA  = "/mnt/hdd/jonathan/data/tts_test.json"
GRID  = "/mnt/hdd/jonathan/runs/full_run_p100"
LENI  = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
COSTS = "/mnt/hdd/jonathan/runs/token_costs/token_costs.json"
QA_SC = "/mnt/hdd/jonathan/runs/sft_router5_qa/val_scores.jsonl"
RET_SC= "/mnt/hdd/jonathan/runs/sft_router5_ret/val_scores.jsonl"

tc = json.load(open(COSTS))["gate_eval_constants"]
C_SS = tc["COST_RET_W"]+tc["COST_RET_S"]+tc["COST_QA_W_GIVEN_S"]+tc["COST_QA_S_GIVEN_S"]
PATH = {("W","W"): tc["COST_RET_W"]+tc["COST_QA_W_GIVEN_W"],
        ("W","S"): tc["COST_RET_W"]+tc["COST_QA_W_GIVEN_W"]+tc["COST_QA_S_GIVEN_W"],
        ("S","W"): tc["COST_RET_W"]+tc["COST_RET_S"]+tc["COST_QA_W_GIVEN_S"],
        ("S","S"): C_SS}

def base_(s): return ''.join(c for c in s.lower() if c not in set(string.punctuation))
def norm_(s): return ' '.join(re.sub(r'\b(a|an|the)\b',' ',base_(s)).split())
def tokenF1(pred, golds):
    pt = norm_(pred).split(); best = 0.0
    for g in golds:
        gt = norm_(g).split()
        if not pt or not gt: continue
        common = sum((Counter(pt)&Counter(gt)).values())
        if common:
            p,r = common/len(pt), common/len(gt)
            best = max(best, 2*p*r/(p+r))
    return best

def store(p):
    d = {}
    for l in open(p):
        r = json.loads(l); d[r["k"]] = r["v"]
    return d

data = json.load(open(DATA))
qa_w, qa_s = store(f"{GRID}/qa_weak.jsonl"), store(f"{GRID}/qa_strong.jsonl")
val_idx = [json.loads(l)["idx"] for l in open(LENI) if json.loads(l)["split"]=="val"]
qa_p  = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(QA_SC))}
ret_p = {r["idx"]: r["pW"] for r in map(json.loads, open(RET_SC))}

def auc_of(ps, ys):
    pairs = sorted(zip(ps, ys)); pos=[i+1 for i,(p,y) in enumerate(pairs) if y==1]
    nP,nN = len(pos), len(pairs)-len(pos)
    return (sum(pos)-nP*(nP+1)/2)/(nP*nN) if nP and nN else float("nan")

for name, downstream in (("A1 frozen answering router", "policy"),
                         ("A2 best answering action   ", "best")):
    Rb = {}
    for i in val_idx:
        per = {}
        for rc, sfx in (("W","weak"), ("S","strong")):
            golds = data[i]["answers"]
            rW = tokenF1(str(qa_w[f"{i}|{sfx}"]), golds) - LAMBDA*PATH[(rc,"W")]/C_SS
            rS = tokenF1(str(qa_s[f"{i}|{sfx}"]), golds) - LAMBDA*PATH[(rc,"S")]/C_SS
            if downstream == "policy":
                p = qa_p[(i, rc)]
                per[rc] = p*rW + (1-p)*rS
            else:
                per[rc] = max(rW, rS)
        Rb[i] = per
    dR    = [Rb[i]["W"] - Rb[i]["S"] for i in val_idx]
    n     = len(val_idx)
    prefW = sum(1 for d in dR if d > 0)
    tiny  = sum(1 for d in dR if abs(d) <= 0.001)
    graded= sum(1 for d in dR if 0.001 < abs(d) < 0.5)
    big   = sum(1 for d in dR if abs(d) >= 0.5)
    mW = sum(Rb[i]["W"] for i in val_idx)/n
    mS = sum(Rb[i]["S"] for i in val_idx)/n
    mO = sum(max(Rb[i].values()) for i in val_idx)/n
    # does Wednesday's retrieval router already rank by this reward?
    a = auc_of([ret_p[i] for i in val_idx], [1 if d > 0 else 0 for d in dR])
    print(f"--- {name} ---")
    print(f"  always weak retrieval : mean R = {mW:.4f}")
    print(f"  always strong retrieval: mean R = {mS:.4f}")
    print(f"  perfect per-row pick   : mean R = {mO:.4f}")
    print(f"  prefers weak retrieval on {100*prefW/n:.1f}% of rows | "
          f"near-tie {100*tiny/n:.1f}% | graded {100*graded/n:.1f}% | decisive {100*big/n:.1f}%")
    print(f"  current retrieval router's ranking vs this reward: AUC = {a:.4f}\n")

print("field check for the passage-overlap candidate (tomorrow):")
print("  instance keys:", sorted(data[val_idx[0]].keys()))
