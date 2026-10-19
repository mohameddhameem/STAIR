# coverage_policy_test.py -- pre-registered 13 Oct (evening): the CEILING of the
# coverage signal. Jonathan's rule, applied with gold knowledge (training-time
# oracle): weak selections cover the gold passages -> stay weak; else if strong's
# cover them -> escalate; neither -> weak (spend nothing). Measures what a router
# that learned this rule PERFECTLY would earn. CPU-only, read-only.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_train_ret as RT
import run_labels4 as R
import router_prompts as RP

REF_RET = "/mnt/hdd/jonathan/runs/reward_train_ret2/val_scores.jsonl"
REF_QA = "/mnt/hdd/jonathan/runs/reward_train_qa4/val_scores.jsonl"
STD = os.path.join(RT.GRID, "router_data_standard.jsonl")
LAMBDA, PATHS, C_SS = RT.LAMBDA, RT.PATH, RT.C_SS
CELLI = {("W","W"):0, ("W","S"):1, ("S","W"):2, ("S","S"):3}

data = json.load(open(RT.DATA))
qa_w = RT.load_store(os.path.join(RT.GRID, "qa_weak.jsonl"))
qa_s = RT.load_store(os.path.join(RT.GRID, "qa_strong.jsonl"))
retr_w = RT.load_store(os.path.join(RT.GRID, "retr_weak.jsonl"))
rs_path = os.path.join(RT.GRID, "retr_strong.jsonl")
if not os.path.exists(rs_path):
    sys.exit("retr_strong.jsonl not found in GRID -- run: ls " + RT.GRID)
retr_s = RT.load_store(rs_path)
em_std = {g["idx"]: g["em_standard"] for g in map(json.loads, open(STD)) if g["split"] == "val"}
retp = {r["idx"]: r["pW"] for r in map(json.loads, open(REF_RET))}
qap = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(REF_QA))}
lab = [json.loads(l) for l in open(RT.LABELS)]
val_idx = sorted(r["idx"] for r in lab if r["split"] == "val")
N = RT.norm_

def sel_text(i, store):
    inst = data[i]; pool = R.trim_pool(inst)
    sel = R.parse_selection(store[str(i)], len(pool))
    return RP.format_selections(sel, pool)

def gold_titles(inst):
    if "supporting_titles" in inst: return [str(t) for t in inst["supporting_titles"]]
    if "supporting_facts" in inst: return [str(t) for t, *_ in inst["supporting_facts"]]
    return None

rows = []
for i in val_idx:
    inst = data[i]; golds = inst["answers"]
    Rr = {}
    for rc, sfx in (("W","weak"), ("S","strong")):
        for ac, store in (("W", qa_w), ("S", qa_s)):
            Rr[(rc,ac)] = RT.tokenF1(str(store[f"{i}|{sfx}"]), golds) - LAMBDA*PATHS[(rc,ac)]/C_SS
    ts = gold_titles(inst) or []
    covW = bool(ts) and all(N(t) in N(sel_text(i, retr_w)) for t in ts)
    covS = bool(ts) and all(N(t) in N(sel_text(i, retr_s)) for t in ts)
    rows.append({"idx": i, "R": Rr, "covW": covW, "covS": covS, "em": em_std[i]})

n = len(rows)
cW = sum(r["covW"] for r in rows); cSnW = sum((not r["covW"]) and r["covS"] for r in rows)
neither = sum((not r["covW"]) and (not r["covS"]) for r in rows)
print(f"val n={n} | weak covers {100*cW/n:.1f}% | strong rescues {100*cSnW/n:.1f}% | neither {100*neither/n:.1f}%")

def policy_cov(r):   return "W" if r["covW"] else ("S" if r["covS"] else "W")
def ans_greedy(r, rc): return "W" if r["R"][(rc,"W")] >= r["R"][(rc,"S")] else "S"
def ans_router(r, rc): return "W" if qap[(r["idx"], rc)] >= 0.5 else "S"
def ret_router(r):     return "W" if retp[r["idx"]] >= 0.5 else "S"

def report(name, pick):
    rew = acc = cost = esc = 0
    for r in rows:
        rc, ac = pick(r)
        rew += r["R"][(rc,ac)]; acc += r["em"][CELLI[(rc,ac)]]
        cost += PATHS[(rc,ac)]; esc += (rc == "S")
    print(f"{name:46s} reward {rew/n:.4f}  acc {100*acc/n:.2f}  "
          f"cost {100*(cost/n)/C_SS:5.1f}%  strong-retrieval {100*esc/n:4.1f}%")

print(f"{'policy':46s} {'mean R':>13s} {'lenient':>5s} {'tok-cost':>9s}")
report("always strong-strong",            lambda r: ("S","S"))
report("stabilized pair (natural 0.5)",   lambda r: (ret_router(r), ans_router(r, ret_router(r))))
report("COVERAGE rule + current QA router", lambda r: (policy_cov(r), ans_router(r, policy_cov(r))))
report("COVERAGE rule + reward-greedy QA", lambda r: (policy_cov(r), ans_greedy(r, policy_cov(r))))
report("reward-best path (ceiling)",      lambda r: max(r["R"], key=r["R"].get))
