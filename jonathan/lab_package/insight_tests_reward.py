# insight_tests_reward.py -- pre-registered 13 Oct (evening): do the three SFT
# insights hold at REWARD level? Validation split, CPU-only, read-only.
# PREDICTIONS, stated before first run:
#   T1 holds -- reward-best path differs from SS on MORE than half of instances.
#   T2 decisive -- router AUC vs the REWARD-best choice expected BELOW its ~0.75
#      AUC vs the labels (information limit; would explain Run J's negative).
#   T3 holds -- double-digit %% on both branch-conditionality and price-flip.
# Certified ingredients by import: tokenF1, prices, stores, labels (reward_train_ret),
# answering-stage rows (reward_train_qa). Nothing re-implemented.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_train_ret as RT
import reward_train_qa as QT

REF_RET = "/mnt/hdd/jonathan/runs/reward_train_ret2/val_scores.jsonl"
REF_QA = "/mnt/hdd/jonathan/runs/reward_train_qa4/val_scores.jsonl"
LAMBDA, PATHS, C_SS = RT.LAMBDA, RT.PATH, RT.C_SS
EPS = 1e-12

def auc(ps, ys):
    pairs = sorted(zip(ps, ys)); pos = [i+1 for i,(p,y) in enumerate(pairs) if y==1]
    nP, nN = len(pos), len(pairs)-len(pos)
    return (sum(pos)-nP*(nP+1)/2)/(nP*nN) if nP and nN else float("nan")

data = json.load(open(RT.DATA))
qa_w = RT.load_store(os.path.join(RT.GRID, "qa_weak.jsonl"))
qa_s = RT.load_store(os.path.join(RT.GRID, "qa_strong.jsonl"))
lab = [json.loads(l) for l in open(RT.LABELS)]
val_ret = {r["idx"]: r["ret_target"] for r in lab if r["split"] == "val"}

qa_rows = QT.build_rows(data)
probe = qa_rows["val"][0]
tkey = "target" if "target" in probe else ("qa_target" if "qa_target" in probe else None)
print("qa row keys:", sorted(probe), "| using target key:", tkey)
qa_t = {}
if tkey:
    for r in qa_rows["val"]:
        qa_t[(r["idx"], r["ret_choice"])] = r[tkey]

rows = []
for i, rt in sorted(val_ret.items()):
    golds = data[i]["answers"]
    F, R = {}, {}
    for rc, sfx in (("W","weak"), ("S","strong")):
        for ac, store in (("W", qa_w), ("S", qa_s)):
            f = RT.tokenF1(str(store[f"{i}|{sfx}"]), golds)
            F[(rc,ac)] = f
            R[(rc,ac)] = f - LAMBDA*PATHS[(rc,ac)]/C_SS
    rows.append({"idx": i, "R": R, "F": F, "ret_target": rt})
print(f"val rows: {len(rows)}")

retp = {r["idx"]: r["pW"] for r in map(json.loads, open(REF_RET))}
qap = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(REF_QA))}

# ---- TEST 1: the band exists in reward currency ----
heads = [max(r["R"].values()) - r["R"][("S","S")] for r in rows]
nonSS = sum(1 for r in rows if max(r["R"].values()) > r["R"][("S","S")] + EPS)
hs = sorted(heads)
print("\nT1 | band in reward currency")
print(f"  mean best-path {sum(max(r['R'].values()) for r in rows)/len(rows):.4f}  "
      f"mean always-SS {sum(r['R'][('S','S')] for r in rows)/len(rows):.4f}")
print(f"  reward-best path != SS on {100*nonSS/len(rows):.1f}% of instances")
print(f"  headroom per instance: median {hs[len(hs)//2]:.4f}  p90 {hs[int(0.9*len(hs))]:.4f}")

# ---- TEST 2: can the current views SEE the reward-best choice? ----
def best_rc(r): return max(r["R"], key=r["R"].get)[0]
y_rew = [1 if best_rc(r)=="W" else 0 for r in rows]
y_lab = [1 if r["ret_target"]=="W" else 0 for r in rows]
ps = [retp[r["idx"]] for r in rows]
print("\nT2 | ranking quality of CURRENT views, label target vs reward target")
print(f"  RET router: AUC vs labels {auc(ps,y_lab):.4f}   AUC vs reward-best {auc(ps,y_rew):.4f}")
live = [k for k,r in enumerate(rows) if max(r["F"].values()) > 0]
print(f"  same, rows with any quality signal (n={len(live)}): "
      f"labels {auc([ps[k] for k in live],[y_lab[k] for k in live]):.4f}   "
      f"reward-best {auc([ps[k] for k in live],[y_rew[k] for k in live]):.4f}")
for b in ("W","S"):
    pq = [qap[(r["idx"],b)] for r in rows]
    yq = [1 if r["R"][(b,"W")] >= r["R"][(b,"S")] else 0 for r in rows]
    line = f"  QA router | branch {b}: AUC vs reward-best {auc(pq,yq):.4f}"
    if tkey:
        yl = [1 if qa_t[(r["idx"],b)]=="W" else 0 for r in rows]
        line += f"   vs labels {auc(pq,yl):.4f}"
    print(line)

# ---- TEST 3: conditional structure does real work inside the reward ----
diff_ans = sum(1 for r in rows
               if (r["R"][("W","W")] >= r["R"][("W","S")]) != (r["R"][("S","W")] >= r["R"][("S","S")]))
RW_, RS_ = RT.tc["COST_RET_W"], RT.tc["COST_RET_S"]
QWS_, QSS_ = RT.tc["COST_QA_W_GIVEN_S"], RT.tc["COST_QA_S_GIVEN_S"]
FLAT = {("W","W"): RW_+QWS_, ("W","S"): RW_+QWS_+QSS_,
        ("S","W"): RW_+RS_+QWS_, ("S","S"): C_SS}   # QA priced branch-independent (cheap)
flips = 0
for r in rows:
    R2 = {c: r["F"][c] - LAMBDA*FLAT[c]/C_SS for c in r["R"]}
    if max(r["R"], key=r["R"].get) != max(R2, key=R2.get): flips += 1
print("\nT3 | conditional structure")
print(f"  reward-best ANSWERING choice differs between branches: {100*diff_ans/len(rows):.1f}%")
print(f"  best path FLIPS if QA cost made branch-independent: {100*flips/len(rows):.1f}%")
