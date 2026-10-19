# miss_set_analysis.py -- STAGE 2 (pre-registered 13 Oct, evening): Lesson-1 analysis.
# THE DATA: the four-cell reward table on validation, restricted to the MISS SET --
# instances where the stabilized pair's natural decision (score >= 0.5 at each
# stage) is not the reward-best path.
# THE QUESTION: what does each miss need -- a different retrieval choice, a
# different answering choice, or both at once (coupled)? And FIRST, per your
# coverage-first rule: did the weak retriever's selections even contain the gold?
# PREDICTIONS, stated before first run:
#   P1 coupled bucket carries UNDER 15% of the missed reward.
#   P2 misses needing a retrieval change are mostly NOT-covered (weak evidence
#      truly insufficient) -- the exact cases a gold-coverage signal would flag.
#   P3 answering-only misses are mostly covered (evidence was there; the
#      answering side failed) -- those need an answering-stage signal instead.
# CPU-only, read-only. Certified ingredients by import.
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reward_train_ret as RT
import run_labels4 as R
import router_prompts as RP

REF_RET = "/mnt/hdd/jonathan/runs/reward_train_ret2/val_scores.jsonl"
REF_QA = "/mnt/hdd/jonathan/runs/reward_train_qa4/val_scores.jsonl"
LAMBDA, PATHS, C_SS = RT.LAMBDA, RT.PATH, RT.C_SS

data = json.load(open(RT.DATA))
qa_w = RT.load_store(os.path.join(RT.GRID, "qa_weak.jsonl"))
qa_s = RT.load_store(os.path.join(RT.GRID, "qa_strong.jsonl"))
retr_w = RT.load_store(os.path.join(RT.GRID, "retr_weak.jsonl"))
lab = [json.loads(l) for l in open(RT.LABELS)]
val_idx = sorted(r["idx"] for r in lab if r["split"] == "val")
retp = {r["idx"]: r["pW"] for r in map(json.loads, open(REF_RET))}
qap = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(REF_QA))}

def weak_sel_text(i):
    inst = data[i]; pool = R.trim_pool(inst)
    sel = R.parse_selection(retr_w[str(i)], len(pool))
    return RP.format_selections(sel, pool)

def gold_titles(inst):
    if "supporting_titles" in inst: return [str(t) for t in inst["supporting_titles"]]
    if "supporting_facts" in inst: return [str(t) for t, *_ in inst["supporting_facts"]]
    return None

N = RT.norm_   # certified normalizer (lowercase, no punct, no articles)
rows, no_titles = [], 0
for i in val_idx:
    inst = data[i]; golds = inst["answers"]
    Rr = {}
    for rc, sfx in (("W", "weak"), ("S", "strong")):
        for ac, store in (("W", qa_w), ("S", qa_s)):
            Rr[(rc, ac)] = RT.tokenF1(str(store[f"{i}|{sfx}"]), golds) - LAMBDA * PATHS[(rc, ac)] / C_SS
    rc = "W" if retp[i] >= 0.5 else "S"                 # the pair's natural decision
    ac = "W" if qap[(i, rc)] >= 0.5 else "S"
    best = max(Rr, key=Rr.get)
    sel_norm = N(weak_sel_text(i))
    ans_cov = any(N(str(g)) and N(str(g)) in sel_norm for g in golds)
    ts = gold_titles(inst)
    if ts is None: no_titles += 1; ttl_cov = None
    else: ttl_cov = all(N(t) in sel_norm for t in ts)
    rows.append({"idx": i, "R": Rr, "chosen": (rc, ac), "best": best,
                 "gap": Rr[best] - Rr[(rc, ac)], "ans_cov": ans_cov, "ttl_cov": ttl_cov})

miss = [r for r in rows if r["gap"] > 1e-12]
tot = sum(r["gap"] for r in miss)
print(f"val n={len(rows)}  miss set={len(miss)} ({100*len(miss)/len(rows):.1f}%)  "
      f"total missed reward={tot:.2f}  (mean {tot/len(rows):.4f}/instance)")
if no_titles: print(f"note: {no_titles} instances lack supporting-title fields")

def bucket(r):
    (rc, ac), (brc, bac), Rr = r["chosen"], r["best"], r["R"]
    if brc == rc: return "answering-only"
    if bac == ac: return "retrieval-only"
    # both differ: decomposable if either single change already improves
    if Rr[(brc, ac)] > Rr[(rc, ac)] + 1e-12 or Rr[(rc, bac)] > Rr[(rc, ac)] + 1e-12:
        return "both (decomposable)"
    return "both (COUPLED)"

print(f"\n{'bucket':22s} {'n':>5s} {'missed R':>9s} {'% of miss':>9s}   "
      f"{'ans-cov%':>8s} {'title-cov%':>10s}")
for b in ("retrieval-only", "answering-only", "both (decomposable)", "both (COUPLED)"):
    sub = [r for r in miss if bucket(r) == b]
    if not sub:
        print(f"{b:22s} {0:5d} {'-':>9s} {'-':>9s}"); continue
    m = sum(r["gap"] for r in sub)
    ac_ = 100 * sum(r["ans_cov"] for r in sub) / len(sub)
    tc_ = [r for r in sub if r["ttl_cov"] is not None]
    tcs = f"{100*sum(r['ttl_cov'] for r in tc_)/len(tc_):9.1f}%" if tc_ else "      n/a"
    print(f"{b:22s} {len(sub):5d} {m:9.2f} {100*m/tot:8.1f}%   {ac_:7.1f}% {tcs}")

print("\ncoverage-first view of the whole miss set:")
for tag, cond in (("weak selections COVER the gold answer", lambda r: r["ans_cov"]),
                  ("weak selections DO NOT cover it", lambda r: not r["ans_cov"])):
    sub = [r for r in miss if cond(r)]
    m = sum(r["gap"] for r in sub)
    rb = sum(1 for r in sub if bucket(r) == "retrieval-only")
    ab = sum(1 for r in sub if bucket(r) == "answering-only")
    print(f"  {tag}: n={len(sub)}  missed R={m:.2f} ({100*m/tot:.1f}%)  "
          f"[retrieval-only {rb}, answering-only {ab}]")
