# configb_ablation_readout.py -- composed accuracy / cost / APGR per router size.
# CPU-only, read-only, VALIDATION split only. Pairs each retrieval-router
# size's val scores (ablation_eval/val_scores_<size>.jsonl) with the FIXED
# label-trained 1.5B answering router (sft_router5_qa/val_scores.jsonl) and
# sweeps the same 21x21x21 threshold grid as configb_pair_sweep.py.
import json, os

B = "/mnt/hdd/jonathan/runs/config_b"
GRID  = f"{B}/router_data_standard.jsonl"
PRICE = [80, 195, 894, 1009]   # WW, WS, SW, SS -- weak-first: count everything that ran
C_SS  = 929                    # budget = baseline strong-strong without the router
# Router overhead in normalized units per decision, parameter-scaled from the
# Table 5 anchor (7B retrieval pass = 77 units):
RET_OVH = {"0.5B": 5.5, "1.5B": 16.5, "3B": 33.0, "7B": 77.0}
QA_OVH  = 16.5                 # answering router is the 1.5B in every pair

SIZES = ["0.5B", "1.5B", "3B", "7B"]
QA_SCORES = f"{B}/sft_router5_qa/val_scores.jsonl"   # SFT (label) answering router

val = [g for g in map(json.loads, open(GRID)) if g["split"] == "val"]
n = len(val)
ss_acc = sum(g["em_standard"][3] for g in val) / n
ww_acc = sum(g["em_standard"][0] for g in val) / n
T = [i/20 for i in range(21)]
print(f"validation n={n} | always-SS {100*ss_acc:.2f} | always-WW {100*ww_acc:.2f}")

qa = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(QA_SCORES))}

for name in SIZES:
    rf = f"{B}/ablation_eval/val_scores_{name}.jsonl"
    if not os.path.exists(rf):
        print(f"\n{name}: no scores yet ({rf})"); continue
    recs = [json.loads(l) for l in open(rf)]
    ret = {r["idx"]: r["pW"] for r in recs}
    ms  = sorted(r["ms"] for r in recs[10:])           # skip warm-up rows
    rows = [(ret[g["idx"]], qa[(g["idx"],"W")], qa[(g["idx"],"S")], g["em_standard"]) for g in val]
    res = []
    for tr in T:
        for tw in T:
            for ts in T:
                acc = cost = 0
                for pr, pqw, pqs, em in rows:
                    if pr >= tr:  c = 0 if pqw >= tw else 1
                    else:         c = 2 if pqs >= ts else 3
                    acc += em[c]; cost += PRICE[c]
                res.append((acc/n, cost/n/C_SS, tr, tw, ts))
    best = max(res, key=lambda r: (r[0], -r[1]))       # peak accuracy, ties -> cheapest
    ovh  = (RET_OVH[name] + QA_OVH) / C_SS             # router units per instance, share of SS
    apgr = (best[0] - ww_acc) / (ss_acc - ww_acc)
    print(f"\n{name}:")
    print(f"  peak accuracy : {100*best[0]:.2f} at {100*best[1]:.1f}% of SS cost"
          f"  (+router overhead: {100*(best[1]+ovh):.1f}%)  thresholds {best[2]:.2f}/{best[3]:.2f}/{best[4]:.2f}")
    print(f"  APGR {100*apgr:.1f}% | ms/decision median {ms[len(ms)//2]:.1f}")
    beats = [r for r in res if r[0] >= ss_acc]
    if beats:
        cheap = min(beats, key=lambda r: r[1])
        print(f"  cheapest >= SS: {100*cheap[0]:.2f} at {100*cheap[1]:.1f}%  (+overhead {100*(cheap[1]+ovh):.1f}%)")
    else:
        print("  no swept point reaches always-SS accuracy")
