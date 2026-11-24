#!/usr/bin/env python3
"""
main_paired_tests.py -- paired significance of each ROUTER against ALWAYS-STRONG on the same held-out instances (Table 1 / §5 protocol).
v1.0 24 Nov 2026, for Jonathan's line-by-line review. CPU, seconds. Reads ONLY stored held-out outcomes and the routers' stored
per-instance scores (the same files budget_allocation.py v1.8 reads; no model runs). Tests: exact two-sided McNemar on the discordant
pairs, 10,000-resample paired bootstrap 95 % CI of the accuracy difference (seed 20261124), plus the Wilson 95 % interval of each
accuracy on its own. Cost on the strong-only axis (HotpotQA 929 units; MedAgentBench cs per task), task-model compute only.

HotpotQA   per instance: path under the router's thresholds (same rule as budget_allocation._locked / path), correctness =
           em_standard[path] (lenient); always-Strong = em_standard[3]. --router NAME:TAG:T_RET:T_QAW:T_QAS, repeatable
           (e.g. champion:grpo_fresh_objective:0.80:0.05:0.05  or  champion_greedy:grpo_fresh_objective:0.5:0.5:0.5).
MedAgentBench  per task: deliver iff pDeliver >= theta (default 0.5): correct = qw, cost cw; else qs, cost cw + cs (or cw2 + cs for
           Training 3); always-Strong = qs. --router NAME:RUN[:THETA], repeatable (e.g. training1:training1_router:0.5).
USAGE  python3 -I main_paired_tests.py --dataset hotpotqa --router champion:grpo_fresh_objective:0.80:0.05:0.05 [--router ...] --out <dir>
       python3 -I main_paired_tests.py --dataset medagentbench --router training1:training1_router --router training3:training3_router --out <dir>
       --list  prints the test-score tags / run folders available, then exits.
"""
import argparse, glob, json, math, os, random

HOT_ROOT, MAB_ROOT = "/mnt/hdd/jonathan/runs/config_b", "/mnt/hdd/jonathan/runs/mab"
PRICE = [80, 195, 894, 1009]; CSS_HOT = 929.0


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def paired(a, b, ids, n_boot=10000, seed=20261124):
    d = [a[i] - b[i] for i in ids]; n = len(d); win = sum(1 for x in d if x > 0); loss = sum(1 for x in d if x < 0); m = win + loss; k = min(win, loss)
    p = min(1.0, sum(math.comb(m, j) for j in range(0, k + 1)) * 2 / 2 ** m) if m else 1.0
    rng = random.Random(seed); boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(n_boot))
    return {"delta": round(sum(d) / n * 100, 2), "wins": win, "losses": loss, "ties": n - m, "mcnemar_p": round(p, 4),
            "ci95": [round(boots[int(0.025 * n_boot)], 2), round(boots[int(0.975 * n_boot) - 1], 2)]}


def hotpotqa(a):
    rows = {g["idx"]: g for g in map(json.loads, (l for l in open(f"{HOT_ROOT}/router_data_standard.jsonl") if l.strip())) if g["split"] == "test"}
    assert len(rows) == 1481, len(rows)
    ids = sorted(rows); ss = {i: int(rows[i]["em_standard"][3]) for i in ids}
    out = {"dataset": "hotpotqa", "n": len(ids), "always_strong": {"acc": round(100 * sum(ss.values()) / len(ids), 2), "wilson95": wilson(sum(ss.values()), len(ids))}, "routers": {}}
    print(f"\nHotpotQA held-out test n={len(ids)}  always-Strong {out['always_strong']['acc']}  Wilson {out['always_strong']['wilson95'][0]:.1f}-{out['always_strong']['wilson95'][1]:.1f}")
    for spec in a.router:
        name, tag, tr, tw, ts = spec.split(":"); tr, tw, ts = float(tr), float(tw), float(ts)
        sc = lambda path: {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(path))}
        ret, qa = sc(f"{HOT_ROOT}/test_scores/ret_{tag}.jsonl"), sc(f"{HOT_ROOT}/test_scores/qa_{tag}.jsonl")
        def path(i):
            if ret[(i, None)] >= tr: return 0 if qa[(i, "W")] >= tw else 1
            return 2 if qa[(i, "S")] >= ts else 3
        corr = {i: int(rows[i]["em_standard"][path(i)]) for i in ids}; cost = sum(PRICE[path(i)] for i in ids) / len(ids) / CSS_HOT * 100
        acc = 100 * sum(corr.values()) / len(ids); t = paired(corr, ss, ids)
        out["routers"][name] = {"tag": tag, "thresholds": [tr, tw, ts], "acc": round(acc, 2), "cost_pct_strong_only": round(cost, 1), "wilson95": wilson(sum(corr.values()), len(ids)), "vs_always_strong": t}
        print(f"  {name:18s} {acc:6.2f} @ {cost:5.1f} %   vs always-Strong Δ {t['delta']:+.2f}  wins {t['wins']} / losses {t['losses']} / ties {t['ties']}   McNemar p = {t['mcnemar_p']}   95% CI [{t['ci95'][0]:+.2f}, {t['ci95'][1]:+.2f}]")
    return out


def medagentbench(a):
    out = {"dataset": "medagentbench", "routers": {}}
    for spec in a.router:
        parts = spec.split(":"); name, run = parts[0], parts[1]; theta = float(parts[2]) if len(parts) > 2 else 0.5
        table_name = "training3_table.json" if run.startswith("training3") else "training1_table.json"
        table = {t["tid"]: t for t in json.load(open(f"{MAB_ROOT}/{table_name}")) if t["split"] == "test"}
        sc = {r["tid"]: r["pDeliver"] for r in map(json.loads, open(f"{MAB_ROOT}/{run}/test_scores.jsonl"))}
        ids = sorted(t for t in table if t in sc); assert len(ids) == 150, len(ids)
        ss = {t: int(table[t]["qs"]) for t in ids}
        corr, cost, css = {}, 0.0, 0.0
        for t in ids:
            row = table[t]; deliver = sc[t] >= theta
            corr[t] = int(row["qw"]) if deliver else int(row["qs"])
            cost += row["cw"] if deliver else ((row["cw2"] + row["cs"]) if "cw2" in row else (row["cw"] + row["cs"])); css += row["cs"]
        acc = 100 * sum(corr.values()) / len(ids); tst = paired(corr, ss, ids)
        out.setdefault("n", len(ids)); out.setdefault("always_strong", {"acc": round(100 * sum(ss.values()) / len(ids), 2), "wilson95": wilson(sum(ss.values()), len(ids))})
        out["routers"][name] = {"run": run, "table": table_name, "theta": theta, "acc": round(acc, 2), "cost_pct_strong_only": round(100 * cost / css, 1), "wilson95": wilson(sum(corr.values()), len(ids)), "vs_always_strong": tst}
        if len(out["routers"]) == 1:
            print(f"\nMedAgentBench held-out test n={len(ids)}  always-Strong {out['always_strong']['acc']}  Wilson {out['always_strong']['wilson95'][0]:.1f}-{out['always_strong']['wilson95'][1]:.1f}")
        print(f"  {name:18s} {acc:6.2f} @ {100 * cost / css:5.1f} %   vs always-Strong Δ {tst['delta']:+.2f}  wins {tst['wins']} / losses {tst['losses']} / ties {tst['ties']}   McNemar p = {tst['mcnemar_p']}   95% CI [{tst['ci95'][0]:+.2f}, {tst['ci95'][1]:+.2f}]")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "medagentbench"]); ap.add_argument("--router", action="append", default=[])
    ap.add_argument("--out", default=None); ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if a.list:
        if a.dataset == "hotpotqa":
            print("test-score tags:", sorted({os.path.basename(p)[4:-6] for p in glob.glob(f"{HOT_ROOT}/test_scores/ret_*.jsonl")}))
        else:
            print("run folders with test_scores.jsonl:", sorted(os.path.basename(os.path.dirname(p)) for p in glob.glob(f"{MAB_ROOT}/*/test_scores.jsonl")))
        return
    assert a.router and a.out, "give --router specs and --out"
    res = hotpotqa(a) if a.dataset == "hotpotqa" else medagentbench(a)
    os.makedirs(a.out, exist_ok=True); json.dump(res, open(os.path.join(a.out, f"paired_{a.dataset}.json"), "w"), indent=1)
    print("wrote", os.path.join(a.out, f"paired_{a.dataset}.json"))


if __name__ == "__main__":
    main()
