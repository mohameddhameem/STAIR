#!/usr/bin/env python3
"""
budget_allocation.py -- ERP I component: allocation of a FIXED CUMULATIVE BUDGET over a bundle of queries.
DRAFT v1.0 17 Nov 2026 for Jonathan's line-by-line review. CPU only. Reads ONLY stored held-out outcomes and the routers'
stored per-instance scores (no model is run, no new test exposure: every file below was written by a declared test read).

QUESTION (Prof. Archan, item 2): given an expected bundle of N queries and a total budget B, how should the router's decisions
be allocated so that the bundle's spend stays within B -- as opposed to optimising a per-query cost term.

FORMULATION. maximise sum_i E[correct_i(a_i)]  s.t.  sum_i cost_i(a_i) <= B.  The Lagrangian relaxation gives the per-query rule
"act to maximise p_i(a) - lambda * c_i(a)"; for a two-action stage with a calibrated score p_i (= the router's P(weak is enough)),
that rule is a THRESHOLD: deliver weak iff p_i >= theta(lambda). Our routers already use such thresholds; R' = correctness -
0.10 * cost fixes lambda = 0.10 for every query. The budget component is: choose theta FOR THE BUNDLE so the cumulative spend
meets B -- offline when the bundle is known, online (pacing) when queries arrive one by one.

POLICIES (all use the same stored scores; they differ only in how theta is set)
  per-query   theta fixed in advance (the registered operating point), blind to B -> reports its REALISED spend and whether it
              violates B on each bundle
  bundle      theta chosen on the bundle itself: the largest escalation level whose spend <= B (sorts queries by score)
  pacing      online: queries in random order; before each query theta is re-set from the remaining budget per remaining
              query, using the cost-vs-theta curve measured on the VALIDATION split (never on the bundle's own outcomes)
  oracle_B    knapsack on the TRUE outcomes (greedy by accuracy gain per unit cost over the cheapest-correct path) -- upper bound
  random_B    escalate uniformly random queries to the strongest path until B is spent -- lower bound
  always_SS   reference (spend = 100 % by definition)

DATASETS (both from the lab's run folders)
  hotpotqa   rows = router_data_standard.jsonl (split == test, 1,481; em_standard = [WW, WS, SW, SS] lenient outcomes), prices
             PRICE = [80, 195, 894, 1009] units, C_SS = 929 (Table 5 convention, as every HotpotQA number); scores = test_scores/
             ret_<tag>.jsonl (idx, pW) and qa_<tag>.jsonl (idx, ret_choice, pW). A common theta drives all three gates
             (ret, qa|W, qa|S), i.e. the one-parameter family through the greedy point (0.5, 0.5, 0.5).
  medagentbench  rows = training1_table.json (split == test, 150; qw/qs outcomes, cw/cs costs; tid = task<k>_<j>); scores =
             training1_router/test_scores.jsonl (pDeliver); tau from tau_scores_v2.json. Deliver = cw, escalate = cw + cs
             (weak-first). HUMAN OPTION on the tau >= 0.8 tier: a SECOND cumulative resource -- at most K_H hand-offs per bundle;
             the allocator hands off the tier queries with the lowest pDeliver first; Q_H = 1 (person assumed correct, the
             Training 6 convention -- stated as an assumption); a hand-off costs cw (weak already run) and no strong call.

BUNDLES. n_bundles bootstrap samples (with replacement) of size N from the test rows; optional COMPOSITION SHIFT: a group key
(HotpotQA: question type if the row carries one; MedAgentBench: task type from tid) and a Dirichlet re-weighting of group shares,
so bundles differ from the validation mix the per-query theta was tuned on.

OUTPUTS <out>/budget_<dataset>.json (+ per-bundle CSV, + .xlsx when openpyxl is present): for each budget level and policy --
mean accuracy, mean realised spend (% of the bundle's always-SS cost), violation rate P(spend > B), mean overshoot, regret to
oracle_B; MedAgentBench adds tier failures with no human and hand-offs used.

USAGE (lab, python3 -I; seconds to a minute)
  python3 -I budget_allocation.py --dataset hotpotqa --tag grpo_fresh_guided --out <dir> [--N 200 --bundles 1000 --shift]
  python3 -I budget_allocation.py --dataset medagentbench --out <dir> [--N 100 --bundles 1000 --shift --kh 0 5 10 15]
"""
import argparse, bisect, collections, csv, json, math, os, random, statistics, sys

BUDGETS = [40, 50, 60, 70, 80, 90, 100, 110, 120]           # % of the bundle's always-strong cost
THETAS = [i / 100 for i in range(0, 101)]                      # the one-parameter family


# ----------------------------------------------------------------------------------------------------- loaders
def load_hotpotqa(a):
    B = a.root or "/mnt/hdd/jonathan/runs/config_b"
    PRICE, C_SS = [80, 195, 894, 1009], 929
    rows = [json.loads(l) for l in open(f"{B}/router_data_standard.jsonl") if l.strip()]
    def scores(fn):
        return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(f"{B}/test_scores/{fn}.jsonl"))}
    ret, qa = scores(f"ret_{a.tag}"), scores(f"qa_{a.tag}")
    def path(i, theta):                                      # the router's own decision structure, common theta
        if ret[(i, None)] >= theta: return 0 if qa[(i, "W")] >= theta else 1
        return 2 if qa[(i, "S")] >= theta else 3
    items = []
    for g in rows:
        if g["split"] != "test": continue
        i, em = g["idx"], g["em_standard"]
        grp = g.get("type") or g.get("qtype")
        items.append({"id": i, "group": grp or "all", "acc": [float(x) for x in em], "cost": PRICE,
                      "path_at": {t: path(i, t) for t in THETAS}, "strong": 3})
    val = [g for g in rows if g["split"] == "val"]
    val_items = [{"acc": [float(x) for x in g["em_standard"]], "cost": PRICE, "path_at": {t: path(g["idx"], t) for t in THETAS}}
                 for g in val if (g["idx"], None) in ret]
    return items, val_items, C_SS, {"B": B, "tag": a.tag, "n_test": len(items), "n_val_scored": len(val_items)}


def load_medagentbench(a):
    R = a.root or "/mnt/hdd/jonathan/runs/mab"
    table = json.load(open(f"{R}/training1_table.json"))
    taus = {s["tid"]: s["tau"] for s in json.load(open(f"{R}/tau_scores_v2.json"))["scores"]}
    sc = {r["tid"]: r["pDeliver"] for r in map(json.loads, open(f"{R}/training1_router/test_scores.jsonl"))}
    vsc = {}
    vpath = f"{R}/training1_router/val_scores.jsonl"
    if os.path.exists(vpath):
        vsc = {r["tid"]: r["pDeliver"] for r in map(json.loads, open(vpath))}
    def mk(t, p):
        # actions: 0 = deliver weak (cw), 1 = escalate (cw + cs), 2 = hand off (cw; Q_H = 1; tier only)
        acc = [float(t["qw"]), float(t["qs"]), 1.0]; cost = [t["cw"], t["cw"] + t["cs"], t["cw"]]
        return {"id": t["tid"], "group": t["tid"].split("_")[0], "acc": acc, "cost": cost, "tau": taus[t["tid"]],
                "p": p, "path_at": {th: (0 if p >= th else 1) for th in THETAS}, "strong": 1, "css": t["cs"] + t["cw"]}
    items = [mk(t, sc[t["tid"]]) for t in table if t["split"] == "test"]
    val_items = [mk(t, vsc[t["tid"]]) for t in table if t["split"] == "val" and t["tid"] in vsc]
    return items, val_items, None, {"root": R, "n_test": len(items), "n_val_scored": len(val_items), "tau_gate": 0.8}


# ----------------------------------------------------------------------------------------------------- helpers
def spend_acc(bundle, paths):
    c = sum(it["cost"][p] for it, p in zip(bundle, paths)); q = sum(it["acc"][p] for it, p in zip(bundle, paths))
    return c, q


def css_of(bundle):
    return sum(it["cost"][it["strong"]] for it in bundle)


def theta_curve(items):
    """spend fraction of always-SS at each theta (monotone: lower theta -> more escalation -> more spend)."""
    css = css_of(items); return {t: spend_acc(items, [it["path_at"][t] for it in items])[0] / css for t in THETAS}


def policy_per_query(bundle, theta0, **_):
    return [it["path_at"][theta0] for it in bundle]


def policy_bundle(bundle, Bfrac, **_):
    """largest escalation (smallest theta) whose spend <= B; falls back to theta = 1 (deliver everything) if even that exceeds B."""
    css = css_of(bundle); best = None
    for t in THETAS:                                         # ascending theta = descending spend
        paths = [it["path_at"][t] for it in bundle]
        if spend_acc(bundle, paths)[0] <= Bfrac * css: best = paths; break
    return best if best is not None else [it["path_at"][1.0] for it in bundle]


def policy_pacing(bundle, Bfrac, curve, rng, **_):
    """online: for each arriving query choose theta whose VALIDATION spend fraction <= remaining budget per remaining query."""
    order = list(range(len(bundle))); rng.shuffle(order)
    css_mean = css_of(bundle) / len(bundle); remaining = Bfrac * css_of(bundle); paths = [None] * len(bundle)
    ts = sorted(THETAS, key=lambda t: curve[t])              # ascending spend
    for k, j in enumerate(order):
        left = len(bundle) - k; target = remaining / left / css_mean
        t = ts[0]
        for cand in ts:
            if curve[cand] <= target: t = cand
            else: break
        p = bundle[j]["path_at"][t]; paths[j] = p; remaining -= bundle[j]["cost"][p]
    return paths


def policy_oracle(bundle, Bfrac, **_):
    """start from the cheapest path per query; buy upgrades that increase accuracy, best gain-per-unit-cost first, within B."""
    css = css_of(bundle); paths = [min(range(len(it["cost"])), key=lambda p: it["cost"][p]) for it in bundle]
    # exclude the hand-off action for the oracle unless allowed (handled by caller via 'allow_h'); here all actions in cost list
    budget = Bfrac * css - spend_acc(bundle, paths)[0]
    cands = []
    for j, it in enumerate(bundle):
        base = paths[j]
        for p in range(len(it["cost"])):
            dq, dc = it["acc"][p] - it["acc"][base], it["cost"][p] - it["cost"][base]
            if dq > 0 and dc > 0: cands.append((dq / dc, j, p, dq, dc))
            elif dq > 0 and dc <= 0: paths[j] = p; budget -= dc; base = p
    cands.sort(reverse=True); taken = set()
    for ratio, j, p, dq, dc in cands:
        if j in taken or dc > budget: continue
        paths[j] = p; budget -= dc; taken.add(j)
    return paths


def policy_random(bundle, Bfrac, rng, **_):
    css = css_of(bundle); paths = [0] * len(bundle); budget = Bfrac * css - spend_acc(bundle, paths)[0]
    for j in rng.sample(range(len(bundle)), len(bundle)):
        dc = bundle[j]["cost"][bundle[j]["strong"]] - bundle[j]["cost"][0]
        if dc <= budget: paths[j] = bundle[j]["strong"]; budget -= dc
    return paths


def apply_handoffs(bundle, paths, kh, by_oracle=False, tau_gate=0.8):
    """MedAgentBench only: up to kh hand-offs on the tau >= gate tier; allocator = lowest pDeliver first; oracle = both-fail first."""
    tier = [j for j, it in enumerate(bundle) if it["tau"] >= tau_gate]
    if by_oracle: tier.sort(key=lambda j: -(1 - max(bundle[j]["acc"][0], bundle[j]["acc"][1])))
    else:         tier.sort(key=lambda j: bundle[j]["p"])
    out = list(paths)
    for j in tier[:kh]: out[j] = 2
    return out


def mean_se(xs):
    m = sum(xs) / len(xs); return m, (statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)


# ----------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "medagentbench"])
    ap.add_argument("--tag", default="grpo_fresh_guided", help="hotpotqa score files: ret_<tag>/qa_<tag> (the F6g read)")
    ap.add_argument("--theta0", type=float, default=0.5, help="the registered per-query operating point (greedy = 0.5)")
    ap.add_argument("--root", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=None); ap.add_argument("--bundles", type=int, default=1000)
    ap.add_argument("--shift", action="store_true", help="composition-shifted bundles (Dirichlet over groups)")
    ap.add_argument("--kh", nargs="*", type=int, default=[0], help="medagentbench: hand-off caps per bundle")
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); rng = random.Random(a.seed)
    items, val_items, _, meta = (load_hotpotqa if a.dataset == "hotpotqa" else load_medagentbench)(a)
    N = a.N or (200 if a.dataset == "hotpotqa" else 100)
    curve = theta_curve(val_items if val_items else items)   # pacing uses VALIDATION; falls back to test only if no val scores (flagged)
    meta["pacing_curve_source"] = "validation" if val_items else "TEST (no val scores found -- flag in the write-up)"
    groups = collections.defaultdict(list)
    for it in items: groups[it["group"]].append(it)
    gnames = sorted(groups); meta["groups"] = {g: len(groups[g]) for g in gnames}
    print(f"{a.dataset}: test items {len(items)} | groups {meta['groups']} | N {N} | bundles {a.bundles} | shift {a.shift} | "
          f"pacing curve from {meta['pacing_curve_source']}", flush=True)

    def draw():
        if not a.shift or len(gnames) < 2: return [rng.choice(items) for _ in range(N)]
        w = [rng.gammavariate(1.0, 1.0) for _ in gnames]; s = sum(w); w = [x / s for x in w]   # Dirichlet(1,...,1)
        return [rng.choice(groups[rng.choices(gnames, w)[0]]) for _ in range(N)]

    khs = a.kh if a.dataset == "medagentbench" else [0]
    agg = collections.defaultdict(list); per_bundle = []
    for b in range(a.bundles):
        bundle = draw(); css = css_of(bundle); n = len(bundle)
        ss_acc = spend_acc(bundle, [it["strong"] for it in bundle])[1] / n
        for Bp in BUDGETS:
            Bf = Bp / 100
            pols = {"per-query": policy_per_query(bundle, a.theta0), "bundle": policy_bundle(bundle, Bf),
                    "pacing": policy_pacing(bundle, Bf, curve, rng), "oracle_B": policy_oracle(bundle, Bf),
                    "random_B": policy_random(bundle, Bf, rng)}
            for kh in khs:
                for name, paths in pols.items():
                    pp = paths
                    if a.dataset == "medagentbench" and kh:
                        pp = apply_handoffs(bundle, paths, kh, by_oracle=(name == "oracle_B"))
                    c, q = spend_acc(bundle, pp)
                    rec = {"bundle": b, "B": Bp, "kh": kh, "policy": name, "acc": 100 * q / n, "spend_pct": 100 * c / css,
                           "violation": int(c > Bf * css + 1e-9), "overshoot_pct": max(0.0, 100 * (c - Bf * css) / css),
                           "ss_acc": 100 * ss_acc}
                    if a.dataset == "medagentbench":
                        tier = [j for j, it in enumerate(bundle) if it["tau"] >= 0.8]
                        rec["tier_n"] = len(tier)
                        rec["tier_fail_no_human"] = sum(1 for j in tier if pp[j] != 2 and bundle[j]["acc"][pp[j]] == 0)
                        rec["handoffs"] = sum(1 for p in pp if p == 2)
                    per_bundle.append(rec); agg[(Bp, kh, name)].append(rec)
        if (b + 1) % 200 == 0: print(f"  bundles {b + 1}/{a.bundles}", flush=True)

    summary = {"version": "budget_allocation v1.0 2026-11-17", "dataset": a.dataset, "meta": meta, "N": N, "bundles": a.bundles,
               "shift": a.shift, "theta0": a.theta0, "budgets": BUDGETS, "results": {}}
    print(f"\n{'B%':>4s} {'kh':>3s} {'policy':10s} {'acc':>6s} {'±SE':>5s} {'spend%':>7s} {'viol%':>6s} {'over%':>6s} {'regret':>7s}"
          + ("  tierFailNoH  handoffs" if a.dataset == "medagentbench" else ""))
    for (Bp, kh, name), recs in sorted(agg.items(), key=lambda k: (k[0][0], k[0][1], ["per-query", "bundle", "pacing", "oracle_B", "random_B"].index(k[0][2]))):
        acc, se = mean_se([r["acc"] for r in recs]); sp = sum(r["spend_pct"] for r in recs) / len(recs)
        viol = 100 * sum(r["violation"] for r in recs) / len(recs); over = sum(r["overshoot_pct"] for r in recs) / len(recs)
        orc = sum(r["acc"] for r in agg[(Bp, kh, "oracle_B")]) / len(recs); regret = orc - acc
        row = {"acc": round(acc, 2), "se": round(se, 2), "spend_pct": round(sp, 1), "violation_pct": round(viol, 1),
               "overshoot_pct": round(over, 2), "regret_to_oracle": round(regret, 2)}
        line = f"{Bp:4d} {kh:3d} {name:10s} {acc:6.2f} {se:5.2f} {sp:7.1f} {viol:6.1f} {over:6.2f} {regret:7.2f}"
        if a.dataset == "medagentbench":
            tf = sum(r["tier_fail_no_human"] for r in recs) / len(recs); ho = sum(r["handoffs"] for r in recs) / len(recs)
            row.update({"tier_fail_no_human": round(tf, 2), "handoffs": round(ho, 2)}); line += f"  {tf:11.2f}  {ho:8.2f}"
        summary["results"][f"B{Bp}|kh{kh}|{name}"] = row; print(line)
    summary["always_SS_acc"] = round(sum(r["ss_acc"] for r in per_bundle) / len(per_bundle), 2)
    json.dump(summary, open(os.path.join(a.out, f"budget_{a.dataset}.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"budget_{a.dataset}_per_bundle.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_bundle[0])); w.writeheader(); w.writerows(per_bundle)
    try:
        import openpyxl
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "summary"
        ws.append(["B%", "kh", "policy"] + list(next(iter(summary["results"].values()))))
        for k, v in summary["results"].items():
            Bp, kh, name = k.split("|"); ws.append([int(Bp[1:]), int(kh[2:]), name] + list(v.values()))
        wb.save(os.path.join(a.out, f"budget_{a.dataset}.xlsx"))
    except ModuleNotFoundError:
        pass
    print("always-SS accuracy (bundle mean):", summary["always_SS_acc"], "| wrote", os.path.join(a.out, f"budget_{a.dataset}.json"))


if __name__ == "__main__":
    main()
