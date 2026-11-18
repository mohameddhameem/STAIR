#!/usr/bin/env python3
"""
budget_allocation.py -- ERP I component: accuracy within a FIXED CUMULATIVE BUDGET over a bundle of queries.
v1.8 18 Nov 2026 (14:20; --theta-locked T_RET T_QAW T_QAS adds the policy row 'locked' = the router at its registered locked thresholds, hotpotqa only) / v1.7 (13:00; --budget strong-only|weak-first, Amendment D: the budget is the STRONG MODEL ALONE -- 929 per HotpotQA question, cs per MedAgentBench task -- so escalate-all, which pays the weak pass too, reads > 100 %; weak-first reproduces the earlier runs) / v1.6 (09:10; --samplek + --k-weak: the weak answerer called k times with a vote, paths WW/SW priced 77+3k / 891+3k (weak-first; the scorer's v1.0 wrote 811+3k, corrected 18 Nov 12:50); v1.5 22:25; --full = the whole held-out set as the one bundle, i.e. the dataset-level budget — Jonathan's reading of Prof. Archan's 'fixed budget'; v1.3 fixed the MedAgentBench budget denominator = always-strong cw + cs; v1.4 adds --data for HotpotQA question type under --shift) -- DRAFT for Jonathan's line-by-line review; CPU only; reads ONLY stored held-out outcomes and the routers'
stored per-instance scores (no model is run; every input file was written by a declared test read or by validation).

RESEARCH QUESTION (Jonathan, 17 Nov): how to improve accuracy within a fixed budget equal to the always-strong cost -- and must
the whole budget be spent to reach the highest accuracy, or is the peak reached below it?

FORMULATION. maximise sum_i E[correct_i(a_i)]  s.t.  sum_i cost_i(a_i) <= B  over a bundle of N queries. The Lagrangian relaxation
gives the per-query rule "maximise p_i(a) - lambda * c_i(a)"; for a two-action stage with a calibrated score p_i = P(weak is
enough) that is a THRESHOLD: deliver weak iff p_i >= theta. Higher theta = more escalation = more spend. Our routers already act
through such thresholds (R' fixes lambda = 0.10 for every query). The budget component sets theta FOR THE BUNDLE.

POLICIES (same stored scores; they differ only in how theta is set and what the remainder buys)
  spend-all      the largest theta whose bundle spend <= B                       (the naive reading of "use the budget")
  spend-to-peak  theta* = argmax accuracy on the VALIDATION curve (tie -> cheaper), capped so spend <= B; the remainder is banked;
                 on MedAgentBench the remainder's instances are NOT escalated -- instead up to K_H hand-offs go to the high-risk
                 tier's lowest-confidence queries (second resource; person assumed correct = Training 6 convention)
  per-query      theta0 fixed in advance (registered operating point 0.5), blind to B -> realised spend, violation of B
  pacing         online: queries arrive one by one; theta_t = min(theta*, largest theta whose VALIDATION spend <= remaining budget
                 per remaining query) -- spend-to-peak executed without seeing the bundle
  oracle_B       knapsack on TRUE outcomes within B (greedy by accuracy gain per unit cost) -- upper bound under the budget
  oracle         unconstrained per-instance best (the ceiling) ; always_SS = strong everywhere (100 % by definition)
  random_B       escalate uniformly random queries to the strongest path until B is spent -- lower bound

DATASETS (lab run folders)
  hotpotqa   rows: router_data_standard.jsonl (split test 1,481 / val; em_standard = [WW, WS, SW, SS] lenient); PRICE [80, 195,
             894, 1009], C_SS 929 (Table 5 convention); TEST scores: test_scores/ret_<tag>.jsonl, qa_<tag>.jsonl (idx, ret_choice,
             pW); VAL scores: <run>/ret_val_scores.jsonl, qa_val_scores.jsonl (the trainer's dumps). One theta drives all three
             gates (ret, qa|W, qa|S): the one-parameter family through the greedy point (0.5, 0.5, 0.5).
  medagentbench  rows: training1_table.json (split test 150 / val 60; qw/qs outcomes; cw/cs costs; tid task<k>_<j>); TEST
             scores: training1_router/test_scores.jsonl (pDeliver); VAL scores: training1_router/val_scores.jsonl; tau from
             tau_scores_v2.json; tier = tau >= 0.8. deliver = cw, escalate = cw + cs (weak-first), hand-off = cw.
  Validation files are REQUIRED (theta* and the pacing curve come from validation, never from the bundle's outcomes).

BUNDLES. --bundles bootstrap samples (with replacement) of size N from the test rows; --shift re-weights groups (HotpotQA question
type if the rows carry one; MedAgentBench task type from tid) by a Dirichlet(1,...,1) draw per bundle.

OUTPUTS <out>/budget_<dataset>.json, _per_bundle.csv, .xlsx (if openpyxl): per B and policy -- mean accuracy (+SE over bundles),
realised spend (% of the bundle's always-SS cost), P(spend > B), overshoot, regret to oracle_B; the accuracy-spend CURVE over theta
(bundle means) with s* = spend at the validation theta*; MedAgentBench: tier failures with no human, hand-offs used.

USAGE (lab, slr-env or plain python3 -I)
  python3 -I budget_allocation.py --dataset hotpotqa --tag grpo_fresh_guided --run grpo_ret_qa_a1_fresh_ep6_guided --out <dir>
  python3 -I budget_allocation.py --dataset medagentbench --out <dir> --kh 0 5 10 15
  optional: --N --bundles --shift --seed --theta0
"""
import argparse, collections, csv, hashlib, json, math, os, random, statistics


def sha(path):
    return hashlib.sha256(open(path, 'rb').read()).hexdigest()

BUDGETS = [40, 50, 60, 70, 80, 90, 100, 110, 120]           # % of the bundle's always-strong cost
THETAS = [i / 100 for i in range(0, 101)]                      # higher theta -> more escalation -> more spend


# ----------------------------------------------------------------------------------------------------- loaders
def _item(id_, group, acc, cost, path_at, strong, **extra):
    d = {"id": id_, "group": group or "all", "acc": acc, "cost": cost, "path_at": path_at, "strong": strong}; d.update(extra); return d


def load_hotpotqa(a):
    B = a.root or "/mnt/hdd/jonathan/runs/config_b"
    PRICE = [80, 195, 894, 1009]
    rows = [json.loads(l) for l in open(f"{B}/router_data_standard.jsonl") if l.strip()]
    def scores(path):
        return {(r["idx"], r.get("ret_choice")): r["pW"] for r in map(json.loads, open(path))}
    ret_t, qa_t = scores(f"{B}/test_scores/ret_{a.tag}.jsonl"), scores(f"{B}/test_scores/qa_{a.tag}.jsonl")
    ret_v, qa_v = scores(f"{B}/{a.run}/ret_val_scores.jsonl"), scores(f"{B}/{a.run}/qa_val_scores.jsonl")
    def path(ret, qa, i, theta):                            # the router's own decision structure, one theta for all gates
        if ret[(i, None)] >= theta: return 0 if qa[(i, "W")] >= theta else 1
        return 2 if qa[(i, "S")] >= theta else 3
    def _locked(ret, qa, i, t):                              # v1.8: three locked thresholds (grpo_pair_sweep.py lines 36-37 rule)
        tr, tw, ts = t
        if ret[(i, None)] >= tr: return 0 if qa[(i, "W")] >= tw else 1
        return 2 if qa[(i, "S")] >= ts else 3
    qtype_of = {}
    if a.data:                                               # v1.4: question type for --shift (rows carry none)
        data = json.load(open(a.data, encoding="utf-8"))
        try:
            import sys; sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
            from stair_common.qtype import question_type
        except Exception:
            question_type = None
        for i, rec in enumerate(data):
            t = rec.get("type")
            if not t and question_type and rec.get("question"): t = question_type("hotpotqa", rec["question"])
            qtype_of[i] = t or "unknown"
    def build(split, ret, qa):
        out = []
        for g in rows:
            if g["split"] != split or (g["idx"], None) not in ret: continue
            out.append(_item(g["idx"], qtype_of.get(g["idx"]) or g.get("type") or g.get("qtype"), [float(x) for x in g["em_standard"]], PRICE,
                             {t: path(ret, qa, g["idx"], t) for t in THETAS}, 3,
                             css=(PRICE[3] - 80) if a.budget == "strong-only" else PRICE[3],   # v1.7 Amendment D
                             locked=(None if a.theta_locked is None else _locked(ret, qa, g["idx"], a.theta_locked))))   # v1.8: strong-only budget = 929 = 814 + 115 (SS without the weak retrieval 77 and weak answer 3 that ran first) = the trainer's C_SS
        return out
    test, val = build("test", ret_t, qa_t), build("val", ret_v, qa_v)
    assert len(test) == 1481, len(test); assert val, "no validation rows scored -- val score files missing"
    if a.samplek and a.k_weak > 1:                           # v1.6: K > 1 for the weak answerer (vote@k outcomes replace the greedy WW / SW cells)
        votes = {}
        for line in open(a.samplek):
            if line.strip():
                r = json.loads(line)
                if r["k"] == a.k_weak: votes[(r["split"], r["i"], r["ctx"])] = (float(r["correct"]), r["price_units"])
        n_miss = 0
        for split, items in (("test", test), ("val", val)):
            for it in items:
                w, sv = votes.get((split, it["id"], "W")), votes.get((split, it["id"], "S"))
                if w is None or sv is None: n_miss += 1; continue
                it["acc"] = [w[0], it["acc"][1], sv[0], it["acc"][3]]
                it["cost"] = [w[1], PRICE[1], sv[1], PRICE[3]]
        assert n_miss == 0, f"{n_miss} questions without vote@{a.k_weak} outcomes in {a.samplek}"
    files = [f"{B}/router_data_standard.jsonl", f"{B}/test_scores/ret_{a.tag}.jsonl", f"{B}/test_scores/qa_{a.tag}.jsonl",
             f"{B}/{a.run}/ret_val_scores.jsonl", f"{B}/{a.run}/qa_val_scores.jsonl"] + ([a.data] if a.data else []) + ([a.samplek] if a.samplek else [])
    return test, val, {"root": B, "tag": a.tag, "run": a.run, "n_test": len(test), "n_val": len(val), "price": PRICE, "data": a.data, "budget": a.budget, "theta_locked": a.theta_locked, "css_per_q": (PRICE[3] - 80) if a.budget == "strong-only" else PRICE[3],
                       "type_source": ("data 'type' field / wording rule" if qtype_of else "none (rows carry no type)"),
                       "k_weak": a.k_weak, "samplek": a.samplek,
                       "inputs_sha256": {os.path.basename(f) if not f.startswith(B) else os.path.relpath(f, B): sha(f) for f in files}}


def load_medagentbench(a):
    R = a.root or "/mnt/hdd/jonathan/runs/mab"
    table_name = a.table or ("training3_table.json" if a.run.startswith("training3") else "training1_table.json")
    table = json.load(open(f"{R}/{table_name}"))
    taus = {s["tid"]: s["tau"] for s in json.load(open(f"{R}/tau_scores_v2.json"))["scores"]}
    def scores(path): return {r["tid"]: r["pDeliver"] for r in map(json.loads, open(path))}
    sc_t, sc_v = scores(f"{R}/{a.run}/test_scores.jsonl"), scores(f"{R}/{a.run}/val_scores.jsonl")
    def build(split, sc):
        out = []
        for t in table:
            if t["split"] != split or t["tid"] not in sc: continue
            p = sc[t["tid"]]
            # actions: 0 deliver weak (cw) | 1 escalate (cw + cs) | 2 hand-off (cw; Q_H = 1, tier only)
            esc = (t["cw2"] + t["cs"]) if "cw2" in t else (t["cw"] + t["cs"])      # Training 3: round-2 escalation = cw2 + cs
            out.append(_item(t["tid"], t["tid"].split("_")[0], [float(t["qw"]), float(t["qs"]), 1.0], [t["cw"], esc, t["cw"]],
                             {th: (0 if p >= th else 1) for th in THETAS}, 1, tau=taus[t["tid"]], p=p,
                             css=(t["cs"] if a.budget == "strong-only" else t["cw"] + t["cs"])))   # v1.7 Amendment D: strong-only budget = cs (strong agent alone); v1.3 weak-first = cw + cs
        return out
    test, val = build("test", sc_t), build("val", sc_v)
    assert len(test) == 150, len(test); assert val, "no validation rows scored -- val_scores.jsonl missing"
    files = [f"{R}/{table_name}", f"{R}/tau_scores_v2.json", f"{R}/{a.run}/test_scores.jsonl", f"{R}/{a.run}/val_scores.jsonl"]
    return test, val, {"root": R, "run": a.run, "table": table_name, "n_test": len(test), "n_val": len(val), "tau_gate": 0.8,
                       "escalation_cost": "cw2 + cs (round-2)" if any("cw2" in t for t in table) else "cw + cs",
                       "budget": a.budget, "budget_denominator": ("strong agent alone = cs per task (v1.7 Amendment D)" if a.budget == "strong-only" else "always-strong weak-first = cw + cs per instance (v1.3)") + "; the 'always_SS' policy row is escalate-all at the router's escalation price",
                       "inputs_sha256": {os.path.relpath(f, R): sha(f) for f in files}}


# ----------------------------------------------------------------------------------------------------- helpers
def spend_acc(bundle, paths):
    return (sum(it["cost"][p] for it, p in zip(bundle, paths)), sum(it["acc"][p] for it, p in zip(bundle, paths)))


def css_of(bundle): return sum(it.get("css", it["cost"][it["strong"]]) for it in bundle)   # v1.3: 'css' overrides when present (MedAgentBench)


def curve_of(items):
    """per theta: (spend fraction of always-SS, accuracy) -- used on VALIDATION to pick theta* and to pace."""
    css, n = css_of(items), len(items); out = {}
    for t in THETAS:
        c, q = spend_acc(items, [it["path_at"][t] for it in items]); out[t] = (c / css, q / n)
    return out


def theta_star(vcurve, n_val):
    """AMENDED 17 Nov 21:40: the CHEAPEST theta whose validation accuracy is within ONE validation instance (1/n_val) of the
    maximum -- the registration's 'tie -> cheaper' made operational (exact argmax picked a +0.3-pt point at +20 pts of cost)."""
    best = max(v[1] for v in vcurve.values()); tol = 1.0 / n_val + 1e-12
    return min((t for t, v in vcurve.items() if v[1] >= best - tol), key=lambda t: (vcurve[t][0], t))


def largest_theta_within(bundle, Bfrac, cap=1.0):
    css = css_of(bundle); chosen = THETAS[0]
    for t in THETAS:
        if t > cap: break
        if spend_acc(bundle, [it["path_at"][t] for it in bundle])[0] <= Bfrac * css: chosen = t
    return chosen


def pol_spend_all(bundle, Bf, **_):
    t = largest_theta_within(bundle, Bf); return [it["path_at"][t] for it in bundle]


def pol_spend_to_peak(bundle, Bf, tstar, **_):
    t = largest_theta_within(bundle, Bf, cap=tstar); return [it["path_at"][t] for it in bundle]
def pol_per_query(bundle, theta0, **_):  return [it["path_at"][theta0] for it in bundle]
def pol_locked(bundle, **_):               return [it["locked"] for it in bundle]                          # v1.8


def pol_pacing(bundle, Bf, vcurve, tstar, rng, **_):
    order = list(range(len(bundle))); rng.shuffle(order)
    css_mean = css_of(bundle) / len(bundle); remaining = Bf * css_of(bundle); paths = [None] * len(bundle)
    for k, j in enumerate(order):
        target = remaining / (len(bundle) - k) / css_mean; t = THETAS[0]
        for cand in THETAS:
            if cand > tstar: break
            if vcurve[cand][0] <= target: t = cand
        paths[j] = bundle[j]["path_at"][t]; remaining -= bundle[j]["cost"][paths[j]]
    return paths


def pol_oracle_B(bundle, Bf, allow_h=False, **_):
    """cheapest path per query, then buy the best accuracy-per-cost upgrades within B (greedy knapsack)."""
    acts = lambda it: range(len(it["cost"])) if allow_h else range(min(len(it["cost"]), 1 + it["strong"]))
    paths = [min(acts(it), key=lambda p: it["cost"][p]) for it in bundle]
    budget = Bf * css_of(bundle) - spend_acc(bundle, paths)[0]; cands = []
    for j, it in enumerate(bundle):
        base = paths[j]
        for p in acts(it):
            dq, dc = it["acc"][p] - it["acc"][base], it["cost"][p] - it["cost"][base]
            if dq > 0 and dc <= 0: paths[j] = p; budget -= dc; base = p
        for p in acts(it):
            dq, dc = it["acc"][p] - it["acc"][paths[j]], it["cost"][p] - it["cost"][paths[j]]
            if dq > 0 and dc > 0: cands.append((dq / dc, j, p, dc))
    cands.sort(reverse=True); done = set()
    for _, j, p, dc in cands:
        if j in done or dc > budget: continue
        paths[j] = p; budget -= dc; done.add(j)
    return paths


def pol_oracle(bundle, **_):             return [max(range(1 + it["strong"]), key=lambda p: (it["acc"][p], -it["cost"][p])) for it in bundle]
def pol_always_ss(bundle, **_):          return [it["strong"] for it in bundle]


def pol_random_B(bundle, Bf, rng, **_):
    paths = [0] * len(bundle); budget = Bf * css_of(bundle) - spend_acc(bundle, paths)[0]
    for j in rng.sample(range(len(bundle)), len(bundle)):
        dc = bundle[j]["cost"][bundle[j]["strong"]] - bundle[j]["cost"][0]
        if dc <= budget: paths[j] = bundle[j]["strong"]; budget -= dc
    return paths


def with_handoffs(bundle, paths, kh, by_oracle=False, gate=0.8):
    """MedAgentBench: up to kh hand-offs on the tier; allocator = lowest pDeliver first; oracle = both-fail first."""
    tier = [j for j, it in enumerate(bundle) if it["tau"] >= gate]
    tier.sort(key=(lambda j: -(1 - max(bundle[j]["acc"][0], bundle[j]["acc"][1]))) if by_oracle else (lambda j: bundle[j]["p"]))
    out = list(paths)
    for j in tier[:kh]: out[j] = 2
    return out


def mean_se(xs):
    m = sum(xs) / len(xs); return m, (statistics.pstdev(xs) / math.sqrt(len(xs)) if len(xs) > 1 else 0.0)


# ----------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["hotpotqa", "medagentbench"])
    ap.add_argument("--tag", default="grpo_fresh_guided", help="hotpotqa TEST score files ret_<tag>/qa_<tag> (the three-insights read)")
    ap.add_argument("--run", default=None, help="run folder holding the VAL score dumps (hotpotqa: grpo_ret_qa_a1_fresh_ep6_guided; mab: training1_router)")
    ap.add_argument("--samplek", default=None, help="hotpotqa: samplek_outcomes.jsonl from hotpotqa_samplek_score.py (vote@k outcomes per question/ctx)")
    ap.add_argument("--k-weak", type=int, default=1, help="hotpotqa: with --samplek, the weak answerer is called k times (vote); paths WW/SW take the vote@k outcome and price 77+3k / 891+3k")
    ap.add_argument("--data", default=None, help="hotpotqa: the store's data file (tts_test.json) indexed by idx, for question type under --shift")
    ap.add_argument("--table", default=None, help="medagentbench replay table (default: training1_table.json, or training3_table.json when --run starts with training3)")
    ap.add_argument("--theta0", type=float, default=0.5, help="the registered per-query operating point (greedy)")
    ap.add_argument("--root", default=None); ap.add_argument("--out", required=True)
    ap.add_argument("--N", type=int, default=None); ap.add_argument("--bundles", type=int, default=1000)
    ap.add_argument("--theta-locked", nargs=3, type=float, default=None, metavar=("T_RET","T_QAW","T_QAS"), help="v1.8: hotpotqa only -- add the policy row 'locked' = the router at its registered locked thresholds (e.g. 0.80 0.05 0.05, the label-free champion's cap point), priced on the same axis; no other change")
    ap.add_argument("--budget", default="strong-only", choices=["strong-only", "weak-first"], help="v1.7 Amendment D: strong-only (default) = the budget is the cost of the strong model ALONE (no weak pass): HotpotQA 929 = 814 + 115 per question, MedAgentBench cs per task; weak-first = the earlier convention (1,009 / cw + cs), kept to reproduce v1.3-v1.6 runs");
    ap.add_argument("--shift", action="store_true"); ap.add_argument("--kh", nargs="*", type=int, default=[0])
    ap.add_argument("--full", action="store_true", help="v1.5: ONE bundle = the whole held-out set (no resampling) -- the dataset-level budget; bootstrap runs give the SEs")
    ap.add_argument("--seed", type=int, default=20261117)
    a = ap.parse_args()
    a.run = a.run or ("grpo_ret_qa_a1_fresh_ep6_guided" if a.dataset == "hotpotqa" else "training1_router")
    os.makedirs(a.out, exist_ok=True); rng = random.Random(a.seed)
    test, val, meta = (load_hotpotqa if a.dataset == "hotpotqa" else load_medagentbench)(a)
    N = a.N or (200 if a.dataset == "hotpotqa" else 100)
    if a.full:
        N, a.bundles, a.shift = len(test), 1, False                      # v1.5: the whole held-out set, once
    vcurve = curve_of(val); tstar = theta_star(vcurve, len(val))
    def levels(items, key):
        vals = [it[key] for it in items] if key else [v for it in items for v in (it["path_at"][0.5],)]
        return len(set(round(v, 3) for v in vals))
    if a.dataset == "medagentbench":
        meta["score_levels"] = {"test_distinct_p": levels(test, "p"), "val_distinct_p": levels(val, "p"),
                                "test_saturated_pct": round(100 * sum(it["p"] >= 0.999 or it["p"] <= 0.001 for it in test) / len(test), 1)}
    meta.update({"theta_star_val": tstar, "val_spend_at_theta_star": round(100 * vcurve[tstar][0], 1),
                 "val_acc_at_theta_star": round(100 * vcurve[tstar][1], 2)})
    groups = collections.defaultdict(list)
    for it in test: groups[it["group"]].append(it)
    gnames = sorted(groups); meta["groups"] = {g: len(groups[g]) for g in gnames}
    khs = a.kh if a.dataset == "medagentbench" else [0]
    print(f"{a.dataset}: test {len(test)} val {len(val)} | theta* (val) = {tstar} at {meta['val_spend_at_theta_star']} % spend, "
          f"val acc {meta['val_acc_at_theta_star']} | groups {meta['groups']} | N {N} bundles {a.bundles} shift {a.shift} kh {khs}", flush=True)

    def draw():
        if a.full: return list(test)
        if not a.shift or len(gnames) < 2: return [rng.choice(test) for _ in range(N)]
        w = [rng.gammavariate(1.0, 1.0) for _ in gnames]; s = sum(w); w = [x / s for x in w]
        return [rng.choice(groups[rng.choices(gnames, w)[0]]) for _ in range(N)]

    POL = [("spend-all", pol_spend_all), ("spend-to-peak", pol_spend_to_peak), ("per-query", pol_per_query), ("pacing", pol_pacing),
           ("oracle_B", pol_oracle_B), ("random_B", pol_random_B), ("oracle", pol_oracle), ("always_SS", pol_always_ss)]
    if a.dataset == "hotpotqa" and a.theta_locked is not None: POL.insert(3, ("locked", pol_locked))             # v1.8
    agg = collections.defaultdict(list); per_bundle = []; curve_acc = collections.defaultdict(list)
    for b in range(a.bundles):
        bundle = draw(); css = css_of(bundle); n = len(bundle)
        for t in THETAS:                                     # the test accuracy-spend curve (bundle means), for s*
            c, q = spend_acc(bundle, [it["path_at"][t] for it in bundle]); curve_acc[t].append((100 * c / css, 100 * q / n))
        for Bp in BUDGETS:
            Bf = Bp / 100
            for kh in khs:
                for name, fn in POL:
                    paths = fn(bundle, Bf=Bf, tstar=tstar, theta0=a.theta0, vcurve=vcurve, rng=rng, allow_h=False)
                    if a.dataset == "medagentbench" and kh and name in ("spend-to-peak", "pacing", "oracle_B"):
                        paths = with_handoffs(bundle, paths, kh, by_oracle=(name == "oracle_B"))
                    c, q = spend_acc(bundle, paths)
                    rec = {"bundle": b, "B": Bp, "kh": kh, "policy": name, "acc": 100 * q / n, "spend_pct": 100 * c / css,
                           "violation": int(c > Bf * css + 1e-9), "overshoot_pct": max(0.0, 100 * (c - Bf * css) / css)}
                    if a.dataset == "medagentbench":
                        tier = [j for j, it in enumerate(bundle) if it["tau"] >= 0.8]
                        rec.update({"tier_n": len(tier), "handoffs": sum(1 for p in paths if p == 2),
                                    "tier_fail_no_human": sum(1 for j in tier if paths[j] != 2 and bundle[j]["acc"][paths[j]] == 0)})
                    per_bundle.append(rec); agg[(Bp, kh, name)].append(rec)
        if (b + 1) % 200 == 0: print(f"  bundles {b + 1}/{a.bundles}", flush=True)

    order = [p[0] for p in POL]
    summary = {"version": "budget_allocation v1.6 2026-11-18 (+ --samplek/--k-weak: K>1 weak answerer)", "full": a.full, "script_sha256": sha(os.path.abspath(__file__)), "dataset": a.dataset, "meta": meta, "N": N, "bundles": a.bundles,
               "shift": a.shift, "theta0": a.theta0, "budgets": BUDGETS, "khs": khs,
               "curve": {str(t): {"spend_pct": round(sum(x[0] for x in v) / len(v), 2), "acc": round(sum(x[1] for x in v) / len(v), 2)}
                         for t, v in curve_acc.items()}, "results": {}}
    pk = max(summary["curve"].items(), key=lambda kv: (kv[1]["acc"], -kv[1]["spend_pct"]))
    summary["test_curve_peak"] = {"theta": float(pk[0]), **pk[1]}; summary["s_star_from_val_theta"] = summary["curve"][str(tstar)]
    hdr = f"{'B%':>4s} {'kh':>3s} {'policy':14s} {'acc':>6s} {'±SE':>5s} {'spend%':>7s} {'viol%':>6s} {'over%':>6s} {'regret':>7s}"
    print("\n" + hdr + ("  tierFailNoH  handoffs" if a.dataset == "medagentbench" else ""))
    for (Bp, kh, name), recs in sorted(agg.items(), key=lambda k: (k[0][0], k[0][1], order.index(k[0][2]))):
        acc, se = mean_se([r["acc"] for r in recs]); sp = sum(r["spend_pct"] for r in recs) / len(recs)
        viol = 100 * sum(r["violation"] for r in recs) / len(recs); over = sum(r["overshoot_pct"] for r in recs) / len(recs)
        orc = sum(r["acc"] for r in agg[(Bp, kh, "oracle_B")]) / len(recs)
        row = {"acc": round(acc, 2), "se": round(se, 2), "spend_pct": round(sp, 1), "violation_pct": round(viol, 1),
               "overshoot_pct": round(over, 2), "regret_to_oracle_B": round(orc - acc, 2)}
        line = f"{Bp:4d} {kh:3d} {name:14s} {acc:6.2f} {se:5.2f} {sp:7.1f} {viol:6.1f} {over:6.2f} {orc - acc:7.2f}"
        if a.dataset == "medagentbench":
            tf = sum(r["tier_fail_no_human"] for r in recs) / len(recs); ho = sum(r["handoffs"] for r in recs) / len(recs)
            row.update({"tier_fail_no_human": round(tf, 2), "handoffs": round(ho, 2)}); line += f"  {tf:11.2f}  {ho:8.2f}"
        summary["results"][f"B{Bp}|kh{kh}|{name}"] = row; print(line)
    print(f"\ntheta* (val) {tstar}: test spend {summary['s_star_from_val_theta']['spend_pct']} % acc {summary['s_star_from_val_theta']['acc']} | "
          f"test-curve peak theta {summary['test_curve_peak']['theta']}: spend {summary['test_curve_peak']['spend_pct']} % acc {summary['test_curve_peak']['acc']}")
    json.dump(summary, open(os.path.join(a.out, f"budget_{a.dataset}.json"), "w"), indent=1)
    with open(os.path.join(a.out, f"budget_{a.dataset}_per_bundle.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(per_bundle[0])); w.writeheader(); w.writerows(per_bundle)
    try:
        import openpyxl
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "summary"
        first = next(iter(summary["results"].values())); ws.append(["B%", "kh", "policy"] + list(first))
        for k, v in summary["results"].items():
            Bp, kh, name = k.split("|"); ws.append([int(Bp[1:]), int(kh[2:]), name] + [v.get(c) for c in first])
        ws2 = wb.create_sheet("curve"); ws2.append(["theta", "spend_pct", "acc"])
        for t, v in summary["curve"].items(): ws2.append([float(t), v["spend_pct"], v["acc"]])
        wb.save(os.path.join(a.out, f"budget_{a.dataset}.xlsx"))
    except ModuleNotFoundError:
        pass
    print("wrote", os.path.join(a.out, f"budget_{a.dataset}.json"))


if __name__ == "__main__":
    main()
