"""The Jev novelty programme's confirmatory tests (docs/experiments/2026-10-24-jev-preregistration.md): H2 (evidence
decides escalation), H3 (the stronger baselines) and H5 (transfer across model pairs), on a judge run that
`judge train` + `judge eval` finished.

A policy is a set of per-seed escalation scores on the pair's test questions, whether it decides after the weak answer
(`post`), and its own price per question beyond the experts (signals that call the weak model again). A judge's AIQ is
the mean over its seeds of each seed's AIQ (one deployed model). Two-level paired bootstrap: test questions resampled
with the same indices for every policy, then each policy's seeds resampled. Tests, each one-sided at 0.05, Holm within
the family (per dataset for H2 / H3):

    superiority      d = AIQ(a) - AIQ(b) > 0                                  p = P*(d <= 0)
    non-inferiority  d > -delta                                               p = P*(d <= -delta)
    equivalence      |d| < delta (TOST)                                       p = max(P*(d <= -delta), P*(d >= delta))
    gap ratio        (AIQ(a) - random) - r x (AIQ(b) - random) > 0            p = P*(stat <= 0)

    confirm --family H2 --plan configs/e2-musique.toml --out runs/e2-musique --data-dir data/musique
    confirm --family H3 --plan ... --signals runs/musique-pairA/signals
    confirm --family H3 --free ...   (descriptive, outside the family: the priced signals with their extra calls free)
    confirm --family H5 --plan configs/e5-transfer.toml --out runs/e5-transfer --data-dir data/hotpotqa --exclude ...
writes <out>/confirm/<family>[-<pair>][-free].json.
"""

import argparse
import json
import logging
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from stair_common.cost import budget

from stair import provenance
from stair.evaluate_routing import CURVE_COLUMNS, aiq, curve, expert_params, random_curve, with_cost_unit
from stair.judge import load_plan
from stair.label_curve import holm
from stair.metrics import metric_view
from stair.router import ROUTER_INPUTS, escalation_label, predict_escalation, train_routers

log = logging.getLogger(__name__)

DELTA = 0.010  # the pre-registered margin (AIQ)


@dataclass
class Policy:
    name: str
    seeds: list[np.ndarray]  # per-seed escalation scores on the test questions (higher = escalate)
    post: bool
    extra: np.ndarray | None = None  # own price per question in the curves' unit (None: free)
    calib: list[np.ndarray] = field(default_factory=list)  # the same on calib, to pick "the stronger" baselines


def policy_aiq(a: dict[str, np.ndarray], p: Policy, idx: np.ndarray | None, seed_idx: np.ndarray | None = None) -> float:
    s = a if idx is None else {k: v[idx] for k, v in a.items()}
    cmin, cmax = s["small_cost"].mean(), s["large_cost"].mean()
    extra = 0.0 if p.extra is None else float((p.extra if idx is None else p.extra[idx]).mean())
    seeds = p.seeds if seed_idx is None else [p.seeds[i] for i in seed_idx]
    vals = []
    for sc in seeds:
        pts = curve(s, sc if idx is None else sc[idx], p.post)
        vals.append(aiq([{**q, "cost": q["cost"] + extra} for q in pts] if extra else pts, cmin, cmax))
    return float(np.mean(vals))


def random_aiq(a: dict[str, np.ndarray], idx: np.ndarray | None = None) -> float:
    s = a if idx is None else {k: v[idx] for k, v in a.items()}
    return aiq(random_curve(s), s["small_cost"].mean(), s["large_cost"].mean())


@dataclass
class Test:
    name: str
    kind: str  # superiority | noninferiority | equivalence | ratio | transfer-gap
    a: str
    b: str | None = None
    ratio: float = 0.0
    a2: str | None = None  # transfer-gap: (a / b) is one transfer ratio, (a2 / b2) the other
    b2: str | None = None


def statistic(t: Test, aiqs: dict[str, float], rand: float) -> float:
    if t.kind == "ratio":
        return (aiqs[t.a] - rand) - t.ratio * (aiqs[t.b] - rand)
    if t.kind == "transfer-gap":  # (frozen - random) / (retrained - random), first minus second
        return (aiqs[t.a] - rand) / (aiqs[t.b] - rand) - (aiqs[t.a2] - rand) / (aiqs[t.b2] - rand)
    if t.kind == "equivalence" and t.b is None:
        return aiqs[t.a] - rand
    return aiqs[t.a] - aiqs[t.b]


def run_tests(a: dict[str, np.ndarray], policies: dict[str, Policy], tests: list[Test], n_boot: int, seed: int,
              delta: float = DELTA) -> dict:
    needed = sorted({x for t in tests for x in (t.a, t.b, t.a2, t.b2) if x})
    point = {n: policy_aiq(a, policies[n], None) for n in needed}
    rand = random_aiq(a)
    rng = np.random.default_rng(seed)
    n = len(a["small_f1"])
    reps = {t.name: [] for t in tests}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        aiqs = {k: policy_aiq(a, policies[k], idx, rng.integers(0, len(policies[k].seeds), len(policies[k].seeds)))
                for k in needed}
        r = random_aiq(a, idx)
        for t in tests:
            reps[t.name].append(statistic(t, aiqs, r))
    out, pvals = {}, {}
    for t in tests:
        v = np.asarray(reps[t.name])
        est = statistic(t, point, rand)
        if t.kind in ("superiority", "ratio", "transfer-gap"):
            p = float((v <= 0).mean())
        elif t.kind == "noninferiority":
            p = float((v <= -delta).mean())
        else:  # equivalence (TOST)
            p = float(max((v <= -delta).mean(), (v >= delta).mean()))
        pvals[t.name] = p
        out[t.name] = {"kind": t.kind, "a": t.a, "b": t.b, "ratio": t.ratio or None, "estimate": est,
                       "ci90": [float(np.percentile(v, 5)), float(np.percentile(v, 95))],
                       "ci95": [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))], "p": p}
    for name, p in holm(pvals).items():
        out[name]["p_holm"] = p
        out[name]["supported"] = bool(p < 0.05)
    return {"n_test": n, "n_boot": n_boot, "delta": delta, "random_aiq": rand,
            "aiq": point, "tests": out}


# ---------------------------------------------------------------------------------------------
# Loading


def pair_arrays(plan: dict, pair: str, split: str, exclude: set[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    run = Path(plan["pairs"][pair])
    cfg = tomllib.loads((run / "config.toml").read_text())
    params, models = expert_params(cfg), {e["name"]: e.get("model", "") for e in cfg["experts"]}
    df = with_cost_unit(metric_view(pd.read_parquet(run / f"{split}.parquet"), plan.get("metric", "f1")),
                        plan.get("cost", "flops"), params, models)
    df = df[~df["id"].isin(exclude)].reset_index(drop=True)
    return df, {c: df[c].to_numpy(dtype=float) for c in CURVE_COLUMNS}


def judge_policies(out: Path, plan: dict, pair: str, ids: dict[str, np.ndarray]) -> dict[str, Policy]:
    """Every judge of the run, per seed, from `judge eval`'s merged score files (the judge's score is P(large))."""
    from stair.judge import INPUTS, judge_name

    frames = {split: pd.read_parquet(out / plan.get("eval_dir", "eval") / pair / f"scores-{split}.parquet").set_index("id")
              for split in ("calib", "test")}
    pols = {}
    for spec in plan["judges"]:
        col = f"judge/{judge_name(spec)}"
        seeds = sorted(c for c in frames["test"].columns if c.startswith(f"{col}#"))
        if not seeds:
            continue
        pols[col] = Policy(col, [frames["test"].loc[ids["test"], c].to_numpy() for c in seeds], INPUTS[spec["inputs"]],
                           calib=[frames["calib"].loc[ids["calib"], c].to_numpy() for c in seeds])
    return pols


def lr_policies(plan: dict, pair: str, frames: dict[str, pd.DataFrame], seed: int) -> dict[str, Policy]:
    """The logistic-regression routers `judge eval` trains (same data, same seed), as policies."""
    run = Path(plan["pairs"][pair])
    train = metric_view(pd.read_parquet(run / "router_train.parquet"), plan.get("metric", "f1"))
    routers = train_routers(train, plan["tau"], seed)
    return {f"router:{name}": Policy(f"router:{name}", [predict_escalation(r, frames["test"])], r["post"],
                                     calib=[predict_escalation(r, frames["calib"])])
            for name, r in routers.items() if r["label"] == "large-helps"}


def entropy_policy(frames: dict[str, pd.DataFrame]) -> Policy:
    return Policy("small-entropy (post)", [frames["test"]["small_uncertainty"].to_numpy(dtype=float)], True,
                  calib=[frames["calib"]["small_uncertainty"].to_numpy(dtype=float)])


def stronger(a_cal: dict[str, np.ndarray], candidates: list[Policy]) -> Policy:
    """The candidate with the higher AIQ on calib (so the comparison's opponent is chosen without the test set)."""
    return max(candidates, key=lambda p: np.mean([aiq(curve(a_cal, s, p.post), a_cal["small_cost"].mean(),
                                                          a_cal["large_cost"].mean()) for s in p.calib]))


def signal_policies(signals: Path, frames: dict[str, pd.DataFrame], unit_price, tau: float, seed: int) -> dict[str, Policy]:
    """E3's weak-model signals, oriented so higher = escalate and priced for their extra calls; the hidden-state probe
    (logistic regression on router_train, 5-fold C), retrieval-feature trees and the NLI check if present."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegressionCV

    sig = {s: pd.read_parquet(signals / f"{s}.parquet").set_index("id") for s in ("router_train", "calib", "test")}
    settings = json.loads((signals / "settings.json").read_text())
    params = settings.get("params_b") or 1.5
    pols = {}
    oriented = {"msp": lambda d: 1 - d["sig_msp"], "mean-logprob": lambda d: -d["sig_mean_logprob"],
                "first-margin": lambda d: -d["sig_first_margin"], "p-true": lambda d: 1 - d["sig_ptrue"],
                "self-consistency": lambda d: 1 - d["sig_sc_agree"], "semantic-entropy": lambda d: d["sig_semantic_entropy"],
                "self-verification": lambda d: 1 - d["sig_verify_yes"]}
    calls = {"p-true": ("ptrue_prompt_tokens", None), "self-consistency": ("sample_prompt_tokens", "sample_generated_tokens"),
             "semantic-entropy": ("sample_prompt_tokens", "sample_generated_tokens"),
             "self-verification": ("verify_prompt_tokens", "verify_generated_tokens")}
    for name, f in oriented.items():
        t, c = sig["test"].loc[frames["test"]["id"]], sig["calib"].loc[frames["calib"]["id"]]
        extra = None
        if name in calls:
            p_col, g_col = calls[name]
            extra = unit_price(np.array([budget(params, int(pt), int(t.at[i, g_col]) if g_col else 1)
                                         for i, pt in zip(t.index, t[p_col])]))
        pols[f"signal:{name}"] = Policy(f"signal:{name}", [f(t).to_numpy(dtype=float)], True, extra,
                                        calib=[f(c).to_numpy(dtype=float)])
    train = frames["router_train"]
    y = escalation_label(train, tau, "large-helps")
    hid = {s: np.load(signals / f"{s}-hidden.npy").astype(np.float32).reshape(len(sig[s]), -1) for s in sig}
    pos = {s: pd.Series(np.arange(len(sig[s])), index=sig[s].index) for s in sig}
    X = {s: hid[s][pos[s].loc[frames[s]["id"]].to_numpy()] for s in ("router_train", "calib", "test")}
    probe = LogisticRegressionCV(Cs=[0.001, 0.01, 0.1, 1.0], cv=5, max_iter=2000, scoring="neg_log_loss",
                                 random_state=seed).fit(X["router_train"], y)
    pols["signal:hidden-probe"] = Policy("signal:hidden-probe", [probe.predict_proba(X["test"])[:, 1]], True,
                                         calib=[probe.predict_proba(X["calib"])[:, 1]])

    def retrieval(df: pd.DataFrame) -> np.ndarray:
        scores = [np.sort(np.asarray(s, dtype=float))[::-1] for s in df["para_scores"]]
        return np.array([[s[0], s[0] - s[1], s[:2].sum(), float(pd.Series(s).skew())] for s in scores])

    gbt = HistGradientBoostingClassifier(random_state=seed).fit(retrieval(train), y)
    pols["retrieval-trees"] = Policy("retrieval-trees", [gbt.predict_proba(retrieval(frames["test"]))[:, 1]], False,
                                     calib=[gbt.predict_proba(retrieval(frames["calib"]))[:, 1]])
    if "sig_nli_entail" in sig["test"]:
        pols["signal:nli"] = Policy("signal:nli", [1 - sig["test"].loc[frames["test"]["id"], "sig_nli_entail"].to_numpy()],
                                    True, calib=[1 - sig["calib"].loc[frames["calib"]["id"], "sig_nli_entail"].to_numpy()])
    return pols


# ---------------------------------------------------------------------------------------------
# Families


def h2_tests(pols: dict[str, Policy], a_cal: dict[str, np.ndarray]) -> tuple[list[Test], dict[str, Policy]]:
    q, qp, qpa = "judge/q-large-helps-nanojev-A", "judge/qp-large-helps-nanojev-A", "judge/qpa-large-helps-nanojev-A"
    before = [pols[k] for k in pols if k.startswith("router:") and not ROUTER_INPUTS[k[7:].split("/")[0]][2]]
    after = [pols[k] for k in pols if k.startswith("router:") and ROUTER_INPUTS[k[7:].split("/")[0]][2]]
    ent = pols["small-entropy (post)"]
    pols["opponent-before"] = stronger(a_cal, before + [ent])
    pols["opponent-after"] = stronger(a_cal, after + [ent])
    tests = [Test("H2a qp > q", "superiority", qp, q), Test("H2b q ~ random", "equivalence", q),
             Test("H2c qp >= stronger(entropy, LR before) - delta", "noninferiority", qp, "opponent-before"),
             Test("H2c qpa >= stronger(entropy, LR cascade) - delta", "noninferiority", qpa, "opponent-after")]
    for x, frozen in (("qp", "judge/qp-large-helps-hotpotqa-A"), ("qpa", "judge/qpa-large-helps-hotpotqa-A")):
        if frozen in pols:
            tests.append(Test(f"H2d {x}: HotpotQA judge keeps >= 75% of the in-domain gain", "ratio", frozen,
                              f"judge/{x}-large-helps-nanojev-A", 0.75))
    return tests, pols


def h3_tests(pols: dict[str, Policy], judge: Policy) -> tuple[list[Test], dict[str, Policy]]:
    pols["nano-jev qpa (priced)"] = judge
    cheap = ["signal:msp", "signal:mean-logprob", "signal:first-margin", "signal:p-true", "signal:hidden-probe",
             "retrieval-trees", "signal:nli"]
    tests = [Test(f"H3a qpa >= {c} - delta", "noninferiority", "nano-jev qpa (priced)", c) for c in cheap if c in pols]
    tests += [Test(f"H3b qpa > {c}", "superiority", "nano-jev qpa (priced)", c)
              for c in ("signal:self-consistency", "signal:semantic-entropy") if c in pols]
    return tests, pols


def h3_free_tests(pols: dict[str, Policy], judge: Policy) -> tuple[list[Test], dict[str, Policy]]:
    """The pre-registration reports every signal free and priced: nano-jev qpa against each signal that pays for extra
    calls, with those calls free. Descriptive, Holm within these tests only, never mixed into the family."""
    pols["nano-jev qpa (priced)"] = judge
    tests = []
    for c in ("signal:p-true", "signal:self-consistency", "signal:semantic-entropy", "signal:self-verification"):
        if c in pols:
            p = pols[c]
            pols[f"{c} (free)"] = Policy(f"{p.name} (free)", p.seeds, p.post, None, p.calib)
            tests.append(Test(f"free: qpa > {c} (free)", "superiority", "nano-jev qpa (priced)", f"{c} (free)"))
    return tests, pols


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--family", choices=["H2", "H3", "H5"], required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="the judge run (judge train + eval)")
    parser.add_argument("--pair", default="A")
    parser.add_argument("--signals", type=Path, help="H3: the pair's signals folder")
    parser.add_argument("--free", action="store_true", help="H3: the priced signals with their extra calls free "
                        "(descriptive, writes <family>-free.json beside the family's own report)")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    if args.free and args.family != "H3":
        parser.error("--free is H3's (the signals that pay for extra calls)")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = load_plan(args.plan)
    exclude = set()
    for path in args.exclude:
        exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    target = args.out / "confirm"
    target.mkdir(parents=True, exist_ok=True)

    if args.family == "H5":
        reports, pvals = {}, {}
        for pair in [p for p in plan["pairs"] if p != "A"]:
            frames, arrays = {}, {}
            for split in ("calib", "test"):
                frames[split], arrays[split] = pair_arrays(plan, pair, split, exclude if split == "test" else set())
            pols = judge_policies(args.out, plan, pair, {s: frames[s]["id"].to_numpy() for s in frames})
            fz, rt = (lambda x: f"judge/{x}-large-helps-nanojev-A"), (lambda x: f"judge/{x}-large-helps-nanojev-{pair}")  # noqa: E731
            tests = [Test(f"H5a {pair}: frozen qp >= retrained qp - delta", "noninferiority", fz("qp"), rt("qp"))]
            if pair in ("C", "D"):  # the weak model changes: qpa reads its answers, qp does not
                tests.append(Test(f"H5b {pair}: qp transfers better than qpa", "transfer-gap", fz("qp"), rt("qp"), 0.0,
                                  fz("qpa"), rt("qpa")))
            report = run_tests(arrays["test"], pols, tests, args.bootstrap, args.seed)
            rand = report["random_aiq"]
            report["transfer_ratio"] = {x: (report["aiq"].get(fz(x), np.nan) - rand) / (report["aiq"].get(rt(x), np.nan) - rand)
                                        for x in ("qp", "qpa") if fz(x) in report["aiq"] and rt(x) in report["aiq"]}
            reports[pair] = report
            pvals |= {name: t["p"] for name, t in report["tests"].items()}
        for name, p in holm(pvals).items():  # Holm over the whole family, across pairs
            reports[name.split()[1].rstrip(":")]["tests"][name]["p_holm_family"] = p
        (target / "H5.json").write_text(json.dumps(reports, indent=1))
        provenance.record(target, "confirm-H5", plan=str(args.plan), **provenance.environment())
        for pair, r in reports.items():
            log.info(f"H5 pair {pair}: transfer ratio {r['transfer_ratio']}")
        return

    frames, arrays = {}, {}
    for split in ("router_train", "calib", "test"):
        frames[split], arrays[split] = pair_arrays(plan, args.pair, split, exclude if split == "test" else set())
    ids = {s: frames[s]["id"].to_numpy() for s in ("calib", "test")}
    pols = judge_policies(args.out, plan, args.pair, ids)
    pols |= lr_policies(plan, args.pair, frames, args.seed)
    pols["small-entropy (post)"] = entropy_policy(frames)
    if args.family == "H2":
        tests, pols = h2_tests(pols, arrays["calib"])
    else:
        if plan.get("cost") != "table5":
            raise SystemExit("H3 prices the weak model's extra calls with budget(): a table5 plan only")
        run = Path(plan["pairs"][args.pair])
        pols |= signal_policies(args.signals or run / "signals", frames, lambda b: b, plan["tau"], args.seed)
        tests, pols = (h3_free_tests if args.free else h3_tests)(pols, pols["judge/qpa-large-helps-nanojev-A"])
    report = run_tests(arrays["test"], pols, tests, args.bootstrap, args.seed)
    report["all_aiq"] = {k: policy_aiq(arrays["test"], p, None) for k, p in pols.items()}
    report["opponents"] = {k: pols[k].name for k in ("opponent-before", "opponent-after") if k in pols}
    tag = ("" if args.pair == "A" else f"-{args.pair}") + ("-free" if args.free else "")
    (target / f"{args.family}{tag}.json").write_text(json.dumps(report, indent=1))
    provenance.record(target, f"confirm-{args.family}{'-free' if args.free else ''}", plan=str(args.plan),
                      **provenance.environment())
    for name, t in report["tests"].items():
        log.info(f"{name:60s} {t['estimate']:+.4f} [{t['ci90'][0]:+.4f}, {t['ci90'][1]:+.4f}] p {t['p']:.4f} "
                 f"Holm {t['p_holm']:.4f} {'SUPPORTED' if t['supported'] else ''}")


if __name__ == "__main__":
    main()
