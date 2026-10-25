"""Jev novelty E8 (pre-registration H6): escalation with a guaranteed error rate among the questions kept on the weak
model, on open-ended multi-hop QA.

A router keeps a question on the weak model when its escalation score is at most t. The *selective risk* at t is the weak
model's error rate (lenient 0) among the kept questions. Learn-then-Test with fixed-sequence testing picks t on a
calibration half: the thresholds keeping 5%, 10%, ..., 100% of it are tested in that order, each with a binomial
(Clopper-Pearson) upper bound on its selective risk at level `delta`, and the scan stops at the first failure. So with
probability at least 1 - delta the chosen t has selective risk at most alpha, whatever the score (no calibration of the
score is needed: only its order matters).

    H6a  within a dataset, over 1,000 random halvings of calib + test: mean realised risk <= alpha and the realised
         risk exceeds alpha in at most 10% of halvings (delta = 0.1)
    H6b  the HotpotQA-trained judge, calibrated on HotpotQA and applied to MuSiQue: the guarantee breaks (expected)

    risk-control --plan configs/e2-musique.toml --out runs/e2-musique [--shift-from runs/slr-judge-models]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from scipy.stats import beta

from stair import provenance
from stair.confirm import entropy_policy, judge_policies, pair_arrays
from stair.judge import load_plan

log = logging.getLogger(__name__)

ALPHAS = (0.05, 0.10, 0.20)  # pre-registered
POSTHOC_ALPHAS = (0.30, 0.40)  # added after HotpotQA kept ~nothing at the others (deviation 6); exploratory
KEEP_GRID = tuple(round(0.05 * k, 2) for k in range(1, 21))  # kept fractions, tested in this order


def upper_bound(errors: int, n: int, delta: float) -> float:
    """Clopper-Pearson one-sided upper bound on a binomial rate."""
    return 1.0 if n == 0 else float(beta.ppf(1 - delta, errors + 1, n - errors)) if errors < n else 1.0


def ltt_threshold(score: np.ndarray, wrong: np.ndarray, alpha: float, delta: float,
                  grid: tuple[float, ...] = KEEP_GRID) -> float:
    """Learn-then-Test, fixed sequence: the thresholds that keep 5%, 10%, ... of the calibration questions (lowest
    scores first) are tested in that order with a Clopper-Pearson bound at level delta; the scan stops at the first
    failure and returns the loosest passing threshold (-inf keeps nothing)."""
    order = np.sort(score)
    t = -np.inf
    for frac in grid:
        k = max(1, int(round(frac * len(order))))
        cut = order[k - 1]
        kept = score <= cut
        if upper_bound(int(wrong[kept].sum()), int(kept.sum()), delta) > alpha:
            break
        t = cut
    return t


def outcome(score: np.ndarray, wrong: np.ndarray, small_cost: np.ndarray, large_cost: np.ndarray, t: float) -> dict:
    kept = score <= t
    return {"risk": float(wrong[kept].mean()) if kept.any() else 0.0, "kept": float(kept.mean()),
            "cost": float(np.where(kept, small_cost, small_cost + large_cost).mean())}


def resplits(score: np.ndarray, wrong: np.ndarray, small: np.ndarray, large: np.ndarray, alpha: float, delta: float,
             n_splits: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n = len(score)
    risks, kept, costs = [], [], []
    for _ in range(n_splits):
        perm = rng.permutation(n)
        cal, test = perm[: n // 2], perm[n // 2:]
        t = ltt_threshold(score[cal], wrong[cal], alpha, delta)
        o = outcome(score[test], wrong[test], small[test], large[test], t)
        risks.append(o["risk"])
        kept.append(o["kept"])
        costs.append(o["cost"])
    risks = np.asarray(risks)
    return {"alpha": alpha, "mean_risk": float(risks.mean()), "violation_rate": float((risks > alpha).mean()),
            "mean_kept": float(np.mean(kept)), "mean_cost": float(np.mean(costs)),
            "h6a": bool(risks.mean() <= alpha and (risks > alpha).mean() <= 0.10)}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", default="A")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--delta", type=float, default=0.10)
    parser.add_argument("--splits", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shift-from", nargs=2, metavar=("PLAN", "OUT"), type=Path,
                        help="H6b: calibrate the frozen HotpotQA judge on this run (e.g. configs/system-one-slr.toml "
                             "runs/slr-judge-models) and test it on --plan / --out's frozen hotpotqa judge")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = load_plan(args.plan)
    exclude = set()
    for path in args.exclude:
        exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    frames, arrays = {}, {}
    for split in ("calib", "test"):
        frames[split], arrays[split] = pair_arrays(plan, args.pair, split, exclude if split == "test" else set())
    pols = judge_policies(args.out, plan, args.pair, {s: frames[s]["id"].to_numpy() for s in frames})
    pols["small-entropy (post)"] = entropy_policy(frames)
    pool = {k: np.concatenate([arrays["calib"][k], arrays["test"][k]]) for k in arrays["test"]}
    wrong = (pool["small_f1"] < 0.5).astype(int)  # lenient: 0 / 1
    report = {"n": len(wrong), "weak_error": float(wrong.mean()), "delta": args.delta, "policies": {}}
    for name in [k for k in pols if k.startswith("judge/qpa") or k == "small-entropy (post)"]:
        p = pols[name]
        score = np.concatenate([np.mean(p.calib, axis=0), np.mean(p.seeds, axis=0)])
        report["policies"][name] = [{**resplits(score, wrong, pool["small_cost"], pool["large_cost"], a, args.delta,
                                                args.splits, args.seed), "posthoc": a in POSTHOC_ALPHAS}
                                    for a in (*ALPHAS, *POSTHOC_ALPHAS)]
    if args.shift_from:
        src_plan, src_out = load_plan(args.shift_from[0]), args.shift_from[1]
        sf, sa = {}, {}
        for split in ("calib", "test"):
            sf[split], sa[split] = pair_arrays(src_plan, "A", split, set())
        src = judge_policies(src_out, src_plan, "A", {s: sf[s]["id"].to_numpy() for s in sf})["judge/qpa-large-helps-nanojev-A"]
        src_score = np.concatenate([np.mean(src.calib, axis=0), np.mean(src.seeds, axis=0)])
        src_wrong = (np.concatenate([sa["calib"]["small_f1"], sa["test"]["small_f1"]]) < 0.5).astype(int)
        target_pol = pols["judge/qpa-large-helps-hotpotqa-A"]
        t_score = np.mean(target_pol.seeds, axis=0)
        t_wrong = (arrays["test"]["small_f1"] < 0.5).astype(int)
        report["shift"] = []
        for a in (*ALPHAS, *POSTHOC_ALPHAS):
            t = ltt_threshold(src_score, src_wrong, a, args.delta)
            o = outcome(t_score, t_wrong, arrays["test"]["small_cost"], arrays["test"]["large_cost"], t)
            report["shift"].append({"alpha": a, **o, "violated": bool(o["risk"] > a), "posthoc": a in POSTHOC_ALPHAS})
    target = args.out / "confirm"
    target.mkdir(parents=True, exist_ok=True)
    (target / "H6.json").write_text(json.dumps(report, indent=1))
    provenance.record(target, "risk-control", plan=str(args.plan), **provenance.environment())
    for name, rows in report["policies"].items():
        for r in rows:
            log.info(f"{name:40s} alpha {r['alpha']:.2f}: risk {r['mean_risk']:.3f}, violations {r['violation_rate']:.2f}, "
                     f"kept {r['mean_kept']:.2f}, cost {r['mean_cost']:.2f} {'H6a ok' if r['h6a'] else ''}")
    for r in report.get("shift", []):
        log.info(f"shift alpha {r['alpha']:.2f}: risk {r['risk']:.3f}, kept {r['kept']:.2f} {'VIOLATED' if r['violated'] else ''}")


if __name__ == "__main__":
    main()
