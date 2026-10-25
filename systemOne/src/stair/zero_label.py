"""Jev novelty E4 (pre-registration H4): routing from nano-jev's untrained typed decisions, with no or few routing labels.

Scores (higher = escalate), all from nano-jev v1.0 with no fine-tuning:
    before the weak answer   insufficient   1 - P(sufficient) over the top 2 passages   (the cache's ev_sufficiency_k2)
                             low-top1       - relevance of the top passage                (ev_top1)
                             low-top2       - relevance of the top two                    (ev_top2_sum)
    after it                 not-grounded   P(no) that the passages support "the answer is X"  (judge zeroshot)
                             not-correct    P(no) that the proposed answer is correct            (judge zeroshot)
Zero-label compositions: every single score; for the pre-answer scores, the max of (insufficient, rank of low-top1)
and the product of the "fine" probabilities (1 - (1 - insufficient)(1 - rank of low-top1)); the same over all five.
Few-label: a logistic combiner of all five scores (standardised) on n = 10, 25 or 50 labelled questions, 20 draws each;
the draws are the policy's "seeds". The labelled questions come from calib (deviation 3: the zero-shot answer scores
exist on calib and test only); test is never used to fit or choose.

H4 (Holm), with the full-budget trained judges of the run (qp for pre-answer scores, qpa for the rest):
    H4a  the zero-label composition with the best calib AIQ closes >= 50% of the random -> trained-judge gap on test
    H4b  the 50-label combiner closes >= 75% of the random -> trained qpa gap (mean over draws)
    H4c  the best pre-answer zero-label composition (calib) beats entropy, which pays the weak answer it needs

    zero-label --plan configs/e2-musique.toml --out runs/e2-musique
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from stair import provenance
from stair.confirm import Policy, Test, pair_arrays, policy_aiq, run_tests
from stair.judge import load_plan, zero_shot_judges
from stair.router import escalation_label

log = logging.getLogger(__name__)

PRE = ("insufficient", "low-top1", "low-top2")
POST = ("not-grounded", "not-correct")
DRAWS = 20
LABELS = (10, 25, 50)


def typed_scores(df: pd.DataFrame, zs: pd.DataFrame, prefix: str) -> pd.DataFrame:
    zs = zs.set_index("id").loc[df["id"]]
    return pd.DataFrame({"insufficient": 1 - df["ev_sufficiency_k2"].to_numpy(dtype=float),
                         "low-top1": -df["ev_top1"].to_numpy(dtype=float),
                         "low-top2": -df["ev_top2_sum"].to_numpy(dtype=float),
                         "not-grounded": zs[f"{prefix}_grounded"].to_numpy(dtype=float),
                         "not-correct": zs[f"{prefix}_correct"].to_numpy(dtype=float)})


def unit(x: np.ndarray) -> np.ndarray:
    """Ranks scaled to (0, 1]: relevance scores are not probabilities, so compositions use ranks."""
    return rankdata(x) / len(x)


def compositions(s: pd.DataFrame) -> dict[str, tuple[np.ndarray, bool]]:
    """name -> (escalation score, post) for every zero-label composition."""
    out = {name: (s[name].to_numpy(), name in POST) for name in (*PRE, *POST)}
    pre = [s["insufficient"].to_numpy(), unit(s["low-top1"].to_numpy())]
    out["max(pre)"] = (np.maximum(*pre), False)
    out["product(pre)"] = (1 - (1 - pre[0]) * (1 - pre[1]), False)
    every = [unit(s[c].to_numpy()) for c in (*PRE, *POST)]
    out["max(all)"] = (np.max(every, axis=0), True)
    out["mean-rank(all)"] = (np.mean(every, axis=0), True)
    return out


def combiner_draws(cal: pd.DataFrame, y_cal: np.ndarray, test: pd.DataFrame, n: int, seed: int) -> tuple[list, list]:
    """DRAWS logistic combiners of the five scores, each fitted on n random calib questions (both classes present)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    rng = np.random.default_rng(seed)
    cols = [*PRE, *POST]
    on_test, on_cal = [], []
    while len(on_test) < DRAWS:
        idx = rng.choice(len(cal), n, replace=False)
        if len(set(y_cal[idx])) < 2:
            continue
        m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=1000)).fit(cal[cols].to_numpy()[idx], y_cal[idx])
        on_test.append(m.predict_proba(test[cols].to_numpy())[:, 1])
        on_cal.append(m.predict_proba(cal[cols].to_numpy())[:, 1])
    return on_test, on_cal


def main(argv: list[str] | None = None) -> None:
    from stair.confirm import judge_policies, entropy_policy

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pair", default="A")
    parser.add_argument("--exclude", type=Path, action="append", default=[])
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = load_plan(args.plan)
    exclude = set()
    for path in args.exclude:
        exclude |= {line.strip() for line in path.read_text().splitlines() if line.strip()}
    path, prefix, folder = next((z for z in zero_shot_judges(plan) if "nano-jev" in z[0]), (None, None, None))
    if path is None:
        raise SystemExit(f"{args.plan} has no nano-jev zero-shot judge")
    frames, arrays, scores = {}, {}, {}
    for split in ("calib", "test"):
        frames[split], arrays[split] = pair_arrays(plan, args.pair, split, exclude if split == "test" else set())
        zs = pd.read_parquet(args.out / folder / args.pair / f"{split}.parquet")
        scores[split] = typed_scores(frames[split], zs, prefix)
    y_cal = escalation_label(frames["calib"], plan["tau"], "large-helps")

    pols = judge_policies(args.out, plan, args.pair, {s: frames[s]["id"].to_numpy() for s in frames})
    pols["small-entropy (post)"] = entropy_policy(frames)
    comps = {s: compositions(scores[s]) for s in scores}
    for name, (sc, post) in comps["test"].items():
        pols[f"zero:{name}"] = Policy(f"zero:{name}", [sc], post, calib=[comps["calib"][name][0]])
    for n in LABELS:
        t, c = combiner_draws(scores["calib"], y_cal, scores["test"], n, args.seed)
        pols[f"few:{n}"] = Policy(f"few:{n}", t, True, calib=c)

    def calib_aiq(name: str) -> float:
        p = pols[name]
        return policy_aiq(arrays["calib"], Policy(name, p.calib, p.post), None)

    zero = [k for k in pols if k.startswith("zero:")]
    best = max(zero, key=calib_aiq)
    best_pre = max((k for k in zero if not pols[k].post), key=calib_aiq)
    trained = {False: "judge/qp-large-helps-nanojev-A", True: "judge/qpa-large-helps-nanojev-A"}
    tests = [Test(f"H4a {best} (best on calib) closes >= 50% of the gap", "ratio", best, trained[pols[best].post], 0.5),
             Test("H4b 50-label combiner closes >= 75% of the gap", "ratio", "few:50", trained[True], 0.75),
             Test(f"H4c {best_pre} (best pre-answer on calib) > entropy", "superiority", best_pre, "small-entropy (post)")]
    report = run_tests(arrays["test"], pols, tests, args.bootstrap, args.seed)
    rand = report["random_aiq"]
    report["all_aiq"] = {k: policy_aiq(arrays["test"], p, None) for k, p in pols.items()}
    report["gap_closed"] = {k: (v - rand) / (report["all_aiq"][trained[pols[k].post]] - rand)
                            for k, v in report["all_aiq"].items() if k.startswith(("zero:", "few:"))}
    report["chosen_on_calib"] = {"best": best, "best_pre": best_pre, "calib_aiq": {k: calib_aiq(k) for k in zero}}
    target = args.out / "confirm"
    target.mkdir(parents=True, exist_ok=True)
    tag = "" if args.pair == "A" else f"-{args.pair}"
    (target / f"H4{tag}.json").write_text(json.dumps(report, indent=1))
    provenance.record(target, "zero-label", plan=str(args.plan), **provenance.environment())
    for k in sorted(report["gap_closed"], key=report["gap_closed"].get, reverse=True):
        log.info(f"  {k:24s} AIQ {report['all_aiq'][k]:.4f}  gap closed {report['gap_closed'][k]:+.2f}")
    for name, t in report["tests"].items():
        log.info(f"{name:62s} {t['estimate']:+.4f} p {t['p']:.4f} Holm {t['p_holm']:.4f} {'SUPPORTED' if t['supported'] else ''}")


if __name__ == "__main__":
    main()
