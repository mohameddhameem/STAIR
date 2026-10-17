"""Experiment 8, phase 3 (docs/experiment-8-plan.md): compare answer-option policies on the pool questions.

An *option* is one answering cell of configs/exp8.toml, for example the 3B answering once, the 3B
sampling ten answers and fusing them, or the 70B answering once. Options are ordered from cheap to
expensive. For each question the cached answers say which options were lenient-correct and what each
cost, in AgentTTS units (one unit = one 3B pass on its cheapest task; every option already includes
nano-jev's scoring cost).

A policy picks one option per question. Its points are (mean cost, mean lenient accuracy). Their
non-decreasing upper convex hull gives the AIQ, as in experiments 1-7 (`evaluate_routing.aiq`).

- Fixed policies: always one option, and random mixtures of them (their hull).
- Oracle: the cheapest option that is correct, else the cheapest option.
- Score baselines (two options, cheapest and strongest): the cheapest option's answer entropy, and
  logistic regression on nano-jev's scores (before the answer) with the entropy added (after it).
- The System One judge (`exp8_judge`) gives P(option i is the cheapest correct one). A threshold t
  picks the cheapest option whose cumulative probability reaches t. t from 0 (cheapest everywhere) to
  1 (the strongest option when the judge is unsure) traces the curve.

The operating point is chosen on calib and applied to test: the most accurate threshold whose calib
cost is within the budget (a fraction of AgentTTS's strong-everything cost), or the cheapest one
within `max_drop` of the strongest option's calib accuracy.
"""

import logging

import numpy as np
import pandas as pd

from stair.evaluate_routing import RATES, aiq

log = logging.getLogger(__name__)

THRESHOLDS = np.linspace(0.0, 1.0, 101)


def cheapest_correct_label(ok: np.ndarray, cost: np.ndarray) -> np.ndarray:
    """Index of the cheapest correct option per question; the cheapest option when none is correct."""
    return np.where(ok.any(axis=1), np.where(ok, cost, np.inf).argmin(axis=1), cost.argmin(axis=1))


def cumulative_choice(probs: np.ndarray, t: float) -> np.ndarray:
    """First option (in cost order) whose cumulative probability reaches t; t = 1 may pick the last."""
    cum = np.cumsum(probs, axis=1)
    return np.minimum((cum < t - 1e-9).sum(axis=1), probs.shape[1] - 1)


def point(ok: np.ndarray, cost: np.ndarray, choice: np.ndarray, extra: np.ndarray | float = 0.0) -> dict:
    rows = np.arange(len(choice))
    return {"f1": float(ok[rows, choice].mean()), "cost": float((cost[rows, choice] + extra).mean())}


def judge_points(ok: np.ndarray, cost: np.ndarray, probs: np.ndarray, extra: np.ndarray | float = 0.0) -> list[dict]:
    return [{"t": float(t), **point(ok, cost, cumulative_choice(probs, t), extra)} for t in THRESHOLDS]


def always_points(ok: np.ndarray, cost: np.ndarray) -> list[dict]:
    return [{"option": k, "f1": float(ok[:, k].mean()), "cost": float(cost[:, k].mean())} for k in range(ok.shape[1])]


def oracle_point(ok: np.ndarray, cost: np.ndarray) -> dict:
    return point(ok, cost, cheapest_correct_label(ok, cost))


def score_points(ok: np.ndarray, cost: np.ndarray, score: np.ndarray, lo: int, hi: int, post: bool) -> list[dict]:
    """Escalate the highest-scoring fraction of questions from option `lo` to option `hi`.

    `post`: the score needs `lo`'s answer, so escalated questions pay for both calls.
    """
    n = len(score)
    order = np.argsort(-np.asarray(score), kind="stable")
    d_ok = ok[:, hi].astype(float) - ok[:, lo]
    d_cost = cost[:, hi] if post else cost[:, hi] - cost[:, lo]
    cum_ok = np.concatenate([[0.0], np.cumsum(d_ok[order])])
    cum_cost = np.concatenate([[0.0], np.cumsum(d_cost[order])])
    out = []
    for r in RATES:
        k = round(r * n)
        out.append({"rate": k / n, "f1": float((ok[:, lo].sum() + cum_ok[k]) / n),
                    "cost": float((cost[:, lo].sum() + cum_cost[k]) / n)})
    return out


def choose_threshold(points: list[dict], budget: float | None = None, floor: float | None = None) -> dict:
    """Operating point on calib: the most accurate point within `budget`, else the cheapest above `floor`."""
    if budget is not None:
        within = [p for p in points if p["cost"] <= budget]
        return max(within or [min(points, key=lambda p: p["cost"])], key=lambda p: (p["f1"], -p["cost"]))
    above = [p for p in points if p["f1"] >= floor]
    return min(above or [max(points, key=lambda p: p["f1"])], key=lambda p: p["cost"])


def evidence_features(nanojev: list[float]) -> np.ndarray:
    s = np.sort(np.asarray(nanojev, dtype=float))[::-1]
    return np.array([s[0], s[1], s[2], s[0] - s[1], s[1] - s[2], s[:5].mean(), s.std()])


def fit_logistic(X: np.ndarray, y: np.ndarray, seed: int):
    """Standardised logistic regression with C chosen by 5-fold CV log loss."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GridSearchCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    grid = GridSearchCV(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)),
                        {"logisticregression__C": [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
                        scoring="neg_log_loss", cv=5)
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_["logisticregression__C"]


def bootstrap(policies: dict, n: int, n_boot: int, seed: int, fixed_range: tuple[float, float]) -> dict:
    """Paired bootstrap over questions. `policies` maps a name to f(idx) -> points on the resampled rows."""
    rng = np.random.default_rng(seed)
    aiqs = {k: [] for k in policies}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        for k, f in policies.items():
            aiqs[k].append(aiq(f(idx), *fixed_range))
    aiqs = {k: np.asarray(v) for k, v in aiqs.items()}
    ci = {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for k, v in aiqs.items()}
    return {"n": n_boot, "seed": seed, "aiq_ci95": ci, "aiqs": aiqs}


def diff(boot: dict, a: str, b: str) -> dict:
    d = boot["aiqs"][a] - boot["aiqs"][b]
    return {"ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], "p_le_0": float((d <= 0).mean())}


def load_tables(out, split: str, options: list[str], state_k: int, metric: str = "lenient", rescore_dir=None):
    """Question table (pool_answer.load_split) plus ok/cost/entropy/pred matrices over the common ids.

    `metric` is the correctness column: "lenient" from the answer files, or a column of the re-scored
    files in `rescore_dir/<split>/<cell>.parquet` (stair.rescore: "correct" from the LLM grader on the
    self-committed answer, "commit_em"). Then each option's cost includes its commit step (`commit_units`).
    """
    from pathlib import Path

    from stair.pool_answer import load_split

    out = Path(out)
    answers = {o: pd.read_parquet(out / "answers" / split / f"{o}.parquet").set_index("id") for o in options}
    ids = [i for i in pd.read_parquet(out / "pools" / f"{split}.parquet", columns=["id"])["id"]
           if all(i in a.index for a in answers.values())]
    df = load_split(out, split, state_k, None).set_index("id").loc[ids].reset_index()
    if metric == "lenient":
        ok = np.stack([answers[o].loc[ids, "lenient"].to_numpy() >= 0.5 for o in options], axis=1)
    else:
        if rescore_dir is None:
            raise ValueError(f"metric {metric!r} needs rescore_dir (the stair.rescore output)")
        graded = {o: pd.read_parquet(Path(rescore_dir) / split / f"{o}.parquet").set_index("id") for o in options}
        missing = [o for o in options if not set(ids) <= set(graded[o].index)]
        if missing:
            raise ValueError(f"re-scored answers missing for {missing} on {split}: run `rescore run --split {split}`")
        ok = np.stack([graded[o].loc[ids, metric].to_numpy() >= 0.5 for o in options], axis=1)
    cost = np.stack([answers[o].loc[ids, "units"].to_numpy() for o in options], axis=1)
    if metric != "lenient":  # the self-commit step (stair.rescore, version 4) is part of each option's cost
        cost = cost + np.stack([graded[o].loc[ids, "commit_units"].to_numpy() if "commit_units" in graded[o]
                                else np.zeros(len(ids)) for o in options], axis=1)
    df["entropy0"] = answers[options[0]].loc[ids, "entropy"].to_numpy()
    df["pred0"] = answers[options[0]].loc[ids, "pred"].to_numpy()
    return df, ok, cost
