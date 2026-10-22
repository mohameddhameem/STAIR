"""Experiment 10: the System One judge routing menu A on the HotpotQA golden set (Jonathan's Stage-Level Routing).

The head-to-head with his SLR routers, on his terms (`stair.slr`): his Config B grid (every question has all four paths),
his labels, his split (train on slr-train, choose thresholds on slr-val, report once on slr-test), his weak-first Table 5
prices and his two threshold rules (peak, parity). Only the router differs.

The judge is experiment 7's typed-decision cross-encoder (`judge.make_judge`, `judge.fine_tune`): nano-jev, or a Strands
Decider (experiment 9). One model answers two typed yes/no questions, one per stage, so P(yes) is P(the weak stage
suffices), his routers' pW:

    retrieval  the weak retriever's (Qwen2.5-7B) selected passages: do they hold what the answer needs?
               label: his ret_target (W -> yes)
    answer     under the chosen retrieval, that branch's passages (qp), or with the weak answer under it too (qpa):
               will the weak answer do? label: his qa_target for that branch (W -> yes)

Weak-first, as he prices it: the weak answer runs before the answer decision, so qpa reads it at no extra price; the
retrieval decision reads only the weak selection, as his does. `slr.route` turns the three scores into a path.

    exp10-slr train --judge nano-jev --plan configs/exp10-slr.toml   # one judge set, every (inputs, seed); resumes
    exp10-slr eval --plan configs/exp10-slr.toml                     # every judge set trained so far, his routers

Eval: for each router (each seed of each judge, the seeds' mean, and his two routers) the 9,261-point threshold sweep on
slr-val picks the peak and parity points, which are applied to slr-test. AIQ is over the test points of the val-Pareto
thresholds, at weak-first prices from 80 (all WW) to 1,009 (all SS), with paired bootstrap intervals. Router cost: none,
as in his rows; and, beside it, each judge call priced with `stair_common.cost.budget` at its measured tokens
(nano-jev's calls price at 0: they are cheaper than one 3B sample).
"""

import argparse
import json
import logging
import time
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from stair_common.cost import SLR_PATH_PRICES, budget
from stair_common.splits import load_split

from stair import exp8_eval as ev
from stair import provenance, slr
from stair.evaluate_routing import aiq
from stair.experts import clean_answer
from stair.judge import fine_tune, judge_params_b, make_judge

log = logging.getLogger(__name__)

QUESTIONS = {
    "ret": "Do these passages hold what is needed to answer this question: {question}",
    "qp": "Will a small model answer this question correctly from these passages: {question}",
    "qpa": "Is the proposed answer correct? Question: {question}",
}
OPTIONS = ("yes", "no")  # option 0 = the weak stage suffices
SCORES = ["ret", "qa_W", "qa_S"]  # P(yes) per question: retrieval, answer under weak / strong retrieval
PRICE_RANGE = (float(SLR_PATH_PRICES["WW"]), float(SLR_PATH_PRICES["SS"]))
PRICES = np.array([SLR_PATH_PRICES[p] for p in slr.PATHS], dtype=float)


# ---------------------------------------------------------------------------------------------
# Data


def selection(text: str) -> str:
    """A retriever's output without its "Query:" line: the selected references, "[k] title: text" per line."""
    return text.split("Selected references:", 1)[-1].strip() or "(no passages selected)"


def load(plan: dict) -> pd.DataFrame:
    """The golden-set grid (slr.load_grid) plus what the judge reads and learns: the question, each retriever's
    selection, his labels."""
    slr_plan = tomllib.loads(Path(plan["slr_plan"]).read_text(encoding="utf-8"))
    df = slr.load_grid(slr_plan)
    root = Path(slr_plan["out"]) / slr.GRID
    df["question"] = pd.read_parquet(slr_plan["dev"], columns=["question"])["question"].to_numpy()
    for x in "WS":
        sel = {r["k"]: r["v"] for r in slr.read_jsonl(root / f"retr_{slr.BRANCH[x]}.jsonl")}
        df[f"sel_{x}"] = [selection(str(sel[str(i)])) for i in df["idx"]]
    ret = {r["idx"]: r["ret_target"] for r in slr.read_jsonl(root / "ret_labels" / "ret_labels.jsonl")}
    qa = {(r["idx"], r["ret_choice"]): r["qa_target"] for r in slr.read_jsonl(root / "qa_labels" / "qa_labels.jsonl")}
    df["y_ret"] = [ret.get(i) for i in df["idx"]]
    for x in "WS":
        df[f"y_qa_{x}"] = [qa.get((i, x)) for i in df["idx"]]
    return df


def items(df: pd.DataFrame, inputs: str) -> dict[str, list[dict]]:
    """Per score (ret, qa_W, qa_S), one typed item per question."""
    out = {"ret": [{"question": QUESTIONS["ret"].format(question=q), "options": OPTIONS, "state": s}
                   for q, s in zip(df["question"], df["sel_W"])]}
    for x in "WS":
        weak_answer = df[f"answer_{x}W"]  # the weak answer under retrieval x: path xW
        states = (df[f"sel_{x}"] if inputs == "qp" else
                  [f"Proposed answer: {clean_answer(a)[:200]}\n{s}" for a, s in zip(weak_answer, df[f"sel_{x}"])])
        out[f"qa_{x}"] = [{"question": QUESTIONS[inputs].format(question=q), "options": OPTIONS, "state": s}
                          for q, s in zip(df["question"], states)]
    return out


def labels(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """0 (yes: weak suffices) or 1, per score; his targets, which map every question to W or S."""
    out = {"ret": df["y_ret"], **{f"qa_{x}": df[f"y_qa_{x}"] for x in "WS"}}
    if any(v.isna().any() or not v.isin(["W", "S"]).all() for v in out.values()):
        raise ValueError("training questions need his W/S targets for both stages")
    return {k: (v == "S").astype(int).to_numpy() for k, v in out.items()}


# ---------------------------------------------------------------------------------------------
# Training


def judge_set(plan: dict, name: str) -> dict:
    return next(j for j in plan["judges"] if j["name"] == name)


def run_train(df: pd.DataFrame, plan: dict, spec: dict, inputs: str, seed: int, device: str | None) -> None:
    target = Path(plan["out"]) / spec["judge_dir"] / inputs / f"seed{seed}"
    if (target / "done.json").exists():
        log.info(f"{spec['name']} {inputs} seed {seed}: done")
        return
    hp = spec["train"]
    train = df[df["split"] == "train"].reset_index(drop=True)
    rng = np.random.default_rng(seed)
    dev_q = rng.random(len(train)) < hp["dev_fraction"]  # held out by question, so a question's three items stay together
    its, ys = items(train, inputs), labels(train)
    pick = lambda mask: ([it for k in SCORES for it, m in zip(its[k], mask) if m],  # noqa: E731
                         np.concatenate([ys[k][mask] for k in SCORES]))
    (tr_items, tr_y), (dev_items, dev_y) = pick(~dev_q), pick(dev_q)
    log.info(f"{spec['name']} {inputs} seed {seed}: {len(tr_items)} items ({int(tr_y.sum())} escalate), "
             f"{len(dev_items)} dev items")
    judge = make_judge(spec["init"], device, hp.get("max_length", 512), spec.get("score_batch"))
    start = time.perf_counter()
    history = fine_tune(judge, tr_items, tr_y, dev_items, dev_y, hp["epochs"], hp["lr"], hp["weight_decay"],
                        hp["warmup"], hp["batch_size"], seed, hp.get("micro_batch"))
    train_seconds = time.perf_counter() - start
    target.mkdir(parents=True, exist_ok=True)
    for split in ("val", "test"):
        part = df[df["split"] == split].reset_index(drop=True)
        cols = {"idx": part["idx"]}
        for k, its_k in items(part, inputs).items():
            p, tokens = judge.probabilities(its_k)
            cols[k], cols[f"tokens_{k}"] = p[:, 0], tokens
        pd.DataFrame(cols).to_parquet(target / f"{split}.parquet", index=False)
    (target / "done.json").write_text(json.dumps({
        "judge": spec["name"], "inputs": inputs, "seed": seed, "init": spec["init"], "train": hp,
        "n_train_items": len(tr_items), "n_dev_items": len(dev_items), "history": history, "train_seconds": train_seconds,
        "init_revision": provenance.hub_revision(spec["init"]), "device": judge.device}, indent=1))


# ---------------------------------------------------------------------------------------------
# Evaluation


class Router:
    """One router's scores on val and test, and its judge's price per question (0 for his routers)."""

    def __init__(self, name: str, val: dict, test: dict, judge_price_test: np.ndarray | None = None):
        self.name, self.val, self.test = name, val, test
        self.judge_price = judge_price_test  # [questions, 3]: the ret call, the qa call under W, under S

    def choice(self, scores: dict, t) -> np.ndarray:
        return slr.route(scores["ret"], {"W": scores["qa_W"], "S": scores["qa_S"]}, t)

    def extra(self, choice: np.ndarray) -> np.ndarray:
        """The judge's own price: the retrieval call, and the answer call on the branch it took."""
        if self.judge_price is None:
            return np.zeros(len(choice))
        return self.judge_price[:, 0] + np.where(choice < 2, self.judge_price[:, 1], self.judge_price[:, 2])


def judge_routers(plan: dict, test_ids: pd.Series, val_ids: pd.Series) -> list[Router]:
    """Every trained (judge set, inputs): each seed, and the mean of the seeds (which pays every seed's calls)."""
    routers = []
    for spec in plan["judges"]:
        params = judge_params_b(spec["init"])
        for inputs in spec["inputs"]:
            root = Path(plan["out"]) / spec["judge_dir"] / inputs
            seeds = [s for s in spec["seeds"] if (root / f"seed{s}" / "done.json").exists()]
            if not seeds:
                log.info(f"{spec['name']} {inputs}: not trained yet")
                continue
            per = {}
            for s in seeds:
                val = pd.read_parquet(root / f"seed{s}" / "val.parquet").set_index("idx").loc[val_ids]
                test = pd.read_parquet(root / f"seed{s}" / "test.parquet").set_index("idx").loc[test_ids]
                price = np.stack([[budget(params, int(t), 0) for t in test[f"tokens_{k}"]] for k in SCORES], axis=1)
                per[s] = ({k: val[k].to_numpy() for k in SCORES}, {k: test[k].to_numpy() for k in SCORES}, price)
                routers.append(Router(f"{spec['name']} {inputs}#seed{s}", *per[s]))
            if len(seeds) > 1:
                mean = lambda i: {k: np.mean([per[s][i][k] for s in seeds], axis=0) for k in SCORES}  # noqa: E731
                routers.append(Router(f"{spec['name']} {inputs}", mean(0), mean(1), sum(per[s][2] for s in seeds)))
    return routers


def his_routers(plan: dict, slr_plan: dict, val_ids: pd.Series, test_ids: pd.Series) -> list[Router]:
    out = []
    for row in plan["his_routers"]:
        scores = {}
        for split, names, ids in (("val", row["val"], val_ids), ("test", row["test"], test_ids)):
            ret, qa = slr.his_scores(slr_plan, *names, ids, split=split)
            scores[split] = {"ret": ret, "qa_W": qa["W"], "qa_S": qa["S"]}
        out.append(Router(row["name"], scores["val"], scores["test"]))
    return out


def pareto(points: list[tuple]) -> list[tuple]:
    """The thresholds of the val points no other point beats on both accuracy and price."""
    best, out = -1.0, []
    for acc, cost, *t in sorted(points, key=lambda p: (p[1], -p[0])):
        if acc > best:
            best = acc
            out.append(tuple(t))
    return out


def evaluate(df: pd.DataFrame, routers: list[Router], plan: dict, keep: np.ndarray | None = None) -> dict:
    """Peak and parity points (thresholds from slr-val) and AIQ on slr-test; `keep` restricts the test questions."""
    val, test = df[df["split"] == "val"].reset_index(drop=True), df[df["split"] == "test"].reset_index(drop=True)
    keep = np.ones(len(test), dtype=bool) if keep is None else keep
    ok_val, ok_test = slr.ok_matrix(val), slr.ok_matrix(test)[keep]
    strong = int(val["ok_SS"].sum()) / len(val)
    n, rows = int(keep.sum()), np.arange(int(keep.sum()))
    report = {"n_test": n, "price_range": PRICE_RANGE, "routers": {}}
    curves = {"fixed paths": lambda ix: [{"f1": float(ok_test[ix, k].mean()), "cost": slr.direct_price(p)}
                                         for k, p in enumerate(slr.PATHS)]}
    for r in routers:
        points = slr.sweep(ok_val, r.val["ret"], {"W": r.val["qa_W"], "S": r.val["qa_S"]})
        chosen = slr.select(points, strong)
        frontier = pareto(points)
        full = np.stack([r.choice(r.test, t) for t in frontier])  # [thresholds, test questions]
        choices, extras = full[:, keep], np.stack([r.extra(c) for c in full])[:, keep]
        entry = {}
        for rule, t in chosen.items():
            if t is None:
                entry[rule] = None
                continue
            c = r.choice(r.test, t)
            entry[rule] = {"t": t, "lenient": float(ok_test[rows, c[keep]].mean()), "price": float(PRICES[c[keep]].mean()),
                           "judge_price": float(r.extra(c)[keep].mean()),
                           "paths": {p: int((c[keep] == k).sum()) for k, p in enumerate(slr.PATHS)}}
        for tag, add in (("", False), (" +judge", True)):
            curves[r.name + tag] = (lambda ix, ch=choices, ex=extras, a=add:
                                    [{"f1": float(ok_test[ix, c[ix]].mean()), "cost": float((PRICES[c[ix]] + a * e[ix]).mean())}
                                     for c, e in zip(ch, ex)])
        entry["aiq"] = aiq(curves[r.name](rows), *PRICE_RANGE)
        entry["aiq_with_judge_price"] = aiq(curves[r.name + " +judge"](rows), *PRICE_RANGE)
        entry["frontier_size"] = len(frontier)
        report["routers"][r.name] = entry
    report["fixed_paths"] = {p: {"lenient": float(ok_test[:, k].mean()), "price": slr.direct_price(p)}
                             for k, p in enumerate(slr.PATHS)}
    choice = ev.cheapest_correct_label(ok_test, np.tile([slr.direct_price(p) for p in slr.PATHS], (n, 1)))
    report["oracle"] = {"lenient": float(ok_test[rows, choice].mean()),
                        "price": float(np.mean([slr.direct_price(slr.PATHS[c]) for c in choice]))}
    report["aiq_fixed_paths"] = aiq(curves["fixed paths"](rows), *PRICE_RANGE)
    boot = ev.bootstrap(curves, n, plan.get("bootstrap", 1000), 0, PRICE_RANGE)
    report["aiq_ci95"] = boot["aiq_ci95"]
    theirs = [r.name for r in routers if r.name.startswith("SLR ")] + ["fixed paths"]
    ours = [r.name for r in routers if not r.name.startswith("SLR ")]
    report["aiq_diff"] = {f"{a} - {b}": ev.diff(boot, a, b) for a in ours for b in theirs}
    report["aiq_diff_with_judge_price"] = {f"{a} +judge - {b}": ev.diff(boot, a + " +judge", b)
                                           for a in ours for b in theirs}
    return report


def run_eval(df: pd.DataFrame, plan: dict) -> dict:
    slr_plan = tomllib.loads(Path(plan["slr_plan"]).read_text(encoding="utf-8"))
    val_ids, test_ids = (df.loc[df["split"] == s, "idx"].reset_index(drop=True) for s in ("val", "test"))
    routers = his_routers(plan, slr_plan, val_ids, test_ids) + judge_routers(plan, test_ids, val_ids)
    out = Path(plan["out"]) / "eval"
    out.mkdir(parents=True, exist_ok=True)
    seen = set(load_split("hotpotqa", "nanojev_v1_seen"))
    clean = ~df.loc[df["split"] == "test", "id"].isin(seen).to_numpy()
    reports = {}
    for tag, keep in (("", None), ("-clean", clean)):
        reports[tag] = evaluate(df, routers, plan, keep)
        (out / f"report{tag}.json").write_text(json.dumps(reports[tag], indent=1) + "\n", encoding="utf-8")
    for name, e in reports[""]["routers"].items():
        pk, pa = e["peak"], e["parity"]
        log.info(f"{name:34s} AIQ {e['aiq']:.4f} | peak {100 * pk['lenient']:.2f} at {pk['price']:.1f}"
                 + (f" | parity {100 * pa['lenient']:.2f} at {pa['price']:.1f}" if pa else " | parity: none"))
    return reports[""]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("step", choices=["train", "eval"])
    parser.add_argument("--plan", type=Path, default=Path("configs/exp10-slr.toml"))
    parser.add_argument("--judge", help="train: the judge set (a [[judges]] name)")
    parser.add_argument("--device", help="cuda or cpu (default: cuda if available)")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    plan = tomllib.loads(args.plan.read_text(encoding="utf-8"))
    df = load(plan)
    out = Path(plan["out"])
    provenance.record(out, f"exp10-slr {args.step}", plan=str(args.plan), judge=args.judge, **provenance.environment())
    if args.step == "train":
        if not args.judge:
            raise SystemExit("train needs --judge <name>")
        spec = judge_set(plan, args.judge)
        for inputs in spec["inputs"]:
            for seed in spec["seeds"]:
                run_train(df, plan, spec, inputs, seed, args.device)
    else:
        run_eval(df, plan)


if __name__ == "__main__":
    main()
