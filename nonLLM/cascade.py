"""Config A cascade with CPU classifiers; lock on validation before test replay."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score

from baselines import candidates, numeric_features


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def evidence(text):
    # Same per-passage 600-character cap as router_prompts.format_evidence.
    parts = re.split(r"(\[\d+\] )", text)
    return parts[0] + "".join(parts[i] + parts[i + 1][:600]
                             for i in range(1, len(parts), 2))


def view(g, stage, branch, answer=None):
    ctx = evidence(g["evidence_W" if stage == "ret" else "evidence_" + branch])
    text = (g["question"] + "\n") * 3 + ctx
    if answer is not None:
        text += "\nWeak answer: " + answer
    row = {"question": g["question"], "evidence_W": ctx}
    nums = numeric_features(row, "qa_W")
    if stage == "qa" and answer is not None:
        words = set(re.findall(r"\w+", answer.lower()))
        ctxwords = set(re.findall(r"\w+", ctx.lower()))
        nums += [len(answer), len(words), len(words & ctxwords) / max(1, len(words)),
                 int(answer.strip().lower() in ("yes", "no"))]
    return text, nums


def pweak(model, x):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x)[:, list(model.classes_).index("W")]
    # LinearSVC: threshold its margin directly; no probability claim.
    z = model.decision_function(x)
    return z if model.classes_[1] == "W" else -z


def costs(path):
    if path:
        c = json.load(open(path))["gate_eval_constants"]
        vals = [c[k] for k in ("COST_RET_W", "COST_RET_S", "COST_QA_W_GIVEN_W",
                               "COST_QA_W_GIVEN_S", "COST_QA_S_GIVEN_W", "COST_QA_S_GIVEN_S")]
        scope = "Six mean token-aware call costs from token_costs.json; excludes routing overhead"
    else:
        vals = [1., 4.7, 1., 1., 4.7, 4.7]
        scope = "Config A flat parameter-ratio proxy, NOT measured token costs; excludes routing overhead"
    rw, rs, qww, qws, qsw, qss = vals
    cascade = np.array([rw + qww, rw + qww + qsw,
                        rw + rs + qws, rw + rs + qws + qss])
    return cascade, rs + qss, scope


def sweep(em, pr, qw, qs, prices, fraction=.95):
    """Enumerate all distinct decision pairs including tied scores and extremes."""
    n = len(em)
    bar = fraction * em[:, 3].mean()
    best = None
    frontier_candidates = []
    for tr in np.r_[np.unique(pr), np.inf]:
        r = pr < tr
        qp = np.where(r, qs, qw)
        aw = em[np.arange(n), r.astype(int) * 2]
        ass = em[np.arange(n), r.astype(int) * 2 + 1]
        base = np.where(r, prices[2], prices[0])
        extra = np.where(r, prices[3] - prices[2], prices[1] - prices[0])
        order = np.argsort(-qp)
        sortedq = qp[order]
        ks = np.r_[0, np.flatnonzero(np.r_[sortedq[:-1] != sortedq[1:], True]) + 1]
        ca = np.r_[0, np.cumsum(aw[order] - ass[order])]
        cc = np.r_[0, np.cumsum(extra[order])]
        acc = (ass.sum() + ca[ks]) / n
        cost = (base.sum() + extra.sum() - cc[ks]) / n
        tq = np.r_[np.inf, sortedq[ks[1:] - 1]]
        good = np.flatnonzero(acc >= bar - 1e-12)
        if len(good):
            j = min(good, key=lambda j: (cost[j], -acc[j]))
            rec = dict(cost=float(cost[j]), accuracy=float(acc[j]),
                       theta_ret=float(tr), theta_qa=float(tq[j]))
            if best is None or (rec["cost"], -rec["accuracy"]) < (best["cost"], -best["accuracy"]):
                best = rec
        frontier_candidates.extend(zip(cost.tolist(), acc.tolist(),
                                       [float(tr)] * len(ks), tq.tolist()))
    front = []
    high = -1
    for row in sorted(frontier_candidates, key=lambda r: (r[0], -r[1])):
        if row[1] > high:
            front.append(row)
            high = row[1]
    if best is None:
        raise ValueError("95%-of-SS validation bar unreachable")
    return best, front


def compose(grid, pr, qw, qs, lock, prices, direct_ss):
    r = pr < lock["theta_ret"]
    q = np.where(r, qs, qw) < lock["theta_qa"]
    cells = 2 * r.astype(int) + q.astype(int)
    em = np.array([g["em_lenient"] for g in grid])
    strict = np.array([g["em_strict"] for g in grid])
    mean = float(prices[cells].mean())
    return {"n": len(grid), "lenient_accuracy": float(em[np.arange(len(grid)), cells].mean()),
            "strict_accuracy": float(strict[np.arange(len(grid)), cells].mean()),
            "mean_cost": mean, "cost_pct_of_direct_SS": 100 * mean / direct_ss,
            "cost_pct_of_cascade_SS": 100 * mean / prices[3],
            "picks": dict(Counter(str(c) for c in np.array(["WW", "WS", "SW", "SS"])[cells]))}, cells


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--labels-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--qa-weak", type=Path, help="Config A qa_weak.jsonl k/v cache")
    ap.add_argument("--token-costs", type=Path, help="Config A token_costs.json")
    ap.add_argument("--without-weak-answer", action="store_true", help="Explicit input ablation")
    ap.add_argument("--flat-costs", action="store_true", help="Explicit unmeasured cost proxy")
    ap.add_argument("--max-features", type=int, default=60000)
    a = ap.parse_args()
    if bool(a.qa_weak) == a.without_weak_answer:
        ap.error("Choose --qa-weak FILE or --without-weak-answer")
    if bool(a.token_costs) == a.flat_costs:
        ap.error("Choose --token-costs FILE or --flat-costs")
    if a.out.exists() and any(a.out.iterdir()):
        ap.error("Use a fresh output directory; completed runs are never overwritten")
    files = {"grid": a.labels_root / "full_run_p100/router_data_lenient.jsonl",
             "ret": a.labels_root / "ret_labels/ret_labels.jsonl",
             "qa": a.labels_root / "qa_labels/qa_labels.jsonl"}
    grid = {g["idx"]: g for g in load(files["grid"])}
    labels = {s: load(files[s]) for s in ("ret", "qa")}
    answers = {r["k"]: str(r["v"]) for r in load(a.qa_weak)} if a.qa_weak else None
    if answers is not None:
        required = {f"{i}|{suffix}" for i in grid for suffix in ("weak", "strong")}
        missing = required - answers.keys()
        if missing:
            ap.error(f"Incomplete weak-answer cache: {len(missing)} missing keys")
    assert len(labels["ret"]) == len(grid) and len(labels["qa"]) == 2 * len(grid)
    assert len({r["idx"] for r in labels["ret"]}) == len(grid)
    assert len({(r["idx"], r["ret_choice"]) for r in labels["qa"]}) == 2 * len(grid)
    for stage in labels:
        for r in labels[stage]:
            assert r["split"] == grid[r["idx"]]["split"]
            assert r[stage + "_target"] in ("W", "S")
    prices, direct_ss, cost_scope = costs(a.token_costs)
    a.out.mkdir(parents=True, exist_ok=True)
    config = {"experiment": "nonLLM Config A cascade",
              "weak_answer_input": answers is not None, "cost_scope": cost_scope,
              "file_sha256": {k: fingerprint(p) for k, p in files.items()},
              "selection": "Stage hyperparameters by validation AUC; family and two thresholds by cheapest >=95% SS validation accuracy",
              "ret_labels": "Provided answer-coverage labels; X mapped to W",
              "qa_labels": "Provided outcome labels; X mapped to W",
              "qa_classifier": "One shared classifier over both retrieval branches",
              "passage_char_cap": 600, "test_status": "Previously used split; new frozen replay, not a first-ever sealed test"}
    if a.qa_weak:
        config["qa_weak_sha256"] = fingerprint(a.qa_weak)
    if a.token_costs:
        config["token_costs_sha256"] = fingerprint(a.token_costs)
    (a.out / "config.json").write_text(json.dumps(config, indent=2))
    datasets, bundles, val_scores, stage_metrics = {}, {}, {}, {}
    t0 = time.perf_counter()
    for stage in ("ret", "qa"):
        datasets[stage] = {}
        for split in ("train", "val"):  # Test features are built only after locking.
            rows = [r for r in labels[stage] if r["split"] == split]
            texts, nums = [], []
            for r in rows:
                branch = r.get("ret_choice", "W")
                ans = answers[f"{r['idx']}|{'weak' if branch == 'W' else 'strong'}"] if answers is not None and stage == "qa" else None
                text, num = view(grid[r["idx"]], stage, branch, ans)
                texts.append(text); nums.append(num)
            datasets[stage][split] = (rows, texts, np.array(nums),
                                      np.array([r[stage + "_target"] for r in rows]))
        tr, va = datasets[stage]["train"], datasets[stage]["val"]
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=.98,
                              max_features=a.max_features, sublinear_tf=True, dtype=np.float32)
        xt = vec.fit_transform(tr[1]); xv = vec.transform(va[1])
        bundles[stage], val_scores[stage], stage_metrics[stage] = {}, {}, {}
        for family, opts in candidates().items():
            tx, vx = (tr[2], va[2]) if family == "random_forest" else (xt, xv)
            best = None
            for param, model in opts:
                model.fit(tx, tr[3]); scores = pweak(model, vx)
                auc = float(roc_auc_score(va[3] == "W", scores))
                if best is None or auc > best[0]:
                    best = auc, model, scores, param
            auc, model, scores, param = best
            bundles[stage][family] = (vec, model)
            val_scores[stage][family] = scores
            stage_metrics[stage][family] = {"val_auc": auc, "parameter": param}
            print(f"{stage} {family}: validation AUC={auc:.4f}", flush=True)
    ids = sorted(i for i, g in grid.items() if g["split"] == "val")
    vr = datasets["ret"]["val"][0]; vq = datasets["qa"]["val"][0]
    em = np.array([grid[i]["em_lenient"] for i in ids])
    locks = {}
    for family in val_scores["ret"]:
        rp = {r["idx"]: p for r, p in zip(vr, val_scores["ret"][family])}
        qp = {(r["idx"], r["ret_choice"]): p for r, p in zip(vq, val_scores["qa"][family])}
        locks[family], front = sweep(em, np.array([rp[i] for i in ids]),
                                    np.array([qp[i, "W"] for i in ids]),
                                    np.array([qp[i, "S"] for i in ids]), prices)
        replay, _ = compose([grid[i] for i in ids], np.array([rp[i] for i in ids]),
                            np.array([qp[i, "W"] for i in ids]),
                            np.array([qp[i, "S"] for i in ids]),
                            locks[family], prices, direct_ss)
        assert np.isclose(replay["lenient_accuracy"], locks[family]["accuracy"])
        assert np.isclose(replay["mean_cost"], locks[family]["cost"])
        with (a.out / f"frontier_{family}.csv").open("w") as f:
            writer = csv.writer(f); writer.writerow(["cost", "accuracy", "theta_ret", "theta_qa"]); writer.writerows(front)
        print(f"{family} locked: {locks[family]}", flush=True)
    chosen = min(locks, key=lambda f: (locks[f]["cost"], -locks[f]["accuracy"], f))
    lock = {"selected_family": chosen, "rule": config["selection"], "bar": float(.95 * em[:, 3].mean()),
            "families": locks, "stage_metrics": stage_metrics}
    # JSON uses Infinity for all-strong sentinel, matching Python's JSON reader.
    (a.out / "theta_lock.json").write_text(json.dumps(lock, indent=2))
    for stage in bundles:
        joblib.dump({"vectorizer": bundles[stage][chosen][0], "model": bundles[stage][chosen][1],
                     "family": chosen, "config": config}, a.out / f"{stage}.joblib")
    testids = sorted(i for i, g in grid.items() if g["split"] == "test")
    predicted = {}
    for stage in ("ret", "qa"):
        rows = [r for r in labels[stage] if r["split"] == "test"]
        views = []
        for r in rows:
            branch = r.get("ret_choice", "W")
            ans = answers[f"{r['idx']}|{'weak' if branch == 'W' else 'strong'}"] if answers is not None and stage == "qa" else None
            views.append(view(grid[r["idx"]], stage, branch, ans))
        vec, model = bundles[stage][chosen]
        x = np.array([v[1] for v in views]) if chosen == "random_forest" else vec.transform([v[0] for v in views])
        predicted[stage] = {(r["idx"], r.get("ret_choice", "W")): p for r, p in zip(rows, pweak(model, x))}
    pr = np.array([predicted["ret"][i, "W"] for i in testids])
    qw = np.array([predicted["qa"][i, "W"] for i in testids])
    qs = np.array([predicted["qa"][i, "S"] for i in testids])
    test, cells = compose([grid[i] for i in testids], pr, qw, qs, locks[chosen], prices, direct_ss)
    fixed = {c: {"accuracy": float(np.mean([grid[i]["em_lenient"][j] for i in testids])),
                 "cascade_cost": float(prices[j])} for j, c in enumerate(("WW", "WS", "SW", "SS"))}
    result = {"config": config, "lock": lock, "test": test, "fixed_test": fixed,
              "direct_SS_cost": direct_ss, "seconds": time.perf_counter() - t0}
    (a.out / "results.json").write_text(json.dumps(result, indent=2))
    with (a.out / "test_predictions.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["idx", "ret_score_W", "qa_W_score_W", "qa_S_score_W", "pick"])
        writer.writerows((i, r, w, s, ["WW", "WS", "SW", "SS"][c]) for i, r, w, s, c in zip(testids, pr, qw, qs, cells))
    note = "Full weak-answer input" if answers is not None else "ABLATION: weak-answer input absent"
    report = f"# Non-LLM Config A cascade\n\n{note}. {cost_scope}.\n\nSelected on validation: {chosen}.\n\nTest n={test['n']}: lenient {100*test['lenient_accuracy']:.2f}%, strict {100*test['strict_accuracy']:.2f}%.\n\nCost: {test['cost_pct_of_direct_SS']:.2f}% of direct SS; {test['cost_pct_of_cascade_SS']:.2f}% of cascade SS.\n\nDirect SS test accuracy: {100*fixed['SS']['accuracy']:.2f}%.\n\nPicks: {test['picks']}.\n\nThis replays recorded Config A outcomes; it does not generate new model answers. Test was used in earlier experiments; this run locks choices before its new test evaluation.\n"
    (a.out / "REPORT.md").write_text(report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
