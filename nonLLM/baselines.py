"""CPU routing baselines: train on train, select on validation, evaluate test once."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path
import re
import time

import joblib
import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

STAGES = {"ret": "ret_label", "qa_W": "qa_label_W", "qa_S": "qa_label_S"}
CELLS = ["WW", "WS", "SW", "SS"]
COSTS = np.array([2.0, 5.7, 5.7, 9.4])
STOP = set("the a an of in on at to for and or is are was were be been what which who when where how".split())


def context(row, stage):
    return row["pool_text"] if stage == "ret" else row["evidence_" + stage[-1]]


def text_features(row, stage):
    # No gold answers, gold passage annotations, judge scores, outcomes, or labels.
    return (row["question"] + "\n") * 3 + context(row, stage)


def numeric_features(row, stage):
    q, ctx = row["question"], context(row, stage)
    qw = re.findall(r"\w+", q.lower())
    cw = re.findall(r"\w+", ctx.lower())
    qs, cs = set(qw) - STOP, set(cw) - STOP
    inter = len(qs & cs)
    return [len(q), len(qw), len(qs), len(ctx), len(cw), len(cs),
            inter / max(1, len(qs)), inter / max(1, len(qs | cs)),
            len(cs) / max(1, len(cw)), ctx.count("\n"),
            len(re.findall(r"\[\d+\]", ctx)), len(re.findall(r"\d+", q)),
            int(q.lower().startswith("who")), int(q.lower().startswith("when")),
            int(q.lower().startswith("where")), int(q.lower().startswith("which")),
            int(q.lower().startswith("what")), int(q.lower().startswith("how")),
            int(" or " in q.lower()), int("both" in qs)]


def metrics(y, pred):
    return {"n": len(y), "label_counts": dict(Counter(y)),
            "accuracy": float(accuracy_score(y, pred)),
            "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, labels=["W", "S"], average="macro", zero_division=0)),
            "confusion_matrix_W_S": confusion_matrix(y, pred, labels=["W", "S"]).tolist(),
            "prediction_counts": dict(Counter(pred))}


def composed(rows, ids, ret, qa_w, qa_s):
    picks = [r + (w if r == "W" else s) for r, w, s in zip(ret, qa_w, qa_s)]
    cells = [CELLS.index(p) for p in picks]
    return {"n": len(ids), "picks": dict(Counter(picks)),
            "lenient_accuracy": float(np.mean([rows[i]["em_lenient"][c] for i, c in zip(ids, cells)])),
            "strict_accuracy": float(np.mean([rows[i]["em_strict"][c] for i, c in zip(ids, cells)])),
            "mean_pipeline_cost": float(np.mean(COSTS[cells])),
            "cost_pct_of_SS": float(100 * np.mean(COSTS[cells]) / COSTS[3])}


def candidates():
    return {
        "majority": [(None, DummyClassifier(strategy="most_frequent"))],
        "logreg": [(c, LogisticRegression(C=c, solver="liblinear", class_weight="balanced",
                                          max_iter=1000, random_state=707)) for c in [0.1, 1.0, 10.0]],
        "linear_svm": [(c, LinearSVC(C=c, class_weight="balanced", dual="auto",
                                     max_iter=10000, random_state=707)) for c in [0.1, 1.0, 10.0]],
        "multinomial_nb": [(alpha, MultinomialNB(alpha=alpha)) for alpha in [0.1, 1.0, 10.0]],
        "random_forest": [(depth, RandomForestClassifier(n_estimators=200, max_depth=depth,
                             min_samples_leaf=5, class_weight="balanced", n_jobs=2,
                             random_state=707)) for depth in [8, None]],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--splits", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-features", type=int, default=60000)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit("Use a fresh output directory to preserve completed experiments")
    a.out.mkdir(parents=True, exist_ok=True)
    splits_bytes = a.splits.read_bytes()
    sp = json.loads(splits_bytes)["splits"]
    assert all(not set(sp[x]) & set(sp[y]) for x, y in [("train", "val"), ("train", "test"), ("val", "test")])
    where = {i: s for s, ids in sp.items() for i in ids}
    rows = {}
    opener = gzip.open if a.data.suffix == ".gz" else open
    with opener(a.data, "rt") as f:
        for line in f:
            r = json.loads(line)
            assert r["idx"] not in rows and r["split"] == where[r["idx"]]
            rows[r["idx"]] = r
    assert set(rows) == set(where)
    results = {"config": {"data": str(a.data), "splits": str(a.splits),
                "split_sha256": hashlib.sha256(splits_bytes).hexdigest(), "seed": 707,
                "sklearn_version": sklearn.__version__, "max_features": a.max_features,
                "selection_metric": "validation balanced accuracy, then macro F1, then accuracy",
                "ret_input": "question + full candidate pool", "qa_input": "question + corresponding evidence",
                "label_source": "fields in supplied file; default experiment uses measured real labels",
                "cost_scope": "stored pipeline FLOP units; excludes CPU routing overhead",
                "unresolved_policy": "skip None for classification; include all rows for composed evaluation"},
               "stages": {}, "composed": {}, "fixed_policies": {}}
    bundles = {}
    # No test transform, prediction, metric, or outcome lookup during model selection.
    for stage, label in STAGES.items():
        tr = [i for i in sp["train"] if rows[i][label] in ("W", "S")]
        va = [i for i in sp["val"] if rows[i][label] in ("W", "S")]
        ytr = np.array([rows[i][label] for i in tr]); yva = np.array([rows[i][label] for i in va])
        print(f"[{stage}] labeled train={len(tr)} val={len(va)}; fitting TF-IDF", flush=True)
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.98,
                             max_features=a.max_features, sublinear_tf=True, dtype=np.float32)
        xtr = vec.fit_transform(text_features(rows[i], stage) for i in tr)
        xva = vec.transform(text_features(rows[i], stage) for i in va)
        ntr = np.array([numeric_features(rows[i], stage) for i in tr])
        nva = np.array([numeric_features(rows[i], stage) for i in va])
        chosen = {}; results["stages"][stage] = {}
        for family, options in candidates().items():
            tx, vx = (ntr, nva) if family == "random_forest" else (xtr, xva)
            attempts = []; best_key = None
            for param, model in options:
                start = time.perf_counter(); model.fit(tx, ytr)
                elapsed = time.perf_counter() - start
                score = metrics(yva, model.predict(vx))
                attempts.append({"parameter": param, "fit_seconds": elapsed, "validation": score})
                key = (score["balanced_accuracy"], score["macro_f1"], score["accuracy"])
                if best_key is None or key > best_key:
                    best_key = key; chosen[family] = model; selected = len(attempts) - 1
            results["stages"][stage][family] = {"selected_parameter": attempts[selected]["parameter"],
                    "validation": attempts[selected]["validation"], "candidates": attempts}
            print(f"[{stage}] {family} validation balanced_acc={best_key[0]:.4f}", flush=True)
        bundles[stage] = {"vectorizer": vec, "models": chosen}
        joblib.dump(bundles[stage], a.out / f"{stage}.joblib")
    print("All settings frozen. Starting the final test evaluation.", flush=True)
    predictions = {}
    for stage, label in STAGES.items():
        ids = sp["test"]
        x = bundles[stage]["vectorizer"].transform(text_features(rows[i], stage) for i in ids)
        numeric = np.array([numeric_features(rows[i], stage) for i in ids])
        mask = np.array([rows[i][label] in ("W", "S") for i in ids])
        y = np.array([rows[i][label] for i in ids], dtype=object)[mask]
        predictions[stage] = {}
        for family, model in bundles[stage]["models"].items():
            start = time.perf_counter()
            pred = model.predict(numeric if family == "random_forest" else x)
            results["stages"][stage][family]["test_prediction_seconds"] = time.perf_counter() - start
            predictions[stage][family] = pred
            results["stages"][stage][family]["test"] = metrics(y.tolist(), pred[mask].tolist())
    for family in predictions["ret"]:
        result = composed(rows, sp["test"], predictions["ret"][family],
                          predictions["qa_W"][family], predictions["qa_S"][family])
        results["composed"][family] = result
        print(f"[TEST {family}] {json.dumps(result)}", flush=True)
    for name in CELLS:
        results["fixed_policies"][name] = composed(rows, sp["test"],
            [name[0]] * len(sp["test"]), [name[1]] * len(sp["test"]), [name[1]] * len(sp["test"]))
    results["oracle_accuracy"] = {metric: float(np.mean([max(rows[i][metric]) for i in sp["test"]]))
                                 for metric in ["em_lenient", "em_strict"]}
    (a.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    with (a.out / "test_predictions.csv").open("w") as f:
        writer = csv.writer(f); writer.writerow(["idx", "stage", "family", "label", "prediction"])
        for stage, families in predictions.items():
            for family, preds in families.items():
                for i, p in zip(sp["test"], preds):
                    writer.writerow([i, stage, family, rows[i][STAGES[stage]], p])
    print(f"Results saved: {a.out}", flush=True)


if __name__ == "__main__":
    main()
