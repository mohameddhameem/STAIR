"""Render the completed experiment; does not retrain or select models on test."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

NAMES = {"majority": "Majority", "logreg": "Logistic regression", "linear_svm": "Linear SVM",
         "multinomial_nb": "Multinomial NB", "random_forest": "Random forest"}


def frontier(policies, budget):
    points = [(r["mean_pipeline_cost"], r["lenient_accuracy"]) for r in policies.values()]
    values = [acc for cost, acc in points if cost <= budget + 1e-9]
    for c1, a1 in points:
        for c2, a2 in points:
            if c1 < c2 and c1 <= budget <= c2:
                values.append(a1 + (a2 - a1) * (budget - c1) / (c2 - c1))
    return max(values)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("run", type=Path)
    a = ap.parse_args()
    r = json.loads((a.run / "results.json").read_text())
    lines = ["# Non-LLM routing baselines: measured labels", "",
             "Fixed split: train 4,814 / validation 1,110 / test 1,481.",
             "TF-IDF and classifiers fit only on labeled training examples; None labels are skipped.",
             "Hyperparameters selected by validation balanced accuracy, then macro F1, then accuracy.",
             "All models were frozen before the final test pass. Full composed scores include unresolved rows.", "",
             "TF-IDF word unigrams/bigrams (up to 60,000 features per stage) feed logistic regression,",
             "linear SVM and multinomial NB. Random forest uses 20 cheap numerical text features.",
             "Retrieval inputs: question and candidate pool. QA inputs: question and the corresponding evidence.",
             "Gold annotations, outcomes and judge scores are not prediction features.", ""]
    for stage, families in r["stages"].items():
        lines += [f"## {stage}", "", "Only labeled test rows are included in the classification metrics.", "",
                  "| Classifier | Test N | Accuracy % | Balanced accuracy % | Macro F1 % | W recall % | S recall % |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for family, info in families.items():
            t = info["test"]; cm = t["confusion_matrix_W_S"]
            wr = cm[0][0] / max(1, sum(cm[0])); sr = cm[1][1] / max(1, sum(cm[1]))
            lines.append(f"| {NAMES[family]} | {t['n']} | {100*t['accuracy']:.2f} | "
                         f"{100*t['balanced_accuracy']:.2f} | {100*t['macro_f1']:.2f} | {100*wr:.2f} | {100*sr:.2f} |")
        lines += [""]
    lines += ["## Composed pipeline: all 1,481 test questions", "",
              "| Classifier | Lenient % | Strict % | Cost % of SS | Gain over fixed-policy mixture (pp) | WW / WS / SW / SS |",
              "|---|---:|---:|---:|---:|---|"]
    gains = {}
    for family, t in r["composed"].items():
        gain = 100 * (t["lenient_accuracy"] - frontier(r["fixed_policies"], t["mean_pipeline_cost"]))
        gains[family] = gain
        picks = " / ".join(str(t["picks"].get(p, 0)) for p in ["WW", "WS", "SW", "SS"])
        lines.append(f"| {NAMES[family]} | {100*t['lenient_accuracy']:.2f} | {100*t['strict_accuracy']:.2f} | "
                     f"{t['cost_pct_of_SS']:.2f} | {gain:+.2f} | {picks} |")
    lines += ["", "The frontier is the best analytical mixture of fixed policies at or below the router's cost.",
              "It gives a cost-matched baseline and does not use per-question routing information.", "",
              "| Fixed policy | Lenient % | Strict % | Cost % of SS |", "|---|---:|---:|---:|"]
    for name, t in r["fixed_policies"].items():
        lines.append(f"| {name} | {100*t['lenient_accuracy']:.2f} | {100*t['strict_accuracy']:.2f} | {t['cost_pct_of_SS']:.2f} |")
    lines += ["", f"Measured lenient oracle: {100*r['oracle_accuracy']['em_lenient']:.2f}%.", "",
              "Costs are the original pipeline FLOP units (2.0 / 5.7 / 5.7 / 9.4); CPU router overhead is excluded.",
              "This evaluates choices against the existing measured grid; it does not run the new GGUF models.",
              "Scores apply to this fixed split. They do not establish statistical significance or performance on other datasets.", "",
              "![Baseline comparison](baseline_comparison.png)", ""]
    (a.run / "REPORT.md").write_text("\n".join(lines))
    (a.run / "frontier_gains.json").write_text(json.dumps(gains, indent=2) + "\n")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    families = list(r["composed"]); stages = list(r["stages"]); x = np.arange(len(stages))
    for j, family in enumerate(families):
        values = [100*r["stages"][s][family]["test"]["balanced_accuracy"] for s in stages]
        axes[0].bar(x + (j-2)*0.15, values, width=0.15, label=NAMES[family])
    axes[0].set_xticks(x, ["Retrieval", "QA after W", "QA after S"])
    axes[0].set_ylim(0, 100); axes[0].axhline(50, color="gray", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Test balanced accuracy (%)"); axes[0].legend(fontsize=8)
    axes[0].set_title("Stage classification (labeled rows)")
    grid = np.linspace(2, 9.4, 100)
    axes[1].plot(100*grid/9.4, [100*frontier(r["fixed_policies"], c) for c in grid],
                 color="gray", linestyle="--", label="Fixed-policy mixture frontier")
    for family, t in r["composed"].items():
        px, py = t["cost_pct_of_SS"], 100*t["lenient_accuracy"]
        axes[1].scatter(px, py, s=50, label=NAMES[family]); axes[1].annotate(NAMES[family], (px, py), fontsize=8)
    for name, t in r["fixed_policies"].items():
        axes[1].scatter(t["cost_pct_of_SS"], 100*t["lenient_accuracy"], marker="x", color="black")
        axes[1].annotate(name, (t["cost_pct_of_SS"], 100*t["lenient_accuracy"]), fontsize=8)
    axes[1].set_xlabel("Pipeline cost (% of SS)"); axes[1].set_ylabel("Lenient QA accuracy (%)")
    axes[1].set_title("Composed routing (all test questions)"); axes[1].grid(alpha=0.2)
    fig.savefig(a.run / "baseline_comparison.png", dpi=180)
    plt.close(fig)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
