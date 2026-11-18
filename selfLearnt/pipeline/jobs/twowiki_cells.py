"""Jonathan's per-question 2Wiki cells (shared folder copied to $STAIR_PROJECT_DIR/2wiki/cells/<split>/scores_cells_per_instance.csv)
-> results/common/jonathan-ss-2wikimultihopqa-<split>.csv (his always-strong rows, read by `--stage ss_greedy --part compare`)
and a check of his reported numbers: lenient per cell (WW / WS / SW / SS) and the oracle over the four cells, on all questions
and on the headline 500 (test600 positions 100-599 = dev[500:1000]). CPU only, a few seconds.
  bash -c 'source jobs/cluster.env; source jobs/cluster.local.env; python jobs/twowiki_cells.py'"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
CELLS = os.path.join(os.environ.get("STAIR_PROJECT_DIR", "."), "2wiki", "cells")
OUT = os.path.join(ROOT, "results", "common")

for split in ("test600", "val400"):
    p = os.path.join(CELLS, split, "scores_cells_per_instance.csv")
    if not os.path.exists(p):
        print(f"{split}: not found ({p})")
        continue
    d = pd.read_csv(p)
    n = d["_id"].nunique()
    parts = [("all", d)]
    if split == "test600":
        parts.append(("headline 500", d[(d.idx >= 100) & (d.idx < 600)]))
    for name, x in parts:
        acc = {c: 100 * x[x.cell == c].lenient.mean() for c in ("WW", "WS", "SW", "SS")}
        ss_u = x[x.cell == "SS"].units.sum()
        best = x.sort_values(["lenient", "units"], ascending=[False, True]).groupby("_id").head(1)   # cheapest right cell
        print(f"{split} {name} ({x['_id'].nunique()} q): " + " / ".join(f"{c} {v:.1f}" for c, v in acc.items())
              + f" | oracle {100 * best.lenient.mean():.1f} at {100 * best.units.sum() / ss_u:.1f}% of always-strong"
              + f" | always-strong {x[x.cell == 'SS'].units.mean():.0f} units/q")
    ss = d[d.cell == "SS"].sort_values("idx")
    assert len(ss) == n, f"{split}: {len(ss)} SS rows for {n} questions"
    out = pd.DataFrame({"dataset": "2wikimultihopqa", "split": split, "id": ss["_id"], "track": "jonathan",
                        "policy": "always-strong", "menu": "A", "seed": 0, "answer": ss["answer"], "lenient": ss["lenient"],
                        "em": ss["em"], "f1": ss["qa_f1"], "units": ss["units"], "idx": ss["idx"]})
    f = os.path.join(OUT, f"jonathan-ss-2wikimultihopqa-{split}.csv")
    out.to_csv(f, index=False)
    print(f"  -> {f} ({len(out)} rows)")
