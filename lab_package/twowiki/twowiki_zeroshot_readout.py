#!/usr/bin/env python3
"""
twowiki_zeroshot_readout.py -- READ-ONLY readout of the zero-shot champion on the 2Wiki 100 (CPU, no model).
DRAFT v1.0 2026-11-01 for Jonathan's review.

Decision rule = the champion's certified readout (lab_package/hotpotqa/grpo_fresh_test_readout.py), unchanged:
    ret pW >= t_ret  -> retrieval W, then QA pW(W) >= t_qaW ? WW : WS
    else             -> retrieval S, then QA pW(S) >= t_qaS ? SW : SS
Points FROZEN from the HotpotQA validation sweep (28 Oct), not chosen on 2Wiki: greedy 0.50/0.50/0.50 and the
cap<=70% point 0.80/0.05/0.05.
Outcomes and costs come from the cells' per-instance scores (twowiki_cells_score.py output, certified rulers,
em_gold = aliases): lenient, em, qa_f1, units per (instance, cell). Cost = the chosen cell's measured FLOPs units
(retrieval call + answering call), as % of always-SS on the same 100 -- the cells' own currency, so every row is
comparable with the cells table and the oracle.
Usage: python3 twowiki_zeroshot_readout.py --scores <zeroshot dir> --cells <cells dir with scores_cells_per_instance.csv>
"""
import argparse, collections, csv, json, os

CELL = {(True, True): "WW", (True, False): "WS", (False, True): "SW", (False, False): "SS"}
POINTS = [("greedy (router's own choice)", 0.50, 0.50, 0.50), ("cap<=70% point (frozen 28 Oct)", 0.80, 0.05, 0.05)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--scores", required=True); ap.add_argument("--cells", required=True); a = ap.parse_args()
    ret = {r["idx"]: r["pW"] for r in map(json.loads, open(os.path.join(a.scores, "ret_scores.jsonl")))}
    qa = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(os.path.join(a.scores, "qa_scores.jsonl")))}
    cells = collections.defaultdict(dict)
    for r in csv.DictReader(open(os.path.join(a.cells, "scores_cells_per_instance.csv"), encoding="utf-8")):
        cells[int(r["idx"])][r["cell"]] = {k: float(r[k]) for k in ("lenient", "em", "qa_f1", "units")}
    ids = sorted(cells); assert len(ids) == 100 and set(ids) == set(ret), "scores and cells do not cover the same 100"
    ss_units = sum(cells[i]["SS"]["units"] for i in ids)
    out = {"n": len(ids), "references": {}}
    for c in ("WW", "WS", "SW", "SS"):
        out["references"][f"always-{c}"] = {k: round(100 * sum(cells[i][c][k] for i in ids) / len(ids), 2) for k in ("lenient", "em", "qa_f1")} | \
                                           {"cost_pct_of_SS": round(100 * sum(cells[i][c]["units"] for i in ids) / ss_units, 1)}
    out["points"] = {}
    for name, tr, tw, ts in POINTS:
        chosen = {}
        for i in ids:
            w_ret = ret[i] >= tr
            w_qa = qa[(i, "W" if w_ret else "S")] >= (tw if w_ret else ts)
            chosen[i] = CELL[(w_ret, w_qa)]
        res = {k: round(100 * sum(cells[i][chosen[i]][k] for i in ids) / len(ids), 2) for k in ("lenient", "em", "qa_f1")}
        res["cost_pct_of_SS"] = round(100 * sum(cells[i][chosen[i]]["units"] for i in ids) / ss_units, 1)
        res["cell_shares"] = dict(collections.Counter(chosen.values()))
        out["points"][f"{name} t={tr:.2f}/{tw:.2f}/{ts:.2f}"] = res
    json.dump(out, open(os.path.join(a.scores, "zeroshot_readout.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
