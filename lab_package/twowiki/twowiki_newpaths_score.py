#!/usr/bin/env python3
"""
twowiki_newpaths_score.py -- score the §15 new-path cells (twowiki_newpaths_cells.py) against the four single-call cells of the
same split, with the certified rulers. CPU, seconds. DRAFT v1.0 12 Nov 2026 for Jonathan's line-by-line review.

INPUT   --run  <merged split folder>  tts_test_2wiki_adapted.json, retr_*/qa_*/meta_* stores (the four cells, as twowiki_cells_score.py)
        --new  <twowiki_newpaths_cells --out>  qa_newpaths_{weak,strong}.jsonl + meta_*, manifest.json
CELLS   reference: WW WS SW SS (from --run) · new: U-S U-W S-all W-all · SWxk-vote@k (majority vote over the first k samples,
        normalised with stair_common.score.normalize_answer, tie -> earliest sample) · SWxk-fuse@k (the 3B fuser's answer) for
        k in --ks (every k <= K recorded) · SWx1 = sample 0 alone (the sampling counterpart of SW, as a sanity line)
SCORE   lenient (standard ruler over the alias list) on every cell; paired difference vs SS with its SE (sqrt(p_disc)/sqrt(n) form:
        SE of a paired mean of -1/0/+1 values); by question type = the record's own `type` field when present, else the wording
        rule (stair_common.qtype) -- the manifest says which was used.
COST    stair_common.cost.units(params_b, n_in, n_out) per generation (retr 7/72, QA 3/70; fuser 3). A cell's cost = the retrieval
        call(s) its context needed + its answer call(s): U-* pays BOTH retrievals (weak-first: the weak call is paid anyway);
        SWxk pays strong retrieval + k sampled decodes (one prompt, k outputs: n_in once + total n_out) + the fuser call;
        *-all pays no retrieval. Reported as % of SS. (Table 5 per-call convention is given alongside for SWxk: 72 + 3k + 3.)
OUTPUT  <new>/newpaths_per_instance.csv (+ .xlsx when openpyxl is present) · <new>/newpaths_summary.json · console table.
USAGE   python3 -I twowiki_newpaths_score.py --run <merged/val400> --new <newpaths out> [--ks 1 5 10 20 50]
"""
import argparse, collections, csv, json, math, os, sys

for p in ("/mnt/hdd/jonathan/STAIR/common", os.path.expanduser("~/slr/STAIR/common")):
    sys.path.insert(0, p)
try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli as _tomli; sys.modules["tomllib"] = _tomli
from stair_common.score import lenient, normalize_answer  # noqa: E402
from stair_common.cost import units  # noqa: E402
from stair_common.qtype import question_type  # noqa: E402

P_RETR = {"weak": 7.0, "strong": 72.0}; P_QA = {"weak": 3.0, "strong": 70.0}


def load_store(path):
    d = {}
    if not os.path.exists(path):
        return d
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def vote(samples):
    c = collections.Counter(normalize_answer(s) for s in samples)
    best = max(c.values());
    for s in samples:                       # earliest sample among the tied modes
        if c[normalize_answer(s)] == best:
            return s
    return samples[0]


def paired(diffs):
    n = len(diffs); m = sum(diffs) / n
    var = sum((d - m) ** 2 for d in diffs) / max(n - 1, 1)
    return m, math.sqrt(var / n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True); ap.add_argument("--new", required=True)
    ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 50])
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.new, "manifest.json")))
    K = man["k"]; lo, hi = man["range"]
    data = json.load(open(os.path.join(a.run, "tts_test_2wiki_adapted.json"), encoding="utf-8"))
    rq = {w: load_store(os.path.join(a.run, f"qa_{w}.jsonl")) for w in ("weak", "strong")}
    rmeta = {f"{st}_{w}": load_store(os.path.join(a.run, f"meta_{st}_{w}.jsonl")) for st in ("retr", "qa") for w in ("weak", "strong")}
    nq = {w: load_store(os.path.join(a.new, f"qa_newpaths_{w}.jsonl")) for w in ("weak", "strong")}
    nmeta = {w: load_store(os.path.join(a.new, f"meta_qa_newpaths_{w}.jsonl")) for w in ("weak", "strong")}
    typed_field = all("type" in data[i] for i in range(lo, hi))

    def cost_retr(i, w): m = rmeta[f"retr_{w}"][str(i)]; return units(P_RETR[w], m["n_in"], m["n_out"])
    def cost_qa_ref(i, w, rsrc): m = rmeta[f"qa_{w}"][f"{i}|{rsrc}"]; return units(P_QA[w], m["n_in"], m["n_out"])
    def cost_new(i, w, key): m = nmeta[w][f"{i}|{key}"]; return units(P_QA[w], m["n_in"], m["n_out"])

    rows = []
    for i in range(lo, hi):
        rec = data[i]; golds = rec["answers"]
        t = rec["type"] if typed_field else question_type("2wiki", rec["question"])
        r = {"idx": i, "type": t.replace("_", "-")}
        cells = {"WW": (rq["weak"].get(f"{i}|weak"), cost_retr(i, "weak") + cost_qa_ref(i, "weak", "weak")),
                 "WS": (rq["strong"].get(f"{i}|weak"), cost_retr(i, "weak") + cost_qa_ref(i, "strong", "weak")),
                 "SW": (rq["weak"].get(f"{i}|strong"), cost_retr(i, "strong") + cost_qa_ref(i, "weak", "strong")),
                 "SS": (rq["strong"].get(f"{i}|strong"), cost_retr(i, "strong") + cost_qa_ref(i, "strong", "strong"))}
        both_retr = cost_retr(i, "weak") + cost_retr(i, "strong")
        if f"{i}|U" in nq["strong"]:   cells["U-S"] = (nq["strong"][f"{i}|U"], both_retr + cost_new(i, "strong", "U"))
        if f"{i}|U" in nq["weak"]:     cells["U-W"] = (nq["weak"][f"{i}|U"], both_retr + cost_new(i, "weak", "U"))
        if f"{i}|ALL" in nq["strong"]: cells["S-all"] = (nq["strong"][f"{i}|ALL"], cost_new(i, "strong", "ALL"))
        if f"{i}|ALL" in nq["weak"]:   cells["W-all"] = (nq["weak"][f"{i}|ALL"], cost_new(i, "weak", "ALL"))
        tag = f"SWx{K}"
        if f"{i}|{tag}|s0" in nq["weak"]:
            samples = [nq["weak"][f"{i}|{tag}|s{j}"] for j in range(K)]
            sm = nmeta["weak"][f"{i}|{tag}"]
            # k decodes share one prompt: n_in once + the first k outputs' tokens, approximated as k/K of the total output tokens
            for k in a.ks:
                if k > K: continue
                c_samp = cost_retr(i, "strong") + units(P_QA["weak"], sm["n_in"], round(sm["n_out_total"] * k / K))
                cells[f"SWx{k}-vote"] = (vote(samples[:k]), c_samp)
                if f"{i}|SWx{k}|fuse" in nq["weak"]:
                    cells[f"SWx{k}-fuse"] = (nq["weak"][f"{i}|SWx{k}|fuse"], c_samp + cost_new(i, "weak", f"SWx{k}|fuse"))
        for c, (ans, cost) in cells.items():
            r[c] = int(lenient(ans, golds)) if ans is not None else None
            r[f"cost_{c}"] = round(cost, 2)
        rows.append(r)

    names = [c for c in rows[0] if not c.startswith("cost_") and c not in ("idx", "type")]
    types = sorted({r["type"] for r in rows})
    summ = {"n": len(rows), "range": [lo, hi], "type_source": "record field" if typed_field else "wording rule (stair_common.qtype)",
            "manifest": man, "cells": {}}
    c_ss = sum(r["cost_SS"] for r in rows) / len(rows)
    print(f"n = {len(rows)}  [{lo}..{hi - 1}]   types from {summ['type_source']}   C_SS = {c_ss:.1f} units/instance")
    print(f"{'cell':14s} {'acc':>6s} {'vs SS':>7s} {'SE':>5s} {'cost%SS':>8s} | " + " ".join(f"{t[:10]:>10s}" for t in types))
    for c in names:
        rs = [r for r in rows if r[c] is not None]
        if not rs: continue
        acc = 100 * sum(r[c] for r in rs) / len(rs)
        d, se = paired([r[c] - r["SS"] for r in rs]); cost = 100 * sum(r[f"cost_{c}"] for r in rs) / len(rs) / c_ss
        byt = {t: 100 * sum(r[c] for r in rs if r["type"] == t) / max(sum(1 for r in rs if r["type"] == t), 1) for t in types}
        summ["cells"][c] = {"n": len(rs), "acc": round(acc, 2), "diff_vs_SS": round(100 * d, 2), "se": round(100 * se, 2),
                            "cost_pct_SS": round(cost, 1), "by_type": {t: round(v, 1) for t, v in byt.items()}}
        print(f"{c:14s} {acc:6.2f} {100 * d:+7.2f} {100 * se:5.2f} {cost:8.1f} | " + " ".join(f"{byt[t]:10.1f}" for t in types))
    print("by-type n:", {t: sum(1 for r in rows if r["type"] == t) for t in types})

    with open(os.path.join(a.new, "newpaths_per_instance.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    try:
        import openpyxl
        wb = openpyxl.Workbook(); ws = wb.active; ws.title = "per_instance"; ws.append(list(rows[0]))
        for r in rows: ws.append([r[k] for k in rows[0]])
        wb.save(os.path.join(a.new, "newpaths_per_instance.xlsx"))
    except ModuleNotFoundError:
        print("(openpyxl absent: csv only)")
    json.dump(summ, open(os.path.join(a.new, "newpaths_summary.json"), "w"), indent=1)
    print("wrote", os.path.join(a.new, "newpaths_summary.json"))


if __name__ == "__main__":
    main()
