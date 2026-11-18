#!/usr/bin/env python3
"""
hotpotqa_samplek_score.py -- vote@k outcomes for the K > 1 weak-answerer cells (hotpotqa_samplek_cells.py). CPU, seconds.
DRAFT v1.0 17 Nov 2026 for Jonathan's line-by-line review. v1.1 18 Nov 2026: SW×k price corrected to 891 + 3k (was 811 + 3k; accuracies unchanged).

For each (question, ctx in {W, S}) and each k in --ks: majority vote over the FIRST k stored samples (stair_common.score.normalize_answer;
tie -> earliest sample), scored with the standard lenient ruler against the gold alias list -- the same ruler as em_standard.
Also reports the single-call reference cells from router_data_standard.jsonl (WW, WS, SW, SS) on the same questions, so the
k = 1 sample can be checked against the stored greedy WW / SW (expected: close, not identical -- greedy vs one sample).

PRICES (weak-first, as budget_allocation.py): WWxk = 77 + 3k, SWxk = 891 + 3k (SW×1 = 894, SW×40 = 1,011 > SS 1,009); measured tokens from the meta sidecar reported alongside.

OUTPUT  <out>/samplek_outcomes.jsonl   {"i", "split", "ctx", "k", "correct", "price_units", "n_out_total"} -- consumed by budget_allocation.py (--samplek)
        <out>/samplek_summary.json     accuracy by (split, ctx, k) + reference cells + the equal-price comparison WWx40 vs WS
USAGE   python3 -I hotpotqa_samplek_score.py --cells <out of the cells run> --data ~/slr/data/tts_test.json --grid <router_data_standard.jsonl> --out <dir>
"""
import argparse, collections, json, os, sys

for p in (os.path.expanduser("~/slr/STAIR/common"), "/mnt/hdd/jonathan/STAIR/common"):
    sys.path.insert(0, p)
try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli as _t; sys.modules["tomllib"] = _t
from stair_common.score import lenient, normalize_answer  # noqa: E402

W_RET, S_RET, W_ANS = 77, 891, 3   # v1.1 18 Nov: S_RET = 77 (weak retrieval, run first) + 814 (strong retrieval) = 891, matching budget_allocation.py's weak-first PRICE [80, 195, 894, 1009]; v1.0 had 811 (wrong: 80 units low)


def vote(samples):
    c = collections.Counter(normalize_answer(s) for s in samples); best = max(c.values())
    for s in samples:
        if c[normalize_answer(s)] == best: return s
    return samples[0]


def golds(rec):
    g = rec.get("answers", rec.get("answer")); return g if isinstance(g, list) else [g]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True); ap.add_argument("--data", required=True); ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True); ap.add_argument("--ks", nargs="+", type=int, default=[1, 5, 10, 20, 40])
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    data = json.load(open(a.data, encoding="utf-8"))
    grid = {g["idx"]: g for g in map(json.loads, (l for l in open(a.grid) if l.strip()))}
    rows = []
    for split in ("val", "test"):
        f = os.path.join(a.cells, f"samples_{split}.jsonl")
        if not os.path.exists(f): continue
        meta = {}
        mf = os.path.join(a.cells, f"meta_{split}.jsonl")
        if os.path.exists(mf):
            for l in open(mf):
                if l.strip(): r = json.loads(l); meta[r["k"]] = r["v"] if "v" in r else r
        for l in open(f):
            if not l.strip(): continue
            r = json.loads(l); v = json.loads(r["v"]) if "v" in r else r
            i, ctx, samples = v["i"], v["ctx"], v["samples"]; g = golds(data[i])
            mt = meta.get(f"{i}|{ctx}", {}); per_sample_out = (mt.get("n_out_total", 0) / max(mt.get("n_seq_out", 1), 1)) if mt else None
            for k in a.ks:
                if k > len(samples): continue
                rows.append({"i": i, "split": split, "ctx": ctx, "k": k, "correct": int(bool(lenient(vote(samples[:k]), g))),
                             "price_units": (W_RET if ctx == "W" else S_RET) + W_ANS * k,
                             "n_out_total": (round(per_sample_out * k, 1) if per_sample_out is not None else None)})
    with open(os.path.join(a.out, "samplek_outcomes.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    summ = {"version": "hotpotqa_samplek_score v1.1 2026-11-18 (S_RET 891)", "ks": a.ks, "by": {}, "reference": {}}
    for split in ("val", "test"):
        qs = sorted({r["i"] for r in rows if r["split"] == split})
        if not qs: continue
        ref = {c: round(100 * sum(grid[i]["em_standard"][j] for i in qs) / len(qs), 2) for j, c in enumerate(("WW", "WS", "SW", "SS"))}
        summ["reference"][split] = {"n": len(qs), **ref, "oracle": round(100 * sum(max(grid[i]["em_standard"]) for i in qs) / len(qs), 2)}
        print(f"\n{split} n={len(qs)}  reference WW {ref['WW']}  WS {ref['WS']}  SW {ref['SW']}  SS {ref['SS']}")
        print(f"  {'ctx':3s} {'k':>3s} {'acc':>6s} {'price':>6s} {'equal-price reference'}")
        for ctx in ("W", "S"):
            for k in a.ks:
                rs = [r for r in rows if r["split"] == split and r["ctx"] == ctx and r["k"] == k]
                if not rs: continue
                acc = 100 * sum(r["correct"] for r in rs) / len(rs); price = rs[0]["price_units"]
                note = ""
                if ctx == "W" and k == 40: note = f"WS = {ref['WS']} at 195"
                if ctx == "S" and k in (20, 40): note = f"SS = {ref['SS']} at 1009" + (" (SWx40 at 1011 EXCEEDS one SS call)" if k == 40 else "")
                summ["by"][f"{split}|{ctx}|{k}"] = {"acc": round(acc, 2), "price_units": price, "n": len(rs)}
                print(f"  {ctx:3s} {k:3d} {acc:6.2f} {price:6d}  {note}")
    json.dump(summ, open(os.path.join(a.out, "samplek_summary.json"), "w"), indent=1); print("wrote", os.path.join(a.out, "samplek_summary.json"))


if __name__ == "__main__":
    main()
