#!/usr/bin/env python3
"""
hotpotqa_retrk_score.py -- score the §18 cells (hotpotqa_retrk_cells.py): WS(k_ret) with union / vote, and WW(k_ret, k_ans) vote@k_ans, against
the registered k_ret = 1 cells. CPU, seconds. DRAFT v1.0 18 Nov 2026 for Jonathan's line-by-line review.
v1.2 24 Nov 2026: --meta-retr1 / --meta-qa1 implemented (they were documented but never parsed in v1.1): the measured units of the registered
WS(k_ret = 1) call on the same questions = units(7B, meta_retr_weak[i]) + units(70B, meta_qa_strong["i|weak"]), printed in the reference
line and as a Δ on every S row, so the union context's longer strong call is compared like for like.

Rulers: lenient (stair_common.score.lenient over the gold alias list) = the headline ruler of every HotpotQA board; vote@k = majority over the
first k samples (normalize_answer; tie -> earliest), as hotpotqa_samplek_score.py.
Prices: per-call weak-first (headline axis): WS(k_ret) = 77*k_ret + 3 + 115; WW(k_ret, k_ans) = 77*k_ret + 3*k_ans; k_ret = 1 rows from the
registered cells (WS 195, WW 80). Measured units (AgentTTS normalised FLOPs, stair_common.cost.units with the Config B parameter counts
7 / 3 / 70 B) from the meta sidecars, reported alongside -- a union context is longer than a single retrieval, so the per-call price understates it.
OUTPUT <out>/retrk_outcomes.jsonl  {"i","split","ctx","k_ret","rule","ans":"S"|"W","k_ans","correct","price_units","meas_units"} -- for the k-router
       <out>/retrk_summary.json + a printed table per split.
USAGE  python3 -I hotpotqa_retrk_score.py --cells <retrk out> --data ~/slr/data/tts_test.json --grid <router_data_standard.jsonl> --out <dir>
       [--meta-retr1 meta_retr_weak.jsonl --meta-qa1 meta_qa_strong.jsonl]  (optional: measured units for the k_ret = 1 reference)
"""
import argparse, collections, json, os, sys

for p in (os.path.expanduser("~/slr/STAIR/common"), "/mnt/hdd/jonathan/STAIR/common"):
    sys.path.insert(0, p)
try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli as _t; sys.modules["tomllib"] = _t
from stair_common.score import lenient, normalize_answer  # noqa: E402
from stair_common.cost import units                        # noqa: E402

W_RET, S_RET, W_ANS, S_ANS = 77, 814, 3, 115
PARAMS = {"retr_weak": 7.0, "qa_weak": 3.0, "qa_strong": 70.0}
KANS = [1, 5, 10, 20, 40]


def vote(samples):
    c = collections.Counter(normalize_answer(s) for s in samples); best = max(c.values())
    for s in samples:
        if c[normalize_answer(s)] == best: return s
    return samples[0]


def golds(rec):
    g = rec.get("answers", rec.get("answer")); return g if isinstance(g, list) else [g]


def load_meta(path):
    d = {}
    if path and os.path.exists(path):
        for l in open(path):
            if l.strip(): r = json.loads(l); d[r["k"]] = r["v"] if "v" in r else r
    return d


def meas(mt, params):
    """AgentTTS units of one call (or of a k-sample batch: prompt once, generated tokens summed -- App. A.2 Eq. 8)."""
    if not mt: return None
    return units(params, mt["n_in"], mt.get("n_out_total", mt.get("n_out", 0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True); ap.add_argument("--data", required=True); ap.add_argument("--grid", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--meta-retr1", default=None, help="Config B meta_retr_weak.jsonl (flat lines, k = idx)")
    ap.add_argument("--meta-qa1", default=None, help="Config B meta_qa_strong.jsonl (flat lines, k = idx|weak / idx|strong)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    data = json.load(open(a.data, encoding="utf-8"))
    grid = {g["idx"]: g for g in map(json.loads, (l for l in open(a.grid) if l.strip()))}
    m_r1, m_q1 = load_meta(a.meta_retr1), load_meta(a.meta_qa1)   # v1.2: k_ret = 1 reference sidecars (empty dicts if not given)
    rows, summ = [], {"version": "hotpotqa_retrk_score v1.2 2026-11-24 (k_ret=1 measured reference)", "by": {}, "reference": {}}
    for split in ("val", "test"):
        import glob
        f_retrs = sorted(glob.glob(os.path.join(a.cells, f"retrk_{split}*.jsonl")))      # v1.1: chunk files of the split
        if not f_retrs: continue
        m_retr = {}
        for mf in sorted(glob.glob(os.path.join(a.cells, f"meta_retrk_{split}*.jsonl"))): m_retr.update(load_meta(mf))
        m_qs = load_meta(os.path.join(a.cells, f"meta_qa_strong_retrk_{split}.jsonl"))
        m_qw = load_meta(os.path.join(a.cells, f"meta_qa_weak_retrk_{split}.jsonl"))
        qs = {}; qw = {}
        for l in open(os.path.join(a.cells, f"qa_strong_retrk_{split}.jsonl")):
            if l.strip(): r = json.loads(l); qs[r["k"]] = r["v"]
        for l in open(os.path.join(a.cells, f"samples_retrk_{split}.jsonl")):
            if l.strip(): r = json.loads(l); qw[r["k"]] = json.loads(r["v"])["samples"]
        qids = []
        for l in (line for f in f_retrs for line in open(f)):
            if not l.strip(): continue
            r = json.loads(l); v = json.loads(r["v"]); i = v["i"]; qids.append(i); g = golds(data[i])
            u_retr = meas(m_retr.get(str(i)), PARAMS["retr_weak"])          # 10 sampled retrievals, prompt once
            for ctx in ("u5", "v5", "u10", "v10"):
                k_ret, rule = int(ctx[1:]), ("union" if ctx[0] == "u" else "vote")
                retr_units = (u_retr * k_ret / len(v["sels"])) if u_retr is not None else None   # share of the sampled batch for the first k
                key = f"{i}|{ctx}"
                if key in qs:
                    us = meas(m_qs.get(key), PARAMS["qa_strong"])
                    rows.append({"i": i, "split": split, "ctx": ctx, "k_ret": k_ret, "rule": rule, "ans": "S", "k_ans": 1,
                                 "correct": int(bool(lenient(qs[key], g))), "price_units": W_RET * k_ret + W_ANS + S_ANS,
                                 "meas_units": (round(retr_units + us, 1) if (retr_units is not None and us is not None) else None)})
                if key in qw:
                    uw = meas(m_qw.get(key), PARAMS["qa_weak"]); per = (uw / len(qw[key])) if uw is not None else None
                    for k in KANS:
                        if k > len(qw[key]): continue
                        rows.append({"i": i, "split": split, "ctx": ctx, "k_ret": k_ret, "rule": rule, "ans": "W", "k_ans": k,
                                     "correct": int(bool(lenient(vote(qw[key][:k]), g))), "price_units": W_RET * k_ret + W_ANS * k,
                                     "meas_units": (round(retr_units + per * k, 1) if (retr_units is not None and per is not None) else None)})
        if not qids: continue
        ref = {c: round(100 * sum(grid[i]["em_standard"][j] for i in qids) / len(qids), 2) for j, c in enumerate(("WW", "WS", "SW", "SS"))}
        ws1 = [meas(m_r1.get(str(i)), PARAMS["retr_weak"]) + meas(m_q1.get(f"{i}|weak"), PARAMS["qa_strong"]) for i in qids
               if m_r1.get(str(i)) and m_q1.get(f"{i}|weak")]                                   # v1.2: measured WS(k_ret = 1), same questions
        ws1_mean = (sum(ws1) / len(ws1)) if ws1 else None
        summ["reference"][split] = {"n": len(qids), **ref, "WS_meas_units": (round(ws1_mean, 1) if ws1_mean else None), "WS_meas_n": len(ws1)}
        print(f"\n{split} n={len(qids)}  k_ret=1 reference: WW {ref['WW']} @ 80   WS {ref['WS']} @ 195   SW {ref['SW']} @ 894   SS {ref['SS']} @ 1009"
              + (f"   | WS measured {ws1_mean:.1f} units (n={len(ws1)})" if ws1_mean else "   | WS measured: n/a (give --meta-retr1/--meta-qa1)"))
        print(f"  {'ctx':4s} {'rule':5s} {'ans':3s} {'k_ans':>5s} {'acc':>6s} {'price':>6s} {'meas':>7s}  note")
        for ctx in ("u5", "v5", "u10", "v10"):
            for ans, ks in (("S", [1]), ("W", KANS)):
                for k in ks:
                    rs = [r for r in rows if r["split"] == split and r["ctx"] == ctx and r["ans"] == ans and r["k_ans"] == k]
                    if not rs: continue
                    acc = 100 * sum(r["correct"] for r in rs) / len(rs); price = rs[0]["price_units"]
                    mv = [r["meas_units"] for r in rs if r["meas_units"] is not None]; mm = (sum(mv) / len(mv)) if mv else float("nan")
                    note = (f"vs WS {ref['WS']} @ 195  (Δ {acc - ref['WS']:+.2f}" + (f"; meas ×{mm / ws1_mean:.2f}" if (ws1_mean and mm == mm) else "") + ")" if ans == "S" else (f"vs WW×{k} at k_ret=1" if k > 1 else f"vs WW {ref['WW']} @ 80"))
                    if price > 929: note += "  > 929 strong-only budget/instance"
                    summ["by"][f"{split}|{ctx}|{ans}|{k}"] = {"acc": round(acc, 2), "price_units": price, "meas_units": round(mm, 1), "n": len(rs)}
                    print(f"  {ctx:4s} {('union' if ctx[0]=='u' else 'vote'):5s} {ans:3s} {k:5d} {acc:6.2f} {price:6d} {mm:7.1f}  {note}")
    with open(os.path.join(a.out, "retrk_outcomes.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    json.dump(summ, open(os.path.join(a.out, "retrk_summary.json"), "w"), indent=1); print("wrote", os.path.join(a.out, "retrk_summary.json"))


if __name__ == "__main__":
    main()
