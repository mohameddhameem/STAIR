#!/usr/bin/env python3
"""
hotpotqa_fuser_score.py -- score AgentTTS's selected strategy on our pipeline (hotpotqa_fuser_cells.py) against the registered cells, same questions.
DRAFT v1.0 24 Nov 2026 for Jonathan's line-by-line review. CPU, seconds.

Ruler: lenient contains-gold (stair_common.score.lenient over the gold alias list) -- the AgentTTS author confirmed (11 Aug / 21 Oct 2026) that their
"EM" column is this check, so their 0.74 and our number are the same metric. The fused output is a paragraph; lenient handles that (as the 2Wiki
length-fairness audit noted, a longer answer has more chances to contain the gold -- reported, not hidden: mean fused length is printed).
Reference rows on the SAME questions: WW / WS / SW / SS from router_data_standard.jsonl (em_standard), and vote@40 over the same stored samples
(hotpotqa_samplek_score.py's rule) so "fuser vs vote" is read on identical candidates.
Paired tests: exact McNemar and 10,000-resample bootstrap of the difference, as analysis/retrk_paired_test.py.
Price (per-call, Table 5 axis): SW x 40 + fuser = 891 + 3*40 + 3 = 1,014 units (> SS 1,009); measured units from the meta sidecars reported alongside.
USAGE  python3 -I hotpotqa_fuser_score.py --fuser <out of fuser cells> --cells <samplek folder> --data tts_test.json --grid router_data_standard.jsonl --split test --out <dir>
"""
import argparse, collections, json, math, os, random, sys

for p in (os.path.expanduser("~/slr/STAIR/common"), "/mnt/hdd/jonathan/STAIR/common"):
    sys.path.insert(0, p)
try:
    import tomllib  # noqa: F401
except ModuleNotFoundError:
    import tomli as _t; sys.modules["tomllib"] = _t
from stair_common.score import lenient, normalize_answer  # noqa: E402
from stair_common.cost import units                        # noqa: E402

PRICE = {"WW": 80, "WS": 195, "SW": 894, "SS": 1009}


def vote(samples):
    c = collections.Counter(normalize_answer(s) for s in samples); best = max(c.values())
    for s in samples:
        if c[normalize_answer(s)] == best: return s
    return samples[0]


def golds(rec):
    g = rec.get("answers", rec.get("answer")); return g if isinstance(g, list) else [g]


def paired(name, a, b, qids, n_boot=10000, seed=20261124):
    d = [a[i] - b[i] for i in qids]; n = len(d); win = sum(1 for x in d if x > 0); loss = sum(1 for x in d if x < 0); m = win + loss; k = min(win, loss)
    p = min(1.0, sum(math.comb(m, j) for j in range(0, k + 1)) * 2 / 2 ** m) if m else 1.0
    rng = random.Random(seed); boots = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n * 100 for _ in range(n_boot))
    lo, hi = boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot) - 1]
    print(f"  {name:34s} Δ {sum(d) / n * 100:+.2f}  wins {win} / losses {loss} / ties {n - m}   McNemar p = {p:.4f}   95% CI [{lo:+.2f}, {hi:+.2f}]")
    return {"delta": round(sum(d) / n * 100, 2), "wins": win, "losses": loss, "p": round(p, 4), "ci": [round(lo, 2), round(hi, 2)]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fuser", required=True); ap.add_argument("--cells", required=True); ap.add_argument("--data", required=True); ap.add_argument("--grid", required=True)
    ap.add_argument("--split", required=True); ap.add_argument("--out", required=True); ap.add_argument("--ctx", default="S"); ap.add_argument("--k", type=int, default=40)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    data = json.load(open(a.data, encoding="utf-8"))
    grid = {g["idx"]: g for g in map(json.loads, (l for l in open(a.grid) if l.strip()))}
    fused, meta = {}, {}
    for l in open(os.path.join(a.fuser, f"fuser_{a.split}.jsonl")):
        if l.strip(): r = json.loads(l); fused[r["k"]] = r["v"]
    mp = os.path.join(a.fuser, f"meta_fuser_{a.split}.jsonl")
    if os.path.exists(mp):
        for l in open(mp):
            if l.strip(): r = json.loads(l); meta[r["k"]] = r
    samples = {}
    for l in open(os.path.join(a.cells, f"samples_{a.split}.jsonl")):
        if l.strip(): r = json.loads(l); samples[r["k"]] = json.loads(r["v"])["samples"]
    qids = sorted({int(k.split("|")[0]) for k in fused if k.endswith(f"|{a.ctx}|f{a.k}")})
    n = len(qids); print(f"\n{a.split} n={n}  ctx {a.ctx}  fuser over {a.k} samples")
    F = {i: int(bool(lenient(fused[f"{i}|{a.ctx}|f{a.k}"], golds(data[i])))) for i in qids}
    V = {i: int(bool(lenient(vote(samples[f"{i}|{a.ctx}"][:a.k]), golds(data[i])))) for i in qids}
    ref = {c: {i: grid[i]["em_standard"][j] for i in qids} for j, c in enumerate(("WW", "WS", "SW", "SS"))}
    acc = lambda d: 100 * sum(d.values()) / n
    flen = [len(fused[f"{i}|{a.ctx}|f{a.k}"].split()) for i in qids]
    mu = [units(3.0, meta[k]["n_in"], meta[k]["n_out"]) for k in (f"{i}|{a.ctx}|f{a.k}" for i in qids) if k in meta]
    print(f"  AgentTTS design on our pipeline ({'S' if a.ctx == 'S' else 'W'}W x{a.k} + 3B fuser, T 0.3): {acc(F):.2f}   per-call {PRICE['SW' if a.ctx == 'S' else 'WW'] + 3 * a.k + 3} units;"
          f" fuser call measured {sum(mu) / len(mu):.1f} units (n={len(mu)}); mean fused length {sum(flen) / n:.1f} words")
    print(f"  same samples, majority vote@{a.k}: {acc(V):.2f}")
    for c in ("WW", "WS", "SW", "SS"): print(f"  {c} (k = 1, registered): {acc(ref[c]):.2f} @ {PRICE[c]}")
    print("paired (fuser vs ...):")
    res = {"n": n, "fuser_acc": round(acc(F), 2), "vote_acc": round(acc(V), 2), "ref": {c: round(acc(ref[c]), 2) for c in ref}, "mean_fused_words": round(sum(flen) / n, 1),
           "fuser_meas_units": (round(sum(mu) / len(mu), 1) if mu else None), "paired": {}}
    for name, b in (("vote@40 (same samples)", V), ("SW greedy k=1", ref["SW"]), ("WS k=1", ref["WS"]), ("SS always-strong", ref["SS"])):
        res["paired"][name] = paired(f"fuser vs {name}", F, b, qids)
    with open(os.path.join(a.out, "fuser_outcomes.jsonl"), "w") as f:
        for i in qids: f.write(json.dumps({"i": i, "split": a.split, "ctx": a.ctx, "k": a.k, "fuser_correct": F[i], "vote_correct": V[i], **{f"{c}_correct": ref[c][i] for c in ref}}) + "\n")
    json.dump(res, open(os.path.join(a.out, "fuser_summary.json"), "w"), indent=1); print("wrote", os.path.join(a.out, "fuser_summary.json"))


if __name__ == "__main__":
    main()
