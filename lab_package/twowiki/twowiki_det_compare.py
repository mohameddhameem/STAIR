#!/usr/bin/env python3
"""
twowiki_det_compare.py -- READ-ONLY comparison step of the determinism spot-check (job 347398), CPU, seconds.
DRAFT v1.0 2026-11-01. Replaces the inline comparison in twowiki_determinism.sbatch, which crashed (KeyError 'v')
because it globbed *.jsonl and so also read the runner's meta_*.jsonl files (different layout).
Compares ONLY the four output stores -- retr_weak, retr_strong, qa_weak, qa_strong -- entry by entry ({"k": key, "v": value})
between run_a and run_b. PASS = all identical. Writes determinism_result.json next to the runs (refuses to overwrite).
Usage: python3 -I twowiki_det_compare.py --root /common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/determinism5
"""
import argparse, json, os

STORES = ["retr_weak.jsonl", "retr_strong.jsonl", "qa_weak.jsonl", "qa_strong.jsonl"]


def store(p):
    d = {}
    for line in open(p, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"]
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", required=True); a = ap.parse_args()
    out = os.path.join(a.root, "determinism_result.json")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    total = diff = 0; bad = []; per = {}
    for n in STORES:
        A = store(os.path.join(a.root, "run_a", n)); B = store(os.path.join(a.root, "run_b", n))
        assert A.keys() == B.keys(), f"{n}: different keys {sorted(A)} vs {sorted(B)}"
        d = [k for k in A if A[k] != B[k]]
        per[n] = {"entries": len(A), "differing": len(d)}; total += len(A); diff += len(d); bad += [f"{n}:{k}" for k in d]
    res = {"stores": per, "entries": total, "differing": diff, "differing_keys": bad, "verdict": "PASS" if diff == 0 else "FAIL"}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res, indent=1))
    for k in bad[:3]:
        n, key = k.split(":", 1)
        print(f"\n{k}\n  run_a: {str(store(os.path.join(a.root, 'run_a', n))[key])[:300]}\n  run_b: {str(store(os.path.join(a.root, 'run_b', n))[key])[:300]}")


if __name__ == "__main__":
    main()
