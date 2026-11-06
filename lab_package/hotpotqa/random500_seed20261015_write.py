#!/usr/bin/env python3
"""
random500_seed20261015_write.py -- persist the registered size-matched sample (seed 20261015, n = 500) of the HotpotQA
held-out test so the cluster can read the same 500 instances out of the 1,481-run (rep40_heldout_test). Lab, CPU, seconds.
DRAFT v1.0 2026-11-06 for Jonathan's review.

The draw is EXACTLY the one in jonathan/lab_package/configb_random500_readout.py (registered 15 Oct):
    test   = rows of router_data_standard.jsonl with split == "test", in file order
    sample = random.Random(20261015).sample(test, 500)
Nothing is re-drawn or re-seeded here; the same call is made and its idx values are written.
Writes (refuses to overwrite):
    <out>/random500_seed20261015_idx.txt       one idx per line, SORTED (idx = position in tts_test.json, dev order)
    <out>/random500_seed20261015_manifest.json source sha256, n_test, n, seed, first/last idx, sha256 of the idx file
Checks printed for review: n_test == 1481, 500 distinct idx, all idx are test rows; the first five rows' idx and _id.
Usage (lab):  python3 random500_seed20261015_write.py --src /mnt/hdd/jonathan/runs/config_b/router_data_standard.jsonl \
                  --out /mnt/hdd/jonathan/runs/config_b
Then copy to the cluster:  scp /mnt/hdd/jonathan/runs/config_b/random500_seed20261015_* dohyung.ahn.2025@<cluster>:~/slr/data/
"""
import argparse, hashlib, json, os, random

SEED, N = 20261015, 500


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    idx_path = os.path.join(a.out, "random500_seed20261015_idx.txt")
    if os.path.exists(idx_path):
        raise SystemExit(f"REFUSING: {idx_path} exists")
    rows = [json.loads(l) for l in open(a.src) if l.strip()]
    test = [g for g in rows if g["split"] == "test"]
    sample = random.Random(SEED).sample(test, N)              # identical call to configb_random500_readout.py
    idx = sorted(g["idx"] for g in sample)
    assert len(test) == 1481, f"expected 1,481 test rows, got {len(test)}"
    assert len(set(idx)) == N, "duplicate idx in sample"
    test_idx = {g["idx"] for g in test}
    assert all(i in test_idx for i in idx)
    with open(idx_path, "w") as f:
        f.write("\n".join(str(i) for i in idx) + "\n")
    man = {"version": "random500_seed20261015_write v1.0 2026-11-06", "seed": SEED, "n": N, "source": a.src,
           "source_sha256": sha(a.src), "n_rows": len(rows), "n_test": len(test), "idx_first_last": [idx[0], idx[-1]],
           "idx_file_sha256": sha(idx_path), "draw": "random.Random(20261015).sample(test_rows_in_file_order, 500); idx written sorted"}
    json.dump(man, open(os.path.join(a.out, "random500_seed20261015_manifest.json"), "w"), indent=1)
    print(f"test rows {len(test)}; sample {len(idx)} distinct idx, {idx[0]}..{idx[-1]}; wrote {idx_path}")
    print("first five sampled rows (draw order):", [(g["idx"], g.get("_id", "?")) for g in sample[:5]])
    print("idx file sha256:", man["idx_file_sha256"])


if __name__ == "__main__":
    main()
