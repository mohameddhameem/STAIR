#!/usr/bin/env python3
# =============================================================================
# make_splits_8020.py — VERSION: make_splits_8020 v1.0 2026-09-14
#
# Writes a NEW splits file (default splits_8020.json) for the enlarged-train
# experiment: train 5,924 / test 1,481 (80/20 of 7,405), with --val_n
# optionally carving a small validation pool OUT OF THE TRAIN SHARE
# (train becomes 5,924 - val_n; test stays exactly 1,481).
#
# WHY val_n matters: sft_router.py draws its every-25-steps probe examples
# from the VAL split.  With val_n = 0 the probe pool is empty and the run
# cannot produce curves (and best-adapter selection has nothing to select
# on).  val_n >= 300 is recommended if the run should behave like Run 1.
#
# NEVER overwrites splits.json — the frozen split behind all prior runs.
#
# USAGE (lab):
#   python3 make_splits_8020.py \
#       --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#       --out splits_8020.json --train_n 5924 --test_n 1481 --val_n 300
# =============================================================================
import json, random, argparse, collections

VERSION = "make_splits_8020 v1.0 2026-09-14"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", default="splits_8020.json")
    ap.add_argument("--train_n", type=int, default=5924)
    ap.add_argument("--test_n", type=int, default=1481)
    ap.add_argument("--val_n", type=int, default=300,
                    help="carved from the train share; 0 = none (probes will have no pool)")
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()
    print(VERSION, flush=True)
    assert a.out != "splits.json", "refusing to overwrite the frozen splits.json"

    rows = [json.loads(l) for l in open(a.data)]
    idxs = [r["idx"] for r in rows]
    resolved = {r["idx"]: (r["ret_label"] is not None) for r in rows}
    assert len(idxs) == len(set(idxs))
    n = len(idxs)
    assert a.train_n + a.test_n == n, f"train+test must equal {n} (got {a.train_n + a.test_n})"
    assert 0 <= a.val_n < a.train_n

    rnd = random.Random(a.seed)
    order = idxs[:]
    rnd.shuffle(order)
    test = sorted(order[: a.test_n])
    rest = order[a.test_n:]
    val = sorted(rest[: a.val_n])
    train = sorted(rest[a.val_n:])

    def stats(ids):
        r = sum(resolved[i] for i in ids)
        return {"n": len(ids), "resolved": r, "unresolved": len(ids) - r}

    out = {"train": train, "val": val, "test": test,
           "meta": {"script": VERSION, "seed": a.seed, "source": a.data,
                    "design": "80/20 of all instances; val carved from train share",
                    "stats": {k: stats(v) for k, v in
                              (("train", train), ("val", val), ("test", test))}}}
    json.dump(out, open(a.out, "w"))
    print(json.dumps(out["meta"]["stats"], indent=1))
    print(f"written -> {a.out}  (train {len(train)} / val {len(val)} / test {len(test)})")


if __name__ == "__main__":
    main()
