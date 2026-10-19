"""
Stage-Level Routing — freeze the train / validation / test split.

Run ONCE, before any training, and commit the output. Splitting after you have seen
results is the single easiest way to invalidate an experiment, and it cannot be undone
credibly afterwards.

  python make_splits.py --combo /mnt/hdd/jonathan/runs/full_run/combo_results_full.jsonl \
                        --data  /mnt/hdd/jonathan/data/tts_test.json \
                        --out   splits.json

Stratifies on (type, level, argmax-R label) so every split carries the same mix of
question types and the same label balance — including the ~39% of instances that no
configuration solves, which must appear in test even though they train nothing.
"""
import os, json, argparse, random, hashlib
from collections import Counter, defaultdict

COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}


def reward(em_by_combo, alpha):
    """R = Q - max(0, C/B - 1).  lambda_t is dormant on low-risk data by design."""
    B = alpha * COST["strong-strong"]
    return {c: em_by_combo[c] - max(0.0, COST[c] / B - 1.0) for c in COMBOS}


def load_combo(path):
    """combo_results_full.jsonl -> {idx: {combo: em}}; only fully-scored instances."""
    em = defaultdict(dict)
    meta = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except Exception:
            continue
        em[r["idx"]][r["combo"]] = r["em_official"]
        meta[r["idx"]] = {"type": r.get("type"), "level": r.get("level"), "_id": r.get("_id")}
    return ({i: d for i, d in em.items() if len(d) == 4},
            meta)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--combo", required=True, help="combo_results_full.jsonl from mint_labels.py")
    ap.add_argument("--data", required=True, help="tts_test.json (for the question text)")
    ap.add_argument("--out", default="splits.json")
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--ratios", default="65,15,20", help="train,val,test percentages")
    ap.add_argument("--seed", type=int, default=20260729)
    args = ap.parse_args()

    tr, va, te = (int(x) for x in args.ratios.split(","))
    assert tr + va + te == 100, "ratios must sum to 100"

    em_by_idx, meta = load_combo(args.combo)
    print(f"fully-scored instances: {len(em_by_idx)}")

    # stratum = (type, level, argmax-R label or 'unsolved')
    strata = defaultdict(list)
    for idx, em in em_by_idx.items():
        R = reward(em, args.alpha)
        lab = "unsolved" if not any(em.values()) else max(COMBOS, key=lambda c: R[c])
        m = meta[idx]
        strata[(m.get("type"), m.get("level"), lab)].append(idx)

    rnd = random.Random(args.seed)
    split = {"train": [], "val": [], "test": []}
    for key in sorted(strata, key=lambda k: str(k)):
        members = sorted(strata[key])
        rnd.shuffle(members)
        n = len(members)
        n_tr = round(n * tr / 100)
        n_va = round(n * va / 100)
        split["train"] += members[:n_tr]
        split["val"] += members[n_tr:n_tr + n_va]
        split["test"] += members[n_tr + n_va:]

    for k in split:
        split[k] = sorted(split[k])

    # report
    print(f"\n{'split':6s} {'n':>6s}  {'solved':>7s}  label distribution")
    for k in ("train", "val", "test"):
        idxs = split[k]
        labs = Counter()
        solved = 0
        for i in idxs:
            em = em_by_idx[i]
            if any(em.values()):
                solved += 1
                R = reward(em, args.alpha)
                labs[max(COMBOS, key=lambda c: R[c])] += 1
        pct = 100 * solved / max(1, len(idxs))
        print(f"{k:6s} {len(idxs):6d}  {solved:4d} ({pct:.0f}%)  {dict(labs)}")

    payload = {
        "seed": args.seed,
        "ratios": {"train": tr, "val": va, "test": te},
        "alpha": args.alpha,
        "combo_source": os.path.abspath(args.combo),
        "combo_md5": hashlib.md5(open(args.combo, "rb").read()).hexdigest(),
        "n_total": len(em_by_idx),
        "counts": {k: len(v) for k, v in split.items()},
        "splits": split,
    }
    json.dump(payload, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}  —  commit this file before training anything")


if __name__ == "__main__":
    main()
