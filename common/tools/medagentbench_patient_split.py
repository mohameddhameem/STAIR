#!/usr/bin/env python3
"""
MedAgentBench patient-disjoint split: check the committed split, or generate one.

The committed split (common/stair_common/splits/medagentbench/{calib,test,calib_sweep}.txt)
partitions the 300 MedAgentBench tasks (test_data_v2.json; 10 task types x 30) into three
groups that share NO patient:

    test   = test.txt                       150 tasks, 15 per task type
    val    = calib_sweep.txt                 60 tasks,  6 per task type   (inside calib)
    train  = calib.txt minus calib_sweep.txt 90 tasks,  9 per task type

A patient is identified by the task's `eval_MRN` field. Two task-1 instances
("Patient not found" cases) carry no patient and may sit anywhere.

Modes
-----
  --check      Verify the committed files against the data file: sizes, per-type quotas,
               calib_sweep inside calib, full coverage, and patient-disjointness at every
               boundary (train/val, train/test, val/test). Exit 1 on any failure.

  --generate   Run a seeded randomized search + hill-climb over patient assignments that
               meets the quotas exactly, write the three files to --out, and report the
               sha256 of each against the MANIFEST (when --manifest is given).

Provenance note
---------------
The committed split (STAIR commit 2a35c9b, 26 Oct 2026) was produced by a seeded randomized
assignment with seed 20260729 whose generator script was not preserved. This script is a
fresh implementation of the same procedure; regenerating with that seed is NOT expected to
reproduce the committed files byte-for-byte. The committed files are the frozen artifact
(hashes in MANIFEST.json); --check verifies their claimed properties independently.

Usage
-----
  python medagentbench_patient_split.py --check \
      --data /path/to/test_data_v2.json --splits common/stair_common/splits/medagentbench

  python medagentbench_patient_split.py --generate --seed 20260729 \
      --data /path/to/test_data_v2.json --out /tmp/new_split \
      [--manifest common/stair_common/splits/MANIFEST.json] [--trials 20000]
"""
import argparse
import collections
import hashlib
import json
import os
import random
import re
import sys

GROUPS = ("train", "val", "test")
TARGET = {"train": 9, "val": 6, "test": 15}          # per task type, per group
MRN_RE = re.compile(r"S\d{7}")


# ----------------------------------------------------------------------------- data

def task_type(tid):
    return tid.split("_")[0]


def sort_key(tid):
    t, i = tid.split("_")
    return (int(t[4:]), int(i))


def load_tasks(path):
    """Return {task_id: patient_or_None}. Patient = eval_MRN. Asserts the data shape."""
    data = json.load(open(path))
    tasks = {}
    for x in data:
        tid = x["id"]
        mrn = x.get("eval_MRN")
        mentioned = set(MRN_RE.findall(json.dumps(x)))
        if mrn is None:
            assert not mentioned, f"{tid}: no eval_MRN but MRNs appear in the record: {mentioned}"
        else:
            assert mentioned == {mrn}, f"{tid}: record mentions {mentioned}, eval_MRN is {mrn}"
        tasks[tid] = mrn
    assert len(tasks) == 300, f"expected 300 tasks, got {len(tasks)}"
    types = collections.Counter(task_type(t) for t in tasks)
    assert set(types) == {f"task{i}" for i in range(1, 11)} and set(types.values()) == {30}, types
    return tasks


def units_of(tasks):
    """Indivisible units: one per patient (all their tasks), one per patient-less task."""
    units = collections.defaultdict(list)
    for tid, mrn in sorted(tasks.items(), key=lambda kv: sort_key(kv[0])):
        units[mrn if mrn is not None else f"nopatient:{tid}"].append(tid)
    return dict(units)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ----------------------------------------------------------------------------- check

def read_ids(path):
    return open(path).read().split()


def check(tasks, splits_dir):
    calib = read_ids(os.path.join(splits_dir, "calib.txt"))
    test = read_ids(os.path.join(splits_dir, "test.txt"))
    sweep = read_ids(os.path.join(splits_dir, "calib_sweep.txt"))
    groups = {"test": set(test), "val": set(sweep), "train": set(calib) - set(sweep)}
    problems = []

    def expect(cond, msg):
        if not cond:
            problems.append(msg)

    expect(len(calib) == 150 and len(set(calib)) == 150, f"calib has {len(calib)} ids ({len(set(calib))} unique)")
    expect(len(test) == 150 and len(set(test)) == 150, f"test has {len(test)} ids ({len(set(test))} unique)")
    expect(len(sweep) == 60 and len(set(sweep)) == 60, f"calib_sweep has {len(sweep)} ids ({len(set(sweep))} unique)")
    expect(set(sweep) <= set(calib), "calib_sweep is not a subset of calib")
    expect(not (set(calib) & set(test)), "calib and test share task ids")
    expect(set(calib) | set(test) == set(tasks), "calib ∪ test does not cover the 300 tasks exactly")
    unknown = (set(calib) | set(test)) - set(tasks)
    expect(not unknown, f"ids not in the data file: {sorted(unknown)[:5]}")

    for g, ids in groups.items():
        per = collections.Counter(task_type(t) for t in ids)
        bad = {t: n for t, n in per.items() if n != TARGET[g]}
        expect(len(per) == 10 and not bad, f"{g}: per-type counts off target {TARGET[g]}: {bad or per}")

    patients = {g: {tasks[t] for t in ids if tasks[t] is not None} for g, ids in groups.items()}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        shared = patients[a] & patients[b]
        expect(not shared, f"{a}/{b} share {len(shared)} patients: {sorted(shared)[:5]}")

    for name, ids in (("calib", calib), ("test", test), ("calib_sweep", sweep)):
        expect(ids == sorted(ids, key=sort_key), f"{name}.txt is not sorted by (type, index)")

    print(f"calib {len(calib)} | test {len(test)} | calib_sweep {len(sweep)} | train {len(groups['train'])}")
    print("patients: " + " | ".join(f"{g} {len(patients[g])}" for g in GROUPS)
          + f" | overlaps train/val {len(patients['train'] & patients['val'])},"
          f" train/test {len(patients['train'] & patients['test'])}, val/test {len(patients['val'] & patients['test'])}")
    for name in ("calib", "test", "calib_sweep"):
        print(f"sha256 {name}.txt = {sha256_file(os.path.join(splits_dir, name + '.txt'))}")
    if problems:
        print("CHECK FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("CHECK PASSED: all split properties hold.")
    return 0


# ----------------------------------------------------------------------------- generate

def deviation(assign, units):
    """Sum over (type, group) of |count - target|; 0 means every quota is met exactly."""
    count = collections.Counter()
    for u, g in assign.items():
        for tid in units[u]:
            count[(task_type(tid), g)] += 1
    return sum(abs(count[(f"task{i}", g)] - TARGET[g]) for i in range(1, 11) for g in GROUPS)


def climb(best, best_dev, names, units):
    """Greedy descent: accept any single-unit move or two-unit swap that lowers the
    deviation; repeat until it is 0 or no move helps. Deterministic (fixed scan order)."""
    improved = True
    while best_dev > 0 and improved:
        improved = False
        for u in names:                                     # single-unit moves
            current = best[u]
            for g in GROUPS:
                if g == current:
                    continue
                best[u] = g
                dev = deviation(best, units)
                if dev < best_dev:
                    best_dev, improved = dev, True
                    break
                best[u] = current
            if improved:
                break
        if improved:
            continue
        for i, u in enumerate(names):                       # two-unit swaps
            for v in names[i + 1:]:
                if best[u] == best[v]:
                    continue
                best[u], best[v] = best[v], best[u]
                dev = deviation(best, units)
                if dev < best_dev:
                    best_dev, improved = dev, True
                    break
                best[u], best[v] = best[v], best[u]
            if improved:
                break
    return best_dev


def generate(tasks, seed, trials, restarts):
    units = units_of(tasks)
    names = sorted(units)                                   # fixed order -> deterministic
    rng = random.Random(seed)
    weights = [TARGET[g] for g in GROUPS]                   # 9 : 6 : 15

    # Stage 1 — randomized search: keep the best of `trials` random assignments.
    best, best_dev = None, None
    for _ in range(trials):
        assign = {u: rng.choices(GROUPS, weights)[0] for u in names}
        dev = deviation(assign, units)
        if best_dev is None or dev < best_dev:
            best, best_dev = assign, dev
            if dev == 0:
                break
    stage1_dev = best_dev

    # Stage 2 — hill-climb; if it stalls above 0, perturb a few units with the same seeded
    # generator and climb again (up to `restarts` times). All randomness comes from `rng`.
    best_dev = climb(best, best_dev, names, units)
    used_restarts = 0
    while best_dev > 0 and used_restarts < restarts:
        used_restarts += 1
        trial = dict(best)
        for u in rng.sample(names, 3):
            trial[u] = rng.choice(GROUPS)
        dev = climb(trial, deviation(trial, units), names, units)
        if dev < best_dev:
            best, best_dev = trial, dev
    return units, best, stage1_dev, best_dev, used_restarts


def write_split(units, assign, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    group_ids = {g: [] for g in GROUPS}
    for u, g in assign.items():
        group_ids[g].extend(units[u])
    files = {
        "calib": sorted(group_ids["train"] + group_ids["val"], key=sort_key),
        "test": sorted(group_ids["test"], key=sort_key),
        "calib_sweep": sorted(group_ids["val"], key=sort_key),
    }
    paths = {}
    for name, ids in files.items():
        p = os.path.join(out_dir, name + ".txt")
        with open(p, "w") as f:
            f.write("\n".join(ids) + "\n")
        paths[name] = p
    return paths


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="verify the committed split files")
    mode.add_argument("--generate", action="store_true", help="generate a new split with --seed")
    ap.add_argument("--data", required=True, help="path to test_data_v2.json")
    ap.add_argument("--splits", default="common/stair_common/splits/medagentbench",
                    help="directory holding calib.txt / test.txt / calib_sweep.txt (for --check)")
    ap.add_argument("--out", help="output directory for --generate")
    ap.add_argument("--seed", type=int, help="random seed for --generate")
    ap.add_argument("--trials", type=int, default=20000, help="random assignments tried before the hill-climb")
    ap.add_argument("--restarts", type=int, default=200, help="perturb-and-climb restarts if the climb stalls")
    ap.add_argument("--manifest", help="MANIFEST.json to compare generated sha256s against")
    args = ap.parse_args()

    tasks = load_tasks(args.data)
    print(f"data: {args.data} (sha256 {sha256_file(args.data)[:16]}...) — 300 tasks, "
          f"{len({m for m in tasks.values() if m})} patients, "
          f"{sum(1 for m in tasks.values() if m is None)} patient-less tasks")

    if args.check:
        sys.exit(check(tasks, args.splits))

    if args.seed is None or not args.out:
        ap.error("--generate needs --seed and --out")
    units, assign, dev1, dev2, used = generate(tasks, args.seed, args.trials, args.restarts)
    print(f"seed {args.seed}: best of {args.trials} random assignments had deviation {dev1}; "
          f"after hill-climb {dev2} ({used} restarts used)")
    if dev2 != 0:
        print("could not reach deviation 0 — try more --trials/--restarts or another seed")
        sys.exit(1)
    paths = write_split(units, assign, args.out)
    manifest = json.load(open(args.manifest))["parts"] if args.manifest else {}
    for name, p in paths.items():
        h = sha256_file(p)
        ref = manifest.get(f"medagentbench/{name}", {}).get("sha256")
        verdict = "" if ref is None else ("  == MANIFEST" if h == ref else "  != MANIFEST")
        print(f"{name}.txt  {h}{verdict}")
    report = {"seed": args.seed, "trials": args.trials, "restarts_used": used,
              "method": "randomized search + hill-climb (single-unit moves and two-unit swaps, seeded restarts)",
              "targets_per_type": TARGET, "deviation_after_search": dev1, "deviation_final": dev2,
              "patient_assignment": {u: g for u, g in sorted(assign.items())}}
    with open(os.path.join(args.out, "split_report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(f"wrote {args.out}/{{calib,test,calib_sweep}}.txt and split_report.json")
    rc = check(tasks, args.out)
    sys.exit(rc)


if __name__ == "__main__":
    main()
