#!/usr/bin/env python3
"""
gpab_pilot_split.py -- the 60-case pilot list from GPAgentBench-2K's OWN test split (registration §3.5). Deterministic.
  all test cases with ground_truth.referral.urgency == "emergency"      (15 expected)        -> included
  + further abstain_and_refer test cases to make 30 refer cases           (seeded sample)
  + 30 diagnose_and_manage test cases                                     (seeded sample, stratified manage_in_gp / manage_after_tests by share)
Writes <out>/pilot60.txt (case_ids, one per line) + pilot60_report.json (counts, sha256). Their other 425 test cases are untouched.
Usage: python3 -I gpab_pilot_split.py --data /mnt/hdd/jonathan/data/gpagentbench --out common/stair_common/splits/gpagentbench
"""
import argparse, collections, glob, hashlib, json, os, random

SEED = 20261031


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--out", required=True); a = ap.parse_args()
    rows = []
    for p in sorted(glob.glob(os.path.join(a.data, "**", "*.jsonl"), recursive=True)):
        for line in open(p, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                gt = r["ground_truth"] if isinstance(r["ground_truth"], dict) else json.loads(r["ground_truth"])
                rows.append({"case_id": r["case_id"], "split": r["split"], "label": r["management_label"],
                             "category": r["management_category"], "urgency": gt["referral"]["urgency"]})
    test = sorted([r for r in rows if r["split"] == "test"], key=lambda r: r["case_id"])
    assert len(test) == 485, len(test)
    rng = random.Random(SEED)
    emergency = [r for r in test if r["urgency"] == "emergency"]
    refer_rest = [r for r in test if r["label"] == "abstain_and_refer" and r["urgency"] != "emergency"]
    manage = [r for r in test if r["label"] == "diagnose_and_manage"]
    pick = list(emergency)
    pick += rng.sample(refer_rest, 30 - len(emergency))
    after = [r for r in manage if r["category"] == "manage_after_tests"]; ingp = [r for r in manage if r["category"] == "manage_in_gp"]
    n_after = max(1, round(30 * len(after) / len(manage)))
    pick += rng.sample(after, min(n_after, len(after))) + rng.sample(ingp, 30 - min(n_after, len(after)))
    assert len(pick) == 60 and len({r["case_id"] for r in pick}) == 60
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "pilot60.txt")
    with open(path, "w") as f:
        f.write("\n".join(r["case_id"] for r in pick) + "\n")
    rep = {"seed": SEED, "n": 60, "by_label": dict(collections.Counter(r["label"] for r in pick)),
           "by_category": dict(collections.Counter(r["category"] for r in pick)),
           "by_urgency": dict(collections.Counter(r["urgency"] for r in pick)),
           "test_total": len(test), "emergency_in_test": len(emergency), "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(rep, open(os.path.join(a.out, "pilot60_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
