#!/usr/bin/env python3
"""
gpab_test485_split.py -- the FULL GPAgentBench-2K test split as a case list (the registered 485-case comparison, 4 Nov decision
rule). Deterministic: every case with split == "test", sorted by case_id. Writes <out>/test485.txt + test485_report.json
(counts by label / category / urgency, sha256). The 60 pilot cases are a subset; the runner's replay cache skips cases already in
an output file only within the same output folder, so the 485 run is a fresh folder and re-executes the 60 (declared: it does).
DRAFT v1.0 2026-11-08 for Jonathan's review.
Usage: python3 -I gpab_test485_split.py --data /mnt/hdd/jonathan/data/gpagentbench --out /mnt/hdd/jonathan/STAIR/common/stair_common/splits/gpagentbench
"""
import argparse, collections, glob, hashlib, json, os


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
    assert len({r["case_id"] for r in test}) == 485
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "test485.txt")
    with open(path, "w") as f:
        f.write("\n".join(r["case_id"] for r in test) + "\n")
    pilot = os.path.join(a.out, "pilot60.txt")
    pilot_ids = set(open(pilot).read().split()) if os.path.exists(pilot) else set()
    rep = {"n": 485, "by_label": dict(collections.Counter(r["label"] for r in test)),
           "by_category": dict(collections.Counter(r["category"] for r in test)),
           "by_urgency": dict(collections.Counter(r["urgency"] for r in test)),
           "pilot60_subset": len(pilot_ids & {r["case_id"] for r in test}),
           "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}
    json.dump(rep, open(os.path.join(a.out, "test485_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
