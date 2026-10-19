#!/usr/bin/env python3
# =============================================================================
# lenient_score.py — VERSION: lenient_score v1.0 2026-08-18
#
# Re-scores a grid run directory (combo_results_full.jsonl) under:
#   strict EM        normalized exact equality            (cross-check vs stored em_official)
#   lenient-sub      normalized gold SUBSTRING of normalized prediction
#   lenient-word     normalized gold as WHOLE-WORD regex match in normalized prediction
#   token F1         official HotpotQA-style token F1 (with yes/no special case)
#
# ASSUMPTION-1  Metric semantics = containment ("does the gold answer appear in
#               the prediction"), per author (Fali) confirmation 11 Aug 2026.
#               Exact reference code (evaluation_2wiki.py) is absent from the
#               public repo; two containment variants are reported side by side
#               until it arrives / cluster scorer reconciliation next week.
# ASSUMPTION-2  Normalization = HotpotQA official normalize_answer:
#               lowercase -> remove punctuation -> remove articles a/an/the ->
#               collapse whitespace. Stand-in, not verified against their code.
# ASSUMPTION-3  Input schema: one JSON object per line with fields
#               idx, combo, answer, gold, em_official (as in armC_n500).
# ASSUMPTION-4  Pure accuracy re-score: no cost columns in v1.
#
# Usage (CPU only, any venv with Python 3.8+, stdlib only):
#   python3 lenient_score.py --run /mnt/hdd/jonathan/runs/armC_n500 \
#       --out /mnt/hdd/jonathan/validation_g0/lenient_rescore_armC_n500.json
# =============================================================================
import argparse, json, os, re, string, sys
from collections import Counter, defaultdict

VERSION = "lenient_score v1.0 2026-08-18"


# ---------------------------------------------------------------- normalization
def normalize_answer(s):
    """HotpotQA official normalization (ASSUMPTION-2)."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = " ".join(s.split())
    return s


# ---------------------------------------------------------------- metrics
def strict_em(pred, gold):
    return int(normalize_answer(pred) == normalize_answer(gold))


def lenient_substring(pred, gold):
    ng, np_ = normalize_answer(gold), normalize_answer(pred)
    return int(ng != "" and ng in np_)


def lenient_word(pred, gold):
    ng, np_ = normalize_answer(gold), normalize_answer(pred)
    if ng == "":
        return 0
    return int(re.search(r"(?<!\w)" + re.escape(ng) + r"(?!\w)", np_) is not None)


def token_f1(pred, gold):
    """Official HotpotQA F1 incl. the yes/no/noanswer special case."""
    np_, ng = normalize_answer(pred), normalize_answer(gold)
    if np_ in ("yes", "no", "noanswer") or ng in ("yes", "no", "noanswer"):
        if np_ != ng:
            return 0.0
    pt, gt = np_.split(), ng.split()
    common = Counter(pt) & Counter(gt)
    n_same = sum(common.values())
    if n_same == 0:
        return 0.0
    precision = n_same / len(pt)
    recall = n_same / len(gt)
    return 2 * precision * recall / (precision + recall)


METRICS = {
    "strict_em": strict_em,
    "lenient_sub": lenient_substring,
    "lenient_word": lenient_word,
    "f1": token_f1,
}


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run directory containing combo_results_full.jsonl")
    ap.add_argument("--file", default="combo_results_full.jsonl", help="results filename inside --run")
    ap.add_argument("--out", default=None, help="output JSON path (default: <run>/lenient_rescore.json)")
    args = ap.parse_args()

    print(VERSION)
    path = os.path.join(args.run, args.file)
    if not os.path.exists(path):
        sys.exit(f"ERROR: {path} not found")

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"records: {len(records)} from {path}")

    manifest = {}
    mpath = os.path.join(args.run, "run_manifest.json")
    if os.path.exists(mpath):
        with open(mpath) as f:
            manifest = json.load(f)

    # ---- score every record under every metric
    cells = sorted({r["combo"] for r in records})
    idxs = sorted({r["idx"] for r in records})
    scored = defaultdict(dict)  # scored[idx][combo] = {metric: value}
    mismatches = []
    for r in records:
        vals = {name: fn(r["answer"], r["gold"]) for name, fn in METRICS.items()}
        scored[r["idx"]][r["combo"]] = vals
        # self-check: recomputed strict EM vs stored em_official
        if "em_official" in r and int(vals["strict_em"]) != int(r["em_official"]):
            mismatches.append((r["idx"], r["combo"], r["answer"][:60], r["gold"], r["em_official"], vals["strict_em"]))

    # ---- self-check report (must be zero before trusting anything below)
    print(f"\nSELF-CHECK strict_em vs stored em_official: {len(mismatches)} mismatches "
          f"/ {len(records)} records")
    for m in mismatches[:5]:
        print(f"  idx {m[0]} {m[1]}: pred={m[2]!r} gold={m[3]!r} stored={m[4]} recomputed={m[5]}")
    if mismatches:
        print("  WARNING: normalization differs from the run that produced em_official. "
              "Investigate before using lenient columns.")

    # ---- per-cell means, oracle, best-fixed under each metric
    def mean(v):
        return sum(v) / len(v) if v else 0.0

    summary = {"per_cell": {}, "oracle": {}, "best_fixed": {}}
    for name in METRICS:
        per_cell = {}
        for c in cells:
            vals = [scored[i][c][name] for i in idxs if c in scored[i]]
            per_cell[c] = round(100 * mean(vals), 2)
        oracle_vals = [max(scored[i][c][name] for c in scored[i]) for i in idxs]
        summary["per_cell"][name] = per_cell
        summary["oracle"][name] = round(100 * mean(oracle_vals), 2)
        bf_cell = max(per_cell, key=per_cell.get)
        summary["best_fixed"][name] = {"cell": bf_cell, "value": per_cell[bf_cell]}

    # ---- print table
    print(f"\ninstances: {len(idxs)}  (idx {min(idxs)}..{max(idxs)})  cells: {', '.join(cells)}")
    hdr = f"{'cell':<14}" + "".join(f"{m:>14}" for m in METRICS)
    print("\n" + hdr)
    print("-" * len(hdr))
    for c in cells:
        row = f"{c:<14}" + "".join(f"{summary['per_cell'][m][c]:>14.2f}" for m in METRICS)
        print(row)
    print("-" * len(hdr))
    print(f"{'best-fixed':<14}" + "".join(
        f"{summary['best_fixed'][m]['value']:>14.2f}" for m in METRICS))
    print(f"{'oracle':<14}" + "".join(f"{summary['oracle'][m]:>14.2f}" for m in METRICS))
    print(f"\nbest-fixed cell by metric: " +
          ", ".join(f"{m}={summary['best_fixed'][m]['cell']}" for m in METRICS))

    # ---- write JSON
    out_path = args.out or os.path.join(args.run, "lenient_rescore.json")
    payload = {
        "version": VERSION,
        "input": path,
        "n_records": len(records),
        "n_instances": len(idxs),
        "idx_range": [min(idxs), max(idxs)],
        "region_note": "idx starts at 0 => first-N region of tts_test.json, NOT dev[500:1000]"
                       if min(idxs) == 0 else "check region against routeA spec",
        "self_check_mismatches": len(mismatches),
        "metric_provenance": (
            "containment semantics author-confirmed 11 Aug 2026 (Fali); exact reference "
            "code (evaluation_2wiki.py) absent from public repo — pending author files; "
            "normalization = HotpotQA official normalize_answer (stand-in); numbers pending "
            "reconciliation with the cluster 75.8 scorer"
        ),
        "manifest": manifest,
        "summary": summary,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
