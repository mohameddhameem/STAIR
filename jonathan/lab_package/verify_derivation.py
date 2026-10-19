#!/usr/bin/env python3
# =============================================================================
# verify_derivation.py — VERSION: verify_derivation v1.1 2026-09-13 (em_lenient 4-list schema)
#
# ONE QUESTION: does every label in router_data_lenient.jsonl follow from its
# own stored per-cell verdicts?  (All 7,405 rows; independent re-derivation —
# cheapest winning cell, cost order WW < {WS, SW} < SS, tie WS&SW -> SW;
# qa_given_W from the W-branch cells, qa_given_S from the S-branch cells;
# retrieval label = winning cell's first letter.)
#
# Expected result: 0 mismatches.
#
# USAGE (lab, CPU, seconds):
#   python3 verify_derivation.py --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl
# =============================================================================
import json, argparse

VERSION = "verify_derivation v1.1 2026-09-13 (em_lenient 4-list schema)"
CELLS = ["WW", "WS", "SW", "SS"]
FIRST = {"WW": "W", "WS": "W", "SW": "S", "SS": "S"}
CAND = {"ret": ["ret_label", "label_retrieval", "ret"],
        "qa_W": ["qa_given_W", "label_QA_given_W", "qa_label_W", "qa_W"],
        "qa_S": ["qa_given_S", "label_QA_given_S", "qa_label_S", "qa_S"],
        "joint": ["joint_label", "joint_cell", "joint"]}
LIST_ORDER = ["WW", "WS", "SW", "SS"]   # = combo order weak-weak, weak-strong, strong-weak, strong-strong
# NOTE: if this order assumption were wrong, Tier A would fail on ~half the
# labeled rows — so a 0-mismatch result simultaneously certifies the order.
JMAP = {"weak-weak": "WW", "weak-strong": "WS", "strong-weak": "SW", "strong-strong": "SS"}


def resolve(row, names):
    for n in names:
        if n in row:
            return n
    return None


def derive(em):
    if em["WW"]:
        j = "WW"
    elif em["SW"] and em["WS"]:
        j = "SW"                      # the declared tie-break
    elif em["WS"]:
        j = "WS"
    elif em["SW"]:
        j = "SW"
    elif em["SS"]:
        j = "SS"
    else:
        j = None
    qw = "W" if em["WW"] else ("S" if em["WS"] else None)
    qs = "W" if em["SW"] else ("S" if em["SS"] else None)
    return (FIRST[j] if j else None), qw, qs, j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    print(VERSION, flush=True)
    rows = [json.loads(l) for l in open(a.data)]
    r0 = rows[0]
    F = {k: resolve(r0, v) for k, v in CAND.items()}
    has_list = isinstance(r0.get("em_lenient"), list) and len(r0["em_lenient"]) == 4
    print(f"rows: {len(rows)} | schema: {F} | verdicts: em_lenient 4-list = {has_list}", flush=True)
    if not has_list or any(F[k] is None for k in F):
        print("SCHEMA STOP — first row keys:", sorted(r0.keys()))
        return
    mm = []
    for r in rows:
        em = {c: int(v) for c, v in zip(LIST_ORDER, r["em_lenient"])}
        ret, qw, qs, j = derive(em)
        js = r[F["joint"]]
        js = JMAP.get(js, js) if isinstance(js, str) else js
        if not (r[F["ret"]] == ret and r[F["qa_W"]] == qw and r[F["qa_S"]] == qs and (js or None) == j):
            mm.append((r.get("idx"), em, (r[F["ret"]], r[F["qa_W"]], r[F["qa_S"]], js), (ret, qw, qs, j)))
    print(f"RESULT: {len(rows)-len(mm)}/{len(rows)} rows match, {len(mm)} mismatch", flush=True)
    for m in mm[: a.show]:
        print("  mismatch idx", m[0], "verdicts", m[1], "stored", m[2], "derived", m[3])
    print("VERDICT:", "DERIVATION CERTIFIED (0 mismatches)" if not mm else f"{len(mm)} rows need review")


if __name__ == "__main__":
    main()
