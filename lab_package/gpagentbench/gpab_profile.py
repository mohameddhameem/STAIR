#!/usr/bin/env python3
"""
gpab_profile.py -- READ-ONLY profile of the GPAgentBench-2K data files (no model, no network).
Answers the questions that decide whether the set fits stage-level routing + the risk factor:
  * size per split and per management category / referral urgency  (is there a high-risk tier? how big?)
  * which fields exist and how long the texts are                      (what the router would read)
  * repeated presentations / possible patient grouping                 (the leakage unit for our split)
  * departments, persona dimensions, evidence sizes                   (stage structure and heterogeneity)
Usage (lab):  python3 -I gpab_profile.py --dir /mnt/hdd/jonathan/data/gpagentbench
Writes nothing; prints a report. Pass --csv out.csv to also dump one row per case (ids + labels only).
"""
import argparse, collections, glob, hashlib, json, os, statistics as st


def load_rows(d):
    rows = []
    for p in sorted(glob.glob(os.path.join(d, "**", "*.jsonl"), recursive=True)):
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                r = json.loads(line); r["_file"] = os.path.relpath(p, d); rows.append(r)
    if not rows:  # parquet fallback
        try:
            import pandas as pd
            for p in sorted(glob.glob(os.path.join(d, "**", "*.parquet"), recursive=True)):
                for r in pd.read_parquet(p).to_dict("records"):
                    r["_file"] = os.path.relpath(p, d); rows.append(r)
        except ImportError:
            pass
    return rows


def get(r, *path):
    cur = r
    for k in path:
        if isinstance(cur, str):
            try: cur = json.loads(cur)
            except Exception: return None
        if not isinstance(cur, dict): return None
        cur = cur.get(k)
    return cur


def text_len(x):
    if x is None: return 0
    if isinstance(x, (dict, list)): x = json.dumps(x, ensure_ascii=False)
    return len(str(x).split())


def ctab(title, counter, total=None):
    print(f"\n{title}")
    for k, v in counter.most_common():
        pct = f" ({100*v/total:.1f} %)" if total else ""
        print(f"  {str(k):40s} {v:6d}{pct}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = load_rows(a.dir)
    print(f"files: {sorted(set(r['_file'] for r in rows))}")
    print(f"rows: {len(rows)}")
    if not rows: return
    print(f"top-level keys: {sorted(k for k in rows[0] if not k.startswith('_'))}")

    n = len(rows)
    ctab("split", collections.Counter(r.get("split") for r in rows), n)
    ctab("management_label", collections.Counter(r.get("management_label") for r in rows), n)
    ctab("management_category", collections.Counter(r.get("management_category") for r in rows), n)
    ctab("ground_truth.referral.urgency", collections.Counter(get(r, "ground_truth", "referral", "urgency") for r in rows), n)
    ctab("ground_truth.referral.referral_needed", collections.Counter(get(r, "ground_truth", "referral", "referral_needed") for r in rows), n)
    ctab("ground_truth.disposition", collections.Counter(get(r, "ground_truth", "disposition") for r in rows), n)
    # category x urgency, per split  -> the candidate high-risk tier
    cx = collections.Counter((r.get("split"), r.get("management_category"), get(r, "ground_truth", "referral", "urgency")) for r in rows)
    print("\nsplit x management_category x urgency")
    for k, v in sorted(cx.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]), str(kv[0][2]))):
        print(f"  {str(k[0]):6s} {str(k[1]):22s} {str(k[2]):12s} {v:6d}")

    # departments / destinations / diagnoses
    ctab("ground_truth.referral.destination (top 25)", collections.Counter(get(r, "ground_truth", "referral", "destination") for r in rows), n)
    dx = collections.Counter(str(get(r, "ground_truth", "diagnosis")).strip().lower() for r in rows)
    print(f"\ndistinct diagnoses: {len(dx)}; top 15:")
    for k, v in dx.most_common(15): print(f"  {k[:60]:60s} {v:5d}")

    # persona dimensions
    pers = rows[0].get("persona")
    if isinstance(pers, str):
        try: pers = json.loads(pers)
        except Exception: pers = None
    if isinstance(pers, dict):
        for dim in pers:
            ctab(f"persona.{dim}", collections.Counter(str(get(r, "persona", dim)) for r in rows), n)

    # patient / evidence structure and text lengths
    pk = collections.Counter()
    ek = collections.Counter()
    for r in rows:
        p = r.get("patient"); e = r.get("evidence")
        if isinstance(p, str):
            try: p = json.loads(p)
            except Exception: p = None
        if isinstance(e, str):
            try: e = json.loads(e)
            except Exception: e = None
        if isinstance(p, dict): pk.update(p.keys())
        if isinstance(e, dict): ek.update(e.keys())
    ctab("patient.* keys (presence)", pk, n)
    ctab("evidence.* keys (presence)", ek, n)
    for field in ("patient", "evidence", "ground_truth"):
        L = [text_len(r.get(field)) for r in rows]
        print(f"\n{field}: words/case mean {st.mean(L):.0f}, median {st.median(L):.0f}, min {min(L)}, max {max(L)}")
    tests = []
    for r in rows:
        e = r.get("evidence")
        if isinstance(e, str):
            try: e = json.loads(e)
            except Exception: e = None
        if isinstance(e, dict):
            t = e.get("test_results") or e.get("tests") or e.get("lab_results")
            tests.append(len(t) if isinstance(t, (list, dict)) else 0)
    if tests: print(f"tests available per case: mean {st.mean(tests):.1f}, median {st.median(tests)}, max {max(tests)}, zero {sum(1 for t in tests if t==0)}")

    # repeated presentations -> leakage unit
    def cc(r):
        p = r.get("patient")
        if isinstance(p, str):
            try: p = json.loads(p)
            except Exception: p = {}
        if not isinstance(p, dict): return ""
        for k in ("chief_complaint", "presenting_complaint", "complaint"):
            if p.get(k): return str(p[k]).strip().lower()
        return ""
    ccs = collections.Counter(cc(r) for r in rows)
    print(f"\ndistinct chief complaints: {len(ccs)} of {n}; repeated (>1): {sum(1 for v in ccs.values() if v>1)}; top 10:")
    for k, v in ccs.most_common(10): print(f"  {k[:70]:70s} {v:5d}")
    full = collections.Counter(hashlib.sha1(json.dumps(r.get('patient'), sort_keys=True, ensure_ascii=False).encode()).hexdigest() for r in rows)
    print(f"exact duplicate patient blocks: {sum(v-1 for v in full.values() if v>1)}")
    # same chief complaint across train/test
    tr = set(cc(r) for r in rows if r.get("split") == "train"); te = set(cc(r) for r in rows if r.get("split") == "test")
    print(f"chief complaints shared between train and test: {len(tr & te)} of {len(te)} test complaints")
    ids = collections.Counter(r.get("case_id") for r in rows)
    print(f"case_id unique: {len(ids) == n}; example ids: {[r.get('case_id') for r in rows[:3]]}")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f); w.writerow(["case_id", "split", "management_label", "management_category", "urgency", "referral_needed", "destination", "patient_words", "evidence_words"])
            for r in rows:
                w.writerow([r.get("case_id"), r.get("split"), r.get("management_label"), r.get("management_category"),
                            get(r, "ground_truth", "referral", "urgency"), get(r, "ground_truth", "referral", "referral_needed"),
                            get(r, "ground_truth", "referral", "destination"), text_len(r.get("patient")), text_len(r.get("evidence"))])
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
