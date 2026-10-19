#!/usr/bin/env python3
"""
make_label_review.py - CPU-only, read-only. Builds label_review.csv: one row per
instance with every label field (Ret side computed here; teacher-1 fields joined
if --t1 file given; grid outcomes as scoreboard columns). Open the csv in Excel.

usage:
  python make_label_review.py                      # Ret fields + outcomes (all 7,405)
  python make_label_review.py --t1 PATH_TO_T1.jsonl  # + qa_t1_label / p_supported
"""
import argparse, csv, json, re, string
import run_labels4 as R

GRID = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
RETW = "/mnt/hdd/jonathan/runs/full_run_p100/retr_weak.jsonl"
RETS = "/mnt/hdd/jonathan/runs/full_run_p100/retr_strong.jsonl"
DATA = "/mnt/hdd/jonathan/data/tts_test.json"
OUT  = "/mnt/hdd/jonathan/runs/label_review.csv"

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def contains(text, golds):
    nt = norm(text)
    return any(re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", nt)
               for g in golds if norm(g))

def retr_map(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        m[int(str(r["k"]).split("|")[0])] = r["v"]
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", default=None, help="teacher-1 label jsonl (idx -> label/prob)")
    a = ap.parse_args()
    data = json.load(open(DATA))
    rw, rs = retr_map(RETW), retr_map(RETS)
    t1 = {}
    if a.t1:
        for l in open(a.t1):
            r = json.loads(l)
            i = int(r.get("idx", r.get("k", -1)))
            t1[i] = r
    grid = {}
    for l in open(GRID):
        r = json.loads(l)
        grid[r["idx"]] = r
    n_written = 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx","split","question","gold","n_weak_sel","covW","covUnion",
                    "ret_label","ret_target","qa_t1_label","qa_t1_target","p_supported",
                    "em_WW","em_WS","em_SW","em_SS"])
        for i, inst in enumerate(data):
            if i not in grid or i not in rw or i not in rs:
                continue
            g = grid[i]
            golds = inst["answers"] if isinstance(inst["answers"], list) else [inst["answers"]]
            n = len(inst["ctxs_candidate"])
            selW = R.parse_selection(rw[i], n)
            selS = R.parse_selection(rs[i], n)
            txtW = " ".join(inst["ctxs_candidate"][k-1] for k in selW)
            txtU = " ".join(inst["ctxs_candidate"][k-1] for k in sorted(set(selW) | set(selS)))
            covW = contains(txtW, golds)
            covU = contains(txtU, golds)
            ret = "W" if covW else ("S" if covU else "X")
            ret_t = "W" if ret == "X" else ret
            q1 = q1t = ps = ""
            if i in t1:
                r1 = t1[i]
                raw = r1.get("label", r1.get("v", ""))
                q1 = "X" if not covU else raw          # the covUnion=0 join
                q1t = "W" if q1 == "X" else q1
                ps = r1.get("p_supported", r1.get("prob", r1.get("p", "")))
            em = g["em_lenient"]
            w.writerow([i, g["split"], inst["question"][:120], " | ".join(golds)[:80],
                        len(selW), covW, covU, ret, ret_t, q1, q1t, ps,
                        em[0], em[1], em[2], em[3]])
            n_written += 1
    print("wrote", OUT, "rows:", n_written)
    from collections import Counter
    c = Counter()
    for l in open(OUT):
        p = l.split(",")
        if p[0] == "idx": continue
        c[p[7]] += 1
    print("ret_label shares:", dict(c))

if __name__ == "__main__":
    main()
