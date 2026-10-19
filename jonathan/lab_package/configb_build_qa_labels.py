# 8 build_qa_labels.py -- the QA label PACKAGER (teacher-1 / outcome lane only;
# the teacher-2 comparative lane was retired by 7's gate, 7 Oct: prize 3.4%,
# danger 4x, net -9.7 val pts).
#
# The outcome QA labels already exist in the grid file (qa_label_W / qa_label_S,
# minted from em_standard by configb_make_grid.py).  This script packages them into a
# certified labels artifact, exactly as ret_labels.jsonl is for the Ret stage:
# two rows per instance (one per retrieval choice), dual targets, manifest with
# class weights, three audits, and a review CSV.
#
#   qa_label  (honest):  W = deliver weak's answer | S = escalate | X = both wrong
#   qa_target (mapped):  X -> W  (when neither answer works, don't pay for strong)
#
# CPU only; reads the cached grid; writes only into OUT.
import os, json, csv, random
from collections import Counter

GRID_FILE  = "/mnt/hdd/jonathan/runs/config_b/router_data_standard.jsonl"
RET_LABELS = "/mnt/hdd/jonathan/runs/config_b/ret_labels/ret_labels.jsonl"
QA_WEAK    = "/mnt/hdd/jonathan/runs/config_b/qa_weak.jsonl"
QA_STRONG  = "/mnt/hdd/jonathan/runs/config_b/qa_strong.jsonl"
DATA       = "/mnt/hdd/jonathan/data/tts_test.json"
OUT        = "/mnt/hdd/jonathan/runs/config_b/qa_labels"
REVIEW_N, SEED = 150, 17

def load_store(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line); d[r["k"]] = r["v"]
        except Exception:
            pass
    return d

def main():
    os.makedirs(OUT, exist_ok=True)
    grid = [json.loads(l) for l in open(GRID_FILE)]
    cov  = {r["idx"]: r for r in (json.loads(l) for l in open(RET_LABELS))}
    qa_w, qa_s = load_store(QA_WEAK), load_store(QA_STRONG)
    data = json.load(open(DATA))

    rows, mismatch = [], Counter()
    for g in grid:
        i, sp, em = g["idx"], g["split"], g["em_standard"]   # em = [WW, WS, SW, SS]
        for rc, stored, (wq, sq) in (("W", g["qa_label_W"], (em[0], em[1])),
                                     ("S", g["qa_label_S"], (em[2], em[3]))):
            # audit 1: recompute from outcomes; must equal the stored field
            expect = "W" if wq else ("S" if sq else None)
            if expect != stored:
                mismatch[f"{rc}:{stored}->{expect}"] += 1
            lab = stored if stored is not None else "X"
            rows.append({"idx": i, "split": sp, "ret_choice": rc,
                         "qa_label": lab, "qa_target": "W" if lab == "X" else lab})

    # audit 2: shares per split / retrieval choice
    print("split  ret_choice  " + "  ".join(f"{c:>5}" for c in ("W","S","X")))
    shares = {}
    for sp in ("train", "val", "test"):
        for rc in ("W", "S"):
            c = Counter(r["qa_label"] for r in rows
                        if r["split"] == sp and r["ret_choice"] == rc)
            shares[f"{sp}|{rc}"] = dict(c)
            print(f"{sp:<6} {rc:<10} " + "  ".join(f"{c.get(k,0):>5}" for k in ("W","S","X")))
    print("label-vs-outcome mismatches (audit 1, expect NONE):", dict(mismatch) or 0)

    # audit 3: X overlap with covUnion=0 (report only -- X here means both answers wrong)
    x_rows = [r for r in rows if r["qa_label"] == "X"]
    x_nocov = sum(1 for r in x_rows if not cov.get(r["idx"], {}).get("covUnion", True))
    print(f"X rows: {len(x_rows)} | of which covUnion=0 (retrieval hopeless too): "
          f"{x_nocov} ({100*x_nocov/len(x_rows):.1f}%)")

    # class weights post-mapping (train rows only), as the Ret manifest does
    tr = Counter(r["qa_target"] for r in rows if r["split"] == "train")
    tot = sum(tr.values())
    weights = {k: round(tot / (2 * v), 4) for k, v in tr.items()}

    with open(os.path.join(OUT, "qa_labels.jsonl"), "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    json.dump({"source": GRID_FILE, "lane": "teacher-1 / outcome-only",
               "teacher2_status": "retired by chooser_ceiling gate, 7 Oct",
               "rule": "W if weak answer right; S if weak wrong and strong right; X if both wrong",
               "mapping": "X -> W (do not pay for strong when neither works)",
               "rows": len(rows), "shares": shares,
               "class_weights_post_mapping": weights,
               "audit_mismatches": dict(mismatch)},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2)

    random.seed(SEED)
    sample = random.sample([r for r in rows if r["split"] != "test"], REVIEW_N)
    with open(os.path.join(OUT, "qa_label_review.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["idx","split","ret_choice","qa_label","qa_target",
                    "question","weak_answer","strong_answer","gold_answers"])
        for r in sample:
            i = r["idx"]; sfx = "weak" if r["ret_choice"] == "W" else "strong"
            w.writerow([i, r["split"], r["ret_choice"], r["qa_label"], r["qa_target"],
                        data[i]["question"],
                        str(qa_w.get(f"{i}|{sfx}", ""))[:200],
                        str(qa_s.get(f"{i}|{sfx}", ""))[:200],
                        " | ".join(map(str, data[i].get("answers", [])))])
    print(f"wrote {OUT}/qa_labels.jsonl ({len(rows)} rows), manifest.json, "
          f"qa_label_review.csv ({REVIEW_N} sampled rows)")

if __name__ == "__main__":
    main()
