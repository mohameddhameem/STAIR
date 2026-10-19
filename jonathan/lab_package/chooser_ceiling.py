# 7 chooser_ceiling.py -- measure BEFORE spend (gate for 8's comparative lane).
# Question: on how many instances does the weak QA answer beat the strong one,
# within the same retrieval branch?  A perfect chooser can gain at most "prize";
# a blind always-weak policy gains "prize - danger" (net).  If even the ceiling
# is small, the chooser retires and 8's teacher-2 machinery is cancelled.
#
# Reads ONLY the cached grid (no GPU, no generation).  Cell order in
# em_lenient is [WW, WS, SW, SS] -- verified against build_ret_labels.py's
# own audit loop (zip(("WW","WS","SW","SS"), em)).  First letter = retrieval
# choice (W or S), second = QA choice.
import os, json, csv

GRID_FILE = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
QA_WEAK   = "/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl"    # weak QA model's answers
QA_STRONG = "/mnt/hdd/jonathan/runs/full_run_p100/qa_strong.jsonl"  # strong QA model's answers
DATA      = "/mnt/hdd/jonathan/data/tts_test.json"
OUT       = "/mnt/hdd/jonathan/runs/chooser_ceiling"
BAR_POINTS = 1.0   # retirement bar, in composed validation points (sign-off 1)

def load_store(path):               # same k/v jsonl format as run_labels4.Store, read-only
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
    grid = [g for g in grid if g["split"] in ("train", "val")]   # test never touched
    qa_w, qa_s = load_store(QA_WEAK), load_store(QA_STRONG)
    data = json.load(open(DATA))

    # buckets[split][branch][bucket] ; branch "W" pairs (WW vs WS), "S" pairs (SW vs SS)
    buckets = {sp: {br: {"prize": 0, "danger": 0, "indifferent": 0, "n": 0}
                    for br in ("W", "S")} for sp in ("train", "val")}
    review = []
    for g in grid:
        i, sp = g["idx"], g["split"]
        em = g["em_lenient"]                     # [WW, WS, SW, SS]
        for br, (wq, sq) in (("W", (em[0], em[1])), ("S", (em[2], em[3]))):
            b = buckets[sp][br]; b["n"] += 1
            if wq == 1 and sq == 0:   kind = "prize"    # weak QA beats strong QA
            elif wq == 0 and sq == 1: kind = "danger"   # picking weak would cost
            else:                     kind = "indifferent"
            b[kind] += 1
            if kind != "indifferent":                    # the disagreement slice -> CSV
                suffix = "weak" if br == "W" else "strong"
                review.append({
                    "idx": i, "split": sp, "ret_branch": br, "bucket": kind,
                    "question": g["question"],
                    "weak_qa_answer":   str(qa_w.get(f"{i}|{suffix}", ""))[:300],
                    "strong_qa_answer": str(qa_s.get(f"{i}|{suffix}", ""))[:300],
                    "gold_answers": " | ".join(map(str, data[i].get("answers", []))),
                })

    # report -----------------------------------------------------------------
    rep = {"bar_composed_val_points": BAR_POINTS, "cell_order": "WW,WS,SW,SS",
           "slices": {"all": "every train/val instance",
                      "disagreement": "prize+danger only (where a choice matters)"},
           "branches": {}}
    print(f"{'split':<6} {'branch':<6} {'n':>5} {'prize':>6} {'danger':>6} "
          f"{'indiff':>6} {'prize%':>7} {'net':>5}")
    for br in ("W", "S"):
        for sp in ("train", "val"):
            b = buckets[sp][br]
            net = b["prize"] - b["danger"]
            print(f"{sp:<6} {br:<6} {b['n']:>5} {b['prize']:>6} {b['danger']:>6} "
                  f"{b['indifferent']:>6} {100*b['prize']/b['n']:>6.2f}% {net:>5}")
        v = buckets["val"][br]
        perfect_pts = 100.0 * v["prize"] / v["n"]                 # perfect-chooser ceiling
        net_pts     = 100.0 * (v["prize"] - v["danger"]) / v["n"] # blind always-weak gain
        rep["branches"][br] = {
            "train": buckets["train"][br], "val": v,
            "val_perfect_ceiling_points": round(perfect_pts, 2),
            "val_net_points": round(net_pts, 2),
            "verdict_perfect_rule": "RETIRE" if perfect_pts < BAR_POINTS else "KEEP",
            "verdict_net_rule":     "RETIRE" if net_pts     < BAR_POINTS else "KEEP",
        }
        print(f"  -> branch {br} (val): perfect-chooser ceiling = {perfect_pts:.2f} pts "
              f"({rep['branches'][br]['verdict_perfect_rule']} at bar {BAR_POINTS}) | "
              f"net = {net_pts:.2f} pts ({rep['branches'][br]['verdict_net_rule']})")

    json.dump(rep, open(os.path.join(OUT, "report.json"), "w"), indent=2)
    with open(os.path.join(OUT, "chooser_review.csv"), "w", newline="",
              encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(review[0].keys()))
        w.writeheader(); w.writerows(review)
    print(f"\nwrote {OUT}/report.json and {OUT}/chooser_review.csv "
          f"({len(review)} disagreement rows)")

if __name__ == "__main__":
    main()
