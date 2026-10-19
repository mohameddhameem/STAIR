# harmonize_config_a.py -- OPTION B: re-grade Config A's stored answers under the
# STANDARD (AgentTTS-comparable) ruler, so Config A and Config B share one measurement.
# Re-certifies the grid rule first (stops on any mismatch, nothing written);
# writes router_data_standard.jsonl NEXT TO the original; original untouched.
# USAGE: python harmonize_config_a.py /path/to/router_data_lenient.jsonl
import json, os, sys

from lenient_match_agenttts import LEN   # standard ruler  (certified: 74.6 = published 0.746)
from lenient_match import em             # grid rule       (certified: 0/29,620 vs stored grid)

DATA = "/mnt/hdd/jonathan/data/tts_test.json"
GRID = "/mnt/hdd/jonathan/runs/full_run_p100"

# Cell order of em_lenient, as stored: [WW, WS, SW, SS] = (retrieval, answering).
# The answer for cell (ret, qa) is the qa-model store at key "idx|weak" or
# "idx|strong" -- the suffix names the RETRIEVAL branch the QA model saw.
CELLS = [("W", "W"), ("W", "S"), ("S", "W"), ("S", "S")]
NAMES = ["WW (weak retrieval + weak answering)",
         "WS (weak retrieval + strong answering)",
         "SW (strong retrieval + weak answering)",
         "SS (strong retrieval + strong answering)"]

def load_store(path):
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["k"]] = r["v"]
    return d

def main():
    if len(sys.argv) != 2:
        sys.exit("usage: python harmonize_config_a.py /path/to/router_data_lenient.jsonl")
    RD = sys.argv[1]
    OUT = os.path.join(os.path.dirname(os.path.abspath(RD)), "router_data_standard.jsonl")
    if os.path.exists(OUT):
        sys.exit(f"refusing to overwrite existing {OUT} -- move it aside first")

    data = json.load(open(DATA))
    qa = {"W": load_store(os.path.join(GRID, "qa_weak.jsonl")),
          "S": load_store(os.path.join(GRID, "qa_strong.jsonl"))}

    rows, n, flips = [], 0, 0
    old_sum, new_sum, flip_sum = [0]*4, [0]*4, [0]*4
    by_split = {}
    for line in open(RD):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        i = r["idx"]
        golds = [str(g) for g in data[i]["answers"]]
        old = r["em_lenient"]
        assert len(old) == 4, f"idx {i}: em_lenient has {len(old)} cells, expected 4"
        new = []
        for c, (rc, qc) in enumerate(CELLS):
            sfx = "weak" if rc == "W" else "strong"
            ans = str(qa[qc][f"{i}|{sfx}"])
            g = em(ans, golds)
            assert g == old[c], (f"GRID-RULE REPLAY MISMATCH at idx {i}, cell {NAMES[c]}: "
                                 f"stored {old[c]}, replayed {g} -- stopping, nothing written")
            v = LEN(ans, golds)
            new.append(v)
            old_sum[c] += old[c]; new_sum[c] += v
            flip_sum[c] += (v != old[c]); flips += (v != old[c]); n += 1
        out = dict(r); out["em_standard"] = new
        rows.append(out)
        if "split" in r:
            s = by_split.setdefault(r["split"], {"n": 0, "old": [0]*4, "new": [0]*4})
            s["n"] += 1
            for c in range(4):
                s["old"][c] += old[c]; s["new"][c] += new[c]

    with open(OUT, "w") as f:
        for o in rows:
            f.write(json.dumps(o) + "\n")

    N = len(rows)
    print(f"re-certification PASSED: all {n} stored grid verdicts reproduced (0 mismatches)")
    print(f"instances={N}  verdicts={n}  standard-ruler flips={flips}\n")
    print("path accuracy, grid rule -> standard ruler (whole set):")
    for c in range(4):
        print(f"  {NAMES[c]}: {100*old_sum[c]/N:.2f} -> {100*new_sum[c]/N:.2f}   (flips {flip_sum[c]})")
    for sp in ("train", "val", "test"):
        if sp in by_split:
            s = by_split[sp]
            cells = "  ".join(f"{NAMES[c][:2]} {100*s['old'][c]/s['n']:.2f}->{100*s['new'][c]/s['n']:.2f}"
                              for c in range(4))
            print(f"split {sp} (n={s['n']}): {cells}")
    print(f"\nwrote {OUT}  (original untouched)")

if __name__ == "__main__":
    main()
