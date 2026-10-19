# configb_make_grid.py -- builds Config B's grid file from the copied stores.
# CPU only. Mirrors harmonize_config_a.py's structure; verdicts minted with the
# STANDARD ruler (lenient_match_agenttts), per the 13 Oct registration.
#
# WRITES: /mnt/hdd/jonathan/runs/config_b/router_data_standard.jsonl
#   one row per instance: {idx, split, em_standard:[WW,WS,SW,SS],
#                          qa_label_W, qa_label_S}
#   qa_label rule (same as Config A's build_qa_labels):
#     under retrieval choice X: "W" if X-weak succeeds, else "S" if X-strong
#     succeeds, else null (packaged to X later).
#
# SELF-CHECK: prints the test-split ladder; must equal the origami numbers
#   WW 50.30 / WS 64.01 / SW 61.78 / SS 76.77  (n=1,481)
import json, os, sys

sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")
from lenient_match_agenttts import LEN      # the standard ruler, certified 74.6

DATA   = "/mnt/hdd/jonathan/data/tts_test.json"
RUN    = "/mnt/hdd/jonathan/runs/config_b"
SPLITS = "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json"
OUT    = os.path.join(RUN, "router_data_standard.jsonl")

CELLS = [("W","weak"), ("S","weak"), ("W","strong"), ("S","strong")]
# em order [WW, WS, SW, SS]: cell c = (answering model, retrieval branch)
#   WW = qa_weak  @ idx|weak    WS = qa_strong @ idx|weak
#   SW = qa_weak  @ idx|strong  SS = qa_strong @ idx|strong

def load_store(path):
    d = {}
    for line in open(path):
        r = json.loads(line); d[r["k"]] = r["v"]
    return d

def main():
    if os.path.exists(OUT):
        sys.exit(f"refusing to overwrite existing {OUT} -- move it aside first")
    data = json.load(open(DATA))
    split_of = {}
    for name, idxs in json.load(open(SPLITS))["splits"].items():
        for i in idxs: split_of[i] = name
    qa = {"W": load_store(os.path.join(RUN, "qa_weak.jsonl")),
          "S": load_store(os.path.join(RUN, "qa_strong.jsonl"))}

    rows, acc, n_test = [], [0]*4, 0
    for i, inst in enumerate(data):
        if i not in split_of: continue
        golds = [str(g) for g in (inst["answers"] if isinstance(inst["answers"], list) else [inst["answers"]])]
        em = []
        for qc, sfx in CELLS:
            key = f"{i}|{sfx}"
            assert key in qa[qc], f"idx {i}: missing {('weak' if qc=='W' else 'strong')} answer at {key}"
            em.append(LEN(str(qa[qc][key]), golds))
        lab_W = "W" if em[0] else ("S" if em[1] else None)   # weak retrieval branch
        lab_S = "W" if em[2] else ("S" if em[3] else None)   # strong retrieval branch
        rows.append({"idx": i, "split": split_of[i], "em_standard": em,
                     "qa_label_W": lab_W, "qa_label_S": lab_S})
        if split_of[i] == "test":
            n_test += 1
            for c in range(4): acc[c] += em[c]

    with open(OUT, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")

    print(f"rows written: {len(rows)}  (test n={n_test})")
    print("test-split ladder (standard ruler) -- must match origami:")
    for name, c in zip(("WW","WS","SW","SS"), range(4)):
        print(f"  {name}: {100*acc[c]/n_test:.2f}")
    print(f"wrote {OUT}")

if __name__ == "__main__":
    main()
