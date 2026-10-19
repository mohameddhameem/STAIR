# rescore_their_pipeline_105.py -- fills the 2x2's empty cell (read-only, CPU).
# Scores the THEIR-PIPELINE stored answers (cluster_72b, Aug runs) with the
# STANDARD ruler on (a) all 500 -- must reproduce slide 11's ladder
# 44.4 / 54.0 / 64.4 / 72.2, certifying the file->path mapping -- and
# (b) the 105 rows of their 500 inside our test split (the comparison set).
import json, sys
sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")
from lenient_match_agenttts import LEN

DIR    = "/mnt/hdd/jonathan/runs/cluster_72b"
DATA   = "/mnt/hdd/jonathan/data/tts_test.json"
SPLITS = "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json"

data = json.load(open(DATA))
test = set(json.load(open(SPLITS))["splits"]["test"])
sub  = [i for i in range(500) if i in test]
print(f"105-set size: {len(sub)} (expect 105)")

FILES = [("WW  (q8_r7: 3B ans, weak retr)",   "q8_r7.jsonl",      44.4),
         ("WS  (q70_weak: 70B ans, weak)",    "q70_weak.jsonl",   54.0),
         ("SW  (q8_r72: 3B ans, strong)",     "q8_r72.jsonl",     64.4),
         ("SS  (q70_strong: 70B ans, strong)","q70_strong.jsonl", 72.2)]
for name, fn, exp in FILES:
    d = {r["idx"]: str(r["cands"][0]) for r in map(json.loads, open(f"{DIR}/{fn}"))}
    out = []
    for scope, idxs in (("all 500", [i for i in range(500) if i in d]),
                        ("the 105", [i for i in sub if i in d])):
        golds = lambda i: [str(g) for g in (data[i]["answers"] if isinstance(data[i]["answers"], list) else [data[i]["answers"]])]
        acc = sum(LEN(d[i], golds(i)) for i in idxs)
        out.append(f"{scope}: {100*acc/len(idxs):.2f} (n={len(idxs)})")
    print(f"{name}  [slide-11 expects {exp} on 500]  " + "  |  ".join(out))
