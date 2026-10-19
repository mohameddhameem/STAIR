#!/usr/bin/env python3
# make_teacher1_data.py v1.1 — build router TRAINING data from verifier scores (teacher #1)
# (1) qa_label_S = W if P(supported) >= tau else S  (verifier-derived labels)
# (2) INPUT MODE (b): appends the weak answerer's produced answer to evidence_S as a
#     labeled block, so the router SEES the candidate answer — with ZERO trainer changes
#     (the certified QA prompt renders evidence_S verbatim). This file is training/eval
#     specific: never merge its evidence_S back into the base grid.
# Keeps ret_label (probe safety), blanks qa_label_W (S-branch campaign), prints floors.
import json, argparse, collections
ap = argparse.ArgumentParser()
ap.add_argument("--grid", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
ap.add_argument("--scores", required=True, help="verifier scores jsonl covering the train split")
ap.add_argument("--tau", type=float, default=0.999, help="keep-weak threshold (default 0.999 per fine sweep)")
ap.add_argument("--answers", default="/mnt/hdd/jonathan/runs/full_run_p100/qa_weak.jsonl")
ap.add_argument("--no_answer_input", action="store_true", help="input mode (a): do NOT append the candidate answer (controlled comparison)")
ap.add_argument("--out", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_teacher1.jsonl")
a = ap.parse_args()
smap = {}
for path in a.scores.split(","):
    for l in open(path.strip()):
        r = json.loads(l); smap[r["idx"]] = r["p_supported"]
amap = {}
for l in open(a.answers):
    r = json.loads(l); i, br = r["k"].split("|")
    if br == "strong": amap[int(i)] = (r.get("v") or "").strip()
cnt = collections.Counter()
with open(a.out, "w") as f:
    for l in open(a.grid):
        r = json.loads(l)
        sc = smap.get(r["idx"])
        r["qa_label_W"] = None
        if not a.no_answer_input:
            ans = amap.get(r["idx"], "") or "(no answer)"
            r["evidence_S"] = r["evidence_S"] + "\n\n[Candidate answer produced by the fast answerer]: " + ans + "\n(Assess this answer against the evidence when making your decision.)"
        if sc is not None:
            r["qa_label_S"] = "W" if sc >= a.tau else "S"
            cnt[(r["split"], r["qa_label_S"])] += 1
        f.write(json.dumps(r) + "\n")
print("wrote", a.out)
for s in ("train", "val", "test"):
    W, S = cnt[(s, "W")], cnt[(s, "S")]
    if W + S: print(f"{s:5s}: teacher1 labels  W={W} S={S}  W-share(floor)={W/(W+S):.4f}")
