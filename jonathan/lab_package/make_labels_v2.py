#!/usr/bin/env python3
# =============================================================================
# make_labels_v2.py — VERSION: make_labels_v2 v1.0 2026-09-15
#
# OPTION D PILOT (stage-matched label construction) — Jonathan's direction:
# build a NEW training-label file router_data_v2.jsonl in which
#   - the RETRIEVAL target comes from the retrieval-selection OVERLAP signal
#     (measured 15 Sep: AUC test 0.746, monotone quintiles 51%->98% W-rate)
#   - the QA targets come from the zero-shot 7B JUDGE's qa scores
#     (measured: qa signal behind the 43.15 policy)
# The real measured file router_data_lenient.jsonl is NEVER modified; it
# remains the evaluation ground truth. This file is a DESIGNED artifact and
# says so in every row (label_source fields).
#
# DESIGN CHOICES — DELIBERATELY EXPLICIT, FOR JONATHAN'S REVIEW (each is a
# flag; defaults are proposals, not decisions):
#   D1 ret target form   --ret_target {hard,soft}   default hard
#       hard: ret_label = W if overlap >= --ret_tau else S
#       soft: ret_p_W = calibrated overlap quantile (trainer would need a
#             soft-target mode — NOT yet supported by sft_router v1.0, so
#             hard is the pilot default; soft recorded for the future)
#   D2 ret threshold     --ret_tau    default 0.50
#       0.50 ~ the empirical Q3/Q4 boundary; W-share at this tau ~ the real
#       train W-share (~65-69%), keeping class balance comparable.
#   D3 qa targets        --qa_source {judge,copy}   default judge
#       judge: qa_label_X = W if p_qaX_W >= --qa_tau else S  (X in {W,S})
#       copy : keep the REAL qa labels (isolates the ret-label change)
#   D4 qa threshold      --qa_tau     default 0.999
#       the saturation regime: the 43.15 policy's val-chosen taus were 1.0;
#       0.999 admits the near-certain mass. Review candidate: 1.0 exactly.
#   D5 coverage          every instance gets labels (overlap + judge exist
#       for all/scored rows). Rows WITHOUT judge scores (train split: the
#       zs run scored only val+test) fall back per --qa_fallback
#       {copy,skip}: copy = real qa labels where judge absent (default);
#       skip = qa labels blank there (QA phase then trains on val/test-scored
#       rows only — NOT possible under the frozen split; hence copy default).
#       *** This asymmetry is the kit's main design caveat — flagged. ***
#   D6 em fields: copied UNCHANGED from the real file (evaluation must stay
#       against measured truth; training reads labels, not ems).
#
# EVERY ROW records: label_source_ret ("overlap>=tau"/...), overlap value,
# label_source_qa, and the original real labels under real_* keys — full
# provenance, reversible, auditable.
#
# USAGE (lab, CPU, ~1 min):
#   python3 make_labels_v2.py \
#     --data   /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#     --scores /mnt/hdd/jonathan/runs/zs_evidence/router_scores.jsonl \
#     --out    /mnt/hdd/jonathan/runs/full_run_p100/router_data_v2.jsonl
# =============================================================================
import json, re, argparse, collections

VERSION = "make_labels_v2 v1.0 2026-09-15"


def pool_title_index(pool_text):
    out = {}
    for m in re.finditer(r"^\[(\d+)\]\s*([^:\n]{2,80}):", pool_text, flags=re.M):
        out[m.group(2).strip()] = int(m.group(1))
    return out


def selection_set(evidence, titles, n_pool):
    blob = str(evidence)
    sel = set()
    part = blob.split("Selected references", 1)[-1]
    for m in re.findall(r"\[(\d+)\]", part):
        k = int(m)
        if 1 <= k <= (n_pool or 10**9):
            sel.add(k)
    for t, k in titles.items():
        if t in blob:
            sel.add(k)
    return sel


def jaccard(a, b):
    u = a | b
    return len(a & b) / len(u) if u else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scores", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ret_target", choices=["hard", "soft"], default="hard")
    ap.add_argument("--ret_tau", type=float, default=0.50)
    ap.add_argument("--qa_source", choices=["judge", "copy"], default="judge")
    ap.add_argument("--qa_tau", type=float, default=0.999)
    ap.add_argument("--qa_fallback", choices=["copy", "skip"], default="copy")
    a = ap.parse_args()
    print(VERSION, flush=True)

    J = {}
    for l in open(a.scores):
        j = json.loads(l)
        J[j["idx"]] = j

    n = 0
    dist = collections.Counter()
    qa_src_count = collections.Counter()
    with open(a.out, "w") as out:
        for l in open(a.data):
            r = json.loads(l)
            titles = pool_title_index(r.get("pool_text", ""))
            sw = selection_set(r.get("evidence_W", ""), titles, r.get("n_pool"))
            ss = selection_set(r.get("evidence_S", ""), titles, r.get("n_pool"))
            ov = jaccard(sw, ss)
            # preserve the real labels under real_* (provenance, evaluation)
            for k in ("ret_label", "qa_label_W", "qa_label_S", "joint_label"):
                r["real_" + k] = r.get(k)
            # D1/D2: ret target from overlap
            if ov is None:
                r["ret_label"] = r["real_ret_label"]      # no selections parsed: fall back
                r["label_source_ret"] = "fallback_real(no_overlap)"
            else:
                r["ret_label"] = "W" if ov >= a.ret_tau else "S"
                r["label_source_ret"] = f"overlap>={a.ret_tau}"
            r["overlap_jaccard"] = None if ov is None else round(ov, 4)
            # D3/D4/D5: qa targets
            j = J.get(r["idx"])
            if a.qa_source == "judge" and j is not None:
                r["qa_label_W"] = "W" if j.get("p_qaW_W", 0) >= a.qa_tau else "S"
                r["qa_label_S"] = "W" if j.get("p_qaS_W", 0) >= a.qa_tau else "S"
                r["label_source_qa"] = f"judge>={a.qa_tau}"
            elif a.qa_source == "judge" and j is None and a.qa_fallback == "skip":
                r["qa_label_W"] = None
                r["qa_label_S"] = None
                r["label_source_qa"] = "skipped(no_judge_score)"
            else:
                r["qa_label_W"] = r["real_qa_label_W"]
                r["qa_label_S"] = r["real_qa_label_S"]
                r["label_source_qa"] = "copy_real" + ("(no_judge_score)" if a.qa_source == "judge" else "")
            # joint label: derived from the new stage labels for consistency
            if r["ret_label"] and (r["qa_label_W"] if r["ret_label"] == "W" else r["qa_label_S"]):
                qa_pick = r["qa_label_W"] if r["ret_label"] == "W" else r["qa_label_S"]
                r["joint_label"] = {"W": "weak-", "S": "strong-"}[r["ret_label"]] + {"W": "weak", "S": "strong"}[qa_pick]
            out.write(json.dumps(r) + "\n")
            n += 1
            dist[r["ret_label"]] += 1
            qa_src_count[r["label_source_qa"]] += 1
    print(f"wrote {n} rows -> {a.out}")
    print(f"ret label dist: {dict(dist)}  (W-share {100*dist['W']/n:.1f}%)")
    print(f"qa label sources: {dict(qa_src_count)}")
    print("real labels preserved under real_* keys; em fields unchanged (evaluation truth intact)")
    print(f"design: ret={a.ret_target}@tau {a.ret_tau} | qa={a.qa_source}@tau {a.qa_tau} | fallback={a.qa_fallback}")


if __name__ == "__main__":
    main()
