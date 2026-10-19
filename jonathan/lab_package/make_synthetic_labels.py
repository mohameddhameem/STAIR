#!/usr/bin/env python3
# =============================================================================
# make_synthetic_labels.py — VERSION: make_synthetic_labels v1.0 2026-09-15
#
# AVENUE #4 (positive control): produce a twin of router_data_lenient.jsonl in
# which every label follows a SIMPLE RULE READABLE FROM THE PROMPT, so the
# UNCHANGED sft_router.py must learn it — the probe curves must CLIMB toward
# 1.0. If they do: the entire training path (loader, shuffle, batching,
# probes, adapter) is certified AT SCALE. If they stay flat: we found a real
# scale bug. Either way the run is decisive.
#
# THE RULE (deliberately trivial, stated openly for the record):
#   label = "W" if the question's first alphabetic character is in a..m
#           else "S"
# applied identically to ret_label, qa_label_W, qa_label_S. Every instance is
# resolved (all labels present). em_lenient is set to [1,1,1,1] so downstream
# test-metric code has consistent (trivial) ground truth — the OBJECT of this
# run is the probe curves, not the test table.
#
# TRAINER INVARIANCE: sft_router.py is not edited in any way. Recommended run
# uses --ret_input question (the INSTR_RET_QONLY prompt, a documented trainer
# flag) so retrieval prompts are short and 246 steps finish in hours, not
# days; the rule is readable from the question, so the flag does not weaken
# the control.
#
# USAGE (lab, CPU, seconds):
#   python3 make_synthetic_labels.py \
#     --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#     --out  /mnt/hdd/jonathan/runs/full_run_p100/router_data_synthetic.jsonl
# =============================================================================
import json, argparse, collections

VERSION = "make_synthetic_labels v1.0 2026-09-15"


def rule_label(question):
    for ch in question.lower():
        if ch.isalpha():
            return "W" if ch <= "m" else "S"
    return "W"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    print(VERSION, flush=True)

    n = 0
    dist = collections.Counter()
    with open(a.out, "w") as out:
        for l in open(a.data):
            r = json.loads(l)
            lab = rule_label(r.get("question", ""))
            r["ret_label"] = lab
            r["qa_label_W"] = lab
            r["qa_label_S"] = lab
            r["joint_label"] = {"W": "weak-weak", "S": "strong-strong"}[lab]
            r["joint_tie"] = False
            r["em_lenient"] = [1, 1, 1, 1]
            r["em_strict"] = [1, 1, 1, 1]
            out.write(json.dumps(r) + "\n")
            n += 1
            dist[lab] += 1
    share = 100 * dist["W"] / n
    print(f"wrote {n} rows -> {a.out}")
    print(f"label distribution: W {dist['W']} ({share:.1f}%) | S {dist['S']} ({100-share:.1f}%)")
    print("rule: W iff first alphabetic char of question <= 'm' (readable from every prompt)")
    print("NOTE: run the UNCHANGED sft_router.py on this file; probe curves must CLIMB.")


if __name__ == "__main__":
    main()
