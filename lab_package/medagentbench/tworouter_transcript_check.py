#!/usr/bin/env python3
# tworouter_transcript_check.py v1.0 (28 Nov 2026) — READ-ONLY pre-registration check for the two-router run.
#
# Answers two questions from the stored episodes before any trainer exists:
#   (1) Stage-type structure per instance: how many rounds, how many information lookups (GET),
#       order decisions (POST) and final answers (FINISH) per task type.  Expected: every instance ends
#       in FINISH or stops at the round cap; types 5 and 9 show POST rounds; type 9 may show two.
#   (2) Reply cut: for every lookup round, is the key field of the record-system reply visible within the
#       first CUT characters?  Registered rule: CUT = 600 if >= 90 % of key-bearing replies pass, else 1000.
#
# Key fields (FHIR JSON): "valueQuantity"/"valueString"/"valueCodeableConcept" (observation values),
# "effectiveDateTime", "medicationCodeableConcept", "code", "name", "birthDate", "total", "id".
# A reply "bears a key" if any of these occurs anywhere in it; it "passes" if the first occurrence of ANY of
# them is within CUT characters.  Replies without any key (errors, empty bundles) are reported separately.
#
# USAGE (lab):  python3 tworouter_transcript_check.py --weak <WeakPilot7B_run1 dir> --strong <MedAgentBenchv1 dir>
#   dirs are the .../medagentbench-std folders holding overall.json.  Writes nothing.
import json, argparse, collections, statistics, re, sys

KEYS = ['"valueQuantity"', '"valueString"', '"valueCodeableConcept"', '"effectiveDateTime"',
        '"medicationCodeableConcept"', '"code"', '"name"', '"birthDate"', '"total"', '"id"']

def load(d):
    recs = json.load(open(f"{d}/overall.json"))["custom"]["raw_results"]
    return {r["index"]: r for r in recs}

def stage_type(action):
    a = action.strip()
    if a.startswith("FINISH"): return "final answer"
    if a.startswith("POST"):   return "order decision"
    if a.startswith("GET"):    return "information lookup"
    return "other"

def walk(rec):
    """Yield (round_no, action, reply) pairs from the history (agent action, following user reply or None)."""
    h = rec["history"]; out = []; k = 0
    for i, m in enumerate(h):
        if m["role"] != "agent": continue
        k += 1
        reply = h[i + 1]["content"] if i + 1 < len(h) and h[i + 1]["role"] == "user" else None
        out.append((k, m["content"], reply))
    return out

def first_key_pos(reply):
    pos = [reply.find(key) for key in KEYS]
    pos = [p for p in pos if p >= 0]
    return min(pos) if pos else None

def analyse(name, R, cuts=(400, 600, 1000, 2000)):
    per_type = collections.defaultdict(lambda: collections.Counter())
    rounds_per_type = collections.defaultdict(list)
    other_actions = collections.Counter()
    reply_lens = []; keypos = []; nokey = 0; nreply = 0
    multi_post = collections.Counter()
    for idx in sorted(R):
        tid = idx // 30 + 1
        steps = walk(R[idx])
        rounds_per_type[tid].append(len(steps))
        nposts = 0
        for k, action, reply in steps:
            st = stage_type(action); per_type[tid][st] += 1
            if st == "other": other_actions[action[:40]] += 1
            if st == "order decision": nposts += 1
            if st == "information lookup" and reply is not None:
                nreply += 1; reply_lens.append(len(reply))
                p = first_key_pos(reply)
                if p is None: nokey += 1
                else: keypos.append(p)
        multi_post[(tid, nposts)] += 1
    print(f"\n=== {name}: {len(R)} instances ===")
    print("task type | n | rounds min/mean/max | lookups | order decisions | final answers | other")
    for tid in sorted(per_type):
        rr = rounds_per_type[tid]; c = per_type[tid]
        print(f"  type {tid:2d} | {len(rr):3d} | {min(rr)}/{statistics.mean(rr):.1f}/{max(rr)} | "
              f"{c['information lookup']:4d} | {c['order decision']:4d} | {c['final answer']:4d} | {c['other']}")
    if other_actions: print("  'other' actions (first 40 chars):", dict(other_actions.most_common(5)))
    mp = {t: {n: v for (tt, n), v in multi_post.items() if tt == t and n} for t in sorted(per_type)}
    print("  instances with >=1 POST, by (type: {n_posts: count}):", {t: d for t, d in mp.items() if d})
    print(f"  lookup replies: {nreply}; without any key field: {nokey} ({100*nokey/max(nreply,1):.1f} %)")
    if reply_lens:
        q = statistics.quantiles(reply_lens, n=10)
        print(f"  reply length chars: median {int(statistics.median(reply_lens))}, p90 {int(q[8])}, max {max(reply_lens)}")
    if keypos:
        for cut in cuts:
            ok = sum(1 for p in keypos if p < cut)
            print(f"  key field within first {cut:4d} chars: {ok}/{len(keypos)} = {100*ok/len(keypos):.1f} %")
    return keypos

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--weak", required=True); ap.add_argument("--strong", required=True)
    a = ap.parse_args()
    W = load(a.weak); S = load(a.strong)
    kw = analyse("WEAK (7B) episodes", W)
    ks = analyse("STRONG (72B) episodes", S)
    both = kw  # the router sees WEAK transcripts; the rule is decided on those
    if both:
        ok600 = 100 * sum(1 for p in both if p < 600) / len(both)
        print(f"\nREGISTERED RULE on weak lookups: 600 passes {ok600:.1f} % -> CUT = {'600' if ok600 >= 90 else '1000'}")
