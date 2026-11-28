#!/usr/bin/env python3
# tworouter_order_attempts.py v1.0 (28 Nov 2026) — READ-ONLY. Do the stored WEAK episodes contain mid-episode ORDER
# ATTEMPTS that the stage classifier counts as lookups?  The 28 Nov check showed type-9 weak actions of the form
# "GET …Observation?…\n\nPOST …ServiceRequest {…}" (an order written inside a lookup, rejected 400). If such attempts
# exist on the high-risk types, Run B can treat them as decision stages (Router 2 -> H -> doctor's order -> weak
# continues), which gives real AI -> doctor -> AI episodes. Writes nothing.
# USAGE: python3 tworouter_order_attempts.py --weak <…/medagentbench-std dir> [--strong <dir>]
import json, argparse, collections, re

def load(d): return {r["index"]: r for r in json.load(open(f"{d}/overall.json"))["custom"]["raw_results"]}

def rounds(rec):
    h = rec["history"]; out = []; k = 0
    for i, m in enumerate(h):
        if m["role"] != "agent": continue
        k += 1
        reply = h[i + 1]["content"] if i + 1 < len(h) and h[i + 1]["role"] == "user" else ""
        out.append((k, m["content"], reply))
    return out

def classify(action):
    a = action.strip()
    head = "FINISH" if a.startswith("FINISH") else "POST" if a.startswith("POST") else "GET" if a.startswith("GET") else "other"
    embedded = bool(re.search(r"(?m)^\s*POST\s+http", a[4:])) or ("\nPOST " in a) or (" POST http" in a and head != "POST")
    return head, (embedded and head != "POST")

def report(name, R):
    print(f"\n=== {name}: {len(R)} instances ===")
    print("type | n | instances with >=1 embedded order attempt | attempts | first-attempt round (min/median/max) | rounds after first attempt (median) | status of those instances")
    for t in range(1, 11):
        idxs = [i for i in sorted(R) if i // 30 + 1 == t]
        inst = []; n_att = 0
        for i in idxs:
            rs = rounds(R[i]); firsts = [k for k, a, _ in rs if classify(a)[1]]
            n_att += len(firsts)
            if firsts: inst.append((i, firsts[0], len(rs) - firsts[0], R[i]["status"]))
        if inst:
            fr = sorted(x[1] for x in inst); after = sorted(x[2] for x in inst)
            st = collections.Counter(x[3] for x in inst)
            print(f"  {t:2d} | {len(idxs)} | {len(inst):2d} | {n_att:3d} | {fr[0]}/{fr[len(fr)//2]}/{fr[-1]} | {after[len(after)//2]} | {dict(st)}")
        else:
            print(f"  {t:2d} | {len(idxs)} |  0 |   0 | - | - | -")
    # show the first embedded attempt verbatim (first 500 chars) for one instance of each type that has one
    shown = set()
    for i in sorted(R):
        t = i // 30 + 1
        if t in shown: continue
        for k, a, rep in rounds(R[i]):
            if classify(a)[1]:
                shown.add(t)
                print(f"\n--- example type {t}, instance index {i}, round {k} (action, first 500 chars) ---\n{a[:500]}\n--- reply (first 200 chars) ---\n{rep[:200]}")
                break

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weak", required=True); ap.add_argument("--strong", default=None)
    a = ap.parse_args()
    report("WEAK (7B)", load(a.weak))
    if a.strong: report("STRONG (72B)", load(a.strong))
