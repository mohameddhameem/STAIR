#!/usr/bin/env python3
# qa_case_review.py v1.0 — human-eyeball review of QA cases (21 Sep 2026)
# Purpose: the feature hunt (section H) found NO simple predictor of "weak answerer
# suffices" on strong evidence. Before concluding the signal is invisible, LOOK at cases.
# Prints N sampled TEST instances per class of the real strong-branch QA label:
#   qa_label_S = S  -> weak answerer FAILED on strong evidence (needs strong answerer)
#   qa_label_S = W  -> weak answerer SUFFICED on strong evidence
# For each: question, type, gold answer(s), gold titles, and the strong evidence excerpt
# (what both answerers actually read). Also prints which pipeline outcomes succeeded
# (em_lenient [WW,WS,SW,SS]) so ambiguous cases are visible.
# CPU-only, read-only, deterministic sampling (seeded).
import json, random, argparse, textwrap

def load_rows(path):
    return [json.loads(l) for l in open(path)]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
    ap.add_argument("--gold", default="/mnt/hdd/jonathan/data/tts_test.json")
    ap.add_argument("--splits", default="/mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json")
    ap.add_argument("--n", type=int, default=6, help="cases per class")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--ev_chars", type=int, default=700, help="evidence excerpt length")
    a = ap.parse_args()

    rows = load_rows(a.data)
    gold = json.load(open(a.gold))
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    test = set(sp["test"])

    S_cases = [r for r in rows if r["idx"] in test and r.get("qa_label_S") == "S"]
    W_cases = [r for r in rows if r["idx"] in test and r.get("qa_label_S") == "W"]
    rng = random.Random(a.seed)
    pickS = rng.sample(S_cases, min(a.n, len(S_cases)))
    pickW = rng.sample(W_cases, min(a.n, len(W_cases)))

    def show(r, tag):
        g = gold[r["idx"]]
        ans = " | ".join(map(str, g.get("answers", [])))
        ev = (r.get("evidence_S") or "")[:a.ev_chars].replace("\n", " ")
        em = r.get("em_lenient")
        print("=" * 100)
        print(f"[{tag}] idx {r['idx']}  type={g.get('type')}  em[WW,WS,SW,SS]={em}")
        print(f"Q: {r.get('question')}")
        print(f"GOLD ANSWER: {ans}   GOLD TITLES: {' | '.join(g.get('supporting_titles', []))}")
        print(f"STRONG EVIDENCE (first {a.ev_chars} chars of what both answerers read):")
        print(textwrap.fill(ev, width=98, initial_indent="   ", subsequent_indent="   "))

    print(f"### CLASS S — weak answerer FAILED on strong evidence (n={len(S_cases)} in test; showing {len(pickS)})")
    for r in pickS: show(r, "S: needs strong answerer")
    print()
    print(f"### CLASS W — weak answerer SUFFICED on strong evidence (n={len(W_cases)} in test; showing {len(pickW)})")
    for r in pickW: show(r, "W: weak suffices")
    print("=" * 100)
    print("Eyeball questions: 1) Do S-cases share anything visible (multi-step chains, yes/no")
    print("inference, answer buried late, longer evidence)? 2) Are there S-cases whose evidence")
    print("looks as clean as W-cases (candidate coin-flips / underivable)? 3) Any label surprises")
    print("(em row contradicting the label)?")

if __name__ == "__main__":
    main()
