#!/usr/bin/env python3
# =============================================================================
# verify_scoring.py — VERSION: verify_scoring v1.1 2026-09-13 (join answers from combo_results_full.jsonl)
#
# ONE QUESTION: do the per-cell lenient verdicts STORED in the file agree with
# an INDEPENDENT reimplementation of lenient v1.2 (normalize + Unicode fold +
# whole-word containment + yes/no guard), recomputed from the stored answers
# and gold?  (All rows x 4 cells; the reimplementation below is written fresh,
# NOT imported from the mint, so the mint is not grading itself.)
#
# Expected result: 0 differing verdicts, or a short list of edge rows to
# hand-review together.
#
# USAGE (lab, CPU, ~1 minute):
#   python3 verify_scoring.py \
#       --data  /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#       --combo /mnt/hdd/jonathan/runs/full_run_p100/combo_results_full.jsonl
# =============================================================================
import json, re, argparse, unicodedata

VERSION = "verify_scoring v1.1 2026-09-13 (join answers from combo_results_full.jsonl)"
CELLS = ["WW", "WS", "SW", "SS"]
LIST_ORDER = ["WW", "WS", "SW", "SS"]
COMBO2CELL = {"weak-weak": "WW", "weak-strong": "WS", "strong-weak": "SW", "strong-strong": "SS"}

# ---------------- INDEPENDENT reimplementation of lenient v1.2 -----------------------
_ART = re.compile(r"\b(a|an|the)\b")
_PUNC = re.compile(r"[^0-9a-z\s]")
_WS = re.compile(r"\s+")
YES = {"yes"}; NO = {"no"}

def _norm(s):
    s = unicodedata.normalize("NFKD", str(s))
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.lower()
    s = s.replace("\u2018", "'").replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = s.replace("\u2013", "-").replace("\u2014", "-")
    s = _PUNC.sub(" ", s)
    s = _ART.sub(" ", s)
    return _WS.sub(" ", s).strip()

def em_lenient_indep(pred, gold):
    """Independent lenient v1.2: whole-word containment of normalized gold
    in normalized prediction; Unicode fold; yes/no guard (a yes/no gold must
    match as the answer's own yes/no stance, not merely appear as a word)."""
    p, g = _norm(pred), _norm(gold)
    if not g:
        return 0
    if g in YES or g in NO:
        toks = p.split()
        head = toks[0] if toks else ""
        if head in YES or head in NO:
            return int(head == g)
        return int(g in toks and (("yes" in toks) != ("no" in toks) or head == g))
    return int(re.search(r"(?<![0-9a-z])" + re.escape(g) + r"(?![0-9a-z])", p) is not None)




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--combo", required=True, help="combo_results_full.jsonl (answers + gold)")
    ap.add_argument("--show", type=int, default=10)
    a = ap.parse_args()
    print(VERSION, flush=True)
    rows = [json.loads(l) for l in open(a.data)]
    r0 = rows[0]
    if not (isinstance(r0.get("em_lenient"), list) and len(r0["em_lenient"]) == 4):
        print("SCHEMA STOP — first row keys:", sorted(r0.keys()))
        return
    answers = {}
    gold = {}
    n_combo = 0
    for l in open(a.combo):
        cr = json.loads(l)
        n_combo += 1
        answers[(cr["idx"], COMBO2CELL[cr["combo"]])] = cr["answer"]
        gold[cr["idx"]] = cr["gold"]
    print(f"rows: {len(rows)} | combo lines: {n_combo} | joined answers: {len(answers)}", flush=True)
    mm = []
    n = 0
    miss = 0
    for r in rows:
        i = r["idx"]
        em = {c: int(v) for c, v in zip(LIST_ORDER, r["em_lenient"])}
        for c in CELLS:
            if (i, c) not in answers:
                miss += 1
                continue
            n += 1
            mine = em_lenient_indep(answers[(i, c)], gold[i])
            if mine != em[c]:
                mm.append((i, c, str(answers[(i, c)])[:60], str(gold[i])[:40], em[c], mine))
    if miss:
        print(f"note: {miss} (idx, cell) pairs had no combo line")
    print(f"RESULT: {n-len(mm)}/{n} cell verdicts agree, {len(mm)} differ", flush=True)
    for m in mm[: a.show]:
        print(f"  differ idx {m[0]} {m[1]} | ans {m[2]!r} | gold {m[3]!r} | stored {m[4]} indep {m[5]}")
    print("VERDICT:", "SCORING CERTIFIED (0 differences)" if not mm else
          f"{len(mm)} verdicts to hand-review (rule-edge candidates)")


if __name__ == "__main__":
    main()
