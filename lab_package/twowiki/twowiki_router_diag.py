#!/usr/bin/env python3
"""
twowiki_router_diag.py -- why did the 2Wiki fresh-adapter router fail? Diagnosis on VALIDATION ONLY (val400), read-only,
CPU, seconds. Trains nothing, scores no new model output, never opens test600. DRAFT v1.0 2026-11-10 for Jonathan's
line-by-line review.

Question (Jonathan, 11:57): "weak model's retrieved passages don't give better training signal for the router."
Four checks, each answering one part of it:
  D1  Retrieval router: does its score pW_ret separate "weak retrieval was ENOUGH" (WW or WS correct) from "not enough"?
      AUC (0.5 = no signal, 1.0 = perfect). Also on the questions where SOME path is correct (the only ones that matter).
  D2  Is the evidence in the weak picks? covW = a gold answer appears in the text of the weak retriever's selected
      passages (same whole-word containment as configb_build_ret_labels.py); covS the same for strong picks.
      Cross-tab covW against "weak retrieval enough", and AUC of pW_ret against covW (does the router see the evidence?).
  D3  Answering router on the STRONG-retrieval branch (where 225 test questions went SW): does pW_qa(S) separate
      "weak answer is as good as strong" (SW correct, or SW = SS) from "strong answer needed" (SS right, SW wrong)?
      Same for the weak branch (WW vs WS).
  D4  By 2Wiki question type: cell accuracies, oracle shares, router behaviour at the locked point, D1 AUC.
  Extra (gold-free signal for the lead enhancement idea): does the WEAK ANSWER on the weak branch -- which exists before
      any strong call -- carry the signal? Rate of "weak retrieval enough" when the weak answer abstains vs answers.
Gold answers are used here for DIAGNOSIS ONLY (as in the label builder's audits); none enters any router input.
Inputs (lab): labels6000_merged/val400/{tts_test_2wiki_adapted.json, retr_*.jsonl, qa_weak.jsonl,
              scores_cells_per_instance.csv}; the run's {ret,qa}_val_scores.jsonl and LOCKED_thresholds.json;
              official 2Wiki json(s) for question type if the records lack it.
Output: printed report + <RUN>/diag_val400.csv (one row per question, opens in Excel).
Usage (lab):  python3 -I twowiki_router_diag.py
"""
import csv, json, os, re, string

L   = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/val400"
RUN = "/mnt/hdd/jonathan/runs/twowiki/grpo_2w_a1_fresh_ep6_objective"
OFF = 5000                                                       # staged idx of val400 position 0
OFFICIAL = ["/mnt/hdd/jonathan/data/2wiki_official/dev.json",     # for question type, joined by _id
            "/mnt/hdd/jonathan/data/2wiki_official/train.json"]
CELLS = ["WW", "WS", "SW", "SS"]


def store(name):
    d = {}
    for line in open(os.path.join(L, name + ".jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def parse_selection(txt, n_refs):                                 # = twowiki_cells_score.parse_selection
    part = txt.split('Selected references')[-1] if 'Selected references' in txt else txt
    nums = [int(x) for x in re.findall(r'\[(\d+)\]', part)]
    return sorted(set(n for n in nums if 1 <= n <= n_refs))


def norm(s):                                                      # = configb_build_ret_labels.norm
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(text, golds):                                        # = configb_build_ret_labels.contains
    nt = norm(text)
    return any(re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", nt) for g in golds if norm(g))


def auc(scores, labels):
    """Mann-Whitney AUC: P(score of a positive > score of a negative), ties count half. None if one class is empty."""
    pos = [s for s, y in zip(scores, labels) if y]; neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    allv = sorted((s, y) for s, y in zip(scores, labels))
    ranks, i = {}, 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        for k in range(i, j):
            ranks.setdefault(allv[k][0], (i + j + 1) / 2)
        i = j
    rp = sum(ranks[s] for s in pos)
    return round((rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)), 3)


ABSTAIN = re.compile(r"\b(unknown|not (?:mentioned|provided|available|given|specified|enough)|cannot|can't|no information|"
                     r"insufficient|unable|none)\b", re.I)

# ---- load -----------------------------------------------------------------------------------------------------------
data = json.load(open(os.path.join(L, "tts_test_2wiki_adapted.json"), encoding="utf-8"))
assert len(data) == 400
ver = {}
for r in csv.DictReader(open(os.path.join(L, "scores_cells_per_instance.csv"))):
    ver.setdefault(int(r["idx"]), {})[r["cell"]] = int(float(r["lenient"]))
retr = {"W": store("retr_weak"), "S": store("retr_strong")}
qa_w = store("qa_weak")
ret = {r["idx"]: r["pW"] for r in map(json.loads, open(os.path.join(RUN, "ret_val_scores.jsonl")))}
qa = {(r["idx"], r["ret_choice"]): r["pW"] for r in map(json.loads, open(os.path.join(RUN, "qa_val_scores.jsonl")))}
lock = json.load(open(os.path.join(RUN, "LOCKED_thresholds.json")))
tr, tw, ts = lock["t_ret"], lock["t_qaW"], lock["t_qaS"]

qtype = {}
if not all("type" in x for x in data):
    for p in OFFICIAL:
        if os.path.exists(p):
            for x in json.load(open(p, encoding="utf-8")):
                qtype[x["_id"]] = x["type"]

rows = []
for i, inst in enumerate(data):
    pool, golds = inst["ctxs_candidate"], list(inst["answers"])
    selW = parse_selection(retr["W"][str(i)], len(pool)); selS = parse_selection(retr["S"][str(i)], len(pool))
    e = ver[i]; k = OFF + i
    pr, pqw, pqs = ret[k], qa[(k, "W")], qa[(k, "S")]
    c = (0 if pqw >= tw else 1) if pr >= tr else (2 if pqs >= ts else 3)
    wans = str(qa_w[f"{i}|weak"])
    rows.append({"idx": i, "type": inst.get("type") or qtype.get(inst.get("_id"), "?"),
                 **{c_: e[c_] for c_ in CELLS}, "any": int(any(e.values())),
                 "weak_ret_enough": int(e["WW"] or e["WS"]),
                 "covW": int(contains(" ".join(pool[j - 1] for j in selW), golds)),
                 "covS": int(contains(" ".join(pool[j - 1] for j in selS), golds)),
                 "n_selW": len(selW), "n_selS": len(selS),
                 "pW_ret": pr, "pW_qa_onW": pqw, "pW_qa_onS": pqs, "locked_cell": CELLS[c],
                 "weak_answer_onW": wans, "weak_answer_abstains": int(bool(ABSTAIN.search(wans)) or not wans.strip())})
n = len(rows)


def pct(a, b): return f"{100 * a / b:.1f}%" if b else "-"


print(f"VAL400 diagnosis (n={n}) -- validation only, test600 untouched")
print(f"cells: " + " · ".join(f"{c_} {pct(sum(r[c_] for r in rows), n)}" for c_ in CELLS)
      + f" · any-correct {pct(sum(r['any'] for r in rows), n)}")

# ---- D1 -------------------------------------------------------------------------------------------------------------
enough = [r["weak_ret_enough"] for r in rows]
print(f"\nD1 retrieval router  -- 'weak retrieval enough' (WW or WS correct) on {sum(enough)}/{n} questions")
print(f"   AUC pW_ret vs enough, all questions        : {auc([r['pW_ret'] for r in rows], enough)}")
sub = [r for r in rows if r["any"]]
print(f"   AUC pW_ret vs enough, any-correct only (n={len(sub)}): {auc([r['pW_ret'] for r in sub], [r['weak_ret_enough'] for r in sub])}")
print(f"   pW_ret mean when enough {sum(r['pW_ret'] for r in rows if r['weak_ret_enough'])/max(1,sum(enough)):.3f} "
      f"| when not {sum(r['pW_ret'] for r in rows if not r['weak_ret_enough'])/max(1,n-sum(enough)):.3f}")

# ---- D2 -------------------------------------------------------------------------------------------------------------
print(f"\nD2 evidence in the picks -- covW {pct(sum(r['covW'] for r in rows), n)} · covS {pct(sum(r['covS'] for r in rows), n)}"
      f" · mean picks W {sum(r['n_selW'] for r in rows)/n:.2f} / S {sum(r['n_selS'] for r in rows)/n:.2f}")
for cw in (1, 0):
    g = [r for r in rows if r["covW"] == cw]
    print(f"   covW={cw}: n={len(g):3d} | weak retrieval enough {pct(sum(r['weak_ret_enough'] for r in g), len(g))}"
          f" | WS {pct(sum(r['WS'] for r in g), len(g))} | SS {pct(sum(r['SS'] for r in g), len(g))}")
print(f"   AUC pW_ret vs covW (does the router see whether the answer is in its picks?): "
      f"{auc([r['pW_ret'] for r in rows], [r['covW'] for r in rows])}")

# ---- D3 -------------------------------------------------------------------------------------------------------------
print("\nD3 answering router")
g = [r for r in rows if r["SS"] or r["SW"]]
lab = [int(r["SW"] >= r["SS"]) for r in g]
print(f"   strong branch (n={len(g)} where SW or SS right): weak answer enough {pct(sum(lab), len(g))}"
      f" | AUC pW_qa_onS {auc([r['pW_qa_onS'] for r in g], lab)}")
g = [r for r in rows if r["WW"] or r["WS"]]
lab = [int(r["WW"] >= r["WS"]) for r in g]
print(f"   weak branch   (n={len(g)} where WW or WS right): weak answer enough {pct(sum(lab), len(g))}"
      f" | AUC pW_qa_onW {auc([r['pW_qa_onW'] for r in g], lab)}")
sw = [r for r in rows if r["locked_cell"] == "SW"]
print(f"   locked point sent {len(sw)} to SW: SW right {pct(sum(r['SW'] for r in sw), len(sw))}"
      f" vs SS right {pct(sum(r['SS'] for r in sw), len(sw))} on those questions")

# ---- Extra: the weak answer as a gold-free signal --------------------------------------------------------------------
print("\nEXTRA weak answer on the weak branch (exists before any strong call)")
for ab in (0, 1):
    g = [r for r in rows if r["weak_answer_abstains"] == ab]
    print(f"   abstains={ab}: n={len(g):3d} | weak retrieval enough {pct(sum(r['weak_ret_enough'] for r in g), len(g))}"
          f" | covW {pct(sum(r['covW'] for r in g), len(g))}")

# ---- D4 -------------------------------------------------------------------------------------------------------------
print("\nD4 by question type")
print(f"   {'type':18s} {'n':>4s} " + " ".join(f"{c_:>6s}" for c_ in CELLS)
      + f" {'enough':>7s} {'covW':>6s} {'AUC_D1':>7s} {'locked W-ret':>12s}")
for t in sorted({r["type"] for r in rows}):
    g = [r for r in rows if r["type"] == t]
    print(f"   {t:18s} {len(g):4d} " + " ".join(f"{pct(sum(r[c_] for r in g), len(g)):>6s}" for c_ in CELLS)
          + f" {pct(sum(r['weak_ret_enough'] for r in g), len(g)):>7s} {pct(sum(r['covW'] for r in g), len(g)):>6s}"
          + f" {str(auc([r['pW_ret'] for r in g], [r['weak_ret_enough'] for r in g])):>7s}"
          + f" {pct(sum(r['locked_cell'] in ('WW', 'WS') for r in g), len(g)):>12s}")

out = os.path.join(RUN, "diag_val400.csv")
with open(out, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
print(f"\nper-question table: {out}")
