#!/usr/bin/env python3
"""
twowiki_answer_analysis.py -- where could 2Wiki accuracy ABOVE always-SS come from, without new training? Read-only, CPU,
under a minute. Uses only answers already in the label stores (all four paths WW/WS/SW/SS per question). Never opens
test600. DRAFT v1.0 2026-11-10 for Jonathan's line-by-line review.

Goal (Jonathan, 12:10): "I still hope we can have 70% accuracy with adjustment." Always-SS is 65.75 % on val400 and the
oracle 77.75 %, so 70 % needs about 17 of the ~48 val questions that SS gets wrong but some other path gets right -- picked
WITHOUT gold. This script asks whether gold-free signals can find them.
Every policy below is a FIXED rule -- nothing is tuned -- so it is scored on val400 AND on train5000 (5,000 questions,
no router involved, far tighter error bars). Gold answers are used only to score.
  A1  Where the extra accuracy is: questions SS gets wrong, broken down by which other paths get them right.
  A2  Agreement as a confidence signal: SS accuracy when SS agrees with WS / with SW / with both, vs when it disagrees.
  A3  Voting over the four answers already produced (AgentTTS itself reaches 72 by sampling the answerer many times and
      fusing; this is the cheapest version of that idea). Cost = both retrievals + all four answers.
        V1 majority of the four answers, ties -> SS's answer
        V2 SS, unless WS and SW agree with each other and not with SS -> their answer
        V3 majority of the three answers that use at least one strong model (WS, SW, SS), ties -> SS
  A4  Agreement cascade (cheap path first, gold-free): run weak retrieval and both answers on it (WW, WS); if they agree,
      stop and output WS; otherwise run strong retrieval and output SS (C1) or vote over all four (C2).
  A5  Evidence potential of a "union" path not yet run: gold answer in weak picks / strong picks / their union.
  All of A1-A5 also by question type.
Agreement = the standard ruler applied between two answers (lenient_match_agenttts.LEN(a, [b]), symmetric).
Check built in: val400 lenient verdicts recomputed here must match the reviewed scorer's CSV exactly (else it stops).
Usage (lab):  python3 -I twowiki_answer_analysis.py
"""
import csv, json, os, re, string, importlib.util

ROOT = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged"
COMMON = "/mnt/hdd/jonathan/STAIR/common/stair_common"
OFFICIAL = ["/mnt/hdd/jonathan/data/2wiki_official/dev.json", "/mnt/hdd/jonathan/data/2wiki_official/train.json"]
P = {"retr": {"W": 7.0, "S": 72.0}, "qa": {"W": 3.0, "S": 70.0}}
CELLS = ["WW", "WS", "SW", "SS"]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m); return m


LEN = load("lenient_match_agenttts", os.path.join(COMMON, "lenient_match_agenttts.py")).LEN
units = load("stair_cost", os.path.join(COMMON, "cost.py")).units


def store(d, name):
    out = {}
    for line in open(os.path.join(d, name + ".jsonl"), encoding="utf-8"):
        if line.strip():
            r = json.loads(line); out[r["k"]] = r["v"] if "v" in r else r
    return out


def parse_selection(txt, n_refs):                                   # = twowiki_cells_score.parse_selection
    part = txt.split('Selected references')[-1] if 'Selected references' in txt else txt
    nums = [int(x) for x in re.findall(r'\[(\d+)\]', part)]
    return sorted(set(n for n in nums if 1 <= n <= n_refs))


def norm(s):
    s = s.lower(); s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s); return " ".join(s.split())


def contains(text, golds):                                          # = configb_build_ret_labels.contains
    nt = norm(text)
    return any(re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", nt) for g in golds if norm(g))


def agree(a, b):
    return bool(str(a).strip()) and bool(str(b).strip()) and bool(LEN(str(a), [str(b)]))


def vote(answers, fallback):
    """Greedy clustering by agreement; the biggest cluster wins; ties -> the cluster holding `fallback`."""
    clusters = []
    for a in answers:
        for c in clusters:
            if agree(a, c[0]):
                c.append(a); break
        else:
            clusters.append([a])
    top = max(len(c) for c in clusters)
    best = [c for c in clusters if len(c) == top]
    for c in best:
        if any(x is fallback for x in c):
            return fallback
    return best[0][0]


def qtypes():
    m = {}
    for p in OFFICIAL:
        if os.path.exists(p):
            for x in json.load(open(p, encoding="utf-8")):
                m[x["_id"]] = x["type"]
    return m


TYPEMAP = None


def build(split):
    global TYPEMAP
    d = os.path.join(ROOT, split)
    data = json.load(open(os.path.join(d, "tts_test_2wiki_adapted.json"), encoding="utf-8"))
    if TYPEMAP is None and not all("type" in x for x in data):
        TYPEMAP = qtypes()
    retr = {w: store(d, f"retr_{n}") for w, n in (("W", "weak"), ("S", "strong"))}
    qa = {w: store(d, f"qa_{n}") for w, n in (("W", "weak"), ("S", "strong"))}
    mr = {w: store(d, f"meta_retr_{n}") for w, n in (("W", "weak"), ("S", "strong"))}
    mq = {w: store(d, f"meta_qa_{n}") for w, n in (("W", "weak"), ("S", "strong"))}
    rows = []
    for i, inst in enumerate(data):
        pool, golds = inst["ctxs_candidate"], list(inst["answers"])
        br = {"W": "weak", "S": "strong"}
        ans = {r + a: str(qa[a][f"{i}|{br[r]}"]) for r in "WS" for a in "WS"}          # key = retrieval + answerer
        rW = units(P["retr"]["W"], mr["W"][str(i)]["n_in"], mr["W"][str(i)]["n_out"])
        rS = units(P["retr"]["S"], mr["S"][str(i)]["n_in"], mr["S"][str(i)]["n_out"])
        qc = {r + a: units(P["qa"][a], mq[a][f"{i}|{br[r]}"]["n_in"], mq[a][f"{i}|{br[r]}"]["n_out"]) for r in "WS" for a in "WS"}
        selW = parse_selection(retr["W"][str(i)], len(pool)); selS = parse_selection(retr["S"][str(i)], len(pool))
        rows.append({"i": i, "type": inst.get("type") or (TYPEMAP or {}).get(inst.get("_id"), "?"), "golds": golds, "ans": ans,
                     "ok": {c: int(LEN(ans[c], golds)) for c in CELLS},
                     "cost": {"rW": rW, "rS": rS, **qc},
                     "covW": int(contains(" ".join(pool[j - 1] for j in selW), golds)),
                     "covS": int(contains(" ".join(pool[j - 1] for j in selS), golds)),
                     "covU": int(contains(" ".join(pool[j - 1] for j in sorted(set(selW) | set(selS))), golds))})
    return rows


def pct(a, b): return f"{100 * a / b:5.1f}%" if b else "   - "


def report(name, rows):
    n = len(rows)
    C_SS = sum(r["cost"]["rS"] + r["cost"]["SS"] for r in rows) / n
    def cpct(x): return f"{100 * x / n / C_SS:5.1f}%"
    ok = lambda r, c: r["ok"][c]
    print(f"\n=================== {name}  (n={n}) ===================")
    print("cells: " + " · ".join(f"{c} {pct(sum(ok(r, c) for r in rows), n)}" for c in CELLS)
          + f" · oracle {pct(sum(max(r['ok'].values()) for r in rows), n)}")
    # A1
    wrong = [r for r in rows if not ok(r, "SS")]
    print(f"\nA1 SS wrong on {len(wrong)} ({pct(len(wrong), n)}); another path right on "
          f"{sum(max(r['ok'].values()) for r in wrong)}  -> by which paths:")
    combo = {}
    for r in wrong:
        k = "+".join(c for c in ("WW", "WS", "SW") if ok(r, c)) or "none"
        combo[k] = combo.get(k, 0) + 1
    for k, v in sorted(combo.items(), key=lambda x: -x[1]):
        print(f"     {k:12s} {v:5d}  ({pct(v, n)} of all questions)")
    # A2
    print("\nA2 SS accuracy by agreement (agreement is gold-free)")
    for lab, f in (("SS agrees with WS", lambda r: agree(r["ans"]["SS"], r["ans"]["WS"])),
                   ("SS agrees with SW", lambda r: agree(r["ans"]["SS"], r["ans"]["SW"])),
                   ("SS agrees with WS and SW", lambda r: agree(r["ans"]["SS"], r["ans"]["WS"]) and agree(r["ans"]["SS"], r["ans"]["SW"])),
                   ("WW agrees with WS (weak retrieval only)", lambda r: agree(r["ans"]["WW"], r["ans"]["WS"]))):
        y = [r for r in rows if f(r)]; nn = [r for r in rows if not f(r)]
        print(f"     {lab:42s} yes n={len(y):5d}: SS {pct(sum(ok(r, 'SS') for r in y), len(y))} WS {pct(sum(ok(r, 'WS') for r in y), len(y))}"
              f" | no n={len(nn):5d}: SS {pct(sum(ok(r, 'SS') for r in nn), len(nn))} WS {pct(sum(ok(r, 'WS') for r in nn), len(nn))}"
              f" SW {pct(sum(ok(r, 'SW') for r in nn), len(nn))}")
    # A3 / A4
    all4 = lambda r: r["cost"]["rW"] + r["cost"]["rS"] + sum(r["cost"][c] for c in CELLS)
    pol = {}
    pol["SS (always-strong)"] = [(ok(r, "SS"), r["cost"]["rS"] + r["cost"]["SS"]) for r in rows]
    def score_ans(r, a): return int(LEN(a, r["golds"]))
    pol["V1 majority of 4, ties->SS"] = [(score_ans(r, vote([r["ans"][c] for c in CELLS], r["ans"]["SS"])), all4(r)) for r in rows]
    def v2(r):
        a = r["ans"]
        return a["WS"] if agree(a["WS"], a["SW"]) and not agree(a["WS"], a["SS"]) else a["SS"]
    pol["V2 SS unless WS=SW!=SS"] = [(score_ans(r, v2(r)), all4(r)) for r in rows]
    pol["V3 majority of WS,SW,SS, ties->SS"] = [(score_ans(r, vote([r["ans"][c] for c in ("WS", "SW", "SS")], r["ans"]["SS"])),
                                                r["cost"]["rW"] + r["cost"]["rS"] + r["cost"]["WS"] + r["cost"]["SW"] + r["cost"]["SS"]) for r in rows]
    def c1(r):
        weak = r["cost"]["rW"] + r["cost"]["WW"] + r["cost"]["WS"]
        if agree(r["ans"]["WW"], r["ans"]["WS"]):
            return ok(r, "WS"), weak
        return ok(r, "SS"), weak + r["cost"]["rS"] + r["cost"]["SS"]
    pol["C1 WW=WS -> WS, else SS"] = [c1(r) for r in rows]
    def c2(r):
        weak = r["cost"]["rW"] + r["cost"]["WW"] + r["cost"]["WS"]
        if agree(r["ans"]["WW"], r["ans"]["WS"]):
            return ok(r, "WS"), weak
        return score_ans(r, vote([r["ans"][c] for c in CELLS], r["ans"]["SS"])), all4(r)
    pol["C2 WW=WS -> WS, else vote of 4"] = [c2(r) for r in rows]
    def c3(r):
        weak = r["cost"]["rW"] + r["cost"]["WW"] + r["cost"]["WS"]
        if agree(r["ans"]["WW"], r["ans"]["WS"]):
            return ok(r, "WS"), weak
        return score_ans(r, v2(r)), all4(r)
    pol["C3 WW=WS -> WS, else V2"] = [c3(r) for r in rows]
    print("\nA3/A4 policies (fixed rules, nothing tuned)               accuracy   cost % of SS   vs SS")
    ss = sum(x[0] for x in pol["SS (always-strong)"])
    for k, v in pol.items():
        a = sum(x[0] for x in v)
        print(f"     {k:44s} {pct(a, n)}     {cpct(sum(x[1] for x in v))}      {100 * (a - ss) / n:+5.2f}")
    # A5
    print(f"\nA5 gold answer in picks: weak {pct(sum(r['covW'] for r in rows), n)} · strong {pct(sum(r['covS'] for r in rows), n)}"
          f" · union {pct(sum(r['covU'] for r in rows), n)}  (union path = strong answer over both pick sets; not yet run)")
    # by type
    print("\nBY TYPE: n | SS | oracle | V1 | V2 | C1 acc/cost | covS | covU")
    for t in sorted({r["type"] for r in rows}):
        idx = [k for k, r in enumerate(rows) if r["type"] == t]; g = [rows[k] for k in idx]; m = len(g)
        cg = sum(r["cost"]["rS"] + r["cost"]["SS"] for r in g) / m
        f = lambda p: pct(sum(pol[p][k][0] for k in idx), m)
        print(f"     {t:18s} {m:5d} | {pct(sum(ok(r, 'SS') for r in g), m)} | {pct(sum(max(r['ok'].values()) for r in g), m)} | "
              f"{f('V1 majority of 4, ties->SS')} | {f('V2 SS unless WS=SW!=SS')} | "
              f"{f('C1 WW=WS -> WS, else SS')}/{100 * sum(pol['C1 WW=WS -> WS, else SS'][k][1] for k in idx) / m / cg:5.1f}% | "
              f"{pct(sum(r['covS'] for r in g), m)} | {pct(sum(r['covU'] for r in g), m)}")


# ---- val400 first, with the built-in check against the reviewed scorer -------------------------------------------
val = build("val400")
ref = {}
for r in csv.DictReader(open(os.path.join(ROOT, "val400", "scores_cells_per_instance.csv"))):
    ref[(int(r["idx"]), r["cell"])] = int(float(r["lenient"]))
mism = sum(ref[(r["i"], c)] != r["ok"][c] for r in val for c in CELLS)
print(f"CHECK: val400 lenient recomputed vs reviewed scorer CSV: {mism} mismatches out of {4 * len(val)}")
assert mism == 0, "verdicts differ from the reviewed scorer -- STOP"
report("VAL400", val)
report("TRAIN5000 (fixed rules only, no router -> usable for tighter estimates)", build("train5000"))
