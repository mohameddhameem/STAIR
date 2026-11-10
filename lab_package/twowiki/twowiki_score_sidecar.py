#!/usr/bin/env python3
"""
twowiki_score_sidecar.py -- authoritative scores for Row A ("AgentTTS (reproduced), their 100") and Row C (dev[500:1000])
from the sidecar that the two import shims wrote during their archon_main.py run. No model, no GPU, no network.
v1.1 2026-11-10 (DRAFT for Jonathan's line-by-line review) -- v1.0 (31 Oct) + four changes agreed 10 Nov 12:20:
  [1] ALIGNMENT: besides the gold-list check (weak: many 2Wiki golds are just "yes"/"no"), the retrieval record's gold
      passage ids must equal instance i's gold-passage positions in its own pool (instance-specific). The id convention
      (1-based or 0-based) is read from the data: one convention must match on EVERY record, else the script stops
      (override only with --skip-align-check, which is printed into the summary).
  [2] TYPES: their data files carry no `type`; --types reads their100_types.json ({question: type} or
      {"question_to_type": {...}}) and/or --official reads the official 2Wiki dev.json (joined by question text).
  [3] PYTHON 3.10: stair_common/score.py is loaded directly from its file (the package __init__ needs tomllib); the
      ruler's sha256 is checked against score.RULER_SHA256 before anything is scored.
  [4] LENGTH-FAIRNESS AUDIT, reported NEXT TO the certified headline, never replacing it. The fuser writes a paragraph
      (median ~590 chars); the standard ruler credits a gold that appears ANYWHERE in the output. Per instance:
        lenient_first_sentence  the standard ruler on the output's first sentence only
        yesno_both              gold is yes/no and the output contains both the word "yes" and the word "no"
        options_both            the question offers two options ("..., A or B?") and the output names both
        lenient_unambiguous     lenient AND NOT (yesno_both OR options_both)
      Purpose: how much of the 70.0 -- and of the gap to our short-answer cells -- is paragraph-length credit.
UNCHANGED from v1.0: certified rulers (lenient / em / f1 from stair_common.score), Ret F1, record pairing, count checks.
v1.0 outputs are kept; v1.1 writes scores_v11_per_instance.csv (+ .xlsx) and scores_v11_summary.json.

USAGE (lab or cluster):
  python3 -I twowiki_score_sidecar.py --run /mnt/hdd/jonathan/runs/twowiki/repro_their100 \
      --data /mnt/hdd/jonathan/agenttts_verify/data/2wikimultihopqa/tts_test.json \
      --types /mnt/hdd/jonathan/STAIR/common/stair_common/splits/2wikimultihopqa/their100_types.json \
      [--official /mnt/hdd/jonathan/data/2wiki_official/dev.json] [--em-gold aliases|single]
"""
import argparse, collections, csv, hashlib, importlib.util, json, os, re

COMMON_CANDIDATES = [os.path.join(os.environ.get("STAIR_ROOT", ""), "common"), "/mnt/hdd/jonathan/STAIR/common",
                     os.path.expanduser("~/slr/STAIR/common")]


def sha256(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_score_module():                                                     # [3]
    for c in COMMON_CANDIDATES:
        p = os.path.join(c, "stair_common", "score.py")
        if c and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("stair_score", p)
            m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
            got = sha256(str(m.RULER))
            assert got == m.RULER_SHA256, f"ruler sha256 {got} != certified {m.RULER_SHA256} -- STOP"
            return m, p
    raise SystemExit("stair_common/score.py not found")


S, SCORE_PATH = load_score_module()
lenient, em, f1 = S.lenient, S.em, S.f1


def ret_f1(gt_ids, pred_ids):
    gt, pred = set(gt_ids), set(pred_ids)
    inter = len(gt & pred)
    if not inter:
        return 0.0
    p, r = inter / len(pred), inter / len(gt)
    return 2 * p * r / (p + r)


# ---- [4] audit helpers ---------------------------------------------------------------------------------------------
def first_sentence(s):
    parts = [p for p in re.split(r"(?<=[.!?])\s+|\n+", s.strip()) if p.strip()]
    return parts[0] if parts else ""


def word_set(s):
    return set(re.findall(r"[a-z0-9]+", s.lower()))


OPT = re.compile(r",\s*(.+?)\s+or\s+(.+?)\s*\?\s*$", re.I)


def options(q):
    m = OPT.search(q)
    return (m.group(1).strip(), m.group(2).strip()) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--types", default=None)
    ap.add_argument("--official", default=None)
    ap.add_argument("--em-gold", choices=["aliases", "single"], default="aliases")
    ap.add_argument("--skip-align-check", action="store_true")
    a = ap.parse_args()

    data = json.load(open(a.data, encoding="utf-8"))
    rows = [json.loads(l) for l in open(os.path.join(a.run, "sidecar.jsonl"), encoding="utf-8") if l.strip()]
    answers = [r for r in rows if r["kind"] == "answer"]
    retrievals = [r for r in rows if r["kind"] == "retrieval"]
    n = len(answers)
    assert n == len(retrievals), f"{n} answer records vs {len(retrievals)} retrieval records"
    assert n <= len(data), f"{n} records but only {len(data)} instances in the data file"
    if n < len(data):
        print(f"WARNING: {n} of {len(data)} instances scored (smoke or partial run) -- not a reportable row")

    # ---- [2] question types ----
    qtype = {}
    if a.types:
        t = json.load(open(a.types, encoding="utf-8")); qtype.update(t.get("question_to_type", t))
    if a.official:
        for x in json.load(open(a.official, encoding="utf-8")):
            qtype.setdefault(x["question"], x["type"])

    # ---- [1] alignment by gold-passage ids ----
    def gold_pos(ex, base):
        pool = ex["ctxs_candidate"]
        return sorted(pool.index(g) + base for g in ex["ctxs_gt"] if g in pool)
    align = {}
    for base in (1, 0):
        align[base] = sum(sorted(ret["gt_ids"]) == gold_pos(data[i], base) for i, ret in enumerate(retrievals))
    conv = next((b for b in (1, 0) if align[b] == n), None)
    print(f"ALIGNMENT (gold-passage ids, instance-specific): 1-based {align[1]}/{n} · 0-based {align[0]}/{n}"
          f" -> {'OK (' + str(conv) + '-based)' if conv is not None else 'NOT CONFIRMED'}")
    if conv is None and not a.skip_align_check:
        raise SystemExit("STOP: no id convention matches every record -- inspect before scoring (or --skip-align-check)")

    per = []
    for i, (ans, ret) in enumerate(zip(answers, retrievals)):
        ex = data[i]
        golds = [ex["answers"]] if isinstance(ex["answers"], str) else list(ex["answers"])
        assert ans["gold_answers"] == golds, f"instance {i}: sidecar gold list differs from the data file -- misaligned"
        out = ans["response"]
        qt = ex.get("type") or qtype.get(ex["question"])
        ws = word_set(out)
        yn = S._ruler().norm(golds[0]) in ("yes", "no")
        yesno_both = int(yn and "yes" in ws and "no" in ws)
        opts = options(ex["question"])
        options_both = int(bool(opts) and all(lenient(out, [o]) for o in opts))
        len_ = lenient(out, golds)
        per.append({
            "idx": i, "type": qt, "question": ex["question"], "gold": golds[0], "n_aliases": len(golds) - 1,
            "output": out, "output_chars": len(out),
            "lenient": len_,
            "em": em(out, golds if a.em_gold == "aliases" else golds[0]),
            "qa_f1": round(f1(out, golds), 4),
            "ret_f1": round(ret_f1(ret["gt_ids"], ret["pred_ids"]), 4),
            "ret_gt_ids": json.dumps(ret["gt_ids"]), "ret_pred_ids": json.dumps(ret["pred_ids"]),
            "lenient_first_sentence": lenient(first_sentence(out), golds),                       # [4]
            "yesno_both": yesno_both, "options_both": options_both,
            "lenient_unambiguous": int(len_ and not (yesno_both or options_both)),
        })

    K = ("lenient", "em", "qa_f1", "ret_f1", "lenient_first_sentence", "lenient_unambiguous", "yesno_both", "options_both")

    def summ(group):
        m = len(group)
        return {"n": m, **{k: round(100 * sum(g[k] for g in group) / m, 2) for k in K},
                "median_output_chars": sorted(g["output_chars"] for g in group)[m // 2]}

    by_type = collections.defaultdict(list)
    for g in per:
        by_type[str(g["type"])].append(g)
    summary = {"version": "twowiki_score_sidecar v1.1 2026-11-10", "n_scored": n, "em_gold": a.em_gold,
               "alignment": {"one_based": align[1], "zero_based": align[0], "convention": conv,
                             "skipped": bool(a.skip_align_check and conv is None)},
               "score_module": SCORE_PATH, "ruler_sha256": S.RULER_SHA256,
               "sidecar_sha256": sha256(os.path.join(a.run, "sidecar.jsonl")), "data_sha256": sha256(a.data),
               "types_source": {"types": a.types, "official": a.official},
               "overall": summ(per), "by_type": {t: summ(v) for t, v in sorted(by_type.items())},
               "note": "headline = lenient (certified standard ruler); the four audit columns are diagnostics only"}

    keys = list(per[0].keys())
    with open(os.path.join(a.run, "scores_v11_per_instance.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(per)
    try:
        import pandas as pd
        with pd.ExcelWriter(os.path.join(a.run, "scores_v11_per_instance.xlsx")) as xw:
            pd.DataFrame(per).to_excel(xw, index=False, sheet_name="per_instance")
            pd.DataFrame([{"group": "overall", **summary["overall"]}] +
                         [{"group": t, **s} for t, s in summary["by_type"].items()]).to_excel(xw, index=False, sheet_name="summary")
    except Exception as e:
        print("xlsx not written (CSV is the record):", e)
    json.dump(summary, open(os.path.join(a.run, "scores_v11_summary.json"), "w"), indent=1)
    o = summary["overall"]
    print(f"\nHEADLINE lenient {o['lenient']:.2f}  (n={n})  | EM {o['em']:.2f} · QA F1 {o['qa_f1']:.2f} · Ret F1 {o['ret_f1']:.2f}"
          f" · median output {o['median_output_chars']} chars")
    print(f"AUDIT    first-sentence lenient {o['lenient_first_sentence']:.2f} · lenient excl. ambiguous credit "
          f"{o['lenient_unambiguous']:.2f} · yes/no-both {o['yesno_both']:.2f}% · both-options {o['options_both']:.2f}%")
    print("BY TYPE  " + " | ".join(f"{t} n={s['n']}: {s['lenient']:.1f} / first-sent {s['lenient_first_sentence']:.1f}"
                                   f" / unamb {s['lenient_unambiguous']:.1f}" for t, s in summary["by_type"].items()))
    print("wrote scores_v11_per_instance.csv, scores_v11_summary.json")


if __name__ == "__main__":
    main()
