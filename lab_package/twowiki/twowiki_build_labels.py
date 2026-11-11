#!/usr/bin/env python3
"""
twowiki_build_labels.py -- the 2Wiki label factory for SFT (label): retrieval labels + answer labels, built with the
SAME RULES as HotpotQA's certified builders (configb_build_ret_labels.py v1.0 and configb_build_qa_labels.py), on the
staged 2Wiki grid (train5000 = idx 0-4999, val400 = idx 5000-5399). CPU-only; writes only into OUT. Never opens test600.
DRAFT v1.0 2026-11-11 for Jonathan's line-by-line review.

RETRIEVAL label (coverage proxy, gold used only through containment -- as HotpotQA):
  R1 cascade : gold in the weak retriever's picks -> W | not there but in weak∪strong picks -> S | nowhere -> X
  R2 mapping : ret_target = W where ret_label == X
  R3 weights : inverse-frequency class weights on the TRAIN split, after the mapping
  R4 audits  : proxy audit, X-rescue table, oracle-agreement table (outcomes consulted for auditing only)
ANSWER label (outcome lane -- as HotpotQA), one row per (instance, retrieval choice):
  qa_label  : W = the weak answer is right | S = weak wrong, strong right | X = both wrong
  qa_target : X -> W (do not pay for strong when neither works)
  Outcomes  : the standard ruler (lenient_match_agenttts.LEN) on the stored answers; on val400 the recomputed verdicts
              must equal the reviewed scorer's CSV exactly (built-in check, else STOP).
OUT layout (= what configb_sft_router5.py reads, see its CFG):
  OUT/ret_labels/ret_labels.jsonl  rows {idx, split, n_weak_sel, covW, covUnion, ret_label, ret_target}
  OUT/qa_labels/qa_labels.jsonl    rows {idx, split, ret_choice, qa_label, qa_target}
  + manifest.json / audit_report.json per folder, + label_review.xlsx (counts, audits, 150 sampled rows for review)
Usage (lab):  python3 -I twowiki_build_labels.py            # full build (5,400 instances)
              python3 -I twowiki_build_labels.py --limit 10 # smoke: prints 10 rows, writes nothing
"""
import argparse, csv, datetime, importlib.util, json, os, random, re, string
from collections import Counter

GRID = "/mnt/hdd/jonathan/runs/twowiki/grpo_grid"                 # staged 9 Nov (twowiki_grpo_stage.py)
DATA = os.path.join(GRID, "tts_2wiki_trainval.json")
VAL_CSV = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged/val400/scores_cells_per_instance.csv"
OUT = "/mnt/hdd/jonathan/runs/twowiki/sft_grid"
COMMON = "/mnt/hdd/jonathan/STAIR/common/stair_common"
OFF_VAL, N_TRAIN, N_VAL = 5000, 5000, 400
CELLS = ["WW", "WS", "SW", "SS"]

_spec = importlib.util.spec_from_file_location("lenient_match_agenttts", os.path.join(COMMON, "lenient_match_agenttts.py"))
_m = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_m); LEN = _m.LEN


# ---- identical helpers to the HotpotQA builders ----------------------------------------------------------------------
def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def contains(text, golds):
    nt = norm(text)
    return any(re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", nt) for g in golds if norm(g))


def parse_selection(txt, n_refs):                                   # = run_labels4.parse_selection / twowiki_cells_score
    part = txt.split('Selected references')[-1] if 'Selected references' in txt else txt
    nums = [int(x) for x in re.findall(r'\[(\d+)\]', part)]
    return sorted(set(n for n in nums if 1 <= n <= n_refs))


def load_store(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    if not a.limit and os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    data = json.load(open(DATA, encoding="utf-8")); assert len(data) == N_TRAIN + N_VAL
    retr = {"W": load_store(os.path.join(GRID, "retr_weak.jsonl")), "S": load_store(os.path.join(GRID, "retr_strong.jsonl"))}
    qa = {"W": load_store(os.path.join(GRID, "qa_weak.jsonl")), "S": load_store(os.path.join(GRID, "qa_strong.jsonl"))}

    ret_rows, qa_rows, shares = [], [], Counter()
    aud = {"W_but_WW_wrong": 0, "S_but_WW_right": 0, "X_rescued": 0, "X_cells": Counter(), "X_yesno": 0, "X_n": 0,
           "agree": Counter(), "winnable_committed": 0}
    em_all = {}
    for i, inst in enumerate(data):
        if a.limit and i >= a.limit: break
        split = "train" if i < OFF_VAL else "val"
        golds = inst["answers"] if isinstance(inst["answers"], list) else [inst["answers"]]
        pool = inst["ctxs_candidate"]; npool = len(pool)
        selW = parse_selection(retr["W"][str(i)], npool); selS = parse_selection(retr["S"][str(i)], npool)
        txtW = " ".join(pool[k - 1] for k in selW)
        txtU = " ".join(pool[k - 1] for k in sorted(set(selW) | set(selS)))
        covW, covU = contains(txtW, golds), contains(txtU, golds)
        lab = "W" if covW else ("S" if covU else "X"); tgt = "W" if lab == "X" else lab
        shares[lab] += 1
        ret_rows.append({"idx": i, "split": split, "n_weak_sel": len(selW), "covW": covW, "covUnion": covU,
                         "ret_label": lab, "ret_target": tgt})
        # outcomes (standard ruler) -- for the QA labels and for the retrieval audits only
        ans = {"WW": qa["W"][f"{i}|weak"], "WS": qa["S"][f"{i}|weak"], "SW": qa["W"][f"{i}|strong"], "SS": qa["S"][f"{i}|strong"]}
        em = [int(bool(LEN(str(ans[c]), golds))) for c in CELLS]; em_all[i] = em
        for rc, (wq, sq) in (("W", (em[0], em[1])), ("S", (em[2], em[3]))):
            ql = "W" if wq else ("S" if sq else "X")
            qa_rows.append({"idx": i, "split": split, "ret_choice": rc, "qa_label": ql, "qa_target": "W" if ql == "X" else ql})
        # R4 audits
        if lab == "W" and not em[0]: aud["W_but_WW_wrong"] += 1
        if lab == "S" and em[0]: aud["S_but_WW_right"] += 1
        if lab == "X":
            aud["X_n"] += 1
            if str(golds[0]).strip().lower() in ("yes", "no"): aud["X_yesno"] += 1
            if any(em): aud["X_rescued"] += 1
            for nm, v in zip(CELLS, em):
                if v: aud["X_cells"][nm] += 1
        if any(em):
            orc = "W" if (em[0] or em[1]) else "S"
            aud["agree"][f"{lab}|{orc}"] += 1
            if lab in ("W", "S"): aud["winnable_committed"] += 1

    if a.limit:
        for r in ret_rows: print(json.dumps(r))
        for r in qa_rows[:2 * a.limit]: print(json.dumps(r))
        print("smoke mode: nothing written"); return

    # ---- built-in check: val400 verdicts == reviewed scorer CSV --------------------------------------------------
    ref = {}
    for r in csv.DictReader(open(VAL_CSV, encoding="utf-8")):
        ref[(int(r["idx"]) + OFF_VAL, r["cell"])] = int(float(r["lenient"]))
    mism = sum(ref[(i, c)] != em_all[i][k] for i in range(OFF_VAL, OFF_VAL + N_VAL) for k, c in enumerate(CELLS))
    print(f"CHECK val400 verdicts vs reviewed scorer CSV: {mism} mismatches of {4 * N_VAL}")
    assert mism == 0, "verdicts differ from the reviewed scorer -- STOP"

    # ---- R3 class weights (train, post-mapping) ----
    tr = [r for r in ret_rows if r["split"] == "train"]
    nW = sum(r["ret_target"] == "W" for r in tr); nS = len(tr) - nW
    wW, wS = len(tr) / (2 * nW), len(tr) / (2 * nS)
    trq = Counter(r["qa_target"] for r in qa_rows if r["split"] == "train"); totq = sum(trq.values())
    wq = {k: round(totq / (2 * v), 4) for k, v in trq.items()}

    os.makedirs(os.path.join(OUT, "ret_labels")); os.makedirs(os.path.join(OUT, "qa_labels"))
    # the SFT trainer reads data + the four stores from GRID next to the labels: link the staged files (read-only reuse)
    for f in ("tts_2wiki_trainval.json", "retr_weak.jsonl", "retr_strong.jsonl", "qa_weak.jsonl", "qa_strong.jsonl", "stage_manifest.json"):
        os.symlink(os.path.join(GRID, f), os.path.join(OUT, f))
    with open(os.path.join(OUT, "ret_labels", "ret_labels.jsonl"), "w") as f:
        for r in ret_rows: f.write(json.dumps(r) + "\n")
    with open(os.path.join(OUT, "qa_labels", "qa_labels.jsonl"), "w") as f:
        for r in qa_rows: f.write(json.dumps(r) + "\n")
    now = datetime.datetime.now().isoformat(timespec="seconds")
    json.dump({"version": "twowiki_build_labels v1.0 2026-11-11 (= configb_build_ret_labels v1.0 rules; 2Wiki staged grid)",
               "retr_weak": "Qwen/Qwen2.5-7B-Instruct", "retr_strong": "Qwen/Qwen2.5-72B-Instruct",
               "inputs": {"grid": GRID, "data": DATA}, "metric": "lenient whole-word containment",
               "n": len(ret_rows), "shares": dict(shares), "train_post_mapping": {"W": nW, "S": nS},
               "class_weights_post_mapping": {"W": round(wW, 3), "S": round(wS, 3)}, "created": now},
              open(os.path.join(OUT, "ret_labels", "manifest.json"), "w"), indent=1)
    json.dump({k: (dict(v) if isinstance(v, Counter) else v) for k, v in aud.items()},
              open(os.path.join(OUT, "ret_labels", "audit_report.json"), "w"), indent=1)
    qshares = {f"{sp}|{rc}": dict(Counter(r["qa_label"] for r in qa_rows if r["split"] == sp and r["ret_choice"] == rc))
               for sp in ("train", "val") for rc in ("W", "S")}
    json.dump({"version": "twowiki_build_labels v1.0 2026-11-11", "source": GRID, "lane": "outcome-only (standard ruler)",
               "rule": "W if weak answer right; S if weak wrong and strong right; X if both wrong", "mapping": "X -> W",
               "rows": len(qa_rows), "shares": qshares, "class_weights_post_mapping": wq, "created": now},
              open(os.path.join(OUT, "qa_labels", "manifest.json"), "w"), indent=1)

    # ---- printed report (as the HotpotQA builders) ----
    print("RET rows:", len(ret_rows), "| shares:", dict(shares), "| train post-mapping:", {"W": nW, "S": nS},
          "| class weights:", {"W": round(wW, 3), "S": round(wS, 3)})
    print("AUDIT 1 (proxy): W-but-WW-wrong %.1f%% | S-but-WW-right %.1f%% | X rescued %.1f%%" % (
        100 * aud["W_but_WW_wrong"] / max(shares["W"], 1), 100 * aud["S_but_WW_right"] / max(shares["S"], 1),
        100 * aud["X_rescued"] / max(shares["X"], 1)))
    print("AUDIT 2 (X cells):", {k: f'{100 * v / max(aud["X_n"], 1):.1f}%' for k, v in aud["X_cells"].items()},
          "| yes/no-gold in X: %.1f%%" % (100 * aud["X_yesno"] / max(aud["X_n"], 1)))
    comm = aud["winnable_committed"]; agree = aud["agree"]["W|W"] + aud["agree"]["S|S"]
    print("AUDIT 3 (oracle agreement, committed winnable): %d/%d = %.1f%%" % (agree, comm, 100 * agree / max(comm, 1)),
          "| cells:", dict(aud["agree"]))
    print("QA rows:", len(qa_rows), "| shares:", qshares, "| class weights:", wq)

    # ---- Excel for Jonathan's review ----
    try:
        from openpyxl import Workbook
        wb = Workbook(); ws = wb.active; ws.title = "counts"
        ws.append(["RET labels", "n"]); [ws.append([k, v]) for k, v in shares.items()]
        ws.append([]); ws.append(["RET train post-mapping", "W", nW, "S", nS, "weights", round(wW, 3), round(wS, 3)])
        ws.append([]); ws.append(["QA shares", "W", "S", "X"])
        for k, v in qshares.items(): ws.append([k, v.get("W", 0), v.get("S", 0), v.get("X", 0)])
        ws.append([]); ws.append(["AUDIT 1 W-but-WW-wrong", aud["W_but_WW_wrong"]]); ws.append(["AUDIT 1 S-but-WW-right", aud["S_but_WW_right"]])
        ws.append(["AUDIT 1 X rescued", aud["X_rescued"], "of", aud["X_n"]]); ws.append(["AUDIT 3 oracle agreement", agree, "of", comm])
        ws2 = wb.create_sheet("review 150"); random.seed(17)
        ws2.append(["idx", "split", "question", "gold", "n_weak_sel", "covW", "covUnion", "ret_label", "ret_target",
                    "WW right", "WS right", "SW right", "SS right", "qa_label|W", "qa_label|S", "weak picks (text)"])
        qmap = {(r["idx"], r["ret_choice"]): r["qa_label"] for r in qa_rows}
        for r in random.sample(ret_rows, 150):
            i = r["idx"]; inst = data[i]; em = em_all[i]
            selW = parse_selection(retr["W"][str(i)], len(inst["ctxs_candidate"]))
            ws2.append([i, r["split"], inst["question"], str(inst["answers"][0] if isinstance(inst["answers"], list) else inst["answers"]),
                        r["n_weak_sel"], int(r["covW"]), int(r["covUnion"]), r["ret_label"], r["ret_target"], *em,
                        qmap[(i, "W")], qmap[(i, "S")], " || ".join(inst["ctxs_candidate"][k - 1][:200] for k in selW)])
        wb.save(os.path.join(OUT, "label_review.xlsx")); print("wrote", os.path.join(OUT, "label_review.xlsx"))
    except Exception as e:
        print("(Excel not written:", e, ")")
    print("DONE ->", OUT)


if __name__ == "__main__":
    main()
