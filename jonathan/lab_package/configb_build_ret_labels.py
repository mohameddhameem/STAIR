#!/usr/bin/env python3
"""
build_ret_labels.py — v1.0 2026-10-02
The RET label factory (code-table row #1). CPU-only, read-only inputs.

RULES IMPLEMENTED (each block names its rule):
  R1 cascade   : covW -> W | no covW but covUnion -> S | covUnion=0 -> X
  R2 mapping   : ret_target = W where ret_label == X  (X -> W at the compare)
  R3 weights   : class weights computed AFTER the mapping, on the TRAIN split only
  R4 audits    : proxy audit + X-rescue table + oracle-agreement table, printed
                 and saved — re-printed on every build (settled 2 Oct)

OUTPUT: one row per instance ->
  {idx, split, n_weak_sel, covW, covUnion, ret_label, ret_target}
plus manifest.json and audit_report.json in the same directory.

usage:
  python build_ret_labels.py                 # full build (7,405 rows, ~minutes)
  python build_ret_labels.py --limit 10      # smoke: first 10 rows printed, nothing written
"""
import argparse, json, os, re, string, datetime

# ---- pinned paths (the grid cache; change only when the retriever pair changes)
GRID   = "/mnt/hdd/jonathan/runs/config_b/router_data_standard.jsonl"
RETR_W = "/mnt/hdd/jonathan/runs/config_b/retr_weak.jsonl"
RETR_S = "/mnt/hdd/jonathan/runs/config_b/retr_strong.jsonl"
DATA   = "/mnt/hdd/jonathan/data/tts_test.json"
OUTDIR = "/mnt/hdd/jonathan/runs/config_b/ret_labels"

import sys
sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")
import run_labels4 as R      # parse_selection — verbatim reuse of the frozen glue

# ---- lenient whole-word containment (identical to make_label_review.py) ----
def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def contains(text, golds):
    nt = norm(text)
    return any(re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", nt)
               for g in golds if norm(g))

def retr_map(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        m[int(str(r["k"]).split("|")[0])] = r["v"]
    return m

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="smoke mode: process N rows, write nothing")
    a = ap.parse_args()

    data = json.load(open(DATA))
    rw, rs = retr_map(RETR_W), retr_map(RETR_S)
    grid = {}
    for l in open(GRID):
        r = json.loads(l); grid[r["idx"]] = r

    rows, shares = [], {"W":0,"S":0,"X":0}
    # audits (R4)
    aud = {"W_but_WW_wrong":0, "S_but_WW_right":0, "X_rescued":0,
           "X_cells":{"WW":0,"WS":0,"SW":0,"SS":0}, "X_yesno":0, "X_n":0,
           "agree":{}, "winnable_committed":0}

    for i, inst in enumerate(data):
        if a.limit and len(rows) >= a.limit: break
        if i not in grid or i not in rw or i not in rs: continue
        g = grid[i]
        golds = inst["answers"] if isinstance(inst["answers"], list) else [inst["answers"]]
        npool = len(inst["ctxs_candidate"])
        selW = R.parse_selection(rw[i], npool)
        selS = R.parse_selection(rs[i], npool)
        txtW = " ".join(inst["ctxs_candidate"][k-1] for k in selW)
        txtU = " ".join(inst["ctxs_candidate"][k-1] for k in sorted(set(selW) | set(selS)))
        covW, covU = contains(txtW, golds), contains(txtU, golds)

        # R1 cascade  |  R2 mapping
        lab = "W" if covW else ("S" if covU else "X")
        tgt = "W" if lab == "X" else lab
        shares[lab] += 1
        rows.append({"idx": i, "split": g["split"], "n_weak_sel": len(selW),
                     "covW": covW, "covUnion": covU,
                     "ret_label": lab, "ret_target": tgt})

        # R4 audits (outcomes consulted for AUDITING only — never for labeling)
        em = g["em_standard"]
        if lab == "W" and not em[0]: aud["W_but_WW_wrong"] += 1
        if lab == "S" and em[0]:     aud["S_but_WW_right"] += 1
        if lab == "X":
            aud["X_n"] += 1
            gl = str(golds[0]).strip().lower()
            if gl in ("yes","no"): aud["X_yesno"] += 1
            if any(em): aud["X_rescued"] += 1
            for nm, v in zip(("WW","WS","SW","SS"), em):
                if v: aud["X_cells"][nm] += 1
        if any(em):
            orc = "W" if (em[0] or em[1]) else "S"
            key = f"{lab}|{orc}"
            aud["agree"][key] = aud["agree"].get(key, 0) + 1
            if lab in ("W","S"): aud["winnable_committed"] += 1

    if a.limit:
        for r_ in rows: print(json.dumps(r_))
        print("smoke mode: nothing written"); return

    # R3 class weights: TRAIN split, post-mapping
    tr = [r_ for r_ in rows if r_["split"] == "train"]
    nW = sum(1 for r_ in tr if r_["ret_target"] == "W"); nS = len(tr) - nW
    wW, wS = len(tr)/(2*nW), len(tr)/(2*nS)        # inverse-frequency, mean 1

    os.makedirs(OUTDIR, exist_ok=True)
    with open(os.path.join(OUTDIR, "ret_labels.jsonl"), "w") as f:
        for r_ in rows: f.write(json.dumps(r_) + "\n")
    manifest = {"version": "configb_build_ret_labels v1.0 2026-10-14 (Config B: paths, em_standard, model names)",
        "retr_weak": "Qwen/Qwen2.5-7B-Instruct", "retr_strong": "Qwen/Qwen2.5-72B-Instruct",
        "inputs": {"grid": GRID, "retr_weak": RETR_W, "retr_strong": RETR_S, "data": DATA},
        "metric": "lenient whole-word containment (make_label_review-compatible)",
        "n": len(rows), "shares": shares,
        "train_post_mapping": {"W": nW, "S": nS},
        "class_weights_post_mapping": {"W": round(wW,3), "S": round(wS,3)},
        "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(manifest, open(os.path.join(OUTDIR, "manifest.json"), "w"), indent=1)
    json.dump(aud, open(os.path.join(OUTDIR, "audit_report.json"), "w"), indent=1)

    # ---- printed report ----
    print("rows:", len(rows), "| shares:", shares)
    print("train post-mapping:", {"W": nW, "S": nS}, "| class weights:", {"W": round(wW,3), "S": round(wS,3)})
    print("\nAUDIT 1 (proxy): W-but-WW-wrong %.1f%% | S-but-WW-right %.1f%% | X rescued %.1f%%" % (
        100*aud["W_but_WW_wrong"]/max(shares["W"],1), 100*aud["S_but_WW_right"]/max(shares["S"],1),
        100*aud["X_rescued"]/max(shares["X"],1)))
    print("AUDIT 2 (X cells):", {k: f'{100*v/max(aud["X_n"],1):.1f}%' for k,v in aud["X_cells"].items()},
          "| yes/no-gold in X: %.1f%%" % (100*aud["X_yesno"]/max(aud["X_n"],1)))
    comm = aud["winnable_committed"]
    agree = aud["agree"].get("W|W",0) + aud["agree"].get("S|S",0)
    print("AUDIT 3 (oracle agreement, committed winnable): %d/%d = %.1f%%" % (agree, comm, 100*agree/max(comm,1)),
          "| cells:", aud["agree"])

if __name__ == "__main__":
    main()
