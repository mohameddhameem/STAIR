#!/usr/bin/env python3
# =============================================================================
# export_probe_pools.py — VERSION: export_probe_pools v1.3 2026-09-09
# (real schema: ctxs_candidate / answers / supporting_titles; retr_*.jsonl)
#
# Excel 1 of Pan's 10-instance overfit probe: one row per passage — the full
# ~100-passage pool for each of the 10 chosen instances — with flags:
#   is_supporting   passage title in the dataset's supporting_titles
#   weak_selected   chosen by the weak retriever
#   strong_selected chosen by the strong retriever
#
# Writes probe_10_passages.xlsx if openpyxl is available, else CSV.
#
# USAGE (lab, CPU, seconds):
#   cd /mnt/hdd/jonathan/slr/stage-level-routing/lab_package
#   python3 export_probe_pools.py \
#       --data /mnt/hdd/jonathan/data/tts_test.json \
#       --run  /mnt/hdd/jonathan/runs/full_run_p100 \
#       --out  /mnt/hdd/jonathan/probe_10_passages
# =============================================================================
import os, json, argparse

VERSION = "export_probe_pools v1.3 2026-09-09 (string pools; evidence-string selections)"
PROBE_IDX = [5, 6, 9, 10, 12, 13, 16, 17, 22, 23]  # first 10 labeled TRAIN instances; 7 W / 3 S


def load_selected(run_dir, which):
    """Selected passages per instance from retr_weak.jsonl / retr_strong.jsonl.
    Robust to value shape: lists of ints (ranks), strings (titles), or dicts
    (uses 'title' and/or rank-like fields). Returns idx -> (ranks, titles)."""
    path = os.path.join(run_dir, f"retr_{which}.jsonl")
    sel = {}
    if not os.path.exists(path):
        print(f"  note: {path} not found — {which}_selected flags will be empty")
        return sel
    for l in open(path):
        r = json.loads(l)
        k = r.get("k", r.get("idx"))
        try:
            i = int(str(k).split("|")[0])
        except Exception:
            continue
        v = r.get("v", "")
        ranks, titles = set(), set()
        if isinstance(v, str):
            import re as _re
            titles = set(t.strip() for t in _re.findall(r"\[\d+\]\s*([^:\n]+):", v))
        sel[i] = (ranks, titles)
    if sel and not any(rk or tl for rk, tl in sel.values()):
        print(f"  note: parsed {path} but found no rank/title info — flags empty")
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True, help="output path WITHOUT extension")
    a = ap.parse_args()
    print(VERSION, flush=True)
    data = json.load(open(a.data))
    selW = load_selected(a.run, "weak")
    selS = load_selected(a.run, "strong")

    header = ["idx", "question", "gold_answer", "passage_rank", "title",
              "is_supporting", "weak_selected", "strong_selected", "passage_text"]
    rows = []
    for i in PROBE_IDX:
        d = data[i]
        q = d["question"]
        g = (d.get("answers") or [""])[0]
        sup = set(d.get("supporting_titles", []))
        ctx = d.get("ctxs_candidate", [])
        assert len(ctx) >= 90, f"idx {i}: pool has {len(ctx)} passages (expected ~100)"
        rw = selW.get(i, (set(), set()))
        rs = selS.get(i, (set(), set()))
        for rank, entry in enumerate(ctx):
            if isinstance(entry, str) and ": " in entry:
                title, body = entry.split(": ", 1)
            else:
                title, body = "", entry
            if isinstance(body, list):
                body = " ".join(str(x) for x in body)
            rows.append([i, q, g, rank, title,
                         1 if title in sup else 0,
                         1 if (rank in rw[0] or title in rw[1]) else 0,
                         1 if (rank in rs[0] or title in rs[1]) else 0,
                         str(body)])
        nW = sum(r[6] for r in rows if r[0] == i)
        nS = sum(r[7] for r in rows if r[0] == i)
        nsup = sum(1 for r in rows if r[0] == i and r[5])
        print(f"  idx {i}: {len(ctx)} passages, supporting={nsup}, weak-selected={nW}, strong-selected={nS}", flush=True)

    try:
        from openpyxl import Workbook
        from openpyxl.styles import Font, PatternFill
        wb = Workbook(); ws = wb.active; ws.title = "probe_10_passages"
        ws.append(header)
        for c in ws[1]:
            c.font = Font(name="Arial", bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="2C8FA3")
        for r in rows:
            ws.append(r)
        for col, w in zip("ABCDEFGHI", [6, 40, 18, 8, 22, 8, 8, 8, 90]):
            ws.column_dimensions[col].width = w
        ws.freeze_panes = "A2"
        out = a.out + ".xlsx"; wb.save(out)
    except ImportError:
        import csv as _csv
        out = a.out + ".csv"
        with open(out, "w", newline="", encoding="utf-8-sig") as f:
            w = _csv.writer(f); w.writerow(header); w.writerows(rows)
    print(f"wrote {out}  ({len(rows)} rows = 10 instances x ~100 passages)", flush=True)


if __name__ == "__main__":
    main()
