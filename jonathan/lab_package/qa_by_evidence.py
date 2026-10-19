#!/usr/bin/env python3
"""
qa_by_evidence.py - fill slide 3's missing cells: QA accuracy when the answerer
reads SHARED-only or UNION evidence. Reuses run_labels4's model loading, prompts,
glue and caching verbatim (imported), so the only new logic is the evidence set.

usage (one GPU job at a time, in this order):
  python qa_by_evidence.py --evidence union  --answerer weak     # ~40-60 min
  python qa_by_evidence.py --evidence shared --answerer weak     # ~40-60 min
  python qa_by_evidence.py --evidence shared --answerer strong   # ~2-3 h (4bit)
  python qa_by_evidence.py --score                               # CPU: prints all cells
"""
import argparse, json, os, re, string, time
import run_labels4 as R

RUN_DIR   = "/mnt/hdd/jonathan/runs/qa_by_evidence"
GRID_FILE = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
RETR_W    = "/mnt/hdd/jonathan/runs/full_run_p100/retr_weak.jsonl"
RETR_S    = "/mnt/hdd/jonathan/runs/full_run_p100/retr_strong.jsonl"
DATA_FILE = "/mnt/hdd/jonathan/data/tts_test.json"
QA_SYSTEM = "You are a helpful assistant."

def val_indices():
    idxs = []
    for l in open(GRID_FILE):
        r = json.loads(l)
        if r["split"] == "val":
            idxs.append(r["idx"])
    return sorted(idxs)

def retr_map(path):
    m = {}
    for l in open(path):
        r = json.loads(l)
        m[int(str(r["k"]).split("|")[0])] = r["v"]
    return m

def evidence_ids(mode, i, inst, rw, rs):
    n = len(inst["ctxs_candidate"])
    W = set(R.parse_selection(rw[i], n))
    S = set(R.parse_selection(rs[i], n))
    return sorted(W & S) if mode == "shared" else sorted(W | S)

def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def lenient(pred, golds):
    np_ = norm(pred)
    return any(re.search(r"(?<!\w)" + re.escape(norm(g)) + r"(?!\w)", np_)
               for g in golds if norm(g))

def generate_mode(mode, which):
    os.makedirs(RUN_DIR, exist_ok=True)
    data = json.load(open(DATA_FILE))
    rw, rs = retr_map(RETR_W), retr_map(RETR_S)
    idxs = val_indices()
    store = R.Store(os.path.join(RUN_DIR, "qa_%s_%s.jsonl" % (mode, which)))
    todo = [i for i in idxs if str(i) not in store and i in rw and i in rs]
    print("[%s_%s] val n=%d | cached=%d | to run=%d"
          % (mode, which, len(idxs), len(idxs) - len(todo), len(todo)))
    if not todo:
        return
    m, tok = R.load_model(which, "4bit" if which == "strong" else "fp16", stage="qa")
    t0 = time.time()
    for k, i in enumerate(todo, 1):
        inst  = data[i]
        block = R.build_block(inst, evidence_ids(mode, i, inst, rw, rs))
        txt, _ = R.generate(m, tok, QA_SYSTEM, R.qa_prompt(inst, block), R.QA_OUT)
        store.put(str(i), txt)
        if k % 50 == 0:
            el = time.time() - t0
            print("  %d/%d | %.1f min | eta %.1f min"
                  % (k, len(todo), el / 60, el / k * (len(todo) - k) / 60))
            store.flush()
    store.flush()
    print("[%s_%s] done" % (mode, which))

def score():
    data = json.load(open(DATA_FILE))
    for mode in ("shared", "union"):
        for which in ("weak", "strong"):
            p = os.path.join(RUN_DIR, "qa_%s_%s.jsonl" % (mode, which))
            if not os.path.exists(p):
                print("%-6s + %-6s: (no file yet)" % (mode, which))
                continue
            n = ok = 0
            for l in open(p):
                r = json.loads(l)
                i = int(r["k"])
                inst = data[i]
                golds = inst["answers"] if isinstance(inst["answers"], list) else [inst["answers"]]
                n += 1
                ok += lenient(r["v"], golds)
            print("%-6s + %-6s: n=%d  lenient=%.2f" % (mode, which, n, 100.0 * ok / n))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--evidence", choices=["shared", "union"])
    ap.add_argument("--answerer", choices=["weak", "strong"])
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.score:
        score()
    else:
        assert a.evidence and a.answerer, "need --evidence and --answerer (or --score)"
        generate_mode(a.evidence, a.answerer)
