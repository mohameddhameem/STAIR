#!/usr/bin/env python3
"""
rep40_store_from_config_b.py -- turn the lab's Config B strong-retrieval store (run_labels_agenttts.py, pool = first 110
candidates + golds swapped in, NO shuffle) into a retrieval store rep40.py can use, for a chosen set of positions.
CPU, seconds. DRAFT v1.0 2026-11-05 for Jonathan's line-by-line review.

WHY (5 Nov 2026): the 4 Nov reruns (dev[500:1000] -> 32.6, held-out 1,481 -> 33.1) are VOID. rep40.py resolves the
store's "[13]" as a position in ITS OWN pool (cluster run_labels.py: first 30 candidates + golds, seeded SHUFFLE), but the
lab store's "[13]" is a position in the LAB pool (110, no shuffle). Verified on idx 500/501/1000: "[13] Forbidden Quest"
in the store resolved to "Starship Entertainment" on the cluster; "[34]" was dropped (pool of 30). Every question was
answered from the wrong passages. The 21 Oct 74.6 is unaffected (its armC_72b store and pool came from the same script).

WHAT: for every requested position i
  1. rebuild the LAB pool exactly: run_labels_agenttts.trim_pool(inst) with SLR_POOL_K=110 and SLR_SHUFFLE=off
     (deterministic from tts_test.json alone: cands[:110], golds swapped into the tail, no shuffle);
  2. parse the stored selection with the LAB parser (parse_selection) -> positions -> passages from that pool
     (this is exactly what the lab's own QA phase did, run_labels_agenttts.py line ~308);
  3. CHECK: the store text names each selected reference as "[n] <title>: ..."; the title of pool[n-1] must equal it for
     every n. Any mismatch -> count it and REFUSE to write (nothing partial is written);
  4. write <out>/retr_strong.jsonl with {"k": "<i>", "v": [passage, ...]}  -- the PASSAGE TEXTS, not positions --
     plus <out>/store_manifest.json (inputs' sha256, pool_k, counts, per-question #selected, zero-selection count).
rep40.py v1.1 uses a list value as the references directly (one guarded line); a string value keeps the old path.
Usage (cluster login node, slr-env):
  SLR_POOL_K=110 SLR_SHUFFLE=off python rep40_store_from_config_b.py --data ~/slr/data/tts_test.json \
      --store ~/slr/hotpotqa_stores/retr_strong.jsonl --out <run dir> --range 500 1000
  ... or --ids <file of _ids>  (held-out test: slr-test.txt, 1,481)
"""
import argparse, hashlib, json, os, re, sys

sys.path.insert(0, os.path.expanduser("~/slr"))          # the lab runner, scp'd for the 2Wiki job (sha 9ee08eac...)
import run_labels_agenttts as L

TITLE_RE = re.compile(r"\[(\d+)\]\s*([^\[]*?):")        # "[13] Forbidden Quest: ..." -> (13, "Forbidden Quest")


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--store", required=True); ap.add_argument("--out", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--range", nargs=2, type=int, metavar=("START", "STOP"))
    g.add_argument("--ids", help="file of _ids, one per line (positions looked up in --data)")
    a = ap.parse_args()
    assert os.environ.get("SLR_POOL_K") == "110" and os.environ.get("SLR_SHUFFLE") == "off", \
        "run with SLR_POOL_K=110 SLR_SHUFFLE=off (the Config B retrieval settings)"
    assert L.POOL_K == 110 and L.SHUFFLE_SCHEME == "as-given/no-shuffle", (L.POOL_K, L.SHUFFLE_SCHEME)
    out_store = os.path.join(a.out, "retr_strong.jsonl")
    if os.path.exists(out_store):
        raise SystemExit(f"REFUSING: {out_store} exists")
    data = json.load(open(a.data))
    if a.range:
        want = list(range(a.range[0], a.range[1]))
    else:
        pos = {r["_id"]: i for i, r in enumerate(data)}
        ids = [l.strip() for l in open(a.ids) if l.strip()]
        want = sorted(pos[i] for i in ids)                  # KeyError = id not in the data file -> stop
        assert len(want) == len(set(want)) == len(ids)
    st = {}
    for l in open(a.store):
        if l.strip():
            r = json.loads(l); st[int(str(r["k"]).split("|")[0])] = r["v"]
    missing = [i for i in want if i not in st]
    assert not missing, f"{len(missing)} positions absent from the store, e.g. {missing[:5]}"

    rows, nsel, zero, mismatch = [], [], 0, []
    for i in want:
        inst = dict(data[i]); inst["ctxs_candidate"] = L.trim_pool(inst); pool = inst["ctxs_candidate"]
        txt = st[i]
        sel = L.parse_selection(txt, len(pool))
        named = {int(n): t.strip() for n, t in TITLE_RE.findall(txt.split("Selected references")[-1])}
        for n in sel:
            if named.get(n) is not None and L.title_of(pool[n - 1]) != named[n]:
                mismatch.append({"idx": i, "n": n, "store_title": named[n], "pool_title": L.title_of(pool[n - 1])})
        if not sel:
            zero += 1
        nsel.append(len(sel))
        rows.append({"k": str(i), "v": [pool[n - 1] for n in sel]})
    if mismatch:
        print(json.dumps(mismatch[:10], indent=1))
        raise SystemExit(f"REFUSING: {len(mismatch)} title mismatches between store text and rebuilt lab pool")
    os.makedirs(a.out, exist_ok=True)
    with open(out_store, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    man = {"version": "rep40_store_from_config_b v1.0 2026-11-05", "data": a.data, "data_sha256": sha(a.data),
           "store": a.store, "store_sha256": sha(a.store), "runner": L.__file__, "runner_sha256": sha(L.__file__),
           "pool_k": L.POOL_K, "shuffle": L.SHUFFLE_SCHEME, "selection": a.range or a.ids, "n": len(rows),
           "positions_first_last": [want[0], want[-1]], "mean_selected": sum(nsel) / len(nsel), "zero_selected": zero,
           "selected_total": sum(nsel), "title_mismatches": 0,
           "output_sha256": sha(out_store)}
    json.dump(man, open(os.path.join(a.out, "store_manifest.json"), "w"), indent=1)
    if a.ids:
        json.dump(want, open(os.path.join(a.out, "positions_slr_test.json"), "w"))
    print(f"wrote {len(rows)} entries -> {out_store}; mean selected {man['mean_selected']:.2f}; zero-selection {zero}; "
          f"selections {man['selected_total']}, title mismatches 0")


if __name__ == "__main__":
    main()
