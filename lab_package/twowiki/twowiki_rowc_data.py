#!/usr/bin/env python3
"""
twowiki_rowc_data.py -- build AgentTTS's test-file name (`tts_test_2_samples500.json`) for Row C from OUR verified pools:
dev[500:1000] = the 500 their released generator would produce (start = sample_number). DRAFT v1.0 2026-11-01.
Source = the two hash-checked label chunks test600_00 / test600_01 (records 100..599 of test600 are dev[500:1000] in
order). Kept fields = exactly theirs: question, answers, ctxs_candidate, ctxs_gt (nothing inside them changed).
Caveat recorded: their paper (A.4) says the 500 were "randomly sampled"; their code slices -- this is the code's 500.
Usage (cluster): python3 -I twowiki_rowc_data.py --chunks ~/slr/twowiki_pools6000/chunks \
                   --ids ~/slr/STAIR/common/stair_common/splits/2wikimultihopqa/test600.txt --out <work>/data/2wikimultihopqa
"""
import argparse, hashlib, json, os


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--chunks", required=True); ap.add_argument("--ids", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.chunks, "chunks_manifest.json")))
    recs = []
    for n in ("test600_00", "test600_01"):
        p = os.path.join(a.chunks, n + ".json"); assert sha(p) == man["chunks"][n]["sha256"], f"{n}: hash mismatch"
        recs += json.load(open(p, encoding="utf-8"))
    ids = [l.strip() for l in open(a.ids) if l.strip()]
    assert [r["_id"] for r in recs] == ids, "chunks are not test600 in order"
    part = recs[100:600]                                            # dev[500:1000]
    assert len(part) == 500
    out = [{k: r[k] for k in ("question", "answers", "ctxs_candidate", "ctxs_gt")} for r in part]
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "tts_test_2_samples500.json")
    if os.path.exists(dst):
        raise SystemExit(f"REFUSING: {dst} exists")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"ids_dev500_1000": [r["_id"] for r in part], "first_id": part[0]["_id"], "last_id": part[-1]["_id"],
               "sha256": sha(dst), "source_chunks": {n: man["chunks"][n]["sha256"] for n in ("test600_00", "test600_01")}},
              open(os.path.join(a.out, "rowc_data_manifest.json"), "w"), indent=1)
    print("wrote", dst, "| n", len(out), "| first", part[0]["_id"], "| sha256", sha(dst))


if __name__ == "__main__":
    main()
