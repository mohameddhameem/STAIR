#!/usr/bin/env python3
"""
twowiki_labels_merge.py -- merge the 21 chunk outputs of the 2Wiki label array (twowiki_labels6000.sbatch) into ONE run
folder per split, in the exact layout twowiki_cells_score.py reads, so the reviewed scorer runs unchanged on test600,
val400 and train5000. CPU, seconds. DRAFT v1.0 2026-11-04 for Jonathan's line-by-line review.

What it does, per split (test600 / val400 / train5000):
  1. reads chunk_list.txt and, for each chunk of that split, re-checks the chunk DATA hash against the lab manifest
     (chunks_manifest.json) and requires CHUNK_DONE in the chunk's output folder;
  2. concatenates the chunk records (same order as the split file) into <out>/<split>/tts_test_2wiki_adapted.json --
     the file name the scorer expects -- adding nothing and changing nothing inside a record;
  3. re-keys the eight stores: the runner keyed them by POSITION WITHIN THE CHUNK ("<i>" and "<i>|<retr>"); the merged
     position = chunk offset + i. Every expected key must be present exactly once (asserted);
  4. writes <out>/<split>/merge_manifest.json (chunk hashes, store entry counts, record count, output hash).
Nothing is scored here. Usage (cluster):
  python3 -I twowiki_labels_merge.py --chunks ~/slr/twowiki_pools6000/chunks \
      --labels /common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/labels6000 \
      --out    /common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/labels6000_merged
"""
import argparse, collections, hashlib, json, os

SPLITS = tuple(os.environ.get("SPLITS","test600,val400,train5000").split(","))
STORES = ["retr_weak", "retr_strong", "qa_weak", "qa_strong", "meta_retr_weak", "meta_retr_strong", "meta_qa_weak", "meta_qa_strong"]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rekey(k, offset):
    """'<i>' -> '<offset+i>' ; '<i>|weak' -> '<offset+i>|weak'."""
    i, sep, rest = k.partition("|")
    return f"{int(i) + offset}{sep}{rest}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunks", required=True); ap.add_argument("--labels", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.chunks, "chunks_manifest.json")))
    names = [l.strip() for l in open(os.path.join(a.chunks, "chunk_list.txt")) if l.strip()]
    for split in SPLITS:
        od = os.path.join(a.out, split)
        if os.path.exists(od):
            raise SystemExit(f"REFUSING: {od} exists")
        os.makedirs(od)
        recs, offset, used = [], 0, []
        stores = {s: [] for s in STORES}
        for n in [x for x in names if x.startswith(split + "_")]:
            dp = os.path.join(a.chunks, n + ".json"); lp = os.path.join(a.labels, n)
            assert sha(dp) == man["chunks"][n]["sha256"], f"{n}: chunk data hash != lab manifest"
            assert os.path.exists(os.path.join(lp, "CHUNK_DONE")), f"{n}: CHUNK_DONE missing (chunk not finished)"
            part = json.load(open(dp, encoding="utf-8"))
            for s in STORES:
                lines = [json.loads(l) for l in open(os.path.join(lp, s + ".jsonl"), encoding="utf-8") if l.strip()]
                for r in lines:
                    r["k"] = rekey(r["k"], offset)
                stores[s] += lines
            used.append({"chunk": n, "n": len(part), "offset": offset, "data_sha256": man["chunks"][n]["sha256"],
                         "store_entries": {s: sum(1 for _ in open(os.path.join(lp, s + ".jsonl"))) for s in STORES}})
            recs += part; offset += len(part)
        n = len(recs)
        # completeness: retrieval stores need keys 0..n-1; QA stores need "<i>|weak" and "<i>|strong" for every i
        for s in STORES:
            keys = [r["k"] for r in stores[s]]
            assert len(keys) == len(set(keys)), f"{split}/{s}: duplicate keys after re-keying"
            want = {str(i) for i in range(n)} if "retr" in s else {f"{i}|{w}" for i in range(n) for w in ("weak", "strong")}
            assert set(keys) == want, f"{split}/{s}: expected {len(want)} keys, got {len(keys)} (missing {len(want - set(keys))})"
        json.dump(recs, open(os.path.join(od, "tts_test_2wiki_adapted.json"), "w", encoding="utf-8"), ensure_ascii=False)
        for s in STORES:
            with open(os.path.join(od, s + ".jsonl"), "w", encoding="utf-8") as f:
                for r in stores[s]:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        mm = {"version": "twowiki_labels_merge v1.0 2026-11-04", "split": split, "n": n, "chunks": used,
              "types": dict(collections.Counter(r["type"] for r in recs)),
              "output_sha256": {f: sha(os.path.join(od, f)) for f in ["tts_test_2wiki_adapted.json"] + [s + ".jsonl" for s in STORES]}}
        json.dump(mm, open(os.path.join(od, "merge_manifest.json"), "w"), indent=1)
        print(f"{split}: {n} records from {len(used)} chunks; stores complete; types {mm['types']}")


if __name__ == "__main__":
    main()
