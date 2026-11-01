#!/usr/bin/env python3
"""
twowiki_label_chunks.py -- cut the three pool files (twowiki_pools.py v1.1 build output) into fixed chunks for the cluster
label array. DRAFT v1.0 2026-11-01 for Jonathan's review. Lab, CPU, seconds. Changes nothing inside a record.

Why chunks: the four cells on the 100 took 1 h 22 m (job 346550); researchshort allows 6 h, so a job can carry ~400
instances; 300 per chunk leaves margin. Chunks never cross a split, and run in registered order: test600, val400,
train5000 (so the held-out test and validation labels finish first). The runner keys its stores by POSITION in its data
file, so every chunk gets its own output folder on the cluster and keeps its `_id`s for joining afterwards.
Output: <out>/chunks/<split>_<kk>.json  +  <out>/chunks/chunk_list.txt (one name per line = the array order)
        + <out>/chunks/chunks_manifest.json (ids, counts, sha256 of every chunk and of the three source files)
Usage:  python3 -I twowiki_label_chunks.py --pools /mnt/hdd/jonathan/runs/twowiki/pools6000 --size 300
"""
import argparse, hashlib, json, os

ORDER = ["test600", "val400", "train5000"]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--pools", required=True); ap.add_argument("--size", type=int, default=300); a = ap.parse_args()
    cdir = os.path.join(a.pools, "chunks")
    if os.path.exists(cdir):
        raise SystemExit(f"REFUSING: {cdir} exists")
    os.makedirs(cdir)
    names, man, seen = [], {"size": a.size, "sources": {}, "chunks": {}}, set()
    for split in ORDER:
        src = os.path.join(a.pools, f"pools_{split}.json")
        recs = json.load(open(src, encoding="utf-8")); man["sources"][split] = {"n": len(recs), "sha256": sha(src)}
        for k in range(0, len(recs), a.size):
            part = recs[k:k + a.size]; name = f"{split}_{k // a.size:02d}"
            assert not seen & {r["_id"] for r in part}, "an _id appears in two chunks"; seen |= {r["_id"] for r in part}
            p = os.path.join(cdir, name + ".json")
            json.dump(part, open(p, "w", encoding="utf-8"), ensure_ascii=False)
            man["chunks"][name] = {"n": len(part), "first_id": part[0]["_id"], "last_id": part[-1]["_id"], "sha256": sha(p)}
            names.append(name)
    open(os.path.join(cdir, "chunk_list.txt"), "w").write("\n".join(names) + "\n")
    man["n_chunks"] = len(names); man["n_instances"] = len(seen)
    json.dump(man, open(os.path.join(cdir, "chunks_manifest.json"), "w"), indent=1)
    print(f"{len(names)} chunks, {len(seen)} instances:", " ".join(f"{n}({man['chunks'][n]['n']})" for n in names))


if __name__ == "__main__":
    main()
