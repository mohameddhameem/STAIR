#!/usr/bin/env python3
"""
twowiki_grpo_stage.py -- stage the 2Wiki labels in the layout grpo_trainer_v15 reads, so the HotpotQA trainer runs
on 2Wiki with only its data/grid paths changed (grpo_trainer_v15_2w.py). CPU, seconds. No model, nothing scored.
DRAFT v1.0 2026-11-09 for Jonathan's line-by-line review.

The trainer reads, from one folder (GRID):
    DATA                       one JSON list; a row's idx = its position in the list
    retr_weak / retr_strong    keys "<idx>"           qa_weak / qa_strong   keys "<idx>|weak", "<idx>|strong"
    ret_labels/ret_labels.jsonl  one row per instance; the trainer uses ONLY idx and split ("train" / "val")
What this does:
    1. concatenates train5000 then val400 (labels6000_merged) -> idx 0..4999 = train, 5000..5399 = val;
    2. re-keys the four stores with the same offset (val keys + 5000) -- every expected key present exactly once;
    3. writes ret_labels.jsonl rows {idx, split}; test600 is NOT staged (the trainer can never see it);
    4. writes stage_manifest.json (input + output hashes, counts). Refuses to overwrite.
Usage (lab):  python3 -I twowiki_grpo_stage.py
"""
import hashlib, json, os

SRC = "/mnt/hdd/jonathan/runs/twowiki/labels6000_merged"
OUT = "/mnt/hdd/jonathan/runs/twowiki/grpo_grid"
PARTS = [("train5000", "train", 5000), ("val400", "val", 400)]
STORES = ["retr_weak", "retr_strong", "qa_weak", "qa_strong"]


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rekey(k, offset):
    i, sep, rest = k.partition("|")
    return f"{int(i) + offset}{sep}{rest}"


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    data, rows, stores, offset, inputs = [], [], {s: [] for s in STORES}, 0, {}
    for folder, split, n in PARTS:
        d = os.path.join(SRC, folder)
        dp = os.path.join(d, "tts_test_2wiki_adapted.json")
        part = json.load(open(dp, encoding="utf-8"))
        assert len(part) == n, (folder, len(part))
        inputs[folder] = {"data": sha(dp)}
        for s in STORES:
            p = os.path.join(d, s + ".jsonl")
            inputs[folder][s] = sha(p)
            recs = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            keys = [r["k"] for r in recs]
            want = [str(i) for i in range(n)] if s.startswith("retr") else \
                   [f"{i}|{b}" for i in range(n) for b in ("weak", "strong")]
            assert len(keys) == len(set(keys)) and set(keys) == set(want), (folder, s, len(keys), len(want))
            for r in recs:
                r["k"] = rekey(r["k"], offset)
            stores[s] += recs
        rows += [{"idx": offset + i, "split": split} for i in range(n)]
        data += part
        offset += n
    assert len(data) == 5400 and len(rows) == 5400

    os.makedirs(os.path.join(OUT, "ret_labels"))
    dp = os.path.join(OUT, "tts_2wiki_trainval.json")
    json.dump(data, open(dp, "w", encoding="utf-8"), ensure_ascii=False)
    for s in STORES:
        with open(os.path.join(OUT, s + ".jsonl"), "w", encoding="utf-8") as f:
            for r in stores[s]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rp = os.path.join(OUT, "ret_labels", "ret_labels.jsonl")
    with open(rp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    man = {"version": "twowiki_grpo_stage v1.0 2026-11-09", "src": SRC, "inputs_sha256": inputs,
           "layout": "idx 0-4999 = train5000 (train), 5000-5399 = val400 (val); test600 not staged",
           "outputs_sha256": {"tts_2wiki_trainval.json": sha(dp), "ret_labels.jsonl": sha(rp),
                              **{s + ".jsonl": sha(os.path.join(OUT, s + ".jsonl")) for s in STORES}},
           "counts": {"rows": len(rows), "train": 5000, "val": 400,
                      **{s: len(stores[s]) for s in STORES}}}
    json.dump(man, open(os.path.join(OUT, "stage_manifest.json"), "w"), indent=1)
    print(json.dumps(man["counts"]))
    print("STAGED OK ->", OUT)


if __name__ == "__main__":
    main()
