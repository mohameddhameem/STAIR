#!/usr/bin/env python3
"""
twowiki_entity_split.py -- the 2WikiMultiHopQA split over the OFFICIAL dev file, entity-disjoint for training.

WHY (31 Oct 2026). AgentTTS's data/2wikimultihopqa/tts_train.json is the official 2Wiki TEST split (12,576 questions
with answer / supporting_facts / evidences withheld; first _id 8a65901a0bdb11eba7f7acde48001122 = official test.json).
The registry split committed over it (test 500 / val 500 / train_extra 11,576) was therefore over unlabeled data.
The labelled file of the same size is the official dev.json (data_ids_april7.zip, zip sha256 95df2bf5...), and
AgentTTS's 100 pooled comparison instances (tts_test.json) are exactly dev[0:100].

RULE
  test        = dev[0:500]      (contains AgentTTS's 100 = dev[0:100]; the "AgentTTS slice", as on HotpotQA)
  val         = dev[500:1000]
  train_extra = dev[1000:] MINUS every question that shares a gold Wikidata entity (entity_ids) with test or val
Positional, like the HotpotQA registry; the entity filter makes training disjoint from evaluation on the only
leakage key the data provides (questions about the same gold entities). Measured 31 Oct: test 500 shares 450 of its
1,204 gold entities with dev[1000:]; the filter drops 1,832 of 11,576 -> train_extra = 9,744.

USAGE
  --generate --dev <dev.json> --out <dir>    writes test.txt / val.txt / train_extra.txt (one _id per line) + report
  --check    --dev <dev.json> --dir <dir>    verifies committed files: sizes, positions, entity-disjointness, the
                                             AgentTTS 100 inside test, type mix; exits non-zero on any failure
"""
import argparse, collections, hashlib, json, os, re, sys


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def norm(q): return re.sub(r"\s+", " ", str(q)).strip().lower()


def ents(r): return {e for e in str(r.get("entity_ids", "")).split("_") if e}


def load_dev(path):
    dev = json.load(open(path, encoding="utf-8"))
    assert len(dev) == 12576, f"dev.json should have 12,576 records, got {len(dev)}"
    assert all(str(r.get("answer", "")).strip() for r in dev), "dev.json must carry answers (the test split does not)"
    assert len({r["_id"] for r in dev}) == len(dev), "_id not unique"
    return dev


def make(dev):
    test, val, rest = dev[:500], dev[500:1000], dev[1000:]
    held = set().union(*[ents(r) for r in test + val])
    train_extra = [r for r in rest if not (ents(r) & held)]
    return test, val, train_extra, len(rest) - len(train_extra)


def types(rows): return dict(sorted(collections.Counter(r["type"] for r in rows).items()))


def report(dev, test, val, train_extra, dropped, agenttts100=None):
    out = {"dev_n": len(dev), "test_n": len(test), "val_n": len(val), "train_extra_n": len(train_extra), "dropped_for_entity_overlap": dropped,
           "test_types": types(test), "val_types": types(val), "train_extra_types": types(train_extra),
           "entities": {"test_distinct": len(set().union(*[ents(r) for r in test])),
                        "shared_test_val": len(set().union(*[ents(r) for r in test]) & set().union(*[ents(r) for r in val])),
                        "shared_test_train_extra": len(set().union(*[ents(r) for r in test]) & set().union(*[ents(r) for r in train_extra])),
                        "shared_val_train_extra": len(set().union(*[ents(r) for r in val]) & set().union(*[ents(r) for r in train_extra]))}}
    if agenttts100 is not None:
        q = {norm(r["question"]) for r in agenttts100}
        pos = sorted(i for i, r in enumerate(dev) if norm(r["question"]) in q)
        out["agenttts_100"] = {"found_in_dev": len(pos), "all_inside_test": bool(pos) and pos[-1] < 500, "is_dev_0_100": pos == list(range(100))}
    return out


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--generate", action="store_true"); g.add_argument("--check", action="store_true")
    ap.add_argument("--dev", required=True, help="official dev.json (data_ids_april7)")
    ap.add_argument("--out", help="output dir for --generate"); ap.add_argument("--dir", help="committed split dir for --check")
    ap.add_argument("--agenttts-test", default=None, help="AgentTTS tts_test.json (100 pooled) to confirm it lies inside test")
    a = ap.parse_args()
    dev = load_dev(a.dev)
    a100 = json.load(open(a.agenttts_test, encoding="utf-8")) if a.agenttts_test else None
    test, val, train_extra, dropped = make(dev)
    rep = report(dev, test, val, train_extra, dropped, a100)
    rep["dev_sha256"] = sha256(a.dev)
    if a.generate:
        assert a.out, "--out required"; os.makedirs(a.out, exist_ok=True)
        for name, rows in (("test", test), ("val", val), ("train_extra", train_extra)):
            with open(os.path.join(a.out, name + ".txt"), "w") as f:
                f.write("\n".join(r["_id"] for r in rows) + "\n")
        rep["files"] = {n: sha256(os.path.join(a.out, n + ".txt")) for n in ("test", "val", "train_extra")}
        json.dump(rep, open(os.path.join(a.out, "twowiki_split_report.json"), "w"), indent=1)
        print(json.dumps(rep, indent=1)); return
    # --check: committed files must equal the rule's output exactly
    assert a.dir, "--dir required"
    ok = True
    for name, rows in (("test", test), ("val", val), ("train_extra", train_extra)):
        got = [l.strip() for l in open(os.path.join(a.dir, name + ".txt")) if l.strip()]
        want = [r["_id"] for r in rows]
        same = got == want
        ok &= same
        print(f"{name}: committed {len(got)} | rule {len(want)} | identical: {same}")
    ok &= rep["entities"]["shared_test_train_extra"] == 0 and rep["entities"]["shared_val_train_extra"] == 0
    if a100 is not None: ok &= rep["agenttts_100"]["all_inside_test"]
    print(json.dumps(rep, indent=1)); print("CHECK", "PASS" if ok else "FAIL"); sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
