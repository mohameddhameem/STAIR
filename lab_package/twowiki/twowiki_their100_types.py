#!/usr/bin/env python3
"""
twowiki_their100_types.py -- join AgentTTS's 100 (tts_test.json, no `_id`/`type`) to the official dev.json (dev[0:100])
by question text, and write their100_types.json: {question: type} plus a side table of official _id per position.
Used by twowiki_cells_score.py / twowiki_score_sidecar.py for the per-type readout. Lab (dev.json lives there):
  python3 -I twowiki_their100_types.py --tts <agenttts>/data/2wikimultihopqa/tts_test.json \
      --dev /mnt/hdd/jonathan/data/2wiki_official/dev.json --out <STAIR>/common/stair_common/splits/2wikimultihopqa/their100_types.json
"""
import argparse, collections, json, re


def norm(q): return re.sub(r"\s+", " ", str(q)).strip().lower()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tts", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tts = json.load(open(a.tts, encoding="utf-8")); dev = json.load(open(a.dev, encoding="utf-8"))
    by_q = {norm(r["question"]): r for r in dev[:100]}
    out, ids = {}, []
    for i, x in enumerate(tts):
        r = by_q.get(norm(x["question"]))
        assert r is not None, f"instance {i} not found in dev[0:100]: {x['question'][:80]}"
        out[x["question"]] = r["type"]; ids.append({"pos": i, "adapted_id": f"2wiki100-{i:03d}", "official_id": r["_id"], "type": r["type"]})
    json.dump({"question_to_type": out, "positions": ids, "type_counts": dict(collections.Counter(v for v in out.values()))},
              open(a.out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(json.dumps(collections.Counter(out.values()), indent=1), "| n", len(out))


if __name__ == "__main__":
    main()
