#!/usr/bin/env python3
"""
twowiki_pools.py -- rebuild AgentTTS-style candidate pools for 2Wiki instances (lab, CPU + local Elasticsearch).
DRAFT v1.0 2026-11-01 for Jonathan's line-by-line review.

Construction = AgentTTS data/retrieval_QA_data_generator.py (repo 3f0c2df), with their three unreleased inputs rebuilt
from the official dev.json as the probe established (twowiki_pool_probe.py, 1 Nov: gold 244/244 exact, aliases via
answer_id 86/86, candidates 10,085/10,085 inside dev contexts, no duplicate texts in any pool):
  corpus   = every DISTINCT sentence of dev.json's contexts (by text; first title seen kept)  -> their corpus.jsonl
  gold     = the supporting_facts sentences, in supporting_facts order                       -> their queries ctxs
  answers  = [answer] + id_aliases[answer_id].aliases  ([] if answer_id has no entry)         -> their `answers`
  index    = mapping {title: text, text: text}  (theirs, unchanged)
  query    = multi_match on ["title","text"], size 100, query = the question                  (theirs, unchanged)
  pool     = gold first, then each hit's text if not already present, then shuffled            (theirs, unchanged)
DIFFERENCES FROM THEIR CODE (each deliberate, recorded in the prereg):
  D1 shuffle is SEEDED per instance: random.Random(f"{seed}:{_id}") (seed 17 default) -- theirs is unseeded, so their
     order cannot be reproduced; per-instance seeding makes a pool independent of processing order / chunking.
  D2 indexing uses helpers.bulk instead of one es.index call per sentence (same documents, same mapping; faster);
     index is created with 1 shard / 0 replicas so BM25 statistics are global and deterministic.
  D3 records carry `_id` (official dev id), `type`, `split`, `answer`, `supporting_titles` (the cells runner's fields,
     exactly as twowiki_cells_adapt.py adds them) -- nothing inside a sentence, question or answer list is changed.
Known residual: ties in BM25 score at rank 100 may order differently from their index (different internal doc ids).

Usage (lab, slr-env, Elasticsearch 7.17 on localhost:9200):
  python3 -I twowiki_pools.py index    --dev DEV.json
  python3 -I twowiki_pools.py validate --dev DEV.json --aliases ALIASES.json --tts <agenttts>/.../tts_test.json --out DIR
  python3 -I twowiki_pools.py build    --dev DEV.json --aliases ALIASES.json --ids ids_a.txt [ids_b.txt ...] --tag NAME --out DIR
"""
import argparse, collections, csv, datetime, hashlib, json, os, random, re

INDEX = "2wiki_dev_sentences"
RETRIEVE_N = 100


def norm(s): return re.sub(r"\s+", " ", str(s)).strip()
def title_of(c): return c.split(':', 1)[0]                     # identical to run_labels_agenttts.title_of
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def es_client():
    from elasticsearch import Elasticsearch
    return Elasticsearch("http://localhost:9200")


def load_aliases(path):
    txt = open(path, encoding="utf-8").read().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    return {r["Q_id"]: r.get("aliases", []) for r in rows}


def corpus_from_dev(dev):
    seen, docs = set(), []
    for d in dev:
        for title, sents in d["context"]:
            for s in sents:
                if s not in seen:
                    seen.add(s); docs.append({"title": title, "text": s})
    return docs


def cmd_index(a):
    from elasticsearch import helpers
    es = es_client(); dev = json.load(open(a.dev, encoding="utf-8"))
    if es.indices.exists(index=INDEX):
        raise SystemExit(f"REFUSING: index {INDEX} exists ({es.count(index=INDEX)['count']} docs) -- delete it deliberately to rebuild")
    es.indices.create(index=INDEX, body={"settings": {"number_of_shards": 1, "number_of_replicas": 0},
                                         "mappings": {"properties": {"title": {"type": "text"}, "text": {"type": "text"}}}})
    docs = corpus_from_dev(dev)
    helpers.bulk(es, ({"_index": INDEX, "_source": d} for d in docs), chunk_size=5000)
    es.indices.refresh(index=INDEX)
    n = es.count(index=INDEX)["count"]
    print(json.dumps({"index": INDEX, "docs_indexed": n, "distinct_sentences": len(docs), "dev_sha256": sha(a.dev),
                      "es_version": es.info()["version"]["number"]}, indent=1))
    assert n == len(docs), "indexed count != distinct sentences"


def make_record(es, d, aliases, seed):
    ctx = {t: s for t, s in d["context"]}
    gold = [ctx[t][i] for t, i in d["supporting_facts"]]
    answers = [d["answer"]] + list(aliases.get(d["answer_id"], []))
    ctxs = gold.copy()
    hits = es.search(index=INDEX, body={"query": {"multi_match": {"query": d["question"], "fields": ["title", "text"]}},
                                        "size": RETRIEVE_N})["hits"]["hits"]
    for h in hits:
        t = h["_source"]["text"]
        if t not in ctxs:
            ctxs.append(t)
    random.Random(f"{seed}:{d['_id']}").shuffle(ctxs)                                      # D1
    return {"_id": d["_id"], "type": d["type"], "question": d["question"], "answers": answers, "answer": answers[0],
            "ctxs_candidate": ctxs, "ctxs_gt": gold, "supporting_titles": [title_of(g) for g in gold],
            "n_hits": len(hits)}


def cmd_validate(a):
    """Rebuild their 100 (dev[0:100]) and compare with the pools they shipped. Writes CSV (+ xlsx if openpyxl)."""
    es = es_client(); dev = json.load(open(a.dev, encoding="utf-8")); al = load_aliases(a.aliases)
    tts = json.load(open(a.tts, encoding="utf-8")); assert len(tts) == 100
    rows = []
    for i, (t, d) in enumerate(zip(tts, dev[:100])):
        assert norm(t["question"]) == norm(d["question"]), f"position {i}: question mismatch"
        r = make_record(es, d, al, a.seed)
        A, B = set(t["ctxs_candidate"]), set(r["ctxs_candidate"]); G = set(r["ctxs_gt"])
        rows.append({"pos": i, "_id": d["_id"], "type": d["type"], "answers_equal": t["answers"] == r["answers"],
                     "gold_equal": t["ctxs_gt"] == r["ctxs_gt"], "gold_in_ours": G <= B,
                     "n_theirs": len(A), "n_ours": len(B), "overlap": len(A & B),
                     "jaccard_all": round(len(A & B) / len(A | B), 4),
                     "jaccard_non_gold": round(len((A - G) & (B - G)) / max(1, len((A - G) | (B - G))), 4)})
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, "pool_validation_their100.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    try:
        from openpyxl import Workbook
        wb = Workbook(); ws = wb.active; ws.append(list(rows[0]))
        for r in rows: ws.append(list(r.values()))
        wb.save(os.path.join(a.out, "pool_validation_their100.xlsx"))
    except ImportError:
        pass
    jac = sorted(r["jaccard_all"] for r in rows)
    summ = {"n": len(rows), "answers_equal": sum(r["answers_equal"] for r in rows), "gold_equal": sum(r["gold_equal"] for r in rows),
            "gold_in_ours": sum(r["gold_in_ours"] for r in rows), "identical_pools": sum(r["jaccard_all"] == 1.0 for r in rows),
            "jaccard_all_mean": round(sum(jac) / len(jac), 4), "jaccard_all_min": jac[0], "jaccard_all_median": jac[len(jac) // 2],
            "jaccard_non_gold_mean": round(sum(r["jaccard_non_gold"] for r in rows) / len(rows), 4),
            "pool_sizes_ours": dict(sorted(collections.Counter(r["n_ours"] for r in rows).items())),
            "pool_sizes_theirs": dict(sorted(collections.Counter(r["n_theirs"] for r in rows).items())),
            "seed": a.seed, "es_version": es.info()["version"]["number"], "tts_sha256": sha(a.tts)}
    json.dump(summ, open(os.path.join(a.out, "pool_validation_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


def cmd_build(a):
    es = es_client(); dev = json.load(open(a.dev, encoding="utf-8")); al = load_aliases(a.aliases)
    by_id = {d["_id"]: d for d in dev}
    out_path = os.path.join(a.out, f"pools_{a.tag}.json")
    if os.path.exists(out_path):
        raise SystemExit(f"REFUSING: {out_path} exists")
    recs, ids_sha = [], {}
    for f in a.ids:
        ids = [l.strip() for l in open(f) if l.strip()]
        if a.limit_per_file:
            ids = ids[:a.limit_per_file]
        ids_sha[f] = {"sha256": sha(f), "n_used": len(ids)}
        for _id in ids:
            r = make_record(es, by_id[_id], al, a.seed); r["split"] = os.path.basename(f).rsplit(".", 1)[0]
            assert set(r["ctxs_gt"]) <= set(r["ctxs_candidate"])
            recs.append(r)
    assert len({r["_id"] for r in recs}) == len(recs), "duplicate _id across the ids files"
    os.makedirs(a.out, exist_ok=True)
    json.dump(recs, open(out_path, "w", encoding="utf-8"), ensure_ascii=False)
    man = {"version": "twowiki_pools v1.0 2026-11-01", "tag": a.tag, "n": len(recs), "seed": a.seed, "retrieve_n": RETRIEVE_N,
           "index": INDEX, "es_version": es.info()["version"]["number"], "dev_sha256": sha(a.dev), "aliases_sha256": sha(a.aliases),
           "ids_files": ids_sha, "max_pool": max(len(r["ctxs_candidate"]) for r in recs),
           "by_split": dict(collections.Counter(r["split"] for r in recs)), "output_sha256": sha(out_path),
           "created": datetime.datetime.now().isoformat(timespec="seconds")}
    json.dump(man, open(os.path.join(a.out, f"pools_{a.tag}_manifest.json"), "w"), indent=1)
    print(json.dumps(man, indent=1))


def main():
    ap = argparse.ArgumentParser(); sub = ap.add_subparsers(dest="op", required=True)
    p = sub.add_parser("index"); p.add_argument("--dev", required=True)
    for name in ("validate", "build"):
        q = sub.add_parser(name); q.add_argument("--dev", required=True); q.add_argument("--aliases", required=True)
        q.add_argument("--out", required=True); q.add_argument("--seed", type=int, default=17)
    sub.choices["validate"].add_argument("--tts", required=True)
    b = sub.choices["build"]; b.add_argument("--ids", nargs="+", required=True); b.add_argument("--tag", required=True)
    b.add_argument("--limit-per-file", type=int, default=0, help="use only the first N ids of each file (0 = all)")
    a = ap.parse_args()
    {"index": cmd_index, "validate": cmd_validate, "build": cmd_build}[a.op](a)


if __name__ == "__main__":
    main()
