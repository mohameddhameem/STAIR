#!/usr/bin/env python3
"""
twowiki_pool_diag.py -- READ-ONLY diagnostic (lab, local Elasticsearch, ~1 min). DRAFT v1.0 2026-11-01.
Why: twowiki_pools.py validate gave gold/answers 100/100 but candidate Jaccard 0.069 vs AgentTTS's shipped pools, although
probe Q3 showed every one of their candidates is a dev.json sentence. This tests WHICH retrieval setting reproduces theirs,
on their 100, by the share of THEIR non-gold candidates that each variant's top-100 recovers:
  V1 multi_match title+text  (what twowiki_pools.py does)   V2 match text only   V3 match title only
plus, under V1 and V2: where their candidates rank in our top-1000, and the share of their candidates that come from
the question's OWN 10-paragraph context (if high, their corpus may have been per-question, not global).
Writes nothing; queries the existing index only.
Usage: python3 -I twowiki_pool_diag.py --dev DEV.json --tts <agenttts>/.../tts_test.json [--n 100]
"""
import argparse, json, statistics
from elasticsearch import Elasticsearch

INDEX = "2wiki_dev_sentences"


def top(es, q, fields, k):
    body = {"query": {"multi_match": {"query": q, "fields": fields}}} if len(fields) > 1 else {"query": {"match": {fields[0]: q}}}
    return [h["_source"]["text"] for h in es.search(index=INDEX, body=body, size=k)["hits"]["hits"]]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--dev", required=True); ap.add_argument("--tts", required=True); ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    es = Elasticsearch("http://localhost:9200")
    dev = json.load(open(a.dev, encoding="utf-8")); tts = json.load(open(a.tts, encoding="utf-8"))
    rec = {"V1_title+text": [], "V2_text": [], "V3_title": []}
    ranks = {"V1_title+text": [], "V2_text": []}; own = []
    for t, d in list(zip(tts, dev))[: a.n]:
        theirs = set(t["ctxs_candidate"]) - set(t["ctxs_gt"])
        own_ctx = {s for _, ss in d["context"] for s in ss}
        own.append(len(theirs & own_ctx) / len(theirs))
        for name, fields in (("V1_title+text", ["title", "text"]), ("V2_text", ["text"]), ("V3_title", ["title"])):
            got = top(es, d["question"], fields, 100)
            rec[name].append(len(theirs & set(got)) / len(theirs))
            if name in ranks:
                deep = top(es, d["question"], fields, 1000); pos = {s: i for i, s in enumerate(deep)}
                ranks[name] += [pos.get(s, 1000) for s in theirs]
    print(f"n = {a.n} instances; THEIR non-gold candidates recovered by our top-100:")
    for k, v in rec.items():
        print(f"  {k:15s} mean {statistics.mean(v):.3f}  median {statistics.median(v):.3f}  min {min(v):.3f}  max {max(v):.3f}")
    for k, v in ranks.items():
        inside = sum(r < 1000 for r in v) / len(v)
        print(f"  {k:15s} their candidates inside our top-1000: {inside:.3f}; median rank {statistics.median(v):.0f}")
    print(f"share of their non-gold candidates from the question's OWN context: mean {statistics.mean(own):.3f}")
    t, d = tts[0], dev[0]
    print("\ninstance 0 -- 5 of THEIR non-gold candidates:"); [print("   ", s[:120]) for s in list(set(t['ctxs_candidate']) - set(t['ctxs_gt']))[:5]]
    print("instance 0 -- OUR V1 top-5:"); [print("   ", s[:120]) for s in top(es, d["question"], ["title", "text"], 5)]


if __name__ == "__main__":
    main()
