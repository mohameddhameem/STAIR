#!/usr/bin/env python3
"""build_routeA_pools.py — rebuild AgentTTS-style pools for dev[500:1000].

Their recipe (retrieval_QA_data_generator.py, verified 17 Aug):
  gold paragraphs first (supporting-fact titles, sentences space-joined,
  BARE — no title prefix), then top-100 retrieved from the dev-paragraph
  corpus, dedup by exact string, shuffle.

Deliberate deviations, both stated in the ledger:
  * BM25 (self-contained) replaces their Elasticsearch multi_match over
    {title, text} — O3 makes their index irreproducible anyway. We score
    "title + ' ' + text" to approximate multi_match's title field.
  * The shuffle is SEEDED (blake2b per _id) — theirs was unseeded; ours is
    reproducible, an improvement we document rather than copy.

Output: /mnt/hdd/jonathan/data/routeA_test500.json
  [{_id, question, answer, answers, ctxs_candidate (bare), ctxs_gt}]
CPU only; ~10-20 min (BM25 over ~74k paragraphs x 500 queries).

Run: [LAB SSH]  python3 build_routeA_pools.py
"""
import json, math, re, random, hashlib
from collections import Counter, defaultdict

DEV = "/mnt/hdd/jonathan/data/hotpotqa-dev.json"
OUT = "/mnt/hdd/jonathan/data/routeA_test500.json"
TOPN = 100
TOK = lambda s: re.findall(r"[a-z0-9]+", s.lower())

print("loading dev ...")
dev = json.load(open(DEV))
print(f"{len(dev)} instances; building corpus from all context paragraphs ...")

# corpus: every (title, text) paragraph from every dev instance, deduped like
# their preprocess.py (which does NOT dedup — ES dedups by doc; we dedup by
# exact (title,text) string to avoid retrieving 40 copies of hot paragraphs)
seen = set(); corpus = []
for ex in dev:
    for t, sents in ex["context"]:
        text = " ".join(sents)
        key = (t, text)
        if key in seen: continue
        seen.add(key); corpus.append((t, text))
print(f"corpus paragraphs (deduped): {len(corpus)}")

print("indexing (BM25 over 'title + text') ...")
docs_tok = [TOK(t + " " + x) for t, x in corpus]
N = len(docs_tok)
avgdl = sum(len(d) for d in docs_tok) / N
df = Counter()
inv = defaultdict(list)
for j, d in enumerate(docs_tok):
    s = set(d)
    df.update(s)
    for w in s: inv[w].append(j)
tf_cache = [Counter(d) for d in docs_tok]
idf = {w: math.log(1 + (N - df[w] + 0.5) / (df[w] + 0.5)) for w in df}
K1, Bp = 1.5, 0.75

def top100(qtok):
    cand = set()
    for w in set(qtok):
        if w in inv and df[w] < N * 0.3:      # skip near-stopwords for speed
            cand.update(inv[w])
    scores = []
    for j in cand:
        tf = tf_cache[j]; dl = len(docs_tok[j]); s = 0.0
        for w in qtok:
            f = tf.get(w)
            if not f: continue
            s += idf[w] * f * (K1 + 1) / (f + K1 * (1 - Bp + Bp * dl / avgdl))
        scores.append((s, j))
    scores.sort(reverse=True)
    return [corpus[j][1] for _, j in scores[:TOPN]]

out = []
for n, ex in enumerate(dev[500:1000]):
    cd = {t: " ".join(sents) for t, sents in ex["context"]}
    gt = []
    for t, _sid in ex["supporting_facts"]:
        if t in cd and cd[t] not in gt:
            gt.append(cd[t])
    pool = gt.copy()
    for text in top100(TOK(ex["question"])):
        if text not in pool:
            pool.append(text)
    seed = int.from_bytes(hashlib.blake2b(ex["_id"].encode(), digest_size=8).digest(), "big")
    random.Random(seed).shuffle(pool)
    out.append({"_id": ex["_id"], "question": ex["question"],
                "answer": ex["answer"], "answers": [ex["answer"]],
                "ctxs_candidate": pool, "ctxs_gt": gt})
    if (n + 1) % 50 == 0:
        print(f"  {n+1}/500  (last pool size {len(pool)})", flush=True)

sizes = sorted(len(x["ctxs_candidate"]) for x in out)
print(f"pool sizes: min {sizes[0]} median {sizes[250]} max {sizes[-1]} "
      f"(theirs measured 100-104)")
gold_ok = sum(1 for x in out if all(g in x["ctxs_candidate"] for g in x["ctxs_gt"]))
print(f"gold fully in pool: {gold_ok}/500 (must be 500 by construction)")
json.dump(out, open(OUT, "w"))
print(f"wrote {OUT}")
