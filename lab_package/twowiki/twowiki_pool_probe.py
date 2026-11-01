#!/usr/bin/env python3
"""
twowiki_pool_probe.py -- READ-ONLY probe (CPU, ~1 min, no Elasticsearch, no model) run BEFORE the pool builder is written.
DRAFT v1.0 2026-11-01 for Jonathan's review.

Why: AgentTTS's pool builder (data/retrieval_QA_data_generator.py) reads queries.jsonl / corpus.jsonl / id_aliases.json,
and only id_aliases.json is public. Before rebuilding their pools for the 6,000 we must learn, from their shipped 100
(= dev[0:100]), how their files were derived from the official dev.json:
  Q1  gold sentences: does each ctxs_gt sentence equal the dev.json supporting-fact sentence EXACTLY, or only after
      stripping something (their instance 0 starts "(Wojna polsko-ruska) is a 2009 ..." -- title missing)?
  Q2  answers: can `answers` (gold + aliases) be rebuilt from dev.json + id_aliases.json, and from which field?
  Q3  corpus: are their candidate sentences all found among the sentences of dev.json's contexts (then the corpus
      = dev contexts, as §7 assumes), or do some come from elsewhere (then we need another corpus)?
Usage (lab): python3 -I twowiki_pool_probe.py --tts <agenttts>/data/2wikimultihopqa/tts_test.json \
               --dev /mnt/hdd/jonathan/data/2wiki_official/dev.json --aliases /mnt/hdd/jonathan/data/2wiki_official/id_aliases.json
Writes nothing except the printed report (redirect to a file if you want to keep it).
"""
import argparse, collections, json, re


def norm(s): return re.sub(r"\s+", " ", str(s)).strip()


def strip_title(sent, title):
    """Variant forms of a dev sentence that their preprocessing might have produced."""
    s = norm(sent); t = norm(title); base = re.sub(r"\s*\([^)]*\)$", "", t)   # "Polish-Russian War (film)" -> "Polish-Russian War"
    out = {s}
    for cand in (t, base):
        if cand and s.lower().startswith(cand.lower()):
            out.add(s[len(cand):].strip())
    return out


def load_aliases(path):
    txt = open(path, encoding="utf-8").read().strip()
    rows = json.loads(txt) if txt.startswith("[") else [json.loads(l) for l in txt.splitlines() if l.strip()]
    return {r["Q_id"]: r for r in rows}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tts", required=True); ap.add_argument("--dev", required=True); ap.add_argument("--aliases", required=True)
    a = ap.parse_args()
    tts = json.load(open(a.tts, encoding="utf-8")); dev = json.load(open(a.dev, encoding="utf-8")); al = load_aliases(a.aliases)
    print("dev.json keys:", sorted(dev[0].keys())); print("id_aliases example:", next(iter(al.values())))
    for k in sorted(dev[0].keys()):
        if k not in ("context",): print(f"  dev[0][{k!r}] =", str(dev[0][k])[:200])
    assert len(tts) == 100 and all(norm(t["question"]) == norm(d["question"]) for t, d in zip(tts, dev[:100])), "tts_test != dev[0:100] by position"

    # ---- Q1 gold sentences ----
    exact = variant = miss = 0; order_ok = 0; miss_ex = []
    for t, d in zip(tts, dev[:100]):
        ctx = {norm(ti): s for ti, s in d["context"]}
        gold_raw, gold_var = [], []
        for ti, sid in d["supporting_facts"]:
            sents = ctx.get(norm(ti), [])
            s = sents[sid] if sid < len(sents) else ""
            gold_raw.append(norm(s)); gold_var.append(strip_title(s, ti))
        for g in t["ctxs_gt"]:
            g = norm(g)
            if g in gold_raw: exact += 1
            elif any(g in v for v in gold_var): variant += 1
            else:
                miss += 1
                if len(miss_ex) < 3: miss_ex.append({"theirs": g[:160], "dev_gold": [x[:160] for x in gold_raw]})
        order_ok += [norm(g) for g in t["ctxs_gt"]] == gold_raw
    print(f"\nQ1 gold sentences (n={exact+variant+miss}): exact {exact} | match after title-strip {variant} | not found {miss} | same list & order as supporting_facts on {order_ok}/100")
    for e in miss_ex: print("   MISS:", json.dumps(e, ensure_ascii=False))

    # ---- Q2 answers ----
    first_ok = sum(norm(t["answers"][0]) == norm(d["answer"]) for t, d in zip(tts, dev[:100]))
    print(f"\nQ2 answers: answers[0] == dev answer on {first_ok}/100; instances with >1 entry: {sum(len(t['answers'])>1 for t in tts)}")
    idfields = [k for k in dev[0] if "id" in k.lower() and k != "_id"]
    print("   candidate id fields in dev.json:", idfields)
    for k in idfields:
        hit = 0
        for t, d in zip(tts, dev[:100]):
            ids = [x for x in re.split(r"[_\s,]+", str(d.get(k, ""))) if x]
            want = [norm(x) for x in t["answers"][1:]]
            for q in ids:
                if q in al and [norm(x) for x in al[q].get("aliases", [])] == want:
                    hit += 1; break
        print(f"   field {k!r}: an id whose aliases == their answers[1:] on {hit}/100")
    # reverse look-up: which Q_ids have alias lists equal to their answers[1:] (diagnostic for 3 instances with aliases)
    by_alias = collections.defaultdict(list)
    for q, r in al.items(): by_alias[tuple(norm(x) for x in r.get("aliases", []))].append(q)
    shown = 0
    for i, t in enumerate(tts):
        if len(t["answers"]) > 1 and shown < 3:
            print(f"   inst {i}: answers {t['answers'][:4]} -> Q_ids with exactly these aliases: {by_alias.get(tuple(norm(x) for x in t['answers'][1:]), [])[:5]}"); shown += 1

    # ---- Q3 corpus ----
    raw, var = set(), set()
    for d in dev:
        for ti, sents in d["context"]:
            for s in sents:
                raw.add(norm(s)); var |= strip_title(s, ti)
    n = r_hit = v_hit = 0; miss_c = []
    for t in tts:
        for c in t["ctxs_candidate"]:
            c = norm(c); n += 1
            if c in raw: r_hit += 1
            elif c in var: v_hit += 1
            elif len(miss_c) < 3: miss_c.append(c[:160])
    print(f"\nQ3 corpus: their candidates n={n}: in dev sentences exactly {r_hit} | after title-strip {v_hit} | not in dev contexts {n-r_hit-v_hit}")
    print(f"   dev corpus size: {len(raw)} distinct raw sentences")
    for c in miss_c: print("   NOT IN DEV:", c)
    lens = collections.Counter(len(t["ctxs_candidate"]) for t in tts)
    dup = sum(len(t["ctxs_candidate"]) - len({norm(c) for c in t["ctxs_candidate"]}) for t in tts)
    print(f"   pool sizes {sorted(lens.items())}; duplicate texts inside pools: {dup}")


if __name__ == "__main__":
    main()
