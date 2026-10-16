"""Question sets, all in AgentTTS's format: {qid, question, answer, type, level, ret_answers, ctxs_candidate}.

From HotpotQA's dev file (as AgentTTS: data/retrieval_QA_data_generator.py):
  search : dev[0:50]     AgentTTS's search set (its 'train' = dev[0:100], of which the search uses the first 50)
  test   : dev[500:1000] AgentTTS's 500 test questions - every workstream is scored on these, once
  Candidates: the gold paragraphs + the top-100 BM25 paragraphs of the whole dev-file corpus (title + text indexed,
  as their Elasticsearch multi_match), duplicates dropped, shuffled.
From HotpotQA's training file (hard level, never overlapping the above):
  train  : fine-tuning (retrievers: shortlist targets; answerers: answer targets)
  val    : checkpoint choice
  dev    : calibration, probes and the rule-learning micro-experiments of workstreams 3-5
  Candidates built the same way from the training-file corpus.
"""
import json
import os
import random

from .common import BM25, log, write_json


def _load_hotpot(path, url):
    if path and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    assert url, f"{path} not found and no download URL given"
    import urllib.request
    os.makedirs(os.path.dirname(path), exist_ok=True)
    log(f"downloading {url} ...")
    urllib.request.urlretrieve(url, path + ".part")
    os.rename(path + ".part", path)
    return json.load(open(path, encoding="utf-8"))


def _corpus(data, idxs):
    seen, titles, texts = set(), [], []
    for i in idxs:
        for title, sents in data[i]["context"]:
            t = " ".join(sents)
            if t not in seen:
                seen.add(t)
                titles.append(title)
                texts.append(t)
    return titles, texts


def _build(items, data, texts, bm25, rng, n_retrieve):
    out = []
    for i in items:
        ex = data[i]
        ctx = {t: " ".join(s) for t, s in ex["context"]}
        gold = []
        for title, _ in ex["supporting_facts"]:
            if title in ctx and ctx[title] not in gold:
                gold.append(ctx[title])
        if len(gold) < 2:
            continue
        cands = gold.copy()
        for j in bm25.search(ex["question"], n_retrieve):
            if texts[j] not in cands:
                cands.append(texts[j])
        rng.shuffle(cands)
        out.append({"qid": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type"), "level": ex.get("level"),
                    "ret_answers": [cands.index(g) for g in gold], "ctxs_candidate": cands})
    return out


SPLITS = ("search", "test", "train", "val", "dev")


def build(cfg):
    d = cfg["data"]
    out = cfg["paths"]["data_dir"]
    if all(os.path.exists(os.path.join(out, f"{s}.json")) for s in SPLITS):
        log(f"question sets already built in {out}")
        return
    os.makedirs(out, exist_ok=True)
    nr = int(d["n_retrieve"])
    # --- AgentTTS sets from the dev file
    dv = _load_hotpot(d["hotpot_dev"], d.get("hotpot_dev_url"))
    titles, texts = _corpus(dv, range(len(dv)))
    log(f"dev file: {len(dv)} questions, {len(texts)} paragraphs; BM25 over title + text ...")
    bm = BM25([a + " " + b for a, b in zip(titles, texts)])
    rng = random.Random(d["seed"])
    s0, s1 = d["search_range"]
    t0, t1 = d["test_range"]
    for name, idxs in (("search", range(s0, min(s1, len(dv)))), ("test", range(t0, min(t1, len(dv))))):
        rows = _build(idxs, dv, texts, bm, rng, nr)
        json.dump(rows, open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        log(f"{name}: {len(rows)} questions (dev file [{idxs.start}:{idxs.stop}])")
    # --- our sets from the training file
    tr = _load_hotpot(d["hotpot_train"], d.get("hotpot_train_url"))
    if d.get("level"):
        tr = [x for x in tr if x.get("level") == d["level"]]
    rng = random.Random(d["seed"] + 1)
    pool = rng.sample(range(len(tr)), len(tr))          # the training and dev files share no questions
    n_tr, n_va, n_de = int(d["n_train"]), int(d["n_val"]), int(d["n_dev"])
    assert n_tr + n_va + n_de <= len(pool), f"only {len(pool)} training-file questions at level {d.get('level')}"
    titles, texts = _corpus(tr, range(len(tr)))
    log(f"training file: {len(tr)} questions, {len(texts)} paragraphs; BM25 ...")
    bm = BM25([a + " " + b for a, b in zip(titles, texts)])
    for name, idxs in (("train", pool[:n_tr]), ("val", pool[n_tr:n_tr + n_va]), ("dev", pool[n_tr + n_va:n_tr + n_va + n_de])):
        rows = _build(idxs, tr, texts, bm, rng, nr)
        json.dump(rows, open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        log(f"{name}: {len(rows)} questions")
    write_json(os.path.join(out, "data_config.json"), d)


_CACHE = {}


def load(cfg, split):
    p = os.path.join(cfg["paths"]["data_dir"], f"{split}.json")
    if p not in _CACHE:
        _CACHE[p] = json.load(open(p, encoding="utf-8"))
    return _CACHE[p]


def qmap(cfg, splits):
    out = {}
    for s in splits:
        for q in load(cfg, s):
            out[q["qid"]] = q
    return out
