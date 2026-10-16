"""Question sets in one format: {qid, question, answer, type, level, ret_answers, ctxs_candidate}.

  train / val : HotpotQA training file (hard level) - teaching the retrievers to shortlist
  dev         : more HotpotQA training questions, never used for training - calibration and rule-learning micro-experiments
  test        : HotpotQA dev file (distractor setting) - used once, at the end

Each question gets its gold paragraphs plus BM25 distractors from the paragraphs of its own file, shuffled (as in the
development notebooks)."""
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


def _build(items, data, corpus, bm25, rng, n_cand):
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
        for j in bm25.search(ex["question"], n_cand + 10):
            if len(cands) >= n_cand:
                break
            if corpus[j] not in cands:
                cands.append(corpus[j])
        rng.shuffle(cands)
        out.append({"qid": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type"), "level": ex.get("level"),
                    "ret_answers": [cands.index(g) for g in gold], "ctxs_candidate": cands})
    return out


def _corpus(data, idxs):
    seen, corpus = set(), []
    for i in idxs:
        for _, sents in data[i]["context"]:
            t = " ".join(sents)
            if t not in seen:
                seen.add(t)
                corpus.append(t)
    return corpus


def build(cfg):
    d = cfg["data"]
    out = cfg["paths"]["data_dir"]
    if all(os.path.exists(os.path.join(out, f"{s}.json")) for s in ("train", "val", "dev", "test")):
        log(f"question sets already built in {out}")
        return
    os.makedirs(out, exist_ok=True)
    # train / val / dev from the training file
    tr = _load_hotpot(d["hotpot_train"], d.get("hotpot_train_url"))
    if d.get("level"):
        tr = [x for x in tr if x.get("level") == d["level"]]
    rng = random.Random(d["seed"])
    pool = rng.sample(range(len(tr)), len(tr))
    n_tr, n_va, n_de = d["n_train"], d["n_val"], d["n_dev"]
    assert n_tr + n_va + n_de <= len(pool), f"only {len(pool)} training-file questions at level {d.get('level')}"
    corpus = _corpus(tr, pool)
    log(f"training file: {len(tr)} questions, {len(corpus)} paragraphs; BM25 ...")
    bm = BM25(corpus)
    for name, idxs in (("train", pool[:n_tr]), ("val", pool[n_tr:n_tr + n_va]), ("dev", pool[n_tr + n_va:n_tr + n_va + n_de])):
        rows = _build(idxs, tr, corpus, bm, rng, d["n_candidates"])
        json.dump(rows, open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        log(f"{name}: {len(rows)} questions")
    # test from the dev file
    dv = _load_hotpot(d["hotpot_dev"], d.get("hotpot_dev_url"))
    corpus = _corpus(dv, range(len(dv)))
    bm = BM25(corpus)
    rng = random.Random(d["test_seed"])
    n_te = d["n_test"] if d["n_test"] else len(dv)
    idxs = rng.sample(range(len(dv)), min(n_te, len(dv)))
    rows = _build(idxs, dv, corpus, bm, rng, d["n_candidates"])
    json.dump(rows, open(os.path.join(out, "test.json"), "w", encoding="utf-8"), ensure_ascii=False)
    log(f"test: {len(rows)} questions")
    write_json(os.path.join(out, "data_config.json"), d)


def load(cfg, split):
    return json.load(open(os.path.join(cfg["paths"]["data_dir"], f"{split}.json"), encoding="utf-8"))
