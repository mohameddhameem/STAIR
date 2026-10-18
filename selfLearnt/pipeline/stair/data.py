"""Question sets on the team standard (STAIR common/, stair_common.splits, seed 20260729), all in AgentTTS's format:
{qid, question, answer, type, level, ret_answers, ctxs_candidate}.

  test   : AgentTTS's 500 test questions, dev file [500:1000] (stair_common split `test`) - scored once
  val    : dev file [0:500] (split `val`) - checkpoint choice (validation loss), probes, rule learning
  search : the first 50 of val = dev file [0:50], AgentTTS's own search questions (workstream AL-01)
  train  : split `train`, 12,000 questions drawn from the official training file - fine-tuning only

Candidates (as AgentTTS: data/retrieval_QA_data_generator.py): the gold paragraphs + the top-100 BM25 paragraphs of the
same file's corpus (title + text indexed, as their Elasticsearch multi_match), duplicates dropped, shuffled.
"""
import json
import os
import random

from .common import BM25, log, write_json


HF_REPO = "hotpotqa/hotpot_qa"          # the same questions and IDs (distractor setting), as parquet on Hugging Face


def _load_hotpot(path, url):
    """The HotpotQA file in its original JSON layout: from disk, else the official URL, else Hugging Face (converted)."""
    if path and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if url:
        import urllib.request
        try:
            log(f"downloading {url} ...")
            urllib.request.urlretrieve(url, path + ".part")
            os.rename(path + ".part", path)
            return json.load(open(path, encoding="utf-8"))
        except Exception as e:  # noqa: BLE001
            log(f"official download failed ({type(e).__name__}: {str(e)[:120]}); using Hugging Face {HF_REPO} instead")
            if os.path.exists(path + ".part"):
                os.remove(path + ".part")
    split = "train" if "train" in os.path.basename(path) else "validation"
    data = _from_hf(split)
    json.dump(data, open(path + ".part", "w", encoding="utf-8"), ensure_ascii=False)
    os.rename(path + ".part", path)
    log(f"{path}: {len(data)} questions (from Hugging Face, {split})")
    return data


def _from_hf(split):
    """HF `hotpot_qa` distractor parquet -> the official JSON layout (_id, context [[title, sentences]], supporting_facts)."""
    import pandas as pd
    from huggingface_hub import hf_hub_download, list_repo_files
    files = sorted(f for f in list_repo_files(HF_REPO, repo_type="dataset")
                   if f.startswith("distractor/") and f.endswith(".parquet") and os.path.basename(f).startswith(split))
    assert files, f"no distractor/{split} parquet files in {HF_REPO}"
    out = []
    for f in files:
        df = pd.read_parquet(hf_hub_download(HF_REPO, f, repo_type="dataset"))
        for r in df.to_dict("records"):
            ctx, sf = r["context"], r["supporting_facts"]
            out.append({"_id": r["id"], "question": r["question"], "answer": r["answer"], "type": r.get("type"), "level": r.get("level"),
                        "supporting_facts": [[t, int(i)] for t, i in zip(list(sf["title"]), list(sf["sent_id"]))],
                        "context": [[t, [str(x) for x in sents]] for t, sents in zip(list(ctx["title"]), list(ctx["sentences"]))]})
    return out


def _corpus(data):
    seen, titles, texts = set(), [], []
    for ex in data:
        for title, sents in ex["context"]:
            t = " ".join(sents)
            if t not in seen:
                seen.add(t)
                titles.append(title)
                texts.append(t)
    return titles, texts


def _build(exs, texts, bm25, rng, n_retrieve):
    out = []
    for ex in exs:
        ctx = {t: " ".join(s) for t, s in ex["context"]}
        gold = []
        for title, _ in ex["supporting_facts"]:
            if title in ctx and ctx[title] not in gold:
                gold.append(ctx[title])
        cands = gold.copy()
        for j in bm25.search(ex["question"], n_retrieve):
            if texts[j] not in cands:
                cands.append(texts[j])
        rng.shuffle(cands)
        out.append({"qid": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type"), "level": ex.get("level"),
                    "ret_answers": [cands.index(g) for g in gold], "ctxs_candidate": cands})
    return out


SPLITS = ("search", "val", "test", "train")


def _ids(dataset, part):
    from stair_common.splits import load_split
    return load_split(dataset, part)


def build(cfg):
    d = cfg["data"]
    out = cfg["paths"]["data_dir"]
    if all(os.path.exists(os.path.join(out, f"{s}.json")) for s in SPLITS):
        log(f"question sets already built in {out}")
        return
    os.makedirs(out, exist_ok=True)
    nr = int(d["n_retrieve"])
    ds = d.get("dataset", "hotpotqa")
    dv = _load_hotpot(d["hotpot_dev"], d.get("hotpot_dev_url"))
    by_id = {x["_id"]: x for x in dv}
    titles, texts = _corpus(dv)
    log(f"dev file: {len(dv)} questions, {len(texts)} paragraphs; BM25 over title + text ...")
    bm = BM25([a + " " + b for a, b in zip(titles, texts)])
    rng = random.Random(d["seed"])
    val_ids, test_ids = _ids(ds, "val"), _ids(ds, "test")
    sets = {"val": _build([by_id[i] for i in val_ids], texts, bm, rng, nr),
            "test": _build([by_id[i] for i in test_ids], texts, bm, rng, nr)}
    lim = d.get("limit") or {}  # dry runs only: keep the first n of a part
    for k in ("val", "test"):
        if lim.get(k):
            sets[k] = sets[k][:int(lim[k])]
    sets["search"] = sets["val"][:int(d.get("n_search", 50))]
    # training questions: the split part named in the config, from the official train file (or the dev file in a dry run)
    tr_ids = _ids(ds, d.get("train_part", "train"))
    if lim.get("train"):
        tr_ids = tr_ids[:int(lim["train"])]
    if d.get("train_source", "train") == "dev":
        src, tb, ttexts = by_id, bm, texts
    else:
        tr = _load_hotpot(d["hotpot_train"], d.get("hotpot_train_url"))
        src = {x["_id"]: x for x in tr}
        t2, ttexts = _corpus(tr)
        log(f"training file: {len(tr)} questions, {len(ttexts)} paragraphs; BM25 ...")
        tb = BM25([a + " " + b for a, b in zip(t2, ttexts)])
    missing = [i for i in tr_ids if i not in src]
    assert not missing, f"{len(missing)} training IDs of split '{d.get('train_part', 'train')}' are not in the training source"
    sets["train"] = _build([src[i] for i in tr_ids], ttexts, tb, rng, nr)
    for name in SPLITS:
        json.dump(sets[name], open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        log(f"{name}: {len(sets[name])} questions")
    write_json(os.path.join(out, "data_config.json"), dict(d, splits={k: len(v) for k, v in sets.items()}))


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
