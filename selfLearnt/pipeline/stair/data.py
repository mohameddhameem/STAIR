"""Question sets on the team standard (STAIR common/, stair_common.splits, seed 20260729), all in AgentTTS's format:
{qid, question, answer, type, level, ret_answers, ctxs_candidate}.

  test   : the stair_common part `data.test_part`: AgentTTS's 500 (`test`, dev [500:1000]) or the HotpotQA golden set's
           `slr-test` (1,481, common 0.3.0) - scored once
  val    : `data.val_part`: dev [0:500] (`val`) or `slr-val` (1,110) - checkpoint choice (validation loss), probes, rule learning
  search : the first 50 of val, the questions AgentTTS's search scores its settings on (workstream AL-01)
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


def _tts_sets(cfg, val_ids, test_ids):
    """val / test from AgentTTS's own HotpotQA file (tts_test.json, the pools Jonathan's always-strong reference uses):
    the first 100 candidates in the file's order, gold passages re-injected as in his trim_pool (stair/ss_jon.py)."""
    from .ss_jon import _tts, title_of, trim_pool
    tts = _tts(cfg)
    pos = {(x.get("_id") or x.get("id")): i for i, x in enumerate(tts)}
    out = {}
    for name, ids in (("val", val_ids), ("test", test_ids)):
        rows = []
        for q in ids:
            inst = dict(tts[pos[q]])
            pool = trim_pool(inst)
            golds = list(dict.fromkeys(inst.get("supporting_titles", [])))
            ret = [i for i, c in enumerate(pool) if title_of(c) in golds]
            g = inst.get("answers", inst.get("answer"))
            g = g[0] if isinstance(g, list) and len(g) == 1 else g
            rows.append({"qid": q, "question": inst["question"], "answer": g, "type": inst.get("type"), "level": inst.get("level"),
                         "ret_answers": ret, "ctxs_candidate": pool})
        out[name] = rows
        log(f"{name}: {len(rows)} questions from tts_test.json; both gold passages in the pool for "
            f"{sum(1 for r in rows if len(r['ret_answers']) >= 2)}")
    return out


# ----------------------------------------------------------------------------------
# 2WikiMultiHopQA on Jonathan's setup (branch jonathan, lab_package/twowiki): his split of the official dev file
# (split6000_manifest.json: test600 / val400 / train5000) and his pre-built candidate pools (twowiki_pools.py v1.1,
# Elasticsearch over the dev supporting-fact sentences, gold + top-100, shuffled with random.Random("17:<_id>")).
# The pools are used as he built them: nothing inside a record is changed; gold = his `answers` list (gold + aliases).
# ----------------------------------------------------------------------------------
TWOWIKI_PARTS = {"val": "val400", "test": "test600", "train": "train5000"}
TWOWIKI_N = {"val400": 400, "test600": 600, "train5000": 5000}
TWOWIKI_SPLIT_DIR = "common/stair_common/splits/2wikimultihopqa"


def _repo_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def jonathan_ids(dataset, part):
    """One part of Jonathan's split, read from branch jonathan (our copy of common/ still has the older split)."""
    import subprocess
    r = subprocess.run(["git", "-C", _repo_root(), "show", f"origin/jonathan:{TWOWIKI_SPLIT_DIR}/{part}.txt"],
                       capture_output=True, text=True)
    if r.returncode:
        raise SystemExit(f"run `git fetch origin jonathan` first (the {dataset} split {part} is on that branch)")
    return [x.strip() for x in r.stdout.splitlines() if x.strip()]


def _sha(p):
    import hashlib
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def _jon_records(pools_dir):
    """Every record of his pool files: pools_<part>.json, or the chunk files of his cluster array (chunks/<part>_kk.json,
    each checked against chunks_manifest.json)."""
    recs, used = {}, {}
    files = [os.path.join(pools_dir, f"pools_{p}.json") for p in TWOWIKI_N]
    if all(os.path.exists(f) for f in files):
        for f in files:
            used[os.path.basename(f)] = _sha(f)
            for r in json.load(open(f, encoding="utf-8")):
                recs[r["_id"]] = r
        return recs, used
    cdir = pools_dir if os.path.exists(os.path.join(pools_dir, "chunks_manifest.json")) else os.path.join(pools_dir, "chunks")
    man_p = os.path.join(cdir, "chunks_manifest.json")
    if not os.path.exists(man_p):
        raise SystemExit(f"Jonathan's 2Wiki pools not found in {pools_dir}: expected pools_test600/val400/train5000.json, "
                         "or chunks/ with chunks_manifest.json")
    man = json.load(open(man_p))
    for name, info in man["chunks"].items():
        p = os.path.join(cdir, name + ".json")
        got = _sha(p)
        assert got == info["sha256"], f"{name}: sha256 {got[:12]}... does not match the manifest ({info['sha256'][:12]}...)"
        used[name] = got
        for r in json.load(open(p, encoding="utf-8")):
            recs[r["_id"]] = r
    return recs, used


def _jon_row(r):
    cands = r["ctxs_candidate"]
    g = r.get("answers") or [r["answer"]]
    gold = [s for s in dict.fromkeys(r["ctxs_gt"]) if s in cands]    # (20 train5000 records have one supporting fact
    return {"qid": r["_id"], "question": r["question"], "answer": g[0] if len(g) == 1 else list(g), "type": r.get("type"),
            "level": None, "ret_answers": [cands.index(s) for s in gold], "ctxs_candidate": cands,     # out of range: kept)
            "_gold_missing": len(set(r["ctxs_gt"])) - len(gold)}


def _build_twowiki(cfg, out):
    d = cfg["data"]
    pdir = d.get("pools_dir") or ""
    if not pdir or "${" in pdir:                 # STAIR_2WIKI_POOLS not set: the default place under the project folder
        pdir = os.path.join(cfg["paths"]["project_dir"], "2wiki", "pools6000")
    recs, used = _jon_records(pdir)
    sets, ids_of = {}, {}
    for split, part in TWOWIKI_PARTS.items():
        ids = jonathan_ids(d["dataset"], part)
        assert len(ids) == TWOWIKI_N[part], f"{part}: {len(ids)} ids, expected {TWOWIKI_N[part]}"
        miss = [i for i in ids if i not in recs]
        assert not miss, f"{part}: {len(miss)} ids have no pool record (first {miss[0]})"
        rows = [_jon_row(recs[i]) for i in ids]
        short = sum(q.pop("_gold_missing") > 0 for q in rows)
        if short:
            log(f"{part}: {short} questions have a supporting fact outside their pool (kept, as Jonathan's runs)")
        assert all(q["ret_answers"] for q in rows), f"{part}: a question has no gold sentence in its pool"
        sets[split], ids_of[split] = rows, ids
    assert not (set(ids_of["train"]) & (set(ids_of["val"]) | set(ids_of["test"]))) and not (set(ids_of["val"]) & set(ids_of["test"]))
    lim = d.get("limit") or {}
    for k in ("val", "test", "train"):
        if lim.get(k):
            sets[k] = sets[k][:int(lim[k])]
    sets["search"] = sets["val"][:int(d.get("n_search", 50))]
    # his split files, for the team schema check of our results tables (stair_common.schema.validate)
    sd = os.path.join(out, "splits", d["dataset"])
    os.makedirs(sd, exist_ok=True)
    for split, part in TWOWIKI_PARTS.items():
        open(os.path.join(sd, f"{part}.txt"), "w").write("\n".join(ids_of[split]) + "\n")
    for name in SPLITS:
        json.dump(sets[name], open(os.path.join(out, f"{name}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        pools = [len(q["ctxs_candidate"]) for q in sets[name]]
        golds = [len(q["ret_answers"]) for q in sets[name]]
        log(f"{name}: {len(sets[name])} questions, {min(pools)}-{max(pools)} candidate sentences, "
            f"{min(golds)}-{max(golds)} gold sentences each")
    write_json(os.path.join(out, "data_config.json"), dict(d, splits={k: len(v) for k, v in sets.items()},
                                                          parts=TWOWIKI_PARTS, pool_files_sha256=used))


def splits_dir(cfg):
    """Where the team schema check finds the split files of this run's results (None = the installed common package)."""
    p = os.path.join(cfg["paths"]["data_dir"], "splits")
    return p if cfg["data"].get("pool_source") == "jonathan_2wiki" else None


def build(cfg):
    d = cfg["data"]
    out = cfg["paths"]["data_dir"]
    if all(os.path.exists(os.path.join(out, f"{s}.json")) for s in SPLITS):
        log(f"question sets already built in {out}")
        return
    os.makedirs(out, exist_ok=True)
    if d.get("pool_source") == "jonathan_2wiki":
        return _build_twowiki(cfg, out)
    nr = int(d["n_retrieve"])
    ds = d.get("dataset", "hotpotqa")
    dv = _load_hotpot(d["hotpot_dev"], d.get("hotpot_dev_url"))
    by_id = {x["_id"]: x for x in dv}
    titles, texts = _corpus(dv)
    log(f"dev file: {len(dv)} questions, {len(texts)} paragraphs; BM25 over title + text ...")
    bm = BM25([a + " " + b for a, b in zip(titles, texts)])
    rng = random.Random(d["seed"])
    val_ids, test_ids = _ids(ds, d.get("val_part", "val")), _ids(ds, d.get("test_part", "test"))
    if d.get("pool_source") == "tts":          # AgentTTS's own pools (the golden set's reference passages)
        sets = _tts_sets(cfg, val_ids, test_ids)
    else:                                       # our own: gold + BM25 top-100 over the dev file, shuffled
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
