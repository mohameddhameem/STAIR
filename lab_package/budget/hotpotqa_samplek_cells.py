#!/usr/bin/env python3
"""
hotpotqa_samplek_cells.py -- K > 1 FOR THE WEAK ANSWERER (HotpotQA). v1.1 18 Nov 2026 (--runner lab: same script on the lab GPU via run_labels4.py; v1.0 ran the K = 40 / T = 0.7 cells on the cluster, jobs 368176/368177).
GPU. Imports the Config B grid runner (~/slr/run_labels_agenttts.py) unchanged: same pools, same QA
prompt, same "[n] passage" block, same models; the ONLY difference from the stored WW / SW cells is that the weak answerer
(Llama-3.2-3B) is SAMPLED K times instead of decoded once.

RULE (Jonathan, 17 Nov 21:45): the strong model is called at most once per stage; the weak model may be called k times,
k in {1, 5, 10, 20, 40}, answers combined by majority vote (no fuser). This script is the MEASUREMENT step: it stores the K
samples per question so the allocation step can price and compare "weak x k" against "strong x 1" inside the fixed budget
(= the always-strong cost of the test set). Nothing here is a spend decision.

CELLS  for every question of the validation split (slr-val, 1,110) and the held-out test split (slr-test, 1,481):
  WWxK  weak-retrieval context (stored weak retriever picks)   -> weak answerer sampled K times
  SWxK  strong-retrieval context (stored strong retriever picks) -> weak answerer sampled K times
One batched generate per question per context (num_return_sequences = K; do_sample, --temp 0.7 --top-p 0.9 -- OUR settings, no
AgentTTS number to match here; seed = --seed + position). Every sample is stored; the scorer forms vote@k for k <= K.

PRICES (Table 5 per-call convention, as every HotpotQA number): weak answer call = 3 units -> WWxk = 77 + 3k, SWxk = 811 + 3k
(WW = 80, WS = 195, SW = 894, SS = 1,009). Measured tokens are written to the meta sidecar for the alternative accounting.

INPUTS   --data   ~/slr/data/tts_test.json   (the AgentTTS HotpotQA dev file the grid used; records carry the HotpotQA id)
         --stores ~/slr/hotpotqa_stores      (retr_weak.jsonl, retr_strong.jsonl keyed "<position>"; the Config B grid stores)
         --ids    slr-val.txt / slr-test.txt (HotpotQA ids; position = index in --data whose id matches)
OUTPUTS  <out>/samples_<split>.jsonl  one row per (position, ctx): {"i", "id", "split", "ctx": "W"|"S", "samples": [K strings]}
         <out>/meta_<split>.jsonl     {"i", "ctx", "n_in", "n_out_total", "n_seq_out", "secs", "truncated"}
         <out>/manifest.json          shas of script, runner, data, stores, id files; settings; counts
Resume-safe (append-only; rerun the same command). --end 10 for the smoke slice (fresh --out).

USAGE  SLR_POOL_K=110 SLR_SHUFFLE=off python3 hotpotqa_samplek_cells.py --split test --out <dir> --k 40 [--end 10]
"""
import argparse, datetime, hashlib, json, os, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
# v1.1 (18 Nov): --runner lab imports the lab's copy of the same runner (run_labels4.py, identical API) with the lab paths as defaults
_RUNNER = "lab" if "--runner" in sys.argv and sys.argv[sys.argv.index("--runner") + 1] == "lab" else "cluster"
if _RUNNER == "lab":
    sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")
    import torch                                                  # noqa: E402
    import run_labels4 as L                                       # noqa: E402
    DEF_DATA, DEF_STORES, DEF_IDS = "/mnt/hdd/jonathan/data/tts_test.json", "/mnt/hdd/jonathan/runs/config_b", "/mnt/hdd/jonathan/STAIR/common/stair_common/splits/hotpotqa"
else:
    sys.path.insert(0, os.path.expanduser("~/slr"))
    import torch                                                  # noqa: E402
    import run_labels_agenttts as L                               # noqa: E402
    DEF_DATA, DEF_STORES, DEF_IDS = os.path.expanduser("~/slr/data/tts_test.json"), os.path.expanduser("~/slr/hotpotqa_stores"), os.path.expanduser("~/slr/STAIR/common/stair_common/splits/hotpotqa")

SYS_QA = "You are a helpful assistant."                           # the runner's QA system message
ID_KEYS = ("_id", "id", "qid", "question_id")


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_store(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def rec_id(rec):
    for k in ID_KEYS:
        if k in rec: return str(rec[k])
    raise KeyError(f"no id field among {ID_KEYS} in record keys {list(rec)[:12]}")


def generate_samples(model, tok, user, max_new, k, temp, top_p, seed):
    msgs = [{"role": "system", "content": SYS_QA}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=L.MAXLEN).to(model.device); n_in = int(ids.input_ids.shape[1])
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=top_p,
                             num_return_sequences=k, pad_token_id=tok.eos_token_id)
    texts = [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]
    n_out = int(sum(int((o[n_in:] != tok.eos_token_id).sum()) for o in out))
    return texts, {"n_in": n_in, "n_out_total": n_out, "n_seq_out": k, "secs": round(time.time() - t0, 3), "truncated": bool(n_in >= L.MAXLEN)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--runner", default="cluster", choices=["cluster", "lab"])
    ap.add_argument("--data", default=DEF_DATA); ap.add_argument("--stores", default=DEF_STORES); ap.add_argument("--ids-dir", default=DEF_IDS)
    ap.add_argument("--out", required=True); ap.add_argument("--k", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20261117); ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    assert os.environ.get("SLR_POOL_K") == "110" and os.environ.get("SLR_SHUFFLE") == "off", "run with SLR_POOL_K=110 SLR_SHUFFLE=off"
    assert L.POOL_K == 110 and L.SHUFFLE_SCHEME == "as-given/no-shuffle", (L.POOL_K, L.SHUFFLE_SCHEME)
    os.makedirs(a.out, exist_ok=True)

    data = json.load(open(a.data, encoding="utf-8"))
    ids_file = os.path.join(a.ids_dir, f"slr-{a.split}.txt"); want = [l.strip() for l in open(ids_file) if l.strip()]
    pos_of = {rec_id(r): i for i, r in enumerate(data)}
    missing = [q for q in want if q not in pos_of]; assert not missing, f"{len(missing)} ids of {ids_file} not in {a.data}: {missing[:3]}"
    positions = [pos_of[q] for q in want]
    positions = positions[a.start:(len(positions) if a.end is None else a.end)]
    rw, rs = load_store(f"{a.stores}/retr_weak.jsonl"), load_store(f"{a.stores}/retr_strong.jsonl")
    for i in positions[:3]: assert str(i) in rw and str(i) in rs, f"position {i} missing in the retrieval stores"

    manifest = {"version": "hotpotqa_samplek_cells v1.1 2026-11-18 (--runner lab)", "runner_kind": _RUNNER, "split": a.split, "n": len(positions), "range": [a.start, a.start + len(positions)],
                "k": a.k, "temp": a.temp, "top_p": a.top_p, "seed": a.seed, "dtype": a.dtype, "qa_out": L.QA_OUT, "maxlen": L.MAXLEN,
                "qa_weak_model": L.QA_WEAK, "sys_qa": SYS_QA, "runner": L.__file__, "started": datetime.datetime.now().isoformat(timespec="seconds"),
                "sha256": {"script": sha(os.path.abspath(__file__)), "runner": sha(L.__file__), "data": sha(a.data), "ids": sha(ids_file),
                           "retr_weak": sha(f"{a.stores}/retr_weak.jsonl"), "retr_strong": sha(f"{a.stores}/retr_strong.jsonl")}}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print(f"{a.split}: {len(positions)} questions, K={a.k}, temp {a.temp} top_p {a.top_p}", flush=True)

    store = L.Store(os.path.join(a.out, f"samples_{a.split}.jsonl")); meta = L.MetaLog(os.path.join(a.out, f"meta_{a.split}.jsonl"))
    m, tok = L.load_model("weak", a.dtype, stage="qa"); print(f"[weak] {L.QA_WEAK} loaded", flush=True)
    t0 = time.time(); done = 0
    for i in positions:
        inst = dict(data[i]); inst["ctxs_candidate"] = L.trim_pool(inst); npool = len(inst["ctxs_candidate"])
        assert npool == len(data[i]["ctxs_candidate"]), f"position {i}: pool trimmed"
        for ctx, st in (("W", rw), ("S", rs)):
            key = f"{i}|{ctx}"
            if key in store: continue
            sel = L.parse_selection(st[str(i)], npool)
            texts, mt = generate_samples(m, tok, L.qa_prompt(inst, L.build_block(inst, sel)), L.QA_OUT, a.k, a.temp, a.top_p, a.seed + i)
            store.put(key, json.dumps({"i": i, "id": rec_id(data[i]), "split": a.split, "ctx": ctx, "samples": texts}))
            meta.put(key, mt)
        done += 1
        if done % 50 == 0:
            store.flush(); meta.flush(); r = (time.time() - t0) / done
            print(f"  {done}/{len(positions)}  {r:.1f}s/question  ETA {r * (len(positions) - done) / 60:.0f} min", flush=True)
    store.flush(); meta.flush()
    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds"); json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("SAMPLEK DONE ->", a.out, "(score with hotpotqa_samplek_score.py)")


if __name__ == "__main__":
    main()
