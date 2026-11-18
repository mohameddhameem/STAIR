#!/usr/bin/env python3
"""
hotpotqa_retrk_cells.py -- k WEAK RETRIEVALS (HotpotQA): the outcomes the k-router needs at the RETRIEVAL stage. DRAFT v1.0 18 Nov 2026
for Jonathan's line-by-line review (§18 of the budget registration). Cluster (two H200s, Config B runner ~/slr/run_labels_agenttts.py imported
unchanged -- same pools, same retrieval prompt, same QA prompt, same models). Delta on hotpotqa_samplek_cells.py (reviewed 17 Nov): new parts
are marked # RETRK.

WHAT IS MEASURED (Jonathan, 18 Nov 15:10): the weak retriever (Qwen2.5-7B) is SAMPLED k_ret times per question (T 0.7 / top-p 0.9) instead of
decoded once; the k pick-sets are combined by UNION (any passage picked in any sample) and by VOTE (picked in >= ceil(k/2) samples) -- both,
because the combination rule is to be a router action, not fixed; k_ret in {5, 10} (k_ret = 1 = the stored greedy cell; 20/40 are over the
per-instance budget). On EACH of the four combined contexts (u5, v5, u10, v10) the answer stage is measured BOTH ways, because the router
decides it: one greedy strong answer (Llama-3.1-70B -> the WS(k_ret) cells) and k_ans = 40 samples of the weak answerer (Llama-3.2-3B, T 0.7 /
top-p 0.9 -> the WW(k_ret, k_ans) cells, vote@k formed by the scorer for k_ans <= 40). Nothing here is a routing decision.

PHASES (resume-safe, --phase retr | qa_strong | qa_weak | all; run per split):
  retr       weak retriever x k_ret samples per question -> <out>/retrk_<split>.jsonl  {"i","id","sels":[[...]*k_ret],"texts":[...]}
  qa_strong  strong answerer, greedy, on u5/v5/u10/v10     -> <out>/qa_strong_retrk_<split>.jsonl  key "<i>|<ctx>" -> answer text
  qa_weak    weak answerer x k_ans samples on u5/v5/u10/v10 -> <out>/samples_retrk_<split>.jsonl key "<i>|<ctx>" -> {"samples":[...]}
  meta sidecars (n_in, n_out, secs, truncated) for every call -> measured-token pricing (AgentTTS units) alongside the per-call price.
PRICES (per-call, weak-first; the scorer applies them): WS(k_ret) = 77 k_ret + 3 + 115 (503 / 888); WW(k_ret, k_ans) = 77 k_ret + 3 k_ans.
INPUTS as the answer-stage job: --data ~/slr/data/tts_test.json, --ids slr-val.txt / slr-test.txt; the stored greedy retrievals are NOT needed
(k_ret = 1 comes from the registered cells at scoring time).
USAGE  SLR_POOL_K=110 SLR_SHUFFLE=off python3 hotpotqa_retrk_cells.py --split val --out <dir> --phase all [--end 10]
"""
import argparse, datetime, hashlib, json, math, os, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/slr"))
import torch                                                      # noqa: E402
import run_labels_agenttts as L                                   # noqa: E402
DEF_DATA, DEF_IDS = os.path.expanduser("~/slr/data/tts_test.json"), os.path.expanduser("~/slr/STAIR/common/stair_common/splits/hotpotqa")

SYS_QA = "You are a helpful assistant."                           # the runner's QA system message (as the answer-stage script)
ID_KEYS = ("_id", "id", "qid", "question_id")
KRETS = (5, 10)                                                   # RETRK: the two k_ret levels combined from the first k samples
CTXS = [f"{r}{k}" for k in KRETS for r in ("u", "v")]             # RETRK: u5 v5 u10 v10


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def rec_id(rec):
    for k in ID_KEYS:
        if k in rec: return str(rec[k])
    raise KeyError(f"no id field among {ID_KEYS} in record keys {list(rec)[:12]}")


def generate_samples(model, tok, system, user, max_new, k, temp, top_p, seed):
    """k sampled continuations of one prompt (the answer-stage function, now with the system message as a parameter)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
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


def combine(sels, k, rule):                                       # RETRK: union / vote over the FIRST k pick-sets
    first = sels[:k]
    if rule == "u": return sorted(set(n for s in first for n in s))
    need = math.ceil(k / 2); cnt = {}
    for s in first:
        for n in set(s): cnt[n] = cnt.get(n, 0) + 1
    return sorted(n for n, c in cnt.items() if c >= need)


def contexts_of(row, inst):                                       # RETRK: the four combined "[n] passage" blocks for one question
    return {f"{r}{k}": L.build_block(inst, combine(row["sels"], k, r)) for k in KRETS for r in ("u", "v")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["val", "test"])
    ap.add_argument("--phase", default="all", choices=["retr", "qa_strong", "qa_weak", "all"])
    ap.add_argument("--data", default=DEF_DATA); ap.add_argument("--ids-dir", default=DEF_IDS)
    ap.add_argument("--out", required=True)
    ap.add_argument("--kret", type=int, default=10); ap.add_argument("--kans", type=int, default=40)
    ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=20261118); ap.add_argument("--dtype", default="fp16")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    assert os.environ.get("SLR_POOL_K") == "110" and os.environ.get("SLR_SHUFFLE") == "off", "run with SLR_POOL_K=110 SLR_SHUFFLE=off"
    assert L.POOL_K == 110 and L.SHUFFLE_SCHEME == "as-given/no-shuffle", (L.POOL_K, L.SHUFFLE_SCHEME)
    assert a.kret >= max(KRETS), f"--kret must cover {KRETS}"
    # model guards (Config B; the cluster runner's defaults -- asserted, never assumed)
    assert "qwen2.5-7b" in L.RETR_WEAK.lower(), L.RETR_WEAK
    assert "llama-3.2-3b" in L.QA_WEAK.lower(), L.QA_WEAK
    assert "llama-3.1-70b" in L.QA_STRONG.lower(), L.QA_STRONG
    os.makedirs(a.out, exist_ok=True)

    data = json.load(open(a.data, encoding="utf-8"))
    ids_file = os.path.join(a.ids_dir, f"slr-{a.split}.txt"); want = [l.strip() for l in open(ids_file) if l.strip()]
    pos_of = {rec_id(r): i for i, r in enumerate(data)}
    missing = [q for q in want if q not in pos_of]; assert not missing, f"{len(missing)} ids of {ids_file} not in {a.data}: {missing[:3]}"
    positions = [pos_of[q] for q in want][a.start:(None if a.end is None else a.end)]

    man_path = os.path.join(a.out, f"manifest_{a.split}.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {}
    manifest.update({"version": "hotpotqa_retrk_cells v1.0 2026-11-18", "split": a.split, "n": len(positions), "range": [a.start, a.start + len(positions)],
                     "k_ret": a.kret, "k_ret_levels": list(KRETS), "rules": ["union", "vote>=ceil(k/2)"], "k_ans": a.kans, "temp": a.temp, "top_p": a.top_p,
                     "seed": a.seed, "dtype": a.dtype, "retr_out": L.RETR_OUT, "qa_out": L.QA_OUT, "maxlen": L.MAXLEN,
                     "models": {"retr_weak": L.RETR_WEAK, "qa_weak": L.QA_WEAK, "qa_strong": L.QA_STRONG}, "runner": L.__file__,
                     "sha256": {"script": sha(os.path.abspath(__file__)), "runner": sha(L.__file__), "data": sha(a.data), "ids": sha(ids_file)}})
    manifest.setdefault("phases", {})
    print(f"{a.split}: {len(positions)} questions, phase {a.phase}, k_ret {a.kret} (levels {KRETS}), k_ans {a.kans}, T {a.temp} top_p {a.top_p}", flush=True)
    phases = ["retr", "qa_strong", "qa_weak"] if a.phase == "all" else [a.phase]
    retr_store = L.Store(os.path.join(a.out, f"retrk_{a.split}.jsonl"))

    def progress(done, t0, n, tag):
        if done % 50 == 0:
            r = (time.time() - t0) / done; print(f"  {tag} {done}/{n}  {r:.1f}s/question  ETA {r * (n - done) / 60:.0f} min", flush=True)

    for ph in phases:
        manifest["phases"][ph] = {"started": datetime.datetime.now().isoformat(timespec="seconds")}; json.dump(manifest, open(man_path, "w"), indent=1)
        t0 = time.time(); done = 0
        if ph == "retr":                                          # RETRK: k_ret sampled retrievals per question
            meta = L.MetaLog(os.path.join(a.out, f"meta_retrk_{a.split}.jsonl"))
            m, tok = L.load_model("weak", a.dtype, stage="retr"); print(f"[retr weak] {L.RETR_WEAK} loaded", flush=True)
            for i in positions:
                if str(i) in retr_store: continue
                inst = dict(data[i]); inst["ctxs_candidate"] = L.trim_pool(inst); npool = len(inst["ctxs_candidate"])
                assert npool == len(data[i]["ctxs_candidate"]), f"position {i}: pool trimmed"
                texts, mt = generate_samples(m, tok, L.SYS_RETR, L.retr_prompt(inst), L.RETR_OUT, a.kret, a.temp, a.top_p, a.seed + i)
                sels = [L.parse_selection(t, npool) for t in texts]
                retr_store.put(str(i), json.dumps({"i": i, "id": rec_id(data[i]), "split": a.split, "sels": sels, "texts": texts})); meta.put(str(i), mt)
                done += 1; progress(done, t0, len(positions), "retr")
            retr_store.flush(); meta.flush(); del m, tok; L.free()
        else:                                                     # RETRK: answer stage on the four combined contexts
            which = "strong" if ph == "qa_strong" else "weak"
            store = L.Store(os.path.join(a.out, f"{'qa_strong_retrk' if which == 'strong' else 'samples_retrk'}_{a.split}.jsonl"))
            meta = L.MetaLog(os.path.join(a.out, f"meta_{ph}_retrk_{a.split}.jsonl"))
            m, tok = L.load_model(which, a.dtype, stage="qa"); print(f"[qa {which}] {L.QA_STRONG if which == 'strong' else L.QA_WEAK} loaded", flush=True)
            for i in positions:
                assert str(i) in retr_store, f"position {i}: run --phase retr first"
                row = json.loads(retr_store[str(i)])          # Store.__getitem__ (runner)
                inst = dict(data[i]); inst["ctxs_candidate"] = L.trim_pool(inst)
                for ctx, block in contexts_of(row, inst).items():
                    key = f"{i}|{ctx}"
                    if key in store: continue
                    user = L.qa_prompt(inst, block)
                    if which == "strong":
                        txt, mt = L.generate(m, tok, SYS_QA, user, L.QA_OUT)          # greedy, exactly the runner's qa call
                        store.put(key, txt)
                    else:
                        texts, mt = generate_samples(m, tok, SYS_QA, user, L.QA_OUT, a.kans, a.temp, a.top_p, a.seed + i)
                        store.put(key, json.dumps({"i": i, "ctx": ctx, "samples": texts}))
                    meta.put(key, mt)
                done += 1; progress(done, t0, len(positions), ph)
            store.flush(); meta.flush(); del m, tok; L.free()
        manifest["phases"][ph]["finished"] = datetime.datetime.now().isoformat(timespec="seconds"); json.dump(manifest, open(man_path, "w"), indent=1)
        print(f"PHASE {ph} DONE ({(time.time() - t0) / 60:.0f} min)", flush=True)
    print("RETRK DONE ->", a.out)


if __name__ == "__main__":
    main()
