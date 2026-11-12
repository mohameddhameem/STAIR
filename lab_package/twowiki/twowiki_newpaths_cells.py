#!/usr/bin/env python3
"""
twowiki_newpaths_cells.py -- §15 NEW PATHS on 2Wiki val400 (cluster, GPU). DRAFT v1.0 12 Nov 2026 for Jonathan's line-by-line
review; nothing launched. Same models, prompts, pools and stores as the Config B grid runner (~/slr/run_labels_agenttts.py,
imported, unchanged); only the CONTEXT handed to the answerer, or the number of answers sampled, differs per cell.

WHY (12 Nov analyses, val400): SS is wrong on 34 % of instances; on 22 % no single-call cell is right (ceiling 78 % for any
router over WW/WS/SW/SS). The reachable remainder splits by cause -- coverage failures on compositional/inference (only a
retrieval-side change helps: UNION) and answerer failures on bridge-comparison where several weak paths succeed (only an
answer-side change helps: SAMPLING, which is AgentTTS's 2Wiki path). This script measures both as fixed paths, no router.

CELLS (first letter(s) = retrieval context, then the answerer; W = Llama-3.2-3B, S = Llama-3.1-70B, as Config B)
  §15-A  U-S   union of the weak AND strong retrievers' stored selections -> strong answerer   (escalation target question)
         U-W   the same union -> weak answerer
  §15-B  SWxk  strong retrieval -> weak answerer SAMPLED k times (do_sample, --temp/--top-p, seed = --seed + idx); the samples
               are stored one per key so the SCORER can form majority-vote@k for every k <= K from one run; optional 3B FUSER
               reads the first k samples (k in --fuse-at) and writes one answer per k
  §15-C  S-all / W-all  ALL pool passages (no retrieval) -> strong / weak answerer      (retrieval-free control; SLR_MAXLEN raised)
NOT changed: retrieval stores (read, not re-run), QA prompt text (L.qa_prompt), reference block format (L.build_block: "[n] passage"
in ascending pool position), greedy decoding for every non-sampled cell, L.MAXLEN truncation (except --maxlen for the -all cells).

STORES (<out>/)   keys are "<idx>|<cell>" exactly like the runner's "<idx>|<retr>"
  qa_newpaths_weak.jsonl    "<i>|U"  "<i>|ALL"  "<i>|SWx<K>|s<j>" (j = 0..K-1)  "<i>|SWx<k>|fuse"      + meta_qa_newpaths_weak.jsonl
  qa_newpaths_strong.jsonl  "<i>|U"  "<i>|ALL"                                                            + meta_qa_newpaths_strong.jsonl
  manifest.json             runner sha, data sha, cells, K, temp, top_p, seed, maxlen, fuser prompt, counts
Resume-safe (append-only stores; rerun the same command). --end 10 for the smoke slice (fresh --out).

USAGE (cluster, slr-env; sbatch twowiki_newpaths.sbatch sets the env)
  SLR_POOL_K=110 SLR_SHUFFLE=off python twowiki_newpaths_cells.py \
      --run /common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/labels6000_merged/val400 --out <fresh dir> \
      --cells U-S U-W SWxk S-all W-all --k 50 --temp 0.7 --top-p 0.9 --seed 20261112 --fuse-at 5 10 20 50 [--end 10]
"""
import argparse, datetime, hashlib, json, os, sys, time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
sys.path.insert(0, os.path.expanduser("~/slr"))                  # the Config B grid runner, unchanged
import torch                                                      # noqa: E402
import run_labels_agenttts as L                                   # noqa: E402

SYS_QA = "You are a helpful assistant."                           # the runner's QA system message (run_labels_agenttts qa phase)
FUSER_PROMPT = ("Several independent attempts were made to answer a question from the same references. "
                "Pick the single best final answer. Answer with ONLY the exact answer phrase - no explanation, no full sentence.\n"
                "Query: {question}\nCandidate answers:\n{answers}")   # OURS (not AgentTTS's text; see registration §15-B, item to confirm)


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_store(path):
    d = {}
    for line in open(path, encoding="utf-8"):
        if line.strip():
            r = json.loads(line); d[r["k"]] = r["v"] if "v" in r else r
    return d


def chat_ids(tok, model, system, user, maxlen):
    """Identical tokenisation to L.generate (chat template, truncation at maxlen)."""
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt", truncation=True, max_length=maxlen).to(model.device)
    return ids, int(ids.input_ids.shape[1])


def generate_greedy(model, tok, user, max_new, maxlen):
    """= L.generate but with the prompt cap as a parameter (the -all cells need more than L.MAXLEN)."""
    ids, n_in = chat_ids(tok, model, SYS_QA, user, maxlen)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.eos_token_id)
    meta = {"n_in": n_in, "n_out": int(out.shape[1]) - n_in, "secs": round(time.time() - t0, 3), "truncated": bool(n_in >= maxlen)}
    return tok.decode(out[0][n_in:], skip_special_tokens=True), meta


def generate_samples(model, tok, user, max_new, k, temp, top_p, seed):
    """k sampled answers in ONE call (num_return_sequences); one prompt forward, k decodes. Seeded per instance."""
    ids, n_in = chat_ids(tok, model, SYS_QA, user, L.MAXLEN)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    t0 = time.time()
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=True, temperature=temp, top_p=top_p,
                             num_return_sequences=k, pad_token_id=tok.eos_token_id)
    texts = [tok.decode(o[n_in:], skip_special_tokens=True) for o in out]
    n_out = [int((o[n_in:] != tok.eos_token_id).sum()) for o in out]
    meta = {"n_in": n_in, "n_out_total": sum(n_out), "n_seq_out": k, "secs": round(time.time() - t0, 3), "truncated": bool(n_in >= L.MAXLEN)}
    return texts, meta


def union_block(inst, sel_w, sel_s):
    return L.build_block(inst, sorted(set(sel_w) | set(sel_s)))      # ascending pool position, the runner's own block format


def all_block(inst):
    return L.build_block(inst, list(range(1, len(inst["ctxs_candidate"]) + 1)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="merged split folder: tts_test_2wiki_adapted.json + retr_weak/retr_strong.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["U-S", "U-W", "SWxk", "S-all", "W-all"],
                    choices=["U-S", "U-W", "SWxk", "S-all", "W-all"])
    ap.add_argument("--k", type=int, default=50); ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9); ap.add_argument("--seed", type=int, default=20261112)
    ap.add_argument("--fuse-at", nargs="*", type=int, default=[5, 10, 20, 50], help="k values the 3B fuser reads; [] = no fuser")
    ap.add_argument("--maxlen", type=int, default=8192, help="prompt cap for the -all cells only (pools of ~100 sentences)")
    ap.add_argument("--dtype", default="fp16", choices=["4bit", "fp16"], help="strong answerer dtype (Config B ran fp16 on 2 H200s)")
    ap.add_argument("--start", type=int, default=0); ap.add_argument("--end", type=int, default=None)
    a = ap.parse_args()
    assert os.environ.get("SLR_POOL_K") == "110" and os.environ.get("SLR_SHUFFLE") == "off", "run with SLR_POOL_K=110 SLR_SHUFFLE=off"
    assert L.POOL_K == 110 and L.SHUFFLE_SCHEME == "as-given/no-shuffle", (L.POOL_K, L.SHUFFLE_SCHEME)
    assert all(k <= a.k for k in a.fuse_at), "--fuse-at values must be <= --k"
    os.makedirs(a.out, exist_ok=True)

    data_path = os.path.join(a.run, "tts_test_2wiki_adapted.json")
    data = json.load(open(data_path, encoding="utf-8"))
    rw, rs = load_store(os.path.join(a.run, "retr_weak.jsonl")), load_store(os.path.join(a.run, "retr_strong.jsonl"))
    idxs = list(range(a.start, len(data) if a.end is None else min(a.end, len(data))))
    insts = {}
    for i in idxs:
        inst = dict(data[i]); inst["ctxs_candidate"] = L.trim_pool(inst)      # the runner's pool (110, no shuffle = untrimmed here)
        assert len(inst["ctxs_candidate"]) == len(data[i]["ctxs_candidate"]), f"idx {i}: pool was trimmed; POOL_K too small"
        insts[i] = inst
    n = len(insts); print(f"instances {n} [{idxs[0]}..{idxs[-1]}]  cells {a.cells}  K {a.k}  temp {a.temp} top_p {a.top_p} seed {a.seed}", flush=True)

    def blocks_for(i, cell):
        inst = insts[i]; npool = len(inst["ctxs_candidate"])
        sel_w, sel_s = L.parse_selection(rw[str(i)], npool), L.parse_selection(rs[str(i)], npool)
        if cell in ("U-S", "U-W"):   return union_block(inst, sel_w, sel_s)
        if cell == "SWxk":           return L.build_block(inst, sel_s)
        if cell in ("S-all", "W-all"): return all_block(inst)
        raise ValueError(cell)

    manifest = {"version": "twowiki_newpaths_cells v1.0 2026-11-12", "runner": L.__file__, "runner_sha256": sha(L.__file__),
                "data": data_path, "data_sha256": sha(data_path), "retr_weak_sha256": sha(os.path.join(a.run, "retr_weak.jsonl")),
                "retr_strong_sha256": sha(os.path.join(a.run, "retr_strong.jsonl")), "cells": a.cells, "k": a.k, "temp": a.temp,
                "top_p": a.top_p, "seed": a.seed, "fuse_at": a.fuse_at, "maxlen_all": a.maxlen, "maxlen_default": L.MAXLEN,
                "qa_out": L.QA_OUT, "dtype_strong": a.dtype, "models": {"qa_weak": L.QA_WEAK, "qa_strong": L.QA_STRONG},
                "sys_qa": SYS_QA, "fuser_prompt": FUSER_PROMPT, "range": [idxs[0], idxs[-1] + 1], "n": n,
                "started": datetime.datetime.now().isoformat(timespec="seconds"), "torch": torch.__version__}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)

    # ---------- weak answerer (3B): U-W, W-all, SWxk samples, fuser ----------
    weak_cells = [c for c in a.cells if c in ("U-W", "W-all", "SWxk")]
    if weak_cells:
        store = L.Store(os.path.join(a.out, "qa_newpaths_weak.jsonl")); meta = L.MetaLog(os.path.join(a.out, "meta_qa_newpaths_weak.jsonl"))
        m, tok = L.load_model("weak", a.dtype, stage="qa"); print(f"[weak] {L.QA_WEAK} loaded", flush=True)
        t0 = time.time(); done = 0
        for i in idxs:
            inst = insts[i]
            if "U-W" in weak_cells and f"{i}|U" not in store:
                txt, mt = generate_greedy(m, tok, L.qa_prompt(inst, blocks_for(i, "U-W")), L.QA_OUT, L.MAXLEN)
                store.put(f"{i}|U", txt); meta.put(f"{i}|U", mt)
            if "W-all" in weak_cells and f"{i}|ALL" not in store:
                txt, mt = generate_greedy(m, tok, L.qa_prompt(inst, blocks_for(i, "W-all")), L.QA_OUT, a.maxlen)
                store.put(f"{i}|ALL", txt); meta.put(f"{i}|ALL", mt)
            if "SWxk" in weak_cells:
                tag = f"SWx{a.k}"
                if f"{i}|{tag}|s0" not in store:
                    texts, mt = generate_samples(m, tok, L.qa_prompt(inst, blocks_for(i, "SWxk")), L.QA_OUT, a.k, a.temp, a.top_p, a.seed + i)
                    for j, t in enumerate(texts):
                        store.put(f"{i}|{tag}|s{j}", t)
                    meta.put(f"{i}|{tag}", mt)
                for k in a.fuse_at:
                    key = f"{i}|SWx{k}|fuse"
                    if key in store:
                        continue
                    cands = "\n".join(f"- {store[f'{i}|{tag}|s{j}'].strip()}" for j in range(k))
                    txt, mt = generate_greedy(m, tok, FUSER_PROMPT.format(question=inst["question"], answers=cands), L.QA_OUT, L.MAXLEN)
                    store.put(key, txt); meta.put(key, mt)
            done += 1
            if done % 25 == 0:
                store.flush(); meta.flush(); r = (time.time() - t0) / done
                print(f"  weak {done}/{n}  {r:.1f}s/inst  ETA {r * (n - done) / 60:.0f} min", flush=True)
        store.flush(); meta.flush(); del m, tok; L.free(); print(f"[weak] DONE {(time.time() - t0) / 60:.1f} min", flush=True)

    # ---------- strong answerer (70B): U-S, S-all ----------
    strong_cells = [c for c in a.cells if c in ("U-S", "S-all")]
    if strong_cells:
        store = L.Store(os.path.join(a.out, "qa_newpaths_strong.jsonl")); meta = L.MetaLog(os.path.join(a.out, "meta_qa_newpaths_strong.jsonl"))
        m, tok = L.load_model("strong", a.dtype, stage="qa"); print(f"[strong] {L.QA_STRONG} loaded ({a.dtype})", flush=True)
        t0 = time.time(); done = 0
        for i in idxs:
            inst = insts[i]
            if "U-S" in strong_cells and f"{i}|U" not in store:
                txt, mt = generate_greedy(m, tok, L.qa_prompt(inst, blocks_for(i, "U-S")), L.QA_OUT, L.MAXLEN)
                store.put(f"{i}|U", txt); meta.put(f"{i}|U", mt)
            if "S-all" in strong_cells and f"{i}|ALL" not in store:
                txt, mt = generate_greedy(m, tok, L.qa_prompt(inst, blocks_for(i, "S-all")), L.QA_OUT, a.maxlen)
                store.put(f"{i}|ALL", txt); meta.put(f"{i}|ALL", mt)
            done += 1
            if done % 25 == 0:
                store.flush(); meta.flush(); r = (time.time() - t0) / done
                print(f"  strong {done}/{n}  {r:.1f}s/inst  ETA {r * (n - done) / 60:.0f} min", flush=True)
        store.flush(); meta.flush(); del m, tok; L.free(); print(f"[strong] DONE {(time.time() - t0) / 60:.1f} min", flush=True)

    manifest["finished"] = datetime.datetime.now().isoformat(timespec="seconds")
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    print("NEWPATHS DONE ->", a.out, "(score with twowiki_newpaths_score.py)")


if __name__ == "__main__":
    main()
