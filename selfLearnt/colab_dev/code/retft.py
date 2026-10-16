"""Set 7 - the retriever learns to shortlist: LoRA fine-tuning of Qwen 3B on HotpotQA's gold passage labels.

No router and no engineered features: the input is the question and its 20 candidate passages as text (the same construction
as every earlier set), the target is the text line 'Selected references: [a] [b]' naming the gold passages. The answerers are
unchanged (Qwen 32B, Llama 8B, one greedy chain-of-thought answer with the notebook 21 prompt).

Commands (each resumable):
  build_train  HotpotQA *training* questions -> the same 20-candidate prompts (gold + BM25 distractors, shuffled); train / validation split
  train        LoRA on Qwen 3B; checkpoints along the way are scored on the validation questions; the best one is merged
  retrieve     shortlists for the 400 test questions from: base 3B (CoT, as in the sweep), base 3B (direct prompt), fine-tuned 3B (direct)
  answer       Qwen 32B and Llama 8B answer on every new shortlist (answers already made by notebook 21 are reused)
  evaluate     retrieval quality, final accuracy and compute against the cheap fixed setup (Qwen 7B retrieves, Qwen 32B answers)
  merge        merge a trained adapter (the checkpoint chosen on validation) into its base model (notebook 26)
  sample       extra sampled answers on given shortlists, for tie-break votes (notebook 26)

Data roles: training and validation questions come from HotpotQA's *training* file; the 400 test questions come from its dev file
and are never used for training, tuning or checkpoint choice.
"""
import argparse
import json
import os
import random
import sys
import time
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S               # noqa: E402
import sweep as W             # noqa: E402
import retriever_check as RC  # noqa: E402
import experts as X           # noqa: E402

VERSION = "7.5"
log = S.log
TRAIN_URL = "http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json"
DIRECT = ("Select only the references needed to answer the query. Answer with one line in exactly this form, listing only the "
          "reference numbers you select:\nSelected references: [n] [n]")
VARIANTS = {   # name -> (model key, prompt, temperature) ; 'q3ft' is the merged fine-tuned model
    "base3_cot": ("q3", "cot", 0.9),      # the sweep's CoT retrieval prompt (select 3), one try - as the Qwen 7B reference was run
    "base3_direct": ("q3", "direct", 0.0),  # zero-shot with the training prompt: separates the prompt from the training
    "ft3_direct": ("q3ft", "direct", 0.0),  # the fine-tuned model
}
LABEL = {"base3_cot": "Qwen 3B, not trained (step-by-step prompt, as in the sweep)", "base3_direct": "Qwen 3B, not trained (direct prompt)",
         "ft3_direct": "Qwen 3B, fine-tuned (direct prompt)", "q7_ref": "Qwen 7B, not trained (step-by-step prompt) = the cheap fixed setup's retriever",
         "gold": "Gold passages only (needs the answer key; ceiling)"}


def direct_prompt(q):
    return ("You have been provided with a query and a set of references. "
            "Your task is to select references that comprehensively cover the information needed to answer the query. "
            f"Here is the query and references: \n {S.build_query(q)}.\n" + DIRECT)


def target_text(q):
    return "Selected references: " + " ".join(f"[{i + 1}]" for i in sorted(q["ret_answers"]))


# ==================================================================================
# 1. Training data
# ==================================================================================
def load_train_file(path):
    """HotpotQA training questions in the original JSON format; downloads it (official file, then the Hugging Face copy) when missing."""
    if path and os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    local = "/content/hotpot_train_v1.1.json" if os.path.isdir("/content") else os.path.join(os.path.expanduser("~"), "hotpot_train_v1.1.json")
    if not os.path.exists(local):
        try:
            import urllib.request
            log(f"downloading {TRAIN_URL} (about 570 MB) ...")
            urllib.request.urlretrieve(TRAIN_URL, local + ".part")
            os.rename(local + ".part", local)
        except Exception as e:  # noqa: BLE001
            log(f"official download failed ({type(e).__name__}: {e}); trying the Hugging Face copy")
            from datasets import load_dataset
            ds = None
            for repo in ("hotpotqa/hotpot_qa", "hotpot_qa"):
                try:
                    ds = load_dataset(repo, "distractor", split="train")
                    break
                except Exception as e2:  # noqa: BLE001
                    log(f"  {repo}: {type(e2).__name__}")
            assert ds is not None, "could not get the HotpotQA training file - download it by hand and pass --hotpot_train"
            out = [{"_id": r["id"], "question": r["question"], "answer": r["answer"], "type": r["type"], "level": r["level"],
                    "supporting_facts": [[t, s] for t, s in zip(r["supporting_facts"]["title"], r["supporting_facts"]["sent_id"])],
                    "context": [[t, s] for t, s in zip(r["context"]["title"], r["context"]["sentences"])]} for r in ds]
            json.dump(out, open(local, "w", encoding="utf-8"))
    return json.load(open(local, encoding="utf-8"))


def cmd_build_train(args):
    """Same construction as the test questions (ssw.cmd_build_data): gold paragraphs + BM25 distractors from a corpus of
    HotpotQA paragraphs, shuffled. The corpus is the paragraphs of a pool of training questions of the same size as the dev file."""
    data = load_train_file(args.hotpot_train)
    log(f"HotpotQA training file: {len(data)} questions")
    if args.level:
        data = [d for d in data if d.get("level") == args.level]
        log(f"  level '{args.level}' only (the dev questions are all of this level): {len(data)}")
    rng = random.Random(args.seed)
    pool = rng.sample(range(len(data)), min(args.pool, len(data)) if args.pool > 0 else len(data))
    seen, corpus = set(), []
    for i in pool:
        for _, sents in data[i]["context"]:
            t = " ".join(sents)
            if t not in seen:
                seen.add(t)
                corpus.append(t)
    log(f"pool of {len(pool)} questions, {len(corpus)} paragraphs; building BM25 ...")
    bm25 = S.BM25(corpus)
    picked = pool[:args.n_train + args.n_val]
    final_dir, args.out_dir = args.out_dir, args.out_dir.rstrip("/") + f".part{os.getpid()}"
    os.makedirs(args.out_dir, exist_ok=True)
    for split, idxs in (("train", picked[:args.n_train]), ("val", picked[args.n_train:])):
        out = []
        for i in idxs:
            ex = data[i]
            ctx = {t: " ".join(s) for t, s in ex["context"]}
            gold = []
            for title, _ in ex["supporting_facts"]:
                if title in ctx and ctx[title] not in gold:
                    gold.append(ctx[title])
            if len(gold) < 2:
                continue
            cands = gold.copy()
            for j in bm25.search(ex["question"], args.n_candidates + 10):
                if len(cands) >= args.n_candidates:
                    break
                if corpus[j] not in cands:
                    cands.append(corpus[j])
            rng.shuffle(cands)
            out.append({"qid": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type"), "level": ex.get("level"),
                        "ret_answers": [cands.index(g) for g in gold], "ctxs_candidate": cands})
        json.dump(out, open(os.path.join(args.out_dir, f"{split}.json"), "w", encoding="utf-8"), ensure_ascii=False)
        log(f"wrote {split}.json: {len(out)} questions")
    S.write_json(os.path.join(args.out_dir, "data_config.json"), {**vars(args), "out_dir": final_dir, "source": "HotpotQA training file (distractor setting)",
                                                                   "pool_questions": len(pool), "corpus_paragraphs": len(corpus), "version": VERSION})
    if os.path.exists(os.path.join(final_dir, "val.json")):        # another notebook built it meanwhile (same seed, same content)
        import shutil
        shutil.rmtree(args.out_dir, ignore_errors=True)
        log(f"{final_dir} already exists - kept the existing copy")
    else:
        os.rename(args.out_dir, final_dir)
        log(f"training questions ready: {final_dir}")


# ==================================================================================
# 2. LoRA training
# ==================================================================================
def chat_ids(tok, user, answer=None):
    msgs = [{"role": "system", "content": S.RET_SYSTEM}, {"role": "user", "content": user}]
    try:      # thinking off (Qwen3 / Qwen3.5 templates); other templates ignore the flag
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(p, add_special_tokens=False)["input_ids"]
    if answer is None:
        return pid, None
    eos = tok.eos_token or ""
    ans = tok(answer + eos, add_special_tokens=False)["input_ids"]
    return list(pid) + list(ans), len(pid)        # built from two pieces, so the prompt/target boundary is exact


def val_score(model, tok, qs, max_new=24):
    """Share of validation questions whose shortlist contains both gold passages, and whose shortlist is exactly the gold pair."""
    import torch
    model.eval()
    both = exact = 0
    sizes = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for q in qs:
            pid, _ = chat_ids(tok, direct_prompt(q))
            out = model.generate(torch.tensor([pid], device=dev), max_new_tokens=max_new, do_sample=False, use_cache=True,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            text = tok.decode(out[0, len(pid):], skip_special_tokens=True)
            sel = {i - 1 for i in W.picks_of(S.selection_text(text), len(q["ctxs_candidate"]))}
            g = set(q["ret_answers"])
            both += g <= sel
            exact += g == sel
            sizes.append(len(sel))
    model.train()
    return {"both gold passages found": both / len(qs), "exactly the gold passages": exact / len(qs), "mean shortlist size": float(np.mean(sizes))}


def load_lm(path):
    """A causal language model in bf16. Multimodal checkpoints (e.g. Qwen3.5) load through their image-text class;
    only the text path is used."""
    import torch
    import transformers as T
    errs = []
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq"):
        if not hasattr(T, cls):
            continue
        for kw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
            try:
                m = getattr(T, cls).from_pretrained(path, **kw)
                log(f"loaded {path} as {type(m).__name__} ({cls})")
                return m
            except TypeError:
                continue
            except (ValueError, KeyError, OSError) as e:
                errs.append(f"{cls}: {type(e).__name__}: {str(e)[:200]}")
                break
    raise RuntimeError("could not load " + path + "\n" + "\n".join(errs))


def lm_parts(cm):
    """(decoder that returns last_hidden_state, lm_head) for text-only and image-text models alike."""
    head = cm.get_output_embeddings() if hasattr(cm, "get_output_embeddings") else cm.lm_head
    body = cm.model
    if hasattr(body, "language_model"):
        body = body.language_model
    return body, head


def copy_side_files(src, dst):
    """Tokenizer, chat template and processor files that save_pretrained may not write (needed by vLLM for multimodal checkpoints)."""
    import shutil
    for f in os.listdir(src):
        if f.endswith((".safetensors", ".bin", ".pt", ".index.json")) or f.startswith(".") or os.path.isdir(os.path.join(src, f)):   # weight indexes name the base's shards
            continue
        if not os.path.exists(os.path.join(dst, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))


def _no_torchao():
    """Colab ships an old torchao (0.10) that peft refuses to work with, even though plain LoRA never needs it.
    Tell every peft module that torchao is absent, so peft uses its ordinary LoRA layers."""
    import peft  # noqa: F401
    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and hasattr(mod, "is_torchao_available"):
            setattr(mod, "is_torchao_available", lambda: False)


def cmd_train(args):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    _no_torchao()
    t0 = time.time()
    out = args.out_dir
    os.makedirs(out, exist_ok=True)
    tr = json.load(open(os.path.join(args.data_dir, "train.json"), encoding="utf-8"))[:args.n_train]
    va = json.load(open(os.path.join(args.data_dir, "val.json"), encoding="utf-8"))[:args.n_val]
    if args.backend == "tiny":
        import hidden as H
        texts = [direct_prompt(q) + " " + target_text(q) for q in tr[:200] + va] + [S.RET_SYSTEM]
        model, tok = H.tiny_model(texts)
        dev = "cpu"
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.base)
        model = load_lm(args.base)
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
    model.config.use_cache = False
    if args.backend != "tiny":
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    # examples, drop any that would not fit (the target is never cut)
    ex, dropped = [], 0
    for q in tr:
        ids, n_prompt = chat_ids(tok, direct_prompt(q), target_text(q))
        if len(ids) > args.max_len:
            dropped += 1
            continue
        ex.append((ids, n_prompt))
    log(f"training examples: {len(ex)} (dropped {dropped} longer than {args.max_len} tokens); mean length {np.mean([len(e[0]) for e in ex]):.0f} tokens")
    rng = random.Random(args.seed)
    rng.shuffle(ex)
    steps_total = (len(ex) * args.epochs) // args.grad_accum
    if args.ckpt_questions:
        marks = sorted({min(steps_total, max(1, q // args.grad_accum)) for q in args.ckpt_questions})
    else:
        marks = sorted({max(1, int(round(steps_total * f))) for f in args.checkpoints})
    save_every = max(1, args.save_every_questions // args.grad_accum)
    state_p = os.path.join(out, "train_state.json")
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"step": 0, "curve": [], "seconds": 0.0}
    last = os.path.join(out, f"ckpt_step{st['step']}") if st["step"] else None
    if last and os.path.isdir(last):
        model = PeftModel.from_pretrained(model, last, is_trainable=True)
        log(f"resuming from {last}")
    else:
        cfg = LoraConfig(r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, task_type="CAUSAL_LM",
                         target_modules=r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$")   # text layers only
        model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()            # the LoRA weights train in float32; the frozen base stays bf16
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"LoRA r={args.r}: {n_tr / 1e6:.1f}M trainable parameters; {steps_total} optimizer steps ({args.grad_accum} questions each, "
        f"micro-batches of {args.micro_batch}); validation checkpoints at steps {marks}; resume point every {save_every} steps")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    if st["step"] and os.path.exists(os.path.join(last, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(last, "optimizer.pt"), map_location=dev))
    warm = max(1, int(0.03 * steps_total))
    lr_at = lambda s: args.lr * min(1.0, (s + 1) / warm) * max(0.0, 1 - max(0, s - warm) / max(1, steps_total - warm))
    order = [i for _ in range(args.epochs) for i in range(len(ex))]
    cm = model.get_base_model()                 # the causal LM inside the LoRA wrapper (LoRA layers are inside it)
    body, head = lm_parts(cm)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ce = torch.nn.CrossEntropyLoss()
    model.train()
    step, t1, losses = st["step"], time.time(), []
    pos = step * args.grad_accum

    def batch_loss(items):
        """Loss on the target tokens only: the language-model head is applied at the few target positions, not the whole prompt."""
        L = max(len(i) for i, _ in items)
        x = torch.full((len(items), L), pad, dtype=torch.long)
        am = torch.zeros((len(items), L), dtype=torch.long)
        lab = torch.full((len(items), L), -100, dtype=torch.long)
        for b, (ids, n_prompt) in enumerate(items):
            x[b, :len(ids)] = torch.tensor(ids)
            am[b, :len(ids)] = 1
            lab[b, n_prompt:len(ids)] = torch.tensor(ids[n_prompt:])
        x, am, lab = x.to(dev), am.to(dev), lab.to(dev)
        h = body(input_ids=x, attention_mask=am).last_hidden_state
        sel = lab[:, 1:] != -100
        logits = head(h[:, :-1][sel]).float()
        return ce(logits, lab[:, 1:][sel])

    def save(step_, keep):
        d = os.path.join(out, f"ckpt_step{step_}")
        model.save_pretrained(d)
        torch.save(opt.state_dict(), os.path.join(d, "optimizer.pt"))
        old = st.get("resume_only")
        if old and old != step_ and old not in marks:
            import shutil
            shutil.rmtree(os.path.join(out, f"ckpt_step{old}"), ignore_errors=True)
        st["resume_only"] = None if keep else step_
        st["step"] = step_

    while step < steps_total:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = order[pos:pos + args.grad_accum]
        for c in range(0, len(idx), args.micro_batch):
            part = [ex[i] for i in idx[c:c + args.micro_batch]]
            loss = batch_loss(part) * len(part) / len(idx)
            loss.backward()
            losses.append(float(loss) * len(idx) / len(part))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        pos += args.grad_accum
        if step % 25 == 0:
            el = time.time() - t1
            log(f"  step {step}/{steps_total} | loss {np.mean(losses[-50:]):.3f} | {el / 60:.1f} min this session")
        if step in marks:
            save(step, keep=True)
            v = val_score(model, tok, va)
            row = {"step": step, "training questions seen": step * args.grad_accum, "training loss (recent)": float(np.mean(losses[-50:])), **v}
            st["curve"].append(row)
            log(f"checkpoint {step}: validation both-found {100 * v['both gold passages found']:.1f}%, exact {100 * v['exactly the gold passages']:.1f}%, "
                f"shortlist {v['mean shortlist size']:.2f}")
        elif step % save_every == 0:
            save(step, keep=False)
        if step in marks or step % save_every == 0:
            st["seconds"] += time.time() - t1
            t1 = time.time()
            json.dump(st, open(state_p, "w"), indent=1)
            if args.time_limit_min and (time.time() - t0) / 60 > args.time_limit_min and step < steps_total:
                log("time limit reached - run again to continue from this point")
                return
    # the checkpoint with the best validation score (ties -> the later one) is merged and kept
    if os.path.exists(os.path.join(args.merged_dir, "config.json")) and st.get("chosen"):
        log(f"already trained and merged ({args.merged_dir})")
        return
    best = max(st["curve"], key=lambda r: (r["both gold passages found"], r["exactly the gold passages"], r["step"]))
    log(f"chosen on the validation questions: step {best['step']} ({100 * best['both gold passages found']:.1f}% both found)")
    if args.backend == "tiny":
        st["chosen"] = best
        json.dump(st, open(state_p, "w"), indent=1)
        os.makedirs(args.merged_dir, exist_ok=True)
        json.dump({"tiny": True}, open(os.path.join(args.merged_dir, "config.json"), "w"))
        return
    del model, opt
    _free()
    base = load_lm(args.base)
    merged = PeftModel.from_pretrained(base, os.path.join(out, f"ckpt_step{best['step']}")).merge_and_unload()
    merged.save_pretrained(args.merged_dir)
    tok.save_pretrained(args.merged_dir)
    copy_side_files(args.base, args.merged_dir)
    st["chosen"] = best
    json.dump(st, open(state_p, "w"), indent=1)
    log(f"merged model saved to {args.merged_dir}; training took {st['seconds'] / 60:.0f} min in total")


# ==================================================================================
# 3. Retrieval on the test questions
# ==================================================================================
def test_sets(args):
    return [("old", args.old_sweep, args.old_data), ("fresh", args.fresh_run, args.fresh_data)]


def ret_file(run, name):
    return os.path.join(run, "step", f"retrieval_{name}.jsonl")


def cmd_retrieve(args):
    models = W.load_models(args.models_json)
    os.makedirs(os.path.join(args.run_dir, "step"), exist_ok=True)
    qs = [q for _, _, d in test_sets(args) for q in S.load_questions(d)]
    for name in args.variants:
        key, kind, temp = VARIANTS[name]
        path = ret_file(args.run_dir, name)
        done = {r["qid"] for r in S.read_jsonl(path)}
        todo = [q for q in qs if q["qid"] not in done]
        if not todo:
            log(f"{name}: complete")
            continue
        eng = W.make_engine(args.backend, models, key, max_model_len=8192, gpu_util=0.85)
        eng.thinking = False          # Qwen3 / Qwen3.5: no thinking block (the training prompts had none); Qwen2.5 ignores it
        P_ = models[key]["params"]
        max_tok = 1024 if kind == "cot" else 48
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            prompts = [eng.encode(S.RET_SYSTEM, W.ret_prompt(q, 1, "3") if kind == "cot" else direct_prompt(q)) for q in chunk]
            meta = [{"kind": "ret", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"]), "q": q["question"], "conf": kind == "cot"} for q in chunk]
            seeds = [zlib.crc32((name + q["qid"]).encode()) % 999983 for q in chunk]
            outs = eng.generate(prompts, 1, temp, max_tok, seeds, meta)
            rows = []
            for q, p, o in zip(chunk, prompts, outs):
                text, k = o[0][0], o[0][1]
                lp = o[0][2] if len(o[0]) > 2 else None
                sel = tuple(W.picks_of(S.selection_text(text) if kind == "cot" else S.selection_text(text), len(q["ctxs_candidate"]))) or (1,)
                sel = tuple(sorted(sel))
                g = {i + 1 for i in q["ret_answers"]}
                rows.append({"qid": q["qid"], "variant": name, "text": text, "sel": list(sel), "prompt_tokens": len(p), "out_tokens": k,
                             "flops": 2 * P_ * (len(p) + k), "found_both": float(g <= set(sel)), "n_passages": len(sel),
                             "precision": len(g & set(sel)) / len(sel), "hit_cap": k >= max_tok, "mean_logprob": lp})
            S.append_jsonl(path, rows)
            log(f"  {name}: {len(S.read_jsonl(path))}/{len(qs)}")
        del eng
        _free()


def _free():
    try:
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ==================================================================================
# 4. Answers on the new shortlists (notebook 21's answers reused where the shortlist is the same)
# ==================================================================================
def ans_file(run, model):
    return os.path.join(run, "step", f"answers_{model}.jsonl")


def shortlist_jobs(args):
    """(qid -> question, set of shortlists to answer): the three 3B variants, the Qwen 7B reference and the gold passages."""
    jobs = {}
    qmap = {}
    only = getattr(args, "only_extra", False)          # notebook 26: answer only the given shortlists
    for _, run, d in test_sets(args):
        rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RC.RET, RC.COT, RC.PICK)))}
        for q in S.load_questions(d):
            qmap[q["qid"]] = q
            jobs.setdefault(q["qid"], set())
            if not only:
                sl = W.shortlists(rets[q["qid"]], RC.COT, len(q["ctxs_candidate"]))
                jobs[q["qid"]].update({tuple(sl[(1, RC.MERGE)]), tuple(sorted(i + 1 for i in q["ret_answers"]))})
    for name in ([] if only else VARIANTS):
        for r in S.read_jsonl(ret_file(args.run_dir, name)):
            jobs[r["qid"]].add(tuple(r["sel"]))
    for f in getattr(args, "extra_shortlists", []) or []:        # e.g. cascade shortlists built from several retrievers
        for r in S.read_jsonl(f):
            jobs[r["qid"]].add(tuple(sorted(r["sel"])))
    return qmap, jobs


def cmd_answer(args):
    models = W.load_models(args.models_json)
    qmap, jobs = shortlist_jobs(args)
    for m in args.models:
        path = ans_file(args.run_dir, m)
        have = {(r["qid"], tuple(r["sel"])) for r in S.read_jsonl(path)}
        if args.cache_run or args.cache_answers:            # notebooks 21 and 22: same prompt, same settings
            cache = [r for r in S.read_jsonl(os.path.join(args.cache_run, "step", f"arms_{m}.jsonl")) if r["kind"] == "fresh"] if args.cache_run else []
            for d in args.cache_answers:
                cache += S.read_jsonl(os.path.join(d, "step", f"answers_{m}.jsonl"))
            add = []
            for r in cache:
                k = (r["qid"], tuple(r["sel"]))
                if args.require_logprob and r.get("mean_logprob") is None:
                    continue                                     # earlier answers without token probabilities are regenerated
                if r["qid"] in jobs and tuple(r["sel"]) in jobs[r["qid"]] and k not in have:
                    add.append({kk: r[kk] for kk in ("qid", "sel", "model", "prompt_tokens", "out_tokens", "flops", "text", "prediction", "EM", "F1",
                                                     "lenient", "stated_conf", "hit_cap")} | {"mean_logprob": r.get("mean_logprob"),
                                                                                              "from": r.get("from", "notebook 21") if "kind" not in r else "notebook 21"})
                    have.add(k)
            if add:
                S.append_jsonl(path, add)
                log(f"{m}: reused {len(add)} answers from earlier notebooks")
        todo = [(qid, sel) for qid, sels in jobs.items() for sel in sorted(sels) if (qid, sel) not in have]
        log(f"{m}: {len(todo)} answers to generate")
        if not todo:
            continue
        eng = W.make_engine(args.backend, models, m, max_model_len=8192, gpu_util=0.88)
        P_ = models[m]["params"]
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            prompts = [eng.encode(S.QA_SYSTEM, S.qa_prompt(qmap[qid]["question"], S.selected_refs(qmap[qid], sel), cot=True) + W.CONF_ANS) for qid, sel in chunk]
            meta = [{"kind": "qa", "gold": qmap[qid]["answer"], "hit": 1.0, "cot": True, "conf": True} for qid, _ in chunk]
            outs = eng.generate(prompts, 1, 0.0, 512, [0] * len(chunk), meta)
            rows = []
            for (qid, sel), p, o in zip(chunk, prompts, outs):
                text, k = o[0][0], o[0][1]
                pred = W.answer_of(text)
                em, f1 = S.em_f1(pred, qmap[qid]["answer"])
                rows.append({"qid": qid, "sel": list(sel), "model": m, "prompt_tokens": len(p), "out_tokens": k, "flops": 2 * P_ * (len(p) + k),
                             "text": text, "prediction": pred, "EM": em, "F1": f1, "lenient": X.lenient(pred, qmap[qid]["answer"]),
                             "stated_conf": W.stated_conf(text), "hit_cap": k >= 512, "mean_logprob": o[0][2] if len(o[0]) > 2 else None,
                             "from": args.tag})
            S.append_jsonl(path, rows)
            log(f"  {m}: {min(c + args.chunk, len(todo))}/{len(todo)}")
        del eng
        _free()


# ==================================================================================
# 5. Evaluation
# ==================================================================================
def boot_ci(a, b, B=4000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    d = [a[w].mean() - b[w].mean() for w in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))


def cmd_evaluate(args):
    import pandas as pd
    out = os.path.join(args.run_dir, "results")
    os.makedirs(out, exist_ok=True)
    models = W.load_models(args.models_json)
    qmap, _ = shortlist_jobs(args)
    order = [q["qid"] for _, _, d in test_sets(args) for q in S.load_questions(d)]
    origin = {q["qid"]: nm for nm, _, d in test_sets(args) for q in S.load_questions(d)}
    # retrieval rows per variant: sel and retrieval compute
    R = {}
    for name in args.variants:
        R[name] = {r["qid"]: r for r in S.read_jsonl(ret_file(args.run_dir, name))}
    R["q7_ref"], R["gold"] = {}, {}
    for _, run, d in test_sets(args):
        rets = {r["qid"]: r for r in S.read_jsonl(W.ret_path(run, W.vid(RC.RET, RC.COT, RC.PICK)))}
        for q in S.load_questions(d):
            row = rets[q["qid"]]
            sel = W.shortlists(row, RC.COT, len(q["ctxs_candidate"]))[(1, RC.MERGE)]
            g = {i + 1 for i in q["ret_answers"]}
            R["q7_ref"][q["qid"]] = {"sel": list(sel), "flops": 2 * models[RC.RET]["params"] * (row["prompt_tokens"] + row["samples"][0]["out_tokens"]),
                                     "prompt_tokens": row["prompt_tokens"], "out_tokens": row["samples"][0]["out_tokens"],
                                     "found_both": float(g <= set(sel)), "n_passages": len(sel), "precision": len(g & set(sel)) / len(sel)}
            R["gold"][q["qid"]] = {"sel": sorted(g), "flops": 0.0, "prompt_tokens": 0, "out_tokens": 0, "found_both": 1.0, "n_passages": len(g), "precision": 1.0}
    A = {m: {(r["qid"], tuple(r["sel"])): r for r in S.read_jsonl(ans_file(args.run_dir, m))} for m in args.models}
    names = ["q7_ref"] + [v for v in args.variants if v in VARIANTS] + ["gold"]
    missing = [(m, n) for m in args.models for n in names for qid in order if (qid, tuple(R[n][qid]["sel"])) not in A[m]]
    assert not missing, f"{len(missing)} answers missing - run `answer` first"
    ref_c = np.array([R["q7_ref"][q]["flops"] + A["q32"][(q, tuple(R["q7_ref"][q]["sel"]))]["flops"] for q in order])
    rows_r, rows_a = [], []
    for subset, keep in (("all 400", lambda q: True), ("old 200", lambda q: origin[q] == "old"), ("fresh 200", lambda q: origin[q] == "fresh")):
        qs = [q for q in order if keep(q)]
        idx = np.array([order.index(q) for q in qs])
        for n in names:
            rr = [R[n][q] for q in qs]
            rows_r.append({"Questions": subset, "Retriever": LABEL[n], "Both gold passages found": float(np.mean([r["found_both"] for r in rr])),
                           "Mean shortlist size": float(np.mean([r["n_passages"] for r in rr])), "Precision (share of picks that are gold)": float(np.mean([r["precision"] for r in rr])),
                           "Mean retrieval output tokens": float(np.mean([r["out_tokens"] for r in rr])),
                           "Retrieval compute vs Qwen 7B retrieval": float(np.sum([r["flops"] for r in rr]) / np.sum([R["q7_ref"][q]["flops"] for q in qs])) if n != "gold" else None})
            for m in args.models:
                for metric in ("EM", "lenient"):
                    y = np.array([float(A[m][(q, tuple(R[n][q]["sel"]))][metric]) for q in qs])
                    yb = np.array([float(A["q32"][(q, tuple(R["q7_ref"][q]["sel"]))][metric]) for q in qs])
                    c = np.array([R[n][q]["flops"] + A[m][(q, tuple(R[n][q]["sel"]))]["flops"] for q in qs])
                    lo, hi = boot_ci(y, yb)
                    rows_a.append({"Questions": subset, "Metric": metric, "Retriever": LABEL[n], "Answerer": W.LABEL.get(m, m), "Accuracy": float(y.mean()),
                                   "Cheap fixed setup (Qwen 7B retrieves, Qwen 32B answers)": float(yb.mean()),
                                   "vs cheap fixed setup (points)": 100 * float(y.mean() - yb.mean()), "low": lo, "high": hi,
                                   "Compute vs the cheap fixed setup": float(c.sum() / ref_c[idx].sum() - 1) if n != "gold" else None})
    st = json.load(open(os.path.join(args.adapter_dir, "train_state.json"))) if args.adapter_dir and os.path.exists(os.path.join(args.adapter_dir, "train_state.json")) else {}
    curve = st.get("curve", [])
    W.write_excel(os.path.join(out, "set7_retriever_finetune.xlsx"), {"Retrieval": pd.DataFrame(rows_r), "Answers": pd.DataFrame(rows_a),
                                                                      "Training curve": pd.DataFrame(curve)})
    for nm, rr in (("retrieval.jsonl", rows_r), ("answers.jsonl", rows_a), ("training_curve.jsonl", curve)):
        with open(os.path.join(out, nm), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, default=float) + "\n")
    T = lambda rr, cols, pct=(), sg=(): ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join(("-" if r.get(c) is None else (r[c] if isinstance(r[c], str) else (f"{100 * r[c]:.1f}%" if c in pct else (f"{r[c]:+.1f}" if c in sg else f"{r[c]:.2f}"))))
                          for c in cols) + " |" for r in rr))
    L = [f"# Set 7: the retriever learns to shortlist (LoRA on HotpotQA gold passages)\n", f"Code version {VERSION}. Test: the 400 questions of notebooks 17-21, never used for training or checkpoint choice. "
         "Answers: one greedy chain-of-thought answer, the notebook 21 prompt. The cheap fixed setup is Qwen 7B retrieving once (step-by-step prompt) and Qwen 32B answering once.\n"]
    if curve:
        L += ["## Training curve (validation questions from HotpotQA's training file)\n",
              T(curve, ["step", "training questions seen", "training loss (recent)", "both gold passages found", "exactly the gold passages", "mean shortlist size"],
                pct={"both gold passages found", "exactly the gold passages"}),
              f"\nChosen checkpoint: step {st.get('chosen', {}).get('step')}. Training time: {st.get('seconds', 0) / 60:.0f} min.\n"]
    L += ["## Retrieval on the test questions\n", T([r for r in rows_r if r["Questions"] == "all 400"], ["Retriever", "Both gold passages found", "Mean shortlist size",
                                                                                                   "Precision (share of picks that are gold)", "Mean retrieval output tokens", "Retrieval compute vs Qwen 7B retrieval"],
                                                    pct={"Both gold passages found", "Precision (share of picks that are gold)"}), "",
          T([r for r in rows_r if r["Questions"] != "all 400"], ["Questions", "Retriever", "Both gold passages found", "Mean shortlist size"], pct={"Both gold passages found"}), ""]
    for metric in ("lenient", "EM"):
        L += [f"## Final accuracy ({'corrected label' if metric == 'lenient' else 'exact match'})\n",
              T([r for r in rows_a if r["Metric"] == metric and r["Questions"] == "all 400"], ["Retriever", "Answerer", "Accuracy", "vs cheap fixed setup (points)", "low", "high",
                                                                                              "Compute vs the cheap fixed setup"], pct={"Accuracy", "Compute vs the cheap fixed setup"},
                sg={"vs cheap fixed setup (points)", "low", "high"}), ""]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log(f"results -> {out}")


def cmd_merge(args):
    """Merge a trained LoRA adapter (the checkpoint chosen on validation) into its base model, for vLLM."""
    import torch  # noqa: F401
    from peft import PeftModel
    _no_torchao()
    if os.path.exists(os.path.join(args.merged_dir, "config.json")):
        log(f"already merged: {args.merged_dir}")
        return
    st = json.load(open(os.path.join(args.adapter_dir, "train_state.json")))
    ck = os.path.join(args.adapter_dir, f"ckpt_step{st['chosen']['step']}")
    from transformers import AutoTokenizer
    if args.backend == "tiny":
        os.makedirs(args.merged_dir, exist_ok=True)
        json.dump({"tiny": True}, open(os.path.join(args.merged_dir, "config.json"), "w"))
        return
    merged = PeftModel.from_pretrained(load_lm(args.base), ck).merge_and_unload()
    merged.save_pretrained(args.merged_dir)
    AutoTokenizer.from_pretrained(args.base).save_pretrained(args.merged_dir)
    copy_side_files(args.base, args.merged_dir)
    log(f"merged {ck} -> {args.merged_dir}")


def cmd_sample(args):
    """Extra sampled answers from one model on given shortlists (tie-breaker votes); same prompt as the greedy answers."""
    models = W.load_models(args.models_json)
    qmap, _ = shortlist_jobs(args)
    path = os.path.join(args.run_dir, "step", f"samples_{args.model}.jsonl")
    have = {(r["qid"], tuple(r["sel"])) for r in S.read_jsonl(path)}
    todo = []
    for f in args.extra_shortlists:
        for r in S.read_jsonl(f):
            k = (r["qid"], tuple(sorted(r["sel"])))
            if k not in have and k not in todo:
                todo.append(k)
    log(f"{args.model}: {len(todo)} shortlists x {args.n} samples to generate")
    if not todo:
        return
    eng = W.make_engine(args.backend, models, args.model, max_model_len=8192, gpu_util=0.88)
    P_ = models[args.model]["params"]
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        prompts = [eng.encode(S.QA_SYSTEM, S.qa_prompt(qmap[qid]["question"], S.selected_refs(qmap[qid], sel), cot=True) + W.CONF_ANS) for qid, sel in chunk]
        meta = [{"kind": "qa", "gold": qmap[qid]["answer"], "hit": 1.0, "cot": True, "conf": True} for qid, _ in chunk]
        seeds = [zlib.crc32((qid + "s").encode()) % 999983 for qid, _ in chunk]
        outs = eng.generate(prompts, args.n, args.temperature, 512, seeds, meta)
        rows = []
        for (qid, sel), p, o in zip(chunk, prompts, outs):
            ss = []
            for t, k, *rest in o:
                pred = W.answer_of(t)
                em, f1 = S.em_f1(pred, qmap[qid]["answer"])
                ss.append({"prediction": pred, "EM": em, "F1": f1, "lenient": X.lenient(pred, qmap[qid]["answer"]), "out_tokens": k,
                           "mean_logprob": rest[0] if rest else None})
            rows.append({"qid": qid, "sel": list(sel), "model": args.model, "prompt_tokens": len(p), "temperature": args.temperature, "samples": ss,
                         "flops": 2 * P_ * (len(ss) * len(p) + sum(x["out_tokens"] for x in ss))})   # each sample charged its full prompt
        S.append_jsonl(path, rows)
        log(f"  {args.model} samples: {min(c + args.chunk, len(todo))}/{len(todo)}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("build_train")
    p.add_argument("--hotpot_train", default="")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--pool", type=int, default=7405)          # the dev file's size, so the distractor corpus is as large as the test questions'
    p.add_argument("--n_candidates", type=int, default=20)
    p.add_argument("--level", default="hard")
    p.add_argument("--seed", type=int, default=7)
    p = sub.add_parser("train")
    for a in ("data_dir", "out_dir", "merged_dir"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--base", default="")
    p.add_argument("--backend", choices=["hf", "tiny"], default="hf")
    p.add_argument("--n_train", type=int, default=4000)
    p.add_argument("--n_val", type=int, default=300)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--r", type=int, default=16)
    p.add_argument("--alpha", type=int, default=32)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--max_len", type=int, default=4096)
    p.add_argument("--checkpoints", type=float, nargs="+", default=[0.25, 0.5, 1.0])
    p.add_argument("--ckpt_questions", type=int, nargs="*", default=[])     # validation checkpoints by questions seen (overrides --checkpoints)
    p.add_argument("--save_every_questions", type=int, default=1000)        # resume points in between (no validation)
    p.add_argument("--micro_batch", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--time_limit_min", type=float, default=0)
    p = sub.add_parser("merge")
    for a in ("adapter_dir", "merged_dir", "base"):
        p.add_argument("--" + a, required=True)
    p.add_argument("--backend", choices=["hf", "tiny"], default="hf")
    for c in ("retrieve", "answer", "evaluate", "sample"):
        p = sub.add_parser(c)
        for a in ("old_sweep", "old_data", "fresh_run", "fresh_data", "run_dir", "models_json"):
            p.add_argument("--" + a, required=True)
        p.add_argument("--backend", choices=["vllm", "mock"], default="vllm")
        p.add_argument("--chunk", type=int, default=128)
        p.add_argument("--variants", nargs="+", default=list(VARIANTS))
        p.add_argument("--models", nargs="+", default=["q32", "l8"])
        p.add_argument("--cache_run", default="")
        p.add_argument("--cache_answers", nargs="*", default=[])
        p.add_argument("--add_variant", nargs="*", default=[], help="name:model_key:label - a further fine-tuned retriever (direct prompt, greedy)")
        p.add_argument("--extra_shortlists", nargs="*", default=[])
        p.add_argument("--require_logprob", action="store_true")
        p.add_argument("--only_extra", action="store_true")
        p.add_argument("--tag", default="notebook 22")
        p.add_argument("--model", default="q32")
        p.add_argument("--n", type=int, default=4)
        p.add_argument("--temperature", type=float, default=0.7)
        p.add_argument("--adapter_dir", default="")
    args = ap.parse_args()
    for v in getattr(args, "add_variant", []) or []:
        name, key, label = v.split(":", 2)
        VARIANTS[name] = (key, "direct", 0.0)
        LABEL[name] = label
    {"build_train": cmd_build_train, "train": cmd_train, "retrieve": cmd_retrieve, "answer": cmd_answer, "evaluate": cmd_evaluate,
     "merge": cmd_merge, "sample": cmd_sample}[args.cmd](args)


if __name__ == "__main__":
    main()
