"""Set 9 - the smallest model learns where each question should go; both stages ensemble, score themselves, and escalate
until the calibrated score passes a threshold; every answer carries a Sure / Unsure flag (notebook 27).

The 0.5B retriever (Qwen2.5 0.5B, trained in notebook 26) is taught two more lines after its shortlist:
    Selected references: [3] [11]
    Shortlist with: 0.5B | 0.6B | 0.8B | 3B
    Answer with: Llama 8B | Qwen 32B | Qwen 32B, likely wrong
The labels come from HotpotQA training questions that none of the retrievers was trained on ('labelling questions'):
  Shortlist with = the cheapest fine-tuned retriever whose shortlist contains both gold passages (none -> 3B)
  Answer with    = Llama 8B if Llama answers right on the 0.5B shortlist, else Qwen 32B if Qwen does, else 'Qwen 32B, likely wrong'
The model's probabilities over the options of each line are read out (not only its top choice).

Retrieval stage: start at the learned retriever, ensemble its shortlist with the 0.5B's (agree -> keep; majority; else union),
score the ensemble (calibrated chance that both gold passages are in it); below the threshold -> add the next larger retriever.
Answer stage: start at the learned answerer; votes are added step by step (Llama start: Llama -> + Qwen -> + 4 Qwen samples;
Qwen start: Qwen -> + 2 Qwen samples -> + 2 more samples and Llama); majority by agreement; score (calibrated chance the answer
is right); below the threshold -> next step. A final score below the threshold -> 'Unsure', also for Qwen 32B.
The calibrations (logistic regressions) and the threshold choice use 1,000 labelling questions held out from the route training;
the 400 test questions are used once.

Commands: build_label, retrieve, answer, sample, route_data, train_route, route, reach, analyze.
"""
import argparse
import json
import os
import random
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S               # noqa: E402
import sweep as W             # noqa: E402
import experts as X           # noqa: E402
import retft as RT            # noqa: E402
import casc8 as C8            # noqa: E402

VERSION = "9.0"
log = S.log
LADDER = ["r05", "r06", "r08", "r3"]
RNAME = {"r05": "0.5B", "r06": "0.6B", "r08": "0.8B", "r3": "3B"}
RFULL = {"r05": "Qwen2.5 0.5B", "r06": "Qwen3 0.6B", "r08": "Qwen3.5 0.8B", "r3": "Qwen2.5 3B"}
ANS_OPTS = ["Llama 8B", "Qwen 32B", "Qwen 32B, likely wrong"]
key, write, agree, vote = C8.key, C8.write, C8.agree, C8.vote


# ==================================================================================
# question sets
# ==================================================================================
def load_set(args, name):
    if name == "test":
        return [q for d in (args.old_data, args.fresh_data) for q in S.load_questions(d)]
    files = {"route_train": ["route_train"], "calib": ["calib"], "label": ["route_train", "calib"]}[name]
    return [q for f in files for q in json.load(open(os.path.join(args.label_dir, f + ".json"), encoding="utf-8"))]


def cmd_build_label(args):
    """The hard-level training questions NOT among notebook 23a's 12,000 + 300 (same pool, same seed, same BM25 corpus)."""
    data = RT.load_train_file(args.hotpot_train)
    data = [d for d in data if d.get("level") == "hard"]
    rng = random.Random(7)
    pool = rng.sample(range(len(data)), len(data))
    used = set(pool[:args.skip_first])
    rest = [i for i in pool if i not in used]
    if args.n_max:
        rest = rest[:args.n_max]
    seen, corpus = set(), []
    for i in pool:
        for _, sents in data[i]["context"]:
            t = " ".join(sents)
            if t not in seen:
                seen.add(t)
                corpus.append(t)
    log(f"{len(pool)} hard training questions; {len(rest)} not used for training or validation; BM25 over {len(corpus)} paragraphs ...")
    bm25 = S.BM25(corpus)
    r2 = random.Random(8)
    out = []
    for i in rest:
        ex = data[i]
        ctx = {t: " ".join(s) for t, s in ex["context"]}
        gold = []
        for title, _ in ex["supporting_facts"]:
            if title in ctx and ctx[title] not in gold:
                gold.append(ctx[title])
        if len(gold) < 2:
            continue
        cands = gold.copy()
        for j in bm25.search(ex["question"], 30):
            if len(cands) >= 20:
                break
            if corpus[j] not in cands:
                cands.append(corpus[j])
        r2.shuffle(cands)
        out.append({"qid": ex["_id"], "question": ex["question"], "answer": ex["answer"], "type": ex.get("type"), "level": ex.get("level"),
                    "ret_answers": [cands.index(g) for g in gold], "ctxs_candidate": cands})
    n_cal = min(args.n_calib, len(out) // 3)
    tmp = args.label_dir + f".part{os.getpid()}"
    os.makedirs(tmp, exist_ok=True)
    json.dump(out[:-n_cal], open(os.path.join(tmp, "route_train.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(out[-n_cal:], open(os.path.join(tmp, "calib.json"), "w", encoding="utf-8"), ensure_ascii=False)
    S.write_json(os.path.join(tmp, "data_config.json"), {"route_train": len(out) - n_cal, "calib": n_cal, "excluded": "notebook 23a's 12,000 + 300 (seed 7)",
                                                         "version": VERSION})
    os.rename(tmp, args.label_dir)
    log(f"labelling questions: {len(out) - n_cal} for teaching the route lines, {n_cal} for calibration -> {args.label_dir}")


# ==================================================================================
# generation (vLLM / mock): retrieval, answers, samples - all resumable, keyed by (question, shortlist)
# ==================================================================================
def ret_path(args, set_name, r):
    return os.path.join(args.run_dir, "step", f"{set_name}_retrieval_{r}.jsonl")


def cmd_retrieve(args):
    models = W.load_models(args.models_json)
    os.makedirs(os.path.join(args.run_dir, "step"), exist_ok=True)
    qs = load_set(args, args.set)
    for spec in args.retrievers:                     # r:model_key
        r, mk = spec.split(":")
        path = ret_path(args, args.set, r)
        done = {x["qid"] for x in S.read_jsonl(path)}
        todo = [q for q in qs if q["qid"] not in done]
        if not todo:
            log(f"{args.set} {r}: complete")
            continue
        eng = W.make_engine(args.backend, models, mk, max_model_len=8192, gpu_util=0.85)
        eng.thinking = False
        P_ = models[mk]["params"]
        for c in range(0, len(todo), args.chunk):
            chunk = todo[c:c + args.chunk]
            prompts = [eng.encode(S.RET_SYSTEM, RT.direct_prompt(q)) for q in chunk]
            meta = [{"kind": "ret", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"]), "q": q["question"], "conf": False} for q in chunk]
            outs = eng.generate(prompts, 1, 0.0, 48, [zlib.crc32((r + q["qid"]).encode()) % 999983 for q in chunk], meta)
            rows = []
            for q, p, o in zip(chunk, prompts, outs):
                text, k = o[0][0], o[0][1]
                sel = key(W.picks_of(S.selection_text(text), len(q["ctxs_candidate"])) or (1,))
                rows.append({"qid": q["qid"], "retriever": r, "text": text, "sel": list(sel), "prompt_tokens": len(p), "out_tokens": k,
                             "flops": 2 * P_ * (len(p) + k), "found_both": float({i + 1 for i in q["ret_answers"]} <= set(sel)),
                             "mean_logprob": o[0][2] if len(o[0]) > 2 else None})
            S.append_jsonl(path, rows)
            log(f"  {args.set} {r}: {len(S.read_jsonl(path))}/{len(qs)}")
        del eng
        RT._free()


def qmap_all(args):
    m = {}
    for nm in ("label", "test"):
        try:
            for q in load_set(args, nm):
                m[q["qid"]] = q
        except FileNotFoundError:
            pass
    return m


def jobs_from(files):
    out = []
    seen = set()
    for f in files:
        for r in S.read_jsonl(f):
            k = (r["qid"], key(r["sel"]))
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def cmd_answer(args):
    models = W.load_models(args.models_json)
    qmap = qmap_all(args)
    jobs = jobs_from(args.shortlists)
    for m in args.models:
        path = os.path.join(args.run_dir, "step", f"answers_{m}.jsonl")
        have = {(r["qid"], key(r["sel"])) for r in S.read_jsonl(path) if (r.get("mean_logprob") is not None or not args.require_logprob)}
        want = set(jobs)
        add = []
        for d in args.cache_dirs:
            for r in S.read_jsonl(os.path.join(d, "step", f"answers_{m}.jsonl")):
                k = (r["qid"], key(r["sel"]))
                if k in want and k not in have and (r.get("mean_logprob") is not None or not args.require_logprob):
                    add.append(dict(r, sel=list(k[1]), **{"from": r.get("from", os.path.basename(d))}))
                    have.add(k)
        if add:
            S.append_jsonl(path, add)
            log(f"{m}: reused {len(add)} answers from {args.cache_dirs}")
        todo = [k for k in jobs if k not in have]
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
                             "stated_conf": W.stated_conf(text), "hit_cap": k >= 512, "mean_logprob": o[0][2] if len(o[0]) > 2 else None, "from": "notebook 27"})
            S.append_jsonl(path, rows)
            log(f"  {m}: {min(c + args.chunk, len(todo))}/{len(todo)}")
        del eng
        RT._free()


def cmd_sample(args):
    models = W.load_models(args.models_json)
    qmap = qmap_all(args)
    path = os.path.join(args.run_dir, "step", "samples_q32.jsonl")
    have = {(r["qid"], key(r["sel"])) for r in S.read_jsonl(path) if len(r["samples"]) >= args.n}
    want = set(jobs_from(args.shortlists))
    add = []
    for d in args.cache_dirs:
        for r in S.read_jsonl(os.path.join(d, "step", "samples_q32.jsonl")):
            k = (r["qid"], key(r["sel"]))
            if k in want and k not in have and len(r["samples"]) >= args.n:
                add.append(dict(r, sel=list(k[1])))
                have.add(k)
    if add:
        S.append_jsonl(path, add)
        log(f"reused {len(add)} sample sets")
    todo = [k for k in jobs_from(args.shortlists) if k not in have]
    log(f"q32: {len(todo)} shortlists x {args.n} samples to generate")
    if not todo:
        return
    eng = W.make_engine(args.backend, models, "q32", max_model_len=8192, gpu_util=0.88)
    P_ = models["q32"]["params"]
    for c in range(0, len(todo), args.chunk):
        chunk = todo[c:c + args.chunk]
        prompts = [eng.encode(S.QA_SYSTEM, S.qa_prompt(qmap[qid]["question"], S.selected_refs(qmap[qid], sel), cot=True) + W.CONF_ANS) for qid, sel in chunk]
        meta = [{"kind": "qa", "gold": qmap[qid]["answer"], "hit": 1.0, "cot": True, "conf": True} for qid, _ in chunk]
        outs = eng.generate(prompts, args.n, 0.7, 512, [zlib.crc32((qid + "s").encode()) % 999983 for qid, _ in chunk], meta)
        rows = []
        for (qid, sel), p, o in zip(chunk, prompts, outs):
            ss = []
            for t, k, *rest in o:
                pred = W.answer_of(t)
                em, f1 = S.em_f1(pred, qmap[qid]["answer"])
                ss.append({"prediction": pred, "EM": em, "F1": f1, "lenient": X.lenient(pred, qmap[qid]["answer"]), "out_tokens": k,
                           "mean_logprob": rest[0] if rest else None, "model": "q32s"})
            rows.append({"qid": qid, "sel": list(sel), "model": "q32", "prompt_tokens": len(p), "temperature": 0.7, "samples": ss,
                         "flops": 2 * P_ * (len(ss) * len(p) + sum(x["out_tokens"] for x in ss))})
        S.append_jsonl(path, rows)
        log(f"  q32 samples: {min(c + args.chunk, len(todo))}/{len(todo)}")
    del eng
    RT._free()


# ==================================================================================
# the route lines: labels, training (LoRA on the 0.5B retriever), read-out of the option probabilities
# ==================================================================================
def route_target(q, ret_label, ans_label):
    return RT.target_text(q) + f"\nShortlist with: {RNAME[ret_label]}\nAnswer with: {ans_label}"


def labels_for(q, rets, A):
    """Shortlist label: the cheapest retriever whose shortlist holds both gold passages (none -> 3B). Answer label on the 0.5B shortlist."""
    g = {i + 1 for i in q["ret_answers"]}
    rl = next((r for r in LADDER if g <= set(rets[r][q["qid"]]["sel"])), "r3")
    s = key(rets["r05"][q["qid"]]["sel"])
    aL, aQ = A["l8"].get((q["qid"], s)), A["q32"].get((q["qid"], s))
    if aL is None or aQ is None:
        return rl, None
    al = ANS_OPTS[0] if aL["lenient"] else (ANS_OPTS[1] if aQ["lenient"] else ANS_OPTS[2])
    return rl, al


def load_answers(args):
    A = {}
    for m in ("l8", "q32"):
        d = {}
        for r in S.read_jsonl(os.path.join(args.run_dir, "step", f"answers_{m}.jsonl")):
            k = (r["qid"], key(r["sel"]))
            if k not in d or (d[k].get("mean_logprob") is None and r.get("mean_logprob") is not None):
                d[k] = r
        A[m] = d
    return A


def cmd_route_data(args):
    qs = load_set(args, "route_train")
    rets = {r: {x["qid"]: x for x in S.read_jsonl(ret_path(args, "label", r))} for r in LADDER}
    A = load_answers(args)
    rows, dist = [], {}
    for q in qs:
        rl, al = labels_for(q, rets, A)
        if al is None:
            continue
        rows.append(dict(q, target=route_target(q, rl, al), route_labels=[rl, al]))
        dist[(rl, al)] = dist.get((rl, al), 0) + 1
    random.Random(3).shuffle(rows)
    n_val = min(150, len(rows) // 10)
    out = os.path.join(args.run_dir, "route_data")
    os.makedirs(out, exist_ok=True)
    json.dump(rows[n_val:], open(os.path.join(out, "train.json"), "w", encoding="utf-8"), ensure_ascii=False)
    json.dump(rows[:n_val], open(os.path.join(out, "val.json"), "w", encoding="utf-8"), ensure_ascii=False)
    tab = [{"Shortlist with": RNAME[a], "Answer with": b, "Questions": n, "Share": n / len(rows)} for (a, b), n in sorted(dist.items(), key=lambda x: -x[1])]
    write(os.path.join(out, "label_distribution.jsonl"), tab)
    log(f"route training examples: {len(rows) - n_val} (+{n_val} validation). Labels:")
    for t in tab:
        log(f"  {t['Shortlist with']:>5} | {t['Answer with']:<24} {t['Questions']:>5} ({100 * t['Share']:.1f}%)")


def cmd_train_route(args):
    """retft's LoRA trainer, with the three-line target, starting from the merged 0.5B retriever."""
    RT.target_text = lambda q: q["target"]
    a = argparse.Namespace(data_dir=os.path.join(args.run_dir, "route_data"), out_dir=args.out_dir, merged_dir=args.merged_dir, base=args.base,
                           backend=args.backend, n_train=10 ** 6, n_val=10 ** 6, epochs=args.epochs, grad_accum=8, lr=args.lr, r=16, alpha=32,
                           dropout=0.05, max_len=args.max_len, checkpoints=[1.0], ckpt_questions=[], save_every_questions=500, micro_batch=4, seed=0,
                           time_limit_min=args.time_limit_min)
    RT.cmd_train(a)


def option_logprobs(model, tok, prompt, prefix, options, tails):
    """log P(option + tail | prompt, prefix) for each option, from one batched forward pass (tokens after the shared prefix)."""
    import torch
    pid, _ = RT.chat_ids(tok, prompt)
    seqs = [list(pid) + tok(prefix + o + t, add_special_tokens=False)["input_ids"] for o, t in zip(options, tails)]
    n0 = 0
    while all(len(s) > n0 for s in seqs) and len({s[n0] for s in seqs}) == 1:
        n0 += 1
    L = max(len(s) for s in seqs)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    dev = next(model.parameters()).device
    x = torch.full((len(seqs), L), pad, dtype=torch.long)
    am = torch.zeros((len(seqs), L), dtype=torch.long)
    for b, s in enumerate(seqs):
        x[b, :len(s)] = torch.tensor(s)
        am[b, :len(s)] = 1
    with torch.no_grad():
        lp = torch.log_softmax(model(input_ids=x.to(dev), attention_mask=am.to(dev)).logits.float(), -1)
    out = []
    for b, s in enumerate(seqs):
        out.append(float(sum(lp[b, i - 1, s[i]] for i in range(n0, len(s)))))
    return out


def softmax(v):
    v = np.array(v, float)
    e = np.exp(v - v.max())
    return e / e.sum()


def cmd_route(args):
    """The route model on a question set: its shortlist, its two route lines, and its probabilities over every option."""
    models = W.load_models(args.models_json)
    path = os.path.join(args.run_dir, "step", f"{args.set}_route.jsonl")
    done = {x["qid"] for x in S.read_jsonl(path)}
    qs = [q for q in load_set(args, args.set) if q["qid"] not in done]
    log(f"route read-out on {args.set}: {len(qs)} questions to do")
    if not qs:
        return
    P_ = models["r05route"]["params"]
    if args.backend == "mock":
        rows = []
        for q in qs:
            rng = random.Random(zlib.crc32(q["qid"].encode()))
            g = [i + 1 for i in q["ret_answers"]]
            sel = key(g if rng.random() < 0.85 else [g[0], rng.randint(1, len(q["ctxs_candidate"]))])
            pr = softmax([rng.gauss(0, 1) + (2 if i == 0 else 0) for i in range(4)])
            pa = softmax([rng.gauss(0, 1) + (1 if i == 0 else 0) for i in range(3)])
            rows.append({"qid": q["qid"], "text": "", "sel": list(sel), "p_ret": dict(zip(LADDER, pr.tolist())), "p_ans": dict(zip(ANS_OPTS, pa.tolist())),
                         "ret_choice": LADDER[int(np.argmax(pr))], "ans_choice": ANS_OPTS[int(np.argmax(pa))], "mean_logprob": -rng.uniform(0, 0.5),
                         "prompt_tokens": 2750, "out_tokens": 25, "flops": 2 * P_ * 2775})
        S.append_jsonl(path, rows)
        return
    import torch
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(models["r05route"]["path"])
    model = RT.load_lm(models["r05route"]["path"]).to("cuda" if torch.cuda.is_available() else "cpu").eval()
    dev = next(model.parameters()).device
    rows = []
    for n, q in enumerate(qs):
        prompt = RT.direct_prompt(q)
        pid, _ = RT.chat_ids(tok, prompt)
        with torch.no_grad():
            g = model.generate(torch.tensor([pid], device=dev), max_new_tokens=48, do_sample=False, output_scores=True, return_dict_in_generate=True,
                               pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
        new = g.sequences[0, len(pid):]
        lps = [float(torch.log_softmax(sc[0].float(), -1)[t]) for sc, t in zip(g.scores, new)]
        text = tok.decode(new, skip_special_tokens=True)
        lines = text.strip().splitlines()
        refs_line = lines[0] if lines else "Selected references: [1]"
        sel = key(W.picks_of(refs_line, len(q["ctxs_candidate"])) or (1,))
        refs_line = "Selected references: " + " ".join(f"[{i}]" for i in sel) if not refs_line.lower().startswith("selected references:") else refs_line
        rl = option_logprobs(model, tok, prompt, refs_line + "\nShortlist with:", [" " + RNAME[r] for r in LADDER], ["\n"] * 4)
        pr = softmax(rl)
        rc = LADDER[int(np.argmax(pr))]
        al = option_logprobs(model, tok, prompt, refs_line + f"\nShortlist with: {RNAME[rc]}\nAnswer with:", [" " + o for o in ANS_OPTS], [tok.eos_token or ""] * 3)
        pa = softmax(al)
        rows.append({"qid": q["qid"], "text": text, "sel": list(sel), "p_ret": dict(zip(LADDER, pr.tolist())), "p_ans": dict(zip(ANS_OPTS, pa.tolist())),
                     "ret_choice": rc, "ans_choice": ANS_OPTS[int(np.argmax(pa))], "mean_logprob": float(np.mean(lps)) if lps else None,
                     "prompt_tokens": len(pid), "out_tokens": len(new), "flops": 2 * P_ * (len(pid) + len(new))})
        if len(rows) >= 50 or n == len(qs) - 1:
            S.append_jsonl(path, rows)
            rows = []
            log(f"  {args.set}: {len(done) + n + 1}/{len(done) + len(qs)}")


# ==================================================================================
# the two ladders
# ==================================================================================
def ret_states(start):
    """Member lists, in the order they are tried, when the route says 'start'. The 0.5B shortlist is always a member (it is free)."""
    i = LADDER.index(start)
    first = ["r05"] if start == "r05" else ["r05", start]
    st = [first]
    for r in LADDER[i + 1:]:
        st.append(st[-1] + [r])
    return st


def ensemble(sels):
    sels = [key(s) for s in sels]
    if len(set(sels)) == 1:
        return sels[0], 1.0, 0
    cnt = {s: sels.count(s) for s in set(sels)}
    best = max(cnt.values())
    tops = [s for s, c in cnt.items() if c == best]
    if best >= 2 and len(tops) == 1:
        return tops[0], best / len(sels), 0
    return key(set().union(*sels)), best / len(sels), 1


class World:
    """Everything known about one question set: route read-out, retrievals, answers, samples."""

    def __init__(self, args, set_name, A, SM):
        self.Q = load_set(args, set_name)
        self.qids = [q["qid"] for q in self.Q]
        self.qd = {q["qid"]: q for q in self.Q}
        rset = "test" if set_name == "test" else "label"
        self.route = {x["qid"]: x for x in S.read_jsonl(os.path.join(args.run_dir, "step", f"{set_name}_route.jsonl"))}
        self.ret = {r: {x["qid"]: x for x in S.read_jsonl(ret_path(args, rset, r))} for r in LADDER[1:]}
        self.A, self.SM = A, SM
        self.cache = {}

    def member(self, q, r):
        if r == "r05":
            x = self.route[q]
            return key(x["sel"]), x["mean_logprob"], 0.0          # charged with the route pass
        x = self.ret[r][q]
        return key(x["sel"]), x["mean_logprob"], x["flops"]

    def gold(self, q):
        return {i + 1 for i in self.qd[q]["ret_answers"]}


def lp_(x):
    return -1.0 if x is None else float(x)


def ret_state(W_, q, members):
    ms = [W_.member(q, r) for r in members]
    sel, share, union = ensemble([m[0] for m in ms])
    p = W_.route[q]["p_ret"]
    feats = [len(members), float(share == 1.0), share, np.mean([lp_(m[1]) for m in ms]), min(lp_(m[1]) for m in ms), p[members[-1] if len(members) > 1 else "r05"],
             p["r05"], len(sel), union, float(np.log10({"r05": 0.5, "r06": 0.6, "r08": 0.8, "r3": 3.0}[members[-1]]))]
    return {"sel": sel, "feats": feats, "flops": sum(m[2] for m in ms), "found": float(W_.gold(q) <= set(sel)), "members": members}


def ans_votes(W_, q, sel, start, step):
    """Votes at each step. Llama start: [L], [L, Q], [L, Q, s1-s4]. Qwen start: [Q], [Q, s1, s2], [Q, s1-s4, L]."""
    aL, aQ = W_.A["l8"][(q, sel)], dict(W_.A["q32"][(q, sel)], model="q32")
    aL = dict(aL, model="l8")
    sm = W_.SM[(q, sel)]
    ss = [dict(s, model="q32s") for s in sm["samples"][:4]]
    per = sm["flops"] / max(1, len(sm["samples"]))
    if start == "l8":
        v = [[aL], [aQ, aL], [aQ, aL] + ss][step - 1]          # Qwen first: preferred in ties
        n_s = [0, 0, 4][step - 1]
    else:
        v = [[aQ], [aQ] + ss[:2], [aQ] + ss + [aL]][step - 1]
        n_s = [0, 2, 4][step - 1]
    c = sum(x["flops"] for x in v if x["model"] != "q32s") + per * n_s
    return v, c


def ans_state(W_, q, sel, start, step, rs):
    v, c = ans_votes(W_, q, sel, start, step)
    win, sup = vote(v)
    supp = [x for x in v if agree(x["prediction"], win["prediction"]) or x is win]
    have_L = any(x["model"] == "l8" for x in v)
    have_Q = any(x["model"] == "q32" for x in v)
    lpL = lp_(next((x.get("mean_logprob") for x in v if x["model"] == "l8"), None)) if have_L else 0.0
    lpQ = lp_(next((x.get("mean_logprob") for x in v if x["model"] == "q32"), None)) if have_Q else 0.0
    pa = W_.route[q]["p_ans"]
    feats = [step, float(start == "q32"), len(v), sup / len(v), float(win["model"] == "q32"), float(win["model"] == "l8"),
             np.mean([lp_(x.get("mean_logprob")) for x in supp]), float(have_L), lpL, float(have_Q), lpQ,
             float(have_L and have_Q and agree(v[0]["prediction"], v[1]["prediction"])) if (have_L and have_Q) else 0.0,
             pa[ANS_OPTS[0]], pa[ANS_OPTS[1]], pa[ANS_OPTS[2]], rs["feats"][1], rs["feats"][0], len(sel)]
    return {"win": win, "feats": feats, "flops": c, "len": float(win["lenient"]), "em": float(win["EM"]), "votes": [x["model"] for x in v]}


def fit_lr(X_, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y, float)
    if y.min() == y.max():            # degenerate (dry runs): a constant
        c = float(y.mean())
        return lambda X: np.full(len(X), c)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(np.asarray(X_, float), y)
    return lambda X: m.predict_proba(np.asarray(X, float))[:, 1]


def ans_start(route_row):
    pa = route_row["p_ans"]
    return "l8" if pa[ANS_OPTS[0]] >= pa[ANS_OPTS[1]] + pa[ANS_OPTS[2]] else "q32"


def run_policy(W_, q, cal, tr, ta, mode="full"):
    """One question through the cascade. mode: full | no_route (always start at 0.5B and Llama) | route_only (learned start, no ensemble,
    no escalation) | everything (every retriever, every vote)."""
    rr = W_.route[q]
    route_c = rr["flops"]
    start_r = "r05" if mode in ("no_route", "everything") else rr["ret_choice"]
    start_a = "l8" if mode in ("no_route", "everything") else ans_start(rr)
    trail = []
    if mode == "route_only":
        sel, _, f = W_.member(q, start_r)
        rs = {"sel": sel, "feats": [1, 1.0], "flops": f, "found": float(W_.gold(q) <= set(sel)), "members": [start_r]}
        r_score, r_sure = None, None
        a = ans_state(W_, q, sel, start_a, 1, rs)
        a_score, a_sure = None, None
    else:
        states = ret_states(start_r)
        if mode == "everything":
            states = [LADDER]
        for members in states:
            rs = ret_state(W_, q, members)
            k = ("r", q, tuple(members))
            if k not in W_.cache:
                W_.cache[k] = float(cal["ret"]([rs["feats"]])[0])
            r_score = W_.cache[k]
            trail.append(f"{'+'.join(RNAME[m] for m in members)}: {r_score:.2f}")
            if r_score >= tr:
                break
        r_sure = r_score >= tr
        sel = rs["sel"]
        steps = [3] if mode == "everything" else [1, 2, 3]
        for st in steps:
            a = ans_state(W_, q, sel, start_a, st, rs)
            k = ("a", q, tuple(rs["members"]), start_a, st)
            if k not in W_.cache:
                W_.cache[k] = float(cal["ans"]([a["feats"] + [r_score]])[0])
            a_score = W_.cache[k]
            trail.append(f"{'+'.join(a['votes'])}: {a_score:.2f}")
            if a_score >= ta:
                break
        a_sure = a_score >= ta
    return {"qid": q, "lenient": a["len"], "EM": a["em"], "flops": route_c + rs["flops"] + a["flops"], "ret_start": RNAME[start_r],
            "retrievers_run": "+".join(RNAME[m] for m in rs["members"]), "ret_score": r_score, "ret_flag": None if r_sure is None else ("Sure" if r_sure else "Unsure"),
            "found_both": rs["found"], "shortlist": list(sel), "ans_start": {"l8": "Llama 8B", "q32": "Qwen 32B"}[start_a], "votes": "+".join(a["votes"]),
            "answer": a["win"]["prediction"], "ans_score": a_score, "flag": None if a_sure is None else ("Sure" if a_sure else "Unsure"), "trail": " | ".join(trail),
            "escalated": float(len(rs["members"]) > (1 if start_r == "r05" else 2) or len(a["votes"]) > 1)}


def calibrate(W_):
    """Logistic regressions on the calibration questions: every ladder state from every start (both stages)."""
    Xr, yr, Xa, ya = [], [], [], []
    for q in W_.qids:
        for s in LADDER:
            for members in ret_states(s):
                rs = ret_state(W_, q, members)
                Xr.append(rs["feats"])
                yr.append(rs["found"])
    cal = {"ret": fit_lr(Xr, yr)}
    for q in W_.qids:
        seen = set()
        for s in LADDER:
            for members in ret_states(s):
                rs = ret_state(W_, q, members)
                if rs["sel"] in seen:
                    continue
                seen.add(rs["sel"])
                r_score = float(cal["ret"]([rs["feats"]])[0])
                for st_a in ("l8", "q32"):
                    for step in (1, 2, 3):
                        a = ans_state(W_, q, rs["sel"], st_a, step, rs)
                        Xa.append(a["feats"] + [r_score])
                        ya.append(a["len"])
    cal["ans"] = fit_lr(Xa, ya)
    log(f"calibration: {len(Xr)} retrieval states, {len(Xa)} answer states from {len(W_.qids)} questions")
    return cal


def reach_shortlists(W_):
    out = set()
    for q in W_.qids:
        for s in LADDER:
            for members in ret_states(s):
                out.add((q, ret_state(W_, q, members)["sel"]))
            out.add((q, W_.member(q, s)[0]))
    return out


def cmd_reach(args):
    """Every shortlist either ladder can reach (answers and samples are made for all of them).
    --sets route_train: the 0.5B shortlists of the route-training questions (their answers give the 'Answer with' labels)."""
    if args.sets == ["route_train"]:
        rt = load_set(args, "route_train")
        r05 = {x["qid"]: x for x in S.read_jsonl(ret_path(args, "label", "r05"))}
        write(os.path.join(args.run_dir, "step", "route_train_shortlists.jsonl"), [{"qid": q["qid"], "sel": r05[q["qid"]]["sel"]} for q in rt])
        log(f"route_train: {len(rt)} shortlists")
        return
    for nm in args.sets:
        Wd = World(args, nm, {}, {})
        rows = [{"qid": q, "sel": list(s)} for q, s in sorted(reach_shortlists(Wd))]
        if nm == "test":
            models = W.load_models(args.models_json)
            ref = C8.ref_rows(args, models)
            write(os.path.join(args.run_dir, "step", "test_ref_shortlists.jsonl"), [{"qid": q, "sel": list(ref[q]["sel"])} for q in Wd.qids])
        write(os.path.join(args.run_dir, "step", f"{nm}_reach.jsonl"), rows)
        log(f"{nm}: {len(rows)} reachable shortlists ({len(rows) / len(Wd.qids):.2f} per question)")


# ==================================================================================
# analysis
# ==================================================================================
def cmd_analyze(args):
    import pandas as pd
    A = load_answers(args)
    SM = {(r["qid"], key(r["sel"])): r for r in S.read_jsonl(os.path.join(args.run_dir, "step", "samples_q32.jsonl"))}
    Wc, Wt = World(args, "calib", A, SM), World(args, "test", A, SM)
    cal = calibrate(Wc)
    grid = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    res = os.path.join(args.run_dir, "results")
    os.makedirs(os.path.join(res, "charts"), exist_ok=True)

    def table(Wx, tr, ta, mode="full"):
        return [run_policy(Wx, q, cal, tr, ta, mode) for q in Wx.qids]

    # threshold pair chosen on the calibration questions: highest accuracy, then the cheapest within 0.5 point
    cand = []
    for tr in grid:
        for ta in grid:
            rr = table(Wc, tr, ta)
            cand.append((np.mean([r["lenient"] for r in rr]), np.mean([r["flops"] for r in rr]), tr, ta))
    best = max(c[0] for c in cand)
    _, _, tr_c, ta_c = min([c for c in cand if c[0] >= best - 0.005], key=lambda c: c[1])
    T_MAIN = (args.target, args.target)
    log(f"thresholds: stated beforehand {T_MAIN}; chosen on calibration questions ({tr_c}, {ta_c})")
    # references on the test questions
    models = W.load_models(args.models_json)
    ref = C8.ref_rows(args, models)
    fixed = np.array([(A["q32"][(q, ref[q]["sel"])]["lenient"], A["q32"][(q, ref[q]["sel"])]["EM"],
                       ref[q]["flops"] + A["q32"][(q, ref[q]["sel"])]["flops"]) for q in Wt.qids], float)
    ret3 = {x["qid"]: x for x in S.read_jsonl(ret_path(args, "test", "r3"))}

    def single(q, sel, m, rf):
        a = A[m][(q, key(sel))]
        return (a["lenient"], a["EM"], rf + a["flops"])
    ref3 = np.array([single(q, ret3[q]["sel"], "q32", ret3[q]["flops"]) for q in Wt.qids], float)
    pols = {
        "Cheap fixed setup: Qwen 7B retrieves, Qwen 32B answers (AgentTTS's pick)": fixed,
        "Qwen 3B fine-tuned retrieves, Qwen 32B answers (notebook 23a's best fixed pipeline)": ref3,
        "0.5B route model retrieves, Llama 8B answers": np.array([single(q, Wt.route[q]["sel"], "l8", Wt.route[q]["flops"]) for q in Wt.qids], float),
        "0.5B route model retrieves, Qwen 32B answers": np.array([single(q, Wt.route[q]["sel"], "q32", Wt.route[q]["flops"]) for q in Wt.qids], float),
    }
    detail = {}
    for name, tr, ta, mode in ((f"LEARNED ROUTE + ENSEMBLE + ESCALATION, thresholds {T_MAIN[0]:.2f}/{T_MAIN[1]:.2f} (stated beforehand)", *T_MAIN, "full"),
                               (f"LEARNED ROUTE + ENSEMBLE + ESCALATION, thresholds {tr_c:.2f}/{ta_c:.2f} (chosen on calibration questions)", tr_c, ta_c, "full"),
                               (f"No learned route (start at 0.5B and Llama), ensemble + escalation, {T_MAIN[0]:.2f}/{T_MAIN[1]:.2f}", *T_MAIN, "no_route"),
                               ("Learned route only (its chosen retriever and answerer, no ensemble, no escalation)", 0, 0, "route_only"),
                               ("Everything: all four retrievers, Llama + Qwen + 4 samples, vote", 0, 0, "everything")):
        rr = table(Wt, tr, ta, mode)
        detail[name] = rr
        pols[name] = np.array([(r["lenient"], r["EM"], r["flops"]) for r in rr], float)
    rows = []
    for name, v in pols.items():
        lo, hi = C8.boot(v[:, 0], fixed[:, 0])
        lo3, hi3 = C8.boot(v[:, 0], ref3[:, 0])
        d = detail.get(name)
        row = {"Policy": name, "Accuracy (corrected label)": float(v[:, 0].mean()), "Exact match": float(v[:, 1].mean()),
               "TFLOPs per question": float(v[:, 2].mean() / 1e12), "Compute vs the cheap fixed setup": float(v[:, 2].sum() / fixed[:, 2].sum() - 1),
               "vs cheap fixed setup (points)": 100 * float(v[:, 0].mean() - fixed[:, 0].mean()), "low": lo, "high": hi,
               "vs 3B -> Qwen 32B (points)": 100 * float(v[:, 0].mean() - ref3[:, 0].mean()), "low ": lo3, "high ": hi3}
        if d and d[0]["flag"] is not None:
            sure = np.array([r["flag"] == "Sure" for r in d])
            y = v[:, 0]
            row |= {"Answers flagged Sure": float(sure.mean()), "Accuracy when Sure": float(y[sure].mean()) if sure.any() else None,
                    "Accuracy when Unsure": float(y[~sure].mean()) if (~sure).any() else None}
        rows.append(row)
    # Sure/Unsure, both stages, for the two main settings
    rows_f = []
    for name in list(detail)[:3]:
        d = detail[name]
        for stage, fl, y in (("Retrieval (both gold passages in the shortlist)", "ret_flag", "found_both"), ("Answer (corrected label)", "flag", "lenient"),
                             ("Answer (exact match)", "flag", "EM")):
            for f in ("Sure", "Unsure"):
                sub = [r for r in d if r[fl] == f]
                rows_f.append({"Policy": name[:60], "Stage": stage, "Flag": f, "Questions": len(sub), "Share": len(sub) / len(d),
                               "Right": float(np.mean([r[y] for r in sub])) if sub else None})
    # reliability of the answer score (main setting)
    main = detail[list(detail)[0]]
    rows_rel = []
    for lo_, hi_ in ((0, 0.5), (0.5, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.01)):
        sub = [r for r in main if lo_ <= r["ans_score"] < hi_]
        rows_rel.append({"Answer score": f"{lo_:.1f}-{min(hi_, 1):.1f}", "Questions": len(sub), "Mean score": float(np.mean([r["ans_score"] for r in sub])) if sub else None,
                         "Actually right": float(np.mean([r["lenient"] for r in sub])) if sub else None})
    # what the route chose on the test questions, and how often that matched the test labels
    rows_route = []
    lab_r = {q: next((r for r in LADDER if Wt.gold(q) <= set(Wt.member(q, r)[0])), "r3") for q in Wt.qids}
    for r in LADDER:
        sub = [q for q in Wt.qids if Wt.route[q]["ret_choice"] == r]
        rows_route.append({"Route said": f"Shortlist with {RNAME[r]}", "Questions": len(sub),
                           "Label agrees (cheapest retriever that finds both)": float(np.mean([lab_r[q] == r for q in sub])) if sub else None,
                           "Questions whose label is this": sum(lab_r[q] == r for q in Wt.qids)})
    for o in ANS_OPTS:
        sub = [q for q in Wt.qids if Wt.route[q]["ans_choice"] == o]
        yl = [A["l8"][(q, key(Wt.route[q]["sel"]))]["lenient"] for q in sub]
        yq = [A["q32"][(q, key(Wt.route[q]["sel"]))]["lenient"] for q in sub]
        rows_route.append({"Route said": f"Answer with {o}", "Questions": len(sub), "Llama right on these": float(np.mean(yl)) if sub else None,
                           "Qwen right on these": float(np.mean(yq)) if sub else None})
    # threshold grid on the test questions (descriptive)
    rows_g = []
    for tr in grid:
        for ta in grid:
            rr = table(Wt, tr, ta)
            rows_g.append({"Retrieval threshold": tr, "Answer threshold": ta, "Accuracy (corrected label)": float(np.mean([r["lenient"] for r in rr])),
                           "Compute vs the cheap fixed setup": float(sum(r["flops"] for r in rr) / fixed[:, 2].sum() - 1),
                           "Answers flagged Sure": float(np.mean([r["flag"] == "Sure" for r in rr])),
                           "Accuracy when Sure": float(np.mean([r["lenient"] for r in rr if r["flag"] == "Sure"])) if any(r["flag"] == "Sure" for r in rr) else None})
    per_q = [dict(r, question=Wt.qd[r["qid"]]["question"], gold=Wt.qd[r["qid"]]["answer"], TFLOPs=r["flops"] / 1e12) for r in main]
    cols = ["qid", "question", "gold", "ret_start", "retrievers_run", "ret_score", "ret_flag", "found_both", "ans_start", "votes", "answer", "ans_score", "flag",
            "lenient", "EM", "TFLOPs", "trail"]

    def df(rr, cols_=None):
        d = pd.DataFrame(rr)
        if cols_:
            d = d[cols_]
        for c in d.columns:
            if d[c].isna().any():
                d[c] = d[c].astype(object).where(d[c].notna(), "-")
        return d
    W.write_excel(os.path.join(res, "set9_learned_route.xlsx"), {"Policies": df(rows), "Sure-Unsure": df(rows_f), "Score reliability": df(rows_rel),
                                                                "What the route chose": df(rows_route), "Threshold grid (test)": df(rows_g),
                                                                "Per question (main)": df(per_q, cols)})
    for nm, rr in (("policies.jsonl", rows), ("flags.jsonl", rows_f), ("reliability.jsonl", rows_rel), ("route.jsonl", rows_route), ("grid.jsonl", rows_g),
                   ("per_question_main.jsonl", per_q)):
        write(os.path.join(res, nm), rr)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(9, 6))
    xs = [100 * (1 + r["Compute vs the cheap fixed setup"]) for r in rows_g]
    ys = [100 * r["Accuracy (corrected label)"] for r in rows_g]
    ax.scatter(xs, ys, s=12, color="tab:gray", label="other threshold pairs (test, descriptive)")
    for r in rows:
        ax.scatter(100 * (1 + r["Compute vs the cheap fixed setup"]), 100 * r["Accuracy (corrected label)"], s=40)
        ax.annotate(r["Policy"][:44], (100 * (1 + r["Compute vs the cheap fixed setup"]), 100 * r["Accuracy (corrected label)"]), fontsize=6.5)
    ax.set_xlabel("Compute, % of the cheap fixed setup")
    ax.set_ylabel("Accuracy, corrected label (%)")
    ax.set_title("Set 9: learned route, ensembles and calibrated escalation (400 test questions)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(res, "charts", "accuracy_vs_compute.png"), dpi=150)
    T = lambda rr, cols_, pct=(), sg=(): ("| " + " | ".join(cols_) + " |\n|" + "---|" * len(cols_) + "\n" + "\n".join(
        "| " + " | ".join(("-" if r.get(c) is None else (r[c] if isinstance(r[c], str) else (f"{100 * r[c]:.1f}%" if c in pct else (f"{r[c]:+.1f}" if c in sg else
                                                                                                                              (str(r[c]) if isinstance(r[c], int) else f"{r[c]:.2f}")))))
                          for c in cols_) + " |" for r in rr))
    lab = json.load(open(os.path.join(args.run_dir, "route_data", "train.json"), encoding="utf-8")) if os.path.exists(os.path.join(args.run_dir, "route_data", "train.json")) else []
    L = [f"# Set 9: the 0.5B learns where each question should go; ensembles; calibrated escalation; Sure / Unsure\n",
         f"Code version {VERSION}. Route lines taught on {len(lab)} labelling questions; calibration and the chosen thresholds on {len(Wc.qids)} other "
         f"labelling questions; tested once on the 400 test questions. Compute counts every model that runs, including the 0.5B route pass.\n",
         "## Policies\n",
         T(rows, ["Policy", "Accuracy (corrected label)", "Exact match", "Compute vs the cheap fixed setup", "vs cheap fixed setup (points)", "low", "high",
                  "vs 3B -> Qwen 32B (points)", "low ", "high ", "Answers flagged Sure", "Accuracy when Sure", "Accuracy when Unsure"],
           pct={"Accuracy (corrected label)", "Exact match", "Compute vs the cheap fixed setup", "Answers flagged Sure", "Accuracy when Sure", "Accuracy when Unsure"},
           sg={"vs cheap fixed setup (points)", "low", "high", "vs 3B -> Qwen 32B (points)", "low ", "high "}), "",
         "## Sure / Unsure flags\n", T(rows_f, ["Policy", "Stage", "Flag", "Questions", "Share", "Right"], pct={"Share", "Right"}), "",
         "## Is the answer score honest? (main setting)\n", T(rows_rel, list(rows_rel[0]), pct={"Mean score", "Actually right"}), "",
         "## What the learned route chose (test questions)\n",
         T(rows_route, ["Route said", "Questions", "Label agrees (cheapest retriever that finds both)", "Questions whose label is this", "Llama right on these",
                        "Qwen right on these"], pct={"Label agrees (cheapest retriever that finds both)", "Llama right on these", "Qwen right on these"}), "",
         "Per-question results (route, retrievers run, scores, Sure/Unsure, answer, compute, the trail of scores): `set9_learned_route.xlsx`, sheet 'Per question (main)'.\n"]
    open(os.path.join(res, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log(f"results -> {res}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = ("old_sweep", "old_data", "fresh_run", "fresh_data", "run_dir", "models_json", "label_dir")
    for c in ("build_label", "retrieve", "answer", "sample", "route_data", "train_route", "route", "reach", "analyze"):
        p = sub.add_parser(c)
        for a in common:
            p.add_argument("--" + a, default="")
        p.add_argument("--backend", default="vllm")
        p.add_argument("--chunk", type=int, default=256)
        p.add_argument("--set", default="label")
        p.add_argument("--sets", nargs="*", default=["calib", "test"])
        p.add_argument("--retrievers", nargs="*", default=[])
        p.add_argument("--models", nargs="*", default=["l8", "q32"])
        p.add_argument("--shortlists", nargs="*", default=[])
        p.add_argument("--cache_dirs", nargs="*", default=[])
        p.add_argument("--require_logprob", action="store_true")
        p.add_argument("--n", type=int, default=4)
        p.add_argument("--hotpot_train", default="")
        p.add_argument("--n_calib", type=int, default=1000)
        p.add_argument("--n_max", type=int, default=0)
        p.add_argument("--skip_first", type=int, default=12300)      # notebook 23a's 12,000 training + 300 validation questions
        p.add_argument("--out_dir", default="")
        p.add_argument("--merged_dir", default="")
        p.add_argument("--base", default="")
        p.add_argument("--epochs", type=int, default=2)
        p.add_argument("--lr", type=float, default=1e-4)
        p.add_argument("--max_len", type=int, default=4096)
        p.add_argument("--time_limit_min", type=float, default=0)
        p.add_argument("--target", type=float, default=0.9)
    args = ap.parse_args()
    {"build_label": cmd_build_label, "retrieve": cmd_retrieve, "answer": cmd_answer, "sample": cmd_sample, "route_data": cmd_route_data,
     "train_route": cmd_train_route, "route": cmd_route, "reach": cmd_reach, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
