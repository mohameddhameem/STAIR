"""Set 5d part 2 - hidden signals from ONE cheap generation (GPU for `extract`, CPU for `analyze`).

Stage A of the plan: measure whether each signal is informative. The signals found here are then handed to the rule-writing
planner and the deciding model (stage B, notebook 15); nothing in this file makes a routing decision by itself.

  extract --role answer    a cheap answerer answers ONCE (greedy) on the shortlist the cheap fixed setup uses; saved per question:
                             1. hidden states  at the last prompt token (before any answer), over the answer tokens, on the final answer line
                             3. attention      share on each passage / the question / its own text, measured at the answer tokens
                                token probabilities of that single answer, its stated confidence, the answer itself
  extract --role retrieve  the retriever's hidden state after reading the question and the 20 passages (no generation)
  analyze                  probes with held-out scores and shuffled-label controls, layer by layer;
                             2. semantic uncertainty probe = a probe trained ONLY to predict whether 5 answers agree, then used as a signal
                           and what each signal would buy as a gate in front of the big answerer, against the cheap fixed setup.

vLLM does not return hidden states or attention, so extraction uses the transformers library directly.
`--backend tiny` builds a tiny random model in memory so that every code path can be tested without a GPU or model files.
"""
import argparse
import json
import os
import re
import sys
import time
import warnings
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S          # noqa: E402
import sweep as W        # noqa: E402

warnings.filterwarnings("ignore")
VERSION = "5d2.3"
RET, COT, PICK, RK, MERGE = "q7", 1, "3", 1, "all"        # the shortlist the cheap fixed setup answers from
BIG = "q32"
LAYER_FRACTIONS = (0.25, 0.5, 0.625, 0.75, 0.875, 1.0)
MAX_NEW = 512
ATTN_MAX_TOKENS = 2200


def log(*a):
    print(time.strftime("[%H:%M:%S]"), *a, flush=True)


def step_paths(run, key, role):
    return os.path.join(run, "step", f"hidden_{role}_{key}.npz"), os.path.join(run, "step", f"hidden_{role}_{key}.jsonl")


# ==================================================================================
# Models
# ==================================================================================
def tiny_model(texts, seed=0):
    """A 4-layer random model with a word-level tokenizer trained on the given texts. For testing the code paths only."""
    import torch
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers
    from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast
    tk = Tokenizer(models.WordLevel(unk_token="<unk>"))
    tk.pre_tokenizer = pre_tokenizers.Sequence([pre_tokenizers.WhitespaceSplit(), pre_tokenizers.Punctuation()])
    tk.train_from_iterator(texts, trainers.WordLevelTrainer(vocab_size=6000, special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<|sys|>", "<|user|>", "<|bot|>"]))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    tok.chat_template = ("{{ '<s>' }}{% for m in messages %}{{ '<|sys|>' if m['role'] == 'system' else '<|user|>' }} {{ m['content'] }} {% endfor %}"
                         "{% if add_generation_prompt %}{{ '<|bot|>' }}{% endif %}")
    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=8192, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval(), tok


def load_model(path, eager):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    cuda = torch.cuda.is_available()
    dt = torch.bfloat16 if cuda else torch.float32
    kw = {"attn_implementation": "eager"} if eager else {}     # the default fast attention does not return attention weights
    try:
        import accelerate  # noqa: F401
        kw["device_map"] = "cuda" if cuda else "cpu"
    except ImportError:
        pass
    try:
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=dt, **kw)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(path, dtype=dt, **kw)
    if next(model.parameters()).dtype != dt:
        model = model.to(dt)
    if cuda and next(model.parameters()).device.type != "cuda":
        model = model.to("cuda")
    return model.eval(), tok


def chat_text(tok, system, user):
    return tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True)


def token_spans(offsets, spans):
    """offsets: (start, end) per token; spans: {name: (char start, char end)} -> {name: token indices}"""
    out = {}
    for name, (a, b) in spans.items():
        out[name] = [i for i, (s, e) in enumerate(offsets) if e > a and s < b and e > s]
    return out


def layer_ids(n_layers):
    return sorted({max(1, int(round(f * n_layers))) for f in LAYER_FRACTIONS})


# ==================================================================================
# Extraction
# ==================================================================================
def attention_summary(att, q_idx, seg):
    """att: [layers, seq, seq] head-averaged attention of the upper half of the layers. q_idx: the tokens that 'look'. seg: token indices of each segment."""
    if len(q_idx) == 0:
        return [float("nan")] * 6
    a = att[:, q_idx, :].mean(0).mean(0)                    # mean over layers and over the looking tokens -> [seq]
    tot = float(a.sum()) or 1.0
    ps = np.array([float(a[idx].sum()) for idx in seg["passages"]]) if seg["passages"] else np.zeros(0)
    pm = float(ps.sum())
    share = ps / pm if pm > 0 else ps
    ent = float(-(share * np.log(share + 1e-12)).sum() / np.log(len(share))) if len(share) > 1 else 0.0
    return [pm / tot, float(a[seg["question"]].sum()) / tot, float(a[seg["generated"]].sum()) / tot, float(a[seg["other"]].sum()) / tot,
            float(share.max()) if len(share) else float("nan"), ent]


ATT_NAMES = ["attention on the passages", "attention on the question", "attention on its own earlier answer text", "attention on the instructions",
             "largest single passage's share of passage attention", "spread of attention across passages (0 = one passage, 1 = even)"]


def extract_answer_one(model, tok, q, sel, want_attention=True):
    import torch
    refs = S.selected_refs(q, sel)
    user = S.qa_prompt(q["question"], refs, cot=bool(COT)) + W.CONF_ANS
    text = chat_text(tok, S.QA_SYSTEM, user)
    enc = tok(text, return_offsets_mapping=True, add_special_tokens=False, return_tensors="pt")
    ids = enc["input_ids"].to(model.device)
    offsets = [tuple(x) for x in enc["offset_mapping"][0].tolist()]
    P = ids.shape[1]
    with torch.no_grad():
        gen = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=MAX_NEW, do_sample=False, output_scores=True,
                             return_dict_in_generate=True, pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
    full = gen.sequences
    new = full[0, P:]
    eos = set(x for x in ([tok.eos_token_id] + list(np.atleast_1d(getattr(model.generation_config, "eos_token_id", None) or []))) if x is not None)
    keep = len(new)
    while keep > 0 and int(new[keep - 1]) in eos:
        keep -= 1
    G = max(keep, 1)
    full = full[:, :P + G]
    out_text = tok.decode(full[0, P:], skip_special_tokens=True)
    lps, ents = [], []
    for t in range(min(G, len(gen.scores))):
        lp = torch.log_softmax(gen.scores[t][0].float(), -1)
        lps.append(float(lp[full[0, P + t]]))
        ents.append(float(-(lp.exp() * lp).nan_to_num(0.0).sum()))
    do_att = want_attention and (P + G) <= ATTN_MAX_TOKENS
    with torch.no_grad():
        fw = model(full, output_hidden_states=True, output_attentions=do_att)
    L = len(fw.hidden_states) - 1
    lids = layer_ids(L)
    # the final answer line among the generated tokens
    pieces = [tok.decode(full[0, P + i:P + i + 1], skip_special_tokens=True) for i in range(G)]
    ends = np.cumsum([len(p) for p in pieces])
    joined = "".join(pieces)
    hits = [m.start() for m in re.finditer(r"answer\s*:", joined, flags=re.I)]
    if hits:
        a0 = hits[-1]
        nl = joined.find("\n", a0)
        a1 = nl if nl > 0 else len(joined)
        tail = [i for i in range(G) if ends[i] > a0 and (ends[i] - len(pieces[i])) < a1]
    else:
        tail = []
    if not tail:
        tail = list(range(max(0, G - 8), G))
    gen_idx = list(range(P, P + G))
    tail_idx = [P + i for i in tail]
    H = torch.stack([fw.hidden_states[k][0] for k in lids]).float()            # [layers, seq, d]
    h_prompt = H[:, P - 1].cpu().numpy()
    h_ans = H[:, gen_idx].mean(1).cpu().numpy()
    h_tail = H[:, tail_idx].mean(1).cpu().numpy()
    att_feats = [float("nan")] * (2 * len(ATT_NAMES))
    if do_att:
        qpos = text.find("Query: " + q["question"])
        spans = {"question": (qpos, qpos + len("Query: " + q["question"])) if qpos >= 0 else (0, 0)}
        for i, r in enumerate(refs):
            p0 = text.find(f"[{i + 1}] " + r)
            spans[f"p{i}"] = (p0, p0 + len(f"[{i + 1}] " + r)) if p0 >= 0 else (0, 0)
        ts = token_spans(offsets, spans)
        used = set(ts["question"]) | {x for i in range(len(refs)) for x in ts[f"p{i}"]}
        seg = {"question": ts["question"], "passages": [ts[f"p{i}"] for i in range(len(refs))], "generated": gen_idx,
               "other": [i for i in range(P) if i not in used]}
        upper = list(range(L // 2, L))
        att = torch.stack([fw.attentions[k][0].float().mean(0) for k in upper]).cpu().numpy()      # [layers, seq, seq], heads averaged
        att_feats = attention_summary(att, gen_idx, seg) + attention_summary(att, tail_idx, seg)
        del att
    del fw, gen, H
    ans = W.answer_of(out_text)
    em, f1 = S.em_f1(ans, q["answer"])
    tail_lp = [lps[i] for i in tail if i < len(lps)]
    row = {"qid": q["qid"], "sel": list(sel), "n_passages": len(refs), "prompt_tokens": int(P), "out_tokens": int(G), "text": out_text, "prediction": ans,
           "gold": q["answer"], "EM": em, "F1": f1, "stated_conf": W.stated_conf(out_text), "hit_cap": bool(G >= MAX_NEW),
           "mean_logprob": float(np.mean(lps)) if lps else None, "min_logprob": float(np.min(lps)) if lps else None,
           "mean_entropy": float(np.mean(ents)) if ents else None, "answer_line_logprob": float(np.mean(tail_lp)) if tail_lp else None,
           "attention_measured": bool(do_att)}
    return row, h_prompt, h_ans, h_tail, np.array(att_feats, float), lids


def extract_retrieve_one(model, tok, q):
    import torch
    text = chat_text(tok, S.RET_SYSTEM, W.ret_prompt(q, COT, PICK))
    ids = tok(text, add_special_tokens=False, return_tensors="pt")["input_ids"].to(model.device)
    with torch.no_grad():
        fw = model(ids, output_hidden_states=True)
    L = len(fw.hidden_states) - 1
    lids = layer_ids(L)
    h = torch.stack([fw.hidden_states[k][0, -1] for k in lids]).float().cpu().numpy()
    lp = torch.log_softmax(fw.logits[0, -1].float(), -1)
    row = {"qid": q["qid"], "prompt_tokens": int(ids.shape[1]), "next_token_entropy": float(-(lp.exp() * lp).nan_to_num(0.0).sum())}
    del fw
    return row, h, lids


def cmd_extract(args):
    import torch
    global MAX_NEW
    MAX_NEW = args.max_new
    run = args.run_dir
    os.makedirs(os.path.join(run, "step"), exist_ok=True)
    models = W.load_models(args.models_json)
    qs = S.load_questions(args.data_dir)
    if args.limit:
        qs = qs[:args.limit]
    npz, jl = step_paths(run, args.model, args.role)
    rows = S.read_jsonl(jl) if os.path.exists(jl) else []
    arrs = dict(np.load(npz)) if os.path.exists(npz) and rows else {}
    done = {r["qid"] for r in rows}
    todo = [q for q in qs if q["qid"] not in done]
    log(f"extract {args.role} {args.model}: {len(done)} done, {len(todo)} to do")
    if not todo:
        return
    t0 = time.time()
    if args.backend == "tiny":
        model, tok = tiny_model([q["question"] for q in qs] + [p for q in qs for p in q["ctxs_candidate"]] + [S.qa_prompt("x", ["y"], cot=True) + W.CONF_ANS,
                                                                                                             W.ret_prompt(qs[0], COT, PICK)],
                                seed=sum(map(ord, args.model)))
    else:
        model, tok = load_model(models[args.model]["path"], eager=(args.role == "answer"))
    log(f"model ready in {time.time() - t0:.0f}s on {model.device}")
    sels = {}
    if args.role == "answer":
        rp = W.ret_path(args.sweep_run, W.vid(RET, COT, PICK))
        for r in S.read_jsonl(rp):
            sels[r["qid"]] = W.shortlists(r, COT, 20)[(RK, MERGE)]
        assert sels, f"no retrieval results at {rp}"
    keys = ["h_prompt", "h_answer", "h_answer_line", "attention"] if args.role == "answer" else ["h_prompt"]
    buf = {k: [arrs[k][i] for i in range(len(rows))] if k in arrs else [] for k in keys}
    lids, t1, since = None, time.time(), 0

    def save():
        S_rows = rows
        with open(jl, "w", encoding="utf-8") as f:
            for r in S_rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        np.savez_compressed(npz, layers=np.array(lids if lids is not None else arrs.get("layers", [])),
                            **{k: np.array(v, np.float16 if k != "attention" else np.float32) for k, v in buf.items()})
    for i, q in enumerate(todo):
        if args.role == "answer":
            row, hp, ha, ht, att, lids = extract_answer_one(model, tok, q, sels[q["qid"]])
            buf["h_prompt"].append(hp); buf["h_answer"].append(ha); buf["h_answer_line"].append(ht); buf["attention"].append(att)
        else:
            row, hp, lids = extract_retrieve_one(model, tok, q)
            buf["h_prompt"].append(hp)
        row["model"] = args.model
        rows.append(row)
        since += 1
        if torch.cuda.is_available() and i % 20 == 0:
            torch.cuda.empty_cache()
        if since >= 20 or i + 1 == len(todo):
            save()
            since = 0
            rate = (time.time() - t1) / (i + 1)
            log(f"  {args.role} {args.model}: {len(rows)}/{len(qs)} ({rate:.1f}s per question)")
        if args.time_limit_min and (time.time() - t0) / 60 > args.time_limit_min:
            save()
            log("time limit reached - stopping cleanly (run again to continue)")
            break
    json.dump({"stage": f"extract_{args.role}_{args.model}", "done": len(rows), "total": len(qs), "seconds": round(time.time() - t0),
               "layers": [int(x) for x in (lids or [])]}, open(os.path.join(run, "stages", f"extract_{args.role}_{args.model}.json"), "w"), indent=1) \
        if os.path.isdir(os.path.join(run, "stages")) else None


# ==================================================================================
# Analysis
# ==================================================================================
def prep_pca(Xtr, Xte, k=16):
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(Xtr)
    sc = StandardScaler().fit(imp.transform(Xtr))
    a, b = sc.transform(imp.transform(Xtr)), sc.transform(imp.transform(Xte))
    if a.shape[1] > k:
        p = PCA(min(k, len(a) - 1), random_state=0).fit(a)
        a, b = p.transform(a), p.transform(b)
        sd = a.std(0) + 1e-9
        a, b = a / sd, b / sd
    return a, b


def reduce_once(X, k=16):
    """Hidden states have thousands of dimensions. They are reduced ONCE to k principal components (no labels are used),
    so that the thousands of cross-validation fits that follow work on a small table instead of the raw activations."""
    from sklearn.decomposition import PCA
    X = np.asarray(X, np.float64)
    ok = ~np.isnan(X).any(1)
    out = np.full((len(X), min(k, max(1, ok.sum() - 1), X.shape[1])), np.nan)
    if ok.sum() < 3:
        return out
    Z = X[ok]
    Z = (Z - Z.mean(0)) / (Z.std(0) + 1e-6)
    out[ok] = PCA(out.shape[1], random_state=0).fit_transform(Z)
    return out


def f_linear(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    a, b = prep_pca(Xtr, Xte)
    return LogisticRegression(C=0.3, max_iter=500).fit(a, ytr).predict_proba(b)[:, 1]


def f_pca2(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    a, b = prep_pca(Xtr, Xte, 2)
    return LogisticRegression(C=1.0, max_iter=500).fit(a[:, :2], ytr).predict_proba(b[:, :2])[:, 1]


def f_knn(Xtr, ytr, Xte):
    from sklearn.neighbors import KNeighborsClassifier
    a, b = prep_pca(Xtr, Xte)
    return KNeighborsClassifier(min(15, len(ytr) - 1)).fit(a, ytr).predict_proba(b)[:, 1]


def f_boost(Xtr, ytr, Xte):
    from sklearn.ensemble import GradientBoostingClassifier
    a, b = prep_pca(Xtr, Xte)
    return GradientBoostingClassifier(n_estimators=40, max_depth=2, subsample=0.8, random_state=0).fit(a, ytr).predict_proba(b)[:, 1]


STRATS = {"linear probe": f_linear, "first 2 principal components (PCA)": f_pca2, "nearest neighbours (t-SNE-style neighbourhoods)": f_knn,
          "boosted trees": f_boost}
SU = "semantic uncertainty probe (trained only to predict whether 5 answers agree)"


def auc(y, s):
    y, s = np.asarray(y, float), np.asarray(s, float)
    ok = ~np.isnan(s) & ~np.isnan(y)
    y, s = y[ok], s[ok]
    if len(y) == 0 or y.min() == y.max():
        return float("nan")
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, s))


def cv(fn, X, y_train, seed, strat=None):
    """Held-out scores. y_train is what the probe is trained on (it may differ from the target it is later scored against)."""
    from sklearn.model_selection import StratifiedKFold
    p = np.full(len(y_train), np.nan)
    ok = ~np.isnan(np.asarray(y_train, float))
    idx = np.where(ok)[0]
    yy = np.asarray(y_train, float)[idx].astype(int)
    if len(idx) < 20 or yy.min() == yy.max() or min(yy.sum(), len(yy) - yy.sum()) < 5:
        return p
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(idx, yy):
        try:
            p[idx[te]] = fn(X[idx[tr]], yy[tr], X[idx[te]])
        except Exception:
            p[idx[te]] = yy[tr].mean()
    return p


def load_extract(run, key, role, qids):
    npz, jl = step_paths(run, key, role)
    if not (os.path.exists(npz) and os.path.exists(jl)):
        return None
    rows = {r["qid"]: (i, r) for i, r in enumerate(S.read_jsonl(jl))}
    a = np.load(npz)
    order = [rows[q][0] if q in rows else -1 for q in qids]
    have = np.array([o >= 0 for o in order])
    out = {"rows": [rows[q][1] if q in rows else None for q in qids], "have": have, "layers": [int(x) for x in a["layers"]]}
    for k in a.files:
        if k == "layers":
            continue
        arr = a[k].astype(np.float32)
        full = np.full((len(qids),) + arr.shape[1:], np.nan, np.float32)
        full[have] = arr[[o for o in order if o >= 0]]
        out[k] = full
    return out


def cmd_analyze(args):
    import pandas as pd
    import routers as R
    t0 = time.time()
    run = args.run_dir
    out = os.path.join(run, "results")
    os.makedirs(out, exist_ok=True)
    log(f"analysis code version {VERSION}")
    D = R.Data(args.sweep_run, args.data_dir)
    models = W.load_models(args.models_json)
    P = {k: v["params"] for k, v in models.items()}
    j_big = D.j(RET, BIG, COT, PICK, MERGE, RK, 1)                         # the cheap fixed setup: Qwen 7B x1 -> Qwen 32B x1
    fixed_acc, fixed_cost = float(D.EM[:, j_big].mean()), float(D.FL[:, j_big].mean() / D.ref)
    ret_fl = D.FR[:, j_big]
    big_tail = D.FA[:, j_big]
    rows, oof, spaces_meta, layer_rows, gens = [], {}, {}, [], {}
    rng = np.random.default_rng(0)
    reps, perms = args.repeats, args.permutations

    def score(space, strat, target, y, p, null_fn, cost):
        a = auc(y, p)
        null = np.array([v for v in (null_fn(i) for i in range(perms)) if v == v])
        rows.append({"What is looked at": space, "How it is read": strat, "What is predicted": target,
                     "Separation on held-out questions (0.5 = useless)": a,
                     "Same method with shuffled labels: 95th percentile": float(np.percentile(null, 95)) if len(null) else float("nan"),
                     "Beats shuffled labels": bool(len(null) and a == a and a > np.percentile(null, 95)), "Extra cost of having this signal": cost,
                     "Questions": int((~np.isnan(np.asarray(p, float))).sum())})
        oof[(space, strat, target)] = p
        return a

    def run_space(space, X, targets, strats, cost, su_label=None):
        """Scores one space against every target with every strategy (+ the semantic-uncertainty probe if su_label is given)."""
        for tg, y in targets.items():
            yf = np.asarray(y, float)
            for st in strats:
                p = np.nanmean([cv(STRATS[st], X, yf, 10 + r) for r in range(reps)], 0)

                def null_fn(i, st=st, yf=yf):
                    ok = ~np.isnan(yf)
                    yp = yf.copy()
                    yp[ok] = rng.permutation(yf[ok])
                    return auc(yp, cv(STRATS[st], X, yp, 500 + i))
                score(space, st, tg, yf, p, null_fn, cost)
        if su_label is not None:
            ya = np.asarray(su_label, float)
            p = np.nanmean([cv(f_linear, X, ya, 10 + r) for r in range(reps)], 0)       # trained on agreement only
            for tg, y in targets.items():
                yf = np.asarray(y, float)

                def null_fn(i, yf=yf):
                    ok = ~np.isnan(ya)
                    yp = ya.copy()
                    yp[ok] = rng.permutation(ya[ok])
                    return auc(yf, cv(f_linear, X, yp, 500 + i))
                score(space, SU, tg, yf, p, null_fn, cost)

    # ------------------------------------------------------------------ answerers
    single = {}
    for key in args.answerers:
        E = load_extract(run, key, "answer", D.qids)
        if E is None:
            log(f"no extraction for answerer {key} - skipped")
            continue
        name = W.LABEL[key]
        have = E["have"]
        if have.sum() < 100:
            log(f"answerer {key}: only {int(have.sum())} of {len(have)} questions extracted so far - left out of the analysis until extraction is complete")
            continue
        em = np.array([r["EM"] if r else np.nan for r in E["rows"]], float)
        single[key] = (E, em)
        j5 = D.j(RET, key, COT, PICK, MERGE, RK, 5)
        agree = D.AG[:, j5]
        targets = {f"{name}'s single answer is right": em,
                   f"{name}'s single answer is no worse than the big answer (routing target)": np.where(have, (em >= D.EM[:, j_big]).astype(float), np.nan),
                   f"{name}'s 5 answers all agree": np.where(have, (agree >= 0.999).astype(float), np.nan)}
        su_y = targets[f"{name}'s 5 answers all agree"]
        gens[key] = [dict(r, **{"split": D.split[i], "question": D.text[i], "big answer right": float(D.EM[i, j_big]), "agreement of 5 answers (Set 4)": float(agree[i])})
                     for i, r in enumerate(E["rows"]) if r]
        n_l = len(E["layers"])
        L = max(E["layers"])
        for pos, lab, cost in (("h_prompt", "hidden state at the last prompt token, before answering", "none: read before the cheap answer is written"),
                               ("h_answer", "hidden state averaged over the answer tokens", "none beyond one cheap answer"),
                               ("h_answer_line", "hidden state on the final answer line", "none beyond one cheap answer")):
            for li in range(n_l):
                sp = f"{name}: {lab} (layer {E['layers'][li]} of {L})"
                Xr = reduce_once(E[pos][:, li, :])
                run_space(sp, Xr, targets, ["linear probe", "first 2 principal components (PCA)", "nearest neighbours (t-SNE-style neighbourhoods)"],
                          cost, su_label=su_y)
                spaces_meta[sp] = {"model": key, "pos": pos, "layer": E["layers"][li], "frac": E["layers"][li] / L, "X": Xr}
            log(f"  {name}: {lab} done")
        tab = lambda cols: np.array([[np.nan if r is None or r.get(c) is None else float(r[c]) for c in cols] for r in E["rows"]], float)
        att0 = E["attention"]
        n_att = len(ATT_NAMES)
        rel = []
        for o in (0, n_att):                                    # shares among passages + question + own text only
            den = att0[:, o] + att0[:, o + 1] + att0[:, o + 2]
            rel += [att0[:, o] / den, att0[:, o + 1] / den, att0[:, o] / (att0[:, o] + att0[:, o + 2])]
        att = np.column_stack([att0] + rel)
        rel_names = ["passages' share of attention, instructions left out", "question's share of attention, instructions left out",
                     "passages versus its own earlier text"]
        att_names = ([f"{n} (from all answer tokens)" for n in ATT_NAMES] + [f"{n} (from the final answer line)" for n in ATT_NAMES]
                     + [f"{n} (from all answer tokens)" for n in rel_names] + [f"{n} (from the final answer line)" for n in rel_names])
        small = {
            f"{name}: evidence-use attention signals": (np.column_stack([att, tab(["n_passages"])]), att_names + ["passages in the shortlist"],
                                                         "none beyond one cheap answer"),
            f"{name}: token probabilities of the single answer": (tab(["mean_logprob", "min_logprob", "mean_entropy", "answer_line_logprob", "out_tokens"]),
                                                                  ["mean token log-probability", "lowest token log-probability", "mean entropy",
                                                                   "log-probability of the answer line", "answer length in tokens"], "none beyond one cheap answer"),
            f"{name}: stated confidence of the single answer": (tab(["stated_conf"]), ["stated confidence"], "none beyond one cheap answer"),
            f"{name}: agreement of 5 answers (the Set 4 signal, for comparison)": (agree[:, None], ["agreement of 5 answers"],
                                                                                   "four more cheap answers: about 1.0x"),
        }
        mid = min(range(n_l), key=lambda i: abs(E["layers"][i] / L - 0.75))
        from sklearn.decomposition import PCA
        from sklearn.impute import SimpleImputer
        hm = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(E["h_answer"][:, mid, :])
        hp8 = PCA(8, random_state=0).fit_transform(hm - hm.mean(0))
        small[f"{name}: all single-generation signals together"] = (
            np.column_stack([hp8, att, small[f"{name}: token probabilities of the single answer"][0], tab(["stated_conf"])]),
            [f"hidden-state component {i + 1}" for i in range(8)] + att_names + small[f"{name}: token probabilities of the single answer"][1] + ["stated confidence"],
            "none beyond one cheap answer")
        for sp, (X, names, cost) in small.items():
            run_space(sp, X, targets, list(STRATS), cost)
            spaces_meta[sp] = {"model": key, "pos": "table", "X": X, "names": names}
        log(f"  {name}: tables done")
    # cross-model agreement of two single answers
    if len(single) >= 2:
        ks = list(single)[:2]
        (Ea, ema), (Eb, emb) = single[ks[0]], single[ks[1]]
        same = np.array([float(S.normalize_answer(a["prediction"]) == S.normalize_answer(b["prediction"]) and a["prediction"] != "") if a and b else np.nan
                         for a, b in zip(Ea["rows"], Eb["rows"])])
        for key, em in ((ks[0], ema), (ks[1], emb)):
            name = W.LABEL[key]
            tg = {f"{name}'s single answer is right": em,
                  f"{name}'s single answer is no worse than the big answer (routing target)": np.where(~np.isnan(em), (em >= D.EM[:, j_big]).astype(float), np.nan)}
            sp = f"{name}: its single answer equals {W.LABEL[[k for k in ks if k != key][0]]}'s single answer"
            other = [k for k in ks if k != key][0]
            run_space(sp, same[:, None], tg, ["linear probe"], f"one single answer from {W.LABEL[other]} as well")
            spaces_meta[sp] = {"model": key, "pos": "table", "X": same[:, None], "names": ["same answer"]}
    # ------------------------------------------------------------------ retriever
    Er = load_extract(run, args.retriever, "retrieve", D.qids)
    if Er is not None:
        name = W.LABEL[args.retriever]
        f1 = D.FOUND[:, D.j(args.retriever, "l8", COT, PICK, "all", 1, 5)]
        f3 = D.FOUND[:, D.j(args.retriever, "l8", COT, PICK, "all", 3, 5)]
        fb = D.FOUND[:, D.j("q32", "l8", COT, PICK, "all", 1, 5)]
        tg = {f"{name} finds both passages in 1 try": np.where(Er["have"], f1, np.nan),
              f"{name} finds both passages in 3 tries (keep all)": np.where(Er["have"], f3, np.nan),
              "a bigger retriever is needed (Qwen 32B finds both, this one does not)": np.where(Er["have"], ((fb == 1) & (f1 == 0)).astype(float), np.nan)}
        L = max(Er["layers"])
        for li in range(len(Er["layers"])):
            sp = f"{name} (retriever): hidden state after reading the question and 20 passages (layer {Er['layers'][li]} of {L})"
            Xr = reduce_once(Er["h_prompt"][:, li, :])
            run_space(sp, Xr, tg, ["linear probe", "first 2 principal components (PCA)", "nearest neighbours (t-SNE-style neighbourhoods)"],
                      "none: part of the retrieval pass")
            spaces_meta[sp] = {"model": args.retriever, "pos": "ret", "layer": Er["layers"][li], "frac": Er["layers"][li] / L, "X": Xr}
        ragree = D.RG[:, D.j(args.retriever, "l8", COT, PICK, "all", 3, 5)]
        sp = f"{name} (retriever): agreement of its 3 tries (the Set 4 signal, for comparison)"
        run_space(sp, ragree[:, None], tg, ["linear probe"], "two more retrieval tries: about 2.2x")
        log("  retriever done")
    key_sep = "Separation on held-out questions (0.5 = useless)"
    assert rows, "nothing was extracted - run the extract stages first"

    # ------------------------------------------------------------------ gate: what a signal would buy
    gate_rows, curve_rows = [], []
    for key, (E, em) in single.items():
        name = W.LABEL[key]
        tg = f"{name}'s single answer is no worse than the big answer (routing target)"
        have = E["have"]
        pt = np.array([r["prompt_tokens"] if r else np.nan for r in E["rows"]], float)
        ot = np.array([r["out_tokens"] if r else np.nan for r in E["rows"]], float)
        cheap = ret_fl + 2 * P[key] * (pt + ot)
        read_only = ret_fl + 2 * P[key] * pt
        idx = np.where(have)[0]
        base = [("always the cheap answerer, one answer", np.zeros(len(idx), bool), False), ("always the big answerer = the cheap fixed setup", None, False)]
        gate_rows.append({"Answerer": name, "Gate": "always the cheap answerer, one answer", "Share escalated": 0.0, "Accuracy": float(em[idx].mean()),
                          "Compute (x cheapest)": float(cheap[idx].mean() / D.ref), "Separation": float("nan")})
        gate_rows.append({"Answerer": name, "Gate": "always the big answerer = the cheap fixed setup", "Share escalated": 1.0,
                          "Accuracy": float(D.EM[idx, j_big].mean()), "Compute (x cheapest)": float(D.FL[idx, j_big].mean() / D.ref), "Separation": float("nan")})
        cands = [r for r in rows if r["What is predicted"] == tg and r[key_sep] == r[key_sep]]
        best = {}
        for r in cands:                                         # best reading of each kind of signal
            sp = r["What is looked at"]
            kind = re.sub(r" \(layer \d+ of \d+\)", "", sp)
            if kind not in best or r[key_sep] > best[kind][key_sep]:
                best[kind] = r
        others = {k2: 2 * P[k2] * np.array([(x["prompt_tokens"] + x["out_tokens"]) if x else np.nan for x in E2["rows"]], float)
                  for k2, (E2, _) in single.items() if k2 != key}
        for kind, r in sorted(best.items(), key=lambda kv: -kv[1][key_sep]):
            p = oof[(r["What is looked at"], r["How it is read"], tg)]
            before = "before answering" in kind
            extra = np.zeros(len(pt))
            if "Set 4 signal" in kind:
                extra = 4 * 2 * P[key] * (pt + ot)                 # the agreement signal needs four more cheap answers
            elif "equals" in kind and others:
                extra = np.nan_to_num(list(others.values())[0])    # the other small model's single answer
            cheap_k, read_k = cheap + extra, read_only + extra
            for share in np.linspace(0.05, 0.95, 19):
                thr = np.nanquantile(p[idx], share)
                esc = p[idx] <= thr
                acc = float(np.where(esc, D.EM[idx, j_big], em[idx]).mean())
                co = np.where(esc, (read_k[idx] if before else cheap_k[idx]) + big_tail[idx], cheap_k[idx])
                rr = {"Answerer": name, "Gate": f"{kind} | {r['How it is read']}", "Share escalated": float(esc.mean()), "Accuracy": acc,
                      "Compute (x cheapest)": float(co.mean() / D.ref), "Separation": r[key_sep], "Beats shuffled labels": r["Beats shuffled labels"],
                      "Accuracy of a random gate with the same share": float(esc.mean() * D.EM[idx, j_big].mean() + (1 - esc.mean()) * em[idx].mean())}
                curve_rows.append(rr)
                if (abs(share - 0.35) < 1e-9 or abs(share - 0.5) < 1e-9) and abs(esc.mean() - share) < 0.1:
                    gate_rows.append(rr)
    for r in gate_rows:
        r["Accuracy vs the cheap fixed setup (points)"] = 100 * (r["Accuracy"] - fixed_acc)
        r["Compute vs the cheap fixed setup"] = r["Compute (x cheapest)"] / fixed_cost - 1
    wins = [r for r in curve_rows if r["Accuracy"] >= fixed_acc - 0.02 and r["Compute (x cheapest)"] <= 0.85 * fixed_cost]

    # ------------------------------------------------------------------ layer table
    for sp, m in spaces_meta.items():
        if "layer" not in m:
            continue
        for r in rows:
            if r["What is looked at"] == sp:
                layer_rows.append({"Model": W.LABEL[m["model"]], "Where": re.sub(r" \(layer \d+ of \d+\)", "", sp.split(": ", 1)[1]) if ": " in sp else sp,
                                   "Layer": m["layer"], "Depth (share of layers)": m["frac"], "How it is read": r["How it is read"],
                                   "What is predicted": r["What is predicted"], key_sep: r[key_sep], "Beats shuffled labels": r["Beats shuffled labels"]})

    # ------------------------------------------------------------------ signals handed to the rules (stage B)
    handed = []
    for key, (E, em) in single.items():
        name = W.LABEL[key]
        tg = f"{name}'s single answer is no worse than the big answer (routing target)"
        good = sorted([r for r in rows if r["What is predicted"] == tg and r["Beats shuffled labels"]], key=lambda r: -r[key_sep])
        seen, keep = set(), []
        for r in good:
            kind = re.sub(r" \(layer \d+ of \d+\)", "", r["What is looked at"]) + (" [SU probe]" if r["How it is read"] == SU else "")
            if kind not in seen and "Set 4 signal" not in kind:
                seen.add(kind)
                keep.append((kind, r))
        for i, q in enumerate(D.qids):
            if not E["have"][i]:
                continue
            row = {"qid": q, "answerer": key, "single answer": E["rows"][i]["prediction"], "stated confidence": E["rows"][i]["stated_conf"]}
            for kind, r in keep[:6]:
                v = oof[(r["What is looked at"], r["How it is read"], tg)][i]
                row[f"signal: {kind}"] = None if v != v else round(float(v), 3)
            handed.append(row)

    def dump(name, rr):
        with open(os.path.join(out, name), "w", encoding="utf-8") as f:
            for r in rr:
                f.write(json.dumps(r, ensure_ascii=False, default=float) + "\n")
    dump("scoreboard.jsonl", rows); dump("layers.jsonl", layer_rows); dump("gates.jsonl", gate_rows); dump("gate_curves.jsonl", curve_rows)
    dump("signals_for_rules.jsonl", handed)
    for key, g in gens.items():
        dump(f"single_answers_{key}.jsonl", g)
    charts(os.path.join(out, "charts"), D, rows, layer_rows, curve_rows, spaces_meta, single, fixed_acc, fixed_cost, key_sep, j_big, args.quick)

    # ------------------------------------------------------------------ summary
    pct = {"Accuracy", "Share escalated", "Compute vs the cheap fixed setup", "Accuracy of a random gate with the same share", "Share that is 'yes'"}
    nums = {key_sep, "Same method with shuffled labels: 95th percentile", "Separation", "Accuracy vs the cheap fixed setup (points)", "Depth (share of layers)"}
    xs = {"Compute (x cheapest)"}
    T = lambda rr, cols: R.md_table(rr, cols, pct, xs, nums)
    kinds = defaultdict(list)
    for r in rows:
        kinds[(re.sub(r" \(layer \d+ of \d+\)", "", r["What is looked at"]), r["What is predicted"])].append(r)
    best_kind = sorted([max(v, key=lambda r: r[key_sep] if r[key_sep] == r[key_sep] else -1) for v in kinds.values()], key=lambda r: -(r[key_sep] if r[key_sep] == r[key_sep] else -1))
    for r in best_kind:
        r["Signal"] = re.sub(r" \(layer \d+ of \d+\)", "", r["What is looked at"])
    routing = [r for r in best_kind if "routing target" in r["What is predicted"]]
    agree_t = [r for r in best_kind if "5 answers all agree" in r["What is predicted"]]
    retr = [r for r in best_kind if "(retriever)" in r["What is looked at"]]
    n_beat = sum(r["Beats shuffled labels"] for r in rows)
    acc_rows = []
    for key, (E, em) in single.items():
        idx = np.where(E["have"])[0]
        j1, j5 = D.j(RET, key, COT, PICK, MERGE, RK, 1), D.j(RET, key, COT, PICK, MERGE, RK, 5)
        acc_rows.append({"Answerer": W.LABEL[key], "Questions": len(idx), "Accuracy": float(em[idx].mean()),
                         "Set 4 accuracy, 1 answer (sampled, with fusion)": float(D.EM[idx, j1].mean()),
                         "Set 4 accuracy, 5 answers + fusion": float(D.EM[idx, j5].mean()), "Big answerer on the same questions": float(D.EM[idx, j_big].mean()),
                         "Mean answer tokens": float(np.mean([r["out_tokens"] for r in E["rows"] if r])),
                         "Answers cut off at the length cap": float(np.mean([r["hit_cap"] for r in E["rows"] if r])),
                         "Attention measured": float(np.mean([r["attention_measured"] for r in E["rows"] if r]))})
    pct |= {"Set 4 accuracy, 1 answer (sampled, with fusion)", "Set 4 accuracy, 5 answers + fusion", "Big answerer on the same questions",
            "Answers cut off at the length cap", "Attention measured"}
    cols = ["Signal", "How it is read", key_sep, "Same method with shuffled labels: 95th percentile", "Beats shuffled labels", "Extra cost of having this signal"]
    L = ["# Set 5d part 2 - hidden signals from one cheap generation\n", f"Analysis code version {VERSION}.\n",
         f"Sweep used: `{os.path.basename(args.sweep_run.rstrip('/'))}`. Shortlist: {W.LABEL[RET]} retrieves once (CoT, per-attempt shortlist {PICK}), as in the cheap fixed setup. "
         f"The cheap answerer answers once, greedily. The bar is the cheap fixed setup ({W.LABEL[RET]} retrieves once, {W.LABEL[BIG]} answers once): "
         f"**{100 * fixed_acc:.1f}% at {fixed_cost:.1f}x**.",
         f"Every separation is on held-out questions (5-fold cross-validation x{reps}); 0.5 = useless. 'Beats shuffled labels' = above the 95th percentile of "
         f"the same method on {perms} label shuffles. {n_beat} of {len(rows)} combinations beat shuffled labels; chance alone would give about {round(0.05 * len(rows))}.",
         "This is stage A (is each signal informative?). The informative signals are saved in `results/signals_for_rules.jsonl` for stage B, "
         "where the planner writes rules over them.\n",
         "## 1. The single cheap answers\n",
         T(acc_rows, ["Answerer", "Questions", "Accuracy", "Set 4 accuracy, 1 answer (sampled, with fusion)", "Set 4 accuracy, 5 answers + fusion",
                      "Big answerer on the same questions", "Mean answer tokens", "Answers cut off at the length cap", "Attention measured"]),
         "\n## 2. Which signals predict the routing target? (best layer and reading of each signal)\n",
         "Picking the best layer after the fact flatters the hidden-state rows a little; section 5 shows every layer.\n",
         T(routing, ["Signal", "How it is read", key_sep, "Same method with shuffled labels: 95th percentile", "Beats shuffled labels",
                     "Extra cost of having this signal"]),
         "\n## 3. Semantic uncertainty: can one generation's hidden states predict whether 5 answers would agree?\n", T(agree_t, cols),
         "\n## 4. What a signal would buy as a gate in front of the big answerer\n",
         "The gate escalates the questions the signal is least sure about. For signals read before answering, an escalated question skips the cheap answer.\n",
         T(sorted(gate_rows, key=lambda r: (r["Answerer"], not r["Gate"].startswith("always"),
                                            -(r["Separation"] if r.get("Separation") == r.get("Separation") else 0), r["Share escalated"])),
           ["Answerer", "Gate", "Share escalated", "Accuracy", "Compute (x cheapest)", "Accuracy vs the cheap fixed setup (points)",
            "Compute vs the cheap fixed setup", "Accuracy of a random gate with the same share"]),
         f"\n**Any point within 2 points of the cheap fixed setup at 15% or more less compute?** "
         + (f"Yes, {len(wins)} points on the curves (in-sample over the escalation share, so provisional): "
            + "; ".join(sorted({w['Gate'] + ' [' + w['Answerer'] + ']' for w in wins})[:6]) + "." if wins else "No."),
         "\n## 5. Layer by layer (linear probe, routing target)\n",
         T([r for r in layer_rows if r["How it is read"] == "linear probe" and "routing target" in r["What is predicted"]],
           ["Model", "Where", "Layer", "Depth (share of layers)", key_sep, "Beats shuffled labels"]),
         "\n## 6. Retriever-side probe\n", T(retr, cols + ["What is predicted"]) if retr else "Not extracted.",
         "\n## 7. Top 20 combinations overall\n",
         T(sorted([r for r in rows if r[key_sep] == r[key_sep]], key=lambda r: -r[key_sep])[:20],
           ["What is looked at", "How it is read", "What is predicted", key_sep, "Same method with shuffled labels: 95th percentile", "Beats shuffled labels"]),
         "\nCharts: `results/charts/`. Excel: `results/set5d_hidden.xlsx`. Every single answer: `results/single_answers_<model>.jsonl`."]
    open(os.path.join(out, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    def xl(rr):
        d = pd.DataFrame(rr)
        for c in d.columns:
            if d[c].isna().all():
                d[c] = ""                                       # an all-empty column breaks the column-width code
            elif d[c].map(lambda v: isinstance(v, (list, dict))).any():
                d[c] = d[c].astype(str)
        return d
    W.write_excel(os.path.join(out, "set5d_hidden.xlsx"), {
        "Scoreboard": xl(rows), "Layers": xl(layer_rows), "Gates": xl(gate_rows), "Gate curves": xl(curve_rows),
        "Single answers (accuracy)": xl(acc_rows), "Signals for rules": xl(handed),
        **{f"Answers {W.LABEL[k]}": xl(g) for k, g in gens.items()}})
    json.dump({"version": VERSION, "seconds": round(time.time() - t0), "combinations": len(rows), "beating shuffled labels": int(n_beat),
               "cheap fixed setup": {"accuracy": fixed_acc, "compute_x": fixed_cost}, "gate points meeting the test (provisional)": len(wins),
               "answerers": list(single), "retriever": args.retriever if Er is not None else None},
              open(os.path.join(run, "analyze_status.json"), "w"), indent=1)
    log(f"done in {time.time() - t0:.0f}s -> {out}")


# ==================================================================================
# Charts
# ==================================================================================
def charts(out, D, rows, layer_rows, curve_rows, meta, single, fixed_acc, fixed_cost, key_sep, j_big, quick):
    plt = S._plt()
    os.makedirs(out, exist_ok=True)
    # 1. layer curves
    g = [r for r in layer_rows if r["How it is read"] == "linear probe"]
    combos = list(dict.fromkeys((r["Model"], r["What is predicted"]) for r in g))
    if combos:
        ncol = min(3, len(combos))
        nrow = int(np.ceil(len(combos) / ncol))
        fig, axs = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 3.9 * nrow), squeeze=False)
        for ax in axs.ravel()[len(combos):]:
            ax.axis("off")
        for ax, (mdl, tg) in zip(axs.ravel(), combos):
            for i, wh in enumerate(dict.fromkeys(r["Where"] for r in g if r["Model"] == mdl and r["What is predicted"] == tg)):
                h = sorted([r for r in g if r["Model"] == mdl and r["What is predicted"] == tg and r["Where"] == wh], key=lambda r: r["Layer"])
                ax.plot([r["Depth (share of layers)"] for r in h], [r[key_sep] for r in h], marker="o", color=S.SERIES[i % len(S.SERIES)], label=wh, lw=1.6)
                for r in h:
                    if r["Beats shuffled labels"]:
                        ax.scatter([r["Depth (share of layers)"]], [r[key_sep]], s=90, facecolor="none", edgecolor=S.SERIES[i % len(S.SERIES)], zorder=4)
            ax.axhline(0.5, color=S.INK2, ls=":", lw=1)
            ax.set_ylim(0.3, 1.0)
            ax.set_title(tg, fontsize=8)
            ax.set_xlabel("depth in the model (share of layers)")
            ax.set_ylabel("held-out separation")
            ax.legend(frameon=False, fontsize=7)
        fig.suptitle("Where in the model is the signal? Linear probe, layer by layer (ringed = beats shuffled labels)", fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "1_layers.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 2. signals scoreboard (best reading per signal kind x target)
    kinds = defaultdict(dict)
    for r in rows:
        k = re.sub(r" \(layer \d+ of \d+\)", "", r["What is looked at"])
        t = r["What is predicted"]
        if r[key_sep] == r[key_sep] and (t not in kinds[k] or r[key_sep] > kinds[k][t][0]):
            kinds[k][t] = (r[key_sep], r["Beats shuffled labels"])
    for tag, sel in (("answerers", lambda k: "(retriever)" not in k), ("retriever", lambda k: "(retriever)" in k)):
        ks = [k for k in kinds if sel(k)]
        ts = list(dict.fromkeys(t for k in ks for t in kinds[k]))
        if not ks:
            continue
        M = np.full((len(ks), len(ts)), np.nan)
        fig, ax = plt.subplots(figsize=(2.3 * len(ts) + 6, 0.42 * len(ks) + 2.4))
        for i, k in enumerate(ks):
            for j, t in enumerate(ts):
                if t in kinds[k]:
                    M[i, j] = kinds[k][t][0]
        im = ax.imshow(M, cmap="RdBu_r", vmin=0.2, vmax=0.8, aspect="auto")
        for i, k in enumerate(ks):
            for j, t in enumerate(ts):
                if t in kinds[k]:
                    ax.text(j, i, f"{M[i, j]:.2f}" + ("*" if kinds[k][t][1] else ""), ha="center", va="center", fontsize=8,
                            color="white" if abs(M[i, j] - 0.5) > 0.2 else "black")
        ax.set_xticks(range(len(ts)))
        ax.set_xticklabels([t.replace(" is ", "\nis ").replace(" (routing", "\n(routing").replace(" all agree", "\nall agree").replace(" finds", "\nfinds") for t in ts], fontsize=7)
        ax.set_yticks(range(len(ks)))
        ax.set_yticklabels(ks, fontsize=7)
        ax.grid(False)
        ax.set_title(f"Hidden signals ({tag}): best held-out separation per signal (0.5 = useless; * = beats shuffled labels)", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        fig.savefig(os.path.join(out, f"2_scoreboard_{tag}.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 3 + 4. PCA and t-SNE maps of hidden states
    from sklearn.decomposition import PCA
    from sklearn.impute import SimpleImputer
    from sklearn.manifold import TSNE
    panels = []
    for key, (E, em) in single.items():
        L = max(E["layers"])
        mid = min(range(len(E["layers"])), key=lambda i: abs(E["layers"][i] / L - 0.75))
        for pos, lab in (("h_prompt", "before answering"), ("h_answer", "over the answer")):
            panels.append((f"{W.LABEL[key]}, {lab}, layer {E['layers'][mid]}", E[pos][:, mid, :], em, (em >= D.EM[:, j_big]).astype(float), E["have"]))
    if panels:
        for fname, kind in (("3_pca_maps.png", "PCA"), ("4_tsne_maps.png", "t-SNE")):
            fig, axs = plt.subplots(2, len(panels), figsize=(4.0 * len(panels), 7.6), squeeze=False)
            for jx, (title, X, em, rt, have) in enumerate(panels):
                Xh = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X[have])
                Xh = (Xh - Xh.mean(0)) / (Xh.std(0) + 1e-6)
                Z = PCA(2, random_state=0).fit_transform(Xh)
                if kind == "t-SNE":
                    Z = TSNE(2, perplexity=min(20, max(5, len(Xh) // 6)), init="pca", random_state=0, max_iter=500 if quick else 1000).fit_transform(
                        PCA(min(30, len(Xh) - 1, Xh.shape[1]), random_state=0).fit_transform(Xh))
                for ix, (y, lab) in enumerate(((em[have], "single answer is right"), (rt[have], "no worse than the big answer"))):
                    ax = axs[ix, jx]
                    ax.scatter(Z[y == 1, 0], Z[y == 1, 1], s=14, color=S.SERIES[0], alpha=0.75, label="yes")
                    ax.scatter(Z[y == 0, 0], Z[y == 0, 1], s=14, color=S.SERIES[1], alpha=0.75, label="no")
                    ax.set_title(f"{title}\ncoloured by: {lab}", fontsize=8)
                    ax.set_xticks([]); ax.set_yticks([])
                    if ix == 0 and jx == 0:
                        ax.legend(frameon=False, fontsize=8)
            fig.suptitle(f"{kind} maps of the hidden states. A picture is not evidence: the held-out numbers are in the scoreboard.", fontsize=10)
            fig.tight_layout()
            fig.savefig(os.path.join(out, fname), dpi=150, bbox_inches="tight")
            plt.close(fig)
    # 5. gate curves
    ans = list(dict.fromkeys(r["Answerer"] for r in curve_rows))
    if ans:
        fig, axs = plt.subplots(1, len(ans), figsize=(7.4 * len(ans), 5.0), squeeze=False)
        for ax, a in zip(axs[0], ans):
            gs = list(dict.fromkeys(r["Gate"] for r in curve_rows if r["Answerer"] == a))
            top = sorted(gs, key=lambda gname: -max(r["Separation"] for r in curve_rows if r["Answerer"] == a and r["Gate"] == gname))[:7]
            for i, gname in enumerate(top):
                h = sorted([r for r in curve_rows if r["Answerer"] == a and r["Gate"] == gname], key=lambda r: r["Compute (x cheapest)"])
                ax.plot([r["Compute (x cheapest)"] for r in h], [r["Accuracy"] for r in h], color=S.SERIES[i % len(S.SERIES)], lw=1.5,
                        label=f"{gname.split(' | ')[0].split(': ', 1)[-1]} ({h[0]['Separation']:.2f})")
            ax.scatter([fixed_cost], [fixed_acc], marker="D", s=90, color=S.INK2, zorder=5, label="cheap fixed setup")
            ax.axhline(fixed_acc - 0.02, color=S.INK2, ls=":", lw=1)
            ax.axvline(0.85 * fixed_cost, color=S.INK2, ls=":", lw=1)
            ax.set_xlabel("compute (x cheapest)")
            ax.set_ylabel("accuracy")
            ax.set_title(f"{a}: one cheap answer, then a gate in front of the big answerer\n(dotted = the success test: left of the vertical line and above the horizontal one)", fontsize=9)
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "5_gate_curves.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)
    # 6. attention: right vs wrong
    att_sp = [k for k in meta if "evidence-use attention" in k]
    if att_sp:
        fig, axs = plt.subplots(1, len(att_sp), figsize=(7.2 * len(att_sp), 4.6), squeeze=False)
        for ax, sp in zip(axs[0], att_sp):
            key = meta[sp]["model"]
            E, em = single[key]
            X, names = meta[sp]["X"][:, :len(ATT_NAMES)], ATT_NAMES
            y = np.arange(len(names))
            for k, (mk, colr, lab) in enumerate(((em == 1, S.SERIES[0], "answer right"), (em == 0, S.SERIES[1], "answer wrong"))):
                ax.barh(y + (k - 0.5) * 0.38, np.nanmean(X[mk], 0) if mk.any() else np.zeros(len(names)), 0.38, color=colr, label=lab)
            ax.set_yticks(y)
            ax.set_yticklabels(names, fontsize=8)
            ax.invert_yaxis()
            ax.set_title(f"{W.LABEL[key]}: where the answer tokens look (mean over questions)", fontsize=9)
            ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "6_attention.png"), dpi=150, bbox_inches="tight")
        plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("extract", "analyze"):
        p = sub.add_parser(name)
        p.add_argument("--sweep_run", required=True)
        p.add_argument("--data_dir", required=True)
        p.add_argument("--run_dir", required=True)
        p.add_argument("--models_json", required=True)
        if name == "extract":
            p.add_argument("--model", required=True)
            p.add_argument("--role", choices=["answer", "retrieve"], required=True)
            p.add_argument("--backend", default="hf", choices=["hf", "tiny"])
            p.add_argument("--limit", type=int, default=0)
            p.add_argument("--time_limit_min", type=float, default=0)
            p.add_argument("--max_new", type=int, default=MAX_NEW)
        else:
            p.add_argument("--answerers", nargs="+", default=["l8", "l3"])
            p.add_argument("--retriever", default="q7")
            p.add_argument("--repeats", type=int, default=3)
            p.add_argument("--permutations", type=int, default=20)
            p.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    {"extract": cmd_extract, "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    main()
