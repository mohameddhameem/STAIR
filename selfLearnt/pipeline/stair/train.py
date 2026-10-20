"""Fine-tuning by backpropagation: retrievers learn to shortlist the gold passages, answerers learn to answer from shortlisted
passages. LoRA on the bf16 model (up to 8B) or QLoRA (4-bit base + LoRA; the 32B, 70B and 72B models).

Recipe (development notebooks 22-26): LoRA r=16, alpha=32, dropout 0.05, lr 1e-4, 3% warm-up then linear decay, 8 questions
per optimizer step, loss on the target only, thinking off for Qwen3/3.5. The validation checkpoint with the best score is kept:
retrievers - both gold passages in the shortlist; answerers - corrected-label accuracy. Resumable at every save point."""
import json
import os
import random
import shutil
import sys
import time

import numpy as np

from .common import (QA_SYSTEM, RET_SYSTEM, ans_target, answer_of, direct_qa_prompt, direct_ret_prompt, scores, log, picks_of, ret_target,
                     selection_text)
from .config import ft_settings
from .engines import free_gpu
from .progress import Tracker, mark_done


def _no_torchao():
    import peft  # noqa: F401
    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and hasattr(mod, "is_torchao_available"):
            setattr(mod, "is_torchao_available", lambda: False)


def chat_ids(tok, system, user, answer=None):
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = list(tok(p, add_special_tokens=False)["input_ids"])
    if answer is None:
        return pid, None
    ans = list(tok(answer + (tok.eos_token or ""), add_special_tokens=False)["input_ids"])
    return pid + ans, len(pid)


# ----------------------------------------------------------------------------------
# training examples
# ----------------------------------------------------------------------------------
def fit_retrieval(tok, ex, max_len, rng):
    """Retrieval example within max_len tokens: drop random distractors (never a gold passage) until it fits."""
    e = dict(ex)
    ids, n = chat_ids(tok, RET_SYSTEM, direct_ret_prompt(e), ret_target(e))
    while len(ids) > max_len and len(e["ctxs_candidate"]) > len(e["ret_answers"]) + 2:
        gold = [e["ctxs_candidate"][i] for i in e["ret_answers"]]
        others = [c for c in e["ctxs_candidate"] if c not in gold]
        per = len(ids) / len(e["ctxs_candidate"])
        drop = max(1, int((len(ids) - max_len) / max(per, 1)) + 1)
        keep = rng.sample(others, max(2, len(others) - drop))
        cands = gold + keep
        rng.shuffle(cands)
        e = dict(e, ctxs_candidate=cands, ret_answers=[cands.index(g) for g in gold])
        ids, n = chat_ids(tok, RET_SYSTEM, direct_ret_prompt(e), ret_target(e))
    return ids, n


def answer_refs(ex, rng, settings):
    gold = [ex["ctxs_candidate"][i] for i in ex["ret_answers"]]
    others = [c for i, c in enumerate(ex["ctxs_candidate"]) if i not in ex["ret_answers"]]
    lo, hi = settings.get("distractors", [0, 2])
    refs = list(gold)
    if len(refs) > 1 and rng.random() < settings.get("drop_gold_share", 0.0):
        refs.pop(rng.randrange(len(refs)))
    refs += rng.sample(others, min(len(others), rng.randint(lo, hi)))
    rng.shuffle(refs)
    return refs


def examples(cfg, role, tok, qs, T, seed=0):
    rng = random.Random(seed)
    out = []
    for q in qs:
        if role == "retriever":
            ids, n = fit_retrieval(tok, q, T["max_len"], rng)
        else:
            ids, n = chat_ids(tok, QA_SYSTEM, direct_qa_prompt(q["question"], answer_refs(q, rng, cfg["finetune"]["answer_inputs"])), ans_target(q))
        if len(ids) <= T["max_len"]:
            out.append((ids, n))
    return out


def val_score(model, tok, role, qs, cfg, max_new=24):
    import torch
    model.eval()
    dev = next(model.parameters()).device
    rng = random.Random(123)
    good = exact = 0
    with torch.no_grad():
        for q in qs:
            if role == "retriever":
                pid, _ = chat_ids(tok, RET_SYSTEM, direct_ret_prompt(q))
            else:
                pid, _ = chat_ids(tok, QA_SYSTEM, direct_qa_prompt(q["question"], answer_refs(q, rng, cfg["finetune"]["answer_inputs"])))
            out = model.generate(torch.tensor([pid], device=dev), max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, use_cache=True,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            text = tok.decode(out[0, len(pid):], skip_special_tokens=True)
            if role == "retriever":
                sel = {i - 1 for i in picks_of(selection_text(text), len(q["ctxs_candidate"]))}
                g = set(q["ret_answers"])
                good += g <= sel
                exact += g == sel
            else:
                good += scores(answer_of(text), q["answer"])["acc"]
    model.train()
    if role == "retriever":
        return {"score": good / len(qs), "both gold passages found": good / len(qs), "exactly the gold passages": exact / len(qs)}
    return {"score": good / len(qs), "answer accuracy (corrected label)": good / len(qs)}


# ----------------------------------------------------------------------------------
def load_lm(path, qlora=False):
    import torch
    import transformers as T
    kw = {}
    if qlora:
        kw["quantization_config"] = T.BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                                         bnb_4bit_compute_dtype=torch.bfloat16)
        kw["device_map"] = "auto"
    errs = []
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText"):
        if not hasattr(T, cls):
            continue
        for dt in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
            try:
                return getattr(T, cls).from_pretrained(path, attn_implementation="sdpa", **dt, **kw)
            except TypeError:
                continue
            except (ValueError, KeyError, OSError) as e:
                errs.append(f"{cls}: {type(e).__name__}: {str(e)[:200]}")
                break
    raise RuntimeError("could not load " + path + "\n" + "\n".join(errs))


def lm_parts(cm):
    head = cm.get_output_embeddings() if hasattr(cm, "get_output_embeddings") else cm.lm_head
    body = cm.model
    if hasattr(body, "language_model"):
        body = body.language_model
    return body, head


def copy_side_files(src, dst):
    for f in os.listdir(src):
        if f.endswith((".safetensors", ".bin", ".pt", ".index.json")) or f.startswith(".") or os.path.isdir(os.path.join(src, f)):
            continue
        if not os.path.exists(os.path.join(dst, f)):
            shutil.copy(os.path.join(src, f), os.path.join(dst, f))


def _tiny_model(texts, seed=0):
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
    c = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                    max_position_embeddings=32768, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id)
    return LlamaForCausalLM(c).eval(), tok


def state_path(cfg, key):
    return os.path.join(cfg["run_dir"], "adapters", key, "train_state.json")


def trained(cfg, key):
    s = state_path(cfg, key)
    return os.path.exists(s) and "chosen" in json.load(open(s))


def train(cfg, key, train_qs, val_qs, time_limit_min=0):
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    _no_torchao()
    t0 = time.time()
    role, T = ft_settings(cfg, key)
    m = cfg["models"][key]
    qlora = T["method"] == "qlora"
    out = os.path.join(cfg["run_dir"], "adapters", key)
    os.makedirs(out, exist_ok=True)
    mock = cfg["backend"] == "mock"
    tr = train_qs[:T["n_train"]]
    va = val_qs
    if mock:
        texts = [direct_ret_prompt(q) + " " + ret_target(q) for q in tr[:100]] + [direct_qa_prompt(q["question"], q["ctxs_candidate"][:3]) + " "
                                                                                  + ans_target(q) for q in tr[:100] + va]
        model, tok = _tiny_model(texts + [RET_SYSTEM, QA_SYSTEM])
        dev = "cpu"
        T = dict(T, max_len=40000)
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(m["train_path"])
        model = load_lm(m["train_path"], qlora=qlora)
        if qlora:
            from peft import prepare_model_for_kbit_training
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False})
            dev = next(model.parameters()).device
        else:
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            model.to(dev)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
    model.config.use_cache = False
    run_seed = int(cfg.get("seed", 0))
    ex = examples(cfg, role, tok, tr, T, seed=run_seed)
    random.Random(run_seed).shuffle(ex)
    vex = examples(cfg, role, tok, va[:T.get("n_val_loss", 300)], T, seed=12345)      # fixed validation-loss examples
    vmet = va[:T.get("n_val_metric", 100)]
    ga, mb = T["grad_accum"], T["micro_batch"]
    steps_epoch = max(1, len(ex) // ga)
    max_epochs = int(T.get("max_epochs", 2))
    steps_total = steps_epoch * max_epochs
    eval_every = max(1, steps_epoch // int(T.get("evals_per_epoch", 2)))
    save_every = max(1, T.get("save_every_questions", 500) // ga)
    state_p = state_path(cfg, key)
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"step": 0, "curve": [], "seconds": 0.0, "best": None, "bad": 0}
    last = os.path.join(out, f"ckpt_step{st['step']}") if st["step"] else None
    if last and os.path.isdir(last):
        model = PeftModel.from_pretrained(model, last, is_trainable=True)
        log(f"resuming from {last}")
    else:
        model = get_peft_model(model, LoraConfig(r=T["r"], lora_alpha=T["alpha"], lora_dropout=T["dropout"], task_type="CAUSAL_LM",
                                                 target_modules=r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    log(f"{key} ({role}, {T['method']}, seed {run_seed}): {len(ex)} examples, up to {max_epochs} epochs = {steps_total} steps; "
        f"validation loss every {eval_every} steps on {len(vex)} val questions; stop after {T.get('patience', 2)} evaluations without improvement")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=T["lr"], weight_decay=0.0)
    if st["step"] and last and os.path.exists(os.path.join(last, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(last, "optimizer.pt"), map_location="cpu"))
    warm = max(1, int(0.03 * steps_total))
    lr_at = lambda s: T["lr"] * min(1.0, (s + 1) / warm) * max(0.0, 1 - max(0, s - warm) / max(1, steps_total - warm))
    order = [i for _ in range(max_epochs) for i in range(len(ex))]
    body, head = lm_parts(model.get_base_model())
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ce = torch.nn.CrossEntropyLoss()
    model.train()
    step, t1, losses = st["step"], time.time(), []
    pos = step * ga
    tracker = Tracker(cfg, "train", key, steps_total, unit="steps")

    def batch_loss(items):
        L = max(len(i) for i, _ in items)
        x = torch.full((len(items), L), pad, dtype=torch.long)
        am = torch.zeros((len(items), L), dtype=torch.long)
        lab = torch.full((len(items), L), -100, dtype=torch.long)
        for b, (ids, n_prompt) in enumerate(items):
            x[b, :len(ids)] = torch.tensor(ids)
            am[b, :len(ids)] = 1
            lab[b, n_prompt:len(ids)] = torch.tensor(ids[n_prompt:])
        x, am = x.to(dev), am.to(dev)
        # one unpadded example: no mask, so PyTorch can use its memory-saving causal attention kernel. With an explicit mask it
        # may fall back to building the full attention matrix (32B at 12k tokens: ~12 GB, out of memory on a 48 GB card)
        h = body(input_ids=x, attention_mask=None if len(items) == 1 else am).last_hidden_state
        lab = lab.to(h.device)
        sel = lab[:, 1:] != -100
        return ce(head(h[:, :-1][sel]).float(), lab[:, 1:][sel])

    def val_loss():
        model.eval()
        with torch.no_grad():
            vals = [float(batch_loss([e])) for e in vex]
        model.train()
        return float(np.mean(vals)) if vals else float("nan")

    def save(step_, name=None):
        d = os.path.join(out, name or f"ckpt_step{step_}")
        if os.path.exists(d):
            shutil.rmtree(d)
        model.save_pretrained(d)
        if name is None:
            torch.save(opt.state_dict(), os.path.join(d, "optimizer.pt"))
            old = st.get("resume_at")
            if old and old != step_:
                shutil.rmtree(os.path.join(out, f"ckpt_step{old}"), ignore_errors=True)
            st["resume_at"] = st["step"] = step_

    stopped = False
    while step < steps_total and not stopped:
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = order[pos:pos + ga]
        for c in range(0, len(idx), mb):
            part = [ex[i] for i in idx[c:c + mb]]
            loss = batch_loss(part) * len(part) / len(idx)
            loss.backward()
            losses.append(float(loss) * len(idx) / len(part))
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        pos += ga
        tracker.tick(step, note=f"loss {np.mean(losses[-50:]):.3f}")
        if step % 25 == 0:
            log(f"  step {step}/{steps_total} | loss {np.mean(losses[-50:]):.3f}")
        if step % eval_every == 0 or step == steps_total:
            vl = val_loss()
            vm = val_score(model, tok, role, vmet, cfg)["score"] if vmet else None
            rec = {"step": step, "epoch": round(step / steps_epoch, 3), "train_loss": float(np.mean(losses)) if losses else float("nan"),
                   "val_loss": vl, "val_metric": vm, "training questions seen": step * ga}
            st["curve"].append(rec)
            losses = []
            if st["best"] is None or vl < st["best"]["val_loss"] - 1e-6:
                st["best"], st["bad"] = rec, 0
                save(step, name="best")
            else:
                st["bad"] += 1
            log(f"evaluation at epoch {rec['epoch']}: val loss {vl:.4f}" + (f", val score {100 * vm:.1f}%" if vm is not None else "")
                + (" (best)" if st["bad"] == 0 else f" (no improvement x{st['bad']})"))
            if st["bad"] >= int(T.get("patience", 2)):
                log(f"early stop: validation loss has not improved for {st['bad']} evaluations")
                stopped = True
        if step % save_every == 0 or step % eval_every == 0:
            save(step)
            st["seconds"] += time.time() - t1
            t1 = time.time()
            json.dump(st, open(state_p, "w"), indent=1)
            if time_limit_min and (time.time() - t0) / 60 > time_limit_min and step < steps_total and not stopped:
                log("time limit reached - submit the job again to continue")
                return False
    best = st["best"]
    st["chosen"] = dict(best, score=best.get("val_metric") or 0.0, stopped_early=stopped)
    st["role"], st["method"] = role, T["method"]
    chosen = os.path.join(out, "chosen")
    if os.path.exists(chosen):
        shutil.rmtree(chosen)
    shutil.copytree(os.path.join(out, "best"), chosen)
    json.dump(st, open(state_p, "w"), indent=1)
    write_history(cfg, key, role, T, st)
    tracker.finish(note=f"best epoch {best['epoch']}: val loss {best['val_loss']:.4f}")
    log(f"{key}: best epoch {best['epoch']} (val loss {best['val_loss']:.4f}){', stopped early' if stopped else ''}")
    del model, opt
    free_gpu()
    return True


def write_history(cfg, key, role, T, st):
    """The training curve in the team layout (stair_common.training), one CSV per model and seed."""
    from stair_common.training import from_records, validate_history
    m = cfg["models"][key]
    recs = [{"epoch": r["epoch"], "train_loss": r["train_loss"], "val_loss": r["val_loss"], "step": r["step"],
             **({"val_metric": r["val_metric"]} if r.get("val_metric") is not None else {})} for r in st["curve"]]
    df = from_records(recs, track="selfLearnt", model=f"{m['label']} {role}", run=f"{T['method']}-r{T['r']}", seed=int(cfg.get("seed", 0)))
    validate_history(df)
    d = os.path.join(cfg["paths"]["results_dir"], "history")
    os.makedirs(d, exist_ok=True)
    df.to_csv(os.path.join(d, f"selfLearnt-{key}-{role}-seed{cfg.get('seed', 0)}.csv"), index=False)


def merge(cfg, key):
    """LoRA-trained models are merged into a full copy for vLLM; QLoRA-trained ones are served as an adapter on the AWQ copy."""
    role, T = ft_settings(cfg, key)
    if T["method"] != "lora":
        mark_done(cfg, "train", key)
        return
    dst = os.path.join(cfg["run_dir"], "merged", key)
    if os.path.exists(os.path.join(dst, "config.json")):
        mark_done(cfg, "train", key)
        return
    os.makedirs(dst, exist_ok=True)
    if cfg["backend"] == "mock":
        json.dump({"mock": True}, open(os.path.join(dst, "config.json"), "w"))
        mark_done(cfg, "train", key)
        return
    from peft import PeftModel
    from transformers import AutoTokenizer
    _no_torchao()
    m = cfg["models"][key]
    merged = PeftModel.from_pretrained(load_lm(m["train_path"]), os.path.join(cfg["run_dir"], "adapters", key, "chosen")).merge_and_unload()
    merged.save_pretrained(dst)
    AutoTokenizer.from_pretrained(m["train_path"]).save_pretrained(dst)
    copy_side_files(m["train_path"], dst)
    log(f"merged {key} -> {dst}")
    mark_done(cfg, "train", key)
