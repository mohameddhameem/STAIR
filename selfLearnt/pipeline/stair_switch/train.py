"""Teach a retriever to shortlist (LoRA, loss on the target line only), keep the checkpoint best on validation, merge it.

Same recipe as the development notebooks 22-26: LoRA r=16, alpha=32, dropout 0.05, lr 1e-4 with warm-up and linear decay,
8 questions per optimizer step (micro-batches of 4), target 'Selected references: [a] [b]', thinking off for Qwen3/3.5.
Resumable: a resume point every `save_every_questions`, validation checkpoints at `ckpt_questions`."""
import json
import os
import random
import shutil
import sys
import time

import numpy as np

from .common import RET_SYSTEM, direct_prompt, free_gpu, log, picks_of, selection_text, target_text


def _no_torchao():
    """Some images ship an old torchao that peft refuses; plain LoRA never needs it."""
    import peft  # noqa: F401
    for name, mod in list(sys.modules.items()):
        if name.startswith("peft") and hasattr(mod, "is_torchao_available"):
            setattr(mod, "is_torchao_available", lambda: False)


def chat_ids(tok, user, answer=None):
    msgs = [{"role": "system", "content": RET_SYSTEM}, {"role": "user", "content": user}]
    try:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        p = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    pid = tok(p, add_special_tokens=False)["input_ids"]
    if answer is None:
        return pid, None
    ans = tok(answer + (tok.eos_token or ""), add_special_tokens=False)["input_ids"]
    return list(pid) + list(ans), len(pid)


def load_lm(path):
    import torch
    import transformers as T
    errs = []
    for cls in ("AutoModelForCausalLM", "AutoModelForImageTextToText", "AutoModelForMultimodalLM", "AutoModelForVision2Seq"):
        if not hasattr(T, cls):
            continue
        for kw in ({"dtype": torch.bfloat16}, {"torch_dtype": torch.bfloat16}):
            try:
                return getattr(T, cls).from_pretrained(path, **kw)
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


def val_score(model, tok, qs, max_new=24):
    import torch
    model.eval()
    both = exact = 0
    sizes = []
    dev = next(model.parameters()).device
    with torch.no_grad():
        for q in qs:
            pid, _ = chat_ids(tok, direct_prompt(q))
            out = model.generate(torch.tensor([pid], device=dev), max_new_tokens=max_new, do_sample=False, repetition_penalty=1.0, use_cache=True,
                                 pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id)
            text = tok.decode(out[0, len(pid):], skip_special_tokens=True)
            sel = {i - 1 for i in picks_of(selection_text(text), len(q["ctxs_candidate"]))}
            g = set(q["ret_answers"])
            both += g <= sel
            exact += g == sel
            sizes.append(len(sel))
    model.train()
    return {"both gold passages found": both / len(qs), "exactly the gold passages": exact / len(qs), "mean shortlist size": float(np.mean(sizes))}


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
    cfg = LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=4,
                      max_position_embeddings=8192, bos_token_id=tok.bos_token_id, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                      attn_implementation="eager")
    return LlamaForCausalLM(cfg).eval(), tok


def trained(r):
    s = os.path.join(r["adapter_dir"], "train_state.json")
    return os.path.exists(s) and "chosen" in json.load(open(s))


def train(cfg, r, train_qs, val_qs, time_limit_min=0):
    """r: retriever entry (base path, adapter_dir, merged_dir, train settings)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    _no_torchao()
    t0 = time.time()
    T = r["train"]
    out = r["adapter_dir"]
    os.makedirs(out, exist_ok=True)
    tiny = cfg["backend"] == "mock"
    tr = train_qs[:T["n_train"]]
    va = val_qs[:T.get("n_val", 300)]
    if tiny:
        model, tok = _tiny_model([direct_prompt(q) + " " + target_text(q) for q in tr[:200] + va] + [RET_SYSTEM])
        dev = "cpu"
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(r["base_path"])
        model = load_lm(r["base_path"])
        dev = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(dev)
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    model.config.use_cache = False
    ex = []
    for q in tr:
        ids, n_prompt = chat_ids(tok, direct_prompt(q), target_text(q))
        if len(ids) <= T.get("max_len", 4096 if not tiny else 9000):
            ex.append((ids, n_prompt))
    random.Random(T.get("seed", 0)).shuffle(ex)
    ga, mb = T.get("grad_accum", 8), T.get("micro_batch", 4)
    steps_total = (len(ex) * T["epochs"]) // ga
    marks = sorted({min(steps_total, max(1, q // ga)) for q in T["ckpt_questions"]})
    save_every = max(1, T.get("save_every_questions", 1000) // ga)
    state_p = os.path.join(out, "train_state.json")
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"step": 0, "curve": [], "seconds": 0.0}
    last = os.path.join(out, f"ckpt_step{st['step']}") if st["step"] else None
    if last and os.path.isdir(last):
        model = PeftModel.from_pretrained(model, last, is_trainable=True)
        log(f"resuming from {last}")
    else:
        model = get_peft_model(model, LoraConfig(r=T.get("r", 16), lora_alpha=T.get("alpha", 32), lora_dropout=T.get("dropout", 0.05), task_type="CAUSAL_LM",
                                                 target_modules=r"^(?!.*(visual|vision)).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"))
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    log(f"{r['key']}: {len(ex)} examples x {T['epochs']} passes = {steps_total} optimizer steps; validation at steps {marks}")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=T.get("lr", 1e-4), weight_decay=0.0)
    if st["step"] and os.path.exists(os.path.join(last, "optimizer.pt")):
        opt.load_state_dict(torch.load(os.path.join(last, "optimizer.pt"), map_location=dev))
    warm = max(1, int(0.03 * steps_total))
    lr0 = T.get("lr", 1e-4)
    lr_at = lambda s: lr0 * min(1.0, (s + 1) / warm) * max(0.0, 1 - max(0, s - warm) / max(1, steps_total - warm))
    order = [i for _ in range(T["epochs"]) for i in range(len(ex))]
    body, head = lm_parts(model.get_base_model())
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ce = torch.nn.CrossEntropyLoss()
    model.train()
    step, t1, losses = st["step"], time.time(), []
    pos = step * ga

    def batch_loss(items):
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
        return ce(head(h[:, :-1][sel]).float(), lab[:, 1:][sel])

    def save(step_, keep):
        d = os.path.join(out, f"ckpt_step{step_}")
        model.save_pretrained(d)
        torch.save(opt.state_dict(), os.path.join(d, "optimizer.pt"))
        old = st.get("resume_only")
        if old and old != step_ and old not in marks:
            shutil.rmtree(os.path.join(out, f"ckpt_step{old}"), ignore_errors=True)
        st["resume_only"] = None if keep else step_
        st["step"] = step_

    while step < steps_total:
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
        if step % 25 == 0:
            log(f"  step {step}/{steps_total} | loss {np.mean(losses[-50:]):.3f}")
        if step in marks:
            save(step, keep=True)
            v = val_score(model, tok, va)
            st["curve"].append({"step": step, "training questions seen": step * ga, "training loss (recent)": float(np.mean(losses[-50:])), **v})
            log(f"checkpoint {step}: validation both-found {100 * v['both gold passages found']:.1f}%")
        elif step % save_every == 0:
            save(step, keep=False)
        if step in marks or step % save_every == 0:
            st["seconds"] += time.time() - t1
            t1 = time.time()
            json.dump(st, open(state_p, "w"), indent=1)
            if time_limit_min and (time.time() - t0) / 60 > time_limit_min and step < steps_total:
                log("time limit reached - submit the job again to continue")
                return False
    best = max(st["curve"], key=lambda x: (x["both gold passages found"], x["exactly the gold passages"], x["step"]))
    st["chosen"] = best
    json.dump(st, open(state_p, "w"), indent=1)
    log(f"{r['key']}: chosen checkpoint step {best['step']} ({100 * best['both gold passages found']:.1f}% both found on validation)")
    del model, opt
    free_gpu()
    return True


def merge(cfg, r):
    if os.path.exists(os.path.join(r["merged_dir"], "config.json")):
        return
    st = json.load(open(os.path.join(r["adapter_dir"], "train_state.json")))
    os.makedirs(r["merged_dir"], exist_ok=True)
    if cfg["backend"] == "mock":
        json.dump({"mock": True}, open(os.path.join(r["merged_dir"], "config.json"), "w"))
        return
    from peft import PeftModel
    from transformers import AutoTokenizer
    _no_torchao()
    m = PeftModel.from_pretrained(load_lm(r["base_path"]), os.path.join(r["adapter_dir"], f"ckpt_step{st['chosen']['step']}")).merge_and_unload()
    m.save_pretrained(r["merged_dir"])
    AutoTokenizer.from_pretrained(r["base_path"]).save_pretrained(r["merged_dir"])
    copy_side_files(r["base_path"], r["merged_dir"])
    log(f"merged {r['key']} -> {r['merged_dir']}")
