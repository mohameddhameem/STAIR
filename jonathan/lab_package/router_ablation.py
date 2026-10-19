"""
Stage-Level Routing — training-method ablation on HotpotQA.

Eight arms, one dataset, one frozen split, reported on EM / cost / realised R /
% of headroom captured.

    always weak-weak          floor
    always strong-strong      best-fixed  <- the comparator C2's gate is defined against
    query-level router        RouteLLM granularity (weak-weak vs strong-strong only)
    SFT on argmax-R           imitate the winning configuration
    reward-weighted SFT       all four actions weighted by R (uses 100% of instances)
    GRPO from base            can the router learn from reward alone, cold?
    SFT -> GRPO               the proposed production recipe
    oracle                    ceiling

WHY THIS IS CHEAP: R is already known for every (instance, configuration) pair, so a
rollout is a short generation plus a dictionary lookup. No 7B inference, no retrieval,
no QA generation. Each arm is minutes-to-hours on the lab card.

SCOPE NOTE (state this in the write-up): the router here makes ONE decision over the 4
stage combinations. That is equivalent to joint per-stage routing for a 2-stage pipeline.
Sequential per-stage routing, where stage 2 conditions on stage 1's confidence, needs a
confidence signal we do not yet have (it comes from multi-sample retrieval) and is a
separate experiment.

Usage
-----
  python router_ablation.py prepare --combo combo_results_full.jsonl --data tts_test.json \
                                    --splits splits.json --out router_data.jsonl

  python router_ablation.py train --arm sft       --data router_data.jsonl --splits splits.json --out runs/sft
  python router_ablation.py train --arm rwsft     --data router_data.jsonl --splits splits.json --out runs/rwsft
  python router_ablation.py train --arm grpo      --data router_data.jsonl --splits splits.json --out runs/grpo
  python router_ablation.py train --arm sft2grpo  --data router_data.jsonl --splits splits.json --out runs/sft2grpo \
                                  --init runs/sft
  python router_ablation.py train --arm query     --data router_data.jsonl --splits splits.json --out runs/query

  python router_ablation.py eval --data router_data.jsonl --splits splits.json \
                                 --arms runs/sft runs/rwsft runs/grpo runs/sft2grpo runs/query \
                                 --out results.json
"""
import os, sys, json, argparse, random, math
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")

COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
LETTERS = ["A", "B", "C", "D"]          # one label token per combination
ROUTER_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

SYS = ("You route a two-stage question-answering pipeline. Each stage runs either a weak "
       "(cheap) or a strong (expensive) model. Choose the configuration that answers the "
       "question correctly at the lowest cost.")
INSTR = ("A = weak retrieval, weak QA\nB = weak retrieval, strong QA\n"
         "C = strong retrieval, weak QA\nD = strong retrieval, strong QA\n\n"
         "Question: {q}\n\nAnswer with one letter only.")


def reward_vec(em, alpha):
    B = alpha * COST["strong-strong"]
    return [em[c] - max(0.0, COST[c] / B - 1.0) for c in COMBOS]


# ─────────────────────────────── prepare ───────────────────────────────
def cmd_prepare(args):
    from collections import defaultdict
    em = defaultdict(dict)
    meta = {}
    for line in open(args.combo):
        try:
            r = json.loads(line)
        except Exception:
            continue
        em[r["idx"]][r["combo"]] = r["em_official"]
        meta[r["idx"]] = {"type": r.get("type"), "level": r.get("level")}
    em = {i: d for i, d in em.items() if len(d) == 4}

    hotpot = json.load(open(args.data))
    keep = set()
    for part in json.load(open(args.splits))["splits"].values():
        keep.update(part)

    n = 0
    with open(args.out, "w") as f:
        for idx in sorted(em):
            if idx not in keep:
                continue
            e = em[idx]
            f.write(json.dumps({
                "idx": idx,
                "question": hotpot[idx]["question"],
                "em": [e[c] for c in COMBOS],
                "R": [round(x, 6) for x in reward_vec(e, args.alpha)],
                "type": meta[idx].get("type"),
                "level": meta[idx].get("level"),
            }) + "\n")
            n += 1
    print(f"wrote {args.out}  ({n} instances, alpha={args.alpha})")


def load_rows(path, splits_path):
    rows = {json.loads(l)["idx"]: json.loads(l) for l in open(path)}
    sp = json.load(open(splits_path))["splits"]
    return rows, {k: [i for i in v if i in rows] for k, v in sp.items()}


# ─────────────────────────────── model ───────────────────────────────
def build_model(tok_only=False, init=None, lora_r=16):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    tok = AutoTokenizer.from_pretrained(ROUTER_MODEL)
    if tok_only:
        return tok, None
    model = AutoModelForCausalLM.from_pretrained(
        ROUTER_MODEL, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa")
    if init:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, init, is_trainable=True)
        print(f"  initialised from {init}")
    else:
        from peft import LoraConfig, get_peft_model
        cfg = LoraConfig(r=lora_r, lora_alpha=2 * lora_r, lora_dropout=0.05,
                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
                         task_type="CAUSAL_LM")
        model = get_peft_model(model, cfg)
    return tok, model


def label_ids(tok):
    ids = []
    for ch in LETTERS:
        enc = tok.encode(ch, add_special_tokens=False)
        assert len(enc) == 1, f"label {ch!r} is not a single token: {enc}"
        ids.append(enc[0])
    return ids


def prompt_for(tok, question):
    msgs = [{"role": "system", "content": SYS},
            {"role": "user", "content": INSTR.format(q=question)}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


def action_logits(model, tok, texts, lab_ids, device="cuda", max_len=512):
    """Logits over the 4 label tokens at the first generated position."""
    import torch
    enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
              max_length=max_len).to(device)
    out = model(**enc).logits                                   # [B, T, V]
    # 17Aug fix: left padding puts real tokens rightmost; the last real
    # token is position -1 for every row, independent of batch composition.
    sel = out[:, -1]                                            # [B, V]
    return sel[:, lab_ids].float()                              # [B, 4]


# ─────────────────────────────── training ───────────────────────────────
def cmd_train(args):
    import torch
    import torch.nn.functional as F

    rows, sp = load_rows(args.data, args.splits)
    train_idx, val_idx = sp["train"], sp["val"]
    print(f"arm={args.arm}  train={len(train_idx)}  val={len(val_idx)}")

    tok, model = build_model(init=args.init, lora_r=args.lora_r)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lab_ids = label_ids(tok)
    n_act = 2 if args.arm == "query" else 4
    act_map = [0, 3] if args.arm == "query" else [0, 1, 2, 3]   # query-level: ww vs ss only

    # frozen reference policy for the GRPO KL term
    ref = None
    if args.arm in ("grpo", "sft2grpo") and args.kl > 0:
        import copy
        ref = copy.deepcopy(model).eval()
        for p in ref.parameters():
            p.requires_grad_(False)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    rnd = random.Random(args.seed)
    order = list(train_idx)
    step = 0

    for epoch in range(args.epochs):
        rnd.shuffle(order)
        for b0 in range(0, len(order), args.batch):
            batch = order[b0:b0 + args.batch]
            texts = [prompt_for(tok, rows[i]["question"]) for i in batch]
            Rfull = torch.tensor([rows[i]["R"] for i in batch], device="cuda")   # [B,4]
            R = Rfull[:, act_map]                                                # [B,n_act]

            logits = action_logits(model, tok, texts, lab_ids)[:, act_map]
            logp = F.log_softmax(logits, dim=-1)

            if args.arm in ("sft", "sft2grpo_init", "query"):
                tgt = R.argmax(dim=-1)
                loss = F.nll_loss(logp, tgt)

            elif args.arm == "rwsft":
                # weight every action by its reward, normalised per instance
                w = torch.softmax(R / args.tau, dim=-1)
                loss = -(w * logp).sum(-1).mean()

            elif args.arm in ("grpo", "sft2grpo"):
                probs = logp.exp()
                # sample a group of k actions per prompt from the current policy
                a = torch.multinomial(probs, args.group, replacement=True)        # [B,k]
                r = torch.gather(R, 1, a)                                         # [B,k]
                adv = (r - r.mean(1, keepdim=True)) / (r.std(1, keepdim=True) + 1e-6)
                lp = torch.gather(logp, 1, a)
                loss = -(adv.detach() * lp).mean()
                if ref is not None:
                    with torch.no_grad():
                        ref_logp = F.log_softmax(
                            action_logits(ref, tok, texts, lab_ids)[:, act_map], dim=-1)
                    loss = loss + args.kl * F.kl_div(ref_logp, logp, log_target=True,
                                                     reduction="batchmean")
            else:
                raise ValueError(f"unknown arm {args.arm}")

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step(); opt.zero_grad()
            step += 1
            if step % args.log_every == 0:
                print(f"  epoch {epoch} step {step}  loss {loss.item():.4f}", flush=True)

        m = score(model, tok, lab_ids, rows, val_idx, act_map)
        print(f"[epoch {epoch}] val  EM {m['em']:.3f}  cost {m['cost']:.3f}  R {m['R']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    model.save_pretrained(args.out)
    json.dump({"arm": args.arm, "epochs": args.epochs, "lr": args.lr, "batch": args.batch,
               "group": args.group, "kl": args.kl, "tau": args.tau, "seed": args.seed,
               "init": args.init, "n_actions": n_act},
              open(os.path.join(args.out, "arm.json"), "w"), indent=2)
    print(f"saved -> {args.out}")


# ─────────────────────────────── evaluation ───────────────────────────────
def score(model, tok, lab_ids, rows, idxs, act_map, batch=32):
    """Greedy policy over the allowed actions; returns mean EM, cost, R."""
    import torch
    model.eval()
    em = cost = R = 0.0
    picks = Counter()
    with torch.no_grad():
        for b0 in range(0, len(idxs), batch):
            b = idxs[b0:b0 + batch]
            texts = [prompt_for(tok, rows[i]["question"]) for i in b]
            lg = action_logits(model, tok, texts, lab_ids)[:, act_map]
            choice = lg.argmax(-1).tolist()
            for i, ci in zip(b, choice):
                c = COMBOS[act_map[ci]]
                picks[c] += 1
                em += rows[i]["em"][act_map[ci]]
                cost += COST[c] / COST["strong-strong"]
                R += rows[i]["R"][act_map[ci]]
    model.train()
    n = max(1, len(idxs))
    return {"em": em / n, "cost": cost / n, "R": R / n, "picks": dict(picks)}


def fixed_metrics(rows, idxs, combo):
    j = COMBOS.index(combo)
    n = max(1, len(idxs))
    return {"em": sum(rows[i]["em"][j] for i in idxs) / n,
            "cost": COST[combo] / COST["strong-strong"],
            "R": sum(rows[i]["R"][j] for i in idxs) / n,
            "picks": {combo: len(idxs)}}


def oracle_metrics(rows, idxs):
    n = max(1, len(idxs))
    em = cost = R = 0.0
    picks = Counter()
    for i in idxs:
        j = max(range(4), key=lambda k: rows[i]["R"][k])
        picks[COMBOS[j]] += 1
        em += rows[i]["em"][j]
        cost += COST[COMBOS[j]] / COST["strong-strong"]
        R += rows[i]["R"][j]
    return {"em": em / n, "cost": cost / n, "R": R / n, "picks": dict(picks)}


def cmd_eval(args):
    rows, sp = load_rows(args.data, args.splits)
    test, train = sp["test"], sp["train"]

    # best-fixed is chosen on TRAIN, then applied to test — no test leakage
    best_fixed = max(COMBOS, key=lambda c: fixed_metrics(rows, train, c)["R"])
    print(f"best-fixed selected on train: {best_fixed}\n")

    results = {"n_test": len(test), "best_fixed": best_fixed, "arms": {}}
    results["arms"]["always weak-weak"] = fixed_metrics(rows, test, "weak-weak")
    results["arms"][f"always {best_fixed} (best-fixed)"] = fixed_metrics(rows, test, best_fixed)

    if args.arms:
        tok, _ = build_model(tok_only=True)
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        lab_ids = label_ids(tok)
        for path in args.arms:
            cfg = json.load(open(os.path.join(path, "arm.json")))
            act_map = [0, 3] if cfg["n_actions"] == 2 else [0, 1, 2, 3]
            _, model = build_model(init=path)
            results["arms"][cfg["arm"]] = score(model, tok, lab_ids, rows, test, act_map)
            del model
            import torch, gc
            gc.collect(); torch.cuda.empty_cache()

    results["arms"]["oracle"] = oracle_metrics(rows, test)

    # % of headroom captured, anchored 0% at best-fixed and 100% at oracle
    bf = results["arms"][f"always {best_fixed} (best-fixed)"]["R"]
    orc = results["arms"]["oracle"]["R"]
    span = orc - bf
    for name, m in results["arms"].items():
        m["headroom_pct"] = 100 * (m["R"] - bf) / span if span > 1e-9 else float("nan")

    order = ["always weak-weak", f"always {best_fixed} (best-fixed)", "query",
             "sft", "rwsft", "grpo", "sft2grpo", "oracle"]
    print(f"{'arm':34s} {'EM':>7s} {'cost':>7s} {'R':>8s} {'headroom':>9s}")
    for name in order:
        if name not in results["arms"]:
            continue
        m = results["arms"][name]
        print(f"{name:34s} {100*m['em']:6.1f}% {100*m['cost']:6.1f}% "
              f"{m['R']:8.4f} {m['headroom_pct']:8.1f}%")

    json.dump(results, open(args.out, "w"), indent=2)
    print(f"\nwrote {args.out}")


# ─────────────────────────────── cli ───────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--combo", required=True); p.add_argument("--data", required=True)
    p.add_argument("--splits", required=True); p.add_argument("--out", default="router_data.jsonl")
    p.add_argument("--alpha", type=float, default=0.6)
    p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("train")
    p.add_argument("--arm", required=True,
                   choices=["sft", "rwsft", "grpo", "sft2grpo", "query"])
    p.add_argument("--data", required=True); p.add_argument("--splits", required=True)
    p.add_argument("--out", required=True); p.add_argument("--init", default=None)
    p.add_argument("--epochs", type=int, default=2); p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4); p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--group", type=int, default=8, help="GRPO rollouts per prompt")
    p.add_argument("--kl", type=float, default=0.02, help="GRPO KL coefficient (0 disables)")
    p.add_argument("--tau", type=float, default=0.5, help="reward-weighted SFT temperature")
    p.add_argument("--seed", type=int, default=20260729)
    p.add_argument("--log_every", type=int, default=20)
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval")
    p.add_argument("--data", required=True); p.add_argument("--splits", required=True)
    p.add_argument("--arms", nargs="*", default=[]); p.add_argument("--out", default="results.json")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
