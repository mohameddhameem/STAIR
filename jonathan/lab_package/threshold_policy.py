#!/usr/bin/env python3
# =============================================================================
# threshold_policy.py — VERSION: threshold_policy v1.0 2026-09-06  (Run 1c)
#
# PURPOSE (slide 11, Run 1c). Two questions about the ALREADY-TRAINED Run 1
# router, no retraining:
#   (1) AUC — do its probabilities RANK instances at all (0.5 = nothing,
#       the collapse is total; >0.6 = argmax wasted usable confidence)?
#   (2) τ-policy — with SS as the DEFAULT and cheaper cells taken only when
#       the router's probability clears a bar τ, what cost–accuracy frontier
#       does this router already buy? (τ=0.5 reproduces Run 1's argmax;
#       τ→1 reproduces always-SS, the null.)
#
# TWO PHASES, separable:
#   score : GPU. For every val+test instance, THREE probabilities from the
#           saved adapter — p_ret_W (retrieval call), p_qaW_W (QA call on the
#           weak retriever's references), p_qaS_W (QA call on the strong's).
#           Written to <out>/router_scores.jsonl. ~2.5k instances × 3 calls;
#           retrieval prompts are ~14k tokens -> a few hours at batch 1.
#   sweep : CPU, seconds. Decision rule per (τ_ret, τ_qa):
#               ret = W  iff  p_ret_W  >= τ_ret   else S      (SS-default)
#               qa  = W  iff  p_qa|ret_W >= τ_qa  else S
#           -> cell -> stored-grid lookup (em_lenient / em_strict / R / cost).
#           Sweep the full τ grid on VAL, freeze the chosen τ* under each
#           selection rule, apply ONCE to TEST. Also full-test curve for the
#           frontier PLOT (illustration only; the frozen-τ numbers are the
#           honest ones).
#
# SELECTION RULES on val (all three reported frozen-on-test):
#   parity  : max cost saving subject to lenient >= always-SS lenient (on val)
#   best_acc: max lenient accuracy, cost ignored
#   knee    : max (accuracy − λ·cost%),  λ=0.1 — a simple knee heuristic
#
# PROMPTS: imported from sft_router.py (safe — its argparse lives in main()),
# so the scoring prompts are BYTE-IDENTICAL to the ones the router was trained
# on. No prompt text is duplicated here.
#
# USAGE (lab, venv):
#   python3 threshold_policy.py --data .../full_run_p100/router_data.jsonl \
#       --splits splits.json --adapter_dir /mnt/hdd/jonathan/runs/sft_v5_1p5b \
#       --adapter best --out /mnt/hdd/jonathan/runs/run1c
#   # later, re-sweep without GPU:  add  --sweep_only
# Outputs: router_scores.jsonl · policy_result.json · frontier.png · manifest
# =============================================================================
import os, json, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")

VERSION = "threshold_policy v1.0 2026-09-06"
COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
SS = 3  # index of strong-strong in COMBOS

# the trained prompts, verbatim from the training script (import is safe: main-guarded)
from sft_router import SYS_STAGE, INSTR_RET, INSTR_QA, CHOICE_WORD, CLASSES, LETTERS


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="router_data.jsonl (pool-100)")
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--adapter_dir", required=True, help="Run 1 output dir")
    ap.add_argument("--adapter", choices=["best", "last"], default="best",
                    help="best = best_adapter.pt (mean-acc checkpoint); last = final saved adapter")
    ap.add_argument("--router_model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--out", required=True)
    ap.add_argument("--maxlen", type=int, default=32768)
    ap.add_argument("--tau_step", type=float, default=0.02, help="τ grid spacing in [0,1]")
    ap.add_argument("--knee_lambda", type=float, default=0.1)
    ap.add_argument("--score_only", action="store_true")
    ap.add_argument("--sweep_only", action="store_true",
                    help="skip GPU; reuse <out>/router_scores.jsonl")
    return ap.parse_args()


def load_rows(path):
    rows = {}
    for l in open(path):
        r = json.loads(l); rows[r["idx"]] = r
    return rows


# ------------------------------ phase 1: score -------------------------------
def score_router(a, rows, ids):
    """Three probabilities per instance from the saved adapter. GPU."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, set_peft_model_state_dict

    tok = AutoTokenizer.from_pretrained(a.router_model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    lab_ids = [tok.encode(c, add_special_tokens=False)[0] for c in LETTERS]  # [A, B]

    model = AutoModelForCausalLM.from_pretrained(
        a.router_model, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa")
    model = PeftModel.from_pretrained(model, a.adapter_dir)          # 'last' state
    if a.adapter == "best":
        set_peft_model_state_dict(
            model, torch.load(os.path.join(a.adapter_dir, "best_adapter.pt")))
    model.eval()
    model.config.use_cache = False

    def prompt(system, user):
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def p_weak(texts):
        """P(letter A = weak) at the answer slot, per text. Batch of 1 or more."""
        enc = tok(texts, return_tensors="pt", padding=True, truncation=False).to("cuda")
        n = enc.input_ids.shape[1]
        assert n <= a.maxlen, f"prompt {n} tokens > maxlen {a.maxlen}"
        out = model(**enc).logits[:, -1][:, lab_ids].float()          # [B, 2] = [A, B]
        return torch.softmax(out, dim=-1)[:, 0].tolist()              # P(A) = P(weak)

    scores, done = {}, 0
    sc_path = os.path.join(a.out, "router_scores.jsonl")
    f = open(sc_path, "w")
    for idx in ids:
        r = rows[idx]
        ret_uc = INSTR_RET.format(q=r["question"], k=r["n_pool"], pool=r["pool_text"])
        p_ret = p_weak([prompt(SYS_STAGE, ret_uc)])[0]                # batch 1: long prompt
        qa_texts = [prompt(SYS_STAGE, INSTR_QA.format(
            q=r["question"], ret_choice=CHOICE_WORD[ch], evidence=r[f"evidence_{ch}"]))
            for ch in ("W", "S")]
        p_qaW, p_qaS = p_weak(qa_texts)                               # batch 2: short prompts
        rec = {"idx": idx, "split": r["split"], "p_ret_W": round(p_ret, 6),
               "p_qaW_W": round(p_qaW, 6), "p_qaS_W": round(p_qaS, 6)}
        f.write(json.dumps(rec) + "\n"); scores[idx] = rec
        done += 1
        if done % 100 == 0:
            print(f"scored {done}/{len(ids)}", flush=True); f.flush()
    f.close()
    print(f"wrote {sc_path} ({done} instances)", flush=True)
    return scores


# ------------------------------ AUC (no sklearn) -----------------------------
def auc(pairs):
    """pairs: list of (score, y) with y in {0,1}; AUC of score for y=1.
    Rank-based (Mann–Whitney), ties get half credit."""
    pos = sorted(p for p, y in pairs if y == 1)
    neg = sorted(p for p, y in pairs if y == 0)
    if not pos or not neg:
        return None
    import bisect
    wins = 0.0
    for p in pos:
        lo = bisect.bisect_left(neg, p); hi = bisect.bisect_right(neg, p)
        wins += lo + 0.5 * (hi - lo)
    return round(wins / (len(pos) * len(neg)), 4)


# ------------------------------ phase 2: sweep -------------------------------
def cell_of(sc, t_ret, t_qa):
    ret = "W" if sc["p_ret_W"] >= t_ret else "S"                      # SS-default
    p_qa = sc["p_qaW_W"] if ret == "W" else sc["p_qaS_W"]
    qa = "W" if p_qa >= t_qa else "S"
    return COMBOS.index(f"{CHOICE_WORD[ret]}-{CHOICE_WORD[qa]}")

def eval_policy(rows, scores, ids, t_ret, t_qa):
    n = len(ids)
    lens = sum(rows[i]["em_lenient"][cell_of(scores[i], t_ret, t_qa)] for i in ids) / n
    strict = sum(rows[i]["em_strict"][cell_of(scores[i], t_ret, t_qa)] for i in ids) / n
    cost = sum(COST[COMBOS[cell_of(scores[i], t_ret, t_qa)]] for i in ids) / n
    return round(100 * lens, 2), round(100 * strict, 2), round(100 * cost / COST["strong-strong"], 1)

def baselines(rows, ids):
    out = {}
    for name, k in (("always_SS", SS), ("always_WW", 0)):
        lens = sum(rows[i]["em_lenient"][k] for i in ids) / len(ids)
        strict = sum(rows[i]["em_strict"][k] for i in ids) / len(ids)
        out[name] = {"lenient": round(100 * lens, 2), "strict": round(100 * strict, 2),
                     "cost_pct_of_SS": round(100 * COST[COMBOS[k]] / COST["strong-strong"], 1)}
    return out


def main():
    a = build_args()
    os.makedirs(a.out, exist_ok=True)
    print(VERSION, flush=True)
    rows = load_rows(a.data)
    sp = json.load(open(a.splits))["splits"]
    va = [i for i in sp["val"] if i in rows]
    te = [i for i in sp["test"] if i in rows]

    if a.sweep_only:
        scores = {}
        for l in open(os.path.join(a.out, "router_scores.jsonl")):
            r = json.loads(l); scores[r["idx"]] = r
        print(f"reusing scores for {len(scores)} instances", flush=True)
    else:
        scores = score_router(a, rows, va + te)
        if a.score_only:
            print("score_only: done."); return

    # ---- AUC per stage, labeled rows only, val+test pooled and test alone
    def stage_pairs(ids):
        rp = [(1 - scores[i]["p_ret_W"], 1 if rows[i]["ret_label"] == "S" else 0)
              for i in ids if rows[i]["ret_label"]]
        qp = []
        for i in ids:
            for ch, key in (("W", "p_qaW_W"), ("S", "p_qaS_W")):
                lab = rows[i][f"qa_label_{ch}"]
                if lab:
                    qp.append((1 - scores[i][key], 1 if lab == "S" else 0))
        return rp, qp
    rp_te, qp_te = stage_pairs(te)
    aucs = {"ret_test": auc(rp_te), "qa_test": auc(qp_te)}
    print(f"AUC (P(S) vs S-labels, test): retrieval {aucs['ret_test']}  qa {aucs['qa_test']}", flush=True)

    # ---- τ sweep on val; freeze three τ* choices; apply to test
    taus = [round(a.tau_step * k, 4) for k in range(int(1 / a.tau_step) + 1)] + [1.01]
    base_va = baselines(rows, va); base_te = baselines(rows, te)
    ss_val = base_va["always_SS"]["lenient"]
    grid_va = {}
    for tr in taus:
        for tq in taus:
            grid_va[(tr, tq)] = eval_policy(rows, scores, va, tr, tq)
    parity = max((g for g in grid_va if grid_va[g][0] >= ss_val),
                 key=lambda g: -grid_va[g][2], default=None)          # min cost at >= SS acc
    best_acc = max(grid_va, key=lambda g: (grid_va[g][0], -grid_va[g][2]))
    knee = max(grid_va, key=lambda g: grid_va[g][0] - a.knee_lambda * grid_va[g][2])

    chosen = {}
    for name, g in (("parity", parity), ("best_acc", best_acc), ("knee", knee)):
        if g is None:
            chosen[name] = None; continue
        lv, sv, cv = grid_va[g]; lt, st, ct = eval_policy(rows, scores, te, *g)
        chosen[name] = {"tau_ret": g[0], "tau_qa": g[1],
                        "val": {"lenient": lv, "strict": sv, "cost_pct_of_SS": cv},
                        "test": {"lenient": lt, "strict": st, "cost_pct_of_SS": ct}}
        print(f"[{name}] τ_ret {g[0]} τ_qa {g[1]} | val {lv} @ {cv}% | TEST {lt} lenient @ {ct}% of SS cost", flush=True)
    print(f"[baselines TEST] SS {base_te['always_SS']} | WW {base_te['always_WW']} | argmax = τ(0.5, 0.5): "
          f"{eval_policy(rows, scores, te, 0.5, 0.5)}", flush=True)

    # ---- frontier plot (test curve for illustration; frozen τ* marked)
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pts = set()
        for tr in taus:
            for tq in taus:
                l, _, c = eval_policy(rows, scores, te, tr, tq)
                pts.add((c, l))                       # (cost, lenient)
        xs = [c for c, _ in pts]; ys = [l for _, l in pts]
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=150)
        ax.scatter(xs, ys, s=8, alpha=0.35, color="#2C8FA3", label="τ-policy points (test)")
        for name, mark, col in (("always_SS", "*", "#1F3A5F"), ("always_WW", "s", "#8a8a8a")):
            b = base_te[name]
            ax.scatter([b["cost_pct_of_SS"]], [b["lenient"]], marker=mark, s=120, color=col, zorder=5, label=name)
        for name, col in (("parity", "#D97A3A"), ("best_acc", "#6B4E8E")):
            ch = chosen.get(name)
            if ch:
                ax.scatter([ch["test"]["cost_pct_of_SS"]], [ch["test"]["lenient"]],
                           marker="o", s=90, color=col, zorder=6, label=f"τ* {name} (frozen on val)")
        ax.set_xlabel("cost, % of always-SS"); ax.set_ylabel("lenient accuracy (test)")
        ax.set_title("Run 1c — SS-default τ-policy frontier from the Run 1 router")
        ax.grid(alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "frontier.png"))
        print("wrote frontier.png", flush=True)
    except Exception as e:
        print(f"(plot skipped: {e})", flush=True)

    json.dump({"version": VERSION, "adapter": a.adapter, "adapter_dir": a.adapter_dir,
               "aucs": aucs, "baselines_test": base_te, "baselines_val": base_va,
               "chosen": chosen, "tau_step": a.tau_step,
               "n_scored": {"val": len(va), "test": len(te)}},
              open(os.path.join(a.out, "policy_result.json"), "w"), indent=2)
    print(f"saved -> {a.out}/policy_result.json", flush=True)


if __name__ == "__main__":
    main()
