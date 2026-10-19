#!/usr/bin/env python3
# =============================================================================
# zeroshot_router.py — VERSION: zeroshot_router v1.1 2026-09-15 (--only_splits)
#
# CAUSE-3 DIAGNOSTIC + PROMPT-CRITERIA TEST (no training anywhere).
# Asks the BASE instruct model (no adapter) to reason about each instance with
# explicit decision criteria — Jonathan's "give the router guidance" idea,
# tested in the one regime where instructions are the mechanism.
#
# TWO INPUT MODES (--input):
#   pool     : question + the raw 100-passage pool (Run 1's input; ~14k tokens,
#              batch 1, ~2 h). Tests: is W/S decidable from the raw input by
#              the strongest reader we have, with reasoning criteria?
#   evidence : question + the WEAK retriever's selected passages (~500 tokens,
#              batch 8, ~15 min). PREVIEWS RUN 3's input with zero training:
#              "are these references sufficient?" — a comprehension question.
#
# TWO PROMPT VARIANTS (--prompt): plain (Run 1's framing, zero-shot) and
# enriched (explicit criteria: hops, entity rarity, direct support). Default
# enriched; run plain only if the enriched result makes the contrast worth it.
#
# SCORER MODEL (--scorer): default the 7B (the best reasoner on the box);
# 1p5b variant for the cost-realistic version. The score is P("A") from the
# A/B letter logits — same read-out as training, so AUCs are comparable.
#
# OUTPUT: <out>/zs_scores.jsonl in EXACTLY threshold_policy.py's format
# (p_ret_W / p_qaW_W / p_qaS_W), so AUC + τ-sweep + frontier come free:
#   python3 threshold_policy.py --sweep_only --out <out> ...
# This script also prints the AUCs directly for the quick read.
#
# USAGE (lab):
#   python3 zeroshot_router.py --data .../router_data.jsonl --splits splits.json \
#       --input evidence --out /mnt/hdd/jonathan/runs/zs_evidence
#   # then: python3 threshold_policy.py --data ... --splits ... \
#   #   --adapter_dir unused --adapter last --out /mnt/hdd/jonathan/runs/zs_evidence --sweep_only
# =============================================================================
import os, json, argparse
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("HF_HOME", "/mnt/hdd/jonathan/hf_cache")

VERSION = "zeroshot_router v1.1 2026-09-15 (--only_splits)"

SYS_ZS = ("You estimate, for one stage of a question-answering pipeline, whether a small "
          "model (Qwen2.5-1.5B) will handle it correctly or the large model (Qwen2.5-7B) "
          "is needed. Answer with one letter: A = the small model suffices, B = the large "
          "model is needed.")

CRITERIA = ("To decide, assess how demanding this instance is for a small model:\n"
            "- Does answering require combining facts from multiple passages (multi-hop), "
            "or is the answer directly stated in one place?\n"
            "- Are the key entities common and unambiguous, or rare and confusable?\n"
            "- Is the question simple in form (yes/no, single lookup) or layered?\n"
            "- Do the provided passages clearly contain what is needed?\n"
            "Simple, direct, well-supported -> A. Multi-hop, ambiguous, or thinly "
            "supported -> B.\n")

U_POOL = ("Stage: RETRIEVAL — the model must select the references relevant to the "
          "question from the candidate pool.\n{crit}Question: {q}\n"
          "Candidate pool ({k} passages): {pool}\nAnswer with one letter only.")
U_EVID = ("Stage: RETRIEVAL escalation check — the SMALL model already selected the "
          "references below. If they are sufficient to answer the question, the small "
          "model's retrieval suffices (A); if key facts are missing, the large model "
          "must redo retrieval (B).\n{crit}Question: {q}\n"
          "References selected by the small model: {ev}\nAnswer with one letter only.")
U_QA = ("Stage: ANSWERING — the model must write the answer using the references "
        "below.\n{crit}Question: {q}\nRetrieved references (selected by the "
        "{choice} retriever): {ev}\nAnswer with one letter only.")


def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", default="splits.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--input", choices=["pool", "evidence"], default="evidence")
    ap.add_argument("--prompt", choices=["enriched", "plain"], default="enriched")
    ap.add_argument("--scorer", default="Qwen/Qwen2.5-7B-Instruct",
                    help="the reasoning model; use Qwen/Qwen2.5-1.5B-Instruct for the cost-realistic variant")
    ap.add_argument("--maxlen", type=int, default=32768)
    ap.add_argument("--limit", type=int, default=0, help="0 = all of val+test; N = first N (smoke)")
    ap.add_argument("--only_splits", default="val,test",
                    help="comma list of splits to score (v1.1; default preserves v1.0 behavior)")
    return ap.parse_args()


def main():
    a = build_args()
    os.makedirs(a.out, exist_ok=True)
    print(VERSION, "| input:", a.input, "| prompt:", a.prompt, "| scorer:", a.scorer, flush=True)
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    rows = {}
    for l in open(a.data):
        r = json.loads(l); rows[r["idx"]] = r
    sp = json.load(open(a.splits))["splits"]
    want = [w.strip() for w in a.only_splits.split(",") if w.strip()]
    ids = [i for w in want for i in sp[w] if i in rows]
    if a.limit: ids = ids[:a.limit]

    tok = AutoTokenizer.from_pretrained(a.scorer)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    lab_ids = [tok.encode(c, add_special_tokens=False)[0] for c in ("A", "B")]
    assert all(len(tok.encode(c, add_special_tokens=False)) == 1 for c in ("A", "B"))
    model = AutoModelForCausalLM.from_pretrained(
        a.scorer, torch_dtype=torch.float16, device_map="cuda", attn_implementation="sdpa")
    model.eval(); model.config.use_cache = False
    crit = CRITERIA if a.prompt == "enriched" else ""

    def render(system, user):
        return tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def p_A(texts):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=False).to("cuda")
        assert enc.input_ids.shape[1] <= a.maxlen, f"prompt {enc.input_ids.shape[1]} > maxlen"
        out = model(**enc).logits[:, -1][:, lab_ids].float()
        return torch.softmax(out, dim=-1)[:, 0].tolist()

    f = open(os.path.join(a.out, "zs_scores.jsonl"), "w")
    done = 0
    for idx in ids:
        r = rows[idx]
        if a.input == "pool":
            ret_u = U_POOL.format(crit=crit, q=r["question"], k=r["n_pool"], pool=r["pool_text"])
            p_ret = p_A([render(SYS_ZS, ret_u)])[0]            # 14k tokens -> batch 1
        else:
            ret_u = U_EVID.format(crit=crit, q=r["question"], ev=r["evidence_W"])
            p_ret = p_A([render(SYS_ZS, ret_u)])[0]            # short; batch 1 keeps code simple
        qa_texts = [render(SYS_ZS, U_QA.format(crit=crit, q=r["question"],
                    choice=("weak" if ch == "W" else "strong"), ev=r[f"evidence_{ch}"]))
                    for ch in ("W", "S")]
        p_qaW, p_qaS = p_A(qa_texts)
        f.write(json.dumps({"idx": idx, "split": r["split"], "p_ret_W": round(p_ret, 6),
                            "p_qaW_W": round(p_qaW, 6), "p_qaS_W": round(p_qaS, 6)}) + "\n")
        done += 1
        if done % 100 == 0: print(f"scored {done}/{len(ids)}", flush=True); f.flush()
    f.close()

    # quick AUC read (same rank-AUC as 1c, imported)
    from threshold_policy import auc
    scores = {json.loads(l)["idx"]: json.loads(l) for l in open(os.path.join(a.out, "zs_scores.jsonl"))}
    te = [i for i in sp["test"] if i in scores]
    rp = [(1 - scores[i]["p_ret_W"], 1 if rows[i]["ret_label"] == "S" else 0)
          for i in te if rows[i]["ret_label"]]
    qp = []
    for i in te:
        for ch, key in (("W", "p_qaW_W"), ("S", "p_qaS_W")):
            if rows[i][f"qa_label_{ch}"]:
                qp.append((1 - scores[i][key], 1 if rows[i][f"qa_label_{ch}"] == "S" else 0))
    print(f"AUC (test) — retrieval {auc(rp)}  qa {auc(qp)}   [Run 1c trained router: 0.580 / 0.550; "
          f"rung-3 features: 0.606]", flush=True)
    # sufficiency framing note: in evidence mode, p_ret_W is P(weak retrieval sufficient)
    json.dump({"version": VERSION, "input": a.input, "prompt": a.prompt, "scorer": a.scorer,
               "n_scored": len(scores)}, open(os.path.join(a.out, "zs_manifest.json"), "w"), indent=2)
    print(f"saved -> {a.out}  (rename zs_scores.jsonl to router_scores.jsonl to reuse "
          f"threshold_policy.py --sweep_only for the frontier)", flush=True)


if __name__ == "__main__":
    main()
