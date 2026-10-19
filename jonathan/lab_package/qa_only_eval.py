#!/usr/bin/env python3
# qa_only_eval.py v1.0 — the pivot architecture, evaluated (18 Sep 2026)
# STRONG retrieval fixed for every instance; the TRAINED router decides only the
# answerer (W/S) from the strong evidence — the exact QA prompt format it trained on.
# Composed scoring on the sealed TEST split: answerer W -> em[SW cell], S -> em[SS].
# Zero training; read-only. Compare vs ladder: WW 31.80 / SW 41.73 / SS 53.54 / oracle 61.24.
import argparse, json, os, glob, sys
sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")
import sft_router as SR
from sft_router import load_rows

COST = {"WW": 2.0, "WS": 5.7, "SW": 5.7, "SS": 9.4}  # project FLOP units

def find_last_pt(run_dir):
    pts = sorted(glob.glob(os.path.join(run_dir, "phase_*_last.pt")), key=os.path.getmtime)
    if not pts: raise SystemExit(f"no phase_*_last.pt in {run_dir}; contents: {sorted(os.listdir(run_dir))}")
    return pts[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)          # the joint-training run
    ap.add_argument("--data", required=True)             # router_data_v2_tau0.50.jsonl (prompts) — em/labels read from lenient file
    ap.add_argument("--em_data", default="/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--adapter", default="last", choices=["best", "last"])
    a = ap.parse_args()

    man = json.load(open(os.path.join(a.run_dir, "manifest.json"))) if os.path.exists(os.path.join(a.run_dir,"manifest.json")) else {}
    evidence_chars = man.get("evidence_chars", -1)
    router_model = man.get("router_model", "Qwen/Qwen2.5-1.5B-Instruct")
    print(f"[cfg] evidence_chars={evidence_chars} model={router_model} adapter={a.adapter}")

    rows = load_rows(a.data)                             # prompt fields (question, evidence_S)
    em_rows = load_rows(a.em_data)                       # em_lenient/em_strict + qa_label_S + type via _id join later if needed
    sp = json.load(open(a.splits)); sp = sp.get("splits", sp)
    test_ids = sp["test"]

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel, PeftConfig, get_peft_model, set_peft_model_state_dict

    tok = AutoTokenizer.from_pretrained(router_model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    let_tok = [tok.encode(c, add_special_tokens=False) for c in SR.LETTERS]
    assert all(len(t) == 1 for t in let_tok)
    lab_ids = [t[0] for t in let_tok]

    base = AutoModelForCausalLM.from_pretrained(router_model, torch_dtype=torch.bfloat16, device_map="auto")
    if a.adapter == "best":
        model = PeftModel.from_pretrained(base, a.run_dir).eval()
        print(f"[adapter] best = root PEFT save")
    else:
        pc = PeftConfig.from_pretrained(a.run_dir)
        model = get_peft_model(base, pc)
        pt = find_last_pt(a.run_dir)
        set_peft_model_state_dict(model, torch.load(pt, map_location="cpu"))
        model = model.eval()
        print(f"[adapter] last = {os.path.basename(pt)}")

    def qa_prompt(r):
        ev = r["evidence_S"]
        ev = ev if evidence_chars <= 0 else ev[:evidence_chars]
        uc = SR.INSTR_QA.format(q=r["question"], ret_choice=SR.CHOICE_WORD["S"], evidence=ev)
        msgs = [{"role": "system", "content": SR.SYS_STAGE}, {"role": "user", "content": uc}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    n=len(test_ids); picks={"W":0,"S":0}; hit_len=0; hit_str=0; cost=0.0
    agree_qaS=[0,0]
    import collections
    by_pick_len=collections.Counter(); done=0
    with torch.no_grad():
        for i in test_ids:
            r = rows[i]; e = em_rows[i]
            t = tok(qa_prompt(r), return_tensors="pt").to(model.device)
            logits = model(**t).logits[0, -1]
            pred = SR.CLASSES[max(range(len(lab_ids)), key=lambda k: logits[lab_ids[k]].item())]
            picks[pred]+=1
            cell = 2 if pred=="W" else 3                  # SW or SS
            hit_len += e["em_lenient"][cell]
            if "em_strict" in e: hit_str += e["em_strict"][cell]
            cost += COST["SW"] if pred=="W" else COST["SS"]
            ql = e.get("qa_label_S")
            if ql in ("W","S"):
                agree_qaS[1]+=1; agree_qaS[0]+=int(pred==ql)
            by_pick_len[(pred, e["em_lenient"][cell])]+=1
            done+=1
            if done % 300 == 0: print(f"  ... {done}/{n}", flush=True)
    print(f"\n[QA-ONLY ROUTER on sealed TEST, n={n}] strong retrieval fixed; router decides answerer")
    print(f"  picks: answerer-W {picks['W']}  answerer-S {picks['S']}")
    print(f"  composed lenient: {100*hit_len/n:.2f}   (ladder: WW 31.80 | SW 41.73 | SS 53.54 | oracle 61.24)")
    print(f"  composed strict:  {100*hit_str/n:.2f}")
    print(f"  cost: {100*cost/(n*COST['SS']):.1f}% of always-SS")
    if agree_qaS[1]:
        print(f"  agreement with real qa_label_S on labeled rows: {agree_qaS[0]/agree_qaS[1]:.4f} (n={agree_qaS[1]})")
    print(f"  verdict: {'POSITIVE CAPTURE' if hit_len/n>0.5354 else 'below always-SS'} "
          f"({100*hit_len/n:.2f} vs 53.54 at {100*cost/(n*COST['SS']):.1f}% cost)")

if __name__ == "__main__":
    main()
