#!/usr/bin/env python3
# eval_train_split.py v1.3 — memorization diagnostic (Prof. Archan, 18 Sep 2026)
# Scores a FINISHED run's saved adapter on the TRAINING split (val alongside, for
# reference) using sft_router's OWN load_rows/make_examples and a byte-identical
# prompt path (same constants, same user_content logic, same chat template, same
# letter-logit argmax). READ-ONLY: nothing in the run dir is modified.
#
#   python3 eval_train_split.py --run_dir /mnt/hdd/jonathan/runs/sft_v7_joint_tau050 \
#       --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_v2_tau0.50.jsonl \
#       --splits splits.json [--adapter best|last] [--sample 400] [--seed 0]
import argparse, json, os, random, sys, glob
sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")
import sft_router as SR                      # certified module, imported unmodified
from sft_router import load_rows, make_examples

def find_last_pt(run_dir):
    """Newest phase_*_last.pt = the run's final state (mixed or sequential)."""
    pts = sorted(glob.glob(os.path.join(run_dir, "phase_*_last.pt")), key=os.path.getmtime)
    if not pts:
        raise SystemExit(f"no phase_*_last.pt in {run_dir}; contents: {sorted(os.listdir(run_dir))}")
    return pts[-1]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--adapter", default="best", choices=["best", "last"])
    ap.add_argument("--sample", type=int, default=400, help="instances per split (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    # ---- run config from the run's own manifest (no guessed settings) --------
    man_path = os.path.join(a.run_dir, "manifest.json")
    man = json.load(open(man_path)) if os.path.exists(man_path) else {}
    class Cfg: pass
    cfg = Cfg()
    cfg.scenario       = man.get("scenario", "A")
    cfg.qa_choices     = man.get("qa_choices", "both")
    cfg.ret_input      = man.get("ret_input", "pool")
    cfg.evidence_chars = man.get("evidence_chars", -1)
    cfg.router_model   = man.get("router_model", "Qwen/Qwen2.5-1.5B-Instruct")
    print(f"[cfg] scenario={cfg.scenario} qa_choices={cfg.qa_choices} ret_input={cfg.ret_input} "
          f"evidence_chars={cfg.evidence_chars} model={cfg.router_model}")

    rows = load_rows(a.data)
    splits = json.load(open(a.splits)); splits = splits.get("splits", splits)

    # ---- trainer-identical prompt path (constants imported, logic mirrored) --
    def evidence(r, ch):
        ev = r[f"evidence_{ch}"]
        return ev if cfg.evidence_chars <= 0 else ev[:cfg.evidence_chars]
    def user_content(e):
        r = rows[e["idx"]]
        if e["stage"] == "ret":
            if cfg.ret_input == "question":
                return SR.INSTR_RET_QONLY.format(q=r["question"])
            return SR.INSTR_RET.format(q=r["question"], k=r["n_pool"], pool=r["pool_text"])
        ch = e["ret_choice"]
        return SR.INSTR_QA.format(q=r["question"], ret_choice=SR.CHOICE_WORD[ch], evidence=evidence(r, ch))

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(cfg.router_model)
    tok.padding_side = "left"
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    let_tok = [tok.encode(c, add_special_tokens=False) for c in SR.LETTERS]
    assert all(len(t) == 1 for t in let_tok), "letters not single tokens"
    lab_ids = [t[0] for t in let_tok]                 # position-paired with SR.CLASSES
    def prompt_for(e):
        msgs = [{"role": "system", "content": SR.SYS_STAGE},
                {"role": "user",   "content": user_content(e)}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    base = AutoModelForCausalLM.from_pretrained(cfg.router_model, torch_dtype=torch.bfloat16, device_map="auto")
    if not os.path.exists(os.path.join(a.run_dir, "adapter_config.json")):
        raise SystemExit(f"{a.run_dir} has no adapter_config.json — run not finished yet? "
                         f"contents: {sorted(os.listdir(a.run_dir))}")
    if a.adapter == "best":
        # root-level PEFT save == the best adapter (trainer reloads best, then save_pretrained)
        model = PeftModel.from_pretrained(base, a.run_dir).eval()
        print(f"[adapter] best = root PEFT save ({a.run_dir}/adapter_model.safetensors)")
    else:
        from peft import PeftConfig, get_peft_model as _gpm, set_peft_model_state_dict as _sps
        pc = PeftConfig.from_pretrained(a.run_dir)
        model = _gpm(base, pc)
        pt = find_last_pt(a.run_dir)
        _sps(model, torch.load(pt, map_location="cpu"))
        model = model.eval()
        print(f"[adapter] last = {os.path.basename(pt)} loaded into shell from adapter_config.json")

    class ExArgs: pass
    xa = ExArgs(); xa.scenario = cfg.scenario; xa.qa_choices = cfg.qa_choices
    rng = random.Random(a.seed)
    out = {}
    for split in ("train", "val"):
        ids = list(splits[split])
        if a.sample and len(ids) > a.sample: ids = rng.sample(ids, a.sample)
        ex = make_examples(rows, ids, xa)
        acc = {"ret": [0, 0], "qa": [0, 0]}
        with torch.no_grad():
            for e in ex:
                t = tok(prompt_for(e), return_tensors="pt").to(model.device)
                logits = model(**t).logits[0, -1]
                pred = SR.CLASSES[max(range(len(lab_ids)), key=lambda i: logits[lab_ids[i]].item())]
                st = e["stage"]; acc[st][1] += 1; acc[st][0] += int(pred == e["label"])
        out[split] = {st: {"acc": round(c / n, 4) if n else None, "n": n} for st, (c, n) in acc.items()}
        # majority floors for THIS sample, from the labels themselves
        for st in ("ret", "qa"):
            labs = [e["label"] for e in ex if e["stage"] == st]
            if labs:
                from collections import Counter
                out[split][st]["floor"] = round(Counter(labs).most_common(1)[0][1] / len(labs), 4)
    print(json.dumps(out, indent=2))
    print("\nREADING: train acc >> floor while val ~ floor  -> memorized, no transfer;"
          "\n         train acc ~ floor                     -> cannot even memorize at this exposure;"
          "\n         both >> floor                         -> healthy signal (positive-control-like).")

if __name__ == "__main__":
    main()
