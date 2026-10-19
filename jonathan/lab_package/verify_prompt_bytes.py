#!/usr/bin/env python3
# =============================================================================
# verify_prompt_bytes.py — VERSION: verify_prompt_bytes v1.0 2026-09-15
#
# AVENUE #1: prove the rendered prompt BYTES are identical between the
# trainer (sft_router.py) and the probe (overfit_probe.py / overfit_probe_qa.py).
#
# METHOD (no guessing, no heavy imports):
#   1. Parse each script with `ast` and extract the prompt constants exactly
#      as written (SYS_STAGE, INSTR_RET, INSTR_QA, CHOICE_WORD, PROBE_IDX).
#   2. Compare the constants byte-for-byte across files.
#   3. Render full prompts for the probe's instances through the real chat
#      template (same tokenizer both paths) using each file's own constants,
#      and assert the final strings are byte-identical. First diff printed
#      with offset + context on any mismatch.
#
# USAGE (lab, CPU, <1 min):
#   python3 verify_prompt_bytes.py \
#       --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#       --sft sft_router.py --probe overfit_probe.py --probe_qa overfit_probe_qa.py
# =============================================================================
import ast, json, argparse, sys

VERSION = "verify_prompt_bytes v1.0 2026-09-15"
WANT = ["SYS_STAGE", "INSTR_RET", "INSTR_RET_QONLY", "INSTR_QA", "CHOICE_WORD", "PROBE_IDX"]


def extract_constants(path):
    """Pull top-level constant assignments from a script without executing it."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in WANT:
                try:
                    out[name] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def first_diff(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) != len(b) else -1


def cmp(tag, a, b):
    if a == b:
        print(f"  {tag}: IDENTICAL ({len(a)} chars)")
        return True
    i = first_diff(a, b)
    print(f"  {tag}: DIFF at char {i}")
    print(f"    trainer: ...{a[max(0,i-40):i+40]!r}...")
    print(f"    probe  : ...{b[max(0,i-40):i+40]!r}...")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--sft", default="sft_router.py")
    ap.add_argument("--probe", default="overfit_probe.py")
    ap.add_argument("--probe_qa", default="overfit_probe_qa.py")
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    a = ap.parse_args()
    print(VERSION, flush=True)

    sft = extract_constants(a.sft)
    prb = extract_constants(a.probe)
    print(f"constants found — trainer: {sorted(sft)} | probe: {sorted(prb)}")

    ok = True
    # ---- step 1: constant-level byte comparison (ret path) -------------------
    print("\n[1] constant comparison, trainer vs ret-probe:")
    for name in ("SYS_STAGE", "INSTR_RET"):
        if name in sft and name in prb:
            ok &= cmp(name, sft[name], prb[name])
        else:
            print(f"  {name}: MISSING in one file — trainer has: {name in sft}, probe has: {name in prb}")
            ok = False

    # ---- step 2: QA probe, if present ----------------------------------------
    try:
        qa = extract_constants(a.probe_qa)
        print("\n[2] constant comparison, trainer vs qa-probe:")
        for name in ("SYS_STAGE", "INSTR_QA"):
            if name in sft and name in qa:
                ok &= cmp(name, sft[name], qa[name])
        if "CHOICE_WORD" in sft and "CHOICE_WORD" in qa:
            same = sft["CHOICE_WORD"] == qa["CHOICE_WORD"]
            print(f"  CHOICE_WORD: {'IDENTICAL' if same else 'DIFF'} ({sft.get('CHOICE_WORD')})")
            ok &= same
    except FileNotFoundError:
        print(f"\n[2] {a.probe_qa} not found — QA constants checked against trainer only if present later.")
        qa = {}

    # ---- step 3: full rendered-prompt byte proof on the probe instances ------
    idxs = prb.get("PROBE_IDX", [5, 6, 9, 10, 12, 13, 16, 17, 22, 23])
    rows = {}
    for l in open(a.data):
        r = json.loads(l)
        if r["idx"] in set(idxs):
            rows[r["idx"]] = r
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)

    def full(sys_c, user_c):
        return tok.apply_chat_template(
            [{"role": "system", "content": sys_c}, {"role": "user", "content": user_c}],
            tokenize=False, add_generation_prompt=True)

    print(f"\n[3] full rendered RET prompts, trainer-constants vs probe-constants, {len(idxs)} instances:")
    bad = 0
    for i in idxs:
        r = rows[i]
        t = full(sft["SYS_STAGE"], sft["INSTR_RET"].format(q=r["question"], k=r["n_pool"], pool=r["pool_text"]))
        p = full(prb["SYS_STAGE"], prb["INSTR_RET"].format(q=r["question"], k=r["n_pool"], pool=r["pool_text"]))
        if t != p:
            bad += 1
            print(f"  idx {i}: DIFF at char {first_diff(t, p)}")
        else:
            print(f"  idx {i}: byte-identical ({len(t)} chars)")
    ok &= (bad == 0)

    print("\nVERDICT:", "PROMPT BYTES CERTIFIED — trainer and probe render identical strings"
          if ok else "DIFFERENCES FOUND — see lines above")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
