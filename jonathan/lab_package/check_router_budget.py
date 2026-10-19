"""
Stage-Level Routing — check_router_budget.py (v1.0, 2026-10-07)  ·  CPU only, ~minutes

PURPOSE
  Put MEASURED reasons behind the two router-input knobs frozen in router_prompts.py:
    CHAR_CAP = 600   (characters of each passage the router sees)
    MAXLEN   = 8192  (token budget of one assembled router call)

  Three measurements over the Config A grid + labels (nothing runs, nothing changes):

  M1  gold-visibility@cap — of the instances LABELED W (gold found in weak's
      passages, uncapped), in what fraction is the gold string still visible
      within the first <cap> characters of at least one selected passage?
      Computed at caps 300 / 400 / 600 / 900 / 1200 / uncapped -> the full curve.
      (Self-validation: our containment re-computes covW uncapped and reports
      agreement with the label file — expect ~100%, proving the check matches
      the label factory's.)

  M2  passage survival — of the passages the weak retriever actually selected:
      fraction whose full text fits within 600 chars, and the median fraction
      of a passage that survives the cap.

  M3  token budget — assemble EVERY Ret-router call (real selections, cap 600,
      the real chat template) and count tokens: distribution, p99, max, and the
      fraction exceeding 8,192. Same for the QA-gate call (evidence + weak answer).

  PRE-DECLARED DECISION RULE (set with Jonathan, 7 Oct):
      keep 600/8192 if M1@600 >= ~90% and M3 overflow < 1%;
      else raise CHAR_CAP to where M1 crosses ~90%, re-check M3, re-run the probe.

USAGE (lab box, from the directory holding run_labels4.py and router_prompts.py):
  python check_router_budget.py \
      --data /mnt/hdd/jonathan/data/tts_test.json \
      --grid /mnt/hdd/jonathan/runs/full_run_p100 \
      --labels /mnt/hdd/jonathan/runs/ret_labels/ret_labels.jsonl \
      --out /mnt/hdd/jonathan/runs/budget_report
  Add --skip-tokens to skip M3 (no tokenizer load; M1/M2 are pure string work).
"""
import os, sys, json, argparse, re, statistics

# pool construction must match the grid exactly: pool_k=100, shuffle OFF
os.environ.setdefault("SLR_POOL_K", "100")
os.environ.setdefault("SLR_SHUFFLE", "off")
import run_labels4 as R            # trim_pool, parse_selection, build_block (frozen)
import router_prompts as RP        # the prompt contract under audit

CAPS = [300, 400, 600, 900, 1200, None]   # None = uncapped

# ---------- containment: same rule family as the label factory ----------
def norm(s):
    return re.sub(r"\s+", " ", s.lower()).strip()

def strip_articles(s):
    return re.sub(r"^(a|an|the)\s+", "", s)

def contains(passage, gold):
    """Whole-word containment of the (normalized, article-stripped) gold string."""
    g = strip_articles(norm(gold))
    if not g:
        return False
    return re.search(r"(?<!\w)" + re.escape(g) + r"(?!\w)", norm(passage)) is not None

def golds_of(inst):
    for key in ("answers", "answer", "golds", "gold"):
        v = inst.get(key)
        if isinstance(v, str): return [v]
        if isinstance(v, list) and v: return [x for x in v if isinstance(x, str)]
    raise KeyError(f"no gold-answer field found; instance keys: {sorted(inst.keys())}")

def load_store(path):
    d = {}
    for line in open(path):
        try:
            r = json.loads(line); d[r["k"]] = r["v"]
        except Exception:
            pass
    return d

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--grid", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="budget_report")
    ap.add_argument("--skip-tokens", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    print("loading data + grid + labels ...")
    hotpot = json.load(open(args.data))
    retr_w = load_store(os.path.join(args.grid, "retr_weak.jsonl"))
    qa_w   = load_store(os.path.join(args.grid, "qa_weak.jsonl"))
    labels = {}
    for line in open(args.labels):
        r = json.loads(line); labels[int(r["idx"])] = r

    # rebuild each instance's pool and selections exactly as the grid saw them
    rows = []
    for i, inst in enumerate(hotpot):
        if i not in labels or str(i) not in retr_w:
            continue
        inst = dict(inst)
        inst["ctxs_candidate"] = R.trim_pool(inst)
        sel = R.parse_selection(retr_w[str(i)], len(inst["ctxs_candidate"]))
        rows.append((i, inst, sel))
    print(f"{len(rows)} instances joined (data ∩ grid ∩ labels)")

    # ---------- self-validation: our containment vs the label file ----------
    agree = disagree = 0
    for i, inst, sel in rows:
        golds = golds_of(inst)
        covW_here = any(contains(inst["ctxs_candidate"][n-1], g) for n in sel for g in golds) if sel else False
        covW_label = labels[i]["covW"] > 0 if isinstance(labels[i].get("covW"), (int, float)) else bool(labels[i].get("covW"))
        agree += (covW_here == covW_label); disagree += (covW_here != covW_label)
    agreement = agree / max(1, agree + disagree)
    print(f"\nSELF-VALIDATION  our covW vs label file: {agree} agree / {disagree} disagree "
          f"({agreement:.1%}) — expect ~100%; below ~99% means this audit's containment "
          f"differs from the factory's and M1 should not be trusted until reconciled")

    # ---------- M1: gold-visibility@cap over W-labeled instances ----------
    w_rows = [(i, inst, sel) for i, inst, sel in rows if labels[i]["ret_label"] == "W"]
    print(f"\nM1  gold-visibility@cap  (n = {len(w_rows)} W-labeled instances)")
    m1 = {}
    for cap_v in CAPS:
        vis = 0
        for i, inst, sel in w_rows:
            golds = golds_of(inst)
            view = [(inst["ctxs_candidate"][n-1] if cap_v is None
                     else inst["ctxs_candidate"][n-1][:cap_v]) for n in sel]
            if any(contains(p, g) for p in view for g in golds):
                vis += 1
        key = "uncapped" if cap_v is None else str(cap_v)
        m1[key] = vis / max(1, len(w_rows))
        print(f"    cap {key:>8}: {m1[key]:6.1%}")

    # ---------- M2: passage survival at 600 ----------
    lens = [len(inst["ctxs_candidate"][n-1]) for _, inst, sel in rows for n in sel]
    fit = sum(1 for L in lens if L <= 600) / max(1, len(lens))
    surv = statistics.median(min(1.0, 600 / L) for L in lens if L > 0)
    print(f"\nM2  selected passages: n = {len(lens)} | fully within 600 chars: {fit:.1%} | "
          f"median fraction surviving the cap: {surv:.1%} | "
          f"median length {statistics.median(lens):.0f} chars, p90 {sorted(lens)[int(0.9*len(lens))]} chars")

    # ---------- M3: token budget of assembled router calls ----------
    m3 = {}
    if not args.skip_tokens:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
        def toklen(user):
            msgs = [{"role": "system", "content": RP.sys_cascade()},
                    {"role": "user", "content": user}]
            text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            return len(tok(text, add_special_tokens=False).input_ids)
        ret_lens, qa_lens = [], []
        for k, (i, inst, sel) in enumerate(rows):
            user = RP.INSTR_RET2.format(q=inst["question"], n_sel=len(sel),
                                        selections=RP.format_selections(sel, inst["ctxs_candidate"]))
            ret_lens.append(toklen(user))
            wa = qa_w.get(f"{i}|weak")
            if wa is not None:
                block = R.build_block(inst, sel)
                user_q = RP.INSTR_QA1.format(q=inst["question"],
                                             evidence=RP.format_evidence(block), weak_answer=wa)
                qa_lens.append(toklen(user_q))
            if (k + 1) % 1000 == 0:
                print(f"    tokenized {k+1}/{len(rows)} ...", flush=True)
        for name, L in (("RET-router call", ret_lens), ("QA-gate call", qa_lens)):
            if not L: continue
            over = sum(1 for x in L if x > RP.MAXLEN)
            s = sorted(L)
            m3[name] = {"n": len(L), "median": s[len(s)//2], "p99": s[int(0.99*len(s))],
                        "max": s[-1], "over_maxlen": over, "over_frac": over/len(L)}
            print(f"\nM3  {name}: median {m3[name]['median']} | p99 {m3[name]['p99']} | "
                  f"max {m3[name]['max']} | > {RP.MAXLEN}: {over} ({m3[name]['over_frac']:.2%})")
    else:
        print("\nM3 skipped (--skip-tokens)")

    report = {"agreement_covW": agreement, "M1_gold_visibility": m1,
              "M2": {"n_selected_passages": len(lens), "fit_600": fit, "median_survival": surv},
              "M3": m3, "caps_tested": [c if c else "uncapped" for c in CAPS],
              "decision_rule": "keep 600/8192 if M1@600>=~90% and M3 overflow <1%; else raise cap to the ~90% crossing, re-check M3, re-run probe"}
    with open(os.path.join(args.out, "budget_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nreport -> {os.path.join(args.out, 'budget_report.json')}")

if __name__ == "__main__":
    main()
