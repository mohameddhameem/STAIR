#!/usr/bin/env python3
# =============================================================================
# build_router_data.py — VERSION: build_router_data v1.2 2026-09-08 (label_metric + unicode fold + yes/no guard)
#
# CPU-only data builder for sft_router.py (stage-level routing, per Prof. Pan
# 22 Aug). Reads a label grid directory written by run_labels4.py and writes
# ONE row per instance carrying everything the router pipeline needs:
#
#   question, pool_text (EXACT grid pool, rendered), n_pool,
#   evidence_weak / evidence_strong (each retriever's own output text —
#       "Query: ...\nSelected references: ..." — the QA-stage router input),
#   em_strict[4], em_lenient[4], R[4] (cells in COMBOS order),
#   joint_label            (cheapest successful cell, tie-break applied; None if unresolved)
#   ret_label              ('W' | 'S' | None)            <- Phase 1 target
#   qa_label_W / qa_label_S ('W' | 'S' | None)           <- Phase 2 targets, one per retrieval choice
#   split, type, level
#
# LABEL RULES (deck slide 8, decided 31 Aug):
#   1. Run all four cells (the grid).  2. joint = the cheapest SUCCESSFUL cell
#   (strict EM).  3. Cost ties (WS vs SW, both 5.7 units) -> --tie_break
#   (default SW: prefer strong retrieval; WS reproduces the pure stage-local
#   rule).  4. Decompose: ret_label = retrieval half of joint;
#   qa_label_X = cheapest answerer that succeeds on retriever X's references
#   (W if X-weak succeeds, else S if X-strong succeeds, else None).
#   Unresolved (nothing succeeds) -> ret_label None; unresolved under X ->
#   qa_label_X None. Dropping them from TRAINING is Option 1 (slide 8);
#   sft_router.py's --scenario B can override to 'W' (retired fail-cheap rule).
#   Ties cannot occur for the binary stage labels; the tie-break only decides
#   WHICH retrieval choice the joint label selects.
#
# PROVENANCE — pool reconstruction: title_of / stable_seed / trim_pool are the
#   VERBATIM copies from run_labels.py (shuffle scheme as-given/no-shuffle),
#   so pool_text is byte-identical, in order, to what the graded retrieval
#   stage saw. pool_k is read from the grid manifest and must equal --pool_k.
#
# SELF-CHECKS (abort loudly on failure):
#   1. recomputed em_strict == em_official stored in combo_results_full.jsonl
#   2. manifest shuffle_scheme == as-given/no-shuffle and pool_k == --pool_k
#   3. every instance has BOTH evidence texts (retr_weak/strong.jsonl)
#   4. every emitted idx belongs to exactly one split
#   5. label consistency: ret_label is None iff no cell succeeded;
#      qa_label_X is None iff no X-cell succeeded
#
# Usage (lab, venv active):
#   python3 build_router_data.py \
#       --run   /mnt/hdd/jonathan/runs/full_run_p100 \
#       --data  /mnt/hdd/jonathan/data/tts_test.json \
#       --splits /mnt/hdd/jonathan/slr/stage-level-routing/lab_package/splits.json \
#       --out   /mnt/hdd/jonathan/runs/full_run_p100/router_data.jsonl \
#       --pool_k 100 --tie_break SW --sample 3
# Writes <out> and <out dir>/build_manifest.json (counts, per-stage class
# weights, tie-break count) — the numbers slide 4 / slide 8 quote.
# =============================================================================
import os, json, argparse, random, hashlib, re, string
from collections import Counter

VERSION = "build_router_data v1.2 2026-09-08 (label_metric + unicode fold + yes/no guard)"
COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
ALPHA = 0.6
B = ALPHA * COST["strong-strong"]
SHUFFLE_SCHEME = "as-given/no-shuffle"
RET_OF = {"weak-weak": "W", "weak-strong": "W", "strong-weak": "S", "strong-strong": "S"}


# ---- copied VERBATIM from run_labels.py (do not edit) ------------------------
def title_of(c): return c.split(':', 1)[0]

def stable_seed(s):
    return int(hashlib.blake2b(s.encode(), digest_size=8).hexdigest(), 16)

def trim_pool(inst, k=30):
    cands = inst['ctxs_candidate']
    golds = inst.get('supporting_titles', [])
    pool = list(cands[:k])
    have = set(title_of(c) for c in pool)
    for t in golds:
        if t not in have:
            g = next((c for c in cands if title_of(c) == t), None)
            if g:
                for j in range(k - 1, -1, -1):
                    if title_of(pool[j]) not in golds:
                        pool[j] = g; break
                have.add(t)
    rnd = random.Random(42 + stable_seed(inst['_id']) % 100000)
    rnd.shuffle(pool)
    return pool
# ---- end verbatim copy -------------------------------------------------------


# ---- strict EM: copied from mint_labels.py ----------------------------------
UNI_FOLD = str.maketrans({"\u2013": "-", "\u2014": "-",
                          "\u2019": "'", "\u2018": "'",
                          "\u201c": '"', "\u201d": '"',
                          "\u00b2": "2"})

def normalize_answer(s):
    s = s.translate(UNI_FOLD)  # v1.2: fold unicode punctuation, then official steps
    s = ''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def em_official(pred, golds):
    if isinstance(golds, str): golds = [golds]
    p = normalize_answer(pred)
    return int(any(p == normalize_answer(g) for g in golds))


# ---- lenient: whole-word containment (lenient_score v1 rules) ---------------
def em_lenient(pred, golds):
    if isinstance(golds, str): golds = [golds]
    np_ = normalize_answer(pred)
    for g in golds:
        ng = normalize_answer(g)
        if not ng: continue
        if ng in ("yes", "no"):  # v1.2 guard: a yes/no gold gets no credit if the answer also contains the opposite
            opp = "no" if ng == "yes" else "yes"
            if re.search(r"(?<!\w)" + opp + r"(?!\w)", np_):
                continue
        if re.search(r"(?<!\w)" + re.escape(ng) + r"(?!\w)", np_):
            return 1
    return 0


def load_jsonl_store(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            try:
                r = json.loads(line); d[r['k']] = r['v']
            except Exception:
                pass
    return d


# ---- the label rule (slide 8) ------------------------------------------------
def joint_label(em, tie_break):
    """Cheapest successful cell under strict EM; cost ties broken by tie_break
    ('SW' -> strong-weak, 'WS' -> weak-strong). Returns (combo, tie_hit)."""
    ok = [c for c in COMBOS if em[c] == 1]
    if not ok:
        return None, False
    cmin = min(COST[c] for c in ok)
    cheapest = [c for c in ok if COST[c] == cmin]
    if len(cheapest) == 1:
        return cheapest[0], False
    pref = "strong-weak" if tie_break == "SW" else "weak-strong"
    return (pref if pref in cheapest else cheapest[0]), True

def qa_label_given(em, ret_choice):
    """Cheapest answerer that succeeds on retriever `ret_choice`'s references."""
    src = "weak" if ret_choice == "W" else "strong"
    if em[f"{src}-weak"] == 1:
        return "W"
    if em[f"{src}-strong"] == 1:
        return "S"
    return None


def _selftest():
    cases = [
        ("Armenian-American", "Armenian\u2013American", 1),          # en-dash gold
        ("Wild Hearts Can't Be Broken", "Wild Hearts Can\u2019t Be Broken", 1),
        ("$10-20 million", "$10\u201320 million", 1),
        ("14 million km2", "14 million km\u00b2", 1),
        ("Yes, both are American.", "yes", 1),                        # prose yes still credited
        ("yes and no", "yes", 0),                                     # contradiction guard
        ("No, one is Australian", "yes", 0),                          # opposite only
        ("human blood, platelets, and plasma", "blood, platelets, and plasma", 1),
        ("Ted 2", "Ted", 1),  # inherent cover-EM property: gold appearing inside a longer title IS credited (faithful to authors' rule; documented)
        ("nothing", "no", 0),                                         # word boundary
    ]
    ok = True
    for pred, gold, want in cases:
        got = em_lenient(pred, gold)
        print(f"  selftest {'OK ' if got==want else 'FAIL'} lenient({pred!r}, {gold!r}) = {got} (want {want})")
        ok &= (got == want)
    assert ok, "SELFTEST FAILED"
    print("selftest: all pinned cases pass")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='grid directory from run_labels4.py')
    ap.add_argument('--data', required=True, help='tts_test.json (7,405-instance pool file)')
    ap.add_argument('--splits', required=True, help='frozen splits.json (index-based)')
    ap.add_argument('--out', required=True)
    ap.add_argument('--pool_k', type=int, default=100, help='must equal the grid manifest pool_k')
    ap.add_argument('--label_metric', choices=['strict', 'lenient'], default='strict',
                    help='which EM vector mints the labels; both vectors are stored either way')
    ap.add_argument('--tie_break', choices=['SW', 'WS'], default='SW',
                    help='cost-tie rule for the joint label (slide 8): SW = prefer strong retrieval')
    ap.add_argument('--sample', type=int, default=0, help='print N worked examples')
    args = ap.parse_args()
    print(VERSION)
    _selftest()

    # -- self-check 2: manifest agreement
    man = json.load(open(os.path.join(args.run, "run_manifest.json")))
    assert man.get("shuffle_scheme") == SHUFFLE_SCHEME, \
        f"manifest shuffle_scheme {man.get('shuffle_scheme')!r} != {SHUFFLE_SCHEME!r}"
    assert int(man.get("pool_k", -1)) == args.pool_k, \
        f"manifest pool_k {man.get('pool_k')} != --pool_k {args.pool_k}"
    print(f"manifest ok: pool_k={man['pool_k']} maxlen={man.get('maxlen')} dtype={man.get('dtype')} "
          f"shuffle={man['shuffle_scheme']} git={man.get('git_sha')}"
          f"{' DIRTY' if man.get('git_dirty') else ''}")

    hotpot = json.load(open(args.data))
    sp = json.load(open(args.splits))["splits"]
    split_of = {}
    for name in ("train", "val", "test"):
        for i in sp[name]:
            assert i not in split_of, f"idx {i} in two splits"   # self-check 4
            split_of[i] = name

    qa = {w: load_jsonl_store(os.path.join(args.run, f"qa_{w}.jsonl")) for w in ("weak", "strong")}
    retr = {w: load_jsonl_store(os.path.join(args.run, f"retr_{w}.jsonl")) for w in ("weak", "strong")}
    print(f"qa_weak {len(qa['weak'])} | qa_strong {len(qa['strong'])} | "
          f"retr_weak {len(retr['weak'])} | retr_strong {len(retr['strong'])}")

    stored_em = {}
    for line in open(os.path.join(args.run, "combo_results_full.jsonl")):
        r = json.loads(line)
        stored_em[(r["idx"], r["combo"])] = r["em_official"]

    n_rows = n_em_mismatch = n_missing_ev = n_tie = 0
    ret_dist = Counter(); qa_dist = {"W": Counter(), "S": Counter()}
    ret_dist_split = {s: Counter() for s in ("train", "val", "test")}
    qa_dist_split = {s: {"W": Counter(), "S": Counter()} for s in ("train", "val", "test")}
    joint_dist = Counter()
    n_unresolved = 0
    samples_left = args.sample

    with open(args.out, "w") as f:
        for idx, inst in enumerate(hotpot):
            em_s, em_l = {}, {}
            complete = True
            for rsrc in ("weak", "strong"):
                for qsrc in ("weak", "strong"):
                    key = f"{idx}|{rsrc}"
                    if key not in qa[qsrc]:
                        complete = False; break
                    ans = qa[qsrc][key]
                    combo = f"{rsrc}-{qsrc}"
                    em_s[combo] = em_official(ans, inst['answers'])
                    em_l[combo] = em_lenient(ans, inst['answers'])
                if not complete: break
            if not complete:
                continue

            # self-check 1
            for c in COMBOS:
                if (idx, c) in stored_em and stored_em[(idx, c)] != em_s[c]:
                    n_em_mismatch += 1
                    if n_em_mismatch <= 5:
                        print(f"  EM MISMATCH idx {idx} {c}: stored {stored_em[(idx, c)]} recomputed {em_s[c]}")

            # self-check 3: evidence for both retrieval choices
            ev_w, ev_s = retr["weak"].get(str(idx)), retr["strong"].get(str(idx))
            if not ev_w or not ev_s:
                n_missing_ev += 1
                if n_missing_ev <= 5:
                    print(f"  MISSING EVIDENCE idx {idx}: weak={bool(ev_w)} strong={bool(ev_s)}")
                continue

            em_lab = em_s if args.label_metric == 'strict' else em_l  # v1.2: label-driving vector
            R = {c: em_lab[c] - max(0.0, COST[c] / B - 1.0) for c in COMBOS}
            joint, tie_hit = joint_label(em_lab, args.tie_break)
            n_tie += tie_hit
            ret_label = RET_OF[joint] if joint else None
            qa_W = qa_label_given(em_lab, "W")
            qa_S = qa_label_given(em_lab, "S")

            # self-check 5: label consistency
            any_ok = any(em_lab.values())
            assert (ret_label is None) == (not any_ok), f"idx {idx}: ret_label/unresolved disagree"
            assert (qa_W is None) == (em_lab["weak-weak"] == 0 and em_lab["weak-strong"] == 0)
            assert (qa_S is None) == (em_lab["strong-weak"] == 0 and em_lab["strong-strong"] == 0)

            pool = trim_pool(dict(inst, ctxs_candidate=inst['ctxs_candidate']), k=args.pool_k)
            pool_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(pool))
            golds = set(inst.get('supporting_titles', []))
            split = split_of.get(idx)

            row = {"idx": idx, "_id": inst["_id"], "split": split,
                   "question": inst["question"],
                   "pool_text": pool_text, "n_pool": len(pool),
                   "gold_in_pool": golds <= set(title_of(c) for c in pool),
                   "evidence_W": ev_w, "evidence_S": ev_s,
                   "em_strict": [em_s[c] for c in COMBOS],
                   "em_lenient": [em_l[c] for c in COMBOS],
                   "R": [round(R[c], 4) for c in COMBOS],
                   "joint_label": joint, "joint_tie": tie_hit,
                   "ret_label": ret_label, "qa_label_W": qa_W, "qa_label_S": qa_S,
                   "type": inst.get("type"), "level": inst.get("level")}
            f.write(json.dumps(row) + "\n")
            n_rows += 1
            if joint: joint_dist[joint] += 1
            else: n_unresolved += 1
            if ret_label:
                ret_dist[ret_label] += 1
                if split: ret_dist_split[split][ret_label] += 1
            for ch, lab in (("W", qa_W), ("S", qa_S)):
                if lab:
                    qa_dist[ch][lab] += 1
                    if split: qa_dist_split[split][ch][lab] += 1

            if samples_left > 0:
                samples_left -= 1
                print("\n" + "=" * 70)
                print(f"WORKED EXAMPLE idx {idx} (split {split})  EM cells "
                      f"WW/WS/SW/SS = {[em_s[c] for c in COMBOS]}")
                print(f"  joint = {joint}{' (tie -> ' + args.tie_break + ')' if tie_hit else ''}  "
                      f"-> ret_label {ret_label} · qa|W {qa_W} · qa|S {qa_S}")
                print(f"  Q: {inst['question'][:120]}")
                print(f"  evidence_W: {ev_w[:160].replace(chr(10), ' | ')}…")
                print(f"  evidence_S: {ev_s[:160].replace(chr(10), ' | ')}…")

    # per-stage class weights over the TRAIN split (what sft_router.py will recompute)
    def weights(cnt):
        n = sum(cnt.values())
        return {c: round(n / (2 * max(1, cnt.get(c, 0))), 3) for c in ("W", "S")}
    tr_ret = ret_dist_split["train"]
    tr_qa = Counter(); tr_qa.update(qa_dist_split["train"]["W"]); tr_qa.update(qa_dist_split["train"]["S"])

    bm = {"version": VERSION, "label_metric": args.label_metric, "run": args.run, "grid_manifest": man,
          "pool_k": args.pool_k, "tie_break": args.tie_break,
          "instances_written": n_rows, "unresolved": n_unresolved,
          "unresolved_pct": round(100 * n_unresolved / max(1, n_rows), 1),
          "joint_label_dist": dict(joint_dist), "tie_break_applied": n_tie,
          "ret_label_dist_all": dict(ret_dist),
          "qa_label_dist_all": {k: dict(v) for k, v in qa_dist.items()},
          "train": {"ret_examples": sum(tr_ret.values()), "ret_dist": dict(tr_ret),
                    "ret_class_weights": weights(tr_ret),
                    "qa_examples_both_choices": sum(tr_qa.values()), "qa_dist": dict(tr_qa),
                    "qa_class_weights": weights(tr_qa),
                    "qa_dist_by_choice": {k: dict(v) for k, v in qa_dist_split["train"].items()}},
          "val": {"ret_examples": sum(ret_dist_split["val"].values()),
                  "qa_examples_both_choices": sum(sum(v.values()) for v in qa_dist_split["val"].values())},
          "test": {"instances": sum(1 for i in split_of if split_of[i] == "test"),
                   "ret_labeled": sum(ret_dist_split["test"].values())},
          "self_checks": {"em_mismatches": n_em_mismatch, "missing_evidence": n_missing_ev}}
    json.dump(bm, open(os.path.join(os.path.dirname(os.path.abspath(args.out)), "build_manifest.json"), "w"), indent=2)

    print(f"\ninstances written: {n_rows}  unresolved: {n_unresolved} ({bm['unresolved_pct']}%)")
    print(f"joint label dist: {dict(joint_dist)}   tie-break applied: {n_tie}")
    print(f"TRAIN retrieval examples: {bm['train']['ret_examples']} {dict(tr_ret)}  weights {bm['train']['ret_class_weights']}")
    print(f"TRAIN QA examples (both choices): {bm['train']['qa_examples_both_choices']} {dict(tr_qa)}  weights {bm['train']['qa_class_weights']}")
    print(f"SELF-CHECK 1 em mismatches: {n_em_mismatch}   SELF-CHECK 3 missing evidence: {n_missing_ev}")
    if n_em_mismatch or n_missing_ev:
        print("*** CHECKS FAILED — do not train on this file until explained ***")
    print(f"wrote {args.out} and build_manifest.json")


if __name__ == "__main__":
    main()
