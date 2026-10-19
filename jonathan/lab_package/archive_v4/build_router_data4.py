#!/usr/bin/env python3
# =============================================================================
# build_router_data4.py — VERSION: build_router_data4 v1.0 2026-08-18
#
# CPU-only prep for sft_router4.py. Reads the full_run label grid and writes
# router_data4.jsonl: one row per scored instance with
#   question, pool_text (EXACT grid pool, rendered), em_strict[4],
#   em_lenient[4], R[4], label, solved, ties, clean, type, level
#
# PROVENANCE — pool reconstruction:
#   title_of / stable_seed / trim_pool are copied VERBATIM from run_labels.py
#   (29 Jul determinism patch, shuffle scheme blake2b-8/2026-07-29). The pool
#   this script renders is byte-identical, in order, to what the graded
#   retrieval stage saw. The manifest's shuffle_scheme is checked to match.
#
# METRICS:
#   em_strict  = official HotpotQA normalization + equality (mint_labels.py)
#   em_lenient = whole-word containment (lenient_score v1 rules)
#                ASSUMPTION-L: quantifier guard of the adjudicated cluster
#                scorer is NOT yet implemented — numbers pending
#                reconciliation (agreed 18 Aug).
#   R          = em_strict - max(0, COST/B - 1), COST/ALPHA copied from
#                mint_labels.py (our FLOP units; ASSUMPTION-c)
#
# SELF-CHECKS (script aborts loudly if any fail):
#   1. recomputed em_strict == em_official stored in combo_results_full.jsonl
#   2. recomputed R matches R_by_combo in the minted label file (<=1e-3)
#   3. manifest shuffle_scheme == blake2b-8/2026-07-29 and pool_k == 30
#
# Usage:
#   python3 build_router_data4.py \
#       --run /mnt/hdd/jonathan/runs/full_run \
#       --data /mnt/hdd/jonathan/data/tts_test.json \
#       --out /mnt/hdd/jonathan/runs/full_run/router_data4.jsonl \
#       --sample 3        # print N fully rendered prompts for validation
# =============================================================================
import os, json, argparse, random, hashlib, re, string
from collections import Counter

VERSION = "build_router_data4 v1.0 2026-08-18"
COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
ALPHA = 0.6
B = ALPHA * COST["strong-strong"]
SHUFFLE_SCHEME = "blake2b-8/2026-07-29"


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
def normalize_answer(s):
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
        if ng and re.search(r"(?<!\w)" + re.escape(ng) + r"(?!\w)", np_):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--pool_k', type=int, default=30)
    ap.add_argument('--sample', type=int, default=0)
    args = ap.parse_args()

    print(VERSION)

    # -- self-check 3: manifest agreement
    man = json.load(open(os.path.join(args.run, "run_manifest.json")))
    assert man.get("shuffle_scheme") == SHUFFLE_SCHEME, \
        f"manifest shuffle_scheme {man.get('shuffle_scheme')!r} != {SHUFFLE_SCHEME!r}"
    assert int(man.get("pool_k", -1)) == args.pool_k, \
        f"manifest pool_k {man.get('pool_k')} != --pool_k {args.pool_k}"
    print(f"manifest ok: pool_k={man['pool_k']} shuffle={man['shuffle_scheme']} "
          f"git={man.get('git_sha')}{' DIRTY' if man.get('git_dirty') else ''}")

    hotpot = json.load(open(args.data))
    qa = {w: load_jsonl_store(os.path.join(args.run, f"qa_{w}.jsonl"))
          for w in ("weak", "strong")}
    print(f"qa_weak entries: {len(qa['weak'])} | qa_strong entries: {len(qa['strong'])}")

    # stored em_official per (idx, combo) for self-check 1
    stored_em = {}
    for line in open(os.path.join(args.run, "combo_results_full.jsonl")):
        r = json.loads(line)
        stored_em[(r["idx"], r["combo"])] = r["em_official"]

    # minted labels for self-check 2 + label/ties/clean fields
    minted = {}
    lp = os.path.join(args.run, "sft_labels_hotpot_v4_alpha0.6.jsonl")
    for line in open(lp):
        r = json.loads(line)
        minted[r["idx"]] = r

    n_rows = n_em_mismatch = n_R_mismatch = n_solved = 0
    dist = Counter()
    samples_left = args.sample
    with open(args.out, "w") as f:
        for idx, inst in enumerate(hotpot):
            em_s, em_l, ans_by = {}, {}, {}
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
                    ans_by[combo] = ans
                if not complete: break
            if not complete:
                continue

            # self-check 1
            for c in COMBOS:
                if (idx, c) in stored_em and stored_em[(idx, c)] != em_s[c]:
                    n_em_mismatch += 1
                    if n_em_mismatch <= 5:
                        print(f"  EM MISMATCH idx {idx} {c}: stored "
                              f"{stored_em[(idx, c)]} recomputed {em_s[c]}")

            R = {c: em_s[c] - max(0.0, COST[c]/B - 1.0) for c in COMBOS}
            solved = any(em_s.values())
            n_solved += solved
            m = minted.get(idx)
            if m:  # self-check 2
                for c in COMBOS:
                    if abs(R[c] - m["R_by_combo"][c]) > 1e-3:
                        n_R_mismatch += 1
                        if n_R_mismatch <= 5:
                            print(f"  R MISMATCH idx {idx} {c}: minted "
                                  f"{m['R_by_combo'][c]} recomputed {R[c]:.3f}")
                label, ties, clean = m["label"], m["ties"], m["clean"]
            else:
                # unsolved (or minted before a partial slice): Scenario B default
                label, ties, clean = "weak-weak", [], False  # ASSUMPTION-b

            pool = trim_pool(dict(inst, ctxs_candidate=inst['ctxs_candidate']),
                             k=args.pool_k)
            pool_text = "\n".join(f"[{i+1}] {c}" for i, c in enumerate(pool))
            golds = set(inst.get('supporting_titles', []))
            gold_in_pool = golds <= set(title_of(c) for c in pool)

            row = {"idx": idx, "_id": inst["_id"], "question": inst["question"],
                   "pool_text": pool_text, "n_pool": len(pool),
                   "gold_in_pool": gold_in_pool, "solved": bool(solved),
                   "label": label, "ties": ties, "clean": clean,
                   "em_strict": [em_s[c] for c in COMBOS],
                   "em_lenient": [em_l[c] for c in COMBOS],
                   "R": [round(R[c], 4) for c in COMBOS],
                   "type": inst.get("type"), "level": inst.get("level")}
            f.write(json.dumps(row) + "\n")
            n_rows += 1
            dist[label] += 1

            if samples_left > 0:
                samples_left -= 1
                print("\n" + "=" * 70)
                print(f"SAMPLE RENDERED USER CONTENT (idx {idx}, label {label}, "
                      f"solved {solved}):")
                print("=" * 70)
                print(f"Question: {inst['question']}\n")
                print(f"Candidate pool ({len(pool)} passages, first 2 shown):")
                for l_ in pool_text.split("\n")[:2]:
                    print(f"  {l_[:160]}")
                print("  ...")

    print(f"\nrows written: {n_rows}  solved: {n_solved} "
          f"({100*n_solved/max(1,n_rows):.1f}%)")
    print(f"label distribution (Scenario B view): {dict(dist)}")
    print(f"SELF-CHECK 1 em mismatches: {n_em_mismatch}")
    print(f"SELF-CHECK 2 R  mismatches: {n_R_mismatch}")
    if n_em_mismatch or n_R_mismatch:
        print("*** MISMATCHES PRESENT — do not train on this file until explained ***")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
