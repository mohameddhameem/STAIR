"""
Stage-Level Routing — score the full run (official EM) and mint labels v4-full.
Pure CPU; run after run_labels.py finishes (or on a partial slice — it scores what exists).

  python mint_labels.py --data /mnt/hdd/jonathan/data/tts_test.json --out /mnt/hdd/jonathan/slr_run
Outputs in --out:
  combo_results_full.jsonl       every (instance x combo) outcome, official EM
  sft_labels_hotpot_v4_alpha0.6.jsonl   argmax-R labels for solved instances
  mint_summary.json              config EMs, solved rate, label distribution
"""
import os, json, argparse, re, string
from collections import Counter

COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
ALPHA = 0.6
B = ALPHA * COST["strong-strong"]

def normalize_answer(s):
    s = ''.join(ch for ch in s.lower() if ch not in set(string.punctuation))
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    return ' '.join(s.split())

def em_official(pred, golds):
    if isinstance(golds, str): golds = [golds]
    p = normalize_answer(pred)
    return int(any(p == normalize_answer(g) for g in golds))

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
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()

    hotpot = json.load(open(args.data))
    qa = {w: load_jsonl_store(os.path.join(args.out, f"qa_{w}.jsonl")) for w in ("weak", "strong")}
    print(f"qa_weak entries: {len(qa['weak'])} | qa_strong entries: {len(qa['strong'])}")

    combo_path = os.path.join(args.out, "combo_results_full.jsonl")
    label_path = os.path.join(args.out, f"sft_labels_hotpot_v4_alpha{ALPHA}.jsonl")
    ce = Counter(); n_scored = solved_n = 0
    dist = Counter(); clean_n = 0

    with open(combo_path, 'w') as cf, open(label_path, 'w') as lf:
        for idx, inst in enumerate(hotpot):
            emap = {}
            complete = True
            for rsrc in ("weak", "strong"):
                for qsrc in ("weak", "strong"):
                    key = f"{idx}|{rsrc}"
                    if key not in qa[qsrc]:
                        complete = False; break
                    ans = qa[qsrc][key]
                    em = em_official(ans, inst['answers'])
                    combo = f"{rsrc}-{qsrc}"
                    emap[combo] = em
                    cf.write(json.dumps({"idx": idx, "_id": inst['_id'], "combo": combo,
                                         "answer": ans[:80], "gold": inst['answers'][0],
                                         "em_official": em, "type": inst.get('type'),
                                         "level": inst.get('level'),
                                         "bm25_gold_recall": inst.get('bm25_gold_recall')}) + '\n')
                if not complete: break
            if not complete:
                continue
            n_scored += 1
            for c in COMBOS: ce[c] += emap[c]
            if not any(emap.values()):
                continue
            solved_n += 1
            R = {c: emap[c] - max(0.0, COST[c]/B - 1.0) for c in COMBOS}
            best = max(R, key=R.get)
            ties = [c for c in COMBOS if abs(R[c] - R[best]) < 1e-9]
            is_clean = (len(ties) == 1)
            is_binary = str(inst.get('type', '')).lower() == 'comparison'
            lf.write(json.dumps({"dataset": "hotpotqa", "idx": idx, "_id": inst['_id'],
                                 "question": inst['question'], "label": best,
                                 "clean": is_clean, "ties": ties, "binary_flag": is_binary,
                                 "em_by_combo": emap,
                                 "R_by_combo": {c: round(R[c], 3) for c in COMBOS},
                                 "alpha": ALPHA, "type": inst.get('type'),
                                 "level": inst.get('level')}) + '\n')
            dist[best] += 1; clean_n += is_clean

    summary = {
        "instances_scored": n_scored,
        "config_em": {c: ce[c] for c in COMBOS},
        "config_em_pct": {c: round(100*ce[c]/max(1, n_scored), 1) for c in COMBOS},
        "best_fixed": max(COMBOS, key=lambda c: ce[c]),
        "solved": solved_n,
        "solved_pct": round(100*solved_n/max(1, n_scored), 1),
        "labels_minted": sum(dist.values()),
        "clean": clean_n,
        "label_distribution": dict(dist),
        "alpha": ALPHA,
    }
    json.dump(summary, open(os.path.join(args.out, "mint_summary.json"), 'w'), indent=2)
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {combo_path}\nwrote {label_path}")

if __name__ == '__main__':
    main()
