#!/usr/bin/env python3
"""
twowiki_cells_adapt.py -- prepare AgentTTS's 2Wiki file (their 100) for the UNCHANGED Config B grid runner
(~/slr/run_labels_agenttts.py, the script that produced the HotpotQA matched-model labels). DRAFT 31 Oct 2026.

Why an adapter and not a runner change: the four cells on 2Wiki must be the HotpotQA pipeline byte for byte --
prompts, decoding (K = 1 greedy), models, phases, stores. Only the data differs, in three harmless ways:
  1. tts_test.json has no `_id`            -> the runner keys its JSONL stores by `_id`; we assign 2wiki100-000 .. 099
                                              (position in the file = dev[0:100] order; registry ids are joined later)
  2. gold passages are given in full (`ctxs_gt`), not as `supporting_titles`
                                           -> the runner's gold re-injection / coverage check uses title_of(c) =
                                              c.split(':',1)[0]; 2Wiki passages are bare sentences, so title_of(c) == c
                                              (or its prefix before a colon) consistently for pool and gold strings.
                                              We set supporting_titles = [title_of(g) for g in ctxs_gt].
  3. pools are 100-104 passages           -> run with SLR_POOL_K >= max pool size so NOTHING is trimmed (the sbatch
                                              sets SLR_POOL_K=110); this script asserts every gold is in its pool.
Nothing is changed inside a passage, question or answer list. Output: <out>/tts_test_2wiki_adapted.json + report.
"""
import argparse, hashlib, json, os


def title_of(c): return c.split(':', 1)[0]          # identical to run_labels_agenttts.title_of


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="AgentTTS data/2wikimultihopqa/tts_test.json (the 100)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    data = json.load(open(a.src, encoding="utf-8"))
    assert len(data) == 100, len(data)
    out, maxpool, covered = [], 0, 0
    for i, x in enumerate(data):
        pool = x["ctxs_candidate"]; golds = x["ctxs_gt"]
        assert all(g in pool for g in golds), f"instance {i}: a gold passage is not in its pool"
        y = dict(x)
        y["_id"] = f"2wiki100-{i:03d}"
        y["supporting_titles"] = [title_of(g) for g in golds]
        y["answer"] = x["answers"][0]                 # the runner's label factories read `answer`; the scorer uses `answers`
        maxpool = max(maxpool, len(pool))
        covered += all(title_of(g) in {title_of(c) for c in pool} for g in golds)
        out.append(y)
    os.makedirs(a.out, exist_ok=True)
    dst = os.path.join(a.out, "tts_test_2wiki_adapted.json")
    json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
    rep = {"n": len(out), "max_pool": maxpool, "gold_covered_by_title_of": covered,
           "src_sha256": hashlib.sha256(open(a.src, "rb").read()).hexdigest(),
           "dst_sha256": hashlib.sha256(open(dst, "rb").read()).hexdigest(),
           "note": "set SLR_POOL_K >= max_pool (sbatch uses 110) so the runner trims nothing"}
    json.dump(rep, open(os.path.join(a.out, "adapt_report.json"), "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
