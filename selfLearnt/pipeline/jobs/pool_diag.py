"""Where the two gold passages sit in each passage set, and how much they stand out from the distractors. Explains why
AgentTTS (untrained models) does better on its own passages (tts_test.json) than on ours (BM25), while our fine-tuned
models do not care. CPU only, a few seconds.
  bash -c 'source jobs/cluster.env; source jobs/cluster.local.env; python jobs/pool_diag.py'"""
import json
import os
import re
import statistics as st

RUNS = os.path.expanduser(os.environ.get("STAIR_RUNS_DIR", "~/stair-runs"))
SETS = [("full_v1", "test", "our passages (BM25), old 500"), ("xcheck_old500", "test", "AgentTTS's passages, old 500"),
        ("golden_v1", "test", "our passages (BM25), slr-test"), ("golden_tts", "test", "AgentTTS's passages, slr-test")]
WORD = re.compile(r"[a-z0-9]+")
STOP = set("the a an of in on at to for and or is was were are be by with from as that this which who what when where how "
           "did does do it its his her their than then also".split())


def words(t):
    return {w for w in WORD.findall(t.lower()) if w not in STOP}


for run, sp, label in SETS:
    p = os.path.join(RUNS, run, "data", f"{sp}.json")
    if not os.path.exists(p):
        print(f"{label}: not found ({p})")
        continue
    qs = json.load(open(p))
    pos, top10, top20, ranks, n_pool = [], 0, 0, [], []
    for q in qs:
        g = q["ret_answers"]
        c = q["ctxs_candidate"]
        n_pool.append(len(c))
        if not g:
            continue
        pos += [i + 1 for i in g]
        top10 += all(i < 10 for i in g)
        top20 += all(i < 20 for i in g)
        qw = words(q["question"])
        ov = [len(qw & words(x)) for x in c]              # how many question words each passage shares
        order = sorted(range(len(c)), key=lambda i: -ov[i])
        rk = {i: r + 1 for r, i in enumerate(order)}
        ranks += [rk[i] for i in g]
    n = len(qs)
    print(f"{label}: {n} questions, {st.mean(n_pool):.0f} passages each")
    print(f"  gold position in the list: median {st.median(pos):.0f}, mean {st.mean(pos):.1f} (random would be ~50)")
    print(f"  both gold passages in the first 10: {100 * top10 / n:.0f}%, first 20: {100 * top20 / n:.0f}%")
    print(f"  gold rank by word overlap with the question: median {st.median(ranks):.0f} (1 = stands out most; "
          f"low = distractors look less like the answer)")
