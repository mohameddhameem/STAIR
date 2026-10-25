"""AL-02 diagnosis: score Llama 70B's own answer (before the fuser) against the fused answer, on the same 500 test questions.
Separates the fuser's effect on accuracy from the answer temperature's. CPU only, reads saved outputs, a few seconds.
  bash -c 'source jobs/cluster.env; source jobs/cluster.local.env; python jobs/al02_split.py'"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stair import agenttts as T, data as D                       # noqa: E402
from stair.common import atts_merge, key, read_jsonl, scores     # noqa: E402
from stair.config import load_config                             # noqa: E402

cfg = load_config(os.environ.get("CONFIG", "configs/default.yaml"))
R = {r["qid"]: r for r in read_jsonl(T.f_ret(cfg, "test", "q72"))}
A = {(r["qid"], tuple(r["skey"])): r for r in read_jsonl(T.f_ans(cfg, "test", "l70"))}
F = {(r["qid"], tuple(r["skey"]), r["k"]): r for r in read_jsonl(T.f_fuse(cfg, "test", "l70"))}
n = raw_ok = fused_ok = only_raw = only_fused = 0
wr = wf = 0
for q in D.load(cfg, "test"):
    r = R.get(q["qid"])
    if not r:
        continue
    sk = key(atts_merge([s["text"] for s in r["samples"][:1]]))
    a, f = A.get((q["qid"], sk)), F.get((q["qid"], sk, 1))
    if not (a and f):
        continue
    raw = a["samples"][0]["text"]
    s_raw, s_f = scores(raw, q["answer"])["lenient"], scores(f["text"], q["answer"])["lenient"]
    n += 1
    raw_ok += s_raw
    fused_ok += s_f
    only_raw += s_raw > s_f
    only_fused += s_f > s_raw
    wr += len(raw.split())
    wf += len(f["text"].split())
print(f"{n} test questions (Qwen 72B x1 -> Llama 70B x1, answer at temperature 0.7)")
print(f"  70B answer, no fuser : lenient {raw_ok / n:.3f}   {wr / n:.0f} words on average")
print(f"  after the 70B fuser  : lenient {fused_ok / n:.3f}   {wf / n:.0f} words on average")
print(f"  right before the fuser, wrong after: {only_raw}   wrong before, right after: {only_fused}")
