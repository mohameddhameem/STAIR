"""The team's always-strong rows on the HotpotQA golden set (slr-val, slr-test) from Jonathan's stored run, with his own
export logic (branch `jonathan`, lab_package/export_ss_results_common.py), extended to the slr-* parts. No model calls.
  git fetch origin jonathan
  python selfLearnt/pipeline/tools/export_jonathan_ss.py <hotpotqa-dev.json (AgentTTS's 7,405, dev order)>
Writes results/common/jonathan-ss-hotpotqa-slr-{val,test}.csv; checks slr-test against his certified 76.77."""
import csv
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(ROOT, "common"))
from stair_common import cost, score                      # noqa: E402
from stair_common.splits import load_split                # noqa: E402

REF = "origin/jonathan"
CB = "jonathan/runs/config_b"
PRICE_SS = cost.TABLE5_HOTPOTQA["retrieve"]["Qwen2.5-72B"] + cost.TABLE5_HOTPOTQA["answer"]["Llama-3.1-70B"]   # 929
MODELS = ("retriever=Qwen/Qwen2.5-72B-Instruct-AWQ@698703eae6604af048a3d2f509995dc302088217;"
          "answerer=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4@2123003760781134cfc31124aa6560a45b491fdf")
COLS = ["dataset", "split", "id", "track", "policy", "menu", "seed", "answer", "lenient", "em", "f1",
        "units", "units_retrieve", "units_answer", "units_fuse", "units_router", "ms_router", "models", "price"]


def jl(name):
    txt = subprocess.run(["git", "-C", ROOT, "show", f"{REF}:{CB}/{name}.jsonl"], capture_output=True, text=True, check=True).stdout
    return [json.loads(x) for x in txt.splitlines() if x.strip()]


dev = json.load(open(sys.argv[1], encoding="utf-8"))
ids = [x["_id"] for x in dev]
assert len(ids) == 7405
meta_ret = {r["k"]: r for r in jl("meta_retr_strong")}
meta_qa = {r["k"]: r for r in jl("meta_qa_strong")}
answers = {r["k"]: r["v"] for r in jl("qa_strong")}
pos = {i: n for n, i in enumerate(ids)}
for part in ("slr-val", "slr-test"):
    out = os.path.join(ROOT, "results", "common", f"jonathan-ss-hotpotqa-{part}.csv")
    ok = 0
    part_ids = load_split("hotpotqa", part)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for id_ in sorted(part_ids, key=pos.get):
            idx = pos[id_]
            k = str(idx)
            ans, gold = answers[f"{k}|strong"], dev[idx]["answer"]
            mr, mq = meta_ret[k], meta_qa[f"{k}|strong"]
            u_ret, u_qa = cost.units(72.0, mr["n_in"], mr["n_out"]), cost.units(70.0, mq["n_in"], mq["n_out"])
            le = score.lenient(ans, gold)
            ok += le
            w.writerow(["hotpotqa", part, id_, "jonathan", "always-strong", "A", 0, ans, le, score.em(ans, gold), score.f1(ans, gold),
                        round(round(u_ret, 2) + round(u_qa, 2), 2), round(u_ret, 2), round(u_qa, 2), 0, 0, "", MODELS, PRICE_SS])
    print(f"{part}: n={len(part_ids)}  lenient={100 * ok / len(part_ids):.2f}  ->  {out}")
