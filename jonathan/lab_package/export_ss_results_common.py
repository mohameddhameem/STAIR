# export_ss_results_common.py -- always-strong (greedy SS) per-question rows for the
# team-standard HotpotQA val/test splits, in stair_common.schema layout + price column.
# Read-only over the Config B grid artifacts; writes two CSVs into the STAIR checkout.
# Usage: python3 export_ss_results_common.py /path/to/STAIR/checkout
import json, sys, hashlib, csv, os

STAIR = sys.argv[1]
sys.path.insert(0, os.path.join(STAIR, "common"))
from stair_common import score, cost

DEV = "/mnt/hdd/jonathan/agenttts_verify/data/HotpotQA/hotpotqa-dev.json"
CB  = "/mnt/hdd/jonathan/runs/config_b"
PRICE_SS = 814 + 115                 # Table-5 always-strong policy (no weak draft runs)
PARAMS_RET, PARAMS_QA = 72.0, 70.0   # models.toml params_b: Qwen2.5-72B / Llama-3.1-70B
MODELS = ("retriever=Qwen/Qwen2.5-72B-Instruct-AWQ@698703eae6604af048a3d2f509995dc302088217;"
          "answerer=hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4@2123003760781134cfc31124aa6560a45b491fdf")

dev = json.load(open(DEV))
ids = [x.get("_id") or x.get("id") for x in dev]
gold = {i: (x.get("answer") or x.get("gold")) for i, x in enumerate(dev)}

def md5(sl):
    return hashlib.md5("\n".join(sorted(sl)).encode()).hexdigest()

# split identity: verified against the committed lists, by hash, before anything is written
splits = {}
for name in ("val", "test"):
    committed = set(open(os.path.join(STAIR, "common/stair_common/splits/hotpotqa", name + ".txt")).read().split())
    sl = {"val": ids[:500], "test": ids[500:1000]}[name]
    assert md5(sl) == md5(sorted(committed)), f"{name}: dev slice does not match committed list -- STOP"
    splits[name] = {id_: idx for idx, id_ in enumerate(ids) if id_ in committed}
print("split identity verified by hash: val = dev[0:500], test = dev[500:1000]")

meta_ret = {r["k"]: r for r in map(json.loads, open(f"{CB}/meta_retr_strong.jsonl"))}
meta_qa  = {r["k"]: r for r in map(json.loads, open(f"{CB}/meta_qa_strong.jsonl"))}
answers  = {r["k"]: r["v"] for r in map(json.loads, open(f"{CB}/qa_strong.jsonl"))}

COLS = ["dataset", "split", "id", "track", "policy", "menu", "seed", "answer", "lenient", "em", "f1",
        "units", "units_retrieve", "units_answer", "units_fuse", "units_router", "ms_router", "models", "price"]

for name, idmap in splits.items():
    out = os.path.join(STAIR, "results/common", f"jonathan-ss-hotpotqa-{name}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    n_ok = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for id_, idx in sorted(idmap.items(), key=lambda kv: kv[1]):
            k = str(idx)
            ans = answers[f"{k}|strong"]
            mr, mq = meta_ret[k], meta_qa[f"{k}|strong"]
            u_ret = cost.units(PARAMS_RET, mr["n_in"], mr["n_out"])
            u_qa  = cost.units(PARAMS_QA,  mq["n_in"], mq["n_out"])
            len_  = score.lenient(ans, gold[idx])
            em_   = score.em(ans, gold[idx])
            f1_   = score.f1(ans, gold[idx])
            n_ok += len_
            w.writerow(["hotpotqa", name, id_, "jonathan", "always-strong", "A", 0, ans,
                        len_, em_, f1_, round(u_ret + u_qa, 2), round(u_ret, 2), round(u_qa, 2),
                        0, 0, "", MODELS, PRICE_SS])
    print(f"{name}: n={len(idmap)}  lenient={100.0 * n_ok / len(idmap):.2f}  ->  {out}")
