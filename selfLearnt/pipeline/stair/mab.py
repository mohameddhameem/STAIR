"""MedAgentBench with our method (configs/medagent.yaml; experiment sheets AL-16 .. AL-19).

Parts (run.py --stage mab_run --part <part> [--model <key>]):
  teach      the untrained teacher (Qwen 72B) on all 300 tasks greedily (= always-strong, also its cost per task) and
             `teach_samples` sampled attempts on the 150 calib tasks; every attempt graded by refsol
  train      fine-tune one agent model (--model q7 / q32 / q72) on the teacher's passing attempts: train = calib minus
             calib_sweep (90 tasks), validation loss on calib_sweep (60); checkpoints as on HotpotQA
  episodes   one model (--model q7:ft / q32:ft / q72:ft / q7) on all 300 tasks greedily, graded
  handoff    AL-19: a larger fine-tuned model (--model q32:ft / q72:ft) continues the weak model's attempt from the
             point where it stopped searching (its first non-GET turn), graded
The rulebooks are learnt and tested on the stored, graded episodes by stair/mab_rules.py (CPU, stage mab_rules)."""
import json
import os
import subprocess

from .common import log, read_jsonl, write_json
from .medagent import FhirServer, MAX_ROUND, ROOT, ep_name, grade_file, harness, run_episodes, wdir

SPLIT_FILES = {"calib": "calib.txt", "val": "calib_sweep.txt", "test": "test.txt"}


def splits():
    """Jonathan's patient-disjoint split (branch jonathan, common/stair_common/splits/medagentbench): train 90, val 60, test 150."""
    out = {}
    for k, f in SPLIT_FILES.items():
        r = subprocess.run(["git", "-C", ROOT, "show", f"origin/jonathan:common/stair_common/splits/medagentbench/{f}"],
                           capture_output=True, text=True)
        if r.returncode:
            raise SystemExit("run `git fetch origin jonathan` first (the MedAgentBench split is on that branch)")
        out[k] = [x.strip() for x in r.stdout.splitlines() if x.strip()]
    val = set(out["val"])
    out["train"] = [t for t in out["calib"] if t not in val]
    assert (len(out["train"]), len(out["val"]), len(out["test"])) == (90, 60, 150), {k: len(v) for k, v in out.items()}
    assert not (set(out["calib"]) & set(out["test"]))
    return out


def tdir(cfg):
    d = os.path.join(wdir(cfg), "teach")
    os.makedirs(d, exist_ok=True)
    return d


def store(cfg, key):
    """Episode store and grades of one model (greedy, all 300 tasks)."""
    return (os.path.join(wdir(cfg), f"episodes_{ep_name(key)}.jsonl"), os.path.join(wdir(cfg), f"graded_{ep_name(key)}.json"))


def hstore(cfg, key):
    return (os.path.join(wdir(cfg), f"handoff_{ep_name(key)}.jsonl"), os.path.join(wdir(cfg), f"graded_handoff_{ep_name(key)}.json"))


def teach(cfg, srv, tasks, funcs, utils, ref):
    M = cfg["medagent"]
    t = M["teacher"]
    sp = splits()
    allt = [x["id"] for x in tasks]
    ep, gr = store(cfg, t)
    run_episodes(cfg, t, allt, srv.url, utils, tasks, funcs, out=ep)
    grade_file(ep, srv.url, ref, tasks, gr)
    for s in range(1, int(M["teach_samples"]) + 1):
        p = os.path.join(tdir(cfg), f"sample{s}.jsonl")
        run_episodes(cfg, t, sp["calib"], srv.url, utils, tasks, funcs, out=p, temperature=M["teach_temperature"], seed=s)
        grade_file(p, srv.url, ref, tasks, os.path.join(tdir(cfg), f"graded_sample{s}.json"))


def sft_sets(cfg):
    """(train, val) lists of {"messages": [...]}: the teacher's passing attempts (greedy + samples), at most keep_per_task
    distinct ones per task, as user / assistant messages."""
    M = cfg["medagent"]
    sp = splits()
    files = [store(cfg, M["teacher"])] + [(os.path.join(tdir(cfg), f"sample{s}.jsonl"), os.path.join(tdir(cfg), f"graded_sample{s}.json"))
                                         for s in range(1, int(M["teach_samples"]) + 1)]
    per = {}
    for ep, gr in files:
        if not os.path.exists(gr):
            continue
        g = json.load(open(gr))
        for e in read_jsonl(ep):
            if g.get(e["tid"]) == 1:
                msgs = [{"role": "assistant" if h["role"] == "agent" else "user", "content": h["content"]} for h in e["history"]]
                sig = json.dumps([m["content"] for m in msgs if m["role"] == "assistant"])
                d = per.setdefault(e["tid"], {})
                if sig not in d and len(d) < int(M["keep_per_task"]):
                    d[sig] = {"tid": e["tid"], "messages": msgs}
    tr = [x for t in sp["train"] for x in per.get(t, {}).values()]
    va = [x for t in sp["val"] for x in per.get(t, {}).values()]
    by_type = {}
    for x in tr:
        by_type.setdefault(x["tid"].split("_")[0], set()).add(x["tid"])
    log(f"training attempts: {len(tr)} from {len({x['tid'] for x in tr})} of 90 train tasks; validation attempts: {len(va)} from "
        f"{len({x['tid'] for x in va})} of 60 val tasks; train tasks with a passing attempt per type: "
        + ", ".join(f"{k[4:]}: {len(v)}" for k, v in sorted(by_type.items(), key=lambda kv: int(kv[0][4:]))))
    return tr, va


def train_model(cfg, key):
    from . import train as T
    tr, va = sft_sets(cfg)
    if not tr:
        raise SystemExit("no passing teacher attempts yet: run --part teach first")
    if not T.trained(cfg, key):
        if not T.train(cfg, key, tr, va, time_limit_min=float(os.environ.get("TIME_LIMIT_MIN", 0))):
            raise SystemExit(3)                    # not finished: the job is requeued and resumes
    T.merge(cfg, key)


def handoff_starts(cfg):
    """The weak model's attempts cut where it stopped searching: everything before its first non-GET turn (or before its
    last turn if it only searched), with that part's rounds and cost."""
    from stair_common.cost import units
    from .config import variant
    weak = cfg["medagent"]["weak"]
    pb = float(variant(cfg, weak)["params_b"])
    out = {}
    for e in read_jsonl(store(cfg, weak)[0]):
        ag = [i for i, h in enumerate(e["history"]) if h["role"] == "agent"]
        if not ag:
            continue
        cut = next((i for i in ag if not h_is_get(e["history"][i]["content"])), ag[-1])
        r = sum(1 for i in ag if i < cut)
        if r >= MAX_ROUND:
            r, cut = MAX_ROUND - 1, ag[MAX_ROUND - 1]
        out[e["tid"]] = {"history": e["history"][:cut], "rounds": r, "from": weak,
                         "units": float(sum(units(pb, a, b) for a, b in list(zip(e["n_in"], e["n_out"]))[:r]))}
    return out


def h_is_get(text):
    return text.strip().replace('```tool_code', '').replace('```', '').strip().startswith("GET")


def run(cfg, part, model=""):
    if part == "train":
        return train_model(cfg, model)
    tasks, funcs, utils, ref = harness(cfg)
    allt = [x["id"] for x in tasks]
    with FhirServer(wdir(cfg)) as srv:
        if part == "teach":
            teach(cfg, srv, tasks, funcs, utils, ref)
        elif part == "episodes":
            ep, gr = store(cfg, model)
            run_episodes(cfg, model, allt, srv.url, utils, tasks, funcs, out=ep)
            grade_file(ep, srv.url, ref, tasks, gr)
        elif part == "handoff":
            ep, gr = hstore(cfg, model)
            run_episodes(cfg, model, allt, srv.url, utils, tasks, funcs, out=ep, starts=handoff_starts(cfg))
            grade_file(ep, srv.url, ref, tasks, gr)
        else:
            raise SystemExit(f"unknown part {part}")
