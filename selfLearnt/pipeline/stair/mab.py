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


def mstore(cfg, key):
    return (os.path.join(wdir(cfg), f"merge_{ep_name(key)}.jsonl"), os.path.join(wdir(cfg), f"graded_merge_{ep_name(key)}.json"))


def _searches(e):
    """The leading searches of an attempt: [(GET turn, server reply)], up to its first non-GET turn."""
    h, out = e["history"], []
    for i in range(1, len(h) - 1):
        if h[i]["role"] == "agent":
            if not h_is_get(h[i]["content"]):
                break
            out.append((h[i], h[i + 1]))
    return out


def merge_starts(cfg, big):
    """Search merge (the MedAgentBench counterpart of merging shortlists on HotpotQA): the task prompt, then the weak
    model's searches and the larger model's own searches (from its fresh attempt) with the server's replies; the larger
    model continues from there and answers. Cost so far = the weak model's searches + the larger model's searches."""
    from stair_common.cost import units
    from .config import variant
    weak = cfg["medagent"]["weak"]
    pw, pb = float(variant(cfg, weak)["params_b"]), float(variant(cfg, big)["params_b"])
    B = {e["tid"]: e for e in read_jsonl(store(cfg, big)[0])}
    out = {}
    for e in read_jsonl(store(cfg, weak)[0]):
        b = B.get(e["tid"])
        if not b:
            continue
        sw, sb = _searches(e), _searches(b)
        seen, turns = set(), []
        for a, r in sw + sb:                      # identical searches once
            k = a["content"].strip()
            if k not in seen:
                seen.add(k)
                turns += [a, r]
        turns = turns[:2 * (MAX_ROUND - 1)]       # leave the answerer at least one round
        cu = sum(units(pw, x, y) for x, y in list(zip(e["n_in"], e["n_out"]))[:len(sw)]) + \
            sum(units(pb, x, y) for x, y in list(zip(b["n_in"], b["n_out"]))[:len(sb)])
        out[e["tid"]] = {"history": [e["history"][0]] + turns, "rounds": len(turns) // 2, "from": f"{weak}+{big}", "units": float(cu)}
    return out


SEARCH_SETS = [["q7:ft"], ["q7:ft", "q32:ft"], ["q7:ft", "q72:ft"], ["q7:ft", "q32:ft", "q72:ft"]]   # W, W+M, W+S, W+M+S


def sets_for(weak):
    """The merged search sets of a ladder that starts with `weak`: W, W+M, W+S, W+M+S."""
    return [[weak], [weak, "q32:ft"], [weak, "q72:ft"], [weak, "q32:ft", "q72:ft"]]


def weaks(cfg):
    return cfg["medagent"].get("weaks", [cfg["medagent"]["weak"]])


def setkey(models):
    return "+".join(m.split(":")[0] for m in models)


def cstore(cfg, answerer, models):
    """One answerer continuing from one merged search set (approaches 1-4, 3 Nov)."""
    b = f"cont_{ep_name(answerer)}_{setkey(models)}"
    return os.path.join(wdir(cfg), f"{b}.jsonl"), os.path.join(wdir(cfg), f"graded_{b}.json")


def search_starts(cfg, models):
    """The task prompt, then every listed model's own leading searches (from its fresh attempt) with the server's
    replies, identical searches once, in ladder order; the answerer continues from there. Cost so far = each listed
    model's search rounds."""
    from stair_common.cost import units
    from .config import variant
    eps = {m: {e["tid"]: e for e in read_jsonl(store(cfg, m)[0])} for m in models}
    out = {}
    for tid, first in eps[models[0]].items():
        seen, turns, cu = set(), [], 0.0
        for m in models:
            e = eps[m].get(tid)
            if not e:
                continue
            sw = _searches(e)
            pb = float(variant(cfg, m)["params_b"])
            cu += sum(units(pb, x, y) for x, y in list(zip(e["n_in"], e["n_out"]))[:len(sw)])
            for a, r in sw:
                k = a["content"].strip()
                if k not in seen:
                    seen.add(k)
                    turns += [a, r]
        turns = turns[:2 * (MAX_ROUND - 1)]            # leave the answerer at least one round
        out[tid] = {"history": [first["history"][0]] + turns, "rounds": len(turns) // 2, "from": setkey(models), "units": float(cu)}
    return out


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


EXPERT_TYPES = ("task5", "task9", "task10")    # the order tasks the teacher never got right (4 Nov, option B)
NOW = "2023-11-13T10:15:00+00:00"                # the benchmark's current time


def _expert_turns(case, base, utils):
    """A correct attempt for one order task, following the task's own instructions: look the value up, place the order
    the instructions call for (if any), report the value. Returns the conversation after the task prompt."""
    from datetime import datetime, timedelta
    ty, mrn = case["id"].split("_")[0], case["eval_MRN"]
    code = {"task5": "MG", "task9": "K", "task10": "A1C"}[ty]
    url = f"{base}Observation?patient={mrn}&code={code}&_count=5000"
    res = utils.send_get_request(url + "&_format=json")
    turns = [{"role": "agent", "content": f"GET {url}"},
             {"role": "user", "content": f"Here is the response from the GET request:\n{res['data']}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"}]
    data = json.loads(res["data"]) if isinstance(res["data"], str) else res["data"]
    now = datetime.fromisoformat(NOW)
    obs = [(datetime.fromisoformat(e["resource"]["effectiveDateTime"]), e["resource"]["effectiveDateTime"], e["resource"]["valueQuantity"]["value"])
           for e in data.get("entry", [])]
    if ty == "task5":                                # magnesium: the last value in the past 24 hours
        obs = [o for o in obs if o[0] >= now - timedelta(hours=24)]
    last = max(obs, key=lambda o: o[0]) if obs else None
    subj = {"reference": f"Patient/{mrn}"}
    posts = []
    if ty == "task5" and last and last[2] <= 1.9:
        dose = 4 if last[2] < 1 else 2 if last[2] < 1.5 else 1
        posts.append(("MedicationRequest", {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Magnesium sulfate in water", "coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "0338-1715-40"}]},
            "authoredOn": NOW, "dosageInstruction": [{"route": "IV", "doseAndRate": [{"doseQuantity": {"value": dose, "unit": "g"},
                                                                                     "rateQuantity": {"value": dose, "unit": "h"}}]}],
            "status": "active", "intent": "order", "subject": subj}))
    if ty == "task9" and last and last[2] < 3.5:
        posts.append(("MedicationRequest", {
            "resourceType": "MedicationRequest",
            "medicationCodeableConcept": {"text": "Potassium chloride", "coding": [{"system": "http://hl7.org/fhir/sid/ndc", "code": "40032-917-01"}]},
            "authoredOn": NOW, "dosageInstruction": [{"route": "oral", "doseAndRate": [{"doseQuantity": {"value": round((3.5 - last[2]) / 0.1 * 10, 1), "unit": "mEq"}}]}],
            "status": "active", "intent": "order", "subject": subj}))
        posts.append(("ServiceRequest", {
            "resourceType": "ServiceRequest", "code": {"coding": [{"system": "http://loinc.org", "code": "2823-3", "display": "Potassium [Moles/volume] in Serum or Plasma"}]},
            "authoredOn": NOW, "status": "active", "intent": "order", "priority": "stat", "subject": subj,
            "occurrenceDateTime": "2023-11-14T08:00:00+00:00"}))
    if ty == "task10" and (not last or last[0] < now - timedelta(days=365)):
        posts.append(("ServiceRequest", {
            "resourceType": "ServiceRequest", "code": {"coding": [{"system": "http://loinc.org", "code": "4548-4", "display": "Hemoglobin A1c/Hemoglobin.total in Blood"}]},
            "authoredOn": NOW, "status": "active", "intent": "order", "priority": "stat", "subject": subj}))
    for kind, payload in posts:
        turns += [{"role": "agent", "content": f"POST {base}{kind}\n{json.dumps(payload)}"},
                  {"role": "user", "content": "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"}]
    ans = [-1] if not last else ([last[2], last[1]] if ty == "task10" else [last[2]])
    turns.append({"role": "agent", "content": f"FINISH({json.dumps(ans)})"})
    return turns, bool(posts)


def experts(cfg, srv, tasks, funcs, utils, ref):
    """Correct attempts for the order tasks (types 5, 9, 10) of the train and val parts, each checked by refsol before it
    is kept -> medagent/experts.jsonl. Test tasks are never touched."""
    from .medagent import MedAgentBench_prompt
    sp = splits()
    keep = set(sp["train"]) | set(sp["val"])
    by_id = {t["id"]: t for t in tasks}
    p = os.path.join(wdir(cfg), "experts.jsonl")
    rows = []
    for tid in sorted(keep):
        if tid.split("_")[0] not in EXPERT_TYPES:
            continue
        case = by_id[tid]
        prompt = MedAgentBench_prompt.format(api_base=srv.url, functions=json.dumps(funcs), context=case["context"], question=case["instruction"])
        turns, ordered = _expert_turns(case, srv.url, utils)
        res = turns[-1]["content"][len("FINISH("):-1]
        rows.append({"tid": tid, "status": "completed", "result": res, "ordered": ordered,
                     "history": [{"role": "user", "content": prompt}] + turns, "rounds": sum(1 for x in turns if x["role"] == "agent"),
                     "n_in": [], "n_out": [], "units": 0.0})
    from .common import write_jsonl
    write_jsonl(p, rows)
    g = grade_file(p, srv.url, ref, tasks, os.path.join(wdir(cfg), "graded_experts.json"))
    bad = [r["tid"] for r in rows if not g.get(r["tid"])]
    log(f"expert attempts: {len(rows)} written, {len(rows) - len(bad)} pass the grader"
        f" ({sum(r['ordered'] for r in rows)} with an order){'; failing: ' + ', '.join(bad) if bad else ''}")


def expert_sets(cfg):
    """The checked expert attempts as training / validation examples; those with an order repeated `order_weight` times."""
    M = cfg["medagent"]
    p, gp = os.path.join(wdir(cfg), "experts.jsonl"), os.path.join(wdir(cfg), "graded_experts.json")
    if not os.path.exists(gp):
        raise SystemExit("no expert attempts yet: run --part expert first")
    g = json.load(open(gp))
    sp = splits()
    tr, va = [], []
    for e in read_jsonl(p):
        if not g.get(e["tid"]):
            continue
        x = {"tid": e["tid"], "messages": [{"role": "assistant" if h["role"] == "agent" else "user", "content": h["content"]} for h in e["history"]]}
        n = int(M.get("order_weight", 3)) if e["ordered"] else 1
        (tr if e["tid"] in sp["train"] else va).extend([x] * (n if e["tid"] in sp["train"] else 1))
    log(f"expert examples: {len(tr)} training (orders x{M.get('order_weight', 3)}), {len(va)} validation")
    return tr, va


def train_model(cfg, key):
    from . import train as T
    tr, va = sft_sets(cfg)
    if cfg["finetune"]["agents"].get(key, {}).get("experts"):     # option B: the teacher's attempts + checked expert ones
        xt, xv = expert_sets(cfg)
        tr, va = tr + xt, va + xv
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
        elif part == "expert":
            experts(cfg, srv, tasks, funcs, utils, ref)
        elif part == "cont":                     # this answerer from every merged search set
            from .common import write_jsonl
            for models in [m for w in weaks(cfg) for m in sets_for(w)]:
                if models == [model] or (model in weaks(cfg) and model != models[0]):
                    continue                          # its own attempt, or a weak model answering another weak model's sets
                ep, gr = cstore(cfg, model, models)
                if os.path.exists(gr):
                    continue                          # done in an earlier job
                starts = search_starts(cfg, models)
                run_episodes(cfg, model, allt, srv.url, utils, tasks, funcs, out=ep, starts=starts)
                rows = read_jsonl(ep)
                for r in rows:
                    r["search_units"] = starts.get(r["tid"], {}).get("units", 0.0)
                write_jsonl(ep, rows)
                grade_file(ep, srv.url, ref, tasks, gr)
        elif part == "merge":
            ep, gr = mstore(cfg, model)
            starts = merge_starts(cfg, model)
            run_episodes(cfg, model, allt, srv.url, utils, tasks, funcs, out=ep, starts=starts)
            rows = read_jsonl(ep)                 # record the merged searches' cost with each continuation
            for r in rows:
                r["merge_units"] = starts.get(r["tid"], {}).get("units", 0.0)
            from .common import write_jsonl
            write_jsonl(ep, rows)
            grade_file(ep, srv.url, ref, tasks, gr)
        else:
            raise SystemExit(f"unknown part {part}")
