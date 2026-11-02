"""MedAgentBench switching rulebooks (AL-16 .. AL-19), learnt and tested on the stored, graded episodes (stair/mab.py).
CPU only, a minute.  python run.py --config configs/medagent.yaml --stage mab_rules

Every attempt is weak-first and every call that ran is paid for (Jonathan's convention). Cost is reported as a share of
always-strong's cost (the untrained Qwen 72B on the same tasks) and in AgentTTS units.

A rulebook (restart designs AL-16 .. AL-18): for each rung below the top and each task type
  early   stop the attempt after round 2 if it is still working, and restart on the next rung (its 2 rounds are paid)
  end     what to do with a finished attempt: accept | up (next rung) | top (largest rung) | person (high-risk types only)
and at the top rung, per task type: accept | person. One confidence rule per rung: escalate as `up` when the attempt's
last reply has a mean token log-probability below a threshold (or no threshold). Locked rule: an attempt that did not
finish (invalid action, no answer, out of rounds or context) escalates (at the top rung: a person if allowed, else it
fails). AL-19 (hand-off): the weak model always runs; per task type, accept its answer, hand its searches to the
32B or 72B answerer (which continues the conversation), or a person.

Learning: start from "accept everything" (the locked rule only); try every single change, keep the best one that
raises the objective on the 60 val tasks without lowering it on either half (30 / 30, every task type split evenly);
repeat until no change qualifies. Objective per task = correct - cost_weight x cost / always-strong cost (Jonathan's R').
A person counts as correct and costs human_cost_factor x always-strong's cost on that task."""
import json
import os

import numpy as np

from .common import log, read_jsonl, write_json
from .mab import hstore, splits, store
from .medagent import wdir

TYPES = [f"task{i}" for i in range(1, 11)]


def _ty(tid):
    return tid.split("_")[0]


class Data:
    def __init__(self, cfg):
        M = cfg["medagent"]
        self.M, self.cfg = M, cfg
        self.sp = splits()
        self.E, self.G = {}, {}
        keys = {M["teacher"], "q7"} | {k for d in M["ladders"].values() for k in d["ladder"]}
        for k in keys:
            ep, gr = store(cfg, k)
            if os.path.exists(gr):
                self.E[k] = {e["tid"]: e for e in read_jsonl(ep)}
                self.G[k] = json.load(open(gr))
            else:
                log(f"{k}: no graded episodes yet")
        self.H, self.HG = {}, {}
        for k in ("q32:ft", "q72:ft"):
            ep, gr = hstore(cfg, k)
            if os.path.exists(gr):
                self.H[k] = {e["tid"]: e for e in read_jsonl(ep)}
                self.HG[k] = json.load(open(gr))
        t = M["teacher"]
        self.ss = {tid: e["units"] for tid, e in self.E[t].items()}      # always-strong's cost per task
        self.hr = set(M["high_risk_types"])

    def two_rounds(self, k, tid):
        from stair_common.cost import units
        from .config import variant
        e = self.E[k][tid]
        pb = float(variant(self.cfg, k)["params_b"])
        return float(sum(units(pb, a, b) for a, b in list(zip(e["n_in"], e["n_out"]))[:2]))

    def lp(self, k, tid):
        v = [x for x in self.E[k][tid].get("lp", []) if x is not None]
        return v[-1] if v else 0.0


def seed_book(D, design):
    L = D.M["ladders"][design]
    n = len(L["ladder"])
    if L["mode"] == "handoff":
        return {"mode": "handoff", "act": {t: "accept" for t in TYPES}, "theta": None}
    return {"mode": "restart", "end": [{t: "accept" for t in TYPES} for _ in range(n)],
            "early": [{t: False for t in TYPES} for _ in range(n)], "theta": [None] * n}


def options(D, design, book):
    """Every single change of a rulebook: (description, new book)."""
    L = D.M["ladders"][design]
    lad, human = L["ladder"], L["human"]
    n = len(lad)
    out = []
    if book["mode"] == "handoff":
        acts = ["accept"] + [f"hand:{k}" for k in ("q32:ft", "q72:ft") if k in D.H]
        for t in TYPES:
            for a in acts + (["person"] if human and t in D.hr else []):
                if a != book["act"][t]:
                    b = json.loads(json.dumps(book))
                    b["act"][t] = a
                    out.append((f"{t}: {book['act'][t]} -> {a}", b))
        for th in [None] + [float(np.quantile([D.lp(lad[0], t) for t in D.sp["val"]], q)) for q in (0.1, 0.25, 0.5)]:
            if th != book["theta"]:
                b = json.loads(json.dumps(book))
                b["theta"] = th
                out.append((f"hand to the 72B when confidence < {th}", b))
        return out
    for i in range(n):
        top = i == n - 1
        for t in TYPES:
            acts = ["accept"] + ([] if top else ["up"] + (["top"] if i < n - 2 else [])) + (["person"] if human and t in D.hr else [])
            for a in acts:
                if a != book["end"][i][t]:
                    b = json.loads(json.dumps(book))
                    b["end"][i][t] = a
                    out.append((f"rung {i + 1} ({lad[i]}), {t}: end {book['end'][i][t]} -> {a}", b))
            if not top:
                b = json.loads(json.dumps(book))
                b["early"][i][t] = not book["early"][i][t]
                out.append((f"rung {i + 1} ({lad[i]}), {t}: stop at round 2 if still working = {b['early'][i][t]}", b))
        if not top:
            for th in [None] + [float(np.quantile([D.lp(lad[i], t) for t in D.sp["val"]], q)) for q in (0.1, 0.25, 0.5)]:
                if th != book["theta"][i]:
                    b = json.loads(json.dumps(book))
                    b["theta"][i] = th
                    out.append((f"rung {i + 1} ({lad[i]}): escalate when confidence < {th}", b))
    return out


def play(D, design, book, tid, k_human=None):
    """One task under a rulebook -> (correct, cost in units, path, answer, person?)."""
    L = D.M["ladders"][design]
    lad, human = L["ladder"], L["human"]
    t = _ty(tid)
    kh = D.M["human_cost_factor"] if k_human is None else k_human
    person_ok = human and t in D.hr
    if book["mode"] == "handoff":
        w = lad[0]
        e = D.E[w][tid]
        cost = e["units"]
        a = book["act"][t]
        if e["status"] != "completed":                     # locked: no answer -> the largest answerer (or a person)
            a = "person" if person_ok and a == "person" else "hand:q72:ft"
        elif book["theta"] is not None and D.lp(w, tid) < book["theta"] and a == "accept":
            a = "hand:q72:ft"
        if a == "accept":
            return D.G[w].get(tid, 0), cost, w, e.get("result"), False
        if a == "person":
            return 1, cost + kh * D.ss[tid], f"{w}->person", None, True
        k = a.split(":", 1)[1]
        h = D.H[k][tid]
        return D.HG[k].get(tid, 0), cost + h["units"], f"{w}->{k}", h.get("result"), False
    i, cost, path = 0, 0.0, []
    n = len(lad)
    while True:
        k = lad[i]
        e = D.E[k][tid]
        path.append(k)
        top = i == n - 1
        if not top and book["early"][i][t] and (e["rounds"] > 2 or e["status"] != "completed"):
            cost += D.two_rounds(k, tid)
            i += 1
            continue
        cost += e["units"]
        a = book["end"][i][t]
        if e["status"] != "completed":                     # locked rule
            a = "up" if not top else ("person" if person_ok else "accept")
        elif not top and book["theta"][i] is not None and D.lp(k, tid) < book["theta"][i] and a == "accept":
            a = "up"
        if a == "accept":
            return D.G[k].get(tid, 0), cost, "->".join(path), e.get("result"), False
        if a == "person" and person_ok:
            return 1, cost + kh * D.ss[tid], "->".join(path) + "->person", None, True
        i = n - 1 if a == "top" else i + 1
        if i >= n:
            return D.G[k].get(tid, 0), cost, "->".join(path), e.get("result"), False


def objective(D, design, book, tids):
    w = D.M["cost_weight"]
    return float(np.mean([c - w * u / D.ss[t] for t in tids for c, u, *_ in [play(D, design, book, t)]]))


def halves(tids):
    a, b = [], []
    for t in TYPES:
        x = sorted((q for q in tids if _ty(q) == t), key=lambda q: int(q.split("_")[1]))
        a += x[0::2]
        b += x[1::2]
    return a, b


def learn(D, design):
    book = seed_book(D, design)
    V = D.sp["val"]
    A, B = halves(V)
    hist = []
    for rnd in range(50):
        base = (objective(D, design, book, V), objective(D, design, book, A), objective(D, design, book, B))
        best = None
        for desc, b in options(D, design, book):
            o = (objective(D, design, b, V), objective(D, design, b, A), objective(D, design, b, B))
            if o[0] > base[0] + 1e-9 and o[1] >= base[1] - 1e-9 and o[2] >= base[2] - 1e-9 and (best is None or o[0] > best[0][0]):
                best = (o, desc, b)
        if best is None:
            break
        hist.append({"round": rnd + 1, "change": best[1], "val_objective": best[0][0], "before": base[0]})
        book = best[2]
    return book, hist


def summary(D, design, book, tids, k_human=None):
    rows = [play(D, design, book, t, k_human) for t in tids]
    acc = np.mean([r[0] for r in rows])
    cost = sum(r[1] for r in rows) / sum(D.ss[t] for t in tids)
    hr = [(t, r) for t, r in zip(tids, rows) if _ty(t) in D.hr]
    return {"acc": float(acc), "cost_share": float(cost), "units": float(np.mean([r[1] for r in rows])),
            "to_person": sum(r[4] for r in rows), "silent_failures": sum(1 for _, r in hr if not r[4] and not r[0]), "n": len(tids)}


def fixed(D, k, tids):
    return {"acc": float(np.mean([D.G[k].get(t, 0) for t in tids])),
            "cost_share": sum(D.E[k][t]["units"] for t in tids) / sum(D.ss[t] for t in tids),
            "units": float(np.mean([D.E[k][t]["units"] for t in tids])), "to_person": 0,
            "silent_failures": sum(1 for t in tids if _ty(t) in D.hr and not D.G[k].get(t, 0)), "n": len(tids)}


JONATHAN = [("Jonathan: always-weak (untrained 7B)", 12.00, 18.7), ("Jonathan: always-strong (untrained 72B)", 54.00, 100.0),
            ("Jonathan: oracle", 63.33, None), ("Jonathan: Training 1 router", 61.33, 54.1),
            ("Jonathan: Training 3 router (round 2)", 63.33, 81.5), ("Jonathan: Training 5b (person on high risk)", 76.00, 94.5)]


def export(D, design, book, seed_bk, T):
    import pandas as pd
    from stair_common.schema import validate
    rows = []

    def add(pol, tid, c, u, ans):
        rows.append({"dataset": "medagentbench", "split": "test", "id": tid, "track": "selfLearnt", "policy": pol, "menu": "B", "seed": 0,
                     "answer": "" if ans is None else str(ans)[:500], "lenient": int(c), "em": float(c), "f1": float(c), "units": float(u),
                     "units_retrieve": 0.0, "units_answer": float(u), "units_fuse": 0.0, "units_router": 0.0, "ms_router": np.nan,
                     "models": "", "price": float(u)})
    for pol, bk in (("rulebook-learnt", book), ("rulebook-seed", seed_bk)):
        for t in T:
            c, u, _, ans, _ = play(D, design, bk, t)
            add(pol, t, c, u, ans)
    for k in sorted(set(D.M["ladders"][design]["ladder"]) | {"q7", D.M["teacher"]}):
        if k in D.G:
            name = f"fixed:{k}" if k.endswith(":ft") else f"untrained:{k}"
            for t in T:
                add(name, t, D.G[k].get(t, 0), D.E[k][t]["units"], D.E[k][t].get("result"))
    df = pd.DataFrame(rows)
    sd = os.path.join(wdir(D.cfg), "splits")                 # Jonathan's split (our common still has the reverted one)
    os.makedirs(os.path.join(sd, "medagentbench"), exist_ok=True)
    open(os.path.join(sd, "medagentbench", "test.txt"), "w").write("\n".join(D.sp["test"]) + "\n")
    validate(df, splits_dir=sd)
    p = os.path.join(D.cfg["paths"]["results_dir"], f"{design}-medagentbench-test-seed0.csv")
    df.to_csv(p, index=False)
    log(f"{design}: {len(df)} rows -> {p}")


def run(cfg, part="all"):
    D = Data(cfg)
    T, V = D.sp["test"], D.sp["val"]
    out = {}
    lines = ["# MedAgentBench rulebooks (test 150, read once; learnt on val 60)", ""]
    lines.append("| Policy | Accuracy % | Cost (% of always-strong) | To a person | High-risk failures no one sees |")
    lines.append("|---|---|---|---|---|")

    def line(name, s):
        lines.append(f"| {name} | {100 * s['acc']:.2f} | {100 * s['cost_share']:.1f} | {s['to_person']} | {s['silent_failures']} |")
    for k, name in (("q7", "untrained Qwen 7B (always-weak)"), (D.M["teacher"], "untrained Qwen 72B (always-strong)"),
                    ("q7:ft", "fine-tuned Qwen 7B alone"), ("q32:ft", "fine-tuned Qwen 32B alone"), ("q72:ft", "fine-tuned Qwen 72B alone")):
        if k in D.G:
            out[name] = fixed(D, k, T)
            line(name, out[name])
    for design, L in D.M["ladders"].items():
        if any(k not in D.G for k in L["ladder"]) or (L["mode"] == "handoff" and len(D.H) < 2):
            log(f"{design}: not all episodes yet - skipped")
            continue
        book, hist = learn(D, design)
        sb = seed_book(D, design)
        s = summary(D, design, book, T)
        s1 = summary(D, design, book, T, k_human=1.0)
        out[design] = {"test": s, "test_person_at_1x": s1, "val": summary(D, design, book, V), "seed_test": summary(D, design, sb, T),
                       "rulebook": book, "changes": hist}
        write_json(os.path.join(wdir(cfg), f"rulebook_{design}.json"), out[design])
        line(f"{design} {L['name']}: learnt rules ({len(hist)} changes)", s)
        if L["human"]:
            lines.append(f"| ... same, a person priced at 1 x always-strong | {100 * s1['acc']:.2f} | {100 * s1['cost_share']:.1f} | "
                         f"{s1['to_person']} | {s1['silent_failures']} |")
        try:
            export(D, design, book, sb, T)
        except Exception as e:  # noqa: BLE001
            log(f"{design}: results table not written ({e})")
    for name, a, c in JONATHAN:
        lines.append(f"| {name} | {a:.2f} | {'-' if c is None else f'{c:.1f}'} | | |")
    txt = "\n".join(lines)
    open(os.path.join(wdir(cfg), "rules_summary.md"), "w").write(txt + "\n")
    print(txt)
    write_json(os.path.join(wdir(cfg), "rules_results.json"), out)
