"""The rulebook: soft rules learnt by micro-experiments on the dev questions, locked rules and guardrails no experiment may change.

Pipeline per question:
  cheapest retriever shortlists -> 'retrieval' rules may escalate to a bigger retriever (shortlists ensembled)
  -> 'retrieval2' rules may escalate once more -> small answerer answers -> 'answer' rules may escalate: the big answerer
  answers; agree -> accept, else sampled big-answerer answers join a majority vote. Every answer gets a calibrated
  Sure / Unsure flag.
Rules are checked in order (locked first); the first rule whose conditions all hold decides; default = keep.

learn: rounds of micro-experiments on the dev questions (halves A and B):
  code tries fine threshold steps for every soft rule; a planner LLM proposes up to 3 changes from evidence tables;
  a change is kept only if, on BOTH halves, it is cheaper at no accuracy loss or more accurate at <= H8 compute per point,
  it beats a random-escalation control of the same size, and it breaks no hard guardrail.
test: the frozen rulebook once on the test questions, against fixed pipelines, random escalation and an oracle."""
import copy
import json
import os
import re
import zlib

import numpy as np

from . import data as D
from .common import agree, ensemble, key, log, lp_, make_engine, normalize_answer, read_jsonl, vote, write_json, write_jsonl
from .generate import ret_file, step_dir

_CANNOT = re.compile(r"(unknown|cannot|can't|not (provided|mentioned|specified|available|enough|stated|given)|no (information|answer|mention)|insufficient|unclear)")
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal}
SIGNALS = {
    "base_conf_pct": ("retrieval", "cheapest retriever's confidence in its own shortlist (token probability), percentile among dev questions"),
    "ret_agree": ("retrieval2", "after escalating: 1 if the bigger retriever picked the same passages as the cheapest, else 0"),
    "target_conf_pct": ("retrieval2", "after escalating: the bigger retriever's confidence, percentile among dev questions"),
    "answer_conf_pct": ("answer", "small answerer's token probability for its answer, percentile among dev questions"),
    "answer_stated": ("answer", "small answerer's stated confidence (0-1)"),
    "words_in_passages": ("answer", "share of the answer's words found in the shortlisted passages (0-1)"),
    "is_yes_no": ("answer", "1 if the answer is yes/no"),
    "is_cannot": ("answer", "1 if the answer is empty or says it cannot be determined"),
    "answer_words": ("answer", "number of words in the answer"),
    "is_comparison": ("answer", "1 if HotpotQA marks the question as a comparison question"),
    "ret_escalated": ("answer", "1 if retrieval was escalated for this question"),
}


def rule_text(r, names=None):
    names = names or {}
    c = " AND ".join(f"{s} {o} {v:g}" for s, o, v in r["when"])
    to = f" (to {names.get(r['to'], r['to'])})" if r.get("to") else ""
    return f"[{r['id']}{', LOCKED' if r.get('hard') else ''}] {r['stage']}: IF {c} THEN {r['then']}{to}"


# ==================================================================================
class World:
    """Saved outputs for one question set."""

    def __init__(self, cfg, split, A, SM):
        self.Q = D.load(cfg, split)
        self.qids = [q["qid"] for q in self.Q]
        self.qd = {q["qid"]: q for q in self.Q}
        self.ladder = [r["key"] for r in cfg["retrievers"]]
        self.R = {r: {x["qid"]: x for x in read_jsonl(ret_file(cfg, split, r))} for r in self.ladder}
        self.A, self.SM = A, SM
        self._st = {}

    def member(self, q, r):
        x = self.R[r][q]
        return key(x["sel"]), x.get("mean_logprob"), x["flops"]

    def state(self, q, members):
        k = (q, tuple(members))
        if k not in self._st:
            ms = [self.member(q, m) for m in members]
            self._st[k] = (ensemble([m[0] for m in ms]), sum(m[2] for m in ms))
        return self._st[k]

    def gold(self, q):
        return {i + 1 for i in self.qd[q]["ret_answers"]}


class Env:
    def __init__(self, cfg):
        self.cfg = cfg
        self.small, self.big = cfg["answerers"]["small"]["key"], cfg["answerers"]["big"]["key"]
        sd = step_dir(cfg)
        self.A = {}
        for m in (self.small, self.big):
            self.A[m] = {(r["qid"], key(r["sel"])): r for r in read_jsonl(os.path.join(sd, f"answers_{m}.jsonl"))}
        self.SM = {(r["qid"], key(r["sel"])): r for r in read_jsonl(os.path.join(sd, f"samples_{self.big}.jsonl"))}
        self.dev = World(cfg, "dev", self.A, self.SM)
        self.ladder = self.dev.ladder
        self.base = self.ladder[0]
        self.next = {r: (self.ladder[i + 1] if i + 1 < len(self.ladder) else None) for i, r in enumerate(self.ladder)}
        self.names = {r["key"]: r["label"] for r in cfg["retrievers"]}
        self.names.update({self.small: cfg["answerers"]["small"]["label"], self.big: cfg["answerers"]["big"]["label"]})
        d = self.dev
        self.dist = {"base_conf": np.sort([np.exp(lp_(d.member(q, self.base)[1])) for q in d.qids]),
                     "target_conf": np.sort([np.exp(lp_(d.member(q, r)[1])) for r in self.ladder[1:] for q in d.qids]),
                     "answer_conf": np.sort([np.exp(lp_(self.A[self.small][(q, d.member(q, self.base)[0])].get("mean_logprob"))) for q in d.qids])}
        rng = np.random.default_rng(cfg["rules"].get("dev_split_seed", 11))
        perm = rng.permutation(len(d.qids))
        self.halfA = sorted(d.qids[i] for i in perm[: len(perm) // 2])
        self.halfB = sorted(d.qids[i] for i in perm[len(perm) // 2:])
        self._sig = {}
        self.H = cfg["rules"]["hard"]

    def pct(self, kind, raw):
        a = self.dist[kind]
        return float(np.searchsorted(a, raw, side="right") / len(a))

    def answer_signals(self, W, q, sel):
        k = (id(W), q, sel)
        if k in self._sig:
            return self._sig[k]
        a = self.A[self.small][(q, sel)]
        qq = W.qd[q]
        pred = a.get("prediction") or ""
        p = normalize_answer(pred)
        ps = set(normalize_answer(" ".join(qq["ctxs_candidate"][i - 1] for i in sel if 1 <= i <= len(qq["ctxs_candidate"]))).split())
        toks = p.split()
        s = {"answer_conf_pct": self.pct("answer_conf", np.exp(lp_(a.get("mean_logprob")))),
             "answer_stated": float(a["stated_conf"]) if a.get("stated_conf") is not None else 0.5,
             "words_in_passages": float(np.mean([t in ps for t in toks])) if toks else 0.0,
             "is_yes_no": float(p in ("yes", "no")), "is_cannot": float((not p) or bool(_CANNOT.search(pred.lower()))),
             "answer_words": float(len(toks)), "is_comparison": float(qq.get("type") == "comparison")}
        self._sig[k] = s
        return s


def match(rule, sig):
    return all(s in sig and OPS[o](sig[s], v) for s, o, v in rule["when"])


def decide(rules, stage, sig):
    rs = [r for r in rules if r["stage"] == stage]
    for r in [r for r in rs if r.get("hard")] + [r for r in rs if not r.get("hard")]:
        if match(r, sig):
            return r
    return None


def run_q(E, W, q, rb, force=None):
    P, rules = rb["params"], rb["rules"]
    base = E.base
    s0, lp0, f0 = W.member(q, base)
    flops = f0
    rsig = {"base_conf_pct": E.pct("base_conf", np.exp(lp_(lp0)))}
    r1 = decide(rules, "retrieval", rsig)
    esc_r = (r1 is not None and r1["then"] == "escalate") if not force else force["ret"]
    fired = [r1["id"] if r1 else "-"]
    sel, members = s0, [base]
    if esc_r:
        tgt = (r1.get("to") if (r1 and not force) else None) or P["ret_target"]
        members = [base, tgt]
        if P["ret_merge"] == "target":
            sel, _, f = W.member(q, tgt)
            flops += f
        else:
            sel, f = W.state(q, members)
            flops = f
        st, lpt, _ = W.member(q, tgt)
        sig2 = dict(rsig, ret_agree=float(st == s0), target_conf_pct=E.pct("target_conf", np.exp(lp_(lpt))))
        r2 = decide(rules, "retrieval2", sig2)
        fired.append(r2["id"] if r2 else "-")
        if r2 is not None and r2["then"] == "escalate" and E.next.get(tgt):
            members = [base, tgt, E.next[tgt]]
            sel, flops = W.state(q, members)
    else:
        fired.append("-")
    aS = dict(E.A[E.small][(q, sel)], model=E.small)
    asig = dict(E.answer_signals(W, q, sel), ret_escalated=float(esc_r), **rsig)
    r3 = decide(rules, "answer", asig)
    esc_a = (r3 is not None and r3["then"] == "escalate") if not force else force["ans"]
    fired.append(r3["id"] if r3 else "-")
    flops += aS["flops"]
    final, votes, agreeSB, support = aS, [E.small], -1.0, 1.0
    if esc_a:
        aB = dict(E.A[E.big][(q, sel)], model=E.big)
        flops += aB["flops"]
        agreeSB = float(agree(aS["prediction"], aB["prediction"]))
        if agreeSB:
            final, votes = aB, [E.small, E.big]
        else:
            sm = E.SM[(q, sel)]
            ss = [dict(s, model=E.big + "_sample") for s in sm["samples"][:int(P["votes"])]]
            flops += sm["flops"] / max(1, len(sm["samples"])) * len(ss)
            vv = [aB, aS] + ss
            final, sup = vote(vv)
            votes, support = [v["model"] for v in vv], sup / len(vv)
    return {"qid": q, "lenient": float(final["lenient"]), "EM": float(final["EM"]), "flops": float(flops), "ret_esc": float(esc_r), "ans_esc": float(esc_a),
            "members": "+".join(E.names.get(m, m) for m in members), "fired": fired, "answer": final.get("prediction"), "votes": "+".join(votes),
            "sure_feats": [asig["answer_conf_pct"], asig["answer_stated"], asig["words_in_passages"], asig["is_yes_no"], asig["is_cannot"], float(esc_a),
                           agreeSB, support, rsig["base_conf_pct"], float(esc_r), float(sel == s0)]}


def evaluate(E, W, qids, rb, forces=None):
    return [run_q(E, W, q, rb, (forces or {}).get(q)) for q in qids]


def summ(rows):
    return {"acc": float(np.mean([r["lenient"] for r in rows])), "em": float(np.mean([r["EM"] for r in rows])), "flops": float(np.sum([r["flops"] for r in rows])),
            "ret_esc": float(np.mean([r["ret_esc"] for r in rows])), "big": float(np.mean([r["ans_esc"] for r in rows]))}


# ==================================================================================
# acceptance
# ==================================================================================
def ok_half(E, c, b, repair=False):
    dA, dC = c["acc"] - b["acc"], c["flops"] / b["flops"] - 1
    if repair:
        return dA > 0, dA, dC
    cheaper = dA >= 0 and dC <= -0.005
    better = dA > 0 and dC <= E.H["H8_max_compute_per_point"] * (100 * dA)
    return cheaper or better, dA, dC


def guardrails(E, rb, rows, best_fixed, cur_acc):
    s = summ(rows)
    soft = [r for r in rb["rules"] if not r.get("hard")]
    why = []
    if s["big"] > E.H["H3_max_share_big"] + 1e-9:
        why.append(f"H3: {100*s['big']:.0f}% reach the big answerer (max {100*E.H['H3_max_share_big']:.0f}%)")
    if s["acc"] < best_fixed - E.H["H4_max_acc_drop_vs_best_fixed"] and s["acc"] < cur_acc:
        why.append(f"H4: dev accuracy {100*s['acc']:.1f}% below the best fixed pipeline ({100*best_fixed:.1f}%) minus {100*E.H['H4_max_acc_drop_vs_best_fixed']:.0f} point")
    if len(soft) > E.H["H6_max_soft_rules"]:
        why.append(f"H6: {len(soft)} soft rules (max {E.H['H6_max_soft_rules']})")
    cnt = {}
    for r in rows:
        for f in r["fired"]:
            cnt[f] = cnt.get(f, 0) + 1
    need = int(np.ceil(E.H["H6_min_support_share"] * len(rows)))
    for r in soft:
        if r["then"] == "escalate" and cnt.get(r["id"], 0) < need:
            why.append(f"H6: rule {r['id']} decides on {cnt.get(r['id'], 0)} dev questions (min {need})")
    return why


def random_control(E, W, qids, cur_rows, cand_rows, cur_rb, draws, seed):
    rng = np.random.default_rng(seed)
    by, cand = {r["qid"]: r for r in cur_rows}, {r["qid"]: r for r in cand_rows}
    out = []
    for _ in range(draws):
        forces = {q: {"ret": bool(by[q]["ret_esc"]), "ans": bool(by[q]["ans_esc"])} for q in qids}
        for st in ("ret", "ans"):
            k = st + "_esc"
            plus = sum(1 for q in qids if cand[q][k] > by[q][k])
            minus = sum(1 for q in qids if cand[q][k] < by[q][k])
            pool_p, pool_m = [q for q in qids if not by[q][k]], [q for q in qids if by[q][k]]
            for q in (rng.choice(pool_p, min(plus, len(pool_p)), replace=False) if plus else []):
                forces[q][st] = True
            for q in (rng.choice(pool_m, min(minus, len(pool_m)), replace=False) if minus else []):
                forces[q][st] = False
        out.append(summ(evaluate(E, W, qids, cur_rb, forces))["acc"])
    return float(np.mean(out))


# ==================================================================================
# changes
# ==================================================================================
def step_unit(sig):
    return 1.0 if sig == "answer_words" else 0.025


def moved(sig, v, v0):
    return abs(v - v0) * (0.05 if sig == "answer_words" else 1.0)


def threshold_steps(rb):
    out = []
    for i, r in enumerate(rb["rules"]):
        if r.get("hard"):
            continue
        for j, (s, o, v) in enumerate(r["when"]):
            if o == "==" or s in ("is_yes_no", "is_cannot", "is_comparison", "ret_agree", "ret_escalated"):
                continue
            for k in (-2, -1, 1, 2):
                nv = round(v + k * step_unit(s), 4)
                hi = 50.0 if s == "answer_words" else 1.0
                if not (0.0 <= nv <= hi):
                    continue
                c = copy.deepcopy(rb)
                c["rules"][i]["when"][j][2] = nv
                out.append((f"threshold: {r['id']} {s} {o} {v:g} -> {nv:g}", c))
    return out


def valid_rule(E, r, errs):
    if not (isinstance(r, dict) and r.get("stage") in ("retrieval", "retrieval2", "answer") and r.get("then") in ("escalate", "keep")
            and isinstance(r.get("when"), list) and r["when"]):
        errs.append("malformed rule")
        return False
    for c in r["when"]:
        if not (isinstance(c, (list, tuple)) and len(c) == 3 and c[0] in SIGNALS and c[1] in OPS and isinstance(c[2], (int, float))):
            errs.append(f"bad condition {c}")
            return False
        st = SIGNALS[c[0]][0]
        if st != r["stage"] and not (r["stage"] in ("answer", "retrieval2") and st == "retrieval"):
            errs.append(f"signal {c[0]} is not available at stage {r['stage']}")
            return False
    if r.get("to") and r["to"] not in E.ladder[1:]:
        errs.append("bad target")
        return False
    return True


def apply_proposal(E, rb, p):
    c = copy.deepcopy(rb)
    act = p.get("action")
    errs = []
    if act == "param":
        name, val = p.get("name"), p.get("value")
        allowed = {"ret_target": tuple(E.ladder[1:]), "ret_merge": ("union", "target"), "votes": (1, 2, 3, 4)}
        if name == "sure_threshold":
            if not isinstance(val, (int, float)) or val < E.H["H5_min_sure_score"] or val > 0.99:
                return None, f"sure_threshold must be between {E.H['H5_min_sure_score']} and 0.99 (H5)"
        elif name not in allowed or val not in allowed[name]:
            return None, f"setting {name}={val} not allowed"
        c["params"][name] = val
        return c, f"setting {name}: {rb['params'].get(name)} -> {val}"
    if act in ("amend", "remove"):
        idx = [i for i, r in enumerate(c["rules"]) if r["id"] == p.get("id")]
        if not idx:
            return None, f"no rule {p.get('id')}"
        if c["rules"][idx[0]].get("hard"):
            return None, f"{p.get('id')} is locked (hard rule)"
        if act == "remove":
            del c["rules"][idx[0]]
            return c, f"remove {p.get('id')}"
        nr = dict(p.get("rule") or {}, id=p.get("id"), hard=False)
        if not valid_rule(E, nr, errs):
            return None, "; ".join(errs)
        nr["text"] = nr.get("text") or rule_text(nr, E.names)
        c["rules"][idx[0]] = nr
        return c, f"amend {rule_text(nr, E.names)}"
    if act == "add":
        nr = dict(p.get("rule") or {}, hard=False)
        ids = [r["id"] for r in c["rules"]]
        nr["id"] = nr["id"] if nr.get("id") and nr["id"] not in ids and not str(nr["id"]).startswith("H") else f"N{len(ids) + 1}"
        if not valid_rule(E, nr, errs):
            return None, "; ".join(errs)
        nr["text"] = nr.get("text") or rule_text(nr, E.names)
        same = [i for i, r in enumerate(c["rules"]) if r["stage"] == nr["stage"] and not r.get("hard")]
        keeps = [i for i in same if c["rules"][i]["then"] == "keep"]
        pos = p.get("position")
        if pos == "first" and same:
            at = same[0]
        elif pos == "last" or not same:
            at = same[-1] + 1 if same else len(c["rules"])
        else:
            at = keeps[0] if keeps else same[-1] + 1
        c["rules"].insert(at, nr)
        return c, f"add {rule_text(nr, E.names)}"
    return None, f"unknown action {act}"


def h7_ok(E, base_rb, cand_rb):
    base = {r["id"]: r for r in base_rb["rules"]}
    for r in cand_rb["rules"]:
        b = base.get(r["id"])
        if not b or r.get("hard") or len(b["when"]) != len(r["when"]):
            continue
        for (s, o, v), (s0, o0, v0) in zip(r["when"], b["when"]):
            if s == s0 and o == o0 and moved(s, v, v0) > E.H["H7_max_step"] + 1e-9:
                return False, f"H7: {r['id']} {s} moved {moved(s, v, v0):.3f} this round (max {E.H['H7_max_step']})"
    return True, ""


# ==================================================================================
# planner
# ==================================================================================
def _bands(vals, y1, y2, n=5):
    v = np.asarray(vals, float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & ((v <= hi) if hi == edges[-1] else (v < hi))
        if m.sum():
            out.append(f"{lo:.3g}-{hi:.3g}: n={int(m.sum())}, small right {100*np.mean(np.asarray(y1)[m]):.0f}%, big right {100*np.mean(np.asarray(y2)[m]):.0f}%")
    return out


def evidence(E, rb, rows):
    W = E.dev
    s = summ(rows)
    L = ["CURRENT RESULTS on dev half A: " + json.dumps({k: round(v, 4) for k, v in s.items() if k != "flops"})
         + f"; compute {s['flops'] / E.base_flops_A:.3f} x of 'cheapest retriever -> big answerer always'"]
    cnt = {}
    for r in rows:
        for f in r["fired"]:
            cnt.setdefault(f, []).append(r)
    L.append("PER RULE (questions on which it decided; final accuracy there):")
    for r in rb["rules"]:
        rr = cnt.get(r["id"], [])
        L.append(f"  {rule_text(r, E.names)} -> decided {len(rr)}, accuracy {100*np.mean([x['lenient'] for x in rr]) if rr else 0:.0f}%")
    keep = [r["qid"] for r in rows if not r["ans_esc"]]
    if keep:
        sel = {q: W.member(q, E.base)[0] for q in keep}
        sig = {q: E.answer_signals(W, q, sel[q]) for q in keep}
        y1 = [E.A[E.small][(q, sel[q])]["lenient"] for q in keep]
        y2 = [E.A[E.big][(q, sel[q])]["lenient"] for q in keep]
        L.append(f"ANSWER STAGE, questions NOT escalated now (n={len(keep)}): small answerer right {100*np.mean(y1):.0f}%, big right {100*np.mean(y2):.0f}% on the same passages. By band:")
        for k in ("answer_conf_pct", "answer_stated", "words_in_passages", "answer_words"):
            L.append(f"  {k}: " + " | ".join(_bands([sig[q][k] for q in keep], y1, y2)))
        for k in ("is_yes_no", "is_comparison"):
            for val in (0.0, 1.0):
                m = [i for i, q in enumerate(keep) if sig[q][k] == val]
                if m:
                    L.append(f"  {k}={int(val)}: n={len(m)}, small right {100*np.mean([y1[i] for i in m]):.0f}%, big right {100*np.mean([y2[i] for i in m]):.0f}%")
    esc = [r for r in rows if r["ans_esc"]]
    if esc:
        L.append(f"ANSWER STAGE, escalated now (n={len(esc)}): final accuracy {100*np.mean([r['lenient'] for r in esc]):.0f}%; the small answerer alone "
                 f"{100*np.mean([E.A[E.small][(r['qid'], W.member(r['qid'], E.base)[0])]['lenient'] for r in esc]):.0f}%")
    qs = [r["qid"] for r in rows]
    vv = np.array([E.pct("base_conf", np.exp(lp_(W.member(q, E.base)[1]))) for q in qs])
    L.append("RETRIEVAL by base_conf_pct band (share whose shortlist holds both gold passages / big-answerer accuracy on it), per retriever:")
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)):
        m = [q for q, x in zip(qs, vv) if lo <= x < hi]
        if m:
            parts = [f"{E.names[r]} {100*np.mean([W.gold(q) <= set(W.member(q, r)[0]) for q in m]):.0f}% / "
                     f"{100*np.mean([E.A[E.big][(q, W.member(q, r)[0])]['lenient'] for q in m]):.0f}%" for r in E.ladder]
            L.append(f"  {lo:.1f}-{min(hi, 1):.1f} (n={len(m)}): " + "; ".join(parts))
    tf = {E.names[r]: np.mean([W.member(q, r)[2] for q in W.qids]) / 1e12 for r in E.ladder}
    tf[E.names[E.small]] = np.mean([E.A[E.small][(q, W.member(q, E.base)[0])]["flops"] for q in W.qids]) / 1e12
    tf[E.names[E.big]] = np.mean([E.A[E.big][(q, W.member(q, E.base)[0])]["flops"] for q in W.qids]) / 1e12
    L.append("COMPUTE per question (TFLOPs): " + ", ".join(f"{k} {v:.1f}" for k, v in tf.items()) + "; each big-answerer sample about as much as one big answer")
    return "\n".join(L)


PLANNER_SYS = ("You improve a rulebook that decides, per question, when a small model's work is good enough and when to escalate to a bigger model. "
               "You propose small, testable changes; every change is tested on held-out questions and kept only if it helps.")


def planner_prompt(E, rb, rows, history):
    H = E.H
    sig = "\n".join(f"  {k} ({v[0]}): {v[1]}" for k, v in SIGNALS.items())
    ladder = ", ".join(f"{r} = {E.names[r]}" for r in E.ladder)
    return f"""THE PIPELINE. The cheapest retriever ({E.names[E.base]}) shortlists passages -> retrieval rules may escalate to a bigger retriever (retrievers: {ladder}; shortlists are ensembled) -> retrieval2 rules may escalate once more -> {E.names[E.small]} answers -> answer rules may escalate: {E.names[E.big]} answers; if it agrees that answer is kept, otherwise {rb['params']['votes']} sampled {E.names[E.big]} answers join a majority vote.
Rules are checked in order (LOCKED first); the first rule whose conditions all hold decides; default = keep.

SIGNALS (name (stage): meaning):
{sig}

CURRENT RULEBOOK (settings: {json.dumps(rb['params'])}):
""" + "\n".join("  " + rule_text(r, E.names) for r in rb["rules"]) + f"""

HOW CHANGES ARE JUDGED: on two separate halves of the dev questions, a change must be no less accurate and cheaper, or more accurate at no more than {int(100*H['H8_max_compute_per_point'])}% extra compute per accuracy point; it must also beat escalating the same number of random questions. While the rulebook is below the accuracy floor (best fixed pipeline minus {100*H['H4_max_acc_drop_vs_best_fixed']:.0f} point), any accuracy gain counts.
LOCKED LIMITS: at most {int(100*H['H3_max_share_big'])}% of questions may reach {E.names[E.big]}; at most {H['H6_max_soft_rules']} soft rules; each escalate rule must decide on at least {100*H['H6_min_support_share']:.0f}% of dev questions; LOCKED rules cannot be changed or removed; thresholds move at most {H['H7_max_step']} per round (percentile for *_pct signals, raw units for 0-1 signals, 1 word for answer_words).

EVIDENCE:
{evidence(E, rb, rows)}

EARLIER PROPOSALS AND OUTCOMES:
{chr(10).join(history[-12:]) or '  (none yet)'}

TASK. Propose up to 3 changes that should raise accuracy or cut compute. Escalate where the big answerer is clearly more often right than the small one; keep where they are close. Answer with ONLY a JSON list, each item one of:
{{"action":"add","position":"first"|"before_keep"|"last","rule":{{"stage":"answer","when":[["signal","<",0.5]],"then":"escalate"|"keep","to":"<retriever key, retrieval only, optional>","text":"plain-English rule"}},"why":"..."}}
{{"action":"amend","id":"A1","rule":{{...same fields...}},"why":"..."}}
{{"action":"remove","id":"A4","why":"..."}}
{{"action":"param","name":"votes"|"ret_target"|"ret_merge"|"sure_threshold","value":...,"why":"..."}}
Position: "before_keep" (default) = before the stage's first keep rule; a rule after rules that already cover every case never fires.
Operators: < <= > >= ==. Values are numbers."""


def parse_json_list(text):
    m = re.search(r"\[.*\]", text or "", flags=re.S)
    if not m:
        return []
    for cand in (m.group(0), re.sub(r",\s*([\]}])", r"\1", m.group(0))):
        try:
            return [x for x in json.loads(cand) if isinstance(x, dict)][:3]
        except Exception:  # noqa: BLE001
            continue
    return []


MOCK_PROPOSALS = [
    [{"action": "amend", "id": "A4", "rule": {"stage": "answer", "when": [["words_in_passages", ">=", 0.97], ["answer_stated", "<=", 0.9]], "then": "escalate"}, "why": "mock"},
     {"action": "param", "name": "votes", "value": 3, "why": "mock"}],
    [{"action": "add", "rule": {"stage": "answer", "when": [["answer_conf_pct", "<", 0.15]], "then": "escalate"}, "why": "mock"},
     {"action": "amend", "id": "H1", "rule": {"stage": "retrieval", "when": [["base_conf_pct", "<", 0.5]], "then": "escalate"}, "why": "mock: must be refused"}],
    [{"action": "remove", "id": "A4", "why": "mock"}],
    [{"action": "param", "name": "sure_threshold", "value": 0.8, "why": "mock: must be refused (H5)"}],
]


# ==================================================================================
def seed_rulebook(cfg):
    R = cfg["rules"]
    return {"version": 0, "params": dict(R["params"]), "rules": [dict(r, when=[list(c) for c in r["when"]]) for r in R["seed_rules"]]}


def fixed_dev(E):
    W = E.dev
    return {r: float(np.mean([E.A[E.big][(q, W.member(q, r)[0])]["lenient"] for q in W.qids])) for r in E.ladder}


def learn(cfg):
    E = Env(cfg)
    out = os.path.join(cfg["run_dir"], "rules")
    os.makedirs(os.path.join(out, "rulebooks"), exist_ok=True)
    W = E.dev
    fx = fixed_dev(E)
    best_fixed = max(fx.values())
    E.base_flops_A = float(sum(W.member(q, E.base)[2] + E.A[E.big][(q, W.member(q, E.base)[0])]["flops"] for q in E.halfA))
    log("dev fixed pipelines (retriever -> big answerer): " + ", ".join(f"{E.names[r]} {100*v:.1f}%" for r, v in fx.items()))
    state_p = os.path.join(out, "learn_state.json")
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"round": 0, "rulebook": seed_rulebook(cfg), "history": [], "log": []}
    rb = st["rulebook"]
    write_json(os.path.join(out, "rulebooks", "rulebook_v0_seed.json"), seed_rulebook(cfg))
    planner = None
    R = cfg["rules"]
    while st["round"] < R["rounds"]:
        rnd = st["round"] + 1
        base_rb = copy.deepcopy(rb)
        cur = {h: evaluate(E, W, qs, rb) for h, qs in (("A", E.halfA), ("B", E.halfB))}
        cur_dev = cur["A"] + cur["B"]
        log(f"round {rnd}: dev accuracy {100*summ(cur_dev)['acc']:.1f}%, big answerer {100*summ(cur_dev)['big']:.0f}%, retrieval escalated {100*summ(cur_dev)['ret_esc']:.0f}%")
        cands = [("code", d, c) for d, c in threshold_steps(rb)]
        if cfg["backend"] == "mock":
            props = MOCK_PROPOSALS[(rnd - 1) % len(MOCK_PROPOSALS)]
        else:
            if planner is None:
                pm = cfg["models"][cfg["planner"]]
                planner = make_engine(cfg, pm, max_model_len=16384)
            ptxt = planner_prompt(E, rb, cur["A"], st["history"])
            raw = planner.generate([planner.encode(PLANNER_SYS, ptxt)], 1, 0.7, 2500, [zlib.crc32(f"r{rnd}".encode()) % 99991], [{"kind": "plan"}])[0][0][0]
            open(os.path.join(out, f"planner_round{rnd}.txt"), "w", encoding="utf-8").write(ptxt + "\n\n=== RESPONSE ===\n" + raw)
            props = parse_json_list(raw)
        log(f"  planner: {len(props)} proposals")
        for p in props:
            c, d = apply_proposal(E, rb, p)
            if c is None:
                st["log"].append({"round": rnd, "source": "planner", "change": json.dumps(p)[:300], "decision": "rejected", "reason": d})
                st["history"].append(f"  round {rnd}: {json.dumps(p)[:200]} -> REJECTED ({d})")
            else:
                cands.append(("planner", d + (f" (why: {p.get('why')})" if p.get("why") else ""), c))
        tried, accepted = set(), True
        while accepted:
            accepted = False
            scored = []
            for src, d, c in cands:
                if d in tried:
                    continue
                ok7, why7 = h7_ok(E, base_rb, c)
                if not ok7:
                    tried.add(d)
                    st["log"].append({"round": rnd, "source": src, "change": d, "decision": "rejected", "reason": why7})
                    continue
                res = {h: evaluate(E, W, qs, c) for h, qs in (("A", E.halfA), ("B", E.halfB))}
                repair = summ(cur_dev)["acc"] < best_fixed - E.H["H4_max_acc_drop_vs_best_fixed"]
                okA, dAa, dCa = ok_half(E, summ(res["A"]), summ(cur["A"]), repair)
                okB, dAb, dCb = ok_half(E, summ(res["B"]), summ(cur["B"]), repair)
                rec = {"round": rnd, "source": src, "change": d, "half A": f"{100*dAa:+.1f} pt, {100*dCa:+.1f}% compute", "half B": f"{100*dAb:+.1f} pt, {100*dCb:+.1f}% compute"}
                if not (okA and okB):
                    tried.add(d)
                    st["log"].append(dict(rec, decision="rejected", reason="not better on both halves" + (" (repair mode)" if repair else "")))
                    if src == "planner":
                        st["history"].append(f"  round {rnd}: {d} -> rejected (A {rec['half A']}, B {rec['half B']})")
                    continue
                dev_rows = res["A"] + res["B"]
                g = guardrails(E, c, dev_rows, best_fixed, summ(cur_dev)["acc"])
                if g:
                    tried.add(d)
                    st["log"].append(dict(rec, decision="rejected", reason="; ".join(g)))
                    if src == "planner":
                        st["history"].append(f"  round {rnd}: {d} -> rejected ({'; '.join(g)})")
                    continue
                changed = any(a["ret_esc"] != b["ret_esc"] or a["ans_esc"] != b["ans_esc"] for a, b in zip(dev_rows, cur_dev))
                ctrl = random_control(E, W, W.qids, cur_dev, dev_rows, rb, R.get("random_control_draws", 10), rnd) if changed else -1.0
                acc = summ(dev_rows)["acc"]
                if changed and acc <= ctrl + 1e-9:
                    tried.add(d)
                    st["log"].append(dict(rec, decision="rejected", reason=f"no better than random escalation of the same size ({100*acc:.1f}% vs {100*ctrl:.1f}%)"))
                    if src == "planner":
                        st["history"].append(f"  round {rnd}: {d} -> rejected (no better than random escalation)")
                    continue
                util = 100 * (dAa + dAb) / 2 - ((dCa + dCb) / 2 / E.H["H8_max_compute_per_point"] if not repair else 0.0)
                scored.append((util, src, d, c, res, rec, ctrl))
            if scored:
                util, src, d, c, res, rec, ctrl = max(scored, key=lambda x: x[0])
                tried.add(d)
                rb, cur = c, res
                cur_dev = res["A"] + res["B"]
                st["log"].append(dict(rec, decision="ACCEPTED", reason=f"both halves pass; random control {100*ctrl:.1f}%" if ctrl >= 0 else "both halves pass"))
                st["history"].append(f"  round {rnd}: {d} -> ACCEPTED (A {rec['half A']}, B {rec['half B']})")
                log(f"  ACCEPTED [{src}] {d} | A {rec['half A']} | B {rec['half B']}")
                accepted = True
        rb["version"] = rnd
        st["round"], st["rulebook"] = rnd, rb
        write_json(os.path.join(out, "rulebooks", f"rulebook_v{rnd}.json"), rb)
        write_json(state_p, st)
        log(f"round {rnd} done: dev accuracy {100*summ(cur_dev)['acc']:.1f}%, big answerer {100*summ(cur_dev)['big']:.0f}%")
    write_json(os.path.join(out, "rulebooks", "rulebook_final.json"), rb)
    log("learning complete")


# ==================================================================================
def _boot(a, b, B=4000):
    rng = np.random.default_rng(0)
    n = len(a)
    d = [a[w].mean() - b[w].mean() for w in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))


def _fit_lr(X, y):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    y = np.asarray(y, float)
    if y.min() == y.max():
        c = float(y.mean())
        return lambda Z: np.full(len(Z), c)
    m = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(np.asarray(X, float), y)
    return lambda Z: m.predict_proba(np.asarray(Z, float))[:, 1]


def test(cfg):
    import pandas as pd
    E = Env(cfg)
    out = os.path.join(cfg["run_dir"], "rules")
    res = os.path.join(cfg["run_dir"], "results")
    os.makedirs(res, exist_ok=True)
    Wt = World(cfg, "test", E.A, E.SM)
    seed, final = seed_rulebook(cfg), json.load(open(os.path.join(out, "rulebooks", "rulebook_final.json")))
    st = json.load(open(os.path.join(out, "learn_state.json")))
    bkey = cfg["baseline"]["key"]
    bl = {x["qid"]: x for x in read_jsonl(ret_file(cfg, "test", bkey))}
    fixed = np.array([(E.A[E.big][(q, key(bl[q]["sel"]))]["lenient"], E.A[E.big][(q, key(bl[q]["sel"]))]["EM"],
                       bl[q]["flops"] + E.A[E.big][(q, key(bl[q]["sel"]))]["flops"]) for q in Wt.qids])

    def single(r, m):
        return np.array([(E.A[m][(q, Wt.member(q, r)[0])]["lenient"], E.A[m][(q, Wt.member(q, r)[0])]["EM"],
                          Wt.member(q, r)[2] + E.A[m][(q, Wt.member(q, r)[0])]["flops"]) for q in Wt.qids])
    fx = fixed_dev(E)
    best_r = max(fx, key=fx.get)
    nm = E.names
    pols = {f"Fixed setup (AgentTTS's pick): {cfg['baseline']['label']} retrieves (untrained, step-by-step), {nm[E.big]} answers": fixed,
            f"{nm[E.base]} retrieves, {nm[E.small]} answers": single(E.base, E.small)}
    for r in E.ladder:
        pols[f"{nm[r]} retrieves, {nm[E.big]} answers"] = single(r, E.big)
    ref_name = f"{nm[best_r]} retrieves, {nm[E.big]} answers"
    detail = {}
    sure_thr = {}
    for name, rb in (("Starting rulebook (before any micro-experiment)", seed), (f"LEARNT rulebook (v{final['version']}, {st['round']} rounds)", final)):
        rows = evaluate(E, Wt, Wt.qids, rb)
        dev_rows = evaluate(E, E.dev, E.dev.qids, rb)
        sure = _fit_lr([r["sure_feats"] for r in dev_rows], [r["lenient"] for r in dev_rows])
        thr = max(E.H["H5_min_sure_score"], rb["params"]["sure_threshold"])
        for r, s in zip(rows, sure([r["sure_feats"] for r in rows])):
            r["sure_score"], r["flag"] = float(s), ("Sure" if s >= thr else "Unsure")
        detail[name], sure_thr[name] = rows, thr
        pols[name] = np.array([(r["lenient"], r["EM"], r["flops"]) for r in rows])
        rng = np.random.default_rng(3)
        accs = []
        for _ in range(cfg["rules"].get("random_control_draws", 10) * 2):
            fr, fa = rng.permutation([bool(r["ret_esc"]) for r in rows]), rng.permutation([bool(r["ans_esc"]) for r in rows])
            accs.append([r["lenient"] for r in evaluate(E, Wt, Wt.qids, rb, {q: {"ret": bool(a), "ans": bool(b)} for q, a, b in zip(Wt.qids, fr, fa)})])
        detail[name + "|ctrl"] = np.mean(accs, 0)
    orc = []
    for q in Wt.qids:
        s, _, f = Wt.member(q, E.base)
        a1, a2 = E.A[E.small][(q, s)], E.A[E.big][(q, s)]
        orc.append((max(a1["lenient"], a2["lenient"]), max(a1["EM"], a2["EM"]), f + a1["flops"] + (0 if a1["lenient"] else a2["flops"])))
    pols[f"Oracle: {nm[E.small]} if right, else {nm[E.big]}, on the {nm[E.base]} shortlist (needs the answer key)"] = np.array(orc)
    yref = pols[ref_name][:, 0]
    rows_p = []
    for name, v in pols.items():
        lo, hi = _boot(v[:, 0], fixed[:, 0])
        lo2, hi2 = _boot(v[:, 0], yref)
        row = {"Policy": name, "Accuracy (corrected label)": float(v[:, 0].mean()), "Exact match": float(v[:, 1].mean()),
               "Compute vs the fixed setup": float(v[:, 2].sum() / fixed[:, 2].sum() - 1),
               "vs fixed setup (points)": 100 * float(v[:, 0].mean() - fixed[:, 0].mean()), "low": lo, "high": hi,
               f"vs best single retriever on dev ({nm[best_r]}) (points)": 100 * float(v[:, 0].mean() - yref.mean()), "low ": lo2, "high ": hi2}
        if name in detail:
            rr, ctrl = detail[name], detail[name + "|ctrl"]
            lc, hc = _boot(v[:, 0], ctrl)
            sure = np.array([r["flag"] == "Sure" for r in rr])
            row |= {"Retrieval escalated": float(np.mean([r["ret_esc"] for r in rr])), "Reached the big answerer": float(np.mean([r["ans_esc"] for r in rr])),
                    "vs random escalation, same counts (points)": 100 * float(v[:, 0].mean() - ctrl.mean()), "low  ": lc, "high  ": hc,
                    "Flagged Sure": float(sure.mean()), "Right when Sure": float(v[sure, 0].mean()) if sure.any() else None,
                    "Right when Unsure": float(v[~sure, 0].mean()) if (~sure).any() else None}
        rows_p.append(row)
    lname = [k for k in detail if k.startswith("LEARNT") and "|" not in k][0]
    rows_r = [{"Rule": rule_text(r, nm), "Plain English": r.get("text", ""), "Decided on test questions": sum(r["id"] in x["fired"] for x in detail[lname]),
               "Accuracy there": (float(np.mean([x["lenient"] for x in detail[lname] if r["id"] in x["fired"]])) if any(r["id"] in x["fired"] for x in detail[lname]) else None)}
              for r in final["rules"]]
    per_q = [{"qid": x["qid"], "question": Wt.qd[x["qid"]]["question"], "gold": Wt.qd[x["qid"]]["answer"], "rules that decided (retrieval | retrieval2 | answer)": " | ".join(x["fired"]),
              "retrievers": x["members"], "votes": x["votes"], "answer": x["answer"], "sure score": x["sure_score"], "flag": x["flag"], "right (corrected)": x["lenient"],
              "exact match": x["EM"], "TFLOPs": x["flops"] / 1e12} for x in detail[lname]]
    for nmf, rr in (("policies.jsonl", rows_p), ("rules_on_test.jsonl", rows_r), ("micro_experiments.jsonl", st["log"]), ("per_question.jsonl", per_q)):
        write_jsonl(os.path.join(res, nmf), rr)

    def df(rr):
        d = pd.DataFrame(rr)
        for c in d.columns:
            if d[c].isna().any():
                d[c] = d[c].astype(object).where(d[c].notna(), "-")
        return d
    with pd.ExcelWriter(os.path.join(res, "rulebook_results.xlsx"), engine="openpyxl") as xw:
        for sheet, rr in (("Test policies", rows_p), ("Learnt rules on test", rows_r), ("Micro-experiment log", st["log"]), ("Per question", per_q)):
            df(rr).to_excel(xw, sheet_name=sheet, index=False)
    pct = {"Accuracy (corrected label)", "Exact match", "Compute vs the fixed setup", "Retrieval escalated", "Reached the big answerer", "Flagged Sure",
           "Right when Sure", "Right when Unsure", "Accuracy there"}
    T = lambda rr, cols: ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join("-" if r.get(c) is None else (r[c] if isinstance(r[c], str) else (f"{100*r[c]:.1f}%" if c in pct else
                                                                                            (str(r[c]) if isinstance(r[c], int) else f"{r[c]:+.1f}"))) for c in cols) + " |" for r in rr))
    cmp_col = [c for c in rows_p[0] if c.startswith("vs best single")][0]
    acc_log = [x for x in st["log"] if x["decision"] == "ACCEPTED"]
    L = ["# Self-learnt rulebook: test results\n",
         f"Dev: {len(E.dev.qids)} questions (halves A and B), {st['round']} rounds of micro-experiments. Test: {len(Wt.qids)} questions, used once. "
         "Compute counts every model that runs. Intervals: paired bootstrap, 95%.\n", "## Accuracy and compute\n",
         T(rows_p, ["Policy", "Accuracy (corrected label)", "Exact match", "Compute vs the fixed setup", "vs fixed setup (points)", "low", "high", cmp_col, "low ", "high "]), "",
         "## Switching against chance; Sure / Unsure\n",
         T([r for r in rows_p if "Retrieval escalated" in r], ["Policy", "Retrieval escalated", "Reached the big answerer", "vs random escalation, same counts (points)",
                                                               "low  ", "high  ", "Flagged Sure", "Right when Sure", "Right when Unsure"]), "",
         "## The learnt rulebook\n", "Settings: " + json.dumps(final["params"]) + "\n", T(rows_r, ["Rule", "Decided on test questions", "Accuracy there"]), "",
         f"## Changes accepted by the micro-experiments ({len(acc_log)} of {len(st['log'])} tried)\n",
         T(acc_log, ["round", "source", "change", "half A", "half B"]) if acc_log else "None: every proposed change failed the evidence test.", ""]
    open(os.path.join(res, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log(f"results -> {res}")
