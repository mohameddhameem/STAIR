"""The rulebook (AL-03 .. AL-06): soft rules learnt by micro-experiments on the val questions; locked rules and guardrails that
no experiment may change.

Per question:
  RETRIEVAL  the first (cheapest) retriever shortlists. After each retriever, the retrieval rules are checked: escalate -> run a
             larger retriever (the next one, or the rule's target / skip) and ensemble all shortlists so far (all agree -> it;
             majority -> it; else union). Repeats until a rule says keep, max_ret_members is reached or the ladder ends.
  ANSWER     the first answerer answers on the final shortlist. After each answer, the answer rules are checked: escalate -> a
             larger answerer answers; if it agrees with an earlier answer, that answer is accepted; if not, the rules are
             checked again (escalate further), and when they stop, the answers so far plus `votes` sampled answers of the last
             answerer go to a majority vote (ties -> the larger model).
  FLAG       a calibrated score (fitted on val) gives Sure / Unsure, also for the largest model's answers.
Rules are checked in order (locked first); the first rule whose conditions all hold decides; default = keep.

learn: rounds of micro-experiments on val halves A and B. Code tries fine threshold steps for every soft rule; the planner LLM
proposes up to 3 changes from evidence tables. A change is kept only if, on BOTH halves, it is cheaper at no accuracy loss or
more accurate at <= H8 compute per point; it beats a random-escalation control of the same size; it breaks no guardrail.
test: the frozen rulebook once on the test questions, next to fixed pipelines, random escalation and the starting rulebook."""
import copy
import json
import os
import re

import numpy as np

from . import data as D
from .common import agree, ensemble, units, key, log, lp_, normalize_answer, read_jsonl, seed_of, vote, write_json, write_jsonl
from .config import model_tag, serve_entry, variant
from .engines import free_gpu, make_engine
from .progress import Tracker, mark_done
from .states import SPLITS, f_ans, f_ret, f_smp

_CANNOT = re.compile(r"(unknown|cannot|can't|not (provided|mentioned|specified|available|enough|stated|given)|no (information|answer|mention)|insufficient|unclear)")
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal}
SIGNALS = {
    "rung": ("both", "position of the current model in its ladder (0 = the smallest)"),
    "ret_conf_pct": ("retrieval", "current retriever's confidence in its shortlist (token probability), percentile among val questions for that retriever"),
    "ret_probe_pct": ("retrieval", "hidden-state probe: how likely the current shortlist holds both gold passages, percentile among val questions"),
    "ret_agree": ("retrieval", "1 if the current retriever picked the same passages as the previous one (1 for the first retriever)"),
    "sel_size": ("retrieval", "number of passages in the current (ensembled) shortlist"),
    "ans_conf_pct": ("answer", "current answerer's token probability for its answer, percentile among val questions for that answerer"),
    "ans_probe_pct": ("answer", "hidden-state probe: how likely the current answer is right, percentile among val questions"),
    "answer_stated": ("answer", "current answerer's stated confidence (0-1; 0.5 when it states none)"),
    "words_in_passages": ("answer", "share of the answer's words found in the shortlisted passages (0-1)"),
    "is_yes_no": ("answer", "1 if the answer is yes/no"),
    "is_cannot": ("answer", "1 if the answer is empty or says it cannot be determined"),
    "answer_words": ("answer", "number of words in the answer"),
    "is_comparison": ("answer", "1 if HotpotQA marks the question as a comparison question"),
    "ret_escalated": ("answer", "1 if retrieval was escalated for this question"),
    "ans_agree": ("answer", "1 if the current answer agrees with the previous answerer's (1 for the first answerer)"),
}
BINARY = {"is_yes_no", "is_cannot", "is_comparison", "ret_escalated", "ret_agree", "ans_agree"}
INTEGER = {"rung", "sel_size", "answer_words"}


def rule_text(r):
    c = " AND ".join(f"{s} {o} {v:g}" for s, o, v in r["when"])
    at = f" at rung {r['at']}" if r.get("at") is not None else ""
    to = f" (to {r['to']})" if r.get("to") else (f" (skip {r['skip']})" if r.get("skip") else "")
    return f"[{r['id']}{', LOCKED' if r.get('hard') else ''}] {r['stage']}{at}: IF {c} THEN {r['then']}{to}"


# ==================================================================================
class Env:
    """Saved outputs of one workstream (val and test), signals and their val distributions."""

    def __init__(self, cfg, ws):
        self.cfg, self.ws = cfg, ws
        W = cfg["workstreams"][ws]
        self.W = W
        self.RL, self.AL = list(W["retrievers"]), list(W["answerers"])
        self.H = cfg["rules"]["hard"]
        self.names = {v: variant(cfg, v)["label_full"] for v in self.RL + self.AL}
        self.tags = {v: model_tag(variant(cfg, v)) for v in self.RL + self.AL}
        self.nominal = {v: cfg["models"][v.split(":")[0]]["nominal"] for v in self.RL + self.AL}
        self.nd = cfg["workstreams"]["ws1"]["np_nd"] if "ws1" in cfg["workstreams"] else {"Retrieval": [2048, 128], "Question Answering": [256, 64]}
        sig = W.get("signals", "all")
        self.allowed = set(SIGNALS) if sig == "all" else set(sig)
        self.Q = {sp: D.load(cfg, sp) for sp in SPLITS}
        self.qd = {q["qid"]: q for sp in SPLITS for q in self.Q[sp]}
        self.R = {sp: {v: {r["qid"]: r for r in read_jsonl(f_ret(cfg, sp, v))} for v in self.RL} for sp in SPLITS}
        self.A = {v: {(r["qid"], key(r["sel"])): r for r in read_jsonl(f_ans(cfg, v))} for v in self.AL}
        self.SM = {v: {(r["qid"], key(r["sel"])): r for r in read_jsonl(f_smp(cfg, v))} for v in self.AL[1:]}
        from .probe import f_probe
        self.PR = {v: {(r["split"], r["qid"]): r["probe"] for r in read_jsonl(f_probe(cfg, v, "ret"))} for v in self.RL}
        self.PA = {v: {(r["split"], r["qid"], key(r["sel"])): r["probe"] for r in read_jsonl(f_probe(cfg, v, "ans"))} for v in self.AL}
        self.qids = {sp: [q["qid"] for q in self.Q[sp] if all(q["qid"] in self.R[sp][v] for v in self.RL)] for sp in SPLITS}
        dq = self.qids["val"]
        base_sel = {q: key(self.R["val"][self.RL[0]][q]["sel"]) for q in dq}
        self.dist = {}
        for v in self.RL:
            self.dist[("ret_conf", v)] = np.sort([np.exp(lp_(self.R["val"][v][q].get("mean_logprob"))) for q in dq])
            self.dist[("ret_probe", v)] = np.sort([self.PR[v].get(("val", q), 0.5) for q in dq])
        for v in self.AL:
            rows = [self.A[v].get((q, base_sel[q])) for q in dq]
            self.dist[("ans_conf", v)] = np.sort([np.exp(lp_(r.get("mean_logprob"))) for r in rows if r])
            self.dist[("ans_probe", v)] = np.sort([self.PA[v].get(("val", q, base_sel[q]), 0.5) for q in dq])
        rng = np.random.default_rng(20260729 + int(cfg.get("seed", 0)))   # val halves A / B; the team seed, shifted by the run seed
        perm = rng.permutation(len(dq))
        self.halfA = sorted(dq[i] for i in perm[: len(perm) // 2])
        self.halfB = sorted(dq[i] for i in perm[len(perm) // 2:])
        self._c = {}

    # ---------------------------------------------------------------- signals
    def pct(self, kind, v, raw):
        a = self.dist[(kind, v)]
        return float(np.searchsorted(a, raw, side="right") / len(a)) if len(a) else 0.5

    def ret_sig(self, sp, q, members, sel):
        k = ("r", sp, q, tuple(members), sel)
        if k not in self._c:
            v = self.RL[members[-1]]
            r = self.R[sp][v][q]
            prev = key(self.R[sp][self.RL[members[-2]]][q]["sel"]) if len(members) > 1 else key(r["sel"])
            s = {"rung": float(members[-1]), "ret_conf_pct": self.pct("ret_conf", v, np.exp(lp_(r.get("mean_logprob")))),
                 "ret_agree": float(key(r["sel"]) == prev), "sel_size": float(len(sel))}
            if (sp, q) in self.PR[v]:
                s["ret_probe_pct"] = self.pct("ret_probe", v, self.PR[v][(sp, q)])
            self._c[k] = s
        return self._c[k]

    def ans_sig(self, sp, q, j, sel, prev_pred, ret_esc):
        k = ("a", sp, q, j, sel, prev_pred, ret_esc)
        if k not in self._c:
            v = self.AL[j]
            a = self.A[v][(q, sel)]
            qq = self.qd[q]
            pred = a.get("prediction") or ""
            p = normalize_answer(pred)
            ps = set(normalize_answer(" ".join(qq["ctxs_candidate"][i - 1] for i in sel if 1 <= i <= len(qq["ctxs_candidate"]))).split())
            toks = p.split()
            s = {"rung": float(j), "ans_conf_pct": self.pct("ans_conf", v, np.exp(lp_(a.get("mean_logprob")))),
                 "answer_stated": float(a["stated_conf"]) if a.get("stated_conf") is not None else 0.5,
                 "words_in_passages": float(np.mean([t in ps for t in toks])) if toks else 0.0,
                 "is_yes_no": float(p in ("yes", "no")), "is_cannot": float((not p) or bool(_CANNOT.search(pred.lower()))),
                 "answer_words": float(len(toks)), "is_comparison": float(qq.get("type") == "comparison"), "ret_escalated": float(ret_esc),
                 "ans_agree": 1.0 if prev_pred is None else float(agree(pred, prev_pred))}
            if (sp, q, sel) in self.PA[v]:
                s["ans_probe_pct"] = self.pct("ans_probe", v, self.PA[v][(sp, q, sel)])
            self._c[k] = s
        return self._c[k]

    def visible(self, sig):
        return {k: v for k, v in sig.items() if k in self.allowed or k in ("rung", "is_cannot")}

    # ---------------------------------------------------------------- one question
    def run_q(self, sp, q, rb, force=None):
        P, rules = rb["params"], rb["rules"]
        maxm = int(P.get("max_ret_members", self.cfg["rules"]["max_ret_members"]))
        maxm = min(maxm, int(self.cfg["rules"]["max_ret_members"]))
        members, fired = [0], []
        r0 = self.R[sp][self.RL[0]][q]
        sel = key(r0["sel"])
        rsig0 = None
        fpath = (force or {}).get("ret_path")
        while True:
            sig = self.ret_sig(sp, q, members, sel)
            rsig0 = rsig0 or sig
            if fpath is not None:
                nxt = fpath[len(members)] if len(members) < len(fpath) else None
                fired.append("forced")
            else:
                if len(members) >= maxm or members[-1] >= len(self.RL) - 1:
                    break
                r = decide(rules, "retrieval", self.visible(sig), members[-1])
                fired.append(r["id"] if r else "-")
                nxt = target(r, members[-1], self.RL) if (r and r["then"] == "escalate") else None
            if nxt is None:
                break
            members.append(nxt)
            sel = self.ens(sp, q, members)
        u_ret = sum(self.R[sp][self.RL[i]][q]["units"] for i in members)
        u_ans = 0.0
        ret_esc = len(members) > 1
        # ---- answers
        j, hist, used = 0, [], {}
        fapath = (force or {}).get("ans_path")
        prev_pred, final, consensus, support = None, None, False, 1.0
        k_a = 0
        while True:
            v = self.AL[j]
            a = dict(self.A[v][(q, sel)], model=v)
            u_ans += a["units"]
            used[v] = used.get(v, 0) + 1
            hist.append(a)
            if prev_pred is not None and any(agree(a["prediction"], h["prediction"]) for h in hist[:-1]):
                final, consensus = a, True
                break
            sig = self.ans_sig(sp, q, j, sel, prev_pred, ret_esc)
            if fapath is not None:
                k_a += 1
                nxt = fapath[k_a] if k_a < len(fapath) else None
                fired.append("forced")
            else:
                if j >= len(self.AL) - 1:
                    nxt = None
                else:
                    r = decide(rules, "answer", self.visible(sig), j)
                    fired.append(r["id"] if r else "-")
                    nxt = target(r, j, self.AL) if (r and r["then"] == "escalate") else None
            if nxt is None:
                break
            prev_pred, j = a["prediction"], nxt
        if final is None:
            if len(hist) == 1:
                final = hist[0]
            else:
                pool = list(reversed(hist))
                nv = int(P.get("votes", 2))
                sm = self.SM.get(self.AL[j], {}).get((q, sel))
                if sm and nv > 0:
                    ss = [dict(s, model=self.AL[j] + "_sample") for s in sm["samples"][:nv]]
                    u_ans += units(self.nominal[self.AL[j]] / 1e9, 0, sum(s["out_tokens"] for s in ss))   # shared prompt counted once
                    used[self.AL[j]] += len(ss)
                    pool += ss
                final, sup = vote(pool)
                support = sup / len(pool)
        fs = self.ans_sig(sp, q, self.AL.index(final["model"].replace("_sample", "")), sel, None, ret_esc) if final["model"] in self.AL else {}
        return {"qid": q, "acc": float(final["acc"]), "lenient": float(final["lenient"]), "EM": float(final["EM"]), "F1": float(final.get("F1", 0.0)),
                "units": float(u_ret + u_ans), "units_retrieve": float(u_ret), "units_answer": float(u_ans),
                "ret_path": members, "ans_path": [self.AL.index(h["model"]) for h in hist], "ret_esc": float(ret_esc), "ans_esc": float(len(hist) > 1),
                "top_ret": float(members[-1] == len(self.RL) - 1), "top_ans": float(self.AL[-1] in used), "fired": fired, "answer": final.get("prediction"),
                "final_model": final["model"], "consensus": float(consensus),
                "models": ";".join([f"retriever={self.tags[self.RL[i]]}" for i in members] + [f"answerer={self.tags[v]}" for v in used]),
                "sure_feats": [float(len(members)), float(len(hist)), float(consensus), float(support), rsig0["ret_conf_pct"], rsig0.get("ret_probe_pct", 0.5),
                               fs.get("ans_conf_pct", 0.5), fs.get("answer_stated", 0.5), fs.get("words_in_passages", 0.0), fs.get("is_yes_no", 0.0),
                               fs.get("is_cannot", 0.0), fs.get("ans_probe_pct", 0.5)]}

    def ens(self, sp, q, members):
        k = ("e", sp, q, tuple(members))
        if k not in self._c:
            self._c[k] = ensemble([self.R[sp][self.RL[i]][q]["sel"] for i in members])
        return self._c[k]

    def evaluate(self, sp, qids, rb, forces=None):
        return [self.run_q(sp, q, rb, (forces or {}).get(q)) for q in qids]

    def fixed(self, sp, i, j, qids=None):
        """Fixed pipeline: retriever i alone -> answerer j once."""
        out = []
        for q in qids or self.qids[sp]:
            r = self.R[sp][self.RL[i]][q]
            a = self.A[self.AL[j]].get((q, key(r["sel"])))
            if a is None:
                continue
            out.append({"qid": q, "acc": a["acc"], "lenient": a["lenient"], "EM": a["EM"], "F1": a.get("F1", 0.0), "answer": a.get("prediction"),
                        "units": r["units"] + a["units"], "units_retrieve": r["units"], "units_answer": a["units"],
                        "models": f"retriever={self.tags[self.RL[i]]};answerer={self.tags[self.AL[j]]}"})
        return out


def variant_params(E, v):
    return E.cfg["models"][v.split(":")[0]]["params"]


def target(rule, cur, ladder):
    if rule.get("to") and rule["to"] in ladder and ladder.index(rule["to"]) > cur:
        return ladder.index(rule["to"])
    return min(len(ladder) - 1, cur + 1 + int(rule.get("skip", 0) or 0))


def match(rule, sig, rung):
    if rule.get("at") is not None and rung not in rule["at"]:
        return False
    return all(s in sig and OPS[o](sig[s], v) for s, o, v in rule["when"])


def decide(rules, stage, sig, rung):
    rs = [r for r in rules if r["stage"] == stage]
    for r in [r for r in rs if r.get("hard")] + [r for r in rs if not r.get("hard")]:
        if match(r, sig, rung):
            return r
    return None


def summ(rows):
    m = lambda k: float(np.mean([r.get(k, 0.0) for r in rows])) if rows else 0.0
    return {"acc": m("acc"), "lenient": m("lenient"), "em": m("EM"), "units": float(np.sum([r["units"] for r in rows])),
            "ret_esc": m("ret_esc"), "ans_esc": m("ans_esc"), "top_ret": m("top_ret"), "top_ans": m("top_ans")}


# ==================================================================================
# acceptance
# ==================================================================================
def ok_half(E, c, b, repair=False):
    dA, dC = c["acc"] - b["acc"], c["units"] / max(b["units"], 1e-9) - 1
    if repair:
        return dA > 0, dA, dC
    cheaper = dA >= 0 and dC <= -0.005
    better = dA > 0 and dC <= E.H["H8_max_compute_per_point"] * (100 * dA)
    return cheaper or better, dA, dC


def floor_of(fx, units_per_q):
    """H4 floor: the best fixed pipeline costing no more than this per question (no switching at the same price)."""
    ok = [a for a, f in fx.values() if f <= units_per_q * 1.0001]
    return max(ok) if ok else min(a for a, _ in fx.values())


def guardrails(E, rb, rows, fx, cur_acc):
    s = summ(rows)
    best_fixed = floor_of(fx, s["units"] / max(1, len(rows)))
    soft = [r for r in rb["rules"] if not r.get("hard")]
    why = []
    if s["top_ans"] > E.H["H3_max_share_top_answerer"] + 1e-9:
        why.append(f"H3: {100*s['top_ans']:.0f}% reach the largest answerer (max {100*E.H['H3_max_share_top_answerer']:.0f}%)")
    if s["top_ret"] > E.H["H3b_max_share_top_retriever"] + 1e-9:
        why.append(f"H3b: {100*s['top_ret']:.0f}% reach the largest retriever (max {100*E.H['H3b_max_share_top_retriever']:.0f}%)")
    if s["acc"] < best_fixed - E.H["H4_max_acc_drop_vs_best_fixed"] and s["acc"] < cur_acc:
        why.append(f"H4: val accuracy {100*s['acc']:.1f}% below the best fixed pipeline at no higher cost ({100*best_fixed:.1f}%) minus {100*E.H['H4_max_acc_drop_vs_best_fixed']:.0f} point")
    if len(soft) > E.H["H6_max_soft_rules"]:
        why.append(f"H6: {len(soft)} soft rules (max {E.H['H6_max_soft_rules']:.0f})")
    cnt = {}
    for r in rows:
        for f in r["fired"]:
            cnt[f] = cnt.get(f, 0) + 1
    need = int(np.ceil(E.H["H6_min_support_share"] * len(rows)))
    for r in soft:
        if r["then"] == "escalate" and cnt.get(r["id"], 0) < need:
            why.append(f"H6: rule {r['id']} decides on {cnt.get(r['id'], 0)} val questions (min {need})")
    return why


def _extend(path, n, top):
    p = list(path)
    while len(p) < n and p[-1] < top:
        p.append(p[-1] + 1)
    return p[:max(1, n)]


def random_control(E, qids, cur_rows, cand_rows, rb, draws, seed):
    """Same number of extra (or fewer) escalation steps as the candidate, given to random questions instead."""
    rng = np.random.default_rng(seed)
    by, cand = {r["qid"]: r for r in cur_rows}, {r["qid"]: r for r in cand_rows}
    # retrieval chains are capped like the rulebook's own (only chains of up to max_ret_members retrievers were answered)
    maxm = min(int(rb["params"].get("max_ret_members", E.cfg["rules"]["max_ret_members"])), int(E.cfg["rules"]["max_ret_members"]))
    out = []
    for _ in range(draws):
        forces = {q: {"ret_path": list(by[q]["ret_path"]), "ans_path": list(by[q]["ans_path"])} for q in qids}
        for kind, top, cap in (("ret_path", len(E.RL) - 1, maxm), ("ans_path", len(E.AL) - 1, len(E.AL))):
            d = {q: len(cand[q][kind]) - len(by[q][kind]) for q in qids}
            plus = [d[q] for q in qids if d[q] > 0]
            minus = [d[q] for q in qids if d[q] < 0]
            pool_p = [q for q in qids if forces[q][kind][-1] < top and len(forces[q][kind]) < cap]
            for q, inc in zip(rng.permutation(pool_p)[:len(plus)], plus):
                forces[q][kind] = _extend(forces[q][kind], min(cap, len(forces[q][kind]) + inc), top)
            pool_m = [q for q in qids if len(forces[q][kind]) > 1]
            for q, dec in zip(rng.permutation(pool_m)[:len(minus)], minus):
                forces[q][kind] = forces[q][kind][:max(1, len(forces[q][kind]) + dec)]
        out.append(summ(E.evaluate("val", qids, rb, forces))["acc"])
    return float(np.mean(out))


# ==================================================================================
# changes
# ==================================================================================
def step_unit(sig):
    return 1.0 if sig in INTEGER else 0.025


def moved(sig, v, v0):
    return abs(v - v0) * (0.05 if sig in INTEGER else 1.0)


def threshold_steps(rb):
    out = []
    for i, r in enumerate(rb["rules"]):
        if r.get("hard"):
            continue
        for j, (s, o, v) in enumerate(r["when"]):
            if o == "==" or s in BINARY or s == "rung":
                continue
            for k in (-2, -1, 1, 2):
                nv = round(v + k * step_unit(s), 4)
                hi = 50.0 if s in INTEGER else 1.0
                if not (0.0 <= nv <= hi):
                    continue
                c = copy.deepcopy(rb)
                c["rules"][i]["when"][j][2] = nv
                out.append((f"threshold: {r['id']} {s} {o} {v:g} -> {nv:g}", c))
    return out


def valid_rule(E, r, errs):
    if not (isinstance(r, dict) and r.get("stage") in ("retrieval", "answer") and r.get("then") in ("escalate", "keep")
            and isinstance(r.get("when"), list) and r["when"]):
        errs.append("malformed rule")
        return False
    for c in r["when"]:
        if not (isinstance(c, (list, tuple)) and len(c) == 3 and c[0] in SIGNALS and c[1] in OPS and isinstance(c[2], (int, float))):
            errs.append(f"bad condition {c}")
            return False
        if c[0] not in E.allowed and c[0] not in ("rung", "is_cannot"):
            errs.append(f"signal {c[0]} is not available in this workstream")
            return False
        st = SIGNALS[c[0]][0]
        if st not in ("both", r["stage"]):
            errs.append(f"signal {c[0]} is not available at stage {r['stage']}")
            return False
    lad = E.RL if r["stage"] == "retrieval" else E.AL
    if r.get("to") and r["to"] not in lad[1:]:
        errs.append("bad target")
        return False
    if r.get("at") is not None and not (isinstance(r["at"], list) and all(isinstance(x, int) and 0 <= x < len(lad) for x in r["at"])):
        errs.append("bad 'at'")
        return False
    if r.get("skip") is not None and not (isinstance(r["skip"], int) and 0 <= r["skip"] < len(lad)):
        errs.append("bad 'skip'")
        return False
    return True


def apply_proposal(E, rb, p):
    c = copy.deepcopy(rb)
    act = p.get("action")
    errs = []
    if act == "param":
        name, val = p.get("name"), p.get("value")
        allowed = {"votes": (0, 1, 2, 3, 4), "max_ret_members": tuple(range(2, int(E.cfg["rules"]["max_ret_members"]) + 1))}
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
        nr["text"] = nr.get("text") or rule_text(nr)
        c["rules"][idx[0]] = nr
        return c, f"amend {rule_text(nr)}"
    if act == "add":
        nr = dict(p.get("rule") or {}, hard=False)
        ids = [r["id"] for r in c["rules"]]
        nr["id"] = nr["id"] if nr.get("id") and nr["id"] not in ids and not str(nr["id"]).startswith("H") else f"N{len(ids) + 1}"
        if not valid_rule(E, nr, errs):
            return None, "; ".join(errs)
        nr["text"] = nr.get("text") or rule_text(nr)
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
        return c, f"add {rule_text(nr)}"
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
def _bands(vals, ys, n=5):
    v = np.asarray(vals, float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        msk = (v >= lo) & ((v <= hi) if hi == edges[-1] else (v < hi))
        if msk.sum():
            out.append(f"{lo:.3g}-{hi:.3g}: n={int(msk.sum())}, " + ", ".join(f"{nm} right {100*np.mean(np.asarray(y)[msk]):.0f}%" for nm, y in ys))
    return out


def evidence(E, rb, rows):
    s = summ(rows)
    L = ["CURRENT RESULTS on val half A: " + json.dumps({k: round(v, 4) for k, v in s.items() if k != "units"})]
    cnt = {}
    for r in rows:
        for f in r["fired"]:
            cnt.setdefault(f, []).append(r)
    L.append("PER RULE (questions on which it decided; final accuracy there):")
    for r in rb["rules"]:
        rr = cnt.get(r["id"], [])
        L.append(f"  {rule_text(r)} -> decided {len(rr)}, accuracy {100*np.mean([x['acc'] for x in rr]) if rr else 0:.0f}%")
    qs = [r["qid"] for r in rows]
    sel = {q: key(E.R["val"][E.RL[0]][q]["sel"]) for q in qs}
    ys = [(E.names[v], [E.A[v][(q, sel[q])]["acc"] if (q, sel[q]) in E.A[v] else 0.0 for q in qs]) for v in E.AL]
    sig = {q: E.visible(E.ans_sig("val", q, 0, sel[q], None, False)) for q in qs}
    L.append(f"ANSWER STAGE on the first retriever's shortlist (n={len(qs)}): " + ", ".join(f"{nm} right {100*np.mean(y):.0f}%" for nm, y in ys) + ". By band of the FIRST answerer's signals:")
    for k in sorted(set().union(*[set(x) for x in sig.values()])):
        if k in BINARY or k == "rung":
            for val in (0.0, 1.0):
                msk = [i for i, q in enumerate(qs) if sig[q].get(k) == val]
                if msk:
                    L.append(f"  {k}={int(val)}: n={len(msk)}, " + ", ".join(f"{nm} right {100*np.mean([y[i] for i in msk]):.0f}%" for nm, y in ys))
        else:
            L.append(f"  {k}: " + " | ".join(_bands([sig[q][k] for q in qs], ys)))
    rs = {q: E.visible(E.ret_sig("val", q, [0], sel[q])) for q in qs}
    for k in ("ret_conf_pct", "ret_probe_pct"):
        if all(k in rs[q] for q in qs):
            vv = np.array([rs[q][k] for q in qs])
            L.append(f"RETRIEVAL by the first retriever's {k} band (share of shortlists holding both gold passages), per retriever:")
            for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)):
                msk = [q for q, x in zip(qs, vv) if lo <= x < hi]
                if msk:
                    L.append(f"  {lo:.1f}-{min(hi, 1):.1f} (n={len(msk)}): " + "; ".join(
                        f"{E.names[v]} {100*np.mean([E.R['val'][v][q]['found_both'] for q in msk]):.0f}%" for v in E.RL))
    tf = {E.names[v]: np.mean([E.R["val"][v][q]["units"] for q in E.qids["val"]]) for v in E.RL}
    for v in E.AL:
        tf[E.names[v]] = np.mean([E.A[v][(q, sel[q])]["units"] for q in qs if (q, sel[q]) in E.A[v]])
    L.append("COMPUTE per question (AgentTTS units per call): " + ", ".join(f"{k} {v:.2f}" for k, v in tf.items()))
    return "\n".join(L)


PLANNER_SYS = ("You improve a rulebook that decides, per question, when a small model's work is good enough and when to escalate to a bigger model. "
               "You propose small, testable changes; every change is tested on held-out questions and kept only if it helps.")


def planner_prompt(E, rb, rows, history):
    H = E.H
    sig = "\n".join(f"  {k} ({v[0]}): {v[1]}" for k, v in SIGNALS.items() if k in E.allowed or k in ("rung", "is_cannot"))
    rl = ", ".join(f"rung {i} = {E.names[v]} ('{v}')" for i, v in enumerate(E.RL))
    al = ", ".join(f"rung {i} = {E.names[v]} ('{v}')" for i, v in enumerate(E.AL))
    return f"""THE PIPELINE. Retrievers (cheapest first): {rl}. Answerers: {al}.
The first retriever shortlists passages; after each retriever, retrieval rules may escalate to a larger retriever (the next one, or 'to' a named one, or 'skip' n sizes); shortlists are ensembled; at most {rb['params'].get('max_ret_members', 3)} retrievers per question.
The first answerer answers; after each answer, answer rules may escalate to a larger answerer; if the new answer agrees with an earlier one it is accepted; otherwise the rules are checked again, and when they stop, the answers plus {rb['params']['votes']} sampled answers of the last answerer go to a majority vote.
Rules are checked in order (LOCKED first); the first rule whose conditions all hold decides; default = keep. A rule may have "at": [rungs] where it applies.

SIGNALS (name (stage): meaning):
{sig}

CURRENT RULEBOOK (settings: {json.dumps(rb['params'])}):
""" + "\n".join("  " + rule_text(r) for r in rb["rules"]) + f"""

HOW CHANGES ARE JUDGED: on two separate halves of the val questions, a change must be no less accurate and cheaper, or more accurate at no more than {int(100*H['H8_max_compute_per_point'])}% extra compute per accuracy point; it must also beat escalating the same number of random questions. While the rulebook is below the accuracy floor (the best fixed pipeline costing no more than it, minus {100*H['H4_max_acc_drop_vs_best_fixed']:.0f} point), any accuracy gain counts.
LOCKED LIMITS: at most {int(100*H['H3_max_share_top_answerer'])}% of questions may reach the largest answerer and {int(100*H['H3b_max_share_top_retriever'])}% the largest retriever; at most {H['H6_max_soft_rules']:.0f} soft rules; each escalate rule must decide on at least {100*H['H6_min_support_share']:.0f}% of val questions; LOCKED rules cannot be changed or removed; thresholds move at most {H['H7_max_step']} per round (percentile for *_pct signals, raw units for 0-1 signals, 1 for whole-number signals).

EVIDENCE:
{evidence(E, rb, rows)}

EARLIER PROPOSALS AND OUTCOMES:
{chr(10).join(history[-12:]) or '  (none yet)'}

TASK. Propose up to 3 changes that should raise accuracy or cut compute. Escalate where a larger model is clearly more often right; keep where they are close; skip sizes when the middle model adds little. Answer with ONLY a JSON list, each item one of:
{{"action":"add","position":"first"|"before_keep"|"last","rule":{{"stage":"retrieval"|"answer","at":[0],"when":[["signal","<",0.5]],"then":"escalate"|"keep","to":"<model key, optional>","skip":0,"text":"plain-English rule"}},"why":"..."}}
{{"action":"amend","id":"A1","rule":{{...same fields...}},"why":"..."}}
{{"action":"remove","id":"A3","why":"..."}}
{{"action":"param","name":"votes"|"max_ret_members"|"sure_threshold","value":...,"why":"..."}}
Position: "before_keep" (default) = before the stage's first keep rule. Operators: < <= > >= ==. Values are numbers."""


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


def mock_proposals(E, rnd):
    s0 = "ret_probe_pct" if "ret_probe_pct" in E.allowed and E.W.get("signals") != "all" else "ret_conf_pct"
    a0 = "ans_probe_pct" if "ans_probe_pct" in E.allowed and E.W.get("signals") != "all" else "ans_conf_pct"
    P = [[{"action": "add", "rule": {"stage": "answer", "when": [[a0, "<", 0.15]], "then": "escalate", "skip": 1}, "why": "mock"},
          {"action": "param", "name": "votes", "value": 3, "why": "mock"}],
         [{"action": "add", "rule": {"stage": "retrieval", "at": [0], "when": [[s0, "<", 0.3]], "then": "escalate"}, "why": "mock"},
          {"action": "amend", "id": "H1", "rule": {"stage": "retrieval", "when": [[s0, "<", 0.5]], "then": "escalate"}, "why": "mock: must be refused"}],
         [{"action": "param", "name": "sure_threshold", "value": 0.8, "why": "mock: must be refused (H5)"}],
         [{"action": "remove", "id": "A1", "why": "mock"}]]
    return P[(rnd - 1) % len(P)]


# ==================================================================================
def seed_rulebook(cfg, ws):
    W = cfg["workstreams"][ws]
    return {"version": 0, "params": dict(cfg["rules"]["params"], max_ret_members=int(cfg["rules"]["max_ret_members"])),
            "rules": [dict(r, when=[list(c) for c in r["when"]]) for r in W["seed_rules"]]}


def fixed_dev(E):
    """{(retriever rung, answerer rung): (val accuracy, mean units per question)} of every fixed pipeline."""
    out = {}
    for i in range(len(E.RL)):
        for j in range(len(E.AL)):
            rows = E.fixed("val", i, j)
            s = summ(rows)
            out[(i, j)] = (s["acc"], s["units"] / max(1, len(rows)))
    return out


def rdir(cfg, ws, *p):
    d = os.path.join(cfg["run_dir"], ws, *p)
    os.makedirs(d, exist_ok=True)
    return d


def learn(cfg, ws, planner=None):
    E = Env(cfg, ws)
    out = rdir(cfg, ws, "rules")
    os.makedirs(os.path.join(out, "rulebooks"), exist_ok=True)
    fx = fixed_dev(E)
    bi, bj = max(fx, key=lambda k: fx[k][0])
    log(f"{ws}: most accurate fixed pipeline on val {100*fx[(bi, bj)][0]:.1f}% ({E.names[E.RL[bi]]} -> {E.names[E.AL[bj]]}); "
        "H4 compares the rulebook with the best fixed pipeline at no higher cost")
    state_p = os.path.join(out, "learn_state.json")
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"round": 0, "rulebook": seed_rulebook(cfg, ws), "history": [], "log": []}
    rb = st["rulebook"]
    write_json(os.path.join(out, "rulebooks", "rulebook_v0_seed.json"), seed_rulebook(cfg, ws))
    R = cfg["rules"]
    tr = Tracker(cfg, "learn", ws, R["rounds"], "rounds")
    tr.tick(st["round"])
    own = False
    while st["round"] < R["rounds"]:
        rnd = st["round"] + 1
        base_rb = copy.deepcopy(rb)
        cur = {h: E.evaluate("val", qs, rb) for h, qs in (("A", E.halfA), ("B", E.halfB))}
        cur_dev = cur["A"] + cur["B"]
        s = summ(cur_dev)
        log(f"{ws} round {rnd}: val accuracy {100*s['acc']:.1f}%, retrieval escalated {100*s['ret_esc']:.0f}%, answer escalated {100*s['ans_esc']:.0f}%")
        cands = [("code", d, c) for d, c in threshold_steps(rb)]
        if cfg["backend"] == "mock":
            ptxt = planner_prompt(E, rb, cur["A"], st["history"])      # built anyway, so the dry run checks it
            open(os.path.join(out, f"planner_round{rnd}.txt"), "w", encoding="utf-8").write(ptxt)
            props = mock_proposals(E, rnd)
        else:
            if planner is None:
                planner, own = make_engine(cfg, serve_entry(variant(cfg, cfg["planner"])), max_model_len=24576), True
            ptxt = planner_prompt(E, rb, cur["A"], st["history"])
            raw = planner.generate([planner.encode(PLANNER_SYS, ptxt)], 1, 0.7, 2500, [seed_of(ws, "r", rnd)], [{"kind": "plan"}])[0][0][0]
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
                res = {h: E.evaluate("val", qs, c) for h, qs in (("A", E.halfA), ("B", E.halfB))}
                sc = summ(cur_dev)
                repair = sc["acc"] < floor_of(fx, sc["units"] / len(cur_dev)) - E.H["H4_max_acc_drop_vs_best_fixed"]
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
                g = guardrails(E, c, dev_rows, fx, summ(cur_dev)["acc"])
                if g:
                    tried.add(d)
                    st["log"].append(dict(rec, decision="rejected", reason="; ".join(g)))
                    if src == "planner":
                        st["history"].append(f"  round {rnd}: {d} -> rejected ({'; '.join(g)})")
                    continue
                changed = any(a["ret_path"] != b["ret_path"] or a["ans_path"] != b["ans_path"] for a, b in zip(dev_rows, cur_dev))
                ctrl = random_control(E, E.halfA + E.halfB, cur_dev, dev_rows, rb, R.get("random_control_draws", 10), rnd) if changed else -1.0
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
        tr.tick(rnd)
        log(f"{ws} round {rnd} done: val accuracy {100*summ(cur_dev)['acc']:.1f}%")
    write_json(os.path.join(out, "rulebooks", "rulebook_final.json"), rb)
    tr.finish()
    if own:
        free_gpu(planner)
    mark_done(cfg, "learn", ws)
    log(f"{ws}: learning complete")


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


def test(cfg, ws):
    """The frozen rulebook once on the test questions -> <ws>/results.json and <ws>/summary.md."""
    E = Env(cfg, ws)
    out = rdir(cfg, ws, "rules")
    seed, final = seed_rulebook(cfg, ws), json.load(open(os.path.join(out, "rulebooks", "rulebook_final.json")))
    st = json.load(open(os.path.join(out, "learn_state.json")))
    T = E.qids["test"]
    nm = E.names
    fx = fixed_dev(E)
    bi, bj = max(fx, key=lambda k: fx[k][0])
    pols = []

    def add(name, rows, kind, detail=None):
        pols.append({"policy": name, "kind": kind, "rows": rows, "detail": detail})

    add(f"Fixed: {nm[E.RL[0]]} -> {nm[E.AL[0]]} (cheapest)", E.fixed("test", 0, 0, T), "fixed")
    add(f"Fixed: {nm[E.RL[-1]]} -> {nm[E.AL[-1]]} (largest)", E.fixed("test", len(E.RL) - 1, len(E.AL) - 1, T), "fixed")
    add(f"Fixed: {nm[E.RL[bi]]} -> {nm[E.AL[bj]]} (best fixed pipeline on val)", E.fixed("test", bi, bj, T), "fixed")
    for name, rb in (("Starting rulebook (before any micro-experiment)", seed), (f"LEARNT rulebook (v{final['version']}, {st['round']} rounds)", final)):
        rows = E.evaluate("test", T, rb)
        dev_rows = E.evaluate("val", E.qids["val"], rb)
        sure = _fit_lr([r["sure_feats"] for r in dev_rows], [r["acc"] for r in dev_rows])
        thr = max(E.H["H5_min_sure_score"], rb["params"]["sure_threshold"])
        for r, sc in zip(rows, sure([r["sure_feats"] for r in rows])):
            r["sure_score"], r["flag"] = float(sc), ("Sure" if sc >= thr else "Unsure")
        rng = np.random.default_rng(3)
        ctrl = []
        for _ in range(cfg["rules"].get("random_control_draws", 10) * 2):
            perm = rng.permutation(len(rows))
            forces = {q: {"ret_path": rows[p]["ret_path"], "ans_path": rows[p]["ans_path"]} for q, p in zip(T, perm)}
            ctrl.append([r["acc"] for r in E.evaluate("test", T, rb, forces)])
        add(name, rows, "rulebook", {"control": np.mean(ctrl, 0), "threshold": thr, "rulebook": rb})
    # oracle: cheapest answerer if right, else the largest, on the first retriever's shortlist
    orc = []
    for q in T:
        r = E.R["test"][E.RL[0]][q]
        s = key(r["sel"])
        a1, a2 = E.A[E.AL[0]][(q, s)], E.A[E.AL[-1]][(q, s)]
        orc.append({"qid": q, "acc": max(a1["acc"], a2["acc"]), "lenient": max(a1["lenient"], a2["lenient"]), "EM": max(a1["EM"], a2["EM"]),
                    "F1": max(a1.get("F1", 0), a2.get("F1", 0)), "answer": (a1 if a1["acc"] else a2).get("prediction"),
                    "units": r["units"] + a1["units"] + (0 if a1["acc"] else a2["units"]), "units_retrieve": r["units"],
                    "units_answer": a1["units"] + (0 if a1["acc"] else a2["units"]), "models": ""})
    add(f"Oracle: {nm[E.AL[0]]} if right, else {nm[E.AL[-1]]} (needs the answer key)", orc, "oracle")
    res = []
    ref = np.array([r["acc"] for r in pols[2]["rows"]])
    for p in pols:
        a = np.array([r["acc"] for r in p["rows"]])
        row = {"policy": p["policy"], "kind": p["kind"], "acc": float(a.mean()), "lenient": float(np.mean([r["lenient"] for r in p["rows"]])),
               "EM": float(np.mean([r["EM"] for r in p["rows"]])), "units_per_q": float(np.mean([r["units"] for r in p["rows"]])), "n": len(a)}
        if len(a) == len(ref):
            row["vs_best_fixed_pts"], row["ci_best_fixed"] = 100 * float(a.mean() - ref.mean()), _boot(a, ref)
        if p["detail"]:
            rr, ctrl = p["rows"], p["detail"]["control"]
            sure = np.array([r["flag"] == "Sure" for r in rr])
            row.update({"vs_random_pts": 100 * float(a.mean() - ctrl.mean()), "ci_random": _boot(a, ctrl),
                        "ret_escalated": float(np.mean([r["ret_esc"] for r in rr])), "ans_escalated": float(np.mean([r["ans_esc"] for r in rr])),
                        "reach_top_retriever": float(np.mean([r["top_ret"] for r in rr])), "reach_top_answerer": float(np.mean([r["top_ans"] for r in rr])),
                        "sure_share": float(sure.mean()), "right_when_sure": float(a[sure].mean()) if sure.any() else None,
                        "right_when_unsure": float(a[~sure].mean()) if (~sure).any() else None})
        res.append(row)
    lrows = pols[4]["rows"]
    rules_t = [{"rule": rule_text(r), "text": r.get("text", ""), "decided_on_test": sum(r["id"] in x["fired"] for x in lrows),
                "acc_there": (float(np.mean([x["acc"] for x in lrows if r["id"] in x["fired"]])) if any(r["id"] in x["fired"] for x in lrows) else None)}
               for r in final["rules"]]
    per_q = [{"qid": x["qid"], "question": E.qd[x["qid"]]["question"], "gold": E.qd[x["qid"]]["answer"], "rules (in order)": " | ".join(x["fired"]),
              "retrievers": "+".join(E.RL[i] for i in x["ret_path"]), "answerers": "+".join(E.AL[i] for i in x["ans_path"]),
              "final from": x["final_model"], "answer": x["answer"], "sure score": round(x["sure_score"], 3), "flag": x["flag"], "right": x["acc"],
              "AgentTTS units": round(x["units"], 3)} for x in lrows]
    R = rdir(cfg, ws)
    # every policy's per-question rows, for the team results table (export stage): all fixed pairs, both rulebooks, the oracle
    slug = {"fixed": None, "rulebook": None, "oracle": "oracle-answer-key"}
    per_policy = {}
    for i in range(len(E.RL)):
        for j in range(len(E.AL)):
            per_policy[f"fixed:{E.RL[i]}->{E.AL[j]}"] = E.fixed("test", i, j, T)
    for p in pols:
        if p["kind"] == "rulebook":
            per_policy["rulebook-learnt" if p["policy"].startswith("LEARNT") else "rulebook-seed"] = p["rows"]
        elif p["kind"] == "oracle":
            per_policy[slug["oracle"]] = p["rows"]
    write_json(os.path.join(R, "policy_rows.json"), per_policy)
    write_json(os.path.join(R, "results.json"), {"workstream": ws, "name": cfg["workstreams"][ws]["name"], "policies": res, "rules": rules_t,
                                                  "accepted": [x for x in st["log"] if x["decision"] == "ACCEPTED"], "tried": len(st["log"]),
                                                  "final_rulebook": final, "n_dev": len(E.qids["val"]), "n_test": len(T)})
    write_jsonl(os.path.join(R, "per_question.jsonl"), per_q)
    write_jsonl(os.path.join(R, "micro_experiments.jsonl"), st["log"])
    _xlsx(R, res, rules_t, st["log"], per_q)
    mark_done(cfg, "test", ws)
    log(f"{ws}: test results -> {R}")


def _xlsx(R, res, rules_t, mlog, per_q):
    try:
        import pandas as pd
        with pd.ExcelWriter(os.path.join(R, "results.xlsx"), engine="openpyxl") as xw:
            for sheet, rr in (("Test policies", res), ("Learnt rules on test", rules_t), ("Micro-experiment log", mlog), ("Per question", per_q)):
                d = pd.DataFrame(rr)
                for c in d.columns:
                    d[c] = d[c].map(lambda v: json.dumps(v) if isinstance(v, (list, tuple, dict)) else v)
                    if d[c].isna().any():
                        d[c] = d[c].astype(object).where(d[c].notna(), "-")
                d.to_excel(xw, sheet_name=sheet, index=False)
    except Exception as e:  # noqa: BLE001
        log(f"results.xlsx not written: {e}")
