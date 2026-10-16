"""Set 10 - a self-learnt rulebook for model switching (notebook 28).

The models are the fine-tuned retrievers (Qwen 0.5B route model, 0.6B, 0.8B, 3B) and the answerers (Llama 8B, Qwen 32B), with all
their outputs already saved by notebooks 26-27. Switching is governed by a RULEBOOK:
  * soft rules  - IF <signal conditions> THEN escalate / keep; tuned on the dev questions by micro-experiments;
  * hard rules  - H1, H2 (always escalate in clear cases) and the guardrails H3-H7, which no experiment may change.
Dev questions = notebook 27's 1,000 calibration questions (split into halves A and B); test = the 400 test questions, used once.

Each round of micro-experiments:
  1. code tries fine threshold steps (+-0.025, +-0.05 in percentile of the dev distribution) for every soft rule;
  2. a planner LLM (Qwen 32B) reads the rulebook and evidence tables from half A and proposes up to 3 new rules / amendments /
     removals / setting changes;
  3. a change is accepted only if on BOTH halves it is cheaper at no accuracy loss or more accurate at little extra compute,
     it beats a random-escalation control of the same size, and it breaks no hard guardrail.

Commands: learn (the rounds; planner on GPU or a mock), test (frozen rulebook on the 400 test questions; CPU).
"""
import argparse
import copy
import json
import os
import re
import sys
import zlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ssw as S               # noqa: E402
import sweep as W             # noqa: E402
import casc8 as C8            # noqa: E402
import route9 as R9           # noqa: E402

VERSION = "10.0"
log = S.log
LADDER = ["r05", "r06", "r08", "r3"]
RNAME = {"r05": "0.5B", "r06": "0.6B", "r08": "0.8B", "r3": "3B"}
NEXT = {"r06": "r08", "r08": "r3", "r3": None}
_CANNOT = re.compile(r"(unknown|cannot|can't|not (provided|mentioned|specified|available|enough|stated|given)|no (information|answer|mention)|insufficient|unclear)")

HARD = {   # locked: the loop may never change these
    "H3_max_share_qwen": 0.60,
    "H4_max_acc_drop_vs_best_fixed": 0.01,
    "H5_min_sure_score": 0.85,
    "H6_max_soft_rules": 8,
    "H6_min_support": 20,
    "H7_max_step_pct": 0.05,
    "H8_max_compute_per_point": 0.10,     # an accuracy gain may cost at most 10% extra compute per point
}
SIGNALS = {   # name -> (stage, description) ; *_pct = percentile rank among the dev questions (0 = least confident)
    "r05_conf_pct": ("retrieval", "0.5B retriever's confidence in its own shortlist (token probability), percentile among dev questions"),
    "route_p05": ("retrieval", "learned route's probability that the 0.5B shortlist suffices (0-1)"),
    "ret_agree": ("retrieval2", "after escalating: 1 if the bigger retriever picked the same passages as the 0.5B, else 0"),
    "target_conf_pct": ("retrieval2", "after escalating: the bigger retriever's confidence, percentile among dev questions"),
    "llama_conf_pct": ("answer", "Llama 8B's token probability for its answer, percentile among dev questions"),
    "llama_stated": ("answer", "Llama 8B's stated confidence (0-1)"),
    "words_in_passages": ("answer", "share of the answer's words found in the shortlisted passages (0-1)"),
    "is_yes_no": ("answer", "1 if the answer is yes/no"),
    "is_cannot": ("answer", "1 if the answer is empty or says it cannot be determined"),
    "answer_words": ("answer", "number of words in the answer"),
    "is_comparison": ("answer", "1 if HotpotQA marks the question as a comparison question"),
    "ret_escalated": ("answer", "1 if retrieval was escalated for this question"),
}
OPS = {"<": np.less, "<=": np.less_equal, ">": np.greater, ">=": np.greater_equal, "==": np.equal}


def seed_rulebook():
    R = lambda i, st, when, then, text, hard=False, to=None: {"id": i, "stage": st, "when": when, "then": then, "text": text, "hard": hard,
                                                              **({"to": to} if to else {})}
    return {"version": 0, "params": {"ret_target": "r08", "ret_merge": "union", "votes": 2, "sure_threshold": 0.90},
            "rules": [
                R("H1", "retrieval", [["r05_conf_pct", "<", 0.10]], "escalate", "IF the 0.5B's confidence is in its lowest 10% THEN escalate retrieval (locked)", True),
                R("R1", "retrieval", [["r05_conf_pct", "<", 0.20]], "escalate", "IF the 0.5B's confidence is in its lowest 20% THEN escalate retrieval to the 0.8B (0.6B skipped) and ensemble", to="r08"),
                R("R2", "retrieval2", [["ret_agree", "==", 0], ["target_conf_pct", "<", 0.20]], "escalate",
                  "IF after escalating the shortlists disagree AND the bigger retriever is also unsure THEN escalate again and ensemble"),
                R("H2", "answer", [["is_cannot", "==", 1]], "escalate", "IF the answer is empty or says it cannot be determined THEN escalate (locked)", True),
                R("A2", "answer", [["is_yes_no", "==", 1]], "keep", "IF the answer is yes/no THEN keep it"),
                R("A1", "answer", [["is_yes_no", "==", 0], ["words_in_passages", "<", 0.97]], "escalate",
                  "IF the answer is not yes/no AND some of its words are not in the passages THEN escalate: Qwen 32B answers; agree -> accept, else Qwen samples join a vote"),
                R("A3", "answer", [["words_in_passages", ">=", 0.97], ["llama_stated", ">", 0.95]], "keep", "IF all its words are in the passages AND stated confidence > 95% THEN keep (Sure)"),
                R("A4", "answer", [["words_in_passages", ">=", 0.97], ["llama_stated", "<=", 0.95]], "keep",
                  "IF all its words are in the passages AND stated confidence <= 95% THEN keep (borderline)"),
            ]}


def rule_text(r):
    c = " AND ".join(f"{s} {o} {v:g}" if isinstance(v, (int, float)) else f"{s} {o} {v}" for s, o, v in r["when"])
    to = f" (to {RNAME.get(r.get('to'), r.get('to'))})" if r.get("to") else ""
    return f"[{r['id']}{', LOCKED' if r['hard'] else ''}] {r['stage']}: IF {c} THEN {r['then']}{to}"


# ==================================================================================
# the world: saved outputs of notebooks 26-27
# ==================================================================================
class Env:
    def __init__(self, args):
        self.args = args
        ns = argparse.Namespace(old_data=args.old_data, fresh_data=args.fresh_data, run_dir=args.route_run, label_dir=args.label_dir,
                                models_json=os.path.join(args.route_run, "models.json"), old_sweep=args.old_sweep, fresh_run=args.fresh_run)
        self.ns = ns
        self.A = R9.load_answers(ns)
        self.SM = {(r["qid"], R9.key(r["sel"])): r for r in S.read_jsonl(os.path.join(args.route_run, "step", "samples_q32.jsonl"))}
        self.dev = R9.World(ns, "calib", self.A, self.SM)
        self.test = R9.World(ns, "test", self.A, self.SM)
        # dev distributions for percentile signals
        d = self.dev
        self.dist = {
            "r05_conf": np.sort([np.exp(R9.lp_(d.route[q]["mean_logprob"])) for q in d.qids]),
            "target_conf": np.sort([np.exp(R9.lp_(d.ret[r][q]["mean_logprob"])) for r in ("r06", "r08", "r3") for q in d.qids]),
            "llama_conf": np.sort([np.exp(R9.lp_(self.A["l8"][(q, R9.key(d.route[q]["sel"]))].get("mean_logprob"))) for q in d.qids]),
        }
        self._sig = {}
        memo, base_rs = {}, R9.ret_state

        def ret_state_m(Wx, q, m):
            k = (id(Wx), q, tuple(m))
            if k not in memo:
                memo[k] = base_rs(Wx, q, m)
            return memo[k]
        R9.ret_state = ret_state_m
        rng = np.random.default_rng(11)
        perm = rng.permutation(len(d.qids))
        self.halfA = sorted(d.qids[i] for i in perm[: len(perm) // 2])
        self.halfB = sorted(d.qids[i] for i in perm[len(perm) // 2:])

    def pct(self, kind, raw):
        a = self.dist[kind]
        return float(np.searchsorted(a, raw, side="right") / len(a))

    def answer_signals(self, Wd, q, sel):
        k = (id(Wd), q, sel)
        if k in self._sig:
            return self._sig[k]
        a = self.A["l8"][(q, sel)]
        qq = Wd.qd[q]
        pred = a.get("prediction") or ""
        p = S.normalize_answer(pred)
        psg = S.normalize_answer(" ".join(qq["ctxs_candidate"][i - 1] for i in sel if 1 <= i <= len(qq["ctxs_candidate"]))).split()
        ps = set(psg)
        toks = p.split()
        s = {"llama_conf_pct": self.pct("llama_conf", np.exp(R9.lp_(a.get("mean_logprob")))),
             "llama_stated": float(a.get("stated_conf") if a.get("stated_conf") is not None else 0.5),
             "words_in_passages": float(np.mean([t in ps for t in toks])) if toks else 0.0,
             "is_yes_no": float(p in ("yes", "no")), "is_cannot": float((not p) or bool(_CANNOT.search(pred.lower()))),
             "answer_words": float(len(toks)), "is_comparison": float(qq.get("type") == "comparison")}
        self._sig[k] = s
        return s


def match(rule, sig):
    for s, o, v in rule["when"]:
        if s not in sig or not OPS[o](sig[s], v):
            return False
    return True


def decide(rules, stage, sig):
    """Locked rules first, then soft rules in order; the first rule whose conditions hold decides. Default: keep."""
    rs = [r for r in rules if r["stage"] == stage]
    for r in [r for r in rs if r["hard"]] + [r for r in rs if not r["hard"]]:
        if match(r, sig):
            return r
    return None


def run_q(E, Wd, q, rb, force=None):
    """One question through the rulebook. force = {"ret": bool, "ans": bool} overrides the decisions (random controls)."""
    P = rb["params"]
    rules = rb["rules"]
    s05, lp05, _ = Wd.member(q, "r05")
    flops = Wd.route[q]["flops"]
    rsig = {"r05_conf_pct": E.pct("r05_conf", np.exp(R9.lp_(lp05))), "route_p05": Wd.route[q]["p_ret"]["r05"]}
    r1 = decide(rules, "retrieval", rsig)
    esc_r = (r1 is not None and r1["then"] == "escalate") if not force else force["ret"]
    fired = [r1["id"] if r1 else "-"]
    sel, members = s05, ["r05"]
    if esc_r:
        tgt = (r1.get("to") if (r1 and not force) else None) or P["ret_target"]
        members = ["r05", tgt]
        if P["ret_merge"] == "target":
            sel, _, f = Wd.member(q, tgt)
            flops += f
        else:
            rs = R9.ret_state(Wd, q, members)
            sel = rs["sel"]
            flops += rs["flops"]
        st, lpt, _ = Wd.member(q, tgt)
        sig2 = dict(rsig, ret_agree=float(st == s05), target_conf_pct=E.pct("target_conf", np.exp(R9.lp_(lpt))))
        r2 = decide(rules, "retrieval2", sig2)
        fired.append(r2["id"] if r2 else "-")
        if r2 is not None and r2["then"] == "escalate" and NEXT[tgt]:
            members = ["r05", tgt, NEXT[tgt]]
            rs = R9.ret_state(Wd, q, members)
            sel = rs["sel"]
            flops = Wd.route[q]["flops"] + rs["flops"]
    else:
        fired.append("-")
    aL = dict(E.A["l8"][(q, sel)], model="l8")
    asig = dict(E.answer_signals(Wd, q, sel), ret_escalated=float(esc_r), **rsig)
    r3 = decide(rules, "answer", asig)
    esc_a = (r3 is not None and r3["then"] == "escalate") if not force else force["ans"]
    fired.append(r3["id"] if r3 else "-")
    flops += aL["flops"]
    final, votes, agreeLQ, support = aL, ["l8"], -1.0, 1.0
    if esc_a:
        aQ = dict(E.A["q32"][(q, sel)], model="q32")
        flops += aQ["flops"]
        agreeLQ = float(C8.agree(aL["prediction"], aQ["prediction"]))
        if agreeLQ:
            final, votes, support = aQ, ["l8", "q32"], 1.0
        else:
            sm = E.SM[(q, sel)]
            n = int(P["votes"])
            ss = [dict(s, model="q32s") for s in sm["samples"][:n]]
            flops += sm["flops"] / max(1, len(sm["samples"])) * len(ss)
            vv = [aQ, aL] + ss
            final, sup = C8.vote(vv)
            votes, support = [v["model"] for v in vv], sup / len(vv)
    return {"qid": q, "lenient": float(final["lenient"]), "EM": float(final["EM"]), "flops": float(flops), "ret_esc": float(esc_r), "ans_esc": float(esc_a),
            "members": "+".join(RNAME[m] for m in members), "fired": fired, "answer": final.get("prediction"), "votes": "+".join(votes),
            "sure_feats": [asig["llama_conf_pct"], asig["llama_stated"], asig["words_in_passages"], asig["is_yes_no"], asig["is_cannot"], float(esc_a),
                           agreeLQ, support, rsig["r05_conf_pct"], float(esc_r), float(sel == s05)]}


def evaluate(E, Wd, qids, rb, forces=None):
    return [run_q(E, Wd, q, rb, (forces or {}).get(q)) for q in qids]


def summ(rows):
    return {"acc": float(np.mean([r["lenient"] for r in rows])), "em": float(np.mean([r["EM"] for r in rows])),
            "flops": float(np.sum([r["flops"] for r in rows])), "ret_esc": float(np.mean([r["ret_esc"] for r in rows])),
            "qwen": float(np.mean([r["ans_esc"] for r in rows]))}


# ==================================================================================
# acceptance: both halves, random control, guardrails
# ==================================================================================
def ok_half(c, b, repair=False):
    """Accept on this half if: no less accurate and cheaper; or more accurate at <= H8 extra compute per point.
    Repair mode (current rulebook below the H4 accuracy floor): any accuracy gain is accepted (H3 still caps the compute)."""
    dA, dC = c["acc"] - b["acc"], c["flops"] / b["flops"] - 1
    if repair:
        return dA > 0, dA, dC
    cheaper = dA >= 0 and dC <= -0.005
    better = dA > 0 and dC <= HARD["H8_max_compute_per_point"] * (100 * dA)
    return cheaper or better, dA, dC


def guardrails(E, rb, rows_dev, best_fixed_dev, cur_acc_dev):
    s = summ(rows_dev)
    soft = [r for r in rb["rules"] if not r["hard"]]
    why = []
    if s["qwen"] > HARD["H3_max_share_qwen"] + 1e-9:
        why.append(f"H3: {100*s['qwen']:.0f}% reach Qwen 32B (max {100*HARD['H3_max_share_qwen']:.0f}%)")
    if s["acc"] < best_fixed_dev - HARD["H4_max_acc_drop_vs_best_fixed"] and s["acc"] < cur_acc_dev:
        why.append(f"H4: dev accuracy {100*s['acc']:.1f}% below the best fixed pipeline ({100*best_fixed_dev:.1f}%) minus 1 point")
    if len(soft) > HARD["H6_max_soft_rules"]:
        why.append(f"H6: {len(soft)} soft rules (max {HARD['H6_max_soft_rules']})")
    cnt = {}
    for r in rows_dev:
        for f in r["fired"]:
            cnt[f] = cnt.get(f, 0) + 1
    for r in soft:
        if r["then"] == "escalate" and cnt.get(r["id"], 0) < HARD["H6_min_support"]:
            why.append(f"H6: rule {r['id']} fires on {cnt.get(r['id'], 0)} dev questions (min {HARD['H6_min_support']})")
    return why


def random_control(E, Wd, qids, cur_rows, cand_rows, cur_rb, draws=20, seed=0):
    """The current rulebook, with the same NUMBER of escalation changes as the candidate made, applied to random questions."""
    rng = np.random.default_rng(seed)
    out = []
    by = {r["qid"]: r for r in cur_rows}
    cand = {r["qid"]: r for r in cand_rows}
    for _ in range(draws):
        forces = {q: {"ret": bool(by[q]["ret_esc"]), "ans": bool(by[q]["ans_esc"])} for q in qids}
        for st in ("ret", "ans"):
            k = st + "_esc"
            plus = sum(1 for q in qids if cand[q][k] > by[q][k])
            minus = sum(1 for q in qids if cand[q][k] < by[q][k])
            pool_p = [q for q in qids if not by[q][k]]
            pool_m = [q for q in qids if by[q][k]]
            for q in rng.choice(pool_p, min(plus, len(pool_p)), replace=False) if plus else []:
                forces[q][st] = True
            for q in rng.choice(pool_m, min(minus, len(pool_m)), replace=False) if minus else []:
                forces[q][st] = False
        out.append(summ(evaluate(E, Wd, qids, cur_rb, forces))["acc"])
    return float(np.mean(out))


# ==================================================================================
# changes: fine threshold steps (code) and planner proposals
# ==================================================================================
def step_unit(sig):
    """Threshold steps: percentile signals (*_pct) and 0-1 signals move in units of 0.025 / 0.05; answer_words in whole words."""
    return 1.0 if sig == "answer_words" else 0.025


def moved(sig, v, v0):
    """How far a threshold moved, in H7 units (0.05 = the per-round limit)."""
    return abs(v - v0) * (0.05 if sig == "answer_words" else 1.0)


def threshold_steps(E, rb):
    out = []
    for i, r in enumerate(rb["rules"]):
        if r["hard"]:
            continue
        for j, (s, o, v) in enumerate(r["when"]):
            if o == "==" or s in ("is_yes_no", "is_cannot", "is_comparison", "ret_agree", "ret_escalated"):
                continue
            for k in (-2, -1, 1, 2):
                nv = v + k * step_unit(s)
                lo, hi = (0.0, 1.0) if s != "answer_words" else (0.0, 50.0)
                if not (lo <= nv <= hi) or abs(nv - v) < 1e-12:
                    continue
                dp = k * step_unit(s)
                c = copy.deepcopy(rb)
                c["rules"][i]["when"][j][2] = round(nv, 4)
                c["rules"][i]["text"] = r["text"]
                out.append((f"threshold: {r['id']} {s} {o} {v:g} -> {nv:.4g}", c))
    return out


def valid_rule(r, errs):
    ok = isinstance(r, dict) and r.get("stage") in ("retrieval", "retrieval2", "answer") and r.get("then") in ("escalate", "keep") and isinstance(r.get("when"), list) and r["when"]
    if not ok:
        errs.append("malformed rule")
        return False
    for c in r["when"]:
        if not (isinstance(c, (list, tuple)) and len(c) == 3 and c[0] in SIGNALS and c[1] in OPS and isinstance(c[2], (int, float))):
            errs.append(f"bad condition {c}")
            return False
        if SIGNALS[c[0]][0] != r["stage"] and not (r["stage"] in ("answer", "retrieval2") and SIGNALS[c[0]][0] == "retrieval"):
            errs.append(f"signal {c[0]} is not available at stage {r['stage']}")
            return False
    if r.get("to") and r["to"] not in ("r06", "r08", "r3"):
        errs.append("bad target")
        return False
    return True


def apply_proposal(rb, p):
    """Returns (candidate rulebook, description) or (None, reason)."""
    c = copy.deepcopy(rb)
    act = p.get("action")
    errs = []
    if act == "param":
        name, val = p.get("name"), p.get("value")
        allowed = {"ret_target": ("r06", "r08", "r3"), "ret_merge": ("union", "target"), "votes": (1, 2, 3, 4)}
        if name == "sure_threshold":
            if not isinstance(val, (int, float)) or val < HARD["H5_min_sure_score"] or val > 0.99:
                return None, f"sure_threshold must be between {HARD['H5_min_sure_score']} and 0.99 (H5)"
        elif name not in allowed or val not in allowed[name]:
            return None, f"setting {name}={val} not allowed"
        c["params"][name] = val
        return c, f"setting {name}: {rb['params'].get(name)} -> {val}"
    if act in ("amend", "remove"):
        rid = p.get("id")
        idx = [i for i, r in enumerate(c["rules"]) if r["id"] == rid]
        if not idx:
            return None, f"no rule {rid}"
        if c["rules"][idx[0]]["hard"]:
            return None, f"{rid} is locked (hard rule)"
        if act == "remove":
            del c["rules"][idx[0]]
            return c, f"remove {rid}"
        nr = dict(p.get("rule") or {})
        nr.update({"id": rid, "hard": False})
        if not valid_rule(nr, errs):
            return None, "; ".join(errs)
        nr["text"] = nr.get("text") or rule_text(nr)
        c["rules"][idx[0]] = nr
        return c, f"amend {rid}: {rule_text(nr)}"
    if act == "add":
        nr = dict(p.get("rule") or {})
        nr["hard"] = False
        nr["id"] = nr.get("id") if nr.get("id") and nr.get("id") not in [r["id"] for r in c["rules"]] and not str(nr.get("id")).startswith("H") else f"N{len(c['rules']) + 1}"
        if not valid_rule(nr, errs):
            return None, "; ".join(errs)
        nr["text"] = nr.get("text") or rule_text(nr)
        pos = p.get("position")
        same = [i for i, r in enumerate(c["rules"]) if r["stage"] == nr["stage"] and not r["hard"]]
        keeps = [i for i in same if c["rules"][i]["then"] == "keep"]
        if pos == "first" and same:
            at = same[0]
        elif pos == "last" or not same:
            at = same[-1] + 1 if same else len(c["rules"])
        else:                      # default: before the stage's first 'keep' rule, so the new rule can fire
            at = keeps[0] if keeps else same[-1] + 1
        c["rules"].insert(at, nr)
        return c, f"add {rule_text(nr)}"
    return None, f"unknown action {act}"


def h7_ok(E, base_rb, cand_rb):
    """No surviving soft rule's threshold may have moved more than H7 (percentile) since the start of the round."""
    base = {r["id"]: r for r in base_rb["rules"]}
    for r in cand_rb["rules"]:
        b = base.get(r["id"])
        if not b or r["hard"] or len(b["when"]) != len(r["when"]):
            continue
        for (s, o, v), (s0, o0, v0) in zip(r["when"], b["when"]):
            if s == s0 and o == o0 and isinstance(v, (int, float)) and moved(s, v, v0) > HARD["H7_max_step_pct"] + 1e-9:
                return False, f"H7: {r['id']} {s} moved {moved(s, v, v0):.3f} this round (max {HARD['H7_max_step_pct']})"
    return True, ""


# ==================================================================================
# planner
# ==================================================================================
def bands(rows, vals, y1, y2, n=5):
    v = np.asarray(vals, float)
    edges = np.unique(np.quantile(v, np.linspace(0, 1, n + 1)))
    out = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (v >= lo) & (v <= hi if hi == edges[-1] else v < hi)
        if m.sum():
            out.append(f"{lo:.3g}-{hi:.3g}: n={int(m.sum())}, Llama right {100*np.mean(np.asarray(y1)[m]):.0f}%, Qwen right {100*np.mean(np.asarray(y2)[m]):.0f}%")
    return out


def evidence(E, rb, rows):
    """Tables from dev half A for the planner."""
    Wd = E.dev
    L = ["CURRENT RESULTS on dev half A: " + json.dumps({k: round(v, 4) for k, v in summ(rows).items() if k != "flops"})
         + f", compute {summ(rows)['flops'] / E.base_flops_A:.3f} x of '0.5B -> Qwen 32B always'"]
    cnt = {}
    for r in rows:
        for i, f in enumerate(r["fired"]):
            cnt.setdefault(f, []).append(r)
    L.append("PER RULE (questions on which it decided; accuracy there):")
    for r in rb["rules"]:
        rr = cnt.get(r["id"], [])
        L.append(f"  {rule_text(r)} -> fired {len(rr)}, final accuracy {100*np.mean([x['lenient'] for x in rr]) if rr else 0:.0f}%")
    keep = [r for r in rows if not r["ans_esc"]]
    if keep:
        sel = {}
        for r in keep:
            q = r["qid"]
            s05 = Wd.member(q, "r05")[0]
            sel[q] = s05
        sigs = {q: E.answer_signals(Wd, q, sel[q]) for q in sel}
        yL = [E.A["l8"][(q, sel[q])]["lenient"] for q in sel]
        yQ = [E.A["q32"][(q, sel[q])]["lenient"] for q in sel]
        L.append(f"ANSWER STAGE, questions currently NOT escalated (n={len(sel)}): Llama right {100*np.mean(yL):.0f}%, Qwen right {100*np.mean(yQ):.0f}% on the same passages. By signal band:")
        for s in ("llama_conf_pct", "llama_stated", "words_in_passages", "answer_words"):
            L.append(f"  {s}: " + " | ".join(bands(keep, [sigs[q][s] for q in sel], yL, yQ)))
        for s in ("is_yes_no", "is_comparison"):
            for val in (0.0, 1.0):
                m = [q for q in sel if sigs[q][s] == val]
                if m:
                    L.append(f"  {s}={int(val)}: n={len(m)}, Llama right {100*np.mean([E.A['l8'][(q, sel[q])]['lenient'] for q in m]):.0f}%, Qwen right {100*np.mean([E.A['q32'][(q, sel[q])]['lenient'] for q in m]):.0f}%")
    esc = [r for r in rows if r["ans_esc"]]
    if esc:
        L.append(f"ANSWER STAGE, questions currently escalated (n={len(esc)}): final accuracy {100*np.mean([r['lenient'] for r in esc]):.0f}%; "
                 f"Llama alone would have had {100*np.mean([E.A['l8'][(r['qid'], Wd.member(r['qid'], 'r05')[0])]['lenient'] for r in esc]):.0f}% on the 0.5B passages")
    qs = [r["qid"] for r in rows]
    v = [E.pct("r05_conf", np.exp(R9.lp_(Wd.member(q, "r05")[1]))) for q in qs]
    L.append("RETRIEVAL STAGE by r05_conf_pct band (share of questions whose shortlist holds both gold passages; Qwen 32B accuracy on that shortlist):")
    vv = np.array(v)
    for lo, hi in ((0, .1), (.1, .2), (.2, .3), (.3, .5), (.5, 1.01)):
        m = [q for q, x in zip(qs, vv) if lo <= x < hi]
        if not m:
            continue
        parts = []
        for r in LADDER:
            sel = [Wd.member(q, r)[0] for q in m]
            parts.append(f"{RNAME[r]} found {100*np.mean([Wd.gold(q) <= set(s) for q, s in zip(m, sel)]):.0f}% / Qwen {100*np.mean([E.A['q32'][(q, s)]['lenient'] for q, s in zip(m, sel)]):.0f}%")
        L.append(f"  {lo:.1f}-{min(hi, 1):.1f} (n={len(m)}): " + "; ".join(parts))
    L.append("COMPUTE per question (TFLOPs): 0.5B route pass 2.8, 0.6B 3.3, 0.8B 4.4, 3B 17, Llama 8B answer ~10.6, Qwen 32B answer ~34.6, each Qwen sample ~34.6")
    return "\n".join(L)


PLANNER_SYS = ("You improve a rulebook that decides, per question, when a small model's work is good enough and when to escalate to a bigger model. "
               "You propose small, testable changes; every change is tested on held-out questions and kept only if it helps.")


def planner_prompt(E, rb, rows, history):
    sig = "\n".join(f"  {k} ({v[0]}): {v[1]}" for k, v in SIGNALS.items())
    hist = "\n".join(history[-12:]) or "  (none yet)"
    return f"""THE PIPELINE. 0.5B retriever shortlists passages -> retrieval rules may escalate to a bigger retriever (0.6B, 0.8B or 3B; the shortlists are ensembled) -> retrieval2 rules may escalate once more -> Llama 8B answers -> answer rules may escalate: Qwen 32B answers; if it agrees with Llama that answer is kept, otherwise {rb['params']['votes']} sampled Qwen answers join a majority vote.
Rules are checked in order (LOCKED rules first); the first rule whose conditions all hold decides; if none holds, the default is keep.

SIGNALS (name (stage): meaning):
{sig}

CURRENT RULEBOOK (settings: {json.dumps(rb['params'])}):
""" + "\n".join("  " + rule_text(r) for r in rb["rules"]) + f"""

HOW CHANGES ARE JUDGED: on two separate halves of the dev questions, a change must be no less accurate and cheaper, or more accurate at no more than {int(100*HARD['H8_max_compute_per_point'])}% extra compute per accuracy point; it must also beat escalating the same number of random questions. While the rulebook is below the accuracy floor (best fixed pipeline minus 1 point), any accuracy gain counts.
LOCKED LIMITS you cannot change: at most {int(100*HARD['H3_max_share_qwen'])}% of questions may reach Qwen 32B; at most {HARD['H6_max_soft_rules']} soft rules; each escalate rule must fire on at least {HARD['H6_min_support']} of 1,000 dev questions; LOCKED rules (H...) cannot be changed or removed; thresholds move at most {HARD['H7_max_step_pct']} per round (percentile for *_pct signals, raw units for 0-1 signals, 1 word for answer_words).

EVIDENCE:
{evidence(E, rb, rows)}

EARLIER PROPOSALS AND OUTCOMES:
{hist}

TASK. Propose up to 3 changes that should raise accuracy or cut compute. Use the evidence: escalate where Qwen is clearly more often right than Llama, keep where they are close. Answer with ONLY a JSON list, each item one of:
{{"action":"add","position":"first"|"before_keep"|"last","rule":{{"stage":"answer","when":[["signal","<",0.5],...],"then":"escalate"|"keep","to":"r08"(retrieval only, optional),"text":"plain-English rule"}},"why":"..."}}
{{"action":"amend","id":"A1","rule":{{...same fields...}},"why":"..."}}
{{"action":"remove","id":"A4","why":"..."}}
{{"action":"param","name":"votes"|"ret_target"|"ret_merge"|"sure_threshold","value":...,"why":"..."}}
Position: "first" = checked before the other soft rules of its stage; "before_keep" (default) = before the stage's first keep rule; "last" = after all of them (a rule placed after rules that already cover every case never fires).
Operators: < <= > >= ==. Values are numbers."""


def parse_json_list(text):
    m = re.search(r"\[.*\]", text or "", flags=re.S)
    if not m:
        return []
    for cand in (m.group(0), re.sub(r",\s*([\]}])", r"\1", m.group(0))):
        try:
            v = json.loads(cand)
            return [x for x in v if isinstance(x, dict)][:3]
        except Exception:  # noqa: BLE001
            continue
    return []


def mock_planner(rnd):
    return [
        [{"action": "amend", "id": "A4", "rule": {"stage": "answer", "when": [["words_in_passages", ">=", 0.97], ["llama_stated", "<=", 0.9]], "then": "escalate"},
          "why": "borderline answers"},
         {"action": "param", "name": "votes", "value": 3, "why": "more votes"}],
        [{"action": "add", "rule": {"stage": "answer", "when": [["llama_conf_pct", "<", 0.15]], "then": "escalate"}, "why": "low token prob"},
         {"action": "amend", "id": "H1", "rule": {"stage": "retrieval", "when": [["r05_conf_pct", "<", 0.5]], "then": "escalate"}, "why": "test lock"}],
        [{"action": "remove", "id": "A4", "why": "no-op"}],
        [{"action": "param", "name": "sure_threshold", "value": 0.8, "why": "test H5"}],
    ][rnd % 4]


# ==================================================================================
def cmd_learn(args):
    E = Env(args)
    out = args.out_dir
    os.makedirs(os.path.join(out, "rulebooks"), exist_ok=True)
    Wd = E.dev
    # signal value lists for percentile moves of raw-valued signals (answer signals on the 0.5B shortlists)
    E.sigvals = {}
    for s in ("llama_stated", "words_in_passages", "answer_words"):
        E.sigvals[s] = np.sort([E.answer_signals(Wd, q, Wd.member(q, "r05")[0])[s] for q in Wd.qids])
    E.sigvals["route_p05"] = np.sort([Wd.route[q]["p_ret"]["r05"] for q in Wd.qids])
    fixed = {r: summ([{"lenient": E.A["q32"][(q, Wd.member(q, r)[0])]["lenient"], "EM": 0, "flops": 0, "ret_esc": 0, "ans_esc": 1} for q in Wd.qids])["acc"]
             for r in LADDER}
    best_fixed = max(fixed.values())
    E.base_flops_A = float(sum(Wd.route[q]["flops"] + E.A["q32"][(q, Wd.member(q, "r05")[0])]["flops"] for q in E.halfA))
    log("dev fixed pipelines (retriever -> Qwen 32B): " + ", ".join(f"{RNAME[r]} {100*v:.1f}%" for r, v in fixed.items()))
    rb = seed_rulebook()
    state_p = os.path.join(out, "learn_state.json")
    st = json.load(open(state_p)) if os.path.exists(state_p) else {"round": 0, "rulebook": rb, "history": [], "log": []}
    rb = st["rulebook"]
    planner = None
    while st["round"] < args.rounds:
        rnd = st["round"] + 1
        base = copy.deepcopy(rb)
        cur = {h: evaluate(E, Wd, qs, rb) for h, qs in (("A", E.halfA), ("B", E.halfB))}
        cur_dev = cur["A"] + cur["B"]
        log(f"round {rnd}: current dev accuracy {100*summ(cur_dev)['acc']:.1f}%, Qwen reached {100*summ(cur_dev)['qwen']:.0f}%, retrieval escalated {100*summ(cur_dev)['ret_esc']:.0f}%")
        cands = [("code", d, c) for d, c in threshold_steps(E, rb)]
        # planner proposals
        if args.planner == "mock":
            props = mock_planner(rnd - 1)
            raw = json.dumps(props)
        else:
            if planner is None:
                models = W.load_models(args.models_json)
                planner = W.make_engine("vllm", models, args.planner_model, max_model_len=16384, gpu_util=0.88)
            prompt_txt = planner_prompt(E, rb, cur["A"], st["history"])
            p = planner.encode(PLANNER_SYS, prompt_txt)
            raw = planner.generate([p], 1, 0.7, 2500, [zlib.crc32(f"r{rnd}".encode()) % 99991], [{}])[0][0][0]
            open(os.path.join(out, f"planner_round{rnd}.txt"), "w", encoding="utf-8").write(prompt_txt + "\n\n=== RESPONSE ===\n" + raw)
            props = parse_json_list(raw)
        log(f"  planner: {len(props)} proposals")
        for p in props:
            c, d = apply_proposal(rb, p)
            if c is None:
                st["log"].append({"round": rnd, "source": "planner", "change": json.dumps(p)[:300], "decision": "rejected", "reason": d})
                st["history"].append(f"  round {rnd}: {json.dumps(p)[:200]} -> REJECTED ({d})")
                continue
            cands.append(("planner", d + (f" (why: {p.get('why')})" if p.get("why") else ""), c))
        # evaluate every candidate against the current rulebook; accept greedily (best first), re-checking after each acceptance
        accepted_any = True
        tried = set()
        while accepted_any:
            accepted_any = False
            scored = []
            for src, d, c in cands:
                if d in tried:
                    continue
                ok7, why7 = h7_ok(E, base, c)
                if not ok7:
                    tried.add(d)
                    st["log"].append({"round": rnd, "source": src, "change": d, "decision": "rejected", "reason": why7})
                    continue
                res = {h: evaluate(E, Wd, qs, c) for h, qs in (("A", E.halfA), ("B", E.halfB))}
                repair = summ(cur_dev)["acc"] < best_fixed - HARD["H4_max_acc_drop_vs_best_fixed"]
                okA, dA_A, dC_A = ok_half(summ(res["A"]), summ(cur["A"]), repair)
                okB, dA_B, dC_B = ok_half(summ(res["B"]), summ(cur["B"]), repair)
                rec = {"round": rnd, "source": src, "change": d, "half A": f"{100*dA_A:+.1f} pt, {100*dC_A:+.1f}% compute",
                       "half B": f"{100*dA_B:+.1f} pt, {100*dC_B:+.1f}% compute"}
                if not (okA and okB):
                    tried.add(d)
                    st["log"].append(dict(rec, decision="rejected", reason="not better on both halves" + (" (repair mode: needs higher accuracy on both)" if repair else "")))
                    if src == "planner":
                        st["history"].append(f"  round {rnd}: {d} -> rejected (half A {rec['half A']}, half B {rec['half B']})")
                    continue
                dev_rows = res["A"] + res["B"]
                g = guardrails(E, c, dev_rows, best_fixed, summ(cur_dev)["acc"])
                if g:
                    tried.add(d)
                    st["log"].append(dict(rec, decision="rejected", reason="; ".join(g)))
                    if src == "planner":
                        st["history"].append(f"  round {rnd}: {d} -> rejected ({'; '.join(g)})")
                    continue
                ctrl = random_control(E, Wd, Wd.qids, cur_dev, dev_rows, rb, draws=args.draws, seed=rnd)
                acc = summ(dev_rows)["acc"]
                if acc <= ctrl + 1e-9 and any(a["ret_esc"] != b["ret_esc"] or a["ans_esc"] != b["ans_esc"] for a, b in zip(dev_rows, cur_dev)):
                    tried.add(d)
                    st["log"].append(dict(rec, decision="rejected", reason=f"does not beat random escalation of the same size ({100*acc:.1f}% vs {100*ctrl:.1f}%)"))
                    if src == "planner":
                        st["history"].append(f"  round {rnd}: {d} -> rejected (no better than random escalation)")
                    continue
                util = 100 * (dA_A + dA_B) / 2 - (dC_A + dC_B) / 2 / HARD["H8_max_compute_per_point"]   # 1 point of accuracy = 10% compute
                if repair:
                    util = 100 * (dA_A + dA_B) / 2
                scored.append((util, src, d, c, res, rec, ctrl))
            if scored:
                util, src, d, c, res, rec, ctrl = max(scored, key=lambda x: x[0])
                tried.add(d)
                rb = c
                rb["version"] = base["version"] + 1 if rb.get("version", 0) <= base.get("version", 0) else rb["version"]
                cur = res
                cur_dev = res["A"] + res["B"]
                st["log"].append(dict(rec, decision="ACCEPTED", reason=f"both halves pass; random control {100*ctrl:.1f}% vs {100*summ(cur_dev)['acc']:.1f}%"))
                st["history"].append(f"  round {rnd}: {d} -> ACCEPTED (half A {rec['half A']}, half B {rec['half B']})")
                log(f"  ACCEPTED [{src}] {d}  | A {rec['half A']} | B {rec['half B']}")
                accepted_any = True
        rb["version"] = rnd
        st["round"], st["rulebook"] = rnd, rb
        json.dump(rb, open(os.path.join(out, "rulebooks", f"rulebook_v{rnd}.json"), "w"), indent=1)
        json.dump(st, open(state_p, "w"), indent=1)
        log(f"round {rnd} done: dev accuracy {100*summ(cur_dev)['acc']:.1f}%, Qwen reached {100*summ(cur_dev)['qwen']:.0f}%")
    json.dump(seed_rulebook(), open(os.path.join(out, "rulebooks", "rulebook_v0_seed.json"), "w"), indent=1)
    json.dump(rb, open(os.path.join(out, "rulebooks", "rulebook_final.json"), "w"), indent=1)
    log("learning complete")


def fit_sure(E, rb):
    """Calibrated chance the final answer is right, fitted on all dev questions under this rulebook."""
    rows = evaluate(E, E.dev, E.dev.qids, rb)
    return R9.fit_lr([r["sure_feats"] for r in rows], [r["lenient"] for r in rows])


def boot(a, b, B=4000):
    rng = np.random.default_rng(0)
    n = len(a)
    d = [a[w].mean() - b[w].mean() for w in (rng.integers(0, n, n) for _ in range(B))]
    return 100 * float(np.percentile(d, 2.5)), 100 * float(np.percentile(d, 97.5))


def cmd_test(args):
    import pandas as pd
    E = Env(args)
    out = args.out_dir
    res = os.path.join(out, "results")
    os.makedirs(res, exist_ok=True)
    Wt = E.test
    E.sigvals = {}
    seed, final = seed_rulebook(), json.load(open(os.path.join(out, "rulebooks", "rulebook_final.json")))
    st = json.load(open(os.path.join(out, "learn_state.json")))
    models = W.load_models(os.path.join(args.route_run, "models.json"))
    ref = C8.ref_rows(E.ns, models)
    fixed = np.array([(E.A["q32"][(q, ref[q]["sel"])]["lenient"], E.A["q32"][(q, ref[q]["sel"])]["EM"], ref[q]["flops"] + E.A["q32"][(q, ref[q]["sel"])]["flops"]) for q in Wt.qids])

    def single(r, m):
        rows = []
        for q in Wt.qids:
            s, _, f = Wt.member(q, r)
            a = E.A[m][(q, s)]
            rows.append((a["lenient"], a["EM"], (Wt.route[q]["flops"] if r == "r05" else f) + a["flops"]))
        return np.array(rows)
    pols = {"Cheap fixed setup: Qwen 7B retrieves, Qwen 32B answers (AgentTTS's pick)": fixed,
            "0.5B retrieves, Llama 8B answers": single("r05", "l8"), "0.5B retrieves, Qwen 32B answers": single("r05", "q32"),
            "0.8B retrieves, Qwen 32B answers": single("r08", "q32"), "3B retrieves, Qwen 32B answers": single("r3", "q32")}
    detail = {}
    for name, rb in (("Starting rulebook (soft rules as agreed, before any micro-experiment)", seed), (f"LEARNT rulebook (v{final['version']}, after {st['round']} rounds)", final)):
        rows = evaluate(E, Wt, Wt.qids, rb)
        sure = fit_sure(E, rb)
        thr = max(HARD["H5_min_sure_score"], rb["params"]["sure_threshold"])
        sc = sure([r["sure_feats"] for r in rows])
        for r, s in zip(rows, sc):
            r["sure_score"], r["flag"] = float(s), ("Sure" if s >= thr else "Unsure")
        detail[name] = rows
        pols[name] = np.array([(r["lenient"], r["EM"], r["flops"]) for r in rows])
        # random control: same escalation counts per stage, random questions (20 draws)
        rng = np.random.default_rng(3)
        accs = []
        for _ in range(20):
            fr = rng.permutation([bool(r["ret_esc"]) for r in rows])
            fa = rng.permutation([bool(r["ans_esc"]) for r in rows])
            rr = evaluate(E, Wt, Wt.qids, rb, {q: {"ret": bool(a), "ans": bool(b)} for q, a, b in zip(Wt.qids, fr, fa)})
            accs.append([r["lenient"] for r in rr])
        detail[name + " | random control"] = np.mean(accs, 0)
    orc = []
    for q in Wt.qids:
        s = Wt.member(q, "r05")[0]
        aL, aQ = E.A["l8"][(q, s)], E.A["q32"][(q, s)]
        orc.append((max(aL["lenient"], aQ["lenient"]), max(aL["EM"], aQ["EM"]), Wt.route[q]["flops"] + aL["flops"] + (0 if aL["lenient"] else aQ["flops"])))
    pols["Oracle: Llama if right, else Qwen 32B, on the 0.5B shortlist (needs the answer key)"] = np.array(orc)
    y08 = pols["0.8B retrieves, Qwen 32B answers"][:, 0]
    rows_p = []
    for name, v in pols.items():
        lo, hi = boot(v[:, 0], fixed[:, 0])
        lo8, hi8 = boot(v[:, 0], y08)
        row = {"Policy": name, "Accuracy (corrected label)": float(v[:, 0].mean()), "Exact match": float(v[:, 1].mean()),
               "Compute vs the cheap fixed setup": float(v[:, 2].sum() / fixed[:, 2].sum() - 1),
               "vs cheap fixed (points)": 100 * float(v[:, 0].mean() - fixed[:, 0].mean()), "low": lo, "high": hi,
               "vs 0.8B -> Qwen 32B (points)": 100 * float(v[:, 0].mean() - y08.mean()), "low ": lo8, "high ": hi8}
        if name in detail:
            rows = detail[name]
            ctrl = detail[name + " | random control"]
            lc, hc = boot(v[:, 0], ctrl)
            sure = np.array([r["flag"] == "Sure" for r in rows])
            row |= {"Retrieval escalated": float(np.mean([r["ret_esc"] for r in rows])), "Reached Qwen 32B": float(np.mean([r["ans_esc"] for r in rows])),
                    "vs random escalation, same counts (points)": 100 * float(v[:, 0].mean() - ctrl.mean()), "low  ": lc, "high  ": hc,
                    "Flagged Sure": float(sure.mean()), "Right when Sure": float(v[sure, 0].mean()) if sure.any() else None,
                    "Right when Unsure": float(v[~sure, 0].mean()) if (~sure).any() else None}
        rows_p.append(row)
    # which rule decided, on the test questions (learnt rulebook)
    lname = [k for k in detail if k.startswith("LEARNT") and "|" not in k][0]
    rows_r = []
    for r in final["rules"]:
        fired = [x for x in detail[lname] if r["id"] in x["fired"]]
        rows_r.append({"Rule": rule_text(r), "Plain English": r.get("text", ""), "Fired on test questions": len(fired),
                       "Final accuracy there": float(np.mean([x["lenient"] for x in fired])) if fired else None})
    per_q = [{"qid": x["qid"], "question": Wt.qd[x["qid"]]["question"], "gold": Wt.qd[x["qid"]]["answer"], "rules that decided (retrieval | retrieval2 | answer)": " | ".join(x["fired"]),
              "retrievers": x["members"], "votes": x["votes"], "answer": x["answer"], "sure score": x["sure_score"], "flag": x["flag"], "right (corrected)": x["lenient"],
              "exact match": x["EM"], "TFLOPs": x["flops"] / 1e12} for x in detail[lname]]

    def df(rr):
        d = pd.DataFrame(rr)
        for c in d.columns:
            if d[c].isna().any():
                d[c] = d[c].astype(object).where(d[c].notna(), "-")
        return d
    W.write_excel(os.path.join(res, "set10_rulebook.xlsx"), {"Test policies": df(rows_p), "Learnt rules on test": df(rows_r), "Micro-experiment log": df(st["log"]),
                                                            "Per question (learnt)": df(per_q)})
    for nm, rr in (("policies.jsonl", rows_p), ("rules_on_test.jsonl", rows_r), ("micro_experiments.jsonl", st["log"]), ("per_question.jsonl", per_q)):
        C8.write(os.path.join(res, nm), rr)
    T = lambda rr, cols, pct=(), sg=(): ("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "\n".join(
        "| " + " | ".join(("-" if r.get(c) is None else (r[c] if isinstance(r[c], str) else (f"{100 * r[c]:.1f}%" if c in pct else (f"{r[c]:+.1f}" if c in sg else
                                                                                                                           (str(r[c]) if isinstance(r[c], int) else f"{r[c]:.2f}")))))
                          for c in cols) + " |" for r in rr))
    acc_log = [x for x in st["log"] if x["decision"] == "ACCEPTED"]
    L = ["# Set 10: a self-learnt rulebook for model switching\n",
         f"Code version {VERSION}. Dev: 1,000 questions (halves A and B); {st['round']} rounds of micro-experiments; test: the 400 test questions, used once. "
         "Compute counts every model that runs.\n",
         "## Test results\n",
         T(rows_p, ["Policy", "Accuracy (corrected label)", "Exact match", "Compute vs the cheap fixed setup", "vs cheap fixed (points)", "low", "high",
                    "vs 0.8B -> Qwen 32B (points)", "low ", "high "], pct={"Accuracy (corrected label)", "Exact match", "Compute vs the cheap fixed setup"},
           sg={"vs cheap fixed (points)", "low", "high", "vs 0.8B -> Qwen 32B (points)", "low ", "high "}), "",
         "## Does the rulebook switch better than chance? Sure / Unsure\n",
         T([r for r in rows_p if "Retrieval escalated" in r], ["Policy", "Retrieval escalated", "Reached Qwen 32B", "vs random escalation, same counts (points)", "low  ", "high  ",
                                                               "Flagged Sure", "Right when Sure", "Right when Unsure"],
           pct={"Retrieval escalated", "Reached Qwen 32B", "Flagged Sure", "Right when Sure", "Right when Unsure"},
           sg={"vs random escalation, same counts (points)", "low  ", "high  "}), "",
         "## The learnt rulebook\n", "Settings: " + json.dumps(final["params"]) + "\n",
         T(rows_r, ["Rule", "Fired on test questions", "Final accuracy there"], pct={"Final accuracy there"}), "",
         f"## Changes accepted by the micro-experiments ({len(acc_log)} of {len(st['log'])} tried)\n",
         T(acc_log, ["round", "source", "change", "half A", "half B"]) if acc_log else "None: every proposed change failed the evidence test.", "",
         "## Changes rejected (first 25)\n",
         T([x for x in st["log"] if x["decision"] != "ACCEPTED"][:25], ["round", "source", "change", "reason"]), "",
         "The full log, the rulebook after every round (`rulebooks/`), the planner's prompts and answers (`planner_round*.txt`) and a per-question table "
         "(rules that decided, retrievers, votes, answer, Sure/Unsure, compute) are saved with this run.\n"]
    open(os.path.join(res, "summary.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    log(f"results -> {res}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("learn", "test"):
        p = sub.add_parser(c)
        for a in ("route_run", "label_dir", "old_data", "fresh_data", "old_sweep", "fresh_run", "out_dir"):
            p.add_argument("--" + a, required=True)
        p.add_argument("--models_json", default="")
        p.add_argument("--planner", choices=["vllm", "mock"], default="vllm")
        p.add_argument("--planner_model", default="q32")
        p.add_argument("--rounds", type=int, default=4)
        p.add_argument("--draws", type=int, default=10)
    args = ap.parse_args()
    {"learn": cmd_learn, "test": cmd_test}[args.cmd](args)


if __name__ == "__main__":
    main()
