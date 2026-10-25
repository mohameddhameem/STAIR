"""Workstreams 1 and 2: AgentTTS on HotpotQA, as in its code (code_others/archon_hotpotqa.py), and the strongest-model baseline.

A setting = (retriever, retrieval samples, answerer, answer samples). Pipeline per question (their 'retrieve-merge-generate'):
  retriever samples k_r shortlists (temperature 0.9, up to 512 tokens) -> Merge (reference numbers voted) ->
  answerer samples k_a answers (temperature 0.7, 512 tokens) on the merged references -> the same answerer fuses them
  (temperature 0.3, 512 tokens) -> the fused response is scored.
ws1_grid   : every setting of the grid on the 50 search questions (samples drawn once at the largest count, first k used)
ws1_search : AgentTTS's planner search over that table (agenttts_search.py), within the budget
ws1_test   : the setting the search chose, AgentTTS's reported best, and workstream 2's setting, on the 500 test questions."""
import itertools
import os
import random

import numpy as np

from . import data as D
from .common import (ATTS_FUSE_SYSTEM, ATTS_QA_SYSTEM, ATTS_RET_SYSTEM, append_jsonl, atts_budget, atts_fuse_prompt, atts_merge, atts_qa_prompt,
                     atts_ret_prompt, atts_selected_refs, key, log, scores, units, read_json, read_jsonl, seed_of, write_json)
from .config import model_tag, serve_entry, variant
from .engines import free_gpu, make_engine
from .progress import Tracker, mark_done

ATTS_NAMES = {"q7": "Qwen/Qwen2.5-7B-Instruct-AWQ", "q32": "Qwen/Qwen2.5-32B-Instruct-AWQ", "q72": "Qwen/Qwen2.5-72B-Instruct-AWQ",
              "l3": "meta-llama/Llama-3.2-3B-Instruct", "l8": "meta-llama/Llama-3.1-8B-Instruct", "l70": "meta-llama/Llama-3.1-70B-Instruct",
              # small models: only for the GPU smoke test (configs/smoke.yaml); not part of AgentTTS's menu
              "q05": "Qwen/Qwen2.5-0.5B-Instruct", "q15": "Qwen/Qwen2.5-1.5B-Instruct", "l1": "meta-llama/Llama-3.2-1B-Instruct"}
FROM_NAME = {v: k for k, v in ATTS_NAMES.items()}
HOTPOT_DESC = ("This task Retreval-then-generate has two subtasks. The first subtask is retrieval: given a question and a set of 100 text chunks, the model "
               "needs to select the most relevant chunks that can help answer the question. The second subtask is question answering: based on the "
               "retrieved chunks, the model should generate a response that answers the question accurately.")


def wdir(cfg, *p):
    d = os.path.join(cfg["run_dir"], "ws1", *p)
    os.makedirs(d, exist_ok=True)
    return d


def f_ret(cfg, split, m):
    return os.path.join(wdir(cfg, "step"), f"{split}_ret_{m}.jsonl")


def f_ans(cfg, split, a):
    return os.path.join(wdir(cfg, "step"), f"{split}_ans_{a}.jsonl")


def f_fuse(cfg, split, a):
    return os.path.join(wdir(cfg, "step"), f"{split}_fuse_{a}.jsonl")


# ==================================================================================
# plans
# ==================================================================================
def grid_plan(cfg):
    W = cfg["workstreams"]["ws1"]
    G = W["grid"]
    return [(rm, rk, am, ak) for rm in W["retrievers"] for rk in G["retrieval"][rm] for am in W["answerers"] for ak in G["answer"][am]]


def test_plan(cfg):
    out = []
    ch = None if cfg["workstreams"]["ws1"].get("skip_search") else read_json(os.path.join(wdir(cfg), "search_result.json"))
    if ch:
        out.append(tuple(ch["chosen"]))
    for c in cfg["workstreams"]["ws1"].get("also_test", []):
        out.append(tuple(c))
    if "ws2" in cfg["workstreams"]:
        out.append(tuple(cfg["workstreams"]["ws2"]["config"]))
    seen = []
    for c in out:
        c = (c[0], int(c[1]), c[2], int(c[3]))
        if c not in seen:
            seen.append(c)
    return seen


def plan_for(cfg, split):
    return grid_plan(cfg) if split == "search" else test_plan(cfg)


# ==================================================================================
# generation
# ==================================================================================
def _load(path, k):
    return {tuple(r[x] if not isinstance(r[x], list) else tuple(r[x]) for x in k): r for r in read_jsonl(path)}


def run_retrieval(cfg, split, rm):
    plan = plan_for(cfg, split)
    n = max([rk for r, rk, _, _ in plan if r == rm], default=0)
    if not n:
        return
    qs = D.load(cfg, split)
    path = f_ret(cfg, split, rm)
    have = {r["qid"]: r for r in read_jsonl(path)}
    todo = [q for q in qs if len(have.get(q["qid"], {}).get("samples", [])) < n]
    tr = Tracker(cfg, f"ws1_{'grid' if split == 'search' else 'test'}", f"retrieve-{rm}", len(qs), "questions")
    tr.tick(len(qs) - len(todo))
    if todo:
        m = variant(cfg, rm)
        eng = make_engine(cfg, serve_entry(m))
        chunk = max(1, cfg["gpu"]["chunk"] // max(1, n // 8))
        for c in range(0, len(todo), chunk):
            part = todo[c:c + chunk]
            prompts = [eng.encode(ATTS_RET_SYSTEM, atts_ret_prompt(q)) for q in part]
            outs = eng.generate(prompts, n, 0.9, 512, [seed_of("r", rm, q["qid"]) for q in part],
                                [{"kind": "ret_atts", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"])} for q in part])
            append_jsonl(path, [{"qid": q["qid"], "prompt_tokens": len(p), "samples": [{"text": t, "out_tokens": k} for t, k, _ in o]}
                                for q, p, o in zip(part, prompts, outs)])
            tr.tick(len(qs) - len(todo) + c + len(part))
        free_gpu(eng)
    tr.finish()


def shortlists(cfg, split, rm, k):
    """{qid: merged reference numbers} from the first k retrieval samples."""
    rows = {r["qid"]: r for r in read_jsonl(f_ret(cfg, split, rm))}
    return {q: atts_merge([s["text"] for s in r["samples"][:k]]) for q, r in rows.items()}


def answer_needs(cfg, split, am):
    """{(qid, sorted shortlist): (shortlist, {answer sample counts})} for one answerer."""
    plan = [c for c in plan_for(cfg, split) if c[2] == am]
    cache, out = {}, {}
    for rm, rk, _, ak in plan:
        if (rm, rk) not in cache:
            cache[(rm, rk)] = shortlists(cfg, split, rm, rk)
        for q, ids in cache[(rm, rk)].items():
            e = out.setdefault((q, key(ids)), (ids, set()))
            e[1].add(ak)
    return out


def run_answers(cfg, split, am):
    """Answer samples (up to the largest count needed) and every fusion the plan needs, for one answerer."""
    needs = answer_needs(cfg, split, am)
    if not needs:
        return
    qm = D.qmap(cfg, [split])
    m = variant(cfg, am)
    stage = f"ws1_{'grid' if split == 'search' else 'test'}"
    ap, fp = f_ans(cfg, split, am), f_fuse(cfg, split, am)
    have = {(r["qid"], tuple(r["skey"])): r for r in read_jsonl(ap)}
    todo = [(q, sk) for (q, sk), (ids, ks) in sorted(needs.items()) if len(have.get((q, sk), {}).get("samples", [])) < max(ks)]
    fhave = {(r["qid"], tuple(r["skey"]), r["k"]) for r in read_jsonl(fp)}
    n_f = sum(len(ks) for ks in (v[1] for v in needs.values()))
    tr = Tracker(cfg, stage, f"answer-{am}", len(needs) + n_f, "answer sets + fusions")
    eng = None
    if todo:
        eng = make_engine(cfg, serve_entry(m))
        chunk = cfg["gpu"]["chunk"]
        for c in range(0, len(todo), chunk):
            part = todo[c:c + chunk]
            nmax = [max(needs[x][1]) for x in part]
            # one vLLM call per sample count (counts differ by setting); prompts sorted by question for prefix caching
            rows = []
            for n in sorted(set(nmax)):
                sub = [x for x, k in zip(part, nmax) if k == n]
                prompts = [eng.encode(ATTS_QA_SYSTEM, atts_qa_prompt(qm[q]["question"], atts_selected_refs(qm[q], needs[(q, sk)][0]))) for q, sk in sub]
                outs = eng.generate(prompts, n, 0.7, 512, [seed_of("a", am, q, sk) for q, sk in sub],
                                    [{"kind": "qa_free", "gold": qm[q]["answer"], "has_gold": _has_gold(qm[q], sk)} for q, sk in sub])
                rows += [{"qid": q, "skey": list(sk), "sel": needs[(q, sk)][0], "prompt_tokens": len(p),
                          "samples": [{"text": t, "out_tokens": k} for t, k, _ in o]} for (q, sk), p, o in zip(sub, prompts, outs)]
            append_jsonl(ap, rows)
            tr.tick(len(needs) - len(todo) + c + len(part))
        have = {(r["qid"], tuple(r["skey"])): r for r in read_jsonl(ap)}
    jobs = sorted((q, sk, k) for (q, sk), (ids, ks) in needs.items() for k in ks if (q, sk, k) not in fhave)
    tr.tick(len(needs) + n_f - len(jobs))
    if jobs:
        eng = eng or make_engine(cfg, serve_entry(m))
        chunk = max(8, cfg["gpu"]["chunk"] // 4)
        for c in range(0, len(jobs), chunk):
            part = jobs[c:c + chunk]
            resp = [[s["text"] for s in have[(q, sk)]["samples"][:k]] for q, sk, k in part]
            prompts = [eng.encode(ATTS_FUSE_SYSTEM, atts_fuse_prompt(qm[q], r)) for (q, sk, k), r in zip(part, resp)]
            outs = eng.generate(prompts, 1, 0.3, 512, [seed_of("f", am, q, sk, k) for q, sk, k in part],
                                [{"kind": "fuse", "gold": qm[q]["answer"], "responses": r} for (q, sk, k), r in zip(part, resp)])
            append_jsonl(fp, [{"qid": q, "skey": list(sk), "k": k, "text": o[0][0], "prompt_tokens": len(p), "out_tokens": o[0][1]}
                              for (q, sk, k), p, o in zip(part, prompts, outs)])
            tr.tick(len(needs) + n_f - len(jobs) + c + len(part))
    if eng is not None:
        free_gpu(eng)
    tr.finish()


def _has_gold(q, sk):
    return set(i + 1 for i in q["ret_answers"]) <= set(sk)


# ==================================================================================
# scoring
# ==================================================================================
def prf(gt, pred):
    gt, pred = set(gt), set(pred)
    if not pred or not gt:
        return 0.0, 0.0, 0.0
    tp = len(gt & pred)
    p, r = tp / len(pred), tp / len(gt)
    return p, r, (2 * p * r / (p + r) if tp else 0.0)


def evaluate(cfg, split, plan=None):
    """Per setting: AgentTTS's metrics (Gen_EM = the team's lenient ruler, which reproduces AgentTTS's published score, on
    the fused response; Gen_F1; Ret_*), AgentTTS's fixed-length budget (for its search) and measured AgentTTS units.
    Returns (table rows, {setting: per-question rows})."""
    plan = plan or plan_for(cfg, split)
    qs = D.load(cfg, split)
    nd = cfg["workstreams"]["ws1"]["np_nd"]
    R = {rm: {r["qid"]: r for r in read_jsonl(f_ret(cfg, split, rm))} for rm in {c[0] for c in plan}}
    A = {am: {(r["qid"], tuple(r["skey"])): r for r in read_jsonl(f_ans(cfg, split, am))} for am in {c[2] for c in plan}}
    F = {am: {(r["qid"], tuple(r["skey"]), r["k"]): r for r in read_jsonl(f_fuse(cfg, split, am))} for am in {c[2] for c in plan}}
    table, per = [], {}
    for rm, rk, am, ak in plan:
        Pr, Pa = cfg["models"][rm]["params_b"], cfg["models"][am]["params_b"]
        tag = f"retriever={model_tag(variant(cfg, rm))};answerer={model_tag(variant(cfg, am))}"
        rows = []
        for q in qs:
            r = R[rm].get(q["qid"])
            if not r:
                continue
            ids = atts_merge([s["text"] for s in r["samples"][:rk]])
            sk = key(ids)
            a, f = A[am].get((q["qid"], sk)), F[am].get((q["qid"], sk, ak))
            if not (a and f):
                continue
            gt = [i + 1 for i in q["ret_answers"]]
            p, rc, f1r = prf(gt, ids)
            sc = scores(f["text"], q["answer"])
            ur = units(Pr, r["prompt_tokens"], sum(s["out_tokens"] for s in r["samples"][:rk]))     # shared prompt counted once
            ua = units(Pa, a["prompt_tokens"], sum(s["out_tokens"] for s in a["samples"][:ak]))
            uf = units(Pa, f["prompt_tokens"], f["out_tokens"])
            rows.append({"qid": q["qid"], **sc, "ret_f1": f1r, "ret_p": p, "ret_r": rc, "ret_acc": float(rc == 1.0),
                         "units": ur + ua + uf, "units_retrieve": ur, "units_answer": ua, "units_fuse": uf, "answer": f["text"], "models": tag})
        if not rows:
            continue
        b = (atts_budget(rk, cfg["models"][rm]["nominal"], *nd["Retrieval"]) + atts_budget(ak, cfg["models"][am]["nominal"], *nd["Question Answering"]))
        table.append({"retrieve_model_name": ATTS_NAMES[rm], "retrieve_samples": rk, "generate_model_name": ATTS_NAMES[am], "generate_samples": ak,
                      "examples": len(rows), "Gen_EM": float(np.mean([x["acc"] for x in rows])), "Gen_F1": float(np.mean([x["F1"] for x in rows])),
                      "Gen_Precision": np.nan, "Gen_Recall": np.nan, "Ret_ACC": float(np.mean([x["ret_acc"] for x in rows])),
                      "Ret_F1": float(np.mean([x["ret_f1"] for x in rows])), "Ret_Precision": float(np.mean([x["ret_p"] for x in rows])),
                      "Ret_Recall": float(np.mean([x["ret_r"] for x in rows])), "budget": b, "units_per_q": float(np.mean([x["units"] for x in rows])),
                      "setting": [rm, rk, am, ak]})
        per[(rm, rk, am, ak)] = rows
    return table, per


def grid_table(cfg):
    import pandas as pd
    table, _ = evaluate(cfg, "search")
    df = pd.DataFrame(table)
    if df.empty:                     # answers not run yet (each grid part is its own job): nothing to tabulate
        return df
    df.drop(columns=["setting"]).to_csv(os.path.join(wdir(cfg), "search_table.csv"), index=False)
    return df


# ==================================================================================
# search
# ==================================================================================
def search(cfg):
    from . import agenttts_search as S
    W = cfg["workstreams"]["ws1"]
    out = wdir(cfg)
    if W.get("skip_search"):
        log("AL-02 only: no AgentTTS search")
        return
    if os.path.exists(os.path.join(out, "search_result.json")):
        log("AgentTTS search already done")
        mark_done(cfg, "ws1_search")
        return
    df = grid_table(cfg)
    log(f"search table: {len(df)} settings measured on the search questions")
    random.seed(0)
    np.random.seed(0)
    lines = []
    S._LOG = lambda *a: lines.append(" ".join(str(x) for x in a))
    S.Np_list = [W["np_nd"]["Retrieval"][0], W["np_nd"]["Question Answering"][0]]
    S.Nd_list = [W["np_nd"]["Retrieval"][1], W["np_nd"]["Question Answering"][1]]
    sub_tasks = ["Retrieval", "Question Answering"]
    model_choices = [[ATTS_NAMES[m] for m in W["retrievers"]], [ATTS_NAMES[m] for m in W["answerers"]]]
    S.model_space = {t: {"model_choice": mc, "prompt_length": a, "generation_length": b} for t, mc, a, b in zip(sub_tasks, model_choices, S.Np_list, S.Nd_list)}
    S._ENV = S.TableEnvironment(df)
    tr = Tracker(cfg, "ws1_search", "all", 1 + W["iterations"] * 2, "planner calls")
    llm = _PlannerLLM(cfg, tr, model_choices, W["batch_size"])
    unit = {"Retrieval": round(atts_budget(1, cfg["models"][W["retrievers"][-1]]["nominal"], *W["np_nd"]["Retrieval"])),
            "Question Answering": round(atts_budget(1, cfg["models"][W["answerers"][-1]]["nominal"], *W["np_nd"]["Question Answering"]))}
    example = {"candidates": [{t: {"model": mc[-1], "samples": 1, "budget": unit[t]} for t, mc in zip(sub_tasks, model_choices)}]}
    agent = S.RAGAgentTTS("hotpotqa", HOTPOT_DESC, W["budget"], model_choices, r"-(\d+)B-", sub_tasks, S.Np_list, S.Nd_list, llm, example)
    hub = S.Task("hotpotqa", HOTPOT_DESC, W["budget"], model_choices, r"-(\d+)B-", sub_tasks, agent)
    results, trace = hub.get_tts_strategy(iterations=W["iterations"], batch_size=W["batch_size"])
    hist = agent.archive.history
    best = max(hist, key=lambda h: (h["score"]["Gen_EM"], -h["cost"]))
    p = best["params"]
    chosen = [FROM_NAME[p["Retrieval"]["model"]], int(p["Retrieval"]["samples"]), FROM_NAME[p["Question Answering"]["model"]], int(p["Question Answering"]["samples"])]
    write_json(os.path.join(out, "search_result.json"), {
        "chosen": chosen, "score_on_search_questions": best["score"], "budget": best["cost"], "trials": len(hist),
        "history": [{"trial": i + 1, "params": h["params"], "score": h["score"], "cost": h["cost"]} for i, h in enumerate(hist)],
        "best_so_far": [{"trial": s + 1, "Gen_EM": sc} for s, sc, _ in trace], "lookups": S._ENV.calls, "experience": agent.archive.experience})
    open(os.path.join(out, "search_log.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n\n" + "\n\n".join(llm.transcript))
    tr.finish(note=f"chosen {chosen}")
    log(f"AgentTTS search: {len(hist)} trials; chosen {chosen} (Gen_EM {best['score']['Gen_EM']:.3f} on the search questions, budget {best['cost']:.0f})")
    llm.close()
    mark_done(cfg, "ws1_search")


class _PlannerLLM:
    """AgentTTS's LLM interface (generate(prompt, system_prompt) -> text) backed by the local planner model."""

    def __init__(self, cfg, tracker, model_choices, batch):
        self.cfg, self.tr, self.n, self.transcript = cfg, tracker, 0, []
        self.mc, self.batch = model_choices, batch
        self.eng = make_engine(cfg, serve_entry(variant(cfg, cfg["planner"])), max_model_len=24576)

    def generate(self, prompt, system_prompt):
        self.n += 1
        kind = "atts_plan" if "strict JSON" in prompt else "atts_insight"
        meta = {"kind": kind, "ret_models": self.mc[0], "gen_models": self.mc[1], "batch": self.batch}
        out = self.eng.generate([self.eng.encode(system_prompt, prompt)], 1, 0.7, 3000, [seed_of("plan", self.n)], [meta])[0][0][0]
        self.transcript.append(f"=== call {self.n} ===\n--- prompt ---\n{prompt}\n--- reply ---\n{out}")
        self.tr.tick(self.n)
        return out

    def close(self):
        free_gpu(self.eng)


# ==================================================================================
def test_results(cfg):
    """Scores of the fixed settings on the test questions -> ws1/test_results.json (used by the report)."""
    plan = test_plan(cfg)
    table, per = evaluate(cfg, "test", plan)
    ch = None if cfg["workstreams"]["ws1"].get("skip_search") else (read_json(os.path.join(wdir(cfg), "search_result.json")) or {}).get("chosen")
    names = {}
    for c in plan:
        tags = []
        if ch and list(c) == list(ch):
            tags.append("Baseline 1: AgentTTS (setting chosen by its search)")
        if [list(x) for x in cfg["workstreams"]["ws1"].get("also_test", [])].count(list(c)):
            tags.append("AgentTTS's reported best setting")
        if "ws2" in cfg["workstreams"] and list(c) == list(cfg["workstreams"]["ws2"]["config"]):
            tags.append("Baseline 2: strongest models")
        names[c] = " = ".join(tags)
    rows = []
    for t in table:
        c = tuple(t["setting"])
        rows.append({"policy": names[c], "setting": f"{cfg['models'][c[0]]['label']} x{c[1]} -> {cfg['models'][c[2]]['label']} x{c[3]} (+ fusion)",
                     "config": list(c), "acc": t["Gen_EM"], "F1": t["Gen_F1"], "ret_found_both": t["Ret_ACC"], "budget_per_q": t["budget"],
                     "units_per_q": t["units_per_q"], "n": t["examples"], "per_question": per[c]})
    write_json(os.path.join(wdir(cfg), "test_results.json"), rows)
    return rows


def all_parts(cfg, stage):
    W = cfg["workstreams"]["ws1"]
    if stage in ("ws1_grid", "ws1_test"):
        return [f"retrieve-{m}" for m in W["retrievers"]] + [f"answer-{a}" for a in W["answerers"]]
    return ["all"]


def run_part(cfg, stage, part):
    split = "search" if stage == "ws1_grid" else "test"
    kind, _, m = part.partition("-")
    if kind == "retrieve":
        run_retrieval(cfg, split, m)
    else:
        run_answers(cfg, split, m)
    mark_done(cfg, stage, part)
    if stage == "ws1_test" and all(_done(cfg, stage, p) for p in all_parts(cfg, stage)):
        test_results(cfg)


def _done(cfg, stage, p):
    from .progress import is_done
    return is_done(cfg, stage, p)


_ = itertools  # (kept for readability of plan construction elsewhere)


# ==================================================================================
# AL-02 replication: the team's always-strong baseline (Jonathan's protocol), on our own stack
# ==================================================================================
# One greedy pass (temperature 0) of the strong retriever with AgentTTS's retrieval prompt, the shortlist parsed as
# AgentTTS's Merge does for one sample, one greedy pass of the strong answerer with AgentTTS's answer prompt, K = 1, no
# fuser. Prompts are AgentTTS's byte for byte (stair.common.atts_*; checked against archon_hotpotqa.py). Saved apart from
# WS1's sampled outputs. The result is compared question by question with the shared rows
# (results/common/jonathan-ss-hotpotqa-test.csv).
SS_PRICE = 814 + 115     # AgentTTS Table 5: Qwen2.5-72B retrieval + Llama-3.1-70B answer (the golden set's always-strong)


def f_ss(cfg, split, kind, m):
    return os.path.join(wdir(cfg, "ss_greedy"), f"{split}_{kind}_{m}.jsonl")


def ss_models(cfg):
    c = cfg["workstreams"].get("ws2", {}).get("config") or ["q72", 1, "l70", 1]
    return c[0], c[2]


def run_ss(cfg, part, split="test"):
    rm, am = ss_models(cfg)
    qs = D.load(cfg, split)
    if part.startswith("retrieve"):
        path = f_ss(cfg, split, "ret", rm)
        have = {r["qid"] for r in read_jsonl(path)}
        todo = [q for q in qs if q["qid"] not in have]
        log(f"always-strong (greedy) retrieval with {rm}: {len(todo)} of {len(qs)} questions to do")
        if todo:
            eng = make_engine(cfg, serve_entry(variant(cfg, rm)))
            chunk = cfg["gpu"]["chunk"]
            for c in range(0, len(todo), chunk):
                part_q = todo[c:c + chunk]
                prompts = [eng.encode(ATTS_RET_SYSTEM, atts_ret_prompt(q)) for q in part_q]
                outs = eng.generate(prompts, 1, 0.0, 512, [0] * len(part_q),
                                    [{"kind": "ret_atts", "gt": q["ret_answers"], "n_refs": len(q["ctxs_candidate"])} for q in part_q])
                append_jsonl(path, [{"qid": q["qid"], "prompt_tokens": len(p), "text": o[0][0], "out_tokens": o[0][1]}
                                    for q, p, o in zip(part_q, prompts, outs)])
            free_gpu(eng)
    elif part.startswith("answer"):
        R = {r["qid"]: r for r in read_jsonl(f_ss(cfg, split, "ret", rm))}
        path = f_ss(cfg, split, "ans", am)
        have = {r["qid"] for r in read_jsonl(path)}
        todo = [q for q in qs if q["qid"] in R and q["qid"] not in have]
        log(f"always-strong (greedy) answers with {am}: {len(todo)} of {len(qs)} questions to do ({len(R)} shortlists)")
        if todo:
            eng = make_engine(cfg, serve_entry(variant(cfg, am)))
            chunk = cfg["gpu"]["chunk"]
            for c in range(0, len(todo), chunk):
                part_q = todo[c:c + chunk]
                ids = [atts_merge([R[q["qid"]]["text"]]) for q in part_q]
                prompts = [eng.encode(ATTS_QA_SYSTEM, atts_qa_prompt(q["question"], atts_selected_refs(q, s))) for q, s in zip(part_q, ids)]
                outs = eng.generate(prompts, 1, 0.0, 512, [0] * len(part_q),
                                    [{"kind": "qa_free", "gold": q["answer"], "has_gold": _has_gold(q, key(s))} for q, s in zip(part_q, ids)])
                append_jsonl(path, [{"qid": q["qid"], "sel": s, "prompt_tokens": len(p), "text": o[0][0], "out_tokens": o[0][1]}
                                    for q, s, p, o in zip(part_q, ids, prompts, outs)])
            free_gpu(eng)
    else:
        ss_compare(cfg, split)
    mark_done(cfg, "ss_greedy", part)


def ss_compare(cfg, split="test"):
    """Our greedy always-strong rows (team schema) and the question-by-question comparison with the shared rows."""
    import pandas as pd
    from stair_common.schema import COLUMNS, validate
    rm, am = ss_models(cfg)
    R = {r["qid"]: r for r in read_jsonl(f_ss(cfg, split, "ret", rm))}
    A = {r["qid"]: r for r in read_jsonl(f_ss(cfg, split, "ans", am))}
    Pr, Pa = cfg["models"][rm]["params_b"], cfg["models"][am]["params_b"]
    tag = f"retriever={model_tag(variant(cfg, rm))};answerer={model_tag(variant(cfg, am))}"
    rows = []
    for q in D.load(cfg, split):
        r, a = R.get(q["qid"]), A.get(q["qid"])
        if not (r and a):
            continue
        sc = scores(a["text"], q["answer"])
        ur, ua = units(Pr, r["prompt_tokens"], r["out_tokens"]), units(Pa, a["prompt_tokens"], a["out_tokens"])
        rows.append({"dataset": cfg["data"].get("dataset", "hotpotqa"), "split": split, "id": q["qid"], "track": "selfLearnt",
                     "policy": "always-strong-replication", "menu": "A", "seed": int(cfg.get("seed", 0)), "answer": a["text"][:2000],
                     "lenient": int(round(sc["lenient"])), "em": sc["EM"], "f1": sc["F1"], "units": ur + ua, "units_retrieve": ur,
                     "units_answer": ua, "units_fuse": 0.0, "units_router": 0.0, "ms_router": None, "models": tag})
    df = pd.DataFrame(rows, columns=COLUMNS)
    validate(df)
    out = os.path.join(cfg["paths"]["results_dir"], f"AL-02-replication-{cfg['data'].get('dataset', 'hotpotqa')}-{split}-seed{cfg.get('seed', 0)}.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    df.to_csv(out, index=False)
    log(f"our greedy always-strong: {len(df)} questions, lenient {df.lenient.mean():.3f}, {df.units.mean():.0f} units -> {out}")
    ref_p = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))),   # repo root
                         "results", "common", f"jonathan-ss-hotpotqa-{split}.csv")
    if not os.path.exists(ref_p):
        log(f"shared rows not found ({ref_p}): git pull, then run --part compare again")
        return
    ref = pd.read_csv(ref_p).set_index("id")
    j = df.set_index("id").join(ref, rsuffix="_ref", how="inner")
    same = (j.lenient == j.lenient_ref).mean()
    same_ans = (j.answer.fillna("").str.strip() == j.answer_ref.fillna("").str.strip()).mean()
    gap = 100 * (j.lenient.mean() - j.lenient_ref.mean())
    verdict = ("REPLICATED" if abs(gap) <= 2 and same >= 0.90 else "NOT REPLICATED - serving differs; do not mix the two baselines")
    msg = (f"replication check on {len(j)} shared questions: ours {j.lenient.mean():.3f} vs shared {j.lenient_ref.mean():.3f} "
           f"({gap:+.1f} pt); same verdict on {100 * same:.1f}% of questions; identical answer text on {100 * same_ans:.1f}%; "
           f"ours right only {int(((j.lenient == 1) & (j.lenient_ref == 0)).sum())}, shared right only "
           f"{int(((j.lenient == 0) & (j.lenient_ref == 1)).sum())}; units {j.units.mean():.0f} vs {j.units_ref.mean():.0f} -> {verdict}")
    log(msg)
    write_json(os.path.join(wdir(cfg, "ss_greedy"), f"{split}_replication.json"),
               {"n": len(j), "ours": j.lenient.mean(), "shared": j.lenient_ref.mean(), "same_verdict": same,
                "same_answer": same_ans, "units_ours": j.units.mean(), "units_shared": j.units_ref.mean(), "verdict": verdict})
