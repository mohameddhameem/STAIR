# 11 gate_eval.py -- VALIDATION: sweep -> frontier -> LOCK (CPU only).
#
# Pre-declared locking rule (signed 8 Oct, before any frontier was seen):
#   Sweep every distinct (theta_Ret, theta_QA) cut on the VALIDATION split.
#   PRIMARY  theta_bar : the CHEAPEST pair whose composed val accuracy is
#                        >= 95% of the SS cell's val accuracy (CPT-95 convention,
#                        as in TRIM / RouteLLM / AutoMix). Only theta_bar is ever
#                        used for the single sealed-test read (by 12).
#   SECONDARY theta_knee: Kneedle knee of the Pareto frontier (axes normalized
#                        to the frontier endpoints; max perpendicular distance
#                        above the endpoint chord; ties -> cheaper point).
#   FALLBACK            : if no pair reaches the bar, theta_knee becomes primary.
#   Also reported (descriptive only): cheapest rows at 90% and 98% of SS.
#
# Inputs: 10's val_scores for both stages + the grid's cached outcomes.
# The sealed test split is never touched here.
import os, json
import numpy as np

# paths are env-overridable so variant experiments (e.g. the reward-weighted
# answering router) sweep into their OWN directory; defaults = the originals,
# so existing behavior and Wednesday's lock are untouched.
RET_SCORES = os.environ.get("SLR_RET_SCORES",
             "/mnt/hdd/jonathan/runs/sft_router5_ret/val_scores.jsonl")
QA_SCORES  = os.environ.get("SLR_QA_SCORES",
             "/mnt/hdd/jonathan/runs/sft_router5_qa/val_scores.jsonl")
GRID_FILE  = "/mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl"
OUT        = os.environ.get("SLR_GATE_OUT", "/mnt/hdd/jonathan/runs/gate_eval")
BAR_FRACTION = 0.95

# ---- unit costs: MEASURED token-aware family (AgentTTS convention), ----
# ---- produced by 13 token_costs.py from the grid's recorded token counts. ----
# ---- Six prices (decided 8 Oct): QA prompts carry the selected passages, ----
# ---- and the strong retriever selects ~2.6x fewer tokens, so each QA    ----
# ---- price is conditional on the retrieval choice (GIVEN_W / GIVEN_S =  ----
# ---- evidence from weak / strong retrieval). The flat family (1.0/4.7,  ----
# ---- params ratio) stays on the Ruler slide for the simple story only.  ----
TOKEN_COSTS = "/mnt/hdd/jonathan/runs/token_costs/token_costs.json"
if not os.path.exists(TOKEN_COSTS):
    raise SystemExit("REFUSED: token_costs.json missing -- run token_costs.py "
                     "(13) first; theta locks on the measured ruler only.")
_tc = json.load(open(TOKEN_COSTS))["gate_eval_constants"]
COST_RET_W        = _tc["COST_RET_W"]         # weak retrieval, always paid (sunk)
COST_RET_S        = _tc["COST_RET_S"]         # strong retrieval, paid on Ret escalation
COST_QA_W_GIVEN_W = _tc["COST_QA_W_GIVEN_W"]  # weak answering, evidence from weak retrieval
COST_QA_W_GIVEN_S = _tc["COST_QA_W_GIVEN_S"]  # weak answering, evidence from strong retrieval
COST_QA_S_GIVEN_W = _tc["COST_QA_S_GIVEN_W"]  # strong answering, evidence from weak retrieval
COST_QA_S_GIVEN_S = _tc["COST_QA_S_GIVEN_S"]  # strong answering, evidence from strong retrieval

def load_jsonl(p): return [json.loads(l) for l in open(p)]

def main():
    for p in (RET_SCORES, QA_SCORES):
        if not os.path.exists(p):
            raise SystemExit(f"missing {p} -- run sft_router5.py for that stage first")
    os.makedirs(OUT, exist_ok=True)
    grid = {g["idx"]: g for g in load_jsonl(GRID_FILE) if g["split"] == "val"}
    ret  = {r["idx"]: r["pW"] for r in load_jsonl(RET_SCORES)}
    qa   = {(r["idx"], r["ret_choice"]): r["pW"] for r in load_jsonl(QA_SCORES)}
    idxs = sorted(set(grid) & set(ret))
    n = len(idxs)
    em   = np.array([grid[i]["em_lenient"] for i in idxs])          # [n,4] WW,WS,SW,SS
    pR   = np.array([ret[i] for i in idxs])
    pQW  = np.array([qa[(i, "W")] for i in idxs])
    pQS  = np.array([qa[(i, "S")] for i in idxs])
    acc_ss = em[:, 3].mean()
    bar = BAR_FRACTION * acc_ss
    print(f"val n={n} | SS ceiling={100*acc_ss:.2f} | bar(95%)={100*bar:.2f}")

    rows = []  # (cost, acc, thetaR, thetaQ)
    thetaR_cuts = np.concatenate([np.unique(pR), [1.01]])  # escalate Ret iff pR < thetaR
    for tR in thetaR_cuts:
        escR  = pR < tR                                     # True -> strong retrieval
        qa_p  = np.where(escR, pQS, pQW)                    # QA score on the branch in use
        acc_w = np.where(escR, em[:, 2], em[:, 0])          # deliver weak answer
        acc_s = np.where(escR, em[:, 3], em[:, 1])          # escalate QA
        # per-instance prices, conditional on the retrieval choice in use
        base_cost = (COST_RET_W + escR * COST_RET_S
                     + np.where(escR, COST_QA_W_GIVEN_S, COST_QA_W_GIVEN_W))
        qa_s_cost = np.where(escR, COST_QA_S_GIVEN_S, COST_QA_S_GIVEN_W)
        order = np.argsort(-qa_p)                           # descending QA score
        aw, as_, bc = acc_w[order], acc_s[order], base_cost[order]
        # k = number of instances delivered weak (QA score >= thetaQ)
        cum_aw = np.concatenate([[0], np.cumsum(aw)])
        cum_as = np.concatenate([[0], np.cumsum(as_)])
        cum_qs = np.concatenate([[0], np.cumsum(qa_s_cost[order])])
        tot_as = cum_as[-1]; tot_bc = bc.sum(); tot_qs = cum_qs[-1]
        sorted_q = qa_p[order]
        ks = np.unique(np.searchsorted(-sorted_q, -np.unique(qa_p), side="right"))
        ks = np.unique(np.concatenate([[0], ks, [n]]))
        for k in ks:
            acc  = (cum_aw[k] + (tot_as - cum_as[k])) / n
            cost = (tot_bc + (tot_qs - cum_qs[k])) / n      # strong-QA spend of the escalated
            thetaQ = sorted_q[k-1] if k > 0 else 1.01       # deliver weak iff qa_p >= thetaQ
            rows.append((cost, acc, float(tR), float(thetaQ)))
    rows = np.array(rows)
    # Pareto frontier (min cost, max acc)
    o = np.lexsort((-rows[:, 1], rows[:, 0]))
    front, best = [], -1
    for r in rows[o]:
        if r[1] > best: front.append(r); best = r[1]
    front = np.array(front)
    print(f"swept {len(rows)} cuts -> frontier of {len(front)} points")

    def cheapest_at(frac):
        ok = front[front[:, 1] >= frac * acc_ss]
        return None if len(ok) == 0 else ok[np.argmin(ok[:, 0])]
    # Kneedle knee on the frontier
    c, a = front[:, 0], front[:, 1]
    cn = (c - c.min()) / max(c.max() - c.min(), 1e-12)
    an = (a - a.min()) / max(a.max() - a.min(), 1e-12)
    dist = an - cn                                          # distance above the y=x chord
    kmax = dist.max()
    cand = np.where(dist >= kmax - 1e-12)[0]
    near = np.where(a >= a[cand[0]] - 0.001)[0]             # ties within 0.1 acc points
    knee = front[cand[np.argmin(c[cand])]]
    bar_row = cheapest_at(BAR_FRACTION)
    primary = ("theta_bar", bar_row) if bar_row is not None else ("theta_knee (FALLBACK)", knee)

    def rec(row):
        return None if row is None else {"cost": round(float(row[0]), 4),
            "val_accuracy": round(100*float(row[1]), 2),
            "theta_ret": round(float(row[2]), 6), "theta_qa": round(float(row[3]), 6)}
    lock = {"rule": "primary=cheapest >= 95% of SS val accuracy (CPT-95); secondary=Kneedle knee; "
                    "fallback=knee if bar unreachable; sealed test reads PRIMARY only, once",
            "ss_val_accuracy": round(100*acc_ss, 2), "bar": round(100*bar, 2),
            "costs_units": {"ret_W": COST_RET_W, "ret_S": COST_RET_S,
                            "qa_W_given_W": COST_QA_W_GIVEN_W, "qa_W_given_S": COST_QA_W_GIVEN_S,
                            "qa_S_given_W": COST_QA_S_GIVEN_W, "qa_S_given_S": COST_QA_S_GIVEN_S,
                            "status": "measured token-aware family (13 token_costs.py)"},
            "primary": primary[0], "theta_bar": rec(bar_row), "theta_knee": rec(knee),
            "context_90": rec(cheapest_at(0.90)), "context_98": rec(cheapest_at(0.98)),
            "max_accuracy_row": rec(front[np.argmax(front[:, 1])]),
            "beats_SS": rec(cheapest_at(1.0)) if (cheapest_at(1.0) is not None and
                        cheapest_at(1.0)[0] < COST_RET_W + COST_RET_S
                                              + COST_QA_W_GIVEN_S + COST_QA_S_GIVEN_S) else None,
            "n_val": n, "frontier_points": len(front)}
    json.dump(lock, open(os.path.join(OUT, "theta_lock.json"), "w"), indent=2)
    np.savetxt(os.path.join(OUT, "frontier.csv"),
               front[:, [0, 1, 2, 3]], delimiter=",", comments="",
               header="cost,val_accuracy,theta_ret,theta_qa")
    print(json.dumps(lock, indent=2))

if __name__ == "__main__":
    main()
