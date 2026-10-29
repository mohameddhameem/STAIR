#!/usr/bin/env python3
"""
training5_board_replay.py -- every Training 5 number on the MedAgentBench results board, replayed from the
committed files. No model is run. Run it to check the board, or to re-derive the numbers under another
human-cost convention (--k).

INPUTS (results/medagentbench/, all committed):
  training1_table.json             per-instance weak/strong correctness (qw, qs) and costs (cw, cs), test split
  tau_scores_v2.json               frozen risk scores (rubric v2); high-risk tier = tau >= 0.8 (30 test instances)
  training5a_test_scores.jsonl     5a router probabilities on the test 150 (pDeliver, pEscalate, pHuman on the tier)
  training5b_test_scores.jsonl     5b router probabilities

THE PIPELINE (weak-first): the weak model always attempts first. The router then reads the task text and the
weak attempt and chooses ONE action -- on high-risk instances W (deliver), S (escalate = run the strong model),
H (route to a person); elsewhere W or S only. On H the strong model is NOT run.

REPORTING CONVENTIONS (decided 29 Oct 2026, Jonathan; registration entry of the same date):
  accuracy  = AI + human execution: an instance routed to a person counts as completed correctly
              (the person is assumed to answer correctly). --no-person scores it as if the hand-off fell back
              to the strong model instead (the strong model's real result).
  cost      = AI cost + human cost, as % of always-strong (weak attempt + strong call on all 150).
              deliver = cw; escalate = cw + cs; routed to a person = cw + C_H.
  human cost C_H = k x cs of that instance (one human execution per hand-off). k = 1 on the board = the floor:
              a person costs no less than the strong call it replaces. No real price is measured; in a real
              environment use human time x loaded rate in the same unit. --k changes the convention.
  operating point: greedy (argmax over allowed actions) is the primary point; --theta-h THETA reads the
              secondary point "route to a person iff P(H) >= THETA, else the greedy W/S choice".
  the router NEVER sees k: it was trained and read without any human price (training5_trainer.py v1.3c).

USAGE:
  python3 -I training5_board_replay.py --results results/medagentbench            # board rows, k = 1
  python3 -I training5_board_replay.py --results results/medagentbench --k 1.5    # sensitivity
  python3 -I training5_board_replay.py --results results/medagentbench --theta-h 0.8 --variant 5b
"""
import argparse, json, os

TAU_GATE = 0.80


def load(results):
    tab = {t["tid"]: t for t in json.load(open(os.path.join(results, "training1_table.json"))) if t["split"] == "test"}
    tj = json.load(open(os.path.join(results, "tau_scores_v2.json")))
    tau = {s["tid"]: s["tau"] for s in (tj["scores"] if "scores" in tj else tj)}
    assert len(tab) == 150 and sum(tau[t] >= TAU_GATE for t in tab) == 30
    return tab, tau


def decisions(results, variant, theta):
    rows = [json.loads(l) for l in open(os.path.join(results, f"training{variant}_test_scores.jsonl"))]
    assert len(rows) == 150
    dec = {}
    for r in rows:
        if r["high"]:
            pr = [r["pDeliver"], r["pEscalate"], r["pHuman"]]
            if theta is None:
                dec[r["tid"]] = "WSH"[max(range(3), key=lambda i: pr[i])]
            else:
                dec[r["tid"]] = "H" if pr[2] >= theta else ("W" if pr[0] >= pr[1] else "S")
        else:
            dec[r["tid"]] = "W" if r["pDeliver"] >= r["pEscalate"] else "S"
    return dec


def tally(tab, tau, dec, k, no_person=False):
    ss = sum(t["cw"] + t["cs"] for t in tab.values())
    acc = ai = hum = 0.0
    hi = {"to_person": 0, "needed": 0, "unnecessary": 0, "escalated": 0, "delivered": 0, "fail_no_human": 0}
    for tid, t in tab.items():
        a = dec[tid]; high = tau[tid] >= TAU_GATE
        if a == "W":
            q, c_ai, c_h = t["qw"], t["cw"], 0.0
        elif a == "S":
            q, c_ai, c_h = t["qs"], t["cw"] + t["cs"], 0.0
        else:  # routed to a person: strong model not run
            q = t["qs"] if no_person else 1
            c_ai, c_h = t["cw"], k * t["cs"]
        acc += q; ai += c_ai; hum += c_h
        if high:
            if a == "H":
                hi["to_person"] += 1
                both_fail = (t["qw"] == 0 and t["qs"] == 0)
                hi["needed"] += both_fail; hi["unnecessary"] += (not both_fail)
            else:
                hi["escalated" if a == "S" else "delivered"] += 1
                hi["fail_no_human"] += (1 - q)
    return {"accuracy_pct": round(100 * acc / 150, 2),
            "ai_cost_pct": round(100 * ai / ss, 1), "human_cost_pct": round(100 * hum / ss, 1),
            "total_cost_pct": round(100 * (ai + hum) / ss, 1), "high_risk_30": hi}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--k", type=float, default=1.0, help="human cost per hand-off as a multiple of that instance's strong-call cost")
    ap.add_argument("--theta-h", type=float, default=None, dest="theta", help="secondary operating point: route to a person iff P(H) >= theta")
    ap.add_argument("--variant", choices=["5a", "5b", "both"], default="both")
    a = ap.parse_args()
    tab, tau = load(a.results)
    point = "greedy (primary)" if a.theta is None else f"P(H) >= {a.theta:g}"
    print(f"human cost k = {a.k:g} x strong call | operating point: {point}")
    print(f"{'router':6s} {'acc%':>7s} {'AI%':>6s} {'human%':>7s} {'total%':>7s} | toPerson needed unnec escal deliv failNoHuman | acc% if no person")
    for v in (["5a", "5b"] if a.variant == "both" else [a.variant]):
        dec = decisions(a.results, v, a.theta)
        r = tally(tab, tau, dec, a.k); n = tally(tab, tau, dec, a.k, no_person=True)
        h = r["high_risk_30"]
        print(f"{v:6s} {r['accuracy_pct']:7.2f} {r['ai_cost_pct']:6.1f} {r['human_cost_pct']:7.1f} {r['total_cost_pct']:7.1f} | "
              f"{h['to_person']:8d} {h['needed']:6d} {h['unnecessary']:5d} {h['escalated']:5d} {h['delivered']:5d} {h['fail_no_human']:11d} | {n['accuracy_pct']:.2f}")


if __name__ == "__main__":
    main()
