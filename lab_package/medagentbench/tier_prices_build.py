#!/usr/bin/env python3
"""
tier_prices_build.py -- turn tier_prices_v1.json (the INPUTS) into tier_prices_v1_computed.json (C_H, C_F and the
ratio rho = C_F / C_H per task type and price setting) that training6_trainer.py and training6_test_read.py read.
CPU, instant. DRAFT v1.0 2026-11-08 for Jonathan's line-by-line review. No model, no data file is touched.

Arithmetic (one line per quantity, so it can be checked by hand):
    C_H        = minutes / 60 * rate
    rework     = rework_multiplier * C_H
    harm       = p_slip * p_event * sum(cost_of_event components)
    C_F        = rework + harm
    rho        = C_F / C_H                        <- the only number the reward uses, per task type
    threshold  = 1 / (1 + rho)                    <- break-even failure risk above which H beats a model (reporting only)
Expected with the v1.0 inputs: task5  C_H 4.67, C_F lower 7.00 / central 72.00 / operating 657.00, rho 1.50 / 15.43 / 140.79
                               task9  C_H 7.58, C_F lower 11.38 / central 76.38 / operating 661.38, rho 1.50 / 10.07 / 87.21
Usage:  python3 tier_prices_build.py --inputs tier_prices_v1.json --out tier_prices_v1_computed.json
"""
import argparse, hashlib, json


def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P = json.load(open(a.inputs))
    rate = P["human_cost"]["rate_sgd_per_hour"]
    mult = P["failure_cost"]["rework_multiplier"]
    p_slip = P["failure_cost"]["p_slip"]
    out = {"version": "tier_prices_build v1.0 2026-11-08", "inputs_file": a.inputs, "inputs_sha256": sha(a.inputs),
           "currency": P["currency"], "tau_gate": P["tau_gate"], "task_types": {}}
    print(f"{'task':6} {'C_H':>7} | " + " | ".join(f"{s:>9} C_F  rho   thr" for s in P["failure_cost"]["settings"]))
    for task, minutes in P["human_cost"]["minutes"].items():
        C_H = minutes / 60.0 * rate
        rework = mult * C_H
        row = {"minutes": minutes, "rate_sgd_per_hour": rate, "C_H": round(C_H, 4), "rework": round(rework, 4), "settings": {}}
        cells = []
        for name, s in P["failure_cost"]["settings"].items():
            event_cost = sum(s["cost_of_event_sgd"].values())
            harm = p_slip * s["p_event"] * event_cost
            C_F = rework + harm
            rho = C_F / C_H
            row["settings"][name] = {"label": s["label"], "p_slip": p_slip, "p_event": s["p_event"],
                                     "cost_of_event": event_cost, "harm": round(harm, 4),
                                     "C_F": round(C_F, 4), "rho": round(rho, 4),
                                     "handoff_threshold_pfail": round(1.0 / (1.0 + rho), 4)}
            cells.append(f"{C_F:9.2f} {rho:6.2f} {100/(1+rho):5.1f}%")
        out["task_types"][task] = row
        print(f"{task:6} {C_H:7.2f} | " + " | ".join(cells))
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out, "sha256", sha(a.out))


if __name__ == "__main__":
    main()
