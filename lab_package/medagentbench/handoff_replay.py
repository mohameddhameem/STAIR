#!/usr/bin/env python3
"""
handoff_replay.py -- who sees a failure? Human hand-off counts per routing policy, by replay.

QUESTION: for each policy on the MedAgentBench held-out test, how many instances are
  (a) escalated to the strong model,
  (b) escalated AND the strong model then fails  -> in a deployment these go to a person,
  (c) delivered from the weak attempt AND that attempt is wrong -> a wrong answer that no person sees.
Reported for the 30 high-risk instances (tau >= 0.8) and for all 150.

NO model is run. Every number is replayed from committed files:
  training1_table.json         per-instance weak/strong correctness (qw, qs) and costs (cw, cs)
  tau_scores_v2.json           frozen risk scores (rubric v2)
  training1_test_scores.jsonl  Training 1's pDeliver per test instance (single test read, 27 Oct)
  training3_test_scores.jsonl  Training 3's pDeliver per test instance (single test read, 27 Oct)
  training3_table.json         the round-2 weak cost cw2 (Training 3 escalation pays cw2 + cs, not cw + cs)
Training 2's decisions are reconstructed from its registered test read (deliver 15/150 = task 1 only,
63.33 @ 94.5%); the script asserts that reconstruction reproduces 95/150 correct before using it.

POLICIES
  T1          Training 1 (risk-free), deliver iff pDeliver >= 0.50
  T1_lean     Training 1, deliver iff pDeliver >= 0.20
  T2          Training 2 (risk in the reward): deliver task 1, escalate everything else
  T3          Training 3 (round-2 router), deliver iff pDeliver >= 0.50   (reported, footnote only)
  gate0.8     T1 decisions, but tau >= 0.8 forced to strong
  gate0.5     T1 decisions, but tau >= 0.5 forced to strong
  SS          always-strong

USAGE (lab):
  python3 -I handoff_replay.py --results /mnt/hdd/jonathan/STAIR/results/medagentbench \
                               --out /mnt/hdd/jonathan/runs/mab/handoff_replay.json
"""
import argparse, hashlib, json, os


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def load_scores(path):
    return {json.loads(l)["tid"]: json.loads(l)["pDeliver"] for l in open(path)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="dir with the committed results files")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = a.results
    files = {k: os.path.join(R, v) for k, v in {
        "table": "training1_table.json", "tau": "tau_scores_v2.json",
        "t1": "training1_test_scores.jsonl", "t3": "training3_test_scores.jsonl",
        "table3": "training3_table.json"}.items()}

    table = json.load(open(files["table"]))
    tauj = json.load(open(files["tau"]))
    tau = {s["tid"]: s["tau"] for s in (tauj["scores"] if "scores" in tauj else tauj)}
    p1, p3 = load_scores(files["t1"]), load_scores(files["t3"])
    cw2 = {r["tid"]: r["cw2"] for r in json.load(open(files["table3"]))}
    test = [r for r in table if r["split"] == "test"]
    assert len(test) == 150 and all(r["tid"] in p1 and r["tid"] in p3 for r in test)

    policies = {
        "T1":      lambda r: "W" if p1[r["tid"]] >= 0.50 else "S",
        "T1_lean": lambda r: "W" if p1[r["tid"]] >= 0.20 else "S",
        "T2":      lambda r: "W" if r["tid"].startswith("task1_") else "S",
        "T3":      lambda r: "W" if p3[r["tid"]] >= 0.50 else "S",
        "gate0.8": lambda r: "S" if tau[r["tid"]] >= 0.8 else ("W" if p1[r["tid"]] >= 0.50 else "S"),
        "gate0.5": lambda r: "S" if tau[r["tid"]] >= 0.5 else ("W" if p1[r["tid"]] >= 0.50 else "S"),
        "SS":      lambda r: "S",
    }

    def tally(dec, rows, round2_cut=False):
        esc = to_person = delivered_wrong = correct = 0
        cost = 0.0
        for r in rows:
            d = dec(r)
            if d == "S":
                esc += 1; to_person += 0 if r["qs"] else 1
                correct += r["qs"]; cost += (cw2[r["tid"]] if round2_cut else r["cw"]) + r["cs"]
            else:
                delivered_wrong += 0 if r["qw"] else 1
                correct += r["qw"]; cost += r["cw"]
        n = len(rows)
        return {"n": n, "escalated": esc, "strong_failed_to_person": to_person,
                "delivered_wrong_no_person": delivered_wrong,
                "correct": correct, "accuracy_pct": round(100 * correct / n, 2),
                "mean_cost_units": round(cost / n, 1)}

    # reconstruction check for Training 2 (registered test read: 63.33 @ 94.5%)
    t2_all = tally(policies["T2"], test)
    assert t2_all["correct"] == 95, f"T2 reconstruction gives {t2_all['correct']}/150, expected 95"
    ss = tally(policies["SS"], test)["mean_cost_units"]

    high = [r for r in test if tau[r["tid"]] >= 0.8]
    assert len(high) == 30
    out = {"inputs": {k: {"path": v, "sha256": sha(v)} for k, v in files.items()},
           "definition": {"escalated": "router chose S", "strong_failed_to_person": "S chosen and qs == 0",
                          "delivered_wrong_no_person": "W chosen and qw == 0",
                          "cost": "weak-first: W pays cw, S pays cw + cs; % of always-strong uses the all-150 SS mean"},
           "always_strong_mean_cost_150": ss, "policies": {}}
    print(f"{'policy':9s} | high-risk 30: esc  ->person  delivWrong | all 150: esc  ->person  delivWrong | acc%   cost%")
    for name, dec in policies.items():
        h, al = tally(dec, high, name == "T3"), tally(dec, test, name == "T3")
        al["cost_pct_of_SS"] = round(100 * al["mean_cost_units"] / ss, 1)
        out["policies"][name] = {"high_risk_30": h, "all_150": al}
        print(f"{name:9s} | {h['escalated']:14d} {h['strong_failed_to_person']:8d} {h['delivered_wrong_no_person']:11d} | "
              f"{al['escalated']:12d} {al['strong_failed_to_person']:8d} {al['delivered_wrong_no_person']:11d} | "
              f"{al['accuracy_pct']:5.2f} {al['cost_pct_of_SS']:6.1f}")
    json.dump(out, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
