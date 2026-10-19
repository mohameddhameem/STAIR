#!/usr/bin/env python3
# =============================================================================
# mutation_kit.py — VERSION: mutation_kit v1.1 2026-09-15 (M2 indent fixed to probe depth)
#
# AVENUE #2: prove the validation checks are SENSITIVE, not blind.
# Creates 4 mutants of overfit_probe.py (exact-match source edits; loud
# failure if the source drifted), runs each SHORT (default 2 epochs) plus an
# unmutated baseline M0, then applies the six-checks machinery to every
# run's probe_steps.csv and prints a mutant-by-check capture table.
#
# MUTANTS (bug -> designated detector):
#   M0 baseline (no edit)            -> every check must PASS
#   M1 label->index map swapped      -> C5 loss-formula (+C6 direction)
#   M2 loss on p(predicted)          -> C5 loss-formula  [Jonathan's 13-Sep
#                                       hand experiment, planted for real]
#   M3 labels shuffled vs instances  -> C2 label lookup vs jsonl
#   M4 renormalization broken        -> C3 sum-to-one
#
# CHECKS (per data row of probe_steps.csv):
#   C2 printed label == jsonl ret_label for that idx
#   C3 |p_W + p_S - 1| <= 1e-3
#   C5 |loss - (-ln p(label))| <= 5e-3   (skipped when p(label) < 1e-3)
#   C6 learning direction: mean p(label) rises epoch 1 -> last (info)
#
# USAGE (lab, GPU, ~20-25 min total for 5 short runs):
#   python3 mutation_kit.py --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#       --workdir /mnt/hdd/jonathan/runs/mutation_kit --epochs 2
# =============================================================================
import os, sys, json, csv, math, argparse, subprocess

VERSION = "mutation_kit v1.1 2026-09-15 (M2 indent fixed to probe depth)"

MUTS = {
    "M1": ("label->index map swapped (A/B readout inverted)", "C5",
           'y = 0 if labels_all[i] == "W" else 1',
           'y = 1 if labels_all[i] == "W" else 0   # MUTANT M1'),
    "M2": ("loss on p(predicted) instead of p(label)", "C5",
           "loss = -logp[y]",
           "pred_ix = 0 if logp[0] >= logp[1] else 1\n            loss = -logp[pred_ix]   # MUTANT M2"),
    "M3": ("labels shuffled against instances (rotation by 1)", "C2",
           "labels_all = dict(labels)",
           "labels_all = dict(labels)\n    _ks = sorted(labels_all); _vs = [labels_all[k] for k in _ks]\n    labels_all = {k: _vs[(_j + 1) % len(_vs)] for _j, k in enumerate(_ks)}   # MUTANT M3"),
    "M4": ("renormalization broken (logsigmoid, pair no longer sums to 1)", "C3",
           "logp = torch.log_softmax(logits2.float(), dim=-1)",
           "logp = torch.nn.functional.logsigmoid(logits2.float())   # MUTANT M4"),
}


def make_mutant(src_text, name):
    if name == "M0":
        return src_text
    _, _, old, new = MUTS[name]
    n = src_text.count(old)
    assert n == 1, f"{name}: target line found {n} times (expected exactly 1) — probe source drifted, refuse to mutate"
    out = src_text.replace(old, new)
    return out.replace('VERSION = "overfit_probe', f'VERSION = "MUTANT-{name} overfit_probe')


def run_one(name, probe_path, a):
    out = os.path.join(a.workdir, name)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, probe_path, "--data", a.data, "--out", out,
           "--epochs", str(a.epochs), "--holdout_n", "0"]
    print(f"\n=== {name}: running {' '.join(cmd[1:])}", flush=True)
    r = subprocess.run(cmd, capture_output=True, text=True)
    tail = (r.stdout + r.stderr).strip().splitlines()[-3:]
    for t in tail:
        print("   |", t)
    return os.path.join(out, "probe_steps.csv"), r.returncode


def check_csv(path, ret_labels):
    res = {"C2": 0, "C3": 0, "C5": 0, "rows": 0, "C6": None}
    by_epoch = {}
    if not os.path.exists(path):
        return None
    for row in csv.DictReader(open(path)):
        if row["kind"] != "step":
            continue
        res["rows"] += 1
        i = int(row["idx"]); lab = row["label"]
        pW = float(row["p_W"]); pS = float(row["p_S"]); loss = float(row["loss"])
        if lab != ret_labels.get(i):
            res["C2"] += 1
        if abs(pW + pS - 1.0) > 1e-3:
            res["C3"] += 1
        p_lab = pW if lab == "W" else pS
        if p_lab >= 1e-3 and abs(loss - (-math.log(p_lab))) > 5e-3:
            res["C5"] += 1
        by_epoch.setdefault(int(row["epoch"]), []).append(p_lab)
    if len(by_epoch) >= 2:
        e = sorted(by_epoch)
        res["C6"] = (sum(by_epoch[e[-1]]) / len(by_epoch[e[-1]])) - (sum(by_epoch[e[0]]) / len(by_epoch[e[0]]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--probe", default="overfit_probe.py")
    ap.add_argument("--workdir", default="/mnt/hdd/jonathan/runs/mutation_kit")
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--only", default="", help="comma list, e.g. M1,M3 (default: M0..M4)")
    a = ap.parse_args()
    print(VERSION, flush=True)

    ret_labels = {}
    for l in open(a.data):
        r = json.loads(l)
        ret_labels[r["idx"]] = r.get("ret_label")

    src = open(a.probe, encoding="utf-8").read()
    names = [n.strip() for n in a.only.split(",") if n.strip()] or ["M0", "M1", "M2", "M3", "M4"]
    os.makedirs(a.workdir, exist_ok=True)

    results = {}
    for name in names:
        mpath = os.path.join(a.workdir, f"probe_{name}.py")
        open(mpath, "w", encoding="utf-8").write(make_mutant(src, name))
        csv_path, rc = run_one(name, mpath, a)
        res = check_csv(csv_path, ret_labels)
        results[name] = (res, rc)

    print("\n================ CAPTURE TABLE ================")
    print(f"{'mutant':6s} {'rows':>4s} {'C2 label':>9s} {'C3 sum1':>8s} {'C5 loss':>8s} {'C6 dP(label)':>12s}  verdict")
    all_ok = True
    for name in names:
        res, rc = results[name]
        if res is None or res["rows"] == 0:
            print(f"{name:6s}  run failed (rc={rc}) — a crash also counts as caught for a planted bug" )
            if name == "M0":
                all_ok = False
            continue
        c6 = "n/a" if res["C6"] is None else f"{res['C6']:+.3f}"
        flags = {k: res[k] for k in ("C2", "C3", "C5")}
        if name == "M0":
            ok = all(v == 0 for v in flags.values()) and (res["C6"] is None or res["C6"] > 0)
            verdict = "CLEAN (control passes)" if ok else "CONTROL FAILED — investigate kit"
            all_ok &= ok
        else:
            det = MUTS[name][1]
            caught = flags.get(det, 0) > 0 or (name == "M1" and res["C6"] is not None and res["C6"] < 0)
            verdict = f"CAUGHT by {det}" if caught else "NOT CAUGHT — checks are blind to this bug!"
            all_ok &= caught
        print(f"{name:6s} {res['rows']:4d} {res['C2']:9d} {res['C3']:8d} {res['C5']:8d} {c6:>12s}  {verdict}")
    print("\nVERDICT:", "SENSITIVITY CERTIFIED — every planted bug caught, control clean"
          if all_ok else "GAP FOUND — see table")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
