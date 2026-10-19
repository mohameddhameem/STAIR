#!/usr/bin/env python3
# =============================================================================
# plot_ablations.py — VERSION: plot_ablations v1.1 2026-09-08 (parsers pinned to real log formats)
#
# Two overlay panels for the evidence spine:
#   A) run2_overlay.png    — v4 weighted (July) vs v4 --no_class_weights:
#                            "training actively destroys judgment; weights only slow it"
#   B) lenient_overlay.png — Run 1 (strict labels) vs Run 1-lenient:
#                            "would fair labels have saved training?"
#
# STRUCTURE (for study): Section 1 parses logs into lists of numbers with
# regular expressions; Section 2 draws. Nothing else.
#
# USAGE (lab, CPU, seconds; panel B needs the lenient run finished):
#   python3 plot_ablations.py \
#     --run1_log  /mnt/hdd/jonathan/validation_g0/run1_v5_full.log \
#     --lenient_log /mnt/hdd/jonathan/validation_g0/run1_lenient.log \
#     --nocw_log  /mnt/hdd/jonathan/validation_g0/run2_nocw.log \
#     --weighted_log /mnt/hdd/jonathan/validation_g0/run_v4_weighted.log \
#     --out /mnt/hdd/jonathan/runs/plots
# (any panel whose two logs are present gets drawn; missing ones are skipped)
# =============================================================================
import os, re, argparse

VERSION = "plot_ablations v1.1 2026-09-08 (parsers pinned to real log formats)"

# ---------------- Section 1: parsing (the new skill) -------------------------
# A training log is just text. re.findall(pattern, text) returns every match;
# each (...) group in the pattern becomes one string in the result tuple.

def read(path):
    """Whole log file as one string ('' if absent, so panels can be skipped)."""
    return open(path).read() if path and os.path.exists(path) else ""


def parse_probe_acc(text, stage="ret"):
    """Real v5 line (both stages on ONE line):
      [probe @ 25] ret loss 0.6334 acc 0.665 | qa loss 0.862 acc 0.63 | chance ...
    -> for stage='ret' return ([25,...],[0.665,...]); for 'qa' the second pair."""
    pairs = re.findall(r"\[probe @ (\d+)\] ret loss [\d.]+ acc ([\d.]+) \| qa loss [\d.]+ acc ([\d.]+)", text)
    steps = [int(s) for s, _, _ in pairs]
    accs = [float(r if stage == "ret" else q) for _, r, q in pairs]
    return steps, accs


def parse_loss(text, stage="ret"):
    """Real v5 line:  [ret ep0] opt_step 5/246 loss 0.3870"""
    pairs = re.findall(r"\[" + stage + r" ep\d+\] opt_step (\d+)/\d+ loss ([\d.]+)", text)
    return [int(s) for s, _ in pairs], [float(v) for _, v in pairs]


def parse_v4_loss(text):
    """Real v4 line:  ep0 opt_step 5/146 loss 1.0857 lr 1.43e-05"""
    pairs = re.findall(r"ep\d+ opt_step (\d+)/\d+ loss ([\d.]+)", text)
    return [int(s) for s, _ in pairs], [float(v) for _, v in pairs]


# ---------------- Section 2: drawing -----------------------------------------
def draw_overlay(ax, series, title, ylab, hlines=()):
    """series: list of (steps, values, label). Two ax.plot calls on ONE axes =
    an overlay; identical axes = honest comparison. hlines: (y, text) refs."""
    for steps, vals, label in series:
        if steps:
            ax.plot(steps, vals, marker="o", ms=3.5, lw=1.4, label=label)
    for y, txt in hlines:
        ax.axhline(y=y, ls="--", lw=0.9, color="grey")
        ax.annotate(txt, xy=(0.99, y), xycoords=("axes fraction", "data"),
                    ha="right", va="bottom", fontsize=7.5, color="grey")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("optimizer step", fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.legend(fontsize=8)
    ax.tick_params(labelsize=8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run1_log"); ap.add_argument("--lenient_log")
    ap.add_argument("--nocw_log"); ap.add_argument("--weighted_log")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print(VERSION)
    import matplotlib
    matplotlib.use("Agg")                       # no display on the lab box
    import matplotlib.pyplot as plt

    # ---- Panel A: v4 weighted vs no-weights (loss) --------------------------
    w, n = read(a.weighted_log), read(a.nocw_log)
    if w and n:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        sw, lw_ = parse_v4_loss(w)
        sn, ln_ = parse_v4_loss(n)
        draw_overlay(ax, [(sw, lw_, "v4 weighted (July)"),
                          (sn, ln_, "v4 --no_class_weights (Run 2)")],
                     "v4 ablation — the loss both runs minimize, one restraint apart",
                     "weighted CE loss",
                     hlines=[(1.386, "chance ln 4")])
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "run2_overlay.png"))
        print("wrote run2_overlay.png")
    else:
        print("panel A skipped (need both v4 logs)")

    # ---- Panel B: Run 1 vs Run 1-lenient (retrieval probe accuracy) ---------
    r1, rl = read(a.run1_log), read(a.lenient_log)
    if r1 and rl:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
        s1, a1 = parse_probe_acc(r1, "ret")
        s2, a2 = parse_probe_acc(rl, "ret")
        draw_overlay(ax, [(s1, a1, "Run 1 — Label-EM (strict)"),
                          (s2, a2, "Run 1-lenient — Label-Lenient v1.2")],
                     "Same trainer, only the labels differ — retrieval probe accuracy",
                     "probe accuracy (val, n=400)",
                     hlines=[(0.680, "majority (strict labels) 0.680"),
                             (0.688, "majority (lenient labels) 0.688")])
        fig.tight_layout(); fig.savefig(os.path.join(a.out, "lenient_overlay.png"))
        print("wrote lenient_overlay.png")
    else:
        print("panel B skipped (need both v5 logs — lenient run must be finished)")


if __name__ == "__main__":
    main()
