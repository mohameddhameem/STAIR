"""The shared training-history layout and its loss-curve plot: one row per (track, model, run, seed, epoch), saved as
CSV. Pooling the team's curves is a concatenation of these files.

    track       systemOne | nonLLM | selfLearnt | jonathan | ...
    model       what was trained, e.g. "nano-jev judge", "Qwen2.5-1.5B retriever"
    run         the track's name for the setting, e.g. "qpa-large-helps", "lora-r16"
    seed        run seed (0, 1, 2)
    epoch       1, 2, ... (or the evaluation step's fractional epoch)
    train_loss  mean training loss over the epoch
    val_loss    loss on the val split (never test); the early-stopping signal
    step, val_metric   optional: the optimiser step, and a val score such as lenient accuracy

    python -m stair_common.training history.csv [more.csv ...] --out curves.png   (needs the "plot" extra)
"""

import numpy as np
import pandas as pd

HISTORY_COLUMNS = ["track", "model", "run", "seed", "epoch", "train_loss", "val_loss"]
OPTIONAL_COLUMNS = ["step", "val_metric"]
KEY = ["track", "model", "run", "seed"]


def validate_history(df) -> None:
    """Raise ValueError if a history table breaks the layout."""
    missing = [c for c in HISTORY_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if df.duplicated(KEY + ["epoch"]).any():
        raise ValueError("duplicate rows for one (track, model, run, seed, epoch)")
    for c in ["epoch", "train_loss", "val_loss"]:
        if not np.isfinite(pd.to_numeric(df[c], errors="coerce")).all():
            raise ValueError(f"{c} must be finite numbers")
    for key, g in df.groupby(KEY):
        if not g["epoch"].is_monotonic_increasing:
            raise ValueError(f"epochs must increase within {dict(zip(KEY, key))}")


def best_epoch(df) -> pd.DataFrame:
    """The early-stopping point of each run: the row with the lowest val_loss, one per (track, model, run, seed)."""
    df = df.reset_index(drop=True)
    return df.loc[df.groupby(KEY)["val_loss"].idxmin()].reset_index(drop=True)


def from_records(records, track, model, run, seed, val_key="val_loss", train_key="train_loss") -> pd.DataFrame:
    """A list of per-epoch dicts in the layout, e.g. from_records(report["history"], ..., val_key="dev_nll")."""
    df = pd.DataFrame(records).rename(columns={val_key: "val_loss", train_key: "train_loss"})
    for c, v in {"track": track, "model": model, "run": run, "seed": seed}.items():
        df[c] = v
    return df[HISTORY_COLUMNS + [c for c in OPTIONAL_COLUMNS if c in df.columns]]


def plot_history(df, out=None):
    """Train and val loss against epoch, one panel per (track, model, run). Seeds are thin lines and their mean bold;
    a dashed line marks the mean's best epoch, flagged when val loss rises after it. Returns the figure."""
    import matplotlib
    if out is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    validate_history(df)
    panels = list(df.groupby(["track", "model", "run"]))
    cols = min(3, len(panels))
    rows = -(-len(panels) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.4 * rows), squeeze=False)
    for ax, ((track, model, run), g) in zip(axes.flat, panels):
        for name, colour in [("train_loss", "tab:blue"), ("val_loss", "tab:orange")]:
            for _, s in g.groupby("seed"):
                ax.plot(s["epoch"], s[name], color=colour, alpha=0.3, lw=0.8)
            mean = g.groupby("epoch")[name].mean()
            ax.plot(mean.index, mean.values, color=colour, lw=2, label=name.replace("_", " "))
        val = g.groupby("epoch")["val_loss"].mean()
        best = val.idxmin()
        rises = (val.loc[best:] > val.min() * 1.01).any()
        ax.axvline(best, color="grey", ls="--", lw=1)
        ax.set_title(f"{model} · {run}\n{track}, best epoch {best:g}" + (", val rises after" if rises else ""),
                     fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(panels):]:
        ax.set_visible(False)
    fig.tight_layout()
    if out is not None:
        fig.savefig(out, dpi=150)
    return fig


def main(argv=None) -> None:
    import argparse

    p = argparse.ArgumentParser(description="Validate training histories and plot their loss curves.")
    p.add_argument("csv", nargs="+")
    p.add_argument("--out", default="loss-curves.png")
    args = p.parse_args(argv)
    df = pd.concat([pd.read_csv(f) for f in args.csv], ignore_index=True)
    plot_history(df, args.out)
    print(best_epoch(df)[KEY + ["epoch", "val_loss"]].to_string(index=False))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
