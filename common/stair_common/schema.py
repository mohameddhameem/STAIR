"""The shared results layout: one row per (dataset, split, id, track, policy, seed), saved as CSV or parquet.
Pooling the team's results is a concatenation of these files.

    dataset    hotpotqa | 2wikimultihopqa | cwq | webqsp | medagentbench
    split      a part of splits/<dataset>/ (test for headline numbers)
    id         the question ID as in splits/<dataset>/<split>.txt
    track      systemOne | nonLLM | selfLearnt | jonathan | ...
    policy     the track's name for the setting, e.g. "always-large", "router@0.5", "agenttts-searched"
    menu       A (AgentTTS menu) | B (full ladder) | C (swap)
    seed       run seed (0, 1, 2)
    answer     the final short answer that was scored
    lenient    standard ruler, 0/1 (score.lenient)
    em, f1     official HotpotQA scoring (score.em / score.f1)
    units      total AgentTTS units for the question (cost.units summed over every call)
    units_retrieve, units_answer, units_fuse, units_router   the same split by stage (0 if the stage is absent)
    ms_router  wall-clock ms of the routing decision itself (empty if no router)
    models     the models called, "role=hf_repo@revision;..." as in models.toml
"""

from pathlib import Path

COLUMNS = ["dataset", "split", "id", "track", "policy", "menu", "seed", "answer", "lenient", "em", "f1", "units",
           "units_retrieve", "units_answer", "units_fuse", "units_router", "ms_router", "models"]
DATASETS = {"hotpotqa", "2wikimultihopqa", "cwq", "webqsp", "medagentbench"}
MENUS = {"A", "B", "C"}


UNIT_COLUMNS = ["units", "units_retrieve", "units_answer", "units_fuse", "units_router"]


def validate(df, splits_dir=None) -> None:
    """Raise ValueError if a results table breaks the layout, or names an ID outside its split."""
    import numpy as np
    import pandas as pd

    from stair_common.splits import SPLITS_DIR, load_split

    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    if bad := set(df["dataset"]) - DATASETS:
        raise ValueError(f"unknown datasets: {bad}")
    if bad := set(df["menu"]) - MENUS:
        raise ValueError(f"unknown menus: {bad}")
    if not df["lenient"].isin([0, 1]).all():
        raise ValueError("lenient must be 0/1")
    for c in ["em", "f1"]:
        if not pd.to_numeric(df[c], errors="coerce").between(0, 1).all():
            raise ValueError(f"{c} must be a number in [0, 1]")
    seed = pd.to_numeric(df["seed"], errors="coerce")
    if not (seed.notna() & (seed == seed.round())).all():
        raise ValueError("seed must be an integer")
    for c in UNIT_COLUMNS:  # NaN would slip past the sum check below
        v = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
        if not (np.isfinite(v) & (v >= 0)).all():
            raise ValueError(f"{c} must be finite and >= 0")
    if df.duplicated(["dataset", "split", "id", "track", "policy", "menu", "seed"]).any():
        raise ValueError("duplicate rows for one (dataset, split, id, track, policy, menu, seed)")
    stages = df[UNIT_COLUMNS[1:]].sum(axis=1)
    if ((stages - df["units"]).abs() > 1e-6 * df["units"].abs().clip(lower=1)).any():
        raise ValueError("units must equal the sum of the per-stage units")
    root = Path(splits_dir or SPLITS_DIR)
    for (dataset, split), ids in df.groupby(["dataset", "split"])["id"]:
        if not (root / dataset / f"{split}.txt").exists():
            parts = sorted(p.stem for p in (root / dataset).glob("*.txt"))
            raise ValueError(f"unknown split {dataset}/{split}; its parts are {parts}")
        allowed = set(load_split(dataset, split, root))
        if extra := set(ids.astype(str)) - allowed:
            raise ValueError(f"{len(extra)} ids are not in splits/{dataset}/{split}.txt")
