"""The team results table (stair_common.schema): one row per (dataset, split, id, track, policy, menu, seed), one CSV per
experiment and seed, checked with stair_common.schema.validate before it is written.

    results/common/AL-01-hotpotqa-test-seed0.csv   AgentTTS (searched setting; its paper's best setting)
    results/common/AL-02-hotpotqa-test-seed0.csv   strongest models
    results/common/AL-03 .. AL-06-...              every fixed retriever -> answerer pair, starting and learnt rulebooks, oracle

Rules and probes add no per-question model calls, so units_router = 0 and ms_router is empty. The planner (AgentTTS's
search, rule proposals) runs once per experiment, not per question; its own cost is reported in the experiment notes."""
import os

from .agenttts import wdir
from .common import log, read_json
from .config import workstreams

COLS = None


def _row(cfg, ws, policy, r):
    W = cfg["workstreams"][ws]
    return {"dataset": cfg["data"].get("dataset", "hotpotqa"), "split": "test", "id": r["qid"], "track": "selfLearnt", "policy": policy,
            "menu": W.get("menu", "A"), "seed": int(cfg.get("seed", 0)), "answer": (r.get("answer") or "")[:2000],
            "lenient": int(round(r["lenient"])), "em": float(r.get("EM", r.get("em", 0.0))), "f1": float(r.get("F1", r.get("f1", 0.0))),
            "units": float(r["units"]), "units_retrieve": float(r.get("units_retrieve", 0.0)), "units_answer": float(r.get("units_answer", 0.0)),
            "units_fuse": float(r.get("units_fuse", 0.0)), "units_router": 0.0, "ms_router": None, "models": r.get("models", "")}


def tables(cfg):
    """{exp_id: list of rows} for every experiment whose test results exist."""
    out = {}
    ws1 = read_json(os.path.join(wdir(cfg), "test_results.json")) or []
    for res in ws1:
        for tag in (res["policy"] or "").split(" = "):
            ws, pol = (("ws2", "always-strongest") if tag.startswith("Baseline 2") else
                       ("ws1", "agenttts-searched") if tag.startswith("Baseline 1") else
                       ("ws1", "agenttts-paper-best") if tag.startswith("AgentTTS") else (None, None))
            if ws and ws in cfg["workstreams"]:
                out.setdefault(cfg["workstreams"][ws]["exp_id"], []).extend(_row(cfg, ws, pol, r) for r in res["per_question"])
    for ws in workstreams(cfg, "rules"):
        pp = read_json(os.path.join(cfg["run_dir"], ws, "policy_rows.json"))
        if pp:
            out[cfg["workstreams"][ws]["exp_id"]] = [_row(cfg, ws, pol, r) for pol, rows in pp.items() for r in rows]
    return out


def export(cfg):
    import pandas as pd
    from stair_common.schema import COLUMNS, validate
    d = cfg["paths"]["results_dir"]
    os.makedirs(d, exist_ok=True)
    written = []
    for exp_id, rows in sorted(tables(cfg).items()):
        df = pd.DataFrame(rows, columns=COLUMNS)
        validate(df)
        p = os.path.join(d, f"{exp_id}-{cfg['data'].get('dataset', 'hotpotqa')}-test-seed{cfg.get('seed', 0)}.csv")
        df.to_csv(p, index=False)
        written.append(p)
        log(f"{exp_id}: {len(df)} rows, {df['policy'].nunique()} policies -> {p}")
    return written
