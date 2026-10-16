"""Run the self-learnt switching pipeline, stage by stage.

  python run.py --config configs/default.yaml --stage <stage> [--retriever r05] [--split dev] [--model small]

Stages (in order; each is resumable, so a job that stops can simply be submitted again):
  setup     download the models (if download_models: true) and the HotpotQA files
  data      build the question sets: train / val / dev (HotpotQA training file) and test (HotpotQA dev file)
  train     fine-tune one retriever (--retriever KEY) or all of them, then merge the best checkpoint
  retrieve  shortlists from every retriever on dev and test, and the AgentTTS baseline retrieval on test
  reach     list every shortlist a rulebook can end on
  answer    small and big answerer on every reachable shortlist (--model small|big, default both)
  sample    sampled big-answerer answers on every reachable shortlist (the votes)
  learn     micro-experiments on the dev questions: learn the rulebook (the planner model runs here)
  test      the frozen rulebook once on the test questions; results/summary.md
  all       everything above, in order
"""
import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stair_switch import data as D           # noqa: E402
from stair_switch import generate as G       # noqa: E402
from stair_switch import rulebook as RB      # noqa: E402
from stair_switch.common import log, write_json  # noqa: E402

STAGES = ["setup", "data", "train", "retrieve", "reach", "answer", "sample", "learn", "test"]


def _interp(obj, env):
    if isinstance(obj, str):
        for _ in range(5):
            new = re.sub(r"\$\{(\w+)\}", lambda m: str(env.get(m.group(1), m.group(0))), obj)
            if new == obj:
                break
            obj = new
        return obj
    if isinstance(obj, dict):
        return {k: _interp(v, env) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interp(v, env) for v in obj]
    return obj


def load_config(path):
    import yaml
    cfg = yaml.safe_load(open(path))
    env = {"run_name": cfg["run_name"]}
    env.update({k: v for k, v in cfg["paths"].items() if isinstance(v, str)})
    env["project_dir"] = cfg["paths"]["project_dir"]
    for _ in range(3):
        cfg["paths"] = _interp(cfg["paths"], env)
        env.update(cfg["paths"])
    cfg = _interp(cfg, env)
    cfg["run_dir"] = os.path.join(cfg["paths"]["runs_dir"], cfg["run_name"])
    models = {}
    for k, m in cfg["models"].items():
        models[k] = dict(m, key=k, params=float(m["params"]), path=os.path.join(cfg["paths"]["models_dir"], m["dir"]))
    cfg["models"] = models
    rets = []
    for r in cfg["retrievers"]:
        m = models[r["model"]]
        rets.append(dict(r, label=m["label"], params=m["params"], base_path=m["path"], no_thinking=m.get("no_thinking", False),
                         adapter_dir=os.path.join(cfg["run_dir"], "adapters", r["key"]), merged_dir=os.path.join(cfg["run_dir"], "merged", r["key"]),
                         path=os.path.join(cfg["run_dir"], "merged", r["key"])))
    cfg["retrievers"] = rets
    b = cfg["baseline"]
    cfg["baseline"] = dict(models[b["model"]], key=b["key"], model_key=b["model"], label=b.get("label", models[b["model"]]["label"]))
    for role in ("small", "big"):
        a = cfg["answerers"][role]
        cfg["answerers"][role] = dict(models[a["model"]])
    for k, v in cfg["rules"]["hard"].items():
        cfg["rules"]["hard"][k] = float(v)
    return cfg


def stage_setup(cfg):
    os.environ.setdefault("HF_HOME", cfg["paths"]["hf_cache"])
    need = {r["model"] for r in cfg["retrievers"]} | {cfg["baseline"]["model_key"], cfg["answerers"]["small"]["key"], cfg["answerers"]["big"]["key"], cfg["planner"]}
    for k in sorted(need):
        m = cfg["models"][k]
        if os.path.exists(os.path.join(m["path"], "config.json")):
            log(f"{k}: present ({m['path']})")
        elif cfg["download_models"] and cfg["backend"] != "mock":
            from huggingface_hub import snapshot_download
            log(f"{k}: downloading {m['hf_id']} ...")
            snapshot_download(repo_id=m["hf_id"], local_dir=m["path"], allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py", "tokenizer*"])
        else:
            log(f"{k}: MISSING - copy {m['hf_id']} into {m['path']}" if cfg["backend"] != "mock" else f"{k}: not needed for a mock run")


def stage_train(cfg, only):
    from stair_switch import train as T
    tr, va = D.load(cfg, "train"), D.load(cfg, "val")
    for r in cfg["retrievers"]:
        if only and r["key"] != only:
            continue
        if not T.trained(r):
            done = T.train(cfg, r, tr, va, time_limit_min=float(os.environ.get("TIME_LIMIT_MIN", 0)))
            if not done:
                sys.exit(3)               # not finished: resubmit the job
        T.merge(cfg, r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=STAGES + ["all"])
    ap.add_argument("--retriever", default="")
    ap.add_argument("--split", default="", choices=["", "dev", "test"])
    ap.add_argument("--model", default="", choices=["", "small", "big"])
    a = ap.parse_args()
    cfg = load_config(a.config)
    os.makedirs(cfg["run_dir"], exist_ok=True)
    write_json(os.path.join(cfg["run_dir"], "config_resolved.json"), cfg)
    t0 = time.time()
    stages = STAGES if a.stage == "all" else [a.stage]
    splits = [a.split] if a.split else ["dev", "test"]
    for st in stages:
        log(f"=== stage {st} ===")
        if st == "setup":
            stage_setup(cfg)
        elif st == "data":
            D.build(cfg)
        elif st == "train":
            stage_train(cfg, a.retriever)
        elif st == "retrieve":
            for sp in splits:
                for r in cfg["retrievers"]:
                    if not a.retriever or r["key"] == a.retriever:
                        G.retrieve(cfg, sp, r)
                if sp == "test" and not a.retriever:
                    G.retrieve(cfg, "test", cfg["baseline"], baseline=True)
        elif st == "reach":
            for sp in splits:
                G.reach(cfg, sp)
        elif st == "answer":
            for role in ([a.model] if a.model else ["small", "big"]):
                G.answer(cfg, splits, cfg["answerers"][role])
        elif st == "sample":
            G.sample(cfg, splits, cfg["answerers"]["big"], int(cfg["answerers"]["samples"]))
        elif st == "learn":
            RB.learn(cfg)
        elif st == "test":
            RB.test(cfg)
    log(f"done in {(time.time() - t0) / 60:.1f} min")
    times = os.path.join(cfg["run_dir"], "stage_times.jsonl")
    with open(times, "a") as f:
        f.write(json.dumps({"stage": a.stage, "retriever": a.retriever, "split": a.split, "model": a.model, "minutes": round((time.time() - t0) / 60, 2),
                            "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


if __name__ == "__main__":
    main()
