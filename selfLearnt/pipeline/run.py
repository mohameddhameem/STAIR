"""Run the five workstreams, stage by stage. Every stage resumes where it stopped: submit the same job again.

  python run.py --config configs/default.yaml --stage <stage> [--part P] [--model M] [--ws W]

Stages, in the order the job chain runs them (WS1-2 first, then the fine-tuning for WS4, then WS3-5):
  setup       download models (only the ones a model/part needs with --model) and the HotpotQA files
  data        question sets (search / test from the dev file as AgentTTS; train / val / dev from the training file)
  ws1_grid    WS1: every AgentTTS setting on the 50 search questions   (--part retrieve-q7 | ... | answer-l70; default all)
  ws1_search  WS1: AgentTTS's planner search over that table (Qwen 72B as planner)
  ws1_test    WS1-2: the chosen setting, AgentTTS's reported best and the strongest models on the 500 test questions (--part as ws1_grid)
  train       fine-tune one model (--model q7) or every model the workstreams need
  retrieve    WS3-5: shortlists of one retriever variant (--model q7:ft) or all
  reach       WS3-5: every shortlist a rulebook can end on (CPU, seconds)
  answer      WS3-5: answers of one answerer variant (--model l70:ft) or all
  probe       WS3: hidden states (--part ret-q7 | ans-l3 | ...) and probe fitting (--part fit)
  sample      WS3-5: sampled answers for votes (--model ...)
  learn       WS3-5: micro-experiments on the dev questions (--ws ws4, or all)
  test        WS3-5: the learnt rulebook once on the test questions (--ws ..., or all)
  report      comparison of all experiments, charts, status page, and the team results tables (results/common/)
  export      only the team results tables: one CSV per experiment and seed, checked by stair_common.schema
  status      rebuild the status page and push it to GitHub now
  all         everything above in order (the CPU dry run)
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stair import data as D                              # noqa: E402
from stair.common import log, write_json                 # noqa: E402
from stair.config import ft_keys, load_config, variant, workstreams  # noqa: E402
from stair.progress import is_done, mark_done, maybe_push  # noqa: E402

ORDER = ["setup", "data", "ws1_grid", "ws1_search", "ws1_test", "train", "retrieve", "reach", "answer", "probe", "sample", "learn", "test", "report"]


def needed_downloads(cfg, only=None):
    """{local path: hf repo id} for every model copy the configured workstreams use."""
    out = {}
    W = cfg["workstreams"]

    def add_inf(v):
        m = variant(cfg, v)
        if not m["finetuned"] or m["ft"]["method"] == "qlora":
            out[m["path"]] = (m["hf_id"], m.get("revision"))
        if m["finetuned"]:
            out[m["train_path"]] = (m.get("train_hf_id", m["hf_id"]), None if m.get("train_hf_id") else m.get("revision"))
    for k, w in W.items():
        if w["kind"] == "agenttts":
            if w.get("skip_search"):        # AL-02 alone: only the strongest pair (added below)
                continue
            for v in w["retrievers"] + w["answerers"]:
                add_inf(v)
        elif w["kind"] == "strong":
            add_inf(w["config"][0])
            add_inf(w["config"][2])
        else:
            for v in w["retrievers"] + w["answerers"]:
                add_inf(v)
    if any(w["kind"] != "strong" and not w.get("skip_search") for w in W.values()):
        add_inf(cfg["planner"])          # AgentTTS's search and the rule proposals
    if only:
        keep = {variant(cfg, only)["path"], variant(cfg, only)["train_path"]}
        out = {p: r for p, r in out.items() if p in keep}
    return out


def stage_setup(cfg, only):
    """Downloads every model copy and the HotpotQA files. One failure (e.g. a gated Llama licence not yet approved) does not
    stop the others; the failures are listed at the end. Re-running resumes: finished files are checked, not re-downloaded."""
    # Only the download cache moves to project storage (HF_HUB_CACHE); HF_HOME stays, so the login token
    # (~/.cache/huggingface/token, from `hf auth login`) is still found. Moving HF_HOME hid it: gated Llama downloads failed.
    os.environ.setdefault("HF_HUB_CACHE", cfg["paths"]["hf_cache"])
    if not os.environ.get("HF_TOKEN"):
        tok = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "token")
        if os.path.exists(tok):
            os.environ["HF_TOKEN"] = open(tok).read().strip()
        else:
            log("no Hugging Face token found (run `hf auth login`): the gated Llama models will fail")
    todo = needed_downloads(cfg, only)
    failed = []
    for path, (repo, rev) in sorted(todo.items(), key=lambda x: x[1][0]):
        if cfg["backend"] == "mock":
            log(f"(mock run) would download {repo}@{rev}")
        elif cfg["download_models"]:
            from huggingface_hub import snapshot_download
            log(f"downloading (or checking) {repo}@{rev} -> {path} ...")
            try:
                snapshot_download(repo_id=repo, revision=rev, local_dir=path,
                                  allow_patterns=["*.json", "*.safetensors", "*.bin", "*.jinja", "*.txt", "*.model", "*.py", "tokenizer*"])
                log(f"ok: {repo}")
            except Exception as e:  # noqa: BLE001
                msg = str(e).splitlines()[0][:200]
                hint = " -> accept the licence on huggingface.co and wait for approval" if any(x in str(e) for x in ("403", "gated", "401", "restricted")) else ""
                log(f"FAILED: {repo}: {type(e).__name__}: {msg}{hint}")
                failed.append(f"{repo}{hint}")
        elif not os.path.exists(os.path.join(path, "config.json")):
            log(f"MISSING: {repo} -> copy it into {path}")
            failed.append(repo)
    files = [(cfg["data"]["hotpot_dev"], cfg["data"].get("hotpot_dev_url"))]
    if cfg["data"].get("train_source", "train") != "dev":
        files.append((cfg["data"]["hotpot_train"], cfg["data"].get("hotpot_train_url")))
    for split_file, url in files:
        if not os.path.exists(split_file):
            try:
                D._load_hotpot(split_file, url)
            except Exception as e:  # noqa: BLE001
                log(f"FAILED: {url}: {e}")
                failed.append(url)
    if failed:
        log("NOT DONE - run setup again after fixing these:\n  " + "\n  ".join(failed))
        sys.exit(1)
    log("all models and data present")
    if not only:
        mark_done(cfg, "setup")


def _parts_or(arg, parts):
    return [arg] if arg else parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stage", required=True, choices=ORDER + ["status", "export", "all"])
    ap.add_argument("--part", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--ws", default="")
    ap.add_argument("--exp", default=None, help="run only one experiment's workstream (AL-01 .. AL-06); outputs are shared")
    ap.add_argument("--seed", type=int, default=None, help="run seed (team standard: 0, 1, 2); default: the config's")
    ap.add_argument("--force", action="store_true", help="re-run a stage part even if marked done")
    a = ap.parse_args()
    cfg = load_config(a.config, seed=a.seed, exp=a.exp)
    os.makedirs(cfg["run_dir"], exist_ok=True)
    write_json(os.path.join(cfg["run_dir"], "config_resolved.json"), cfg)
    t0 = time.time()
    stages = ORDER if a.stage == "all" else [a.stage]
    from stair import agenttts as AT
    from stair import probe as PB
    from stair import rulebook as RB
    from stair import states as S
    from stair.report import final_report, stage_parts
    has = cfg["workstreams"]
    for st in stages:
        log(f"=== stage {st} ===")
        if st == "setup":
            stage_setup(cfg, a.model)
        elif st == "data":
            D.build(cfg)
            mark_done(cfg, "data")
        elif st in ("ws1_grid", "ws1_test"):
            if "ws1" not in has:
                continue
            if st == "ws1_test" and not os.path.exists(os.path.join(AT.wdir(cfg), "search_result.json")):
                log("WS1 search not done yet: the test settings would miss the chosen one")
                if "ws2" not in has:
                    continue
            parts = _parts_or(a.part, AT.all_parts(cfg, st))
            for p in parts:   # test parts always re-check: another experiment may add a setting (saved work is skipped)
                if a.force or st == "ws1_test" or not is_done(cfg, st, p):
                    AT.run_part(cfg, st, p)
            if st == "ws1_grid":
                AT.grid_table(cfg)
        elif st == "ws1_search":
            if "ws1" in has:
                AT.search(cfg)
        elif st == "train":
            from stair import train as T
            tr_q, va_q = D.load(cfg, "train"), D.load(cfg, "val")
            for k in _parts_or(a.model, ft_keys(cfg)):
                if is_done(cfg, "train", k) and not a.force:
                    continue
                if not T.trained(cfg, k):
                    if not T.train(cfg, k, tr_q, va_q, time_limit_min=float(os.environ.get("TIME_LIMIT_MIN", 0))):
                        maybe_push(cfg, force=True)
                        sys.exit(3)             # not finished: submit the job again
                T.merge(cfg, k)
        elif st == "retrieve":
            for v in _parts_or(a.model, S.ret_variants(cfg)):
                if a.force or not is_done(cfg, "retrieve", v):
                    S.retrieve(cfg, v)
        elif st == "reach":
            if workstreams(cfg, "rules"):
                S.reach(cfg, a.ws or None)
        elif st == "answer":                    # always checks for new shortlists (a later workstream may add some)
            for v in _parts_or(a.model, S.ans_variants(cfg)):
                S.answer(cfg, v)
        elif st == "probe":
            parts = _parts_or(a.part, stage_parts(cfg, "probe"))
            for p in parts:
                if p == "fit":
                    PB.fit(cfg)
                else:
                    role, _, v = p.partition("-")
                    PB.extract(cfg, v, role)
        elif st == "sample":
            for v in _parts_or(a.model, stage_parts(cfg, "sample")):
                S.sample(cfg, v)
        elif st == "learn":
            for ws in _parts_or(a.ws, workstreams(cfg, "rules")):
                if a.force or not is_done(cfg, "learn", ws):
                    RB.learn(cfg, ws)
        elif st == "test":
            for ws in _parts_or(a.ws, workstreams(cfg, "rules")):
                RB.test(cfg, ws)
        elif st == "report":
            final_report(cfg)
        elif st == "export":
            from stair.export import export
            export(cfg)
        elif st == "status":
            maybe_push(cfg, force=True)
        maybe_push(cfg, force=st not in ("status",))
    mins = (time.time() - t0) / 60
    log(f"done in {mins:.1f} min")
    with open(os.path.join(cfg["run_dir"], "stage_times.jsonl"), "a") as f:
        f.write(json.dumps({"stage": a.stage, "part": a.part, "model": a.model, "ws": a.ws, "minutes": round(mins, 2),
                            "job": os.environ.get("SLURM_JOB_ID", ""), "finished": time.strftime("%Y-%m-%d %H:%M:%S")}) + "\n")


if __name__ == "__main__":
    main()
