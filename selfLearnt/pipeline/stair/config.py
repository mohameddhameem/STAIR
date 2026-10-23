"""Configuration: YAML with ${...} references; model entries resolved to paths; model variants ('q7' = as released,
'q7:ft' = after our fine-tuning)."""
import copy
import os
import re

import yaml


def _interp(obj, env):
    if isinstance(obj, str):
        for _ in range(6):
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


def _merge(a, b):
    out = copy.deepcopy(a)
    for k, v in (b or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _toml_entry(ref):
    """'B.retriever.Qwen2.5-0.5B' -> the pinned entry of stair_common/models.toml."""
    from stair_common import load_models
    menu, role, name = ref.split(".", 2)
    return load_models()[menu][role][name]


EXPERIMENTS = ("AL-01", "AL-02", "AL-03", "AL-04", "AL-05", "AL-06")


def _only_experiment(raw, exp):
    """Keep only the workstream(s) one experiment needs. All experiments share run_dir, so saved outputs (shortlists,
    answers, fine-tuned models) are reused between them; each stage resumes from what is already there."""
    W = raw["workstreams"]
    keep = {k: w for k, w in W.items() if w.get("exp_id") == exp}
    if not keep:
        raise SystemExit(f"--exp {exp}: no workstream with that exp_id (have {sorted(w.get('exp_id') for w in W.values())})")
    if exp == "AL-02":          # AL-02 runs inside AgentTTS's test stage: keep ws1's models, but test only ws2's setting
        keep = {"ws1": dict(W["ws1"], also_test=[], skip_search=True), **keep}
    raw["_all_workstreams"] = W
    raw["workstreams"] = keep


def full_run(cfg):
    """The config with every workstream (reports and the status page always cover the whole run)."""
    return dict(cfg, workstreams=cfg.get("_all_workstreams", cfg["workstreams"]))


def _read_raw(path):
    """A YAML config with its `include:` chain merged in (the including file wins)."""
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    if raw.get("include"):
        raw = _merge(_read_raw(os.path.join(os.path.dirname(path), raw.pop("include"))), raw)
    return raw


def load_config(path, seed=None, exp=None):
    raw = _read_raw(path)
    raw["workstreams"] = {k: w for k, w in raw["workstreams"].items() if w}   # `wsN: null` in a config drops it
    if os.environ.get("STAIR_STATUS_PUSH") == "1":     # cluster.env switches the status page on for the real run
        raw.setdefault("progress", {})["github"] = True
    if seed is not None:
        raw["seed"] = int(seed)
    raw.setdefault("seed", 0)
    if exp:
        _only_experiment(raw, exp)
    if os.environ.get("STAIR_RUNS_DIR"):     # runs (training outputs, merged models) on another disk than the downloaded models
        raw["paths"]["runs_dir"] = "${STAIR_RUNS_DIR}"
        raw["paths"]["data_dir"] = "${STAIR_RUNS_DIR}/${run_name}/data"
    env = {k: v for k, v in os.environ.items() if k.startswith("STAIR_")}
    env["run_name"] = raw["run_name"]
    env.update({k: v for k, v in raw["paths"].items() if isinstance(v, str)})
    for _ in range(4):
        raw["paths"] = _interp(raw["paths"], env)
        env.update(raw["paths"])
    cfg = _interp(raw, env)
    if "${" in cfg["paths"]["project_dir"]:
        raise SystemExit("set STAIR_PROJECT_DIR (storage for models, data and runs) before running")
    if os.environ.get("STAIR_RESULTS_DIR") and cfg.get("run_name") == "full_v1":
        cfg["paths"]["results_dir"] = os.environ["STAIR_RESULTS_DIR"]     # outside the git copy (no clash with committed tables)
    rd = cfg["paths"].get("results_dir", "results/common")
    if not os.path.isabs(rd):   # relative to the STAIR repo root, wherever the command is run from
        cfg["paths"]["results_dir"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", rd)
    cfg["paths"]["results_dir"] = os.path.normpath(cfg["paths"]["results_dir"])
    cfg["run_dir"] = os.path.join(cfg["paths"]["runs_dir"], cfg["run_name"], f"seed{cfg['seed']}")
    for k, m in cfg["models"].items():
        m["key"] = k
        if m.get("toml"):
            t = _toml_entry(m["toml"])
            m["hf_id"], m["revision"], m["dtype"] = t["repo"], t["revision"], t.get("dtype", "bf16")
            m["params_b"] = float(t["params_b"])
            if m["dtype"] == "awq":
                m["quantization"] = "awq"
        m["params_b"] = float(m.get("params_b", float(m.get("params", 1e9)) / 1e9))
        m["params"] = m["nominal"] = m["params_b"] * 1e9      # cost uses the nominal size (AgentTTS units)
        m.setdefault("dir", m["hf_id"].split("/")[-1] + (f"@{m['revision'][:7]}" if m.get("revision") else ""))
        m["path"] = os.path.join(cfg["paths"]["models_dir"], m["dir"])
        if m.get("train_toml"):          # pinned bf16 copy that QLoRA trains (models.toml [training.qlora])
            t = _toml_entry(m["train_toml"])
            m["train_hf_id"], m["train_revision"] = t["repo"], t["revision"]
        if m.get("train_hf_id"):
            m.setdefault("train_dir", m["train_hf_id"].split("/")[-1] + "-train")
            m["train_path"] = os.path.join(cfg["paths"]["models_dir"], m["train_dir"])
        else:
            m["train_path"] = m["path"]
    for k, v in cfg["rules"]["hard"].items():
        cfg["rules"]["hard"][k] = float(v)
    return cfg


def ft_settings(cfg, key):
    """(role, settings) of a model's fine-tuning, or (None, None)."""
    F = cfg["finetune"]
    for role in ("retrievers", "answerers"):
        if key in (F.get(role) or {}):
            return role[:-1], dict(F["defaults"], **F[role][key])
    return None, None


def variant(cfg, name):
    """Resolve 'q7' or 'q7:ft' to a model entry with everything needed to run it."""
    base, _, tag = name.partition(":")
    m = dict(cfg["models"][base])
    m["variant"] = name
    m["finetuned"] = tag == "ft"
    m["label_full"] = m["label"] + (" (fine-tuned)" if m["finetuned"] else "")
    if m["finetuned"]:
        role, s = ft_settings(cfg, base)
        assert s, f"{base} has no fine-tuning settings"
        m["ft"] = s
        m["role"] = role
        m["adapter_dir"] = os.path.join(cfg["run_dir"], "adapters", base)
        m["merged_dir"] = os.path.join(cfg["run_dir"], "merged", base)
        if s["method"] == "lora":
            m["serve_path"], m["lora_path"] = m["merged_dir"], None
        else:
            m["serve_path"], m["lora_path"] = m["path"], os.path.join(m["adapter_dir"], "chosen")
    else:
        m["serve_path"], m["lora_path"] = m["path"], None
    return m


def model_tag(m):
    """'role=repo@revision' piece of the results' models column."""
    return f"{m['hf_id']}@{m.get('revision', 'main')}" + (" +LoRA" if m.get("finetuned") else "")


def serve_entry(m):
    """The entry an engine loads (the merged copy for LoRA-trained small models)."""
    e = dict(m)
    e["path"] = m["serve_path"]
    if m.get("finetuned") and m["ft"]["method"] == "lora":
        e.pop("quantization", None)
    return e


def workstreams(cfg, kind=None):
    return [k for k, w in cfg["workstreams"].items() if kind is None or w["kind"] == kind]


def all_variants(cfg, role):
    """Every retriever / answerer variant used by the rule-based workstreams."""
    out = []
    for k in workstreams(cfg, "rules"):
        for v in cfg["workstreams"][k]["retrievers" if role == "ret" else "answerers"]:
            if v not in out:
                out.append(v)
    return out


def ft_keys(cfg):
    out = []
    for k in workstreams(cfg, "rules"):
        w = cfg["workstreams"][k]
        for v in w["retrievers"] + w["answerers"]:
            b, _, t = v.partition(":")
            if t == "ft" and b not in out:
                out.append(b)
    return out
