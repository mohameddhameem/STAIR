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


def load_config(path):
    raw = yaml.safe_load(open(path, encoding="utf-8"))
    if raw.get("include"):
        base = yaml.safe_load(open(os.path.join(os.path.dirname(path), raw.pop("include")), encoding="utf-8"))
        raw = _merge(base, raw)
    env = {"run_name": raw["run_name"]}
    env.update({k: v for k, v in raw["paths"].items() if isinstance(v, str)})
    for _ in range(4):
        raw["paths"] = _interp(raw["paths"], env)
        env.update(raw["paths"])
    cfg = _interp(raw, env)
    cfg["run_dir"] = os.path.join(cfg["paths"]["runs_dir"], cfg["run_name"])
    for k, m in cfg["models"].items():
        m["key"] = k
        m["params"] = float(m["params"])
        m["nominal"] = float(m.get("nominal", m["params"]))
        m.setdefault("dir", m["hf_id"].split("/")[-1])
        m["path"] = os.path.join(cfg["paths"]["models_dir"], m["dir"])
        if m.get("train_hf_id"):
            m.setdefault("train_dir", m["train_hf_id"].split("/")[-1])
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
