"""Regenerate an Instruct grid on four GPU shards, then train CPU cascade routers.

Each worker owns one GPU and disjoint instance IDs. Stores are append-only and
resumable. The coordinator checks all keys, rejects truncated calls, builds new
labels/costs, and automatically invokes cascade.py after successful generation.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = Path(__file__).resolve()
KINDS = ("retr_weak", "retr_strong", "qa_weak", "qa_strong")


def digest(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()


def write_json(p, obj):
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2))
    tmp.replace(p)


def source():
    with gzip.open(ROOT / "extracted/Dataset/AgentTTS-HotpotQA.json.gz", "rt") as f:
        return json.load(f)


def core():
    os.environ["SLR_POOL_K"] = "100"
    os.environ["SLR_SHUFFLE"] = "off"
    spec = importlib.util.spec_from_file_location("configA_grid", ROOT / "October 2026/slr_code_configA/run_labels4.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def instance(inst):
    # Preserve the supplied 100-passage ordering; no gold re-injection.
    return {"_id": inst["_id"], "question": inst["question"],
            "ctxs_candidate": [p[:600] for p in inst["ctxs_candidate"][:100]]}


def read_store(path):
    if not path.exists(): return {}
    records = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)  # Fail on corruption rather than skipping records.
            if r["k"] in records: raise ValueError(f"Duplicate key {r['k']} in {path}")
            records[r["k"]] = r
    return records


def worker(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    R = core()
    torch.set_num_threads(2)
    torch.manual_seed(17)
    data = source()
    ids = list(range(a.shard, len(data), 4))
    if a.limit: ids = ids[:a.limit]
    shard = a.out / f"shard_{a.shard}"
    shard.mkdir(parents=True, exist_ok=True)
    info = {"shard": a.shard, "n": len(ids), "ids_sha256": hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            "grid_code_sha256": digest(ROOT / "October 2026/slr_code_configA/run_labels4.py"),
            "generation": "FP16, greedy, SDPA, 100 passages capped at 600 chars, supplied order, no gold insertion, max input 16384",
            "weak_model": str(a.weak_model), "strong_model": str(a.strong_model)}
    manifest = shard / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != info:
        raise ValueError("Shard configuration differs from existing outputs")
    write_json(manifest, info)
    stores = {k: read_store(shard / (k + ".jsonl")) for k in KINDS}
    metas = {k: read_store(shard / ("meta_" + k + ".jsonl")) for k in KINDS}
    for kind in KINDS:
        if stores[kind].keys() != metas[kind].keys():
            raise ValueError(f"Store/meta mismatch for {kind}; inspect interrupted last write before resuming")
        jobs = [(i, None) for i in ids] if kind.startswith("retr") else [(i, b) for i in ids for b in ("weak", "strong")]
        todo = [(i, b) for i, b in jobs if (str(i) if b is None else f"{i}|{b}") not in stores[kind]]
        if not todo: continue
        modelpath = a.weak_model if kind.endswith("weak") else a.strong_model
        print(f"shard={a.shard} stage={kind} pending={len(todo)} model={modelpath}", flush=True)
        tok = AutoTokenizer.from_pretrained(modelpath, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(modelpath, torch_dtype=torch.float16,
                    device_map="cuda", attn_implementation="sdpa", local_files_only=True).eval()
        start = time.monotonic()
        with (shard / (kind + ".jsonl")).open("a") as sf, (shard / ("meta_" + kind + ".jsonl")).open("a") as mf:
            for j, (i, branch) in enumerate(todo):
                inst = instance(data[i]); key = str(i) if branch is None else f"{i}|{branch}"
                if branch is None:
                    user, system, max_new = R.retr_prompt(inst), R.SYS_RETR, 768
                else:
                    selected = R.parse_selection(stores["retr_" + branch][str(i)]["v"], len(inst["ctxs_candidate"]))
                    user, system, max_new = R.qa_prompt(inst, R.build_block(inst, selected)), "You are a helpful assistant.", 128
                text = tok.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                               tokenize=False, add_generation_prompt=True)
                enc = tok(text, return_tensors="pt", add_special_tokens=False).to("cuda")
                n_in = enc.input_ids.shape[1]
                if n_in > 16384:
                    raise ValueError(f"{kind}/{key}: {n_in} input tokens exceeds cap; refusing truncation")
                before = time.monotonic()
                with torch.inference_mode():
                    out = model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                         pad_token_id=tok.eos_token_id)
                answer = tok.decode(out[0, n_in:], skip_special_tokens=True)
                record = {"k": key, "v": answer}
                meta = {"k": key, "n_in": n_in, "n_out": out.shape[1] - n_in,
                        "truncated": False, "secs": round(time.monotonic() - before, 4)}
                sf.write(json.dumps(record, ensure_ascii=False) + "\n"); sf.flush()
                mf.write(json.dumps(meta) + "\n"); mf.flush()
                stores[kind][key] = record; metas[kind][key] = meta
                write_json(shard / "status.json", {"phase": kind, "done": j + 1, "pending_at_start": len(todo),
                    "seconds_per_call": (time.monotonic() - start) / (j + 1), "last_key": key,
                    "last_input_tokens": n_in, "last_output_tokens": meta["n_out"]})
                if j == 0 or (j + 1) % 10 == 0:
                    print(f"{kind} {j+1}/{len(todo)} elapsed={time.monotonic()-start:.1f}s last={key} tokens={n_in}+{meta['n_out']}", flush=True)
                del enc, out
        del model, tok
        import gc
        gc.collect(); torch.cuda.empty_cache()
    write_json(shard / "status.json", {"phase": "complete", "n": len(ids)})


def normalize(s):
    import string
    s = "".join(c for c in str(s).lower() if c not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", s).split())


def contains(text, gold):
    text = normalize(text)
    return any(re.search(r"(?<!\w)" + re.escape(normalize(g)) + r"(?!\w)", text)
               is not None for g in gold if normalize(g))


def score(pred, gold):
    p = normalize(pred)
    strict = any(p == normalize(g) for g in gold)
    # Explicit locally defined lenient metric; author's mint/scoring script is absent.
    guarded = bool(re.match(r"^(more than|less than|at least|at most|over|under)\b", p))
    loose = strict or (not guarded and any(
        re.search(r"(?<!\w)" + re.escape(normalize(g)) + r"(?!\w)", p) or
        re.search(r"(?<!\w)" + re.escape(p) + r"(?!\w)", normalize(g))
        for g in gold if normalize(g) and p))
    return int(strict), int(bool(loose))


def finalize(a):
    R = core()
    data = source()
    idxs = list(range(len(data)))
    if a.limit: idxs = [i for i in idxs if i // 4 < a.limit]
    grid_dir = a.out / "labels/full_run_p100"; grid_dir.mkdir(parents=True, exist_ok=True)
    all_stores, all_meta = {}, {}
    for kind in KINDS:
        all_stores[kind], all_meta[kind] = {}, {}
        for s in range(4):
            all_stores[kind].update(read_store(a.out / f"shard_{s}/{kind}.jsonl"))
            all_meta[kind].update(read_store(a.out / f"shard_{s}/meta_{kind}.jsonl"))
        expected = {str(i) for i in idxs} if kind.startswith("retr") else {f"{i}|{b}" for i in idxs for b in ("weak", "strong")}
        assert all_stores[kind].keys() == expected and all_meta[kind].keys() == expected, kind
        assert not any(r.get("truncated") or r.get("oom") for r in all_meta[kind].values())
        for prefix, records in (("", all_stores[kind]), ("meta_", all_meta[kind])):
            with (grid_dir / (prefix + kind + ".jsonl")).open("w") as f:
                for k in sorted(records): f.write(json.dumps(records[k], ensure_ascii=False) + "\n")
    splits = json.load(open(ROOT / "extracted/Dataset/splits.json"))["splits"]
    splitof = {i: sp for sp, ids in splits.items() for i in ids}
    retrows, qarows = [], []
    with (grid_dir / "router_data_lenient.jsonl").open("w") as f:
        for i in idxs:
            inst = instance(data[i]); gold = data[i]["answers"]
            if isinstance(gold, str): gold = [gold]
            sels = {b: R.parse_selection(all_stores["retr_" + b][str(i)]["v"], len(inst["ctxs_candidate"])) for b in ("weak", "strong")}
            evid = {b: R.build_block(inst, ids) for b, ids in sels.items()}
            sc = [score(all_stores["qa_" + answerer][f"{i}|{retr}"]["v"], gold)
                  for retr in ("weak", "strong") for answerer in ("weak", "strong")]
            ems, eml = [r[0] for r in sc], [r[1] for r in sc]
            covw = contains(" ".join(inst["ctxs_candidate"][n - 1] for n in sels["weak"]), gold)
            covu = contains(" ".join(inst["ctxs_candidate"][n - 1] for n in set(sels["weak"]) | set(sels["strong"])), gold)
            rl = "W" if covw else ("S" if covu else "X")
            retrows.append({"idx": i, "split": splitof[i], "ret_label": rl, "ret_target": "W" if rl == "X" else rl})
            qls = []
            for j, rc in enumerate(("W", "S")):
                ql = "W" if eml[j*2] else ("S" if eml[j*2+1] else "X")
                qarows.append({"idx": i, "split": splitof[i], "ret_choice": rc, "qa_label": ql, "qa_target": "W" if ql == "X" else ql})
                qls.append(None if ql == "X" else ql)
            row = {"idx": i, "_id": inst["_id"], "split": splitof[i], "question": inst["question"],
                   "evidence_W": evid["weak"], "evidence_S": evid["strong"], "em_lenient": eml, "em_strict": ems,
                   "ret_label": None if rl == "X" else rl, "qa_label_W": qls[0], "qa_label_S": qls[1]}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    for stage, rows in (("ret", retrows), ("qa", qarows)):
        dest = a.out / f"labels/{stage}_labels"; dest.mkdir(parents=True, exist_ok=True)
        with (dest / f"{stage}_labels.jsonl").open("w") as f:
            for r in rows: f.write(json.dumps(r) + "\n")
        cnt = Counter(r[stage + "_target"] for r in rows if r["split"] == "train")
        write_json(dest / "manifest.json", {"n": len(rows), "train_counts": cnt,
            "class_weights_post_mapping": {k: sum(cnt.values())/(2*v) for k, v in cnt.items()},
            "rule": "Ret: answer coverage; QA: generated outcome; X mapped to W"})
    raw = {}
    for kind in KINDS:
        for branch in ((None,) if kind.startswith("retr") else ("weak", "strong")):
            records = [r for k, r in all_meta[kind].items() if branch is None or k.endswith("|" + branch)]
            name = kind if branch is None else kind + "_given_" + ("W" if branch == "weak" else "S")
            raw[name] = {"calls": len(records), "mean_tokens": sum(r["n_in"] + r["n_out"] for r in records)/len(records)}
            raw[name]["params_times_tokens"] = (1.5 if kind.endswith("weak") else 7.) * raw[name]["mean_tokens"]
    unit = min(v["params_times_tokens"] for v in raw.values())
    names = ("retr_weak", "retr_strong", "qa_weak_given_W", "qa_weak_given_S", "qa_strong_given_W", "qa_strong_given_S")
    keys = ("COST_RET_W", "COST_RET_S", "COST_QA_W_GIVEN_W", "COST_QA_W_GIVEN_S", "COST_QA_S_GIVEN_W", "COST_QA_S_GIVEN_S")
    write_json(a.out / "token_costs.json", {"measured": raw, "gate_eval_constants":
        {k: raw[n]["params_times_tokens"]/unit for k, n in zip(keys, names)}, "excludes": "CPU router overhead"})
    write_json(a.out / "generation_summary.json", {"n": len(idxs), "calls": 6*len(idxs),
        "strict_metric": "normalized exact match", "lenient_metric": "normalized bidirectional whole-word containment; prefix guard more than/less than/at least/at most/over/under",
        "reproduction_limit": "Original mint/scoring implementation absent; local explicit scorer. New outputs and labels, not original Config A result reproduction."})


def coordinator(a):
    a.out.mkdir(parents=True, exist_ok=True)
    locked = {"weak_model": str(a.weak_model), "strong_model": str(a.strong_model), "limit_per_gpu": a.limit,
              "script_sha256": digest(SCRIPT), "source_sha256": digest(ROOT / "extracted/Dataset/AgentTTS-HotpotQA.json.gz"),
              "router_code_sha256": digest(SCRIPT.parents[1] / "cascade.py"),
              "splits_sha256": digest(ROOT / "extracted/Dataset/splits.json"),
              "model_files": {str(p): digest(p) for m in (a.weak_model, a.strong_model)
                              for p in sorted(m.glob("*.safetensors"))}}
    p = a.out / "manifest.json"
    if p.exists() and json.loads(p.read_text()) != locked: raise ValueError("Coordinator manifest changed; fresh output required")
    write_json(p, locked)
    children, logs = [], []
    try:
        for s in range(4):
            logs.append((a.out / f"gpu_{s}.log").open("a"))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(s), OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2")
            cmd = [sys.executable, "-u", str(SCRIPT), "--worker", "--shard", str(s), "--out", str(a.out),
                   "--weak-model", str(a.weak_model), "--strong-model", str(a.strong_model), "--limit", str(a.limit)]
            children.append(subprocess.Popen(cmd, env=env, stdout=logs[-1], stderr=subprocess.STDOUT))
        write_json(a.out / "status.json", {"phase": "generation", "worker_pids": [p.pid for p in children]})
        while any(p.poll() is None for p in children):
            if any(p.poll() not in (None, 0) for p in children): raise RuntimeError("GPU worker failed; see gpu logs")
            time.sleep(5)
        if any(p.returncode for p in children): raise RuntimeError("GPU worker failed")
        write_json(a.out / "status.json", {"phase": "building_labels_and_costs"})
        finalize(a)
        if not a.limit:
            write_json(a.out / "status.json", {"phase": "CPU_router_training"})
            with (a.out / "router.log").open("a") as f:
                subprocess.run([sys.executable, "-u", str(SCRIPT.parents[1] / "cascade.py"),
                    "--labels-root", str(a.out / "labels"), "--qa-weak", str(a.out / "labels/full_run_p100/qa_weak.jsonl"),
                    "--token-costs", str(a.out / "token_costs.json"), "--out", str(a.out / "router")],
                    env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                    stdout=f, stderr=subprocess.STDOUT, check=True)
        write_json(a.out / "status.json", {"phase": "complete", "report": str(a.out / "router/REPORT.md") if not a.limit else "smoke only"})
    except BaseException as e:
        for p in children:
            if p.poll() is None: p.terminate()
        for p in children:
            try: p.wait(timeout=15)
            except subprocess.TimeoutExpired: p.kill(); p.wait()
        write_json(a.out / "status.json", {"phase": "failed", "error": repr(e)})
        raise
    finally:
        for f in logs: f.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--weak-model", type=Path, default=ROOT / "models/Qwen2.5-1.5B-Instruct-HF")
    ap.add_argument("--strong-model", type=Path, default=ROOT / "models/Qwen2.5-7B-Instruct-HF")
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--shard", type=int, choices=range(4))
    ap.add_argument("--limit", type=int, default=0, help="Smoke: max questions per GPU; 0 = all")
    a = ap.parse_args()
    if a.worker:
        if a.shard is None: ap.error("worker needs shard")
        worker(a)
    else:
        coordinator(a)


if __name__ == "__main__": main()
