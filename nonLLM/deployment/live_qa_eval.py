"""Actual GGUF QA pilot on internal validation, using cached strong retrieval evidence."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import re
import socket
import string
import subprocess
import sys
import time
import urllib.request

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import numeric_features, text_features


def official_metrics(path):
    """Use the official answer-scoring functions without its optional ujson dependency."""
    source = path.read_text()
    names = {"normalize_answer", "f1_score", "exact_match_score"}
    tree = ast.parse(source)
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert {node.name for node in selected} == names
    namespace = {"re": re, "string": string, "Counter": Counter}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"), namespace)
    return namespace, hashlib.sha256(source.encode()).hexdigest()


def request_json(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120 if body else 1) as r:
        return json.load(r)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/ssd1/zmcheng/cs707"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--seed", type=int, default=707)
    ap.add_argument("--port", type=int, default=18087)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists() and any(a.out.iterdir()):
        raise SystemExit("Use a fresh output directory")
    a.out.mkdir(parents=True, exist_ok=True)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", a.port))
    dataset = a.root / "extracted/Dataset"
    splits = json.loads((dataset / "splits.json").read_text())
    ids = sorted(random.Random(a.seed).sample(splits["splits"]["val"], a.n))
    with gzip.open(dataset / "AgentTTS-HotpotQA.json.gz", "rt") as f:
        source = json.load(f)
    gold = {i: source[i]["answers"] for i in ids}
    rows = {}
    with gzip.open(dataset / "router_data_lenient.jsonl.gz", "rt") as f:
        for line in f:
            row = json.loads(line)
            if row["idx"] in gold:
                assert row["split"] == "val" and row["_id"] == source[row["idx"]]["_id"]
                rows[row["idx"]] = row
    del source
    assert set(rows) == set(ids)
    scorer, metric_sha = official_metrics(a.root / "tools/hotpotqa_reference/hotpot_evaluate_v1.py")
    models = json.loads((a.root / "models/manifest.json").read_text())
    predictions = {}; records = []
    config = {"split": "internal validation", "n": a.n, "sample_seed": a.seed,
              "sample_indices": ids, "gpu": a.gpu, "context": 8192, "max_new_tokens": 64,
              "temperature": 0, "retrieval": "cached evidence_S; no actual retrieval run",
              "metrics": "official answer EM/F1; diagnostic case-insensitive whole-word containment",
              "official_metric_sha256": metric_sha,
              "official_metric_url": "https://raw.githubusercontent.com/hotpotqa/hotpot/master/hotpot_evaluate_v1.py",
              "prompt": "References: {evidence_S}\nQuestion: {question}\nGive only a short answer.\nAnswer:",
              "models": {"W": models[1], "S": models[0]}}
    (a.out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for choice, m in [("W", models[1]), ("S", models[0])]:
        free = int(subprocess.check_output(["nvidia-smi", f"--id={a.gpu}",
                   "--query-gpu=memory.free", "--format=csv,noheader,nounits"], text=True).strip())
        required = 8000 if choice == "S" else 3500
        if free < required:
            raise SystemExit(f"GPU {a.gpu} has {free} MiB free; require {required}")
        command = [str(a.root / "tools/llama.cpp/build/bin/llama-server"),
                   "-m", str(a.root / "models" / m["directory"] / m["filename"]),
                   "-ngl", "99", "-c", "8192", "-np", "1", "-b", "128", "-ub", "128",
                   "-t", "2", "--host", "127.0.0.1", "--port", str(a.port)]
        predictions[choice] = {}
        with (a.out / f"server_{choice}.log").open("w") as log:
            proc = subprocess.Popen(command, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu)),
                                    stdout=log, stderr=log)
            try:
                base = f"http://127.0.0.1:{a.port}"; started = time.monotonic()
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Model server exited {proc.returncode}")
                    if time.monotonic() - started > 180:
                        raise TimeoutError("Model startup timed out")
                    try:
                        request_json(base, "/health")
                        break
                    except OSError:
                        time.sleep(0.5)
                for j, i in enumerate(ids):
                    row = rows[i]
                    # Gold answers and passage annotations never enter the prompt.
                    prompt = (f"References: {row['evidence_S']}\nQuestion: {row['question']}\n"
                              "Give only a short answer.\nAnswer:")
                    tokens = request_json(base, "/tokenize", {"content": prompt})["tokens"]
                    if len(tokens) + 64 > 8192:
                        raise ValueError(f"idx {i} exceeds context; refusing silent truncation")
                    before = time.perf_counter()
                    response = request_json(base, "/completion", {"prompt": prompt, "n_predict": 64,
                         "temperature": 0, "seed": a.seed, "stop": ["\n\n", "Question:", "References:"]})
                    answer = response["content"].strip()
                    em = max(float(scorer["exact_match_score"](answer, g)) for g in gold[i])
                    f1 = max(scorer["f1_score"](answer, g)[0] for g in gold[i])
                    containment = any(re.search(r"(?<!\w)" + re.escape(g.lower()) + r"(?!\w)",
                                               answer.lower()) is not None for g in gold[i] if g)
                    record = {"idx": i, "_id": row["_id"], "question": row["question"],
                              "model_choice": choice, "answers": gold[i], "prediction": answer,
                              "answer_em": em, "answer_f1": f1, "diagnostic_containment": int(containment),
                              "prompt_tokens": len(tokens), "seconds": time.perf_counter() - before,
                              "timings": response.get("timings")}
                    predictions[choice][i] = record; records.append(record)
                    with (a.out / "predictions.jsonl").open("a") as sink:
                        sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                    print(f"[{choice}] {j+1}/{a.n} idx={i} EM={em:g} F1={f1:.3f}", flush=True)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill(); proc.wait()
        time.sleep(1)
    bundle = joblib.load(a.root / "runs/baselines_real_v1/qa_S.joblib")
    text = bundle["vectorizer"].transform(text_features(rows[i], "qa_S") for i in ids)
    numeric = np.array([numeric_features(rows[i], "qa_S") for i in ids])
    policies = {"always_weak": ["W"] * a.n, "always_strong": ["S"] * a.n}
    for family, model in bundle["models"].items():
        policies[family] = model.predict(numeric if family == "random_forest" else text).tolist()
    summary = {}
    for name, picks in policies.items():
        chosen = [predictions[ch][i] for i, ch in zip(ids, picks)]
        summary[name] = {"n": a.n, "answer_em_pct": 100*np.mean([r["answer_em"] for r in chosen]),
                         "answer_f1_pct": 100*np.mean([r["answer_f1"] for r in chosen]),
                         "diagnostic_containment_pct": 100*np.mean([r["diagnostic_containment"] for r in chosen]),
                         "answerer_picks": dict(Counter(picks))}
    (a.out / "results.json").write_text(json.dumps({"config": config, "policies": summary}, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
