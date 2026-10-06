"""Resumable actual retrieval + QA benchmark for frozen non-LLM routers."""
import argparse
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from contextlib import contextmanager
from collections import Counter
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import joblib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baselines import numeric_features, text_features
from live_qa_eval import official_metrics

VERSION = "actual-pipeline-v1"
RET_TEMPLATE = ("Select up to 4 references needed to answer a question. "
                "Return only reference numbers in square brackets.\n"
                "Example references:\n[1] Mars: Mars is a planet.\n"
                "[2] France: The capital of France is Paris.\n"
                "Example question: What is the capital of France?\nSelected references: [2]\n\n"
                "Example references:\n[1] Alice: Alice wrote the novel River.\n"
                "[2] Alice biography: Alice was born in London.\n[3] Ocean: An ocean is a large body of water.\n"
                "Example question: Where was the author of River born?\nSelected references: [1], [2]\n\n"
                "References:\n{pool}\nQuestion: {question}\n"
                "Select up to 4 relevant references.\nSelected references:")
QA_TEMPLATE = "References: {evidence}\nQuestion: {question}\nGive only a short answer.\nAnswer:"
ID_GRAMMAR = ('root ::= "[" id "]" (", [" id "]")? (", [" id "]")? (", [" id "]")?\n'
              'id ::= ' + " | ".join(json.dumps(str(i)) for i in range(1, 101)))


def passages(pool):
    matches = list(re.finditer(r"(?m)^\[(\d+)\]\s*", pool))
    return {int(m.group(1)): pool[m.start(): matches[j+1].start() if j+1 < len(matches) else len(pool)].strip()
            for j, m in enumerate(matches)}


def selected_evidence(question, pool, output):
    documents = passages(pool)
    ids = list(dict.fromkeys(int(n) for n in re.findall(r"\[(\d+)\]", output)))
    if not ids or len(ids) > 4 or any(i not in documents for i in ids):
        raise ValueError(f"Invalid retrieval selection: {output!r}")
    return ids, "Query: " + question + "\nSelected references:\n" + "\n".join(documents[i] for i in ids)


def http(base, path, body=None):
    data = None if body is None else json.dumps(body).encode()
    req = urllib.request.Request(base + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600 if body else 2) as response:
        return json.load(response)


def status(out, value):
    tmp = out / "status.json.tmp"
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(out / "status.json")


def gpu_processes(gpu, own_pid=None):
    raw = subprocess.check_output(["nvidia-smi", f"--id={gpu}",
           "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return [int(line.strip()) for line in raw.splitlines()
            if line.strip().isdigit() and int(line.strip()) != own_pid]


@contextmanager
def server(a, choice, log_name):
    minimum = 10000 if choice == "S" else 4500
    free = int(subprocess.check_output(["nvidia-smi", f"--id={a.gpu}", "--query-gpu=memory.free",
                 "--format=csv,noheader,nounits"], text=True).strip())
    occupied = gpu_processes(a.gpu)
    if occupied or free < minimum:
        raise RuntimeError(f"GPU {a.gpu} cannot start: other compute PIDs={occupied}, "
                           f"free={free} MiB, required={minimum}. User notification required; no automatic retry.")
    m = a.models[choice]
    cmd = [str(a.root / "tools/llama.cpp/build/bin/llama-server"),
           "-m", str(a.root / "models" / m["directory"] / m["filename"]),
           "-ngl", "99", "-c", str(a.context * a.workers), "-np", str(a.workers),
           "-b", "2048", "-ub", "1024", "-fa", "on", "-ctk", "q8_0", "-ctv", "q8_0",
           "-t", "2", "--host", "127.0.0.1", "--port", str(a.port)]
    with (a.out / log_name).open("a") as log:
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
        runtime = a.root / "tools/runtime"
        if runtime.exists():
            env["LD_LIBRARY_PATH"] = str(runtime) + ":" + env.get("LD_LIBRARY_PATH", "")
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        watch_stop = threading.Event()
        def watch():
            while not watch_stop.wait(2):
                try:
                    occupied = gpu_processes(a.gpu, proc.pid)
                    if occupied:
                        print(f"Stopping own model to avoid GPU contention with PIDs {occupied}", flush=True)
                        proc.terminate()
                        return
                except Exception as error:
                    print(f"GPU occupancy monitoring failed; stopping own model: {error}", flush=True)
                    if proc.poll() is None:
                        proc.terminate()
                    return
        watchdog = threading.Thread(target=watch, daemon=True)
        watchdog.start()
        try:
            base = f"http://127.0.0.1:{a.port}"; start = time.monotonic()
            while True:
                if proc.poll() is not None:
                    raise RuntimeError(f"Server failed: see {a.out / log_name}")
                if time.monotonic() - start > 180:
                    raise TimeoutError("Server startup timed out")
                try:
                    http(base, "/health")
                    break
                except OSError:
                    time.sleep(0.5)
            yield base
        finally:
            watch_stop.set()
            watchdog.join(timeout=5)
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill(); proc.wait()
    time.sleep(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/ssd1/zmcheng/cs707"))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--split", choices=["val", "test"], default="test")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--context", type=int, default=32768)
    parser.add_argument("--port", type=int, default=18089)
    a = parser.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    # Prevent two resumed workers from using one database or port.
    import fcntl
    lock = (a.out / "run.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", a.port))
    dataset = a.root / "extracted/Dataset"
    split_bytes = (dataset / "splits.json").read_bytes()
    ids = json.loads(split_bytes)["splits"][a.split]
    if a.limit:
        ids = ids[:a.limit]
    keep = set(ids)
    models = json.loads((a.root / "models/manifest.json").read_text())
    a.models = {"W": models[1], "S": models[0]}
    bundle_dir = a.root / "runs/baselines_real_v1"
    config = {"version": VERSION, "split": a.split, "ids": ids, "models": a.models,
              "workers": a.workers, "gpu": a.gpu, "context_per_request": a.context,
              "kv_cache": "q8_0", "temperature": 0, "seed": 707,
              "ret_prompt": RET_TEMPLATE, "qa_prompt": QA_TEMPLATE, "retrieval_grammar": ID_GRAMMAR,
              "retrieval_max_selected": 4, "retrieval_max_tokens": 64, "qa_max_tokens": 64,
              "split_sha256": hashlib.sha256(split_bytes).hexdigest(),
              "routers_sha256": {s: hashlib.sha256((bundle_dir / f"{s}.joblib").read_bytes()).hexdigest()
                                 for s in ["ret", "qa_W", "qa_S"]},
              "note": "Frozen routers; new actual evidence; full six-call grid per question. No silent truncation."}
    manifest = a.out / "config.json"
    if manifest.exists() and json.loads(manifest.read_text()) != config:
        raise SystemExit("Existing run has a different configuration")
    manifest.write_text(json.dumps(config, indent=2) + "\n")
    rows = {}
    with gzip.open(dataset / "router_data_lenient.jsonl.gz", "rt") as src:
        for line in src:
            row = json.loads(line)
            if row["idx"] in keep:
                assert row["split"] == a.split
                assert len(passages(row["pool_text"])) == row["n_pool"] == 100
                rows[row["idx"]] = row
    with gzip.open(dataset / "AgentTTS-HotpotQA.json.gz", "rt") as src:
        originals = json.load(src)
    gold = {i: originals[i]["answers"] for i in ids}
    assert all(rows[i]["_id"] == originals[i]["_id"] for i in ids)
    del originals
    scorer, scorer_hash = official_metrics(a.root / "tools/hotpotqa_reference/hotpot_evaluate_v1.py")
    database = sqlite3.connect(a.out / "calls.sqlite")
    database.execute("PRAGMA journal_mode=WAL")
    database.execute("CREATE TABLE IF NOT EXISTS calls(stage TEXT, idx INTEGER, payload TEXT, PRIMARY KEY(stage,idx))")
    cached = {(stage, i): json.loads(payload) for stage, i, payload in database.execute("SELECT stage,idx,payload FROM calls")}
    planned = 6 * len(ids); began = time.monotonic()

    def execute(base, stage, idx):
        row = rows[idx]
        retrieval = stage.startswith("ret")
        if retrieval:
            prompt = RET_TEMPLATE.format(question=row["question"], pool=row["pool_text"])
        else:
            prompt = QA_TEMPLATE.format(question=row["question"], evidence=cached[("ret"+stage[0], idx)]["evidence"])
        tokens = http(base, "/tokenize", {"content": prompt})["tokens"]
        if len(tokens) + 64 > a.context:
            raise ValueError(f"{stage}/{idx}: {len(tokens)} prompt tokens exceed context; no truncation")
        body = {"prompt": prompt, "n_predict": 64, "temperature": 0, "seed": 707, "cache_prompt": True}
        if retrieval:
            body["grammar"] = ID_GRAMMAR
        else:
            body["stop"] = ["\n\n", "Question:", "References:"]
        start = time.monotonic()
        response = http(base, "/completion", body)
        record = {"idx": idx, "stage": stage, "output": response["content"].strip(),
                  "prompt_tokens": len(tokens), "seconds": time.monotonic()-start,
                  "timings": response.get("timings"), "stopped_limit": response.get("stopped_limit"),
                  "execution_host": socket.gethostname(), "execution_gpu": a.gpu}
        if retrieval:
            selected, evidence = selected_evidence(row["question"], row["pool_text"], record["output"])
            record.update(selected_ids=selected, evidence=evidence)
        else:
            record["answer_em"] = max(float(scorer["exact_match_score"](record["output"], g)) for g in gold[idx])
            record["answer_f1"] = max(scorer["f1_score"](record["output"], g)[0] for g in gold[idx])
            record["containment"] = int(any(re.search(r"(?<!\w)"+re.escape(g.lower())+r"(?!\w)",
                                                      record["output"].lower()) for g in gold[idx] if g))
        return record

    def phase(base, stage):
        missing = [i for i in ids if (stage, i) not in cached]
        print(f"PHASE {stage}: {len(missing)} pending / {len(ids)}", flush=True)
        with ThreadPoolExecutor(max_workers=a.workers) as executor:
            todo = iter(missing)
            pending = {executor.submit(execute, base, stage, i) for i in list(missing)[:a.workers]}
            for _ in range(min(a.workers, len(missing))):
                next(todo)
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    record = future.result(); idx = record["idx"]
                    database.execute("INSERT INTO calls VALUES(?,?,?)", (stage, idx, json.dumps(record)))
                    database.commit(); cached[(stage, idx)] = record
                    done = sum((stage, i) in cached for i in ids)
                    status(a.out, {"state": "running", "stage": stage, "stage_done": done,
                                   "stage_total": len(ids), "calls_done": len(cached), "calls_total": planned,
                                   "elapsed_current_session_seconds": time.monotonic()-began})
                    if done % 10 == 0 or done == len(ids):
                        print(f"{stage}: {done}/{len(ids)} | calls {len(cached)}/{planned}", flush=True)
                    next_id = next(todo, None)
                    if next_id is not None:
                        pending.add(executor.submit(execute, base, stage, next_id))

    schedule = [("W", ["retW", "WW"]), ("S", ["retS", "WS", "SS"]), ("W", ["SW"])]
    try:
        for choice, stages in schedule:
            if all((stage, i) in cached for stage in stages for i in ids):
                continue
            with server(a, choice, f"server_{choice}_{stages[0]}.log") as base:
                for stage in stages:
                    phase(base, stage)
    except Exception as error:
        status(a.out, {"state": "failed", "error": str(error), "calls_done": len(cached), "calls_total": planned})
        raise
    assert len(cached) == planned
    print("Actual model grid complete; evaluating frozen routers on NEW evidence.", flush=True)
    predictions = {}; router_seconds = {}
    for stage in ["ret", "qa_W", "qa_S"]:
        bundle = joblib.load(bundle_dir / f"{stage}.joblib")
        fresh = []
        for i in ids:
            row = dict(rows[i])
            if stage != "ret":
                row["evidence_"+stage[-1]] = cached[("ret"+stage[-1], i)]["evidence"]
            fresh.append(row)
        t = time.perf_counter()
        text = bundle["vectorizer"].transform(text_features(row, stage) for row in fresh)
        numeric = np.array([numeric_features(row, stage) for row in fresh])
        feature_seconds = time.perf_counter()-t
        predictions[stage] = {}
        for family, model in bundle["models"].items():
            t = time.perf_counter()
            predictions[stage][family] = model.predict(numeric if family == "random_forest" else text).tolist()
            router_seconds[stage+"/"+family] = {"predict_seconds": time.perf_counter()-t,
                                               "shared_feature_extraction_seconds": feature_seconds}
    policies = {}
    for family in predictions["ret"]:
        policies[family] = [rc + predictions["qa_"+rc][family][j]
                            for j, rc in enumerate(predictions["ret"][family])]
    for cell in ["WW", "WS", "SW", "SS"]:
        policies["always_"+cell] = [cell] * len(ids)
    results = {}
    decisions = []
    for name, picks in policies.items():
        chosen = [cached[(cell, i)] for cell, i in zip(picks, ids)]
        elapsed = [cached[("ret"+cell[0], i)]["seconds"]+cached[(cell, i)]["seconds"] for cell, i in zip(picks, ids)]
        results[name] = {"n": len(ids), "answer_em_pct": 100*float(np.mean([v["answer_em"] for v in chosen])),
                         "answer_f1_pct": 100*float(np.mean([v["answer_f1"] for v in chosen])),
                         "diagnostic_containment_pct": 100*float(np.mean([v["containment"] for v in chosen])),
                         "picks": dict(Counter(picks)), "mean_observed_call_seconds": float(np.mean(elapsed)),
                         "mean_generated_tokens": float(np.mean([
                             cached[("ret"+cell[0], i)]["timings"]["predicted_n"]+
                             cached[(cell, i)]["timings"]["predicted_n"] for cell, i in zip(picks, ids)]))}
        for cell, i in zip(picks, ids):
            decisions.append({"idx": i, "method": name, "cell": cell,
                              "answer_em": cached[(cell, i)]["answer_em"]})
    oracle = 100*float(np.mean([max(cached[(cell, i)]["answer_em"] for cell in ["WW","WS","SW","SS"]) for i in ids]))
    report = {"config": config, "official_metric_sha256": scorer_hash, "policies": results,
              "oracle_em_pct": oracle, "router_timing": router_seconds,
              "runtime_note": "Two concurrent requests share a GPU with other jobs. Cached call times are not isolated routed latency or FLOPs."}
    if (a.out / "migration.json").exists():
        report["execution_provenance"] = json.loads((a.out / "migration.json").read_text())
    (a.out / "results.json").write_text(json.dumps(report, indent=2)+"\n")
    with (a.out / "decisions.jsonl").open("w") as sink:
        for row in decisions:
            sink.write(json.dumps(row)+"\n")
    status(a.out, {"state": "complete", "calls_done": planned, "calls_total": planned, "results": str(a.out / "results.json")})
    print(json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    main()
