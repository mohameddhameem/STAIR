"""Start one temporary llama.cpp server, test generation, and release its GPU memory."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=Path("/ssd1/zmcheng/cs707"))
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=18087)
    ap.add_argument("--model", choices=["all", "7b", "1.5b", "instruct"], default="all")
    a = ap.parse_args()
    binary = a.root / "tools/llama.cpp/build/bin/llama-server"
    if not binary.is_file():
        raise SystemExit(f"Build llama-server first: {binary}")
    with socket.socket() as s:
        s.bind(("127.0.0.1", a.port))
    models = json.loads((a.root / "models/manifest.json").read_text())
    selected = {"7b": 0, "1.5b": 1, "instruct": 2}
    if a.model != "all":
        models = [models[selected[a.model]]]
    out = a.root / "runs/gguf_smoke"
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for m in models:
        free = int(subprocess.check_output([
            "nvidia-smi", f"--id={a.gpu}", "--query-gpu=memory.free",
            "--format=csv,noheader,nounits"], text=True).strip())
        required = 8000 if "7B" in m["directory"] else 3000
        if free < required:
            raise SystemExit(f"GPU {a.gpu}: only {free} MiB free; need {required} MiB headroom")
        path = a.root / "models" / m["directory"] / m["filename"]
        command = [str(binary), "-m", str(path), "-ngl", "99", "-c", "1024",
                   "-np", "1", "-b", "128", "-ub", "128", "-t", "2",
                   "--host", "127.0.0.1", "--port", str(a.port)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(a.gpu))
        start = time.monotonic()
        log = out / (m["directory"] + ".log")
        with log.open("w") as sink:
            proc = subprocess.Popen(command, env=env, stdout=sink, stderr=sink)
            try:
                base = f"http://127.0.0.1:{a.port}"
                while True:
                    if proc.poll() is not None:
                        raise RuntimeError(f"Server exited {proc.returncode}; see {log}")
                    if time.monotonic() - start > 180:
                        raise TimeoutError(f"Server startup timed out; see {log}")
                    try:
                        with urllib.request.urlopen(base + "/health", timeout=1) as r:
                            if r.status == 200:
                                break
                    except OSError:
                        pass
                    time.sleep(0.5)
                memory = subprocess.check_output([
                    "nvidia-smi", "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader,nounits"], text=True)
                own = [line.strip() for line in memory.splitlines()
                       if line.split(",")[0].strip() == str(proc.pid)]
                prompt = "The capital of France is"
                if "Instruct" in m["directory"]:
                    prompt = ("<|im_start|>user\nWhat is the capital of France? "
                              "Answer briefly.<|im_end|>\n<|im_start|>assistant\n")
                body = json.dumps({"prompt": prompt, "n_predict": 32,
                                   "temperature": 0, "seed": 707}).encode()
                request = urllib.request.Request(base + "/completion", data=body,
                                                 headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(request, timeout=120) as r:
                    response = json.load(r)
                if not response.get("content", "").strip():
                    raise RuntimeError("Generation returned empty text")
                result = {"repo": m["repo"], "gpu": a.gpu, "context": 1024,
                          "free_mib_before": free, "process_memory_after_load": own,
                          "prompt": prompt, "content": response["content"],
                          "timings": response.get("timings"), "log": str(log)}
                results.append(result)
                print(json.dumps(result, ensure_ascii=False), flush=True)
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait()
        (out / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
        time.sleep(1)


if __name__ == "__main__":
    main()
