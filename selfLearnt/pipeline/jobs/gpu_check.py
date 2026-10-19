"""Quick check that one model starts in vLLM on the GPUs of this job and answers one prompt.
   srun ... python jobs/gpu_check.py q32        (keys as in configs/default.yaml: q7, q32, q72, l8, l70, ...)"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stair.config import load_config, serve_entry, variant  # noqa: E402
from stair.engines import free_gpu, make_engine  # noqa: E402

def main():
    cfg = load_config(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "configs", "default.yaml"))
    for key in sys.argv[1:]:
        t = time.time()
        m = variant(cfg, key)
        eng = make_engine(cfg, serve_entry(m))
        out = eng.generate([eng.encode("You are a helpful assistant.", "What is the capital of France? One word.")], 1, 0.0, 16, [0])[0][0][0]
        print(f"OK {key}: {out.strip()!r} ({time.time() - t:.0f} s)", flush=True)
        del eng
        free_gpu()


if __name__ == "__main__":      # vLLM starts its workers with 'spawn', which re-imports this file
    main()
