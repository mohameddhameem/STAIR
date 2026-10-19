"""
Lab GPU smoke test — Stage-Level Routing project
Machine: 10.4.8.30 (HMCS lab, RTX 6000 Ada 48GB expected)

Run in two phases:
  Phase 1 (no downloads, <10s):   python3 lab_smoketest.py
  Phase 2 (downloads Qwen 1.5B):  python3 lab_smoketest.py --model

Phase 2 pulls ~3GB from HuggingFace and runs one tiny generation on GPU.
"""
import os, sys, shutil, subprocess

# Same allocator guard we use on Colab — must be set before torch import
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DATA_DIR = "/mnt/hdd/jonathan"
PASS, FAIL, WARN = "[PASS]", "[FAIL]", "[WARN]"
problems = 0

def check(label, ok, detail="", warn=False):
    global problems
    tag = PASS if ok else (WARN if warn else FAIL)
    if not ok and not warn:
        problems += 1
    print(f"{tag} {label}" + (f" — {detail}" if detail else ""))

print("=" * 60)
print("PHASE 1 — environment checks")
print("=" * 60)

# 1. Python version
v = sys.version_info
check("Python >= 3.9", v >= (3, 9), f"found {v.major}.{v.minor}.{v.micro}")

# 2. nvidia-smi
try:
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.free",
                          "--format=csv,noheader"], capture_output=True, text=True, timeout=15)
    line = out.stdout.strip()
    check("nvidia-smi / GPU visible", out.returncode == 0 and line, line)
    if "6000" in line:
        print("       -> RTX 6000 Ada confirmed (48GB class: single-GPU GRPO is feasible)")
except FileNotFoundError:
    check("nvidia-smi / GPU visible", False, "nvidia-smi not found — driver missing or not on PATH")

# 3. Data directory writable + free space
try:
    os.makedirs(DATA_DIR, exist_ok=True)
    testfile = os.path.join(DATA_DIR, ".write_test")
    with open(testfile, "w") as f:
        f.write("ok")
    os.remove(testfile)
    total, used, free = shutil.disk_usage(DATA_DIR)
    free_gb = free / 1e9
    check(f"{DATA_DIR} writable", True, f"{free_gb:.0f} GB free of {total/1e9:.0f} GB")
    check("Free space >= 60 GB (models + pools + checkpoints)", free_gb >= 60,
          f"{free_gb:.0f} GB", warn=(free_gb >= 30))
except Exception as e:
    check(f"{DATA_DIR} writable", False, str(e))

# 4. Torch + CUDA
try:
    import torch
    check("torch import", True, torch.__version__)
    cuda_ok = torch.cuda.is_available()
    check("torch.cuda.is_available()", cuda_ok)
    if cuda_ok:
        free_b, total_b = torch.cuda.mem_get_info()
        print(f"       -> GPU: {torch.cuda.get_device_name(0)}, "
              f"{free_b/1e9:.1f} GB free / {total_b/1e9:.1f} GB total")
        x = torch.randn(2048, 2048, device="cuda")
        y = (x @ x).sum().item()
        check("GPU matmul", True, f"result finite: {y == y}")
except ImportError:
    check("torch import", False, "pip install torch  (or use the lab conda env)")
except Exception as e:
    check("GPU compute", False, repr(e))

# 5. Project libraries (needed for the pipeline)
for lib in ["transformers", "peft", "trl", "datasets", "bitsandbytes", "bm25s", "accelerate"]:
    try:
        m = __import__(lib)
        check(f"import {lib}", True, getattr(m, "__version__", "?"))
    except ImportError:
        check(f"import {lib}", False, f"pip install {lib}", warn=True)

# 6. Internet access to HuggingFace (needed to pull Qwen weights once)
try:
    import urllib.request
    urllib.request.urlopen("https://huggingface.co", timeout=10)
    check("HuggingFace reachable", True)
except Exception as e:
    check("HuggingFace reachable", False,
          "no outbound internet? Ask Dhanuja about proxy, or transfer weights via scp", warn=True)

print()
if problems == 0:
    print("PHASE 1 RESULT: machine is usable.")
else:
    print(f"PHASE 1 RESULT: {problems} blocking issue(s) above — fix before Phase 2.")

# ---------------- PHASE 2 ----------------
if "--model" in sys.argv and problems == 0:
    print()
    print("=" * 60)
    print("PHASE 2 — load weak model (Qwen2.5-1.5B) + one generation")
    print("=" * 60)
    os.environ.setdefault("HF_HOME", os.path.join(DATA_DIR, "hf_cache"))
    print(f"HF cache -> {os.environ['HF_HOME']} (keeps home dir clean)")
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    name = "Qwen/Qwen2.5-1.5B-Instruct"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="sdpa")
    msgs = [{"role": "user", "content": "Reply with exactly: LAB GPU OK"}]
    ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=False).to("cuda")
    out = model.generate(ids, max_new_tokens=10, do_sample=False)
    print("Model says:", tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True))
    free_b, _ = torch.cuda.mem_get_info()
    print(f"GPU free after load: {free_b/1e9:.1f} GB")
    print("PHASE 2 RESULT: full stack works. Ready for the big label run.")
elif "--model" in sys.argv:
    print("Skipping Phase 2 until Phase 1 issues are fixed.")
