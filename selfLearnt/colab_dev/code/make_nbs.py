import os, json
import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "notebooks_generated")
FROZEN_JSON = "/content/drive/MyDrive/0_EngD_assignments/0_Small_Big_Model_Switching/frozen/set5c_gate_v1/frozen_gate.json"   # local mirror, read when generating notebook 17
PROJECT = "/content/drive/MyDrive/0_EngD_assignments/0_Small_Big_Model_Switching"


def nb(cells, gpu_type="A100"):
    n = nbf.v4.new_notebook()
    n.metadata = {"accelerator": "GPU", "colab": {"provenance": [], "gpuType": gpu_type},
                  "kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}}
    n.cells = [nbf.v4.new_markdown_cell(s) if k == "md" else nbf.v4.new_code_cell(s) for k, s in cells]
    return n


MOUNT = 'from google.colab import drive\ndrive.mount("/content/drive")'

GPU_CHECK = '''import subprocess, sys, os, json, shutil, time
gpu = subprocess.run("nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader 2>/dev/null",
                     shell=True, capture_output=True, text=True).stdout.strip()
print("GPU:", gpu or "none")
if not DRY_RUN:
    assert gpu, "No GPU: Runtime -> Change runtime type -> A100 GPU (or set DRY_RUN = True)."
    if "T4" in gpu:
        raise SystemExit("This runtime has a T4, which cannot run this: the 7B model alone (15 GB) does not fit in its memory.\\n"
                         "Runtime -> Change runtime type -> A100 GPU (or G4 / L4) -> Save, then run all cells again.")
    if "A100" not in gpu and "G4" not in gpu and "RTX PRO 6000" not in gpu:
        print("Note: planned for an A100 or G4; this GPU works but timings will differ.")
GPU_TAG = "cpu" if not gpu else gpu.split(",")[0].replace("NVIDIA", "").replace("Tesla", "").strip().split()[0].split("-")[0]
T_START = time.time()'''

INSTALL = '''# vLLM runs only inside the stage processes, so no runtime restart is needed (~3-5 min)
if not DRY_RUN:
    !pip install -q vllm
    !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
    chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                         capture_output=True, text=True)
    print(chk.stdout or chk.stderr[-2000:])
    assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."'''

RUN_STAGE = '''BACKEND = "mock" if DRY_RUN else "vllm"
CODE = "/content/ssw.py"
shutil.copy(f"{PROJECT_DIR}/code/ssw.py", CODE)
shutil.copy(CODE, f"{RUN_DIR}/code/ssw.py")          # keep the exact code with the results

def minutes_left():
    return max(1.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name):
    """Runs one ssw.py command in a fresh process; streams output here and into logs/<log_name>."""
    cmd = [sys.executable, "-u", CODE] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log:
        log.write("\\n$ " + " ".join(cmd) + "\\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            if any(k in line for k in ("it/s]", "s/it]", "INFO ", "(EngineCore")):
                continue                      # keep the notebook readable; everything is in the log file
            print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = round(time.time() - t0)
    json.dump(TIMES, open(f"{RUN_DIR}/logs/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    if p.returncode != 0:
        raise RuntimeError(f"stage failed - see {RUN_DIR}/logs/{log_name}")
TIMES = {}'''


def copy_models_cell(keys_expr):
    return f'''LOCAL = "/content/models"
MODELS = {{
    "S": {{"id": "Qwen/Qwen2.5-3B-Instruct", "dir": "Qwen2.5-3B-Instruct", "params": 3.09e9, "nominal": 3e9}},
    "L": {{"id": "Qwen/Qwen2.5-7B-Instruct", "dir": LARGE_MODEL_DIR,       "params": 7.62e9, "nominal": 7e9}},
}}
_t = time.time()
for key in {keys_expr}:
    m = MODELS[key] if key in MODELS else {{"dir": key}}
    src, dst = f"{{PROJECT_DIR}}/models/{{m['dir']}}", f"{{LOCAL}}/{{m['dir']}}"
    m["path"] = dst
    if DRY_RUN:
        continue
    assert os.path.exists(f"{{src}}/config.json"), f"{{src}} missing - run 00_setup_download_qwen7b first."
    if not os.path.exists(f"{{dst}}/config.json"):
        print("copying", m["dir"], "to local disk ...")
        shutil.copytree(src, dst, dirs_exist_ok=True)
for m in MODELS.values():
    m["path"] = f"{{LOCAL}}/{{m['dir']}}"
json.dump(MODELS, open(f"{{RUN_DIR}}/models.json", "w"), indent=1)
TIMES["copy_models"] = round(time.time() - _t)
print(f"models ready in {{time.time() - _t:.0f}}s")'''


def run_dir_cell(experiment_var, extra=""):
    return f'''from datetime import datetime
from zoneinfo import ZoneInfo
if RESUME_RUN_DIR:
    RUN_DIR = RESUME_RUN_DIR
else:
    stamp = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
    RUN_DIR = f"{{PROJECT_DIR}}/runs/{{{experiment_var}}}/{{stamp}}_{{GPU_TAG}}" + ("_dryrun" if DRY_RUN else "")
for sub in ["logs", "code"]:
    os.makedirs(f"{{RUN_DIR}}/{{sub}}", exist_ok=True)
print("Run folder:", RUN_DIR){extra}'''


def show_cell(files):
    return f'''from IPython.display import Image, Markdown, display
for f in {files}:
    p = f"{{RUN_DIR}}/{{f}}"
    if not os.path.exists(p):
        continue
    display(Markdown(open(p).read()) if p.endswith(".md") else Image(p))
TIMES["total_notebook"] = round(time.time() - T_START)
json.dump(TIMES, open(f"{{RUN_DIR}}/logs/stage_times.json", "w"), indent=1)
print("Wall-clock (s):", TIMES)'''


FIND_STEP0 = '''import glob
if not STEP0_RUN_DIR:   # empty = the latest finished results-table run in runs/<STEP0_EXPERIMENT> (dry runs skipped)
    done = [p for p in sorted(glob.glob(f"{PROJECT_DIR}/runs/{STEP0_EXPERIMENT}/*/table/per_question.csv")) if "_dryrun" not in p or DRY_RUN]
    assert done, f"No finished results table in runs/{STEP0_EXPERIMENT} - run notebook {'04b' if 'cot' in STEP0_EXPERIMENT else '04'} first."
    STEP0_RUN_DIR = os.path.dirname(os.path.dirname(done[-1]))
assert os.path.exists(f"{STEP0_RUN_DIR}/table/per_question.csv"), f"{STEP0_RUN_DIR} is not a finished results-table run."
print("Results table:", STEP0_RUN_DIR)
DATA_DIR = json.load(open(f"{STEP0_RUN_DIR}/config.json"))["data_dir"]'''

# ------------------------------------------------------------------------------ 00
nb0 = nb([
    ("md", """# 00 · Setup: download Qwen2.5-7B-Instruct to Google Drive

Downloads the large model into `models/` in your project folder (once). The small model, Qwen2.5-3B-Instruct, is already there.

- **Default:** `Qwen/Qwen2.5-7B-Instruct`, full precision, ~15 GB. Runs on an A100, G4 or L4.
- **Option:** the AWQ (compressed) version, ~5.5 GB, as used in the AgentTTS paper. If you use it, set `LARGE_MODEL_DIR = "Qwen2.5-7B-Instruct-AWQ"` in notebooks 04–06.
- **Option:** Qwen3-4B as the agent for Scripts 1 and 2 (thinking mode, ~8 GB). Not needed by default: the agent is Qwen2.5-7B.

Qwen models are not gated, so no licence approval is needed. A CPU runtime is fine."""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
DOWNLOAD = {{
    "Qwen2.5-7B-Instruct":     "Qwen/Qwen2.5-7B-Instruct",        # default large model (~15 GB)
    # "Qwen2.5-7B-Instruct-AWQ": "Qwen/Qwen2.5-7B-Instruct-AWQ",  # optional compressed version (~5.5 GB)
    # "Qwen3-4B":                "Qwen/Qwen3-4B",                  # optional agent with thinking mode (~8 GB)
}}
DELETE_LLAMA = False   # True removes models/Llama-3.2-3B-Instruct (6.4 GB); the plan is Qwen-only now
ALLOW  = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.tiktoken"]
IGNORE = ["original/*", "*.pth", "*.bin", "*.gguf"]'''),
    ("code", MOUNT),
    ("code", '''import os, shutil, glob, time
!pip install -q -U huggingface_hub hf_transfer
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
try:                                    # optional: a token only raises download rate limits
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    tok = tok.get("value") if isinstance(tok, dict) else tok
except Exception:
    tok = None
from huggingface_hub import snapshot_download

MODEL_DIR = f"{PROJECT_DIR}/models"
os.makedirs(MODEL_DIR, exist_ok=True)
def complete(p):
    return os.path.exists(f"{p}/config.json") and glob.glob(f"{p}/*.safetensors") and os.path.exists(f"{p}/tokenizer.json")

for name, repo in DOWNLOAD.items():
    dst = f"{MODEL_DIR}/{name}"
    if complete(dst):
        print(f"✓ {name} already in Drive"); continue
    t = time.time()
    tmp = f"/content/_dl/{name}"
    print(f"downloading {repo} ...")
    snapshot_download(repo_id=repo, local_dir=tmp, allow_patterns=ALLOW, ignore_patterns=IGNORE, token=tok or None)
    print(f"copying to {dst} ...")
    shutil.copytree(tmp, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
    shutil.rmtree(tmp)
    print(f"✓ {name} in {time.time() - t:.0f}s")

if DELETE_LLAMA and os.path.isdir(f"{MODEL_DIR}/Llama-3.2-3B-Instruct"):
    shutil.rmtree(f"{MODEL_DIR}/Llama-3.2-3B-Instruct"); print("deleted Llama-3.2-3B-Instruct")'''),
    ("code", '''!du -sh "{MODEL_DIR}"/*
total, used, free = shutil.disk_usage("/content/drive")
print(f"Drive: {used/1e9:.1f} GB used, {free/1e9:.1f} GB free")'''),
    ("md", "Done. Next: **04_step0_results_table.ipynb** on an A100 GPU."),
])

# ------------------------------------------------------------------------------ 04
nb4 = nb([
    ("md", """# 04 · Step 0: shared per-question results table (GPU, ~30 min on A100)

Runs every configuration once on the same questions, so Scripts 1 and 2 can be compared fairly and rerun in minutes:

- **16 configurations** = retriever Qwen 3B/7B × answerer Qwen 3B/7B × retrieval samples {1, 3} × answer samples {1, 5}
- **200 dev + 200 test questions**, random and non-overlapping, **20 paragraphs** each (2 gold + 18 BM25 distractors)
- pipeline: retrieve → merge (vote) → answer samples → fuse, with a short-answer line scored by HotpotQA EM/F1

Output: `runs/env_qwen_hotpotqa/<timestamp>_<GPU>/` with `table/per_question.csv` (one row per question × config: answer, EM, F1, tokens, FLOPs, estimated GPU seconds), `table/signals.csv`, `table/configs.csv`, `table/summary.md` (with the **go/no-go check**) and a plot. `status.json` shows live progress; every stage resumes if interrupted (set `RESUME_RUN_DIR`).

**Runtime → Change runtime type → A100 GPU** (G4 also works; L4 is ~2× slower; T4 is refused)."""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "env_qwen_hotpotqa"
LARGE_MODEL_DIR = "Qwen2.5-7B-Instruct"    # or "Qwen2.5-7B-Instruct-AWQ"

DEV_N, TEST_N, N_CANDIDATES, DATA_SEED = 200, 200, 20, 42   # 40/100 took 20 min on an A100; 200/200 is estimated ~30 min
TIME_LIMIT_MIN = 55          # every stage stops cleanly at this limit and keeps what is done
RESUME_RUN_DIR = None        # an interrupted run folder, to continue it
DRY_RUN = False              # True = fake models on CPU, tests the whole notebook in ~1 min'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("code", run_dir_cell("EXPERIMENT", extra='''
DATA_DIR = f"{PROJECT_DIR}/data/hotpotqa_q/dev{DEV_N}_test{TEST_N}_c{N_CANDIDATES}_s{DATA_SEED}"
json.dump({"experiment": EXPERIMENT, "data_dir": DATA_DIR, "large_model_dir": LARGE_MODEL_DIR, "gpu": gpu,
           "dev_n": DEV_N, "test_n": TEST_N, "n_candidates": N_CANDIDATES, "data_seed": DATA_SEED, "dry_run": DRY_RUN,
           "started": datetime.now(ZoneInfo("Asia/Singapore")).isoformat()}, open(f"{RUN_DIR}/config.json", "w"), indent=1)''')),
    ("code", RUN_STAGE),
    ("md", "### Data: random dev/test samples (CPU, <1 min, built once and reused)"),
    ("code", '''if not os.path.exists(f"{DATA_DIR}/test.json"):
    run_stage(["build_data", "--hotpot_dev", f"{PROJECT_DIR}/AgentTTS/data/HotpotQA/hotpotqa-dev.json", "--out_dir", DATA_DIR,
               "--dev_n", DEV_N, "--test_n", TEST_N, "--n_candidates", N_CANDIDATES, "--seed", DATA_SEED], "00_build_data.log")
else:
    print("using existing data in", DATA_DIR)'''),
    ("code", INSTALL + '\nTIMES["install_vllm"] = round(time.time() - T_START)'),
    ("code", copy_models_cell('["S", "L"]')),
    ("md", "### Retrieval (3 samples per question) with 3B, then 7B"),
    ("code", '''COMMON = ["--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json", "--backend", BACKEND]
for k in ["S", "L"]:
    run_stage(["retrieve", "--model_key", k, "--time_limit_min", minutes_left()] + COMMON, f"01_retrieve_{k}.log")'''),
    ("md", "### Answer (5 samples) + fuse, with 7B then 3B, on both retrievers' selections"),
    ("code", '''for k in ["L", "S"]:
    run_stage(["answer", "--model_key", k, "--time_limit_min", minutes_left()] + COMMON, f"02_answer_{k}.log")'''),
    ("md", "### Table, costs, signals and the go/no-go check"),
    ("code", '''run_stage(["table", "--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json"], "03_table.log")
print("\\nSTEP0_RUN_DIR for notebooks 05 and 06:\\n", RUN_DIR)'''),
    ("code", show_cell('["table/summary.md", "table/configs_em_vs_flops.png"]')),
])

# ------------------------------------------------------------------------------ 05
AGENT_SETTINGS = '''AGENT_MODEL_DIR = "Qwen2.5-7B-Instruct"   # agent LLM; "Qwen3-4B" (downloaded in 00) + THINKING = True for thinking mode
THINKING = False'''

nb5 = nb([
    ("md", """# 05 · AgentTTS baseline — one global configuration (GPU, ~15–20 min)

**Set 1:** `ALLOW_COT = False` (as in the paper). **Set 2b control:** `ALLOW_COT = True` searches the 24-setup table from notebook 04b,
where the answer model can also be "7B/3B with chain-of-thought"; the budgets stay exactly the Set 1 budgets.

The AgentTTS search, ported from the repo (`AgentTTS.py`, same prompts and budget formula), runs on the **dev** rows of the Step 0 table:
initial trials from Insight 1 → written guidelines → new candidates → budget check → dev score, until 8 distinct configurations are tried.
The chosen configuration is then applied unchanged to **every test question** (scores read from the table; the agent never sees test results).

Also reported: the **dev-optimal global config** (exhaustive search of the 16), the ceiling for any single-configuration method.
Runs 3 budgets × 3 seeds, batched together on the GPU. The agent's thinking is saved in `traces/` but never fed back (condition B1).

Output: `runs/agenttts_global/<timestamp>_<GPU>/` (`summary.md`, `agenttts_runs.csv`, `dev_optimal.csv`, `traces/`)."""),
    ("code", f'''# ==== settings ====
PROJECT_DIR   = "{PROJECT}"
ALLOW_COT     = False         # False = Set 1 (paper setting); True = Set 2b control (needs notebook 04b)
STEP0_EXPERIMENT = "env_qwen_hotpotqa_cot" if ALLOW_COT else "env_qwen_hotpotqa"
STEP0_RUN_DIR = ""            # empty = latest finished run in runs/<STEP0_EXPERIMENT>; or paste a run folder
EXPERIMENT    = "set2b_agenttts_cot" if ALLOW_COT else "agenttts_global"
{AGENT_SETTINGS}
LARGE_MODEL_DIR = "Qwen2.5-7B-Instruct"
BUDGET_QUANTILES = [0.25, 0.5, 1.0]   # budgets = these quantiles of the 16 configs' Eq. 4 costs
SEEDS, ITERATIONS, BATCH_SIZE, MAX_TRIALS = 3, 4, 2, 8
TIME_LIMIT_MIN = 55
RESUME_RUN_DIR = None
DRY_RUN = False'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("code", FIND_STEP0),
    ("code", run_dir_cell("EXPERIMENT", extra='''
json.dump({"experiment": EXPERIMENT, "step0_run_dir": STEP0_RUN_DIR, "allow_cot": ALLOW_COT, "agent": AGENT_MODEL_DIR, "thinking": THINKING,
           "budget_quantiles": BUDGET_QUANTILES, "seeds": SEEDS, "iterations": ITERATIONS, "batch_size": BATCH_SIZE,
           "max_trials": MAX_TRIALS, "gpu": gpu, "dry_run": DRY_RUN}, open(f"{RUN_DIR}/config.json", "w"), indent=1)''')),
    ("code", RUN_STAGE),
    ("code", INSTALL + '\nTIMES["install_vllm"] = round(time.time() - T_START)'),
    ("code", copy_models_cell('[AGENT_MODEL_DIR]') + '\nAGENT_PATH = f"{LOCAL}/{AGENT_MODEL_DIR}"'),
    ("code", '''run_stage(["agenttts", "--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--table_dir", f"{STEP0_RUN_DIR}/table",
           "--agent_path", AGENT_PATH, "--backend", BACKEND, "--budget_quantiles", *BUDGET_QUANTILES,
           "--seeds", SEEDS, "--iterations", ITERATIONS, "--batch_size", BATCH_SIZE, "--max_trials", MAX_TRIALS,
           "--time_limit_min", minutes_left(), "--agent_max_tokens", 4096 if THINKING else 1536]
          + (["--thinking"] if THINKING else []) + (["--allow_cot"] if ALLOW_COT else []), "01_agenttts.log")
print("\\nAgentTTS run (picked up automatically by notebook " + ("07" if ALLOW_COT else "06") + "):\\n", RUN_DIR)'''),
    ("code", show_cell('["summary.md"]')),
])

# ------------------------------------------------------------------------------ 04b
nb4b = nb([
    ("md", """# 04b · Set 2b data: add chain-of-thought answers to the Set 1 results table (GPU, ~20–30 min on A100)

Copies the Set 1 table's inputs (the **same 400 questions, the same retrievals, the same plain answers**) into a new run folder and
adds answers written **with chain-of-thought (CoT)**: the answer prompt asks the model to write out the first-hop fact, then the second,
then the short `Answer:` line (up to 512 tokens instead of 384). Everything else is unchanged: 5 samples, the same fuser, the same scoring.

CoT is added on the **1-sample retrievals** only, for both answer models, so the table grows from 16 to **24 setups**:
`S1-Sc1, S1-Sc5, S1-Lc1, S1-Lc5, L1-Sc1, L1-Sc5, L1-Lc1, L1-Lc5` (`c` = chain-of-thought).
The Set 1 run folder is only read, never written. The last cell checks that the 16 Set 1 setups come out identical.

Output: `runs/env_qwen_hotpotqa_cot/<timestamp>_<GPU>/` (same layout as notebook 04). Used by notebook 05 with `ALLOW_COT = True` and by notebook 07.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
STEP0_EXPERIMENT = "env_qwen_hotpotqa"   # Set 1 results tables live here
STEP0_RUN_DIR = ""                       # empty = latest finished Set 1 table (20260928-155333_A100); or paste it
EXPERIMENT  = "env_qwen_hotpotqa_cot"
LARGE_MODEL_DIR = "Qwen2.5-7B-Instruct"
TIME_LIMIT_MIN = 55          # every stage stops cleanly at this limit and keeps what is done
RESUME_RUN_DIR = None        # an interrupted run folder, to continue it
DRY_RUN = False              # True = fake models on CPU, tests the whole notebook in ~1 min'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("code", FIND_STEP0 + '''
SET1_DIR = STEP0_RUN_DIR
for f in ["retrieval_S", "retrieval_L", "answers_S", "answers_L", "answer_samples_S", "answer_samples_L"]:
    assert os.path.exists(f"{SET1_DIR}/step0/{f}.jsonl"), f"{SET1_DIR}/step0/{f}.jsonl missing"'''),
    ("code", run_dir_cell("EXPERIMENT", extra='''
# copy (never move) the Set 1 inputs; on resume, files already here are kept
os.makedirs(f"{RUN_DIR}/step0", exist_ok=True)
os.makedirs(f"{RUN_DIR}/stages", exist_ok=True)
for f in os.listdir(f"{SET1_DIR}/step0"):
    if not os.path.exists(f"{RUN_DIR}/step0/{f}"):
        shutil.copy(f"{SET1_DIR}/step0/{f}", f"{RUN_DIR}/step0/{f}")
for f in os.listdir(f"{SET1_DIR}/stages"):
    if not os.path.exists(f"{RUN_DIR}/stages/{f}"):
        shutil.copy(f"{SET1_DIR}/stages/{f}", f"{RUN_DIR}/stages/{f}")
print("copied Set 1 retrievals, plain answers and stage timings from", SET1_DIR)
if not os.path.exists(f"{RUN_DIR}/config.json"):
    json.dump({"experiment": EXPERIMENT, "set": "2b-data", "source_set1_run": SET1_DIR, "data_dir": DATA_DIR,
               "large_model_dir": LARGE_MODEL_DIR, "gpu": gpu, "cot_answer_max_tokens": 512, "dry_run": DRY_RUN,
               "started": datetime.now(ZoneInfo("Asia/Singapore")).isoformat()}, open(f"{RUN_DIR}/config.json", "w"), indent=1)''')),
    ("code", RUN_STAGE),
    ("code", INSTALL + '\nTIMES["install_vllm"] = round(time.time() - T_START)'),
    ("code", copy_models_cell('["S", "L"]')),
    ("md", "### Chain-of-thought answers (5 samples) + fuse, with 7B then 3B, on both retrievers' 1-sample selections"),
    ("code", '''COMMON = ["--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json", "--backend", BACKEND]
for k in ["L", "S"]:
    run_stage(["answer", "--cot", "--model_key", k, "--time_limit_min", minutes_left()] + COMMON, f"02_answer_{k}c.log")'''),
    ("md", "### 24-setup table"),
    ("code", '''run_stage(["table", "--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json"], "03_table.log")
print("\\nResults table for notebooks 05 (ALLOW_COT = True) and 07:\\n", RUN_DIR)'''),
    ("md", "### Check: the 16 Set 1 setups must be unchanged"),
    ("code", '''import pandas as pd
a = pd.read_csv(f"{SET1_DIR}/table/per_question.csv")
b = pd.read_csv(f"{RUN_DIR}/table/per_question.csv")
m = a.merge(b, on=["qid", "config"], suffixes=("_set1", "_new"))
diff = {c: float((m[c + "_set1"] - m[c + "_new"]).abs().max()) for c in ["EM", "F1", "flops", "est_gpu_s", "budget_eq4"]}
print(f"setups: Set 1 {a.config.nunique()}, now {b.config.nunique()} | matched rows {len(m)} of {len(a)} | max differences {diff}")
print("OK - Set 1 setups unchanged" if len(m) == len(a) and max(diff.values()) < 1e-9 else "WARNING - Set 1 setups changed, check before using this table")
cot = b[b.ans_model.str.endswith("c")]
print(f"CoT answers: mean {cot.ans_output_tokens.div(cot.ans_k).mean():.0f} output tokens per sample "
      f"(plain: {b[~b.ans_model.str.endswith('c')].ans_output_tokens.div(b.ans_k).mean():.0f})")
json.dump({"set1_rows_matched": int(len(m)), "set1_rows": int(len(a)), "max_diff": diff},
          open(f"{RUN_DIR}/table/set1_check.json", "w"), indent=1)'''),
    ("code", show_cell('["table/summary.md", "table/configs_em_vs_flops.png"]')),
])


# ------------------------------------------------------------------------------ 06 / 07
def switching_nb(number, title, intro, step0_exp, routes, experiment_expr, router_default=2, cot=False):
    return nb([
        ("md", f"""# {number} · {title} (GPU, ~25–30 min)

{intro}

**Router v2** (Set 2) changes, each fixing a Set 1 flaw:

| Set 1 (router v1) | Set 2 (router v2) |
|---|---|
| 7–16 possible setups as routes | a short route menu: `{"`, `".join(routes)}` |
| agent's rule list grew each round (31 / 84 rules) | the new list **replaces** the old; at most `MAX_RULES` lines, **enforced** when parsing |
| conditions on topics, free text | only `IF <feature> <op> <value> [AND …] THEN <route>` over named, checkable features |
| a 3B model applied the rules | rules applied **exactly** by code, and separately by the 3B (fidelity reported) |
| classifier labels dominated by questions nobody solves | **cost-weighted** classifier: questions where the route makes no difference carry no weight |
| — | agent sees route accuracy by feature value, and the questions where routes disagree first |

`ROUTER_VERSION = 1` reruns the Set 1 router exactly.
Also reported: each route used alone, the dev-optimal global setup (over the whole table), the AgentTTS picks from notebook 05, and two hindsight oracles.

Output: `runs/<EXPERIMENT>/<timestamp>_<GPU>/` (`summary.md`, `policies.csv`, `test_decisions.csv`, `rules_v2_*.txt`, `nl_rules_log.json`, plot, `speedcheck.csv`)."""),
        ("code", f'''# ==== settings ====
PROJECT_DIR     = "{PROJECT}"
STEP0_EXPERIMENT = "{step0_exp}"
STEP0_RUN_DIR   = ""          # empty = latest finished run in runs/<STEP0_EXPERIMENT>; or paste a run folder
SCRIPT1_RUN_DIR = "auto"      # "auto" = latest notebook-05 run on the same table{" with ALLOW_COT = True" if cot else ""}; "" = leave out
ROUTER_VERSION  = {router_default}           # 2 = Set 2 router; 1 = the Set 1 router
ROUTES          = {routes!r}   # router v2 route menu (cheapest first)
EXPERIMENT      = {experiment_expr}
{AGENT_SETTINGS}
LARGE_MODEL_DIR = "Qwen2.5-7B-Instruct"
LAMBDAS    = [0.0, 0.01, 0.03, 0.1]   # accuracy-vs-cost trade-offs (utility = EM - lambda x cost/cheapest)
NL_LAMBDAS = [0.03]                   # trade-offs for the NL rule router (each costs NL_ROUNDS agent calls x 2 settings)
NL_ROUNDS, MAX_RULES, MAX_EVIDENCE = 5, 8, 90
SPEEDCHECK, SPEED_N = True, 10        # live timing of the main policies on SPEED_N test questions
TIME_LIMIT_MIN = 55
RESUME_RUN_DIR = None
DRY_RUN = False'''),
        ("code", GPU_CHECK),
        ("code", MOUNT),
        ("code", FIND_STEP0),
        ("code", f'''if SCRIPT1_RUN_DIR == "auto":   # latest AgentTTS run (notebook 05) built on this same results table
    SCRIPT1_RUN_DIR = ""
    runs = sorted(glob.glob(f"{{PROJECT_DIR}}/runs/*/*/agenttts_runs.csv"), key=lambda p: os.path.basename(os.path.dirname(p)))
    for p in runs:
        d = os.path.dirname(p)
        c = json.load(open(f"{{d}}/config.json"))
        if ("_dryrun" not in d or DRY_RUN) and c.get("step0_run_dir") == STEP0_RUN_DIR and bool(c.get("allow_cot")) == {cot}:
            SCRIPT1_RUN_DIR = d
print("AgentTTS results from notebook 05:", SCRIPT1_RUN_DIR or "none yet (the comparison will leave them out)")'''),
        ("code", run_dir_cell("EXPERIMENT", extra='''
json.dump({"experiment": EXPERIMENT, "step0_run_dir": STEP0_RUN_DIR, "script1_run_dir": SCRIPT1_RUN_DIR,
           "router_version": ROUTER_VERSION, "routes": ROUTES, "agent": AGENT_MODEL_DIR, "thinking": THINKING,
           "lambdas": LAMBDAS, "nl_lambdas": NL_LAMBDAS, "nl_rounds": NL_ROUNDS, "max_rules": MAX_RULES, "max_evidence": MAX_EVIDENCE,
           "gpu": gpu, "dry_run": DRY_RUN}, open(f"{RUN_DIR}/config.json", "w"), indent=1)''')),
        ("code", RUN_STAGE),
        ("code", INSTALL + '\nTIMES["install_vllm"] = round(time.time() - T_START)'),
        ("code", copy_models_cell('["S", "L", AGENT_MODEL_DIR]') + '\nAGENT_PATH = f"{LOCAL}/{AGENT_MODEL_DIR}"'),
        ("md", "### Routers (agent and 3B rule reader share the GPU: 50% / 30% of memory)"),
        ("code", '''run_stage(["switching", "--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--table_dir", f"{STEP0_RUN_DIR}/table",
           "--models_json", f"{RUN_DIR}/models.json", "--agent_path", AGENT_PATH, "--backend", BACKEND,
           "--router_version", ROUTER_VERSION, "--routes", *ROUTES, "--max_evidence", MAX_EVIDENCE,
           "--lambdas", *LAMBDAS, "--nl_lambdas", *NL_LAMBDAS, "--nl_rounds", NL_ROUNDS, "--max_rules", MAX_RULES,
           "--time_limit_min", minutes_left(), "--agent_max_tokens", 4096 if THINKING else 1536, "--max_model_len", 32768]
          + (["--thinking"] if THINKING else []) + (["--script1_dir", SCRIPT1_RUN_DIR] if SCRIPT1_RUN_DIR else []),
          "01_switching.log")'''),
        ("md", "### Live speed check (3B and 7B loaded together, one question at a time)"),
        ("code", '''if SPEEDCHECK and minutes_left() > 8:
    run_stage(["speedcheck", "--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json",
               "--switching_dir", RUN_DIR, "--n_questions", SPEED_N, "--backend", BACKEND], "02_speedcheck.log")
else:
    print("speed check skipped")'''),
        ("code", show_cell('["summary.md", "policies_em_vs_flops.png"]') + '''
import glob as _g
for p in sorted(_g.glob(f"{RUN_DIR}/rules_v2_*.txt")):
    print(f"\\n--- {os.path.basename(p)} ---\\n" + open(p).read())
if os.path.exists(f"{RUN_DIR}/speedcheck.csv"):
    import pandas as pd
    display(pd.read_csv(f"{RUN_DIR}/speedcheck.csv").round(2))'''),
    ])


nb6 = switching_nb("06", "Per-question switching between Qwen 3B and 7B — Set 2a (router v2)",
    "**Set 2a:** the Set 1 results table (16 setups, no chain-of-thought), with the router rebuilt. "
    "Compared against Set 1's router v1 run (`runs/switching/20260929-100917_A100`), this isolates the effect of the router fixes.",
    "env_qwen_hotpotqa", ["S1-S5", "S1-L5", "L1-L5"], '"set2a_switching" if ROUTER_VERSION == 2 else "switching"')
nb7 = switching_nb("07", "Per-question switching with a chain-of-thought route — Set 2b",
    "**Set 2b:** the 24-setup table from notebook 04b. The route menu gains a fourth, most expensive rung: "
    "`L1-Lc5` (7B retrieval, 7B answers **with chain-of-thought** ×5). Compared against Set 2a (same router, no CoT route) and "
    "against AgentTTS with CoT allowed (notebook 05, `ALLOW_COT = True`).",
    "env_qwen_hotpotqa_cot", ["S1-S5", "S1-L5", "L1-L5", "L1-Lc5"], '"set2b_switching_cot"', cot=True)

# ------------------------------------------------------------------------------ Set 3: 08 / 09
COPY_SRC = '''SRC_DIR = STEP0_RUN_DIR   # the 24-setup table from notebook 04b: its retrievals and answers are the baselines here
os.makedirs(f"{RUN_DIR}/step0", exist_ok=True)
os.makedirs(f"{RUN_DIR}/stages", exist_ok=True)
for sub in ["step0", "stages"]:
    for f in os.listdir(f"{SRC_DIR}/{sub}"):
        if not os.path.exists(f"{RUN_DIR}/{sub}/{f}"):
            shutil.copy(f"{SRC_DIR}/{sub}/{f}", f"{RUN_DIR}/{sub}/{f}")
print("copied baseline retrievals, answers and stage timings from", SRC_DIR)'''


def set3_nb(number, title, intro, experiment, settings_extra, config_extra, stage_cells):
    return nb([
        ("md", f"# {number} · {title}\n\n{intro}\n\n**Runtime → Change runtime type → A100 GPU.**"),
        ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
STEP0_EXPERIMENT = "env_qwen_hotpotqa_cot"   # the 24-setup table from notebook 04b
STEP0_RUN_DIR = ""                           # empty = latest finished run there (20260929-140148_A100); or paste it
EXPERIMENT  = "{experiment}"
LARGE_MODEL_DIR = "Qwen2.5-7B-Instruct"
{settings_extra}
TIME_LIMIT_MIN = 55          # every stage stops cleanly at this limit and keeps what is done
RESUME_RUN_DIR = None        # an interrupted run folder, to continue it
DRY_RUN = False              # True = fake models on CPU, tests the whole notebook in ~1 min'''),
        ("code", GPU_CHECK),
        ("code", MOUNT),
        ("code", FIND_STEP0),
        ("code", run_dir_cell("EXPERIMENT", extra="\n" + COPY_SRC + f'''
if not os.path.exists(f"{{RUN_DIR}}/config.json"):
    json.dump({{"experiment": EXPERIMENT, "set": "3", "source_table_run": SRC_DIR, "data_dir": DATA_DIR, "gpu": gpu,
               "dry_run": DRY_RUN, {config_extra}
               "started": datetime.now(ZoneInfo("Asia/Singapore")).isoformat()}}, open(f"{{RUN_DIR}}/config.json", "w"), indent=1)''')),
        ("code", RUN_STAGE),
        ("code", INSTALL + '\nTIMES["install_vllm"] = round(time.time() - T_START)'),
        ("code", copy_models_cell('["S", "L"]')),
        ("code", '''COMMON = ["--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json", "--backend", BACKEND]'''),
    ] + stage_cells + [
        ("md", "### Tables: retrieval quality, accuracy by retrieval source x answerer, paired differences"),
        ("code", '''run_stage(["diagnose", "--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json"], "99_diagnose.log")'''),
        ("code", show_cell('["diagnose/summary.md"]')),
    ])


nb8 = set3_nb("08", "Set 3a: answering ceiling — answerers given the two gold paragraphs (GPU, ~25–30 min on A100)",
    """Removes retrieval errors: every answerer gets **exactly the two paragraphs HotpotQA marks as needed** (the "gold" paragraphs),
nothing else. This shows how well each answerer can answer when retrieval is perfect.

- Same 400 questions (200 dev + 200 test), same answer prompt, 5 samples + the same fuser, same scoring as Sets 1–2.
- Four answerers: 3B, 3B + chain-of-thought (CoT), 7B, 7B + CoT.
- The final table puts these next to the same answerers on the 3B and 7B retrievals (copied from notebook 04b), so the gap
  between "gold" and "retrieved" is the cost of imperfect retrieval, and the gap between answerers on "gold" is pure answering skill.

Output: `runs/set3_gold_answering/<timestamp>_<GPU>/diagnose/summary.md` (+ per-question CSVs).""",
    "set3_gold_answering", "", '"mode": "gold paragraphs", "answerers": ["S", "Sc", "L", "Lc"],',
    [("md", "### Answer from the gold paragraphs: 3B then 7B, each without and with CoT"),
     ("code", '''for k in ["S", "L"]:
    run_stage(["answer", "--gold", "--model_key", k, "--out_suffix", "__gold", "--time_limit_min", minutes_left()] + COMMON, f"01_gold_{k}.log")
    run_stage(["answer", "--gold", "--cot", "--model_key", k, "--out_suffix", "__gold", "--time_limit_min", minutes_left()] + COMMON, f"01_gold_{k}c.log")''')])

nb9 = set3_nb("09", "Set 3b: can chain-of-thought fix the 3B's retrieval? (GPU, ~35–40 min on A100)",
    """The 3B retriever hands over both needed paragraphs for only ~15% of questions (the 7B: ~45%). Here the **3B retrieves
with chain-of-thought**: it first names what the first part of the question needs and which paragraph has it, then uses that
fact to find the second paragraph, and ends with a `Selected references:` line. Only that final line counts as its selection.

- Same 400 questions, same 20 candidate paragraphs, 3 retrieval samples (as before; results for 1 and 3 merged samples).
- **Retrieval quality**: how often both gold paragraphs are found, one, or none; paragraph F1; tokens and compute.
- **End-to-end**: the four answerers (3B, 3B + CoT, 7B, 7B + CoT) answer on the new 3B + CoT selections, compared with the
  same answerers on the plain 3B and 7B selections (from notebook 04b), and on the gold paragraphs if notebook 08 has run.
- `RUN_7B_COT_RETRIEVAL = True` also runs the 7B with CoT retrieval as a reference (~+15 min).

Output: `runs/set3_retrieval_cot/<timestamp>_<GPU>/diagnose/summary.md` (+ per-question CSVs).""",
    "set3_retrieval_cot", "RUN_7B_COT_RETRIEVAL = False   # True adds 7B + CoT retrieval as a reference (~+15 min)\nINCLUDE_GOLD_FROM_08 = True    # add notebook 08's gold-paragraph answers to the final table, if that run exists",
    '"mode": "chain-of-thought retrieval", "run_7b_cot_retrieval": RUN_7B_COT_RETRIEVAL,',
    [("md", "### Retrieval with chain-of-thought (3B; optionally 7B)"),
     ("code", '''RET_TAGS = ["Sc"] + (["Lc"] if RUN_7B_COT_RETRIEVAL else [])
for k in ["S"] + (["L"] if RUN_7B_COT_RETRIEVAL else []):
    run_stage(["retrieve", "--cot", "--model_key", k, "--time_limit_min", minutes_left()] + COMMON, f"01_retrieve_{k}c.log")'''),
     ("md", "### Answer on the new selections: 3B then 7B, each without and with CoT"),
     ("code", '''for k in ["S", "L"]:
    run_stage(["answer", "--model_key", k, "--ret_tags", *RET_TAGS, "--ret_ks", 1, "--out_suffix", "__retc",
               "--time_limit_min", minutes_left()] + COMMON, f"02_answer_{k}.log")
    run_stage(["answer", "--cot", "--model_key", k, "--ret_tags", *RET_TAGS, "--ret_ks", 1, "--out_suffix", "__retc",
               "--time_limit_min", minutes_left()] + COMMON, f"02_answer_{k}c.log")'''),
     ("code", '''if INCLUDE_GOLD_FROM_08:   # latest notebook-08 run built on the same table
    import glob as _g
    runs08 = [d for d in sorted(_g.glob(f"{PROJECT_DIR}/runs/set3_gold_answering/*"))
              if ("_dryrun" not in d or DRY_RUN) and os.path.exists(f"{d}/diagnose/summary.md")
              and json.load(open(f"{d}/config.json")).get("source_table_run") == SRC_DIR]
    if runs08:
        for f in os.listdir(f"{runs08[-1]}/step0"):
            if f.startswith("answers_") and f.endswith("__gold.jsonl"):
                shutil.copy(f"{runs08[-1]}/step0/{f}", f"{RUN_DIR}/step0/{f}")
        print("added gold-paragraph answers from", runs08[-1])
    else:
        print("no finished notebook-08 run yet - the gold row is left out")''')])

# ------------------------------------------------------------------------------ Set 4: 10 / 11 / 12
SET4_NAME = "Set_1__3-modality-size"
SET4_MODELS = '''MODELS = {   # key: model, folder under models/, parameters (for compute), nominal size (for the AgentTTS budget formula)
    "q3":  {"id": "Qwen/Qwen2.5-3B-Instruct",        "dir": "Qwen2.5-3B-Instruct",        "params": 3.09e9,  "nominal": 3e9},
    "q7":  {"id": "Qwen/Qwen2.5-7B-Instruct",        "dir": "Qwen2.5-7B-Instruct",        "params": 7.62e9,  "nominal": 7e9},
    "q32": {"id": "Qwen/Qwen2.5-32B-Instruct-AWQ",   "dir": "Qwen2.5-32B-Instruct-AWQ",   "params": 32.76e9, "nominal": 32e9},
    "l3":  {"id": "meta-llama/Llama-3.2-3B-Instruct", "dir": "Llama-3.2-3B-Instruct",     "params": 3.21e9,  "nominal": 3e9},
    "l8":  {"id": "meta-llama/Llama-3.1-8B-Instruct", "dir": "Llama-3.1-8B-Instruct",     "params": 8.03e9,  "nominal": 8e9},
}
LOCAL = "/content/models"
for m in MODELS.values():
    m["path"] = f"{LOCAL}/{m['dir']}"

def ensure_local(key):
    """Copies one model from Drive to the runtime's local disk (only when a stage needs it)."""
    m = MODELS[key]
    src = f"{PROJECT_DIR}/models/{m['dir']}"
    if DRY_RUN or os.path.exists(f"{m['path']}/config.json"):
        return
    assert os.path.exists(f"{src}/config.json"), f"{src} missing - run notebook 10 first."
    idx = f"{src}/model.safetensors.index.json"
    if os.path.exists(idx):
        lost = [f for f in sorted(set(json.load(open(idx))["weight_map"].values())) if not os.path.exists(f"{src}/{f}")]
        assert not lost, f"{src} is incomplete (missing {lost}) - run notebook 10 again."
    t = time.time()
    print(f"copying {m['dir']} to local disk ...")
    shutil.copytree(src, m["path"], dirs_exist_ok=True)
    TIMES[f"copy_{key}"] = round(time.time() - t)'''

SET4_RUN = '''BACKEND = "mock" if DRY_RUN else "vllm"
CODE_DIR = "/content/set4code"
os.makedirs(CODE_DIR, exist_ok=True)
for f in ["ssw.py", "sweep.py"]:
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{RUN_DIR}/code/{f}")      # keep the exact code with the results
TIMES = {}

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name, quiet=False):
    """Runs one sweep.py command in a fresh process; streams output here and into logs/<log_name>."""
    cmd = [sys.executable, "-u", f"{CODE_DIR}/sweep.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        log.write("\\n$ " + " ".join(cmd) + "\\n")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line)
            slog.write(line)
            if quiet or any(k in line for k in ("it/s]", "s/it]", "INFO ", "(EngineCore", "WARNING ")):
                continue
            print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    if not quiet:
        print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    if p.returncode != 0:
        raise RuntimeError(f"stage failed - see {RUN_DIR}/logs/{log_name}")

COMMON = ["--run_dir", RUN_DIR, "--data_dir", DATA_DIR, "--models_json", f"{RUN_DIR}/models.json", "--backend", BACKEND]
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)'''

SET4_FIND_RUN = '''import glob
from datetime import datetime
from zoneinfo import ZoneInfo
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
# which session this is: the models it is restricted to (ONLY_MODELS), or "all"
SESSION_LABEL = globals().get("SESSION_LABEL") or ("-".join(globals().get("ONLY_MODELS") or []) or "all")
import re
CODE_VERSION = re.search(r'^CODE_VERSION = "(.*)"', open(f"{PROJECT_DIR}/code/sweep.py").read(), re.M).group(1)
def run_version(d):
    """The prompt version a run folder was made with (None = empty folder)."""
    if os.path.exists(f"{d}/code_version.txt"):
        return open(f"{d}/code_version.txt").read().strip()
    return "1: before confidence tracking" if glob.glob(f"{d}/step/*.jsonl") else None
runs = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*")) if os.path.isdir(d) and (("_dryrun" in d) == DRY_RUN)]
skipped = [d for d in runs if run_version(d) not in (None, CODE_VERSION)]
runs = [d for d in runs if d not in skipped]
for d in skipped:
    print("Not continuing (made with older prompts, kept untouched):", os.path.basename(d))
if RUN_DIR_OVERRIDE:
    RUN_DIR = RUN_DIR_OVERRIDE
elif runs and not NEW_RUN:
    RUN_DIR = runs[-1]                      # continue the latest run of this set
else:
    RUN_DIR = f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}" + ("_dryrun" if DRY_RUN else "")
for sub in ["logs", "code", "step", "stages"]:
    os.makedirs(f"{RUN_DIR}/{sub}", exist_ok=True)
# everything specific to this session goes in its own labelled folder inside the shared run folder
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}_{SESSION_LABEL}"
os.makedirs(SESSION_DIR, exist_ok=True)
print("Run folder (shared by all sessions):", RUN_DIR)
print("This session's folder:", SESSION_DIR)'''

nb10 = nb([
    ("md", f"""# 10 · Set 4 setup: download Qwen2.5-32B (4-bit) and Llama-3.1-8B to Google Drive

Set 4 (`{SET4_NAME}`) uses three sizes per role:

| Role | Small | Medium | Large |
|---|---|---|---|
| Retriever | Qwen2.5-3B (have) | Qwen2.5-7B (have) | **Qwen2.5-32B, 4-bit AWQ (~19 GB)** |
| Answerer | Llama-3.2-3B (have) | **Llama-3.1-8B (~16 GB)** | Qwen2.5-32B (same files as the retriever) |

This notebook downloads the two models in bold (about 35 GB). A CPU runtime is fine.

**Llama-3.1-8B is gated.** Before running: open https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct, accept the licence, wait for the approval
email, and make sure your Hugging Face token is saved in Colab's Secrets (key icon) as `HF_TOKEN` with notebook access on."""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
DOWNLOAD = {{
    "Qwen2.5-32B-Instruct-AWQ": "Qwen/Qwen2.5-32B-Instruct-AWQ",      # ~19 GB, not gated
    "Llama-3.1-8B-Instruct":    "meta-llama/Llama-3.1-8B-Instruct",   # ~16 GB, gated (needs HF_TOKEN + accepted licence)
    "Llama-3.2-3B-Instruct":    "meta-llama/Llama-3.2-3B-Instruct",   # already in Drive from earlier; skipped if complete
}}
ALLOW  = ["*.json", "*.safetensors", "tokenizer*", "*.model", "*.txt", "*.tiktoken", "*.jinja"]
IGNORE = ["original/*", "*.pth", "*.bin", "*.gguf"]'''),
    ("code", MOUNT),
    ("code", '''import os, shutil, glob, time, json
!pip install -q hf_transfer        # huggingface_hub is already in Colab; upgrading it only triggers version warnings
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
try:
    from google.colab import userdata
    tok = userdata.get("HF_TOKEN")
    tok = tok.get("value") if isinstance(tok, dict) else tok
except Exception:
    tok = None
print("Hugging Face token found:", bool(tok))
from huggingface_hub import snapshot_download

MODEL_DIR = f"{PROJECT_DIR}/models"
os.makedirs(MODEL_DIR, exist_ok=True)
total, used, free = shutil.disk_usage("/content/drive")
print(f"Drive before: {used/1e9:.1f} GB used, {free/1e9:.1f} GB free")
def missing_files(p):
    """Files a model folder still lacks: config, tokenizer and every weight file named in its index."""
    need = ["config.json", "tokenizer.json"]
    idx = f"{p}/model.safetensors.index.json"
    if os.path.exists(idx):
        need += sorted(set(json.load(open(idx))["weight_map"].values()))
    elif not glob.glob(f"{p}/*.safetensors"):
        need.append("model.safetensors")
    return [f for f in need if not (os.path.exists(f"{p}/{f}") and os.path.getsize(f"{p}/{f}") > 0)]
def complete(p):
    return os.path.isdir(p) and not missing_files(p)

for name, repo in DOWNLOAD.items():
    dst = f"{MODEL_DIR}/{name}"
    if complete(dst):
        print(f"✓ {name} already in Drive"); continue
    if os.path.isdir(dst):
        print(f"{name}: incomplete in Drive, missing {missing_files(dst)} - downloading again")
    t = time.time()
    tmp = f"/content/_dl/{name}"
    print(f"downloading {repo} ...")
    try:
        snapshot_download(repo_id=repo, local_dir=tmp, allow_patterns=ALLOW, ignore_patterns=IGNORE, token=tok or None)
    except Exception as e:
        if "meta-llama" in repo:
            print(f"✗ {name}: could not download ({type(e).__name__}). Accept the licence at https://huggingface.co/{repo}, "
                  "wait for approval, check the HF_TOKEN secret, then run this cell again.")
            continue
        raise
    print(f"copying to {dst} ...")
    shutil.copytree(tmp, dst, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
    shutil.rmtree(tmp)
    print(f"✓ {name} in {time.time() - t:.0f}s")'''),
    ("code", '''!du -sh "{MODEL_DIR}"/*
total, used, free = shutil.disk_usage("/content/drive")
print(f"Drive after: {used/1e9:.1f} GB used, {free/1e9:.1f} GB free")
missing = {n: missing_files(f"{MODEL_DIR}/{n}") for n in DOWNLOAD if not complete(f"{MODEL_DIR}/{n}")}
print("All Set 4 models are complete in Drive." if not missing else f"STILL INCOMPLETE: {missing}")
from google.colab import drive
drive.flush_and_unmount()          # waits until every file is really uploaded to Drive
print("Drive upload finished - safe to close this notebook. Next: 11_set4_sweep.ipynb on an A100.")'''),
])
nb10.metadata.pop("accelerator", None)
nb10.metadata["colab"].pop("gpuType", None)

nb11 = nb([
    ("md", f"""# 11 · Set 4 sweep: 3 retrievers × 3 answerers × CoT × per-attempt shortlist × final shortlist (GPU, several sessions)

Runs the broad sweep of `{SET4_NAME}` on **200 questions** (the first 100 dev + 100 test of the Set 1–3 questions, 20 passages each).

| Variable | Levels |
|---|---|
| Retriever | Qwen2.5-3B, Qwen2.5-7B, Qwen2.5-32B (4-bit) |
| Answerer | Llama-3.2-3B, Llama-3.1-8B, Qwen2.5-32B (4-bit) |
| CoT | off / on (retriever and answerer together) |
| Per-attempt shortlist | paper default (no number given), "select exactly 3", "select exactly 5" |
| Final shortlist | paper default (vote merge), 2 (two most-voted), all (every passage any try picked) |

That is **162 setups** at 3 retrieval tries and 5 answer tries + fusion (1-try versions are also recorded).

**Confidence is recorded for every decision**, two ways: *stated* (the model writes a last line `Confidence: <0-100>`) and *measured*
(the probability the model gave its own output, and how much its repeated tries agree). The paper-default retrieval prompt is left
untouched, so it has measured confidence only. The report shows how well each signal tells right from wrong (Excel sheets
"Confidence" and "Calibration", chart 5).

**This notebook is resumable. Run it, and run it again, until it prints `SWEEP COMPLETE`.** Each session works through the pending
stages in order (retrieval with each retriever, then answering with each answerer, without and with CoT), stops cleanly before the time
limit, and always ends by rebuilding the tables, Excel file and charts from whatever is finished. My estimate is 5–8 GPU hours in total,
most of it the 32B; that is an estimate, not a measurement.

**Two runtimes at once:** use the two ready-made copies instead of this notebook — **11a_set4_sweep_q3-q7-l3-l8.ipynb** and
**11b_set4_sweep_q32.ipynb**. They are this notebook with `ONLY_MODELS` already set. Start 11a first; start 11b once 11a has printed its
run folder, so both use the same one. (Two tabs of the *same* notebook would overwrite each other's saved copy.)

**Where things are saved.** All sessions write into one shared run folder, because every answerer needs every retriever's shortlists
and the tables combine everything. Inside it, each session has its own labelled folder,
`sessions/<time>_<models>/` (for example `sessions/20261004-150000_q3-q7-l3-l8/` or `..._q32/`), with that session's logs, timings,
`session.json` and a snapshot of the results at its end. The raw files in `step/` are already named by model (`retrieval_q32_...`, `answers_l8_...`).

A run folder made with older prompts is never continued: the notebook skips it and starts a new one.

Output: `runs/{SET4_NAME}/<timestamp>_<GPU>/` with `results/summary.md`, `results/set4_results.xlsx`, `results/per_question.jsonl`,
`results/charts/`, raw outputs in `step/`. Next: **12_set4_agenttts_switching.ipynb**.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET4_NAME}"
SOURCE_DATA = f"{{PROJECT_DIR}}/data/hotpotqa_q/dev200_test200_c20_s42"   # the Set 1-3 questions
DEV_N, TEST_N = 100, 100      # the sweep uses the first 100 of each split
ONLY_MODELS = []              # [] = every pending stage; or e.g. ["q3", "q7", "l3", "l8"] / ["q32"] to split across two runtimes
TIME_LIMIT_MIN = 55           # the session stops cleanly at this limit and keeps what is done
NEW_RUN = False               # True starts a fresh run folder instead of continuing the latest compatible one
RUN_DIR_OVERRIDE = None       # or paste a run folder to continue that one
DRY_RUN = False               # True = fake models on CPU, tests the whole notebook in ~1 min'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("code", SET4_FIND_RUN),
    ("md", "### Questions: first 100 dev + 100 test of the Set 1–3 sample (built once)"),
    ("code", '''DATA_DIR = f"{PROJECT_DIR}/data/hotpotqa_q/dev{DEV_N}_test{TEST_N}_c20_s42_first"
if not os.path.exists(f"{DATA_DIR}/test.json"):
    os.makedirs(DATA_DIR, exist_ok=True)
    for split, n in (("dev", DEV_N), ("test", TEST_N)):
        rows = json.load(open(f"{SOURCE_DATA}/{split}.json", encoding="utf-8"))[:n]
        json.dump(rows, open(f"{DATA_DIR}/{split}.json", "w", encoding="utf-8"), ensure_ascii=False)
    json.dump({"source": SOURCE_DATA, "dev_n": DEV_N, "test_n": TEST_N, "rule": "first n of each split"}, open(f"{DATA_DIR}/data_config.json", "w"), indent=1)
print("questions:", DATA_DIR)'''),
    ("code", SET4_MODELS),
    ("code", SET4_RUN + '''
if not os.path.exists(f"{RUN_DIR}/config.json"):
    json.dump({"experiment": EXPERIMENT, "set": "4", "data_dir": DATA_DIR, "source_data": SOURCE_DATA, "dev_n": DEV_N, "test_n": TEST_N,
               "gpu": gpu, "dry_run": DRY_RUN, "started": SESSION,
               "design": {"retrievers": ["q3", "q7", "q32"], "answerers": ["l3", "l8", "q32"], "cot": "both steps together",
                          "per_attempt_shortlist": ["silent", "3", "5"], "final_shortlist": ["vote", "top2", "all"],
                          "retrieval_tries": 3, "answer_tries": 5}}, open(f"{RUN_DIR}/config.json", "w"), indent=1)'''),
    ("md", "### What is done, what is pending"),
    ("code", '''def pending():
    run_stage(["status"] + COMMON, "status.log", quiet=True)
    st = json.load(open(f"{RUN_DIR}/sweep_status.json"))
    todo = [s for s in st["stages"] if not s["complete"] and s["total"] > 0 and s["done"] < s["total"]
            and (not ONLY_MODELS or s["model"] in ONLY_MODELS)]
    return todo, st
todo, st = pending()
print(open(f"{RUN_DIR}/logs/status.log").read().split("$")[-1].split("\\n", 1)[1])'''),
    ("code", '''# vLLM runs only inside the stage processes, so no runtime restart is needed (~3-5 min)
if todo and not DRY_RUN:
    !pip install -q vllm
    !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
    chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                         capture_output=True, text=True)
    print(chk.stdout or chk.stderr[-2000:])
    assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
    TIMES["install_vllm"] = round(time.time() - T_START)'''),
    ("md", "### Work through the pending stages until the time limit"),
    ("code", '''MIN_LEFT_TO_START = 8     # do not start a new stage with less than this many minutes left
while True:
    todo, st = pending()
    if not todo:
        break
    if minutes_left() < MIN_LEFT_TO_START:
        print(f"\\nTime limit close ({minutes_left():.0f} min left) - stopping here. Run this notebook again to continue.")
        break
    s = todo[0]
    key = (s["stage"], s["model"], s["cot"], s["done"])
    if key == globals().get("_last_key"):
        raise RuntimeError(f"no progress on {key[:3]} - see the log in {RUN_DIR}/logs")
    _last_key = key
    ensure_local(s["model"])
    limit = max(1.0, minutes_left() - 3)
    if s["stage"] == "retrieve":
        run_stage(["retrieve", "--model", s["model"], "--time_limit_min", limit] + COMMON, f"retrieve_{s['model']}.log")
    else:
        run_stage(["answer", "--model", s["model"], "--cot", s["cot"], "--time_limit_min", limit] + COMMON, f"answer_{s['model']}_c{s['cot']}.log")
run_stage(["status"] + COMMON, "status.log")'''),
    ("md", "### Tables, Excel workbook and charts from everything finished so far"),
    ("code", '''run_stage(["report"] + COMMON, "report.log", quiet=True)
from IPython.display import Image, Markdown, display
if os.path.exists(f"{RUN_DIR}/results/summary.md"):
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
        display(Image(f))
    print("Excel workbook:", f"{RUN_DIR}/results/set4_results.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)

# ---- record of this session: which models it was for, what it ran, and a snapshot of the results as they stood at its end ----
import shutil
ran = [k for k in TIMES if k.startswith(("retrieve_", "answer_"))]
json.dump({"session": SESSION, "label": SESSION_LABEL, "only_models": ONLY_MODELS or "all", "gpu": gpu, "code_version": CODE_VERSION,
           "stages_run": ran, "seconds": TIMES, "run_dir": RUN_DIR,
           "sweep_complete": json.load(open(f"{RUN_DIR}/sweep_status.json"))["complete"]}, open(f"{SESSION_DIR}/session.json", "w"), indent=1)
os.makedirs(f"{SESSION_DIR}/results_snapshot", exist_ok=True)
for f in ["summary.md", "set4_results.xlsx", "setups.jsonl"]:
    if os.path.exists(f"{RUN_DIR}/results/{f}"):
        shutil.copy(f"{RUN_DIR}/results/{f}", f"{SESSION_DIR}/results_snapshot/{f}")
if os.path.isdir(f"{RUN_DIR}/results/charts"):
    shutil.copytree(f"{RUN_DIR}/results/charts", f"{SESSION_DIR}/results_snapshot/charts", dirs_exist_ok=True)
with open(f"{SESSION_DIR}/README.md", "w") as f:
    f.write(f"# Session {SESSION} - ONLY_MODELS = {ONLY_MODELS or 'all'}\\n\\nGPU: {gpu}\\n\\nStages run in this session: {', '.join(ran) or 'none'}\\n\\n"
            f"Seconds: {TIMES}\\n\\n`results_snapshot/` = the tables and charts as they stood when this session ended "
            f"(built from everything finished in the shared run folder at that time, not only this session's models).\\n")
print("Session record:", SESSION_DIR)
print("\\nSWEEP COMPLETE - next: 12_set4_agenttts_switching.ipynb" if json.load(open(f"{RUN_DIR}/sweep_status.json"))["complete"]
      else "\\nNOT COMPLETE YET - run this notebook again to continue.")'''),
])

nb12 = nb([
    ("md", f"""# 12 · Set 4: AgentTTS baseline and per-question switching, planner = Qwen2.5-32B (GPU, ~30–45 min)

Uses the finished sweep from notebook 11 (no new model answers are generated; everything is scored from the sweep's results).

1. **AgentTTS baseline, for every condition.** For each of the 18 conditions (CoT × per-attempt shortlist × final shortlist), the AgentTTS
   search (paper's prompts and budget formula) picks one retriever, one answerer and the number of tries for all questions, on the dev
   questions, at three budgets. Its pick is then scored on the test questions and compared with the best possible single setup.
2. **Per-question switching, for every condition** (and once across all conditions). The routes are the pairings that are best for their
   cost on dev. Two routers: a cost-weighted classifier, and **rules written by the planner model**, applied exactly by code.
   Two settings: *question only*, and *after small pipeline* (also uses how much the small retriever's tries and the small answerer's
   answers agree, and pays for that run).
3. **The rules the planner wrote are printed below** and saved in `plan/rules/`, each with the planner's own confidence and how the
   rule actually did on dev.
4. **Confidence of the deciders.** The classifier's probability for its chosen route, the planner's confidence on each rule, and the
   planner's confidence in each AgentTTS choice are recorded; `plan/decisions.jsonl` lists every routed test question with the route,
   the router's confidence and whether the answer was right. The routers may also use the small pipeline's own stated confidence as a signal.

With 100 dev and 100 test questions, differences under about 10 points are noise; this stage shows which directions are promising, not final numbers.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET4_NAME}"
PLANNER     = "q32"                   # the model that runs the AgentTTS search and writes the routing rules
BUDGET_QUANTILES = [0.25, 0.5, 1.0]   # AgentTTS budgets per condition (quantiles of that condition's setup costs)
SEEDS, ITERATIONS, BATCH_SIZE, MAX_TRIALS = 1, 4, 2, 8
LAMBDAS     = [0.0, 0.03]             # accuracy-vs-cost weights for the routers (0 = accuracy only)
RULE_LAMBDA, RULE_ROUNDS, MAX_RULES, MAX_ROUTES = 0.03, 3, 8, 4
TIME_LIMIT_MIN = 55
SESSION_LABEL = "planner"             # this session's folder is runs/.../sessions/<time>_planner
NEW_RUN = False
RUN_DIR_OVERRIDE = None               # or paste the sweep's run folder
DRY_RUN = False'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("code", SET4_FIND_RUN + '''
assert os.path.exists(f"{RUN_DIR}/config.json"), "No sweep run found - run notebook 11 first."
DATA_DIR = json.load(open(f"{RUN_DIR}/config.json"))["data_dir"]'''),
    ("code", SET4_MODELS),
    ("code", SET4_RUN),
    ("code", '''run_stage(["status"] + COMMON, "status.log")
if not json.load(open(f"{RUN_DIR}/sweep_status.json"))["complete"]:
    print("\\nNOTE: the sweep is not complete; this will plan on the setups that are finished.")'''),
    ("code", INSTALL.replace("if not DRY_RUN:", "if not DRY_RUN:", 1) + '\nTIMES["install_vllm"] = round(time.time() - T_START)\nensure_local(PLANNER)'),
    ("md", "### AgentTTS baseline + switching (the planner's rules are printed at the end)"),
    ("code", '''run_stage(["plan", "--planner", PLANNER, "--budget_quantiles", *BUDGET_QUANTILES, "--seeds", SEEDS, "--iterations", ITERATIONS,
           "--batch_size", BATCH_SIZE, "--max_trials", MAX_TRIALS, "--lambdas", *LAMBDAS, "--rule_lambda", RULE_LAMBDA,
           "--rule_rounds", RULE_ROUNDS, "--max_rules", MAX_RULES, "--max_routes", MAX_ROUTES] + COMMON, "plan.log")'''),
    ("md", "### Full report with the AgentTTS and switching results added (Excel sheets 'AgentTTS baseline', 'Switching', 'Rules')"),
    ("code", '''run_stage(["report"] + COMMON, "report.log", quiet=True)
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
    display(Image(f))
TIMES["total_session"] = round(time.time() - T_START)
json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
print("Excel workbook:", f"{RUN_DIR}/results/set4_results.xlsx")
print("Wall-clock this session (s):", TIMES)'''),
])


SET5_NAME = "Set_2__question-reading-routers"
nb13 = nb([
    ("md", f"""# 13 · Set 5a: offline switching policies on the Set 4 results (CPU, a few minutes)

Tests, **without generating any new model answers**, whether per-question switching can match a fixed setup at lower compute.
Everything is computed from the finished Set 4 sweep (`runs/{SET4_NAME}/…/results/per_question.jsonl`).

| Policy | What decides | Extra cost of deciding |
|---|---|---|
| A1 answerer cascade | cheap answerer first; bigger answerer only if its 5 answers disagree (retrieval shared) | none |
| A2 retriever cascade | small retriever first; bigger retriever only if its 3 tries disagree | none |
| A3 both cascades | A2 then A1 | none |
| A4 big answerer with 1 try | A1 / A3 with a single answer try for the bigger answerer | none |
| A5 learned router, question text | a small trained model reads the question before anything runs | negligible |
| A6 learned router, 5 facts | the Set 4 inputs (yes/no, comparison, words, digit, names) | none |
| A8 / A9 learned cascade | a trained gate on the cheap run's metadata (A9: plus the question text) | none |
| A7 Set 4 rules, smoothed | the planner's rules, with each rule's dev accuracy shrunk by an m-estimate or Laplace smoothing | none |

**Selection rule, the same for fixed setups and for every policy:** on the training questions, the cheapest candidate whose accuracy is
within 2 points of the most accurate fixed setup. **Scores are always on questions the selection did not see** (5-fold cross-validation
repeated `REPEATS` times over all 200 questions, and the Set 4 dev → test split).

Baselines in every table: the most accurate fixed setup, the cheapest fixed setup within 2 points of it, a random router with the same
mix, and a perfect router. The policies chosen on all 200 questions are saved in `frozen_policies.json` for the fresh-question test (step 5c).

Output: `runs/{SET5_NAME}/<timestamp>_cpu/` with `results/summary.md`, `results/set5a_results.xlsx`, `results/charts/` (6 charts),
`results/decisions.jsonl`, `results/rules_smoothed*.jsonl`, `frozen_policies.json`, and a `sessions/<time>_offline/` folder.

**No GPU needed: Runtime → Change runtime type → CPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
SWEEP_EXPERIMENT = "{SET4_NAME}"     # the finished Set 4 sweep these policies are computed from
SWEEP_RUN_OVERRIDE = None             # or paste a sweep run folder; None = the latest complete one
REPEATS = 5                           # repeats of the 5-fold cross-validation
USE_EMBEDDING = True                  # also try a sentence-embedding model for the question text (downloads ~90 MB once)
NEW_RUN = True                        # each run of this notebook gets its own folder (it only takes minutes)'''),
    ("code", MOUNT),
    ("code", '''import glob, json, os, shutil, subprocess, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
T_START = time.time()
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
if SWEEP_RUN_OVERRIDE:
    SWEEP_RUN = SWEEP_RUN_OVERRIDE
else:
    done = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d
            and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")
            and os.path.exists(f"{d}/results/per_question.jsonl")]
    assert done, f"No complete sweep found in runs/{SWEEP_EXPERIMENT} - finish notebook 11 first."
    SWEEP_RUN = done[-1]
DATA_DIR = json.load(open(f"{SWEEP_RUN}/config.json"))["data_dir"]
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_cpu"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_cpu"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}_offline"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
CODE_DIR = "/content/set5code"
os.makedirs(CODE_DIR, exist_ok=True)
for f in ("ssw.py", "sweep.py", "routers.py"):
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{RUN_DIR}/code/{f}")       # keep the exact code with the results
json.dump({"experiment": EXPERIMENT, "set": "5a", "sweep_run": SWEEP_RUN, "data_dir": DATA_DIR, "repeats": REPEATS,
           "use_embedding": USE_EMBEDDING, "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)
print("Set 4 sweep used:", SWEEP_RUN)
print("Run folder:", RUN_DIR)'''),
    ("code", '''# the embedding model is optional: without it the TF-IDF text features still run
if USE_EMBEDDING:
    try:
        import sentence_transformers
    except ImportError:
        print("installing sentence-transformers ...")
        !pip install -q sentence-transformers'''),
    ("md", "### Run every policy (a few minutes)"),
    ("code", '''cmd = [sys.executable, "-u", f"{CODE_DIR}/routers.py", "offline", "--sweep_run", SWEEP_RUN, "--data_dir", DATA_DIR,
       "--run_dir", RUN_DIR, "--repeats", str(REPEATS)] + ([] if USE_EMBEDDING else ["--no_embedding"])
with open(f"{RUN_DIR}/logs/offline.log", "a") as log, open(f"{SESSION_DIR}/offline.log", "a") as slog:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        log.write(line); slog.write(line)
        print(line, end="")
    p.wait()
assert p.returncode == 0, f"failed - see {RUN_DIR}/logs/offline.log"'''),
    ("md", "### Results"),
    ("code", '''from IPython.display import Image, Markdown, display
display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
    display(Image(f))
os.makedirs(f"{SESSION_DIR}/results_snapshot", exist_ok=True)
for f in ("summary.md", "set5a_results.xlsx", "policies.jsonl"):
    shutil.copy(f"{RUN_DIR}/results/{f}", f"{SESSION_DIR}/results_snapshot/{f}")
shutil.copytree(f"{RUN_DIR}/results/charts", f"{SESSION_DIR}/results_snapshot/charts", dirs_exist_ok=True)
shutil.copy(f"{RUN_DIR}/frozen_policies.json", f"{SESSION_DIR}/frozen_policies.json")
TIMES = {"total_session": round(time.time() - T_START)}
json.dump({"session": SESSION, "label": "offline", "sweep_run": SWEEP_RUN, "run_dir": RUN_DIR, "seconds": TIMES,
           "status": json.load(open(f"{RUN_DIR}/offline_status.json"))}, open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Excel workbook:", f"{RUN_DIR}/results/set5a_results.xlsx")
print("Frozen policies for step 5c:", f"{RUN_DIR}/frozen_policies.json")
print("Wall-clock this session (s):", TIMES)
try:
    drive.flush_and_unmount()          # make sure everything reaches Drive
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])
nb13.metadata.pop("accelerator", None)
nb13.metadata["colab"].pop("gpuType", None)

nb14 = nb([
    ("md", f"""# 14 · Set 5d part 1: is there hidden structure that separates easy questions from hard ones? (CPU, about 10 minutes)

Looks for structure in the finished Set 4 results, **without generating any new model answers**. Several ways of looking are scored on
one scoreboard so they can be compared.

| Spaces (what is looked at) | Strategies (how structure is read) | Targets (what "easy" means) |
|---|---|---|
| question facts | linear probe | the small pipeline is right |
| question text, topics, embedding | first 2 principal components (PCA) | the cheap answer is no worse than the big one |
| question–passage similarity | nearest neighbours (the neighbourhoods t-SNE draws) | the best setup is right |
| metadata of the small pipeline and of a cheap run | k-means clusters | the small retriever finds both passages |
| agreement between two small answerers | a small decision tree; boosted trees | the question is easy for most setups |
| all signals available after the cheap run | association rules | |

**Every score is on held-out questions**, shown next to the score the same method gets when the labels are shuffled. PCA and t-SNE maps
are drawn, but a picture is not the evidence; the held-out number is. Association rules are found on the dev half and checked on the test half.

Output: `runs/{SET5_NAME}/<timestamp>_cpu_structure/` with `results/summary.md`, `results/set5d_structure.xlsx`, `results/charts/`
(9 charts), `results/association_rules.jsonl`, `results/tree_rules.txt`, `results/topics.jsonl`, `results/clusters.jsonl`.

**No GPU needed: Runtime → Change runtime type → CPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
SWEEP_EXPERIMENT = "{SET4_NAME}"     # the finished Set 4 sweep these policies are computed from
SWEEP_RUN_OVERRIDE = None             # or paste a sweep run folder; None = the latest complete one
REPEATS = 3                           # repeats of the 5-fold cross-validation
PERMUTATIONS = 20                     # label shuffles per combination for the control
USE_EMBEDDING = True                  # also use a sentence-embedding model for questions and passages (downloads ~90 MB once)
NEW_RUN = True                        # each run of this notebook gets its own folder (it only takes minutes)'''),
    ("code", MOUNT),
    ("code", '''import glob, json, os, shutil, subprocess, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
T_START = time.time()
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
if SWEEP_RUN_OVERRIDE:
    SWEEP_RUN = SWEEP_RUN_OVERRIDE
else:
    done = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d
            and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")
            and os.path.exists(f"{d}/results/per_question.jsonl")]
    assert done, f"No complete sweep found in runs/{SWEEP_EXPERIMENT} - finish notebook 11 first."
    SWEEP_RUN = done[-1]
DATA_DIR = json.load(open(f"{SWEEP_RUN}/config.json"))["data_dir"]
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_cpu_structure"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_cpu_structure"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}_structure"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
CODE_DIR = "/content/set5code"
os.makedirs(CODE_DIR, exist_ok=True)
for f in ("ssw.py", "sweep.py", "routers.py", "structure.py"):
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{RUN_DIR}/code/{f}")       # keep the exact code with the results
json.dump({"experiment": EXPERIMENT, "set": "5d part 1", "sweep_run": SWEEP_RUN, "data_dir": DATA_DIR, "repeats": REPEATS, "permutations": PERMUTATIONS,
           "use_embedding": USE_EMBEDDING, "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)
print("Set 4 sweep used:", SWEEP_RUN)
print("Run folder:", RUN_DIR)'''),
    ("code", '''# the embedding model is optional: without it the TF-IDF text features still run
if USE_EMBEDDING:
    try:
        import sentence_transformers
    except ImportError:
        print("installing sentence-transformers ...")
        !pip install -q sentence-transformers'''),
    ("md", "### Run every strategy (about 10 minutes)"),
    ("code", '''cmd = [sys.executable, "-u", f"{CODE_DIR}/structure.py", "offline", "--sweep_run", SWEEP_RUN, "--data_dir", DATA_DIR,
       "--run_dir", RUN_DIR, "--repeats", str(REPEATS), "--permutations", str(PERMUTATIONS)] + ([] if USE_EMBEDDING else ["--no_embedding"])
with open(f"{RUN_DIR}/logs/structure.log", "a") as log, open(f"{SESSION_DIR}/structure.log", "a") as slog:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        log.write(line); slog.write(line)
        print(line, end="")
    p.wait()
assert p.returncode == 0, f"failed - see {RUN_DIR}/logs/structure.log"'''),
    ("md", "### Results"),
    ("code", '''from IPython.display import Image, Markdown, display
display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
    display(Image(f))
os.makedirs(f"{SESSION_DIR}/results_snapshot", exist_ok=True)
for f in ("summary.md", "set5d_structure.xlsx", "scoreboard.jsonl", "tree_rules.txt"):
    shutil.copy(f"{RUN_DIR}/results/{f}", f"{SESSION_DIR}/results_snapshot/{f}")
shutil.copytree(f"{RUN_DIR}/results/charts", f"{SESSION_DIR}/results_snapshot/charts", dirs_exist_ok=True)
TIMES = {"total_session": round(time.time() - T_START)}
json.dump({"session": SESSION, "label": "structure", "sweep_run": SWEEP_RUN, "run_dir": RUN_DIR, "seconds": TIMES,
           "status": json.load(open(f"{RUN_DIR}/structure_status.json"))}, open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Excel workbook:", f"{RUN_DIR}/results/set5d_structure.xlsx")
print("Readable tree rules:", f"{RUN_DIR}/results/tree_rules.txt")
print("Wall-clock this session (s):", TIMES)
try:
    drive.flush_and_unmount()          # make sure everything reaches Drive
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])
nb14.metadata.pop("accelerator", None)
nb14.metadata["colab"].pop("gpuType", None)


nb16 = nb([
    ("md", f"""# 16 · Set 5d part 2: hidden signals from one cheap generation (A100, about one session)

Stage A of the hidden-signal plan: **measure whether each signal is informative**. Nothing here routes questions; the informative
signals are saved for stage B (notebook 15), where the planner writes rules over them.

A cheap answerer (Llama 8B, then Llama 3B) answers each of the 200 Set 4 questions **once**, on the same shortlist the cheap fixed
setup uses (Qwen 7B retrieves once). While it does, these are recorded:

| Priority | Signal | What is extracted | Extra inference |
|---|---|---|---|
| 1 | Hidden-state probe | the model's internal activations at the last prompt token (before any answer), over the answer tokens, and on the final answer line; 6 layers | none |
| 2 | Semantic uncertainty probe | a probe trained only to predict whether 5 answers would agree (labels saved in Set 4), then used as a signal | none |
| 3 | Evidence-use attention | share of attention on the passages, the question, and the model's own earlier text | none |
| added | Retriever-side probe | Qwen 7B's hidden state after reading the question and the 20 passages | none |

Passage-perturbation sensitivity (priority "later") is not in this notebook.

**The bar** is the cheap fixed setup: Qwen 7B retrieves once, Qwen 32B answers once (63.5% at 4.2x on these questions). Every score is
on held-out questions, next to the same method run on shuffled labels. PCA and t-SNE maps are drawn, but the held-out number is the evidence.

Extraction uses the `transformers` library directly, because vLLM does not return hidden states or attention. **This notebook is
resumable: if it stops at the time limit, run it again.** My estimate is 45–60 minutes in total (model copies about 18, extraction
about 30, analysis about 5); that is an estimate, not a measurement.

Output: `runs/{SET5_NAME}/<timestamp>_<GPU>_hidden/` with `results/summary.md`, `results/set5d_hidden.xlsx`, `results/charts/`,
`results/signals_for_rules.jsonl`, `results/single_answers_<model>.jsonl`, raw arrays in `step/`.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
SWEEP_EXPERIMENT = "{SET4_NAME}"     # the finished Set 4 sweep: shortlists, the big answerer's results and the agreement labels come from it
SWEEP_RUN_OVERRIDE = None
ANSWERERS = ["l8", "l3"]              # cheap answerers to probe, in this order
RETRIEVER = "q7"                      # retriever to probe
REPEATS, PERMUTATIONS = 3, 20         # cross-validation repeats and label shuffles in the analysis
TIME_LIMIT_MIN = 55                   # extraction stops cleanly at this limit; run the notebook again to continue
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = a tiny random model on CPU, tests the whole notebook in a few minutes'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("code", '''import glob
from datetime import datetime
from zoneinfo import ZoneInfo
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
if SWEEP_RUN_OVERRIDE:
    SWEEP_RUN = SWEEP_RUN_OVERRIDE
else:
    done = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d
            and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")
            and os.path.exists(f"{d}/results/per_question.jsonl")]
    assert done, f"No complete sweep found in runs/{SWEEP_EXPERIMENT} - finish notebook 11 first."
    SWEEP_RUN = done[-1]
DATA_DIR = json.load(open(f"{SWEEP_RUN}/config.json"))["data_dir"]
tag = "_hidden_dryrun" if DRY_RUN else "_hidden"
prev = sorted(d for d in glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}_hidden"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", f"{RUN_DIR}/step", f"{RUN_DIR}/stages", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
CODE_DIR = "/content/set5code"
os.makedirs(CODE_DIR, exist_ok=True)
for f in ("ssw.py", "sweep.py", "routers.py", "structure.py", "hidden.py"):
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{RUN_DIR}/code/{f}")       # keep the exact code with the results
TIMES = {}
print("Set 4 sweep used:", SWEEP_RUN)
print("Run folder:", RUN_DIR)'''),
    ("code", SET4_MODELS + '''
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "5d part 2", "sweep_run": SWEEP_RUN, "data_dir": DATA_DIR, "answerers": ANSWERERS, "retriever": RETRIEVER,
           "gpu": gpu, "dry_run": DRY_RUN, "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/hidden.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

COMMON = ["--sweep_run", SWEEP_RUN, "--data_dir", DATA_DIR, "--run_dir", RUN_DIR, "--models_json", f"{RUN_DIR}/models.json"]
N_QUESTIONS = sum(len(json.load(open(f"{DATA_DIR}/{s}.json", encoding="utf-8"))) for s in ("dev", "test"))

def stage_done(role, key):
    p = f"{RUN_DIR}/step/hidden_{role}_{key}.jsonl"
    return os.path.exists(p) and sum(1 for _ in open(p, encoding="utf-8")) >= N_QUESTIONS'''),
    ("md", "### Extract the signals (resumable)"),
    ("code", '''STAGES = [("answer", k) for k in ANSWERERS] + [("retrieve", RETRIEVER)]
extra = ["--backend", "tiny", "--max_new", 30] if DRY_RUN else []
for role, key in STAGES:
    if stage_done(role, key):
        print(f"{role} {key}: already complete")
        continue
    if minutes_left() < 6:
        print(f"Time limit close ({minutes_left():.0f} min left) - {role} {key} not started. Run this notebook again to continue.")
        break
    ensure_local(key)
    run_stage(["extract", "--model", key, "--role", role, "--time_limit_min", max(1.0, minutes_left() - 4)] + COMMON + extra, f"extract_{role}_{key}.log")
ALL_DONE = all(stage_done(r, k) for r, k in STAGES)
print("\\nExtraction complete." if ALL_DONE else "\\nEXTRACTION NOT COMPLETE YET - the analysis below uses what is finished; run this notebook again to continue.")'''),
    ("md", "### Analyse: probes, shuffled-label controls, layer by layer, and what each signal would buy"),
    ("code", '''have = [k for k in ANSWERERS if os.path.exists(f"{RUN_DIR}/step/hidden_answer_{k}.jsonl")]
if have:
    run_stage(["analyze", "--answerers", *have, "--retriever", RETRIEVER, "--repeats", 1 if DRY_RUN else REPEATS,
               "--permutations", 3 if DRY_RUN else PERMUTATIONS] + (["--quick"] if DRY_RUN else []) + COMMON, "analyze.log")
    from IPython.display import Image, Markdown, display
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
        display(Image(f))
    os.makedirs(f"{SESSION_DIR}/results_snapshot", exist_ok=True)
    for f in ("summary.md", "set5d_hidden.xlsx", "scoreboard.jsonl", "signals_for_rules.jsonl"):
        shutil.copy(f"{RUN_DIR}/results/{f}", f"{SESSION_DIR}/results_snapshot/{f}")
    shutil.copytree(f"{RUN_DIR}/results/charts", f"{SESSION_DIR}/results_snapshot/charts", dirs_exist_ok=True)
    print("Excel workbook:", f"{RUN_DIR}/results/set5d_hidden.xlsx")
    print("Signals for the rule-writing stage:", f"{RUN_DIR}/results/signals_for_rules.jsonl")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "label": "hidden", "sweep_run": SWEEP_RUN, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES,
           "extraction_complete": ALL_DONE}, open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])


# ------------------------------------------------------------------------------ 17: final test of the frozen gate
import hashlib as _hl
def _sha(f):
    return _hl.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), "rb").read()).hexdigest()
CODE17 = ["ssw.py", "sweep.py", "routers.py", "hidden.py", "final.py"]
EXPECTED17 = {f: _sha(f) for f in CODE17}
FROZEN_SHA = json.load(open(FROZEN_JSON))["npz_sha256"] if os.path.exists(FROZEN_JSON) else "missing"
nb17 = nb([
    ("md", f"""# 17 · Set 5c: final test of the frozen gate on 200 fresh questions (A100, about 65–75 minutes)

**The question:** does the gate from notebook 16 still match the cheap fixed setup at lower compute when nothing is tuned on the test questions?

| | |
|---|---|
| The bar | Cheap fixed setup: Qwen 7B retrieves once, Qwen 32B answers once (this is what AgentTTS picks) |
| The gate | Llama 8B answers once. A probe reads its hidden state on the final answer line. The least-sure questions go to Qwen 32B |
| Frozen beforehand | Probe, layer and threshold, fitted on the 200 old questions only and saved with a checksum (`frozen/set5c_gate_v1/`) |
| Fresh questions | 200 HotpotQA questions that no earlier set used, built the same way (`data/hotpotqa_q/fresh200_c20_s43/`) |
| Success test | Accuracy within 2 points of the bar, at 15% or more less compute (gate cost included) |

What runs, in order (each stage is resumable):

| Stage | Model | Engine | My estimate |
|---|---|---|---|
| 1. Retrieve | Qwen 7B, CoT, select 3, 3 tries | vLLM (the Set 4 code, unchanged) | about 12 min including install and copy |
| 2. Big answers | Qwen 32B on the "keep all" shortlists | vLLM (the Set 4 code, unchanged) | about 12 min |
| 3. Cheap answer + hidden state | Llama 8B, one greedy answer | transformers (the notebook 16 code, unchanged) | about 40 min |
| 4. Evaluate | none | CPU | under 1 min |

The estimates come from the earlier sessions' timings; they are estimates, not measurements. **If the notebook stops at the time limit,
run it again and it continues.** Models are copied from Drive in the background while earlier stages run.

The notebook checks that Drive is serving the current code files and the unchanged frozen gate before it starts, and stops if not.

Output: `runs/{SET5_NAME}/<timestamp>_<GPU>_final/` with `results/summary.md`, `results/set5c_final_test.xlsx`, `results/per_question.jsonl`,
`results/verdict.json`, `results/charts/`.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"     # 200 questions no earlier set used
FROZEN_DIR  = f"{{PROJECT_DIR}}/frozen/set5c_gate_v1"                 # the gate, frozen on the old questions
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"   # only used by DRY_RUN
TIME_LIMIT_MIN = 80                   # stops cleanly at this limit; run the notebook again to continue (55 = two shorter sessions)
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = fake models on CPU, tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED17, indent=1)}
EXPECTED_FROZEN = "{FROZEN_SHA}"'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("md", "### Check that Drive is serving the current code and the unchanged frozen gate"),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set5code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. This has happened before when Drive had not "
                   "finished syncing. Wait two minutes, then Runtime -> Disconnect and delete runtime, and run again.")
assert os.path.exists(f"{FROZEN_DIR}/frozen_gate.json"), f"{FROZEN_DIR} is missing"
FROZEN = json.load(open(f"{FROZEN_DIR}/frozen_gate.json"))
assert sha(f"{FROZEN_DIR}/frozen_gate.npz") == FROZEN["npz_sha256"] == EXPECTED_FROZEN, "the frozen gate on Drive is not the one that was frozen"
assert os.path.exists(f"{FRESH_DATA}/test.json"), f"{FRESH_DATA} is missing"
N_QUESTIONS = sum(len(json.load(open(f"{FRESH_DATA}/{s}.json", encoding="utf-8"))) for s in ("dev", "test"))
print("Code files: current.  Frozen gate: unchanged (made", FROZEN["created"] + ", checksum", FROZEN["npz_sha256"][:16] + ").")
print("Fresh questions:", N_QUESTIONS)
print("\\nThe gate being tested:\\n ", FROZEN["rule in words"])
print("\\nThe success test:\\n ", FROZEN["success test"])

SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_final_dryrun" if DRY_RUN else "_final"
prev = sorted(d for d in glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}_final"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", f"{RUN_DIR}/step", f"{RUN_DIR}/stages", f"{RUN_DIR}/frozen", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")               # keep the exact code with the results
for f in ("frozen_gate.json", "frozen_gate.npz"):
    shutil.copy(f"{FROZEN_DIR}/{f}", f"{RUN_DIR}/frozen/{f}")            # and the exact gate
TIMES = {}
print("\\nRun folder:", RUN_DIR)'''),
    ("code", SET4_MODELS.split("def ensure_local")[0] + '''json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "5c final test", "data_dir": FRESH_DATA, "frozen_dir": FROZEN_DIR, "frozen_checksum": FROZEN["npz_sha256"],
           "gpu": gpu, "dry_run": DRY_RUN, "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(script, args, log_name):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/{script}"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

def n_lines(p):
    return sum(1 for _ in open(p, encoding="utf-8")) if os.path.exists(p) else 0

def done_retrieve():
    return n_lines(f"{RUN_DIR}/step/retrieval_q7_c1_p3.jsonl") >= N_QUESTIONS

def done_answer():
    """Every 'keep all' shortlist (after 1 try and after 3 tries) has a Qwen 32B answer."""
    if not done_retrieve():
        return False
    sys.path.insert(0, CODE_DIR)
    import sweep as W
    rows = [json.loads(l) for l in open(f"{RUN_DIR}/step/retrieval_q7_c1_p3.jsonl", encoding="utf-8")]
    need = {(r["qid"], s) for r in rows for s in (W.shortlists(r, 1, 20)[(1, "all")], W.shortlists(r, 1, 20)[(3, "all")])}
    p = f"{RUN_DIR}/step/answers_q32_c1.jsonl"
    have = {(r["qid"], tuple(r["sel"])) for r in map(json.loads, open(p, encoding="utf-8"))} if os.path.exists(p) else set()
    return need <= have

def done_extract():
    return n_lines(f"{RUN_DIR}/step/hidden_answer_l8.jsonl") >= N_QUESTIONS

STAGES = [("retrieve", "q7", done_retrieve), ("answer", "q32", done_answer), ("extract", "l8", done_extract)]
print({name: fn() for name, _, fn in STAGES})'''),
    ("md", "### Copy the models in the background, install vLLM"),
    ("code", '''import threading
COPY = {"err": None, "done": set(), "seconds": {}}
def _copy_all(keys):
    try:
        for key in keys:
            m = MODELS[key]
            src = f"{PROJECT_DIR}/models/{m['dir']}"
            if not os.path.exists(f"{m['path']}/config.json"):
                assert os.path.exists(f"{src}/config.json"), f"{src} missing - run notebook 10 first."
                idx = f"{src}/model.safetensors.index.json"
                if os.path.exists(idx):
                    lost = [f for f in sorted(set(json.load(open(idx))["weight_map"].values())) if not os.path.exists(f"{src}/{f}")]
                    assert not lost, f"{src} is incomplete (missing {lost}) - run notebook 10 again."
                t = time.time()
                shutil.rmtree(m["path"] + ".part", ignore_errors=True)
                shutil.copytree(src, m["path"] + ".part")
                os.rename(m["path"] + ".part", m["path"])               # the folder appears only when the copy is complete
                COPY["seconds"][key] = round(time.time() - t)
            COPY["done"].add(key)
    except Exception as e:
        COPY["err"] = e

def wait_local(key):
    if DRY_RUN:
        return
    t = time.time()
    while key not in COPY["done"]:
        if COPY["err"]:
            raise COPY["err"]
        time.sleep(5)
    if time.time() - t > 10:
        print(f"waited {time.time() - t:.0f}s for the copy of {MODELS[key]['dir']}")
    TIMES[f"copy_{key}"] = COPY["seconds"].get(key, 0)

todo_models = [key for _, key, fn in STAGES if not fn()]
if not DRY_RUN:
    os.makedirs(LOCAL, exist_ok=True)
    threading.Thread(target=_copy_all, args=(todo_models,), daemon=True).start()
    print("copying in the background:", [MODELS[k]["dir"] for k in todo_models] or "nothing")
NEED_VLLM = (not DRY_RUN) and not (done_retrieve() and done_answer())
if NEED_VLLM:
    t = time.time()
    !pip install -q vllm
    !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
    chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                         capture_output=True, text=True)
    print(chk.stdout or chk.stderr[-2000:])
    assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
    TIMES["install_vllm"] = round(time.time() - t)
ver = subprocess.run([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"],
                     capture_output=True, text=True).stdout.strip()
print(ver)'''),
    ("md", "### Run the three model stages on the fresh questions (resumable)"),
    ("code", '''BACKEND = "mock" if DRY_RUN else "vllm"
COMMON = ["--run_dir", RUN_DIR, "--data_dir", FRESH_DATA, "--models_json", f"{RUN_DIR}/models.json"]
for name, key, fn in STAGES:
    if fn():
        print(f"{name} ({MODELS[key]['dir']}): already complete")
        continue
    if minutes_left() < 6:
        print(f"Time limit close ({minutes_left():.0f} min left) - {name} not started. Run this notebook again to continue.")
        break
    wait_local(key)
    limit = max(1.0, minutes_left() - 3)
    if name == "retrieve":
        run_stage("final.py", ["retrieve", "--backend", BACKEND, "--time_limit_min", limit] + COMMON, "retrieve_q7.log")
    elif name == "answer":
        run_stage("final.py", ["answer", "--backend", BACKEND, "--time_limit_min", limit] + COMMON, "answer_q32.log")
    else:
        run_stage("hidden.py", ["extract", "--model", "l8", "--role", "answer", "--sweep_run", RUN_DIR, "--time_limit_min", limit] + COMMON
                  + (["--backend", "tiny", "--max_new", 30] if DRY_RUN else []), "extract_answer_l8.log")
    if not fn():
        print(f"{name} is not complete yet. Run this notebook again to continue.")
        break
ALL_DONE = all(fn() for _, _, fn in STAGES)
print("\\nAll model stages complete." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")'''),
    ("md", "### Evaluate: apply the frozen gate (nothing is fitted or chosen on the fresh questions)"),
    ("code", '''if ALL_DONE:
    USE_FROZEN = f"{RUN_DIR}/frozen"
    if DRY_RUN:      # the real gate reads a real Llama 8B; a dry run makes a throwaway gate for the fake model, from the old questions
        sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")]
        OLD_SWEEP = sweeps[-1]
        OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
        DRY_OLD = f"{RUN_DIR}/dry_old"
        os.makedirs(f"{DRY_OLD}/step", exist_ok=True)
        run_stage("hidden.py", ["extract", "--model", "l8", "--role", "answer", "--sweep_run", OLD_SWEEP, "--data_dir", OLD_DATA, "--run_dir", DRY_OLD,
                                "--models_json", f"{RUN_DIR}/models.json", "--backend", "tiny", "--max_new", 30], "dry_extract_old.log")
        USE_FROZEN = f"{RUN_DIR}/frozen_dry"
        run_stage("final.py", ["freeze", "--sweep_run", OLD_SWEEP, "--hidden_run", DRY_OLD, "--data_dir", OLD_DATA, "--models_json", f"{OLD_SWEEP}/models.json",
                               "--out_dir", USE_FROZEN, "--repeats", 1], "dry_freeze.log")
    run_stage("final.py", ["evaluate", "--frozen_dir", USE_FROZEN] + COMMON, "evaluate.log")
    from IPython.display import Image, Markdown, display
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
        display(Image(f))
    os.makedirs(f"{SESSION_DIR}/results_snapshot", exist_ok=True)
    for f in ("summary.md", "set5c_final_test.xlsx", "verdict.json", "policies.jsonl", "per_question.jsonl"):
        shutil.copy(f"{RUN_DIR}/results/{f}", f"{SESSION_DIR}/results_snapshot/{f}")
    shutil.copytree(f"{RUN_DIR}/results/charts", f"{SESSION_DIR}/results_snapshot/charts", dirs_exist_ok=True)
    print("Excel workbook:", f"{RUN_DIR}/results/set5c_final_test.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "label": "final", "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "versions": ver, "code": EXPECTED_CODE,
           "frozen_checksum": FROZEN["npz_sha256"], "complete": ALL_DONE}, open(f"{SESSION_DIR}/session.json", "w"), indent=1)
json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])


# ------------------------------------------------------------------------------ 18: retriever-side check on the fresh questions
CODE18 = ["ssw.py", "sweep.py", "routers.py", "hidden.py", "final.py", "retriever_check.py"]
EXPECTED18 = {f: _sha(f) for f in CODE18}
FROZEN18_JSON = FROZEN_JSON.replace("set5c_gate_v1/frozen_gate.json", "set5e_retriever_gate_v1/frozen_retriever_gate.json")
FROZEN18_SHA = json.load(open(FROZEN18_JSON))["npz_sha256"] if os.path.exists(FROZEN18_JSON) else "missing"
nb18 = nb([
    ("md", f"""# 18 · Set 5e: does the retriever's hidden state tell us when one retrieval try is not enough? (A100, about 10 minutes)

**The question:** notebook 16 found that the Qwen 7B's hidden state predicts "the first try finds both passages" at 0.73 on the old
questions. Does that hold on the 200 fresh questions, and is it good enough to decide where to spend extra retrieval tries?

| | |
|---|---|
| The gate | Qwen 7B reads the question and passages. A probe reads its hidden state (free: part of the retrieval pass). Where the probe says one try is likely incomplete, retrieve three times and keep every passage picked; otherwise retrieve once. Qwen 32B answers once |
| Frozen beforehand | Probe, layer and threshold, fitted on the 200 old questions only (`frozen/set5e_retriever_gate_v1/`) |
| The test (stricter than before) | The gate must find both passages more often than a random gate retrying the same share, with a 95% interval that excludes zero |
| Already available | The 3 retrieval tries and the Qwen 32B answers for the fresh questions, from notebook 17 |
| What this notebook adds | Only the Qwen 7B's hidden states on the fresh questions (about 1–2 minutes of GPU after the model is copied) |

Results are written into the notebook 17 run folder, under `results_retriever/`.

**Runtime → Change runtime type → A100 GPU.** (Any GPU with 24 GB or more also works.)"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
FROZEN_DIR  = f"{{PROJECT_DIR}}/frozen/set5e_retriever_gate_v1"
GATE_DIR    = f"{{PROJECT_DIR}}/frozen/set5c_gate_v1"                 # only for the compute unit; used by DRY_RUN
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"   # only used by DRY_RUN
DRY_RUN = False                       # True = a tiny random model on CPU (needs a dry run of notebook 17 first)

EXPECTED_CODE = {json.dumps(EXPECTED18, indent=1)}
EXPECTED_FROZEN = "{FROZEN18_SHA}"'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set5code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
FROZEN = json.load(open(f"{FROZEN_DIR}/frozen_retriever_gate.json"))
assert sha(f"{FROZEN_DIR}/frozen_retriever_gate.npz") == FROZEN["npz_sha256"] == EXPECTED_FROZEN, "the frozen retriever gate on Drive is not the one that was frozen"
tag = "_final_dryrun" if DRY_RUN else "_final"
runs = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}")) if os.path.exists(f"{d}/results/verdict.json")]
assert runs, "No finished notebook 17 run found - run notebook 17 first."
RUN_DIR = runs[-1]
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}_retriever"
for d in (SESSION_DIR, f"{RUN_DIR}/frozen_retriever"):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
for f in ("frozen_retriever_gate.json", "frozen_retriever_gate.npz"):
    shutil.copy(f"{FROZEN_DIR}/{f}", f"{RUN_DIR}/frozen_retriever/{f}")
N_QUESTIONS = sum(len(json.load(open(f"{FRESH_DATA}/{s}.json", encoding="utf-8"))) for s in ("dev", "test"))
TIMES = {}
print("Code files: current.  Frozen retriever gate: unchanged (made", FROZEN["created"] + ", checksum", FROZEN["npz_sha256"][:16] + ").")
print("Notebook 17 run used:", RUN_DIR)
print("\\nThe gate being tested:\\n ", FROZEN["rule in words"])
print("\\nThe test:\\n ", FROZEN["test"])'''),
    ("code", SET4_MODELS + '''

def run_stage(script, args, log_name):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/{script}"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = round(time.time() - t0)
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

COMMON = ["--run_dir", RUN_DIR, "--data_dir", FRESH_DATA, "--models_json", f"{RUN_DIR}/models.json"]
p = f"{RUN_DIR}/step/hidden_retrieve_q7.jsonl"
if os.path.exists(p) and sum(1 for _ in open(p, encoding="utf-8")) >= N_QUESTIONS:
    print("retriever hidden states: already extracted")
else:
    ensure_local("q7")
    run_stage("hidden.py", ["extract", "--model", "q7", "--role", "retrieve", "--sweep_run", RUN_DIR] + COMMON + (["--backend", "tiny"] if DRY_RUN else []),
              "extract_retrieve_q7.log")'''),
    ("md", "### Evaluate: apply the frozen retriever gate"),
    ("code", '''USE_FROZEN = f"{RUN_DIR}/frozen_retriever"
if DRY_RUN:      # a throwaway gate for the fake model, made from the old questions
    sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")]
    OLD_SWEEP = sweeps[-1]
    OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
    DRY_OLD = f"{RUN_DIR}/dry_old"
    os.makedirs(f"{DRY_OLD}/step", exist_ok=True)
    run_stage("hidden.py", ["extract", "--model", "q7", "--role", "retrieve", "--sweep_run", OLD_SWEEP, "--data_dir", OLD_DATA, "--run_dir", DRY_OLD,
                            "--models_json", f"{RUN_DIR}/models.json", "--backend", "tiny"], "dry_extract_old_retrieve.log")
    USE_FROZEN = f"{RUN_DIR}/frozen_retriever_dry"
    run_stage("retriever_check.py", ["freeze", "--sweep_run", OLD_SWEEP, "--hidden_run", DRY_OLD, "--data_dir", OLD_DATA, "--models_json", f"{OLD_SWEEP}/models.json",
                                     "--gate_dir", GATE_DIR, "--out_dir", USE_FROZEN, "--repeats", 1], "dry_freeze_retriever.log")
run_stage("retriever_check.py", ["evaluate", "--frozen_dir", USE_FROZEN] + COMMON, "evaluate_retriever.log")
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{RUN_DIR}/results_retriever/summary.md").read()))
for f in sorted(glob.glob(f"{RUN_DIR}/results_retriever/charts/*.png")):
    display(Image(f))
TIMES["total_session"] = round(time.time() - T_START)
ver = subprocess.run([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"],
                     capture_output=True, text=True).stdout.strip()
json.dump({"session": SESSION, "label": "retriever", "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "versions": ver, "code": EXPECTED_CODE,
           "frozen_checksum": FROZEN["npz_sha256"]}, open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("Excel workbook:", f"{RUN_DIR}/results_retriever/set5e_retriever_gate.xlsx")
print("\\nALL DONE.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])


# ------------------------------------------------------------------------------ 19: cross-family ensembles (CPU)
CODE19 = ["ssw.py", "sweep.py", "routers.py", "hidden.py", "final.py", "retriever_check.py", "ensemble.py"]
EXPECTED19 = {f: _sha(f) for f in CODE19}
nb19 = nb([
    ("md", f"""# 19 · Set 5f: do signals from different families add up? Cross-family ensembles (CPU, about 10–15 minutes)

**The question:** earlier steps combined signals only within one family. Signals from different families overlap little, so combining
them might give more than any one of them. This notebook tests that properly. **No model is run and no GPU is needed.**

| Group | Signal families |
|---|---|
| The question (free) | hand-made facts, words (TF-IDF), topics (topic model, 6 and 12 topics), similarity to the passages, sentence embedding |
| The retriever's first pass (free once it has run) | facts about the first try, hidden state at two layers |
| One cheap Llama 8B answer | hidden state at three places and two layers, evidence-use attention, token probabilities, stated confidence |

Each family is read three ways (linear, boosted trees, nearest neighbours). The resulting base models are combined 13 ways: plain and
group-balanced averages, averages of one group only, greedy selection of models that add to each other, stacking (logistic regression,
with class weighting, with SMOTE, boosted trees), and all features in one model.

Two decisions: keep the Llama 8B answer or ask the Qwen 32B (answerer); stop after one retrieval try or try twice more (retriever).

Three evaluations, so that no single sample decides: **old → fresh**, **fresh → old**, and **all 400 questions cross-validated** with the
whole pipeline refitted inside every fold.

**Named beforehand:** the primary ensemble is the average of every base model that separates better than chance on the training
questions. It "adds value" only if it beats the best single base model in the 400-question evaluation with a 95% interval that excludes
zero and is ahead in both transfer directions. Its gate must likewise beat a random gate.

Output: `runs/{SET5_NAME}/<timestamp>_cpu_ensembles/results/` with `summary.md`, `set5f_ensembles.xlsx`, `verdict.json`, and charts.

**Runtime → Change runtime type → CPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
USE_EMBEDDING = True      # also use a small sentence-embedding model for the question and passages (downloads ~90 MB once)
CV_REPEATS = 2            # repeats of the 5-fold cross-validation over all 400 questions

EXPECTED_CODE = {json.dumps(EXPECTED19, indent=1)}'''),
    ("code", MOUNT),
    ("code", '''import glob, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
T_START = time.time()
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set5code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")
          and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")]
hidden = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_hidden")) if os.path.exists(f"{d}/step/hidden_retrieve_q7.jsonl")
          and os.path.exists(f"{d}/step/hidden_answer_l8.jsonl")]
fresh = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_final")) if os.path.exists(f"{d}/step/hidden_retrieve_q7.jsonl")
         and os.path.exists(f"{d}/step/hidden_answer_l8.jsonl")]
assert sweeps and hidden and fresh, "Needs the finished runs of notebooks 11, 16, 17 and 18."
OLD_SWEEP, OLD_HIDDEN, FRESH_RUN = sweeps[-1], hidden[-1], fresh[-1]
OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
RUN_DIR = f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_cpu_ensembles"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs"):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
json.dump({"experiment": EXPERIMENT, "set": "5f cross-family ensembles", "old_sweep": OLD_SWEEP, "old_hidden": OLD_HIDDEN, "fresh_run": FRESH_RUN,
           "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "use_embedding": USE_EMBEDDING, "cv_repeats": CV_REPEATS, "code": EXPECTED_CODE, "started": SESSION},
          open(f"{RUN_DIR}/config.json", "w"), indent=1)
print("Code files: current.")
print("Old questions:  ", OLD_SWEEP, "\\n                ", OLD_HIDDEN)
print("Fresh questions:", FRESH_RUN)
print("Run folder:     ", RUN_DIR)'''),
    ("code", '''# optional packages: without them the notebook still runs (no embedding families, no SMOTE row)
import importlib.util
if USE_EMBEDDING and importlib.util.find_spec("sentence_transformers") is None:
    print("installing sentence-transformers ...")
    !pip install -q sentence-transformers
if importlib.util.find_spec("imblearn") is None:
    print("installing imbalanced-learn ...")
    !pip install -q imbalanced-learn'''),
    ("md", "### Run the study (about 10–15 minutes)"),
    ("code", '''cmd = [sys.executable, "-u", f"{CODE_DIR}/ensemble.py", "run", "--old_sweep", OLD_SWEEP, "--old_hidden", OLD_HIDDEN, "--old_data", OLD_DATA,
       "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR, "--cv_repeats", str(CV_REPEATS)] + ([] if USE_EMBEDDING else ["--no_embedding"])
with open(f"{RUN_DIR}/logs/ensemble.log", "a") as log:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        log.write(line)
        if "Warning" not in line and "warn" not in line:
            print(line, end="")
    p.wait()
assert p.returncode == 0, f"failed - see {RUN_DIR}/logs/ensemble.log"
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
    display(Image(f))
json.dump({"session": SESSION, "seconds": round(time.time() - T_START)}, open(f"{RUN_DIR}/session.json", "w"), indent=1)
print("Excel workbook:", f"{RUN_DIR}/results/set5f_ensembles.xlsx")
print(f"\\nALL DONE in {(time.time() - T_START) / 60:.0f} min.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])
nb19.metadata.pop("accelerator", None)
nb19.metadata["colab"].pop("gpuType", None)

# ------------------------------------------------------------------------------ 20: the ideas document (CPU)
CODE20 = ["ssw.py", "sweep.py", "routers.py", "hidden.py", "final.py", "retriever_check.py", "ensemble.py", "experts.py"]
EXPECTED20 = {f: _sha(f) for f in CODE20}
nb20 = nb([
    ("md", f"""# 20 · Set 5g: the ideas from Routing_Experiment_Ideas.docx, tested on this task (CPU, about 10–15 minutes)

**Step 0, scoring audit.** Exact match marks "Gardnerville, Nevada" wrong when the gold answer is "Gardnerville". Every idea below is
therefore scored twice: with a corrected label ("substantially right": exact match, or token F1 of at least 0.5, or the gold answer
contained in the answer) and with exact match as before. Every answer with its three scores is saved for hand checking.

| Idea from the document | What is tested here | Variant named beforehand |
|---|---|---|
| 1 Five-voter gate | TF-IDF cosine, Jaccard, rare-token overlap, Dice and overlap coefficient between the question and the passages picked in the first try: majority vote with the original thresholds, three voters only, thresholds retuned, and a learned model on the continuous scores | Majority vote, original thresholds |
| 2 Rare routing margins | Rarity of the question's words (training-corpus frequency), the gap between the top passage scores, and their interaction | Rarity, margin and interaction |
| 3 Benefit-targeted probes | Predict "helps" directly, and expected gain = P(helps) − P(hurts) | Expected gain |
| 4 Mixture of experts (MoE) | Three experts (evidence, structure, confidence): equal weights, fixed tuned weights, question-dependent weights from a small gate, and a gate that selects one expert | Question-dependent weights |
| 5 Bundle of experts (BoE) | The same three experts, every output kept: votes, and a joint model with disagreement, rarity, margin and named interactions | Joint model with interactions |
| 6 Adaptive weighting and sharpening | A simple adaptive rule, and per-expert temperatures and weights | Per-expert temperatures |

Sequence-aware transitions (bridge retrieval) need new model runs and are not in this notebook.

Each idea is judged three ways, on all 400 questions cross-validated and in both transfer directions (old → fresh, fresh → old):
does its gate beat a random gate at the same budget; does it add to the fixed hidden-state probe; and how does it compare with the
cheap fixed setup (what AgentTTS picks). **No model is run and no GPU is needed.**

Output: `runs/{SET5_NAME}/<timestamp>_cpu_ideas/results/` with `summary.md`, `set5g_ideas.xlsx`, `verdicts.jsonl`, `labels.jsonl`, and charts.

**Runtime → Change runtime type → CPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
USE_EMBEDDING = True      # also use a small sentence-embedding model for the question and passages (downloads ~90 MB once)
CV_REPEATS = 2            # repeats of the 5-fold cross-validation over all 400 questions

EXPECTED_CODE = {json.dumps(EXPECTED20, indent=1)}'''),
    ("code", MOUNT),
    ("code", '''import glob, hashlib, json, os, shutil, subprocess, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
T_START = time.time()
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set5code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")
          and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")]
hidden = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_hidden")) if os.path.exists(f"{d}/step/hidden_retrieve_q7.jsonl")
          and os.path.exists(f"{d}/step/hidden_answer_l8.jsonl")]
fresh = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_final")) if os.path.exists(f"{d}/step/hidden_retrieve_q7.jsonl")
         and os.path.exists(f"{d}/step/hidden_answer_l8.jsonl")]
assert sweeps and hidden and fresh, "Needs the finished runs of notebooks 11, 16, 17 and 18."
OLD_SWEEP, OLD_HIDDEN, FRESH_RUN = sweeps[-1], hidden[-1], fresh[-1]
OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
RUN_DIR = f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_cpu_ideas"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs"):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
json.dump({"experiment": EXPERIMENT, "set": "5g ideas document", "old_sweep": OLD_SWEEP, "old_hidden": OLD_HIDDEN, "fresh_run": FRESH_RUN,
           "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "use_embedding": USE_EMBEDDING, "cv_repeats": CV_REPEATS, "code": EXPECTED_CODE, "started": SESSION},
          open(f"{RUN_DIR}/config.json", "w"), indent=1)
print("Code files: current.")
print("Old questions:  ", OLD_SWEEP, "\\n                ", OLD_HIDDEN)
print("Fresh questions:", FRESH_RUN)
print("Run folder:     ", RUN_DIR)'''),
    ("code", '''# optional packages: without them the notebook still runs (no embedding families, no SMOTE row)
import importlib.util
if USE_EMBEDDING and importlib.util.find_spec("sentence_transformers") is None:
    print("installing sentence-transformers ...")
    !pip install -q sentence-transformers
if importlib.util.find_spec("imblearn") is None:
    print("installing imbalanced-learn ...")
    !pip install -q imbalanced-learn'''),
    ("md", "### Run every idea (about 10–15 minutes)"),
    ("code", '''cmd = [sys.executable, "-u", f"{CODE_DIR}/experts.py", "run", "--old_sweep", OLD_SWEEP, "--old_hidden", OLD_HIDDEN, "--old_data", OLD_DATA,
       "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR, "--cv_repeats", str(CV_REPEATS)] + ([] if USE_EMBEDDING else ["--no_embedding"])
with open(f"{RUN_DIR}/logs/experts.log", "a") as log:
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in p.stdout:
        log.write(line)
        if "Warning" not in line and "warn" not in line:
            print(line, end="")
    p.wait()
assert p.returncode == 0, f"failed - see {RUN_DIR}/logs/experts.log"
from IPython.display import Image, Markdown, display
display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
for f in sorted(glob.glob(f"{RUN_DIR}/results/charts/*.png")):
    display(Image(f))
json.dump({"session": SESSION, "seconds": round(time.time() - T_START)}, open(f"{RUN_DIR}/session.json", "w"), indent=1)
print("Excel workbook:", f"{RUN_DIR}/results/set5g_ideas.xlsx")
print(f"\\nALL DONE in {(time.time() - T_START) / 60:.0f} min.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])
nb20.metadata.pop("accelerator", None)
nb20.metadata["colab"].pop("gpuType", None)

# ------------------------------------------------------------------------------ 21: the next-experiments plan (4_2), GPU answers + CPU analysis
CODE21 = ["ssw.py", "sweep.py", "routers.py", "hidden.py", "final.py", "retriever_check.py", "ensemble.py", "experts.py", "next42.py"]
EXPECTED21 = {f: _sha(f) for f in CODE21}
nb21 = nb([
    ("md", f"""# 21 · Set 6: the next-experiments plan (4_2_Next_Routing_Experiments_Plan.docx) on the 400 development questions (A100)

**The questions:** when the retrieval looks incomplete, is it better to retry it than to escalate to Qwen 32B? Does a short, focused
context after a retry help more than the full union of passages? Can Llama 8B repair its own answer cheaply once new evidence arrives?

| Plan item | What this notebook does | Primary variant (named before running) |
|---|---|---|
| 1 Different remedies | Every question gets every action: keep the Llama 8B answer; Qwen 32B on the same passages; retry retrieval and answer again. A three-action rule (retriever probe, then answerer probe) is compared with the answerer-only gate, the retriever-only gate, random with the same action mix, and the cheap fixed setup | Retry if the retriever probe is low; else escalate if the answer probe is low; else keep. Two thresholds fitted on training questions under the fixed setup's compute |
| 2 Adaptive context | After a retry: the full union of passages vs a bounded shortlist of 2, 3 or 4 (relevance + coverage − redundancy, fixed weights). Gold-only context as a diagnostic ceiling | Probe-triggered retry, then the bounded shortlist with k chosen on training questions |
| 3 Answer revision | After a retry: Llama 8B answers from scratch, Llama 8B revises its first answer, Qwen 32B answers | One Llama 8B revision, no extra escalation |
| 6, 5, 4 (CPU) | Expert ablation, stable escalation benefit, cheap support features (same code as the CPU run already reported) | As in the plan |

What runs, in order (every stage is resumable: if the session stops, run the notebook again and it continues):

| Stage | Model | Answers | My estimate |
|---|---|---|---|
| Install vLLM, copy the two models | - | - | about 8–12 min |
| 1. Llama 8B: first answer, answers on every new context, revisions | Llama 8B, greedy, up to 512 tokens | about 2,400 | about 10–20 min |
| 2. Qwen 32B (AWQ): first shortlist and every new context | Qwen 32B, greedy, up to 512 tokens | about 1,500 | about 15–30 min |
| 3. Analysis (items 1–3) and the CPU items (6, 5, 4) | none | - | about 10–15 min |

These are estimates, not measurements; notebook 19 took more than twice my estimate, so allow up to 1.5 hours. No new retrieval is run: the
three retrieval tries saved by notebooks 11 and 17 are reused. Nothing is frozen and nothing touches new questions: everything here is development.

Output: `runs/{SET5_NAME}/<timestamp>_<GPU>_set6/` with `results_actions/` (summary.md, set6_actions.xlsx, charts) and `results_cpu/`.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
TIME_LIMIT_MIN = 100                  # stops cleanly at this limit; run the notebook again to continue
CV_REPEATS = 2                        # repeats of the 5-fold cross-validation over all 400 questions
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = fake models on CPU, tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED21, indent=1)}'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("md", "### Check that Drive is serving the current code; find the earlier runs"),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set6code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")
          and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")]
hidden = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_hidden")) if os.path.exists(f"{d}/step/hidden_retrieve_q7.jsonl")
          and os.path.exists(f"{d}/step/hidden_answer_l8.jsonl")]
fresh = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_final")) if os.path.exists(f"{d}/step/hidden_retrieve_q7.jsonl")
         and os.path.exists(f"{d}/step/hidden_answer_l8.jsonl")]
assert sweeps and hidden and fresh, "Needs the finished runs of notebooks 11, 16, 17 and 18."
OLD_SWEEP, OLD_HIDDEN, FRESH_RUN = sweeps[-1], hidden[-1], fresh[-1]
OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_set6_dryrun" if DRY_RUN else "_set6"
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", f"{RUN_DIR}/step", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
TIMES = {}
print("Code files: current.")
print("Old questions:  ", OLD_SWEEP, "\\n                ", OLD_HIDDEN)
print("Fresh questions:", FRESH_RUN)
print("Run folder:     ", RUN_DIR)'''),
    ("code", SET4_MODELS.split("def ensure_local")[0] + '''MODELS = {k: v for k, v in MODELS.items() if k in ("q7", "l8", "q32")}
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "6 next-experiments plan", "old_sweep": OLD_SWEEP, "old_hidden": OLD_HIDDEN, "fresh_run": FRESH_RUN,
           "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "cv_repeats": CV_REPEATS, "gpu": gpu, "dry_run": DRY_RUN, "code": EXPECTED_CODE, "started": SESSION},
          open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/next42.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ", "Warning", "warn")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

GEN = ["--old_sweep", OLD_SWEEP, "--old_data", OLD_DATA, "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR,
       "--models_json", f"{RUN_DIR}/models.json"]
def pending(key):
    r = subprocess.run([sys.executable, f"{CODE_DIR}/next42.py", "generate", "--model", key, "--count_only"] + GEN, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])["pending"]
TODO = {k: pending(k) for k in ("l8", "q32")}
print("answers still to generate:", TODO)'''),
    ("md", "### Copy the models in the background, install vLLM"),
    ("code", '''import threading
COPY = {"err": None, "done": set(), "seconds": {}}
def _copy_all(keys):
    try:
        for key in keys:
            m = MODELS[key]
            src = f"{PROJECT_DIR}/models/{m['dir']}"
            if not os.path.exists(f"{m['path']}/config.json"):
                assert os.path.exists(f"{src}/config.json"), f"{src} missing - run notebook 10 first."
                t = time.time()
                shutil.rmtree(m["path"] + ".part", ignore_errors=True)
                shutil.copytree(src, m["path"] + ".part")
                os.rename(m["path"] + ".part", m["path"])               # the folder appears only when the copy is complete
                COPY["seconds"][key] = round(time.time() - t)
            COPY["done"].add(key)
    except Exception as e:
        COPY["err"] = e

def wait_local(key):
    if DRY_RUN:
        return
    t = time.time()
    while key not in COPY["done"]:
        if COPY["err"]:
            raise COPY["err"]
        time.sleep(5)
    if time.time() - t > 10:
        print(f"waited {time.time() - t:.0f}s for the copy of {MODELS[key]['dir']}")
    TIMES[f"copy_{key}"] = COPY["seconds"].get(key, 0)

todo_models = [k for k in ("l8", "q32") if TODO[k]]
if not DRY_RUN and todo_models:
    os.makedirs(LOCAL, exist_ok=True)
    threading.Thread(target=_copy_all, args=(todo_models,), daemon=True).start()
    print("copying in the background:", [MODELS[k]["dir"] for k in todo_models])
if not DRY_RUN and todo_models:
    t = time.time()
    !pip install -q vllm
    !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
    chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                         capture_output=True, text=True)
    print(chk.stdout or chk.stderr[-2000:])
    assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
    TIMES["install_vllm"] = round(time.time() - t)
else:
    print("nothing to generate - no install needed")
ver = subprocess.run([sys.executable, "-c", "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__)"],
                     capture_output=True, text=True).stdout.strip()
print(ver)'''),
    ("md", "### Generate the answers (Llama 8B, then Qwen 32B; resumable)"),
    ("code", '''BACKEND = "mock" if DRY_RUN else "vllm"
for key in ("l8", "q32"):
    if not pending(key):
        print(f"{MODELS[key]['dir']}: every answer is already there")
        continue
    if minutes_left() < 8:
        print(f"Time limit close ({minutes_left():.0f} min left) - {key} not started. Run this notebook again to continue.")
        break
    wait_local(key)
    run_stage(["generate", "--model", key, "--backend", BACKEND, "--time_limit_min", max(1.0, minutes_left() - 4)] + GEN, f"generate_{key}.log")
    if pending(key):
        print(f"{key} is not complete yet. Run this notebook again to continue.")
        break
ALL_DONE = not pending("l8") and not pending("q32")
print("\\nAll answers generated." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")'''),
    ("md", "### Analysis: items 1–3 (different remedies, context size, revision), then the CPU items 6, 5 and 4"),
    ("code", '''if ALL_DONE:
    from IPython.display import Image, Markdown, display
    SETS = ["--old_sweep", OLD_SWEEP, "--old_hidden", OLD_HIDDEN, "--old_data", OLD_DATA, "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR]
    for flag, sub in (([], "results_actions"), (["--first_from_extraction"], "results_actions_probe_answer")):
        if not os.path.exists(f"{RUN_DIR}/{sub}/summary.md"):
            run_stage(["actions", "--cv_repeats", CV_REPEATS] + flag + SETS, f"{sub}.log")
    # results_actions_probe_answer is the fair item 1 test: the kept Llama answer is the one whose hidden state the probe reads
    display(Markdown(open(f"{RUN_DIR}/results_actions_probe_answer/summary.md").read()))
    for f in sorted(glob.glob(f"{RUN_DIR}/results_actions_probe_answer/charts/*.png")):
        display(Image(f))
    if not os.path.exists(f"{RUN_DIR}/results_cpu/summary.md"):
        import importlib.util
        if importlib.util.find_spec("sentence_transformers") is None:
            print("installing sentence-transformers ...")
            !pip install -q sentence-transformers
        run_stage(["cpu", "--cv_repeats", CV_REPEATS] + SETS, "cpu.log")
    display(Markdown(open(f"{RUN_DIR}/results_cpu/summary.md").read()))
    print("Excel workbooks:", f"{RUN_DIR}/results_actions/set6_actions.xlsx", f"{RUN_DIR}/results_cpu/set6_cpu.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "versions": ver, "code": EXPECTED_CODE, "complete": ALL_DONE},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])

# ------------------------------------------------------------------------------ 22: Set 7, Qwen 3B learns to shortlist (LoRA)
SET7_NAME = "Set_3__self-learnt-retrieval"
CODE22 = ["ssw.py", "sweep.py", "routers.py", "hidden.py", "final.py", "retriever_check.py", "ensemble.py", "experts.py", "retft.py"]
EXPECTED22 = {f: _sha(f) for f in CODE22}
nb22 = nb([
    ("md", f"""# 22 · Set 7: Qwen 3B learns to shortlist the right passages (LoRA fine-tuning, A100)

**The question:** if the small retriever is taught, from HotpotQA's own labels, which passages answer a question, does it shortlist
better than today's retriever, and do the answers get better and cheaper? No router: the model learns the rule itself.

| | |
|---|---|
| Input | The question and its 20 candidate passages, as text: the same construction as every earlier set |
| Target | The text line "Selected references: [a] [b]" naming the gold passages |
| Model | Qwen 2.5 3B Instruct with a LoRA adapter (about 30 million trainable weights; the base model is unchanged) |
| Training questions | 4,000 questions from HotpotQA's **training** file (hard level, as the test questions), built like our questions |
| Validation | 300 further training-file questions: used only to pick the best checkpoint |
| Test | The 400 questions of notebooks 17–21 (HotpotQA dev): never used for training or choices |
| Answerers | Unchanged: Qwen 32B and Llama 8B, one greedy step-by-step answer (notebook 21's prompt; its answers are reused) |
| Compared with | Qwen 7B retrieving (the cheap fixed setup's retriever), Qwen 3B before training (two prompts), the gold passages (ceiling) |

What runs, in order (every stage is resumable: if the session stops, run the notebook again):

| Stage | My estimate |
|---|---|
| Build the training questions (downloads HotpotQA's training file, about 570 MB) | 5–10 min, once |
| Copy Qwen 3B, install peft, LoRA training with 3 checkpoints scored on validation | 45–90 min |
| Install vLLM; retrieval on the 400 test questions (3 variants) | 10–15 min |
| Copy Llama 8B and Qwen 32B (in the background during training); about 950 new answers each | 15–25 min |
| Evaluation | under 1 min |

These are estimates, not measurements; allow up to 2.5 hours in total. Nothing touches the untouched questions kept for a final confirmation.

Output: `runs/{SET7_NAME}/<timestamp>_<GPU>_ft3b/` with `adapter/` (the trained LoRA weights and the training curve) and `results/`
(`summary.md`, `set7_retriever_finetune.xlsx`).

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET7_NAME}"
EXP_SET5    = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
N_TRAIN = 4000                        # training questions (the curve is scored at 1/4, 1/2 and all of them)
N_VAL   = 300                         # validation questions (checkpoint choice only)
TIME_LIMIT_MIN = 170                  # stops cleanly at this limit; run the notebook again to continue
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = tiny model and fake answers on CPU: tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED22, indent=1)}'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("md", "### Check that Drive is serving the current code; find the earlier runs"),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set7code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")
          and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")]
fresh = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXP_SET5}/*_final")) if os.path.exists(f"{d}/step/retrieval_q7_c1_p3.jsonl")]
set6 = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXP_SET5}/*_set6")) if os.path.exists(f"{d}/step/arms_q32.jsonl")]
assert sweeps and fresh, "Needs the finished runs of notebooks 11 and 17."
OLD_SWEEP, FRESH_RUN = sweeps[-1], fresh[-1]
CACHE_RUN = set6[-1] if set6 else ""
OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
TRAIN_DATA = f"{PROJECT_DIR}/data/hotpotqa_train/train{N_TRAIN}_val{N_VAL}_c20_hard_s7" + ("_dryrun" if DRY_RUN else "")
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_ft3b_dryrun" if DRY_RUN else "_ft3b"
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", f"{RUN_DIR}/step", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
TIMES = {}
print("Code files: current.")
print("Old questions:   ", OLD_SWEEP)
print("Fresh questions: ", FRESH_RUN)
print("Answers to reuse:", CACHE_RUN or "none")
print("Training data:   ", TRAIN_DATA)
print("Run folder:      ", RUN_DIR)'''),
    ("code", SET4_MODELS.split("def ensure_local")[0] + '''MODELS = {k: v for k, v in MODELS.items() if k in ("q3", "q7", "l8", "q32")}
MODELS["q3ft"] = dict(MODELS["q3"], id="Qwen 3B + LoRA trained in this run", dir="q3ft_merged", path=f"{LOCAL}/q3ft_merged")
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "7 retriever fine-tuning (Qwen 3B, LoRA)", "old_sweep": OLD_SWEEP, "fresh_run": FRESH_RUN, "cache_run": CACHE_RUN,
           "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "train_data": TRAIN_DATA, "n_train": N_TRAIN, "n_val": N_VAL, "gpu": gpu, "dry_run": DRY_RUN,
           "code": EXPECTED_CODE, "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/retft.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ", "Warning", "warn")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

ADAPTER = f"{RUN_DIR}/adapter"
def trained():
    s = f"{ADAPTER}/train_state.json"
    return os.path.exists(s) and "chosen" in json.load(open(s))
def merged():
    return os.path.exists(f"{MODELS['q3ft']['path']}/config.json")
print("training data built:", os.path.exists(f"{TRAIN_DATA}/val.json"), "| trained:", trained())'''),
    ("md", "### 1. Build the training questions (once; kept on Drive)"),
    ("code", '''if not os.path.exists(f"{TRAIN_DATA}/val.json"):
    if DRY_RUN:     # stand-in for the training file: the dev file itself (only to test the code; never used for real)
        run_stage(["build_train", "--hotpot_train", f"{PROJECT_DIR}/AgentTTS/data/HotpotQA/hotpotqa-dev.json", "--out_dir", TRAIN_DATA,
                   "--n_train", 80, "--n_val", 12, "--pool", 1500], "build_train.log")
    else:
        run_stage(["build_train", "--out_dir", TRAIN_DATA, "--n_train", N_TRAIN, "--n_val", N_VAL], "build_train.log")
else:
    print("training questions already built:", TRAIN_DATA)
print(json.load(open(f"{TRAIN_DATA}/data_config.json")))'''),
    ("md", "### 2. Copy the models (Llama 8B and Qwen 32B in the background), install peft, train the LoRA adapter"),
    ("code", '''import threading
COPY = {"err": None, "done": set(), "seconds": {}}
def _copy_all(keys):
    try:
        for key in keys:
            m = MODELS[key]
            src = f"{PROJECT_DIR}/models/{m['dir']}"
            if not os.path.exists(f"{m['path']}/config.json"):
                assert os.path.exists(f"{src}/config.json"), f"{src} missing - run notebook 10 first."
                t = time.time()
                shutil.rmtree(m["path"] + ".part", ignore_errors=True)
                shutil.copytree(src, m["path"] + ".part")
                os.rename(m["path"] + ".part", m["path"])
                COPY["seconds"][key] = round(time.time() - t)
            COPY["done"].add(key)
    except Exception as e:
        COPY["err"] = e

def wait_local(key):
    if DRY_RUN:
        return
    t = time.time()
    while key not in COPY["done"]:
        if COPY["err"]:
            raise COPY["err"]
        time.sleep(5)
    if time.time() - t > 10:
        print(f"waited {time.time() - t:.0f}s for the copy of {MODELS[key]['dir']}")
    TIMES[f"copy_{key}"] = COPY["seconds"].get(key, 0)

if not DRY_RUN:
    os.makedirs(LOCAL, exist_ok=True)
    threading.Thread(target=_copy_all, args=(["q3", "l8", "q32"],), daemon=True).start()
    print("copying in the background: Qwen 3B first, then Llama 8B and Qwen 32B")
import importlib.util
if importlib.util.find_spec("peft") is None:
    print("installing peft ...")
    !pip install -q peft
TRAIN = ["train", "--data_dir", TRAIN_DATA, "--out_dir", ADAPTER, "--merged_dir", MODELS["q3ft"]["path"], "--n_train", N_TRAIN, "--n_val", N_VAL]
if DRY_RUN:
    run_stage(TRAIN + ["--backend", "tiny", "--n_train", 80, "--n_val", 6, "--grad_accum", 4, "--max_len", 9000], "train.log")
elif not (trained() and merged()):
    wait_local("q3")
    run_stage(TRAIN + ["--base", MODELS["q3"]["path"], "--time_limit_min", max(5.0, minutes_left() - 50)], "train.log")
TRAIN_DONE = trained() and (merged() or DRY_RUN)
print("\\nTraining complete." if TRAIN_DONE else "\\nTraining NOT COMPLETE YET - run this notebook again to continue from the last checkpoint.")'''),
    ("md", "### 3. Retrieval on the 400 test questions, then answers (vLLM)"),
    ("code", '''BACKEND = "mock" if DRY_RUN else "vllm"
SETS = ["--old_sweep", OLD_SWEEP, "--old_data", OLD_DATA, "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR,
        "--models_json", f"{RUN_DIR}/models.json", "--backend", BACKEND]
ALL_DONE = False
if TRAIN_DONE:
    if not DRY_RUN and importlib.util.find_spec("vllm") is None:
        t = time.time()
        !pip install -q vllm
        !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
        chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                             capture_output=True, text=True)
        print(chk.stdout or chk.stderr[-2000:])
        assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
        TIMES["install_vllm"] = round(time.time() - t)
    run_stage(["retrieve"] + SETS, "retrieve.log")
    for key in ("l8", "q32"):
        wait_local(key)
    run_stage(["answer", "--cache_run", CACHE_RUN] + SETS, "answer.log")
    ALL_DONE = True'''),
    ("md", "### 4. Results"),
    ("code", '''if ALL_DONE:
    from IPython.display import Markdown, display
    run_stage(["evaluate", "--adapter_dir", ADAPTER] + SETS, "evaluate.log")
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    print("Excel workbook:", f"{RUN_DIR}/results/set7_retriever_finetune.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "code": EXPECTED_CODE, "complete": ALL_DONE},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])

# ------------------------------------------------------------------------------ 23a / 23b: Set 7b, more training (3B) and the 7B retriever
CODE23 = CODE22
EXPECTED23 = {f: _sha(f) for f in CODE23}
N_TRAIN_DATA23 = 12000


def nb23(model, n_train, letter, est_train, est_total):
    size = {"q3": "3B", "q7": "7B"}[model]
    name = f"ft{size[0]}_{n_train // 1000}k"
    key = f"{model}ft{n_train // 1000}k"
    ck = [2000, 4000, 8000] + ([12000] if n_train >= 12000 else [])
    v = nb([
        ("md", f"""# 23{letter} · Set 7b: Qwen {size} learns to shortlist, trained on {n_train:,} questions (LoRA, A100)

Notebook 22 trained Qwen 3B on about 4,000 questions: it then found both gold passages on 84.8% of the test questions (Qwen 7B
untrained: 79.5%), and with Qwen 32B answering it beat the cheap fixed setup by about 4 points at 42% less compute. Its
validation curve was still rising. This notebook trains **Qwen {size}** on **{n_train:,}** questions, with the same recipe.

Notebooks 23a (Qwen 3B, 12,000 questions) and 23b (Qwen 7B, 8,000 questions) are independent: run them in two Colab sessions at
the same time if you can. They share the training questions (whichever starts first builds them; about 10 minutes, once).

| | |
|---|---|
| Same as notebook 22 | Input (question + 20 passages as text), target ("Selected references: [a] [b]"), LoRA r = 16, learning rate, test questions, answerers |
| Training questions | {n_train:,} of the 12,000 hard-level questions built from HotpotQA's training file (all 15,661 hard questions form the distractor corpus) |
| Validation | 300 further training-file questions, scored after {", ".join(f"{c:,}" for c in ck if c <= n_train)} questions; the best checkpoint is kept |
| Faster training | 4 questions per GPU pass, and the loss is computed only on the answer tokens (same result, less memory and time) |
| Resumable | A resume point is saved every 1,000 questions; if the session stops, run the notebook again |

**My estimate:** training {est_train}; about {est_total} in total with the retrieval, the answers and the model copies. These are
estimates (notebook 22's 4,000 questions took 67 minutes on the 3B model, one question per pass).

Output: `runs/{SET7_NAME}/<timestamp>_<GPU>_{name}/` with `adapter/` (training curve, checkpoints) and `results/summary.md`.

**Runtime → Change runtime type → A100 GPU.**"""),
        ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET7_NAME}"
EXP_SET5    = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
MODEL   = "{model}"                   # the retriever to train: Qwen {size}
N_TRAIN = {n_train}                      # training questions used (of the {N_TRAIN_DATA23:,} built)
N_VAL   = 300                         # validation questions (checkpoint choice only)
CKPT_QUESTIONS = {ck}     # validation checkpoints, in questions seen
TIME_LIMIT_MIN = 600                  # stops cleanly at this limit; run the notebook again to continue
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = tiny model and fake answers on CPU: tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED23, indent=1)}'''),
        ("code", GPU_CHECK),
        ("code", MOUNT),
        ("md", "### Check that Drive is serving the current code; find the earlier runs"),
        ("code", f'''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set7code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{{PROJECT_DIR}}/code/{{f}}", f"{{CODE_DIR}}/{{f}}")
    if sha(f"{{CODE_DIR}}/{{f}}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {{stale}} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{OLD_SWEEP_EXPERIMENT}}/*")) if "_dryrun" not in d and os.path.exists(f"{{d}}/results/per_question.jsonl")
          and os.path.exists(f"{{d}}/sweep_status.json") and json.load(open(f"{{d}}/sweep_status.json")).get("complete")]
fresh = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXP_SET5}}/*_final")) if os.path.exists(f"{{d}}/step/retrieval_q7_c1_p3.jsonl")]
set6 = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXP_SET5}}/*_set6")) if os.path.exists(f"{{d}}/step/arms_q32.jsonl")]
set7 = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXPERIMENT}}/*_ft*")) if "_dryrun" not in d and os.path.exists(f"{{d}}/step/answers_q32.jsonl")]
assert sweeps and fresh, "Needs the finished runs of notebooks 11 and 17."
OLD_SWEEP, FRESH_RUN = sweeps[-1], fresh[-1]
CACHE_RUN = set6[-1] if set6 else ""
OLD_DATA = json.load(open(f"{{OLD_SWEEP}}/config.json"))["data_dir"]
TRAIN_DATA = f"{{PROJECT_DIR}}/data/hotpotqa_train/train{N_TRAIN_DATA23}_val300_c20_hard_all_s7" + ("_dryrun" if DRY_RUN else "")
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_{name}_dryrun" if DRY_RUN else "_{name}"
prev = sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXPERIMENT}}/*{{tag}}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{{PROJECT_DIR}}/runs/{{EXPERIMENT}}/{{SESSION}}_{{GPU_TAG}}{{tag}}"
CACHE_ANSWERS = [d for d in set7 if d != RUN_DIR]
SESSION_DIR = f"{{RUN_DIR}}/sessions/{{SESSION}}"
for d in (f"{{RUN_DIR}}/code", f"{{RUN_DIR}}/logs", f"{{RUN_DIR}}/step", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{{CODE_DIR}}/{{f}}", f"{{RUN_DIR}}/code/{{f}}")
TIMES = {{}}
print("Code files: current.")
print("Old questions:   ", OLD_SWEEP)
print("Fresh questions: ", FRESH_RUN)
print("Answers to reuse:", [CACHE_RUN] + CACHE_ANSWERS)
print("Training data:   ", TRAIN_DATA)
print("Run folder:      ", RUN_DIR)'''),
        ("code", SET4_MODELS.split("def ensure_local")[0] + f'''MODELS = {{k: v for k, v in MODELS.items() if k in ("q3", "q7", "l8", "q32")}}
FT_KEY, FT_NAME = "{key}", "{name}"
FT_LABEL = f"Qwen {size}, fine-tuned on {{N_TRAIN:,}} questions (direct prompt)"
MODELS[FT_KEY] = dict(MODELS[MODEL], id=f"{{MODELS[MODEL]['id']}} + LoRA trained in this run", dir=FT_KEY + "_merged", path=f"{{LOCAL}}/{{FT_KEY}}_merged")
json.dump(MODELS, open(f"{{RUN_DIR}}/models.json", "w"), indent=1)
json.dump({{"experiment": EXPERIMENT, "set": f"7b retriever fine-tuning ({{MODEL}}, {{N_TRAIN}} questions)", "old_sweep": OLD_SWEEP, "fresh_run": FRESH_RUN,
           "cache": [CACHE_RUN] + CACHE_ANSWERS, "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "train_data": TRAIN_DATA, "n_train": N_TRAIN, "n_val": N_VAL,
           "gpu": gpu, "dry_run": DRY_RUN, "code": EXPECTED_CODE, "started": SESSION}}, open(f"{{RUN_DIR}}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name):
    cmd = [sys.executable, "-u", f"{{CODE_DIR}}/retft.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{{RUN_DIR}}/logs/{{log_name}}", "a") as log, open(f"{{SESSION_DIR}}/{{log_name}}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ", "Warning", "warn", "Loading weights", "Writing model")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{{SESSION_DIR}}/stage_times.json", "w"), indent=1)
    print(f"--- {{log_name}}: exit {{p.returncode}}, {{time.time() - t0:.0f}}s, {{minutes_left():.0f}} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {{RUN_DIR}}/logs/{{log_name}}"

ADAPTER = f"{{RUN_DIR}}/adapter"
def trained():
    s = f"{{ADAPTER}}/train_state.json"
    return os.path.exists(s) and "chosen" in json.load(open(s))
def merged():
    return os.path.exists(f"{{MODELS[FT_KEY]['path']}}/config.json")
print("training data built:", os.path.exists(f"{{TRAIN_DATA}}/val.json"), "| trained:", trained())'''),
        ("md", "### 1. Build the training questions (once, shared by 23a and 23b; kept on Drive)"),
        ("code", f'''if not os.path.exists(f"{{TRAIN_DATA}}/val.json"):
    if DRY_RUN:     # stand-in for the training file: the dev file itself (only to test the code; never used for real)
        run_stage(["build_train", "--hotpot_train", f"{{PROJECT_DIR}}/AgentTTS/data/HotpotQA/hotpotqa-dev.json", "--out_dir", TRAIN_DATA,
                   "--n_train", 80, "--n_val", 12, "--pool", 1500], "build_train.log")
    else:
        run_stage(["build_train", "--out_dir", TRAIN_DATA, "--n_train", {N_TRAIN_DATA23}, "--n_val", 300, "--pool", 0], "build_train.log")
else:
    print("training questions already built:", TRAIN_DATA)
print(json.load(open(f"{{TRAIN_DATA}}/data_config.json")))'''),
        ("md", f"### 2. Copy the models (Llama 8B and Qwen 32B in the background), install peft, train the LoRA adapter on Qwen {size}"),
        ("code", '''import threading
COPY = {"err": None, "done": set(), "seconds": {}}
def _copy_all(keys):
    try:
        for key in keys:
            m = MODELS[key]
            src = f"{PROJECT_DIR}/models/{m['dir']}"
            if not os.path.exists(f"{m['path']}/config.json"):
                assert os.path.exists(f"{src}/config.json"), f"{src} missing - run notebook 10 first."
                t = time.time()
                shutil.rmtree(m["path"] + ".part", ignore_errors=True)
                shutil.copytree(src, m["path"] + ".part")
                os.rename(m["path"] + ".part", m["path"])
                COPY["seconds"][key] = round(time.time() - t)
            COPY["done"].add(key)
    except Exception as e:
        COPY["err"] = e

def wait_local(key):
    if DRY_RUN:
        return
    t = time.time()
    while key not in COPY["done"]:
        if COPY["err"]:
            raise COPY["err"]
        time.sleep(5)
    if time.time() - t > 10:
        print(f"waited {time.time() - t:.0f}s for the copy of {MODELS[key]['dir']}")
    TIMES[f"copy_{key}"] = COPY["seconds"].get(key, 0)

if not DRY_RUN:
    os.makedirs(LOCAL, exist_ok=True)
    threading.Thread(target=_copy_all, args=([MODEL, "l8", "q32"],), daemon=True).start()
    print(f"copying in the background: {MODELS[MODEL]['dir']} first, then Llama 8B and Qwen 32B")
import importlib.util
if importlib.util.find_spec("peft") is None:
    print("installing peft ...")
    !pip install -q peft
TRAIN = ["train", "--data_dir", TRAIN_DATA, "--out_dir", ADAPTER, "--merged_dir", MODELS[FT_KEY]["path"], "--n_train", N_TRAIN, "--n_val", N_VAL,
         "--ckpt_questions"] + CKPT_QUESTIONS + ["--save_every_questions", 1000, "--micro_batch", 4]
if DRY_RUN:
    run_stage(TRAIN + ["--backend", "tiny", "--n_train", 80, "--n_val", 6, "--grad_accum", 4, "--micro_batch", 2, "--max_len", 9000,
                       "--ckpt_questions", 20, 40, 80, "--save_every_questions", 16], "train.log")
elif not (trained() and merged()):
    wait_local(MODEL)
    run_stage(TRAIN + ["--base", MODELS[MODEL]["path"], "--time_limit_min", max(5.0, minutes_left() - 45)], "train.log")
TRAIN_DONE = trained() and (merged() or DRY_RUN)
print("\\nTraining complete." if TRAIN_DONE else "\\nTraining NOT COMPLETE YET - run this notebook again to continue from the last resume point.")'''),
        ("md", "### 3. Retrieval on the 400 test questions, then answers (vLLM)"),
        ("code", '''BACKEND = "mock" if DRY_RUN else "vllm"
SETS = ["--old_sweep", OLD_SWEEP, "--old_data", OLD_DATA, "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR,
        "--models_json", f"{RUN_DIR}/models.json", "--backend", BACKEND, "--variants", FT_NAME, "--add_variant", f"{FT_NAME}:{FT_KEY}:{FT_LABEL}"]
ALL_DONE = False
if TRAIN_DONE:
    if not DRY_RUN and importlib.util.find_spec("vllm") is None:
        t = time.time()
        !pip install -q vllm
        !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
        chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                             capture_output=True, text=True)
        print(chk.stdout or chk.stderr[-2000:])
        assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
        TIMES["install_vllm"] = round(time.time() - t)
    run_stage(["retrieve"] + SETS, "retrieve.log")
    for key in ("l8", "q32"):
        wait_local(key)
    run_stage(["answer", "--cache_run", CACHE_RUN, "--cache_answers"] + CACHE_ANSWERS + SETS, "answer.log")
    ALL_DONE = True'''),
        ("md", "### 4. Results"),
        ("code", '''if ALL_DONE:
    from IPython.display import Markdown, display
    run_stage(["evaluate", "--adapter_dir", ADAPTER] + SETS, "evaluate.log")
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    print("Excel workbook:", f"{RUN_DIR}/results/set7_retriever_finetune.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "code": EXPECTED_CODE, "complete": ALL_DONE},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
    ])
    return v


nb23a = nb23("q3", 12000, "a", "about 1.5–3 hours", "2–3.5 hours")
nb23b = nb23("q7", 8000, "b", "about 2.5–4.5 hours", "3–5 hours")

# ------------------------------------------------------------------------------ 24: Set 7c, the answer gate on top of the fine-tuned 3B retriever
CODE24 = CODE22 + ["gate7.py"]
EXPECTED24 = {f: _sha(f) for f in CODE24}
nb24 = nb([
    ("md", f"""# 24 · Set 7c: can an answer gate keep the new pipeline's accuracy at lower compute? (A100, about 1–1.5 hours)

**The question:** with the fine-tuned Qwen 3B retriever (notebook 23a), Qwen 32B answering every question reaches 84.2% (corrected
label) / 64.0% (exact match). Can Llama 8B answer most questions and hand only its least-sure answers to Qwen 32B, keeping most of
that accuracy while saving more compute?

**Why this needs the GPU:** the probe must read the hidden state of the exact Llama 8B answer it judges. The new retriever changes
the passages, hence Llama's answers and hidden states. So Llama 8B answers again on the new shortlists, and the hidden states of
that same pass are saved (same code as notebooks 16–17).

| | |
|---|---|
| Shortlists | The fine-tuned 3B retriever's (notebook 23a), unchanged |
| Llama 8B | One greedy step-by-step answer (same prompt as before), hidden states saved from that same pass |
| Qwen 32B | Its answers on the same shortlists already exist (notebook 23a) |
| Fitted inside training folds | The probe (layer 28 of 32, final answer line, 16 components, logistic regression), its target ("Llama's answer is no worse than Qwen's"), and the share escalated |
| Policy named beforehand | The smallest escalation share whose training accuracy is within 1 point of always-Qwen; also fixed shares of 30%, 40% and 50% |
| Judged against | Always Qwen 32B on the new shortlists (the improved pipeline), the cheap fixed setup (AgentTTS's pick), Llama 8B alone, and a random gate with the same share |
| Evaluation | All 400 questions, 5-fold cross-validation repeated 5 times; and old → fresh, fresh → old |

**My estimate:** Llama 8B copy about 5–8 minutes, answers with hidden states about 45–75 minutes, analysis about 2 minutes.
If the session stops, run the notebook again: it continues.

Output: `runs/{SET7_NAME}/<timestamp>_<GPU>_gate3_12k/results/` (`summary.md`, `set7c_answer_gate.xlsx`, `charts/gate_curve.png`).

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET7_NAME}"
EXP_SET5    = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
RETRIEVER_RUN_TAG = "_ft3_12k"        # the fine-tuned retriever whose shortlists are used (notebook 23a)
CV_REPEATS = 5
TIME_LIMIT_MIN = 150                  # stops cleanly at this limit; run the notebook again to continue
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = tiny model on CPU: tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED24, indent=1)}'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("md", "### Check that Drive is serving the current code; find the earlier runs"),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set7code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")
          and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")]
fresh = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXP_SET5}/*_final")) if os.path.exists(f"{d}/step/retrieval_q7_c1_p3.jsonl")]
ft = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{RETRIEVER_RUN_TAG}")) if os.path.exists(f"{d}/step/answers_q32.jsonl")]
assert sweeps and fresh and ft, "Needs the finished runs of notebooks 11, 17 and 23a."
OLD_SWEEP, FRESH_RUN, FT_RUN = sweeps[-1], fresh[-1], ft[-1]
RET_FILE = glob.glob(f"{FT_RUN}/step/retrieval_ft*.jsonl")[0]
OTHER_ANSWERS = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_ft*")) if d != FT_RUN and "_dryrun" not in d and os.path.exists(f"{d}/step/answers_q32.jsonl")]
OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_gate3_12k_dryrun" if DRY_RUN else "_gate3_12k"
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", f"{RUN_DIR}/step", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
TIMES = {}
print("Code files: current.")
print("Fine-tuned retriever run:", FT_RUN)
print("Its shortlists:          ", RET_FILE)
print("Run folder:              ", RUN_DIR)'''),
    ("code", SET4_MODELS.split("def ensure_local")[0] + '''MODELS = {k: v for k, v in MODELS.items() if k in ("q7", "l8", "q32")}
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "7c answer gate on the fine-tuned 3B retriever", "retriever_run": FT_RUN, "ret_file": RET_FILE,
           "old_sweep": OLD_SWEEP, "fresh_run": FRESH_RUN, "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "cv_repeats": CV_REPEATS,
           "gpu": gpu, "dry_run": DRY_RUN, "code": EXPECTED_CODE, "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/gate7.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "Loading weights", "Warning", "warn")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

COMMON = ["--old_data", OLD_DATA, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR, "--models_json", f"{RUN_DIR}/models.json", "--ret_file", RET_FILE]
def n_done():
    p = f"{RUN_DIR}/step/hidden_answer_l8.jsonl"
    return sum(1 for _ in open(p)) if os.path.exists(p) else 0
print("Llama 8B answers with hidden states so far:", n_done(), "of 400")'''),
    ("md", "### 1. Llama 8B answers on the new shortlists, with the hidden states of that same pass (resumable)"),
    ("code", '''if n_done() < 400:
    if DRY_RUN:
        run_stage(["extract", "--backend", "tiny", "--max_new", 12] + COMMON, "extract.log")
    else:
        m = MODELS["l8"]
        if not os.path.exists(f"{m['path']}/config.json"):
            t = time.time()
            print("copying Llama 8B to local disk ...")
            shutil.copytree(f"{PROJECT_DIR}/models/{m['dir']}", m["path"] + ".part", dirs_exist_ok=True)
            os.rename(m["path"] + ".part", m["path"])
            TIMES["copy_l8"] = round(time.time() - t)
        run_stage(["extract", "--time_limit_min", max(5.0, minutes_left() - 5)] + COMMON, "extract.log")
ALL_DONE = n_done() >= 400
print("\\nAll 400 answers extracted." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")'''),
    ("md", "### 2. The gate, fitted inside training folds (CPU)"),
    ("code", '''if ALL_DONE:
    from IPython.display import Image, Markdown, display
    run_stage(["analyze", "--old_sweep", OLD_SWEEP, "--fresh_run", FRESH_RUN, "--answers_run", FT_RUN, "--extra_answers"] + OTHER_ANSWERS
              + ["--repeats", CV_REPEATS] + COMMON, "analyze.log")
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    display(Image(f"{RUN_DIR}/results/charts/gate_curve.png"))
    print("Excel workbook:", f"{RUN_DIR}/results/set7c_answer_gate.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "code": EXPECTED_CODE, "complete": ALL_DONE},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])

# ------------------------------------------------------------------------------ 25a / 25b: Set 7d, can a sub-1B model learn to shortlist as well as Qwen 3B?
CODE25 = CODE22
EXPECTED25 = {f: _sha(f) for f in CODE25}
SMALL25 = {
    "q06": {"id": "Qwen/Qwen3-0.6B", "dir": "Qwen3-0.6B", "params": 0.6e9, "label": "Qwen3 0.6B", "extra_pip": ""},
    "q08": {"id": "Qwen/Qwen3.5-0.8B", "dir": "Qwen3.5-0.8B", "params": 0.8e9, "label": "Qwen3.5 0.8B", "extra_pip": "flash-linear-attention"},
}


def nb25(key, letter):
    sm = SMALL25[key]
    lab = sm["label"]
    name = f"ft{key}_12k"
    ftkey = f"{key}ft12k"
    note = ("Qwen3.5 0.8B mixes ordinary attention with linear-attention (Gated DeltaNet) layers and ships with an image encoder. Only the "
            "text layers are trained and used. The notebook installs `flash-linear-attention` for its fast kernels." if key == "q08" else
            "Qwen3 0.6B is a reasoning model: its 'thinking' block is switched off, both in training and when it shortlists.")
    v = nb([
        ("md", f"""# 25{letter} · Set 7d: can {lab} learn to shortlist as well as Qwen 3B? (LoRA, A100)

Notebook 23a taught Qwen 2.5 3B, on 12,000 HotpotQA training questions, which passages answer a question: it found both gold
passages on 92.0% of the 400 test questions, and with Qwen 32B answering reached 84.2% (corrected label) at 42% less compute than
the cheap fixed setup. This notebook applies **exactly the same recipe** to **{lab}** (about {sm['params'] / 1e9:.1f} billion
parameters, roughly a fifth of the 3B model).

| | |
|---|---|
| Same as notebook 23a | The 12,000 training questions and 300 validation questions, input and target, LoRA r = 16, learning rate, batch, test questions, answerers |
| Model | `{sm['id']}`, downloaded from Hugging Face into `models/{sm['dir']}` on Drive the first time |
| Also measured | {lab} before training (same direct prompt), to show what the training adds |
| Validation checkpoints | after 2,000, 4,000, 8,000 and 12,000 questions; the best is kept |

{note}

**My estimate:** download a few minutes; training about 1–2.5 hours (the 3B model took 3.4 hours; a model this size should be
faster, but the per-question overhead does not shrink in proportion); retrieval and answers about 25 minutes. Resumable.

Notebooks 25a (Qwen3 0.6B) and 25b (Qwen3.5 0.8B) are independent: run them at the same time if you can.

Output: `runs/{SET7_NAME}/<timestamp>_<GPU>_{name}/` with `adapter/` and `results/summary.md`.

**Runtime → Change runtime type → A100 GPU.**"""),
        ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET7_NAME}"
EXP_SET5    = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
SMALL = {json.dumps(dict(SMALL25[key], key=key))}
N_TRAIN = 12000                       # training questions (the same 12,000 as notebook 23a)
N_VAL   = 300
CKPT_QUESTIONS = [2000, 4000, 8000, 12000]
TIME_LIMIT_MIN = 600                  # stops cleanly at this limit; run the notebook again to continue
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = tiny model and fake answers on CPU: tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED25, indent=1)}'''),
        ("code", GPU_CHECK),
        ("code", MOUNT),
        ("md", "### Check that Drive is serving the current code; find the earlier runs and the training questions"),
        ("code", f'''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set7code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{{PROJECT_DIR}}/code/{{f}}", f"{{CODE_DIR}}/{{f}}")
    if sha(f"{{CODE_DIR}}/{{f}}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {{stale}} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{OLD_SWEEP_EXPERIMENT}}/*")) if "_dryrun" not in d and os.path.exists(f"{{d}}/results/per_question.jsonl")
          and os.path.exists(f"{{d}}/sweep_status.json") and json.load(open(f"{{d}}/sweep_status.json")).get("complete")]
fresh = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXP_SET5}}/*_final")) if os.path.exists(f"{{d}}/step/retrieval_q7_c1_p3.jsonl")]
set6 = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXP_SET5}}/*_set6")) if os.path.exists(f"{{d}}/step/arms_q32.jsonl")]
set7 = [d for d in sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXPERIMENT}}/*_ft*")) if "_dryrun" not in d and os.path.exists(f"{{d}}/step/answers_q32.jsonl")]
assert sweeps and fresh, "Needs the finished runs of notebooks 11 and 17."
OLD_SWEEP, FRESH_RUN = sweeps[-1], fresh[-1]
CACHE_RUN = set6[-1] if set6 else ""
OLD_DATA = json.load(open(f"{{OLD_SWEEP}}/config.json"))["data_dir"]
TRAIN_DATA = f"{{PROJECT_DIR}}/data/hotpotqa_train/train12000_val300_c20_hard_all_s7" + ("_dryrun" if DRY_RUN else "")
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_{name}_dryrun" if DRY_RUN else "_{name}"
prev = sorted(glob.glob(f"{{PROJECT_DIR}}/runs/{{EXPERIMENT}}/*{{tag}}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{{PROJECT_DIR}}/runs/{{EXPERIMENT}}/{{SESSION}}_{{GPU_TAG}}{{tag}}"
CACHE_ANSWERS = [d for d in set7 if d != RUN_DIR]
SESSION_DIR = f"{{RUN_DIR}}/sessions/{{SESSION}}"
for d in (f"{{RUN_DIR}}/code", f"{{RUN_DIR}}/logs", f"{{RUN_DIR}}/step", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{{CODE_DIR}}/{{f}}", f"{{RUN_DIR}}/code/{{f}}")
TIMES = {{}}
assert DRY_RUN or os.path.exists(f"{{TRAIN_DATA}}/val.json"), f"{{TRAIN_DATA}} is missing - it is built by notebook 23a"
print("Code files: current.")
print("Training data:   ", TRAIN_DATA)
print("Answers to reuse:", [CACHE_RUN] + CACHE_ANSWERS)
print("Run folder:      ", RUN_DIR)'''),
        ("code", SET4_MODELS.split("def ensure_local")[0] + f'''MODELS = {{k: v for k, v in MODELS.items() if k in ("q7", "l8", "q32")}}
SK = SMALL["key"]
MODELS[SK] = {{"id": SMALL["id"], "dir": SMALL["dir"], "params": SMALL["params"], "nominal": SMALL["params"], "path": f"{{LOCAL}}/{{SMALL['dir']}}"}}
FT_KEY, FT_NAME, BASE_NAME = "{ftkey}", "{name}", f"base_{{SK}}"
FT_LABEL = f"{{SMALL['label']}}, fine-tuned on {{N_TRAIN:,}} questions (direct prompt)"
BASE_LABEL = f"{{SMALL['label']}}, not trained (direct prompt)"
MODELS[FT_KEY] = dict(MODELS[SK], id=SMALL["id"] + " + LoRA trained in this run", dir=FT_KEY + "_merged", path=f"{{LOCAL}}/{{FT_KEY}}_merged")
json.dump(MODELS, open(f"{{RUN_DIR}}/models.json", "w"), indent=1)
json.dump({{"experiment": EXPERIMENT, "set": f"7d retriever fine-tuning ({{SMALL['label']}}, {{N_TRAIN}} questions)", "old_sweep": OLD_SWEEP, "fresh_run": FRESH_RUN,
           "cache": [CACHE_RUN] + CACHE_ANSWERS, "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "train_data": TRAIN_DATA, "n_train": N_TRAIN, "n_val": N_VAL,
           "model": SMALL, "gpu": gpu, "dry_run": DRY_RUN, "code": EXPECTED_CODE, "started": SESSION}}, open(f"{{RUN_DIR}}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name):
    cmd = [sys.executable, "-u", f"{{CODE_DIR}}/retft.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{{RUN_DIR}}/logs/{{log_name}}", "a") as log, open(f"{{SESSION_DIR}}/{{log_name}}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ", "Warning", "warn", "Loading weights", "Writing model")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{{SESSION_DIR}}/stage_times.json", "w"), indent=1)
    print(f"--- {{log_name}}: exit {{p.returncode}}, {{time.time() - t0:.0f}}s, {{minutes_left():.0f}} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {{RUN_DIR}}/logs/{{log_name}}"

ADAPTER = f"{{RUN_DIR}}/adapter"
def trained():
    s = f"{{ADAPTER}}/train_state.json"
    return os.path.exists(s) and "chosen" in json.load(open(s))
def merged():
    return os.path.exists(f"{{MODELS[FT_KEY]['path']}}/config.json")
print("trained:", trained())'''),
        ("md", f"### 1. Download {lab} to Drive (first time only), copy the models, install peft"),
        ("code", '''import importlib.util, threading
SRC = f"{PROJECT_DIR}/models/{SMALL['dir']}"
if not DRY_RUN and not (os.path.exists(f"{SRC}/config.json") and glob.glob(f"{SRC}/*.safetensors")):
    from huggingface_hub import snapshot_download
    t = time.time()
    tmp = f"/content/_dl/{SMALL['dir']}"
    print(f"downloading {SMALL['id']} ...")
    snapshot_download(repo_id=SMALL["id"], local_dir=tmp, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py", "tokenizer*"])
    shutil.copytree(tmp, SRC, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
    shutil.rmtree(tmp, ignore_errors=True)
    TIMES["download"] = round(time.time() - t)
    print(f"saved to {SRC} in {time.time() - t:.0f}s")
COPY = {"err": None, "done": set(), "seconds": {}}
def _copy_all(keys):
    try:
        for key in keys:
            m = MODELS[key]
            src = f"{PROJECT_DIR}/models/{m['dir']}"
            if not os.path.exists(f"{m['path']}/config.json"):
                assert os.path.exists(f"{src}/config.json"), f"{src} missing."
                t = time.time()
                shutil.rmtree(m["path"] + ".part", ignore_errors=True)
                shutil.copytree(src, m["path"] + ".part")
                os.rename(m["path"] + ".part", m["path"])
                COPY["seconds"][key] = round(time.time() - t)
            COPY["done"].add(key)
    except Exception as e:
        COPY["err"] = e

def wait_local(key):
    if DRY_RUN:
        return
    t = time.time()
    while key not in COPY["done"]:
        if COPY["err"]:
            raise COPY["err"]
        time.sleep(5)
    TIMES[f"copy_{key}"] = COPY["seconds"].get(key, 0)

if not DRY_RUN:
    os.makedirs(LOCAL, exist_ok=True)
    threading.Thread(target=_copy_all, args=([SK, "l8", "q32"],), daemon=True).start()
if importlib.util.find_spec("peft") is None:
    print("installing peft ...")
    !pip install -q peft
if SMALL["extra_pip"] and not DRY_RUN and importlib.util.find_spec("fla") is None:
    EXTRA = SMALL["extra_pip"]
    print("installing", EXTRA, "...")
    !pip install -q $EXTRA'''),
        ("md", f"### 2. Train the LoRA adapter on {lab} (resumable)"),
        ("code", '''TRAIN = ["train", "--data_dir", TRAIN_DATA, "--out_dir", ADAPTER, "--merged_dir", MODELS[FT_KEY]["path"], "--n_train", N_TRAIN, "--n_val", N_VAL,
         "--ckpt_questions"] + CKPT_QUESTIONS + ["--save_every_questions", 1000, "--micro_batch", 4]
if DRY_RUN:
    if not os.path.exists(f"{TRAIN_DATA}/val.json"):
        run_stage(["build_train", "--hotpot_train", f"{PROJECT_DIR}/AgentTTS/data/HotpotQA/hotpotqa-dev.json", "--out_dir", TRAIN_DATA,
                   "--n_train", 80, "--n_val", 12, "--pool", 1500], "build_train.log")
    run_stage(TRAIN + ["--backend", "tiny", "--n_train", 80, "--n_val", 6, "--grad_accum", 4, "--micro_batch", 2, "--max_len", 9000,
                       "--ckpt_questions", 20, 40, 80, "--save_every_questions", 16], "train.log")
elif not (trained() and merged()):
    wait_local(SK)
    run_stage(TRAIN + ["--base", MODELS[SK]["path"], "--time_limit_min", max(5.0, minutes_left() - 45)], "train.log")
TRAIN_DONE = trained() and (merged() or DRY_RUN)
print("\\nTraining complete." if TRAIN_DONE else "\\nTraining NOT COMPLETE YET - run this notebook again to continue from the last resume point.")'''),
        ("md", "### 3. Retrieval on the 400 test questions (trained and untrained), then answers (vLLM)"),
        ("code", '''BACKEND = "mock" if DRY_RUN else "vllm"
SETS = ["--old_sweep", OLD_SWEEP, "--old_data", OLD_DATA, "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR,
        "--models_json", f"{RUN_DIR}/models.json", "--backend", BACKEND, "--variants", BASE_NAME, FT_NAME,
        "--add_variant", f"{FT_NAME}:{FT_KEY}:{FT_LABEL}", f"{BASE_NAME}:{SK}:{BASE_LABEL}"]
ALL_DONE = False
if TRAIN_DONE:
    if not DRY_RUN and importlib.util.find_spec("vllm") is None:
        t = time.time()
        !pip install -q vllm
        !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
        chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                             capture_output=True, text=True)
        print(chk.stdout or chk.stderr[-2000:])
        assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
        TIMES["install_vllm"] = round(time.time() - t)
    wait_local(SK)
    run_stage(["retrieve"] + SETS, "retrieve.log")
    for key in ("l8", "q32"):
        wait_local(key)
    run_stage(["answer", "--cache_run", CACHE_RUN, "--cache_answers"] + CACHE_ANSWERS + SETS, "answer.log")
    ALL_DONE = True'''),
        ("md", "### 4. Results"),
        ("code", '''if ALL_DONE:
    from IPython.display import Markdown, display
    run_stage(["evaluate", "--adapter_dir", ADAPTER] + SETS, "evaluate.log")
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    print("Excel workbook:", f"{RUN_DIR}/results/set7_retriever_finetune.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "code": EXPECTED_CODE, "complete": ALL_DONE},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
    ])
    return v


nb25a = nb25("q06", "a")
nb25b = nb25("q08", "b")

# ------------------------------------------------------------------------------ 26: Set 8, Qwen2.5 0.5B trained longer + the 0.5B -> 0.6B -> 0.8B cascade, one notebook
CODE26 = CODE22 + ["casc8.py"]
EXPECTED26 = {f: _sha(f) for f in CODE26}
PREV26 = {"r06": ["_ftq06_12k", "q06", "Qwen3-0.6B", 0.6e9, "Qwen3 0.6B"], "r08": ["_ftq08_12k", "q08", "Qwen3.5-0.8B", 0.8e9, "Qwen3.5 0.8B"],
          "r3": ["_ft3_12k", "q3", "Qwen2.5-3B-Instruct", 3.09e9, "Qwen2.5 3B"]}
nb26 = nb([
    ("md", f"""# 26 · Set 8: train Qwen2.5 0.5B longer, then cascade 0.5B → 0.6B → 0.8B and Llama → Qwen with votes (A100)

**One notebook, resumable.** It trains the smallest model, then runs the whole cascade on the 400 test questions.

| Stage | What happens |
|---|---|
| 1. Train | **Qwen2.5 0.5B** (`Qwen/Qwen2.5-0.5B-Instruct`) learns to shortlist, same recipe as notebooks 23a/25a/25b but **trained longer: 2 passes over the 12,000 training questions** (24,000 examples seen). Validation checkpoints after 4k, 8k, 12k, 16k, 20k, 24k; the best on the 300 validation questions is kept |
| 2. Merge | The adapters already trained in 25a (Qwen3 0.6B), 25b (Qwen3.5 0.8B) and 23a (Qwen 3B, kept as the reference) are merged again from Drive |
| 3. Retrieve | All four shortlist the 400 test questions, now **recording their token probability** (their confidence) |
| 4. Answer | Llama 8B and Qwen 32B answer on every shortlist a cascade can end on (with token probabilities); Qwen 32B also gives 4 sampled answers (temperature 0.7) where a vote may be needed |
| 5. Analyse (CPU) | The policies below, thresholds fitted inside training folds |

**The cascade being tested:**

| Step | Rule |
|---|---|
| Retrieve | 0.5B shortlists. If it is **unsure** (low token probability) → 0.6B also shortlists. They **agree** → accept. They differ → 0.8B breaks the tie; no majority → the union of the three |
| Answer | Llama 8B answers. If it is **unsure** → Qwen 32B answers too. They **agree** → accept. They differ → 4 sampled Qwen answers join a **majority vote** |
| Fitted in training folds | The two "unsure" thresholds: the cheapest setting within 0.5 point of the best training accuracy (5-fold cross-validation × 5) |
| Compared with | The cheap fixed setup (AgentTTS's pick), each fine-tuned retriever alone with Qwen 32B (incl. 3B: 84.2%), each step always run, a random gate escalating the same share, Llama-or-Qwen oracle |
| Compute | Every model actually run is charged (each sample with its full prompt) |

**My estimate:** training 2.5–3 hours (25a's 0.6B took 87 min for one pass); merging about 5 min; retrieval about 10 min;
answers and samples about 50–70 min; analysis a few minutes. **About 4–4.5 hours in total.** If the session stops, run the
notebook again: every stage continues from where it stopped.

Output: `runs/{SET7_NAME}/<timestamp>_<GPU>_casc05/` with `adapter/` and `results/summary.md`, `set8_cascades.xlsx`, `charts/accuracy_vs_compute.png`.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET7_NAME}"
EXP_SET5    = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
SMALL = {{"key": "q05", "id": "Qwen/Qwen2.5-0.5B-Instruct", "dir": "Qwen2.5-0.5B-Instruct", "params": 0.49e9, "label": "Qwen2.5 0.5B"}}
PREV = {json.dumps(PREV26)}       # retriever -> [run tag of its training notebook, base key, base folder, parameters, label]
N_TRAIN = 12000                       # the same 12,000 training questions as notebooks 23a / 25a / 25b
N_VAL   = 300
EPOCHS  = 2                           # trained longer: two passes
CKPT_QUESTIONS = [4000, 8000, 12000, 16000, 20000, 24000]
N_SAMPLES = 4                         # Qwen 32B sampled answers for the votes
CV_REPEATS = 5
TIME_LIMIT_MIN = 600                  # stops cleanly at this limit; run the notebook again to continue
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = tiny model and fake answers on CPU: tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED26, indent=1)}'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("md", "### Check that Drive is serving the current code; find the earlier runs"),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set8code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")
          and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")]
fresh = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXP_SET5}/*_final")) if os.path.exists(f"{d}/step/retrieval_q7_c1_p3.jsonl")]
set6 = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXP_SET5}/*_set6")) if os.path.exists(f"{d}/step/arms_q32.jsonl")]
set7 = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_ft*")) if "_dryrun" not in d and os.path.exists(f"{d}/step/answers_q32.jsonl")]
assert sweeps and fresh, "Needs the finished runs of notebooks 11 and 17."
OLD_SWEEP, FRESH_RUN = sweeps[-1], fresh[-1]
CACHE_RUN = set6[-1] if set6 else ""
OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
TRAIN_DATA = f"{PROJECT_DIR}/data/hotpotqa_train/train12000_val300_c20_hard_all_s7" + ("_dryrun" if DRY_RUN else "")
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_casc05_dryrun" if DRY_RUN else "_casc05"
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
ADAPTER = f"{RUN_DIR}/adapter"
ADAPTERS = {"r05": ADAPTER}
for r, (rtag, *_ ) in PREV.items():
    found = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{rtag}" + ("_dryrun" if DRY_RUN else "")))
             if os.path.exists(f"{d}/adapter/train_state.json") and "chosen" in json.load(open(f"{d}/adapter/train_state.json"))]
    if found:
        ADAPTERS[r] = f"{found[-1]}/adapter"
    else:
        assert DRY_RUN, f"No finished training run *{rtag} with an adapter - needs notebooks 23a, 25a and 25b."
        ADAPTERS[r] = ADAPTER            # dry run: the tiny adapter stands in
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", f"{RUN_DIR}/step", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
TIMES = {}
assert DRY_RUN or os.path.exists(f"{TRAIN_DATA}/val.json"), f"{TRAIN_DATA} is missing - it is built by notebook 23a"
print("Code files: current.")
print("Adapters:        ", json.dumps(ADAPTERS, indent=1))
print("Answers to reuse:", [CACHE_RUN] + set7)
print("Run folder:      ", RUN_DIR)'''),
    ("code", SET4_MODELS.split("def ensure_local")[0] + '''MODELS = {k: v for k, v in MODELS.items() if k in ("q3", "q7", "l8", "q32")}
SK = SMALL["key"]
MODELS[SK] = {"id": SMALL["id"], "dir": SMALL["dir"], "params": SMALL["params"], "nominal": SMALL["params"], "path": f"{LOCAL}/{SMALL['dir']}"}
BASES = {"r05": (SK, SMALL["dir"], SMALL["params"], SMALL["label"])} | {r: (v[1], v[2], v[3], v[4]) for r, v in PREV.items()}
FTKEY = {r: f"{r}ft" for r in BASES}
for r, (bk, bdir, params, lab) in BASES.items():
    MODELS[FTKEY[r]] = {"id": f"{lab} + LoRA ({ADAPTERS[r]})", "dir": FTKEY[r] + "_merged", "params": params, "nominal": params, "path": f"{LOCAL}/{FTKEY[r]}_merged"}
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "8 cascades: Qwen2.5 0.5B (trained longer) -> 0.6B -> 0.8B; Llama -> Qwen with votes", "old_sweep": OLD_SWEEP,
           "fresh_run": FRESH_RUN, "adapters": ADAPTERS, "cache": [CACHE_RUN] + set7, "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "train_data": TRAIN_DATA,
           "n_train": N_TRAIN, "epochs": EPOCHS, "n_samples": N_SAMPLES, "cv_repeats": CV_REPEATS, "gpu": gpu, "dry_run": DRY_RUN, "code": EXPECTED_CODE,
           "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name, script="retft.py"):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/{script}"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ", "Warning", "warn", "Loading weights", "Writing model")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

def trained():
    s = f"{ADAPTER}/train_state.json"
    return os.path.exists(s) and "chosen" in json.load(open(s))
print("0.5B trained:", trained())'''),
    ("md", "### 1. Download Qwen2.5 0.5B to Drive (first time only), copy the models, install peft"),
    ("code", '''import importlib.util, threading
SRC = f"{PROJECT_DIR}/models/{SMALL['dir']}"
if not DRY_RUN and not (os.path.exists(f"{SRC}/config.json") and glob.glob(f"{SRC}/*.safetensors")):
    from huggingface_hub import snapshot_download
    t = time.time()
    tmp = f"/content/_dl/{SMALL['dir']}"
    print(f"downloading {SMALL['id']} ...")
    snapshot_download(repo_id=SMALL["id"], local_dir=tmp, allow_patterns=["*.json", "*.safetensors", "*.jinja", "*.txt", "*.model", "*.py", "tokenizer*"])
    shutil.copytree(tmp, SRC, dirs_exist_ok=True, ignore=shutil.ignore_patterns(".cache"))
    shutil.rmtree(tmp, ignore_errors=True)
    TIMES["download"] = round(time.time() - t)
    print(f"saved to {SRC} in {time.time() - t:.0f}s")
COPY = {"err": None, "done": set(), "seconds": {}}
def _copy_all(keys):
    try:
        for key in keys:
            m = MODELS[key]
            src = f"{PROJECT_DIR}/models/{m['dir']}"
            if not os.path.exists(f"{m['path']}/config.json"):
                assert os.path.exists(f"{src}/config.json"), f"{src} missing."
                t = time.time()
                shutil.rmtree(m["path"] + ".part", ignore_errors=True)
                shutil.copytree(src, m["path"] + ".part")
                os.rename(m["path"] + ".part", m["path"])
                COPY["seconds"][key] = round(time.time() - t)
            COPY["done"].add(key)
    except Exception as e:
        COPY["err"] = e

def wait_local(key):
    if DRY_RUN:
        return
    t = time.time()
    while key not in COPY["done"]:
        if COPY["err"]:
            raise COPY["err"]
        time.sleep(5)
    TIMES[f"copy_{key}"] = COPY["seconds"].get(key, 0)

if not DRY_RUN:
    os.makedirs(LOCAL, exist_ok=True)
    threading.Thread(target=_copy_all, args=([SK, "l8", "q32"],), daemon=True).start()
if importlib.util.find_spec("peft") is None:
    print("installing peft ...")
    !pip install -q peft
if not DRY_RUN and importlib.util.find_spec("fla") is None:
    print("installing flash-linear-attention (for Qwen3.5 0.8B) ...")
    !pip install -q flash-linear-attention'''),
    ("md", "### 2. Train Qwen2.5 0.5B for two passes over the training questions (resumable)"),
    ("code", '''TRAIN = ["train", "--data_dir", TRAIN_DATA, "--out_dir", ADAPTER, "--merged_dir", MODELS[FTKEY["r05"]]["path"], "--n_train", N_TRAIN, "--n_val", N_VAL,
         "--epochs", EPOCHS, "--ckpt_questions"] + CKPT_QUESTIONS + ["--save_every_questions", 1000, "--micro_batch", 4]
if DRY_RUN:
    if not os.path.exists(f"{TRAIN_DATA}/val.json"):
        run_stage(["build_train", "--hotpot_train", f"{PROJECT_DIR}/AgentTTS/data/HotpotQA/hotpotqa-dev.json", "--out_dir", TRAIN_DATA,
                   "--n_train", 80, "--n_val", 12, "--pool", 1500], "build_train.log")
    if not trained():
        run_stage(TRAIN + ["--backend", "tiny", "--n_train", 40, "--n_val", 6, "--grad_accum", 4, "--micro_batch", 2, "--max_len", 9000,
                           "--ckpt_questions", 20, 40, 80, "--save_every_questions", 16], "train.log")
elif not trained():
    wait_local(SK)
    run_stage(TRAIN + ["--base", MODELS[SK]["path"], "--time_limit_min", max(5.0, minutes_left() - 120)], "train.log")
TRAIN_DONE = trained()
if TRAIN_DONE:
    print("Training curve:")
    for row in json.load(open(f"{ADAPTER}/train_state.json"))["curve"]:
        print(f"  {row['training questions seen']:>6} seen: both gold found {100 * row['both gold passages found']:.1f}%, exact {100 * row['exactly the gold passages']:.1f}%")
print("\\nTraining complete." if TRAIN_DONE else "\\nTraining NOT COMPLETE YET - run this notebook again to continue from the last resume point.")'''),
    ("md", "### 3. Merge the four fine-tuned retrievers (0.5B from this run; 0.6B, 0.8B, 3B from notebooks 25a, 25b, 23a)"),
    ("code", '''if TRAIN_DONE:
    for r, (bk, bdir, params, lab) in BASES.items():
        run_stage(["merge", "--adapter_dir", ADAPTERS[r], "--merged_dir", MODELS[FTKEY[r]]["path"], "--base", f"{PROJECT_DIR}/models/{bdir}"]
                  + (["--backend", "tiny"] if DRY_RUN else []), "merge.log")'''),
    ("md", "### 4. Retrieval with confidence, then answers and samples (vLLM)"),
    ("code", '''BACKEND = "mock" if DRY_RUN else "vllm"
BASE_ARGS = ["--old_sweep", OLD_SWEEP, "--old_data", OLD_DATA, "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR,
             "--models_json", f"{RUN_DIR}/models.json"]
SETS = BASE_ARGS + ["--backend", BACKEND]
STEP = f"{RUN_DIR}/step"
ALL_DONE = False
if TRAIN_DONE:
    if not DRY_RUN and importlib.util.find_spec("vllm") is None:
        t = time.time()
        !pip install -q vllm
        !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
        chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                             capture_output=True, text=True)
        print(chk.stdout or chk.stderr[-2000:])
        assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
        TIMES["install_vllm"] = round(time.time() - t)
    run_stage(["retrieve"] + SETS + ["--variants"] + list(BASES) + ["--add_variant"] + [f"{r}:{FTKEY[r]}:{BASES[r][3]}, fine-tuned" for r in BASES], "retrieve.log")
    run_stage(["shortlists"] + BASE_ARGS, "shortlists.log", script="casc8.py")
    for key in ("l8", "q32"):
        wait_local(key)
    # every shortlist a cascade can end on: fresh answers with token probabilities (answers from notebook 25c-style runs with them are reused)
    run_stage(["answer", "--only_extra", "--require_logprob", "--extra_shortlists", f"{STEP}/cascade_shortlists.jsonl", "--tag", "notebook 26",
               "--cache_answers"] + set7 + SETS, "answer.log")
    # the cheap fixed setup's shortlists: Qwen 32B answers, mostly reused from notebooks 21-25
    run_stage(["answer", "--only_extra", "--models", "q32", "--extra_shortlists", f"{STEP}/ref_shortlists.jsonl", "--tag", "notebook 26",
               "--cache_run", CACHE_RUN, "--cache_answers"] + set7 + SETS, "answer_ref.log")
    run_stage(["sample", "--model", "q32", "--n", N_SAMPLES, "--temperature", 0.7, "--extra_shortlists", f"{STEP}/sample_shortlists.jsonl"] + SETS, "sample.log")
    ALL_DONE = True'''),
    ("md", "### 5. The cascades, thresholds fitted inside training folds (CPU)"),
    ("code", '''if ALL_DONE:
    from IPython.display import Image, Markdown, display
    run_stage(["analyze", "--repeats", CV_REPEATS, "--ref_answers"] + set7 + BASE_ARGS, "analyze.log", script="casc8.py")
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    display(Image(f"{RUN_DIR}/results/charts/accuracy_vs_compute.png"))
    print("Excel workbook:", f"{RUN_DIR}/results/set8_cascades.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "code": EXPECTED_CODE, "complete": ALL_DONE},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])

# ------------------------------------------------------------------------------ 27: Set 9, the 0.5B learns the route; ensembles + calibrated escalation + Sure/Unsure
CODE27 = CODE26 + ["route9.py"]
EXPECTED27 = {f: _sha(f) for f in CODE27}
nb27 = nb([
    ("md", f"""# 27 · Set 9: the 0.5B learns where each question should go; ensembles, calibrated escalation, Sure / Unsure (A100)

Needs notebook 26 to be finished (it trains the 0.5B retriever and makes most of the test answers this notebook reuses).

**What the 0.5B learns.** After its shortlist it writes two more lines:

```
Selected references: [3] [11]
Shortlist with: 0.5B | 0.6B | 0.8B | 3B
Answer with: Llama 8B | Qwen 32B | Qwen 32B, likely wrong
```

| | |
|---|---|
| Labelling questions | About 3,300 hard HotpotQA training questions that **no retriever was trained on** (not among notebook 23a's 12,000 + 300) |
| Labels | *Shortlist with* = the cheapest fine-tuned retriever whose shortlist holds both gold passages. *Answer with* = Llama if Llama is right on the 0.5B shortlist, else Qwen if Qwen is, else "Qwen 32B, likely wrong" |
| Teaching | About 2,300 of them: a short LoRA on the 0.5B from notebook 26 (2 passes) |
| Calibration | The other 1,000: the scores of both stages and the threshold pair chosen there |
| Test | The 400 test questions, used once |

**Both stages work the same way:**

| Step | Retrieval | Answer |
|---|---|---|
| 1. Learned start | The retriever the 0.5B named | The answerer the 0.5B named |
| 2. Ensemble | With the 0.5B's own shortlist: agree → keep; majority; else union | Votes: Llama start → Llama, + Qwen, + 4 Qwen samples; Qwen start → Qwen, + 2 samples, + 2 samples and Llama. Majority by agreement |
| 3. Score | Calibrated chance that both gold passages are in the shortlist | Calibrated chance that the answer is right |
| 4. Below the threshold | Add the next larger retriever, score again | Next step of votes, score again |
| Flag | Retrieval **Sure / Unsure** | Answer **Sure / Unsure** (also when Qwen 32B is unsure) |

Thresholds: **0.90 / 0.90 stated beforehand** (so "Sure" should mean about 90% right), and the pair chosen on the calibration
questions. Compared with: the cheap fixed setup, 3B → Qwen 32B (84.2%), the same cascade without the learned start, the learned
start without ensembles, and everything always run. Output includes a per-question table: route, retrievers run, scores, flags,
answer, compute.

**My estimate:** labelling questions built 10–15 min; retrieval about 20 min; Llama and Qwen answers on the labelling questions
about 80–100 min; teaching the route about 20–30 min; calibration and test answers and samples about 60–80 min; analysis a
few minutes. **About 4–4.5 hours.** Resumable: if the session stops, run the notebook again.

Output: `runs/{SET7_NAME}/<timestamp>_<GPU>_route05/results/` (`summary.md`, `set9_learned_route.xlsx`, `charts/accuracy_vs_compute.png`).

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET7_NAME}"
EXP_SET5    = "{SET5_NAME}"
OLD_SWEEP_EXPERIMENT = "{SET4_NAME}"
FRESH_DATA  = f"{{PROJECT_DIR}}/data/hotpotqa_q/fresh200_c20_s43"
N_CALIB = 1000                        # labelling questions kept for calibration (the rest teach the route lines)
ROUTE_EPOCHS = 2
TARGET = 0.90                         # threshold stated beforehand: a 'Sure' answer should be right about 90% of the time
TIME_LIMIT_MIN = 600                  # stops cleanly at this limit; run the notebook again to continue
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = tiny model and fake answers on CPU: tests the whole notebook in a few minutes

EXPECTED_CODE = {json.dumps(EXPECTED27, indent=1)}'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("md", "### Check that Drive is serving the current code; find notebook 26's run"),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set9code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
sweeps = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{OLD_SWEEP_EXPERIMENT}/*")) if "_dryrun" not in d and os.path.exists(f"{d}/results/per_question.jsonl")
          and os.path.exists(f"{d}/sweep_status.json") and json.load(open(f"{d}/sweep_status.json")).get("complete")]
fresh = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXP_SET5}/*_final")) if os.path.exists(f"{d}/step/retrieval_q7_c1_p3.jsonl")]
casc = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_casc05" + ("_dryrun" if DRY_RUN else "")))
        if os.path.exists(f"{d}/results/summary.md") and os.path.exists(f"{d}/adapter/train_state.json")]
set7 = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_ft*")) if "_dryrun" not in d and os.path.exists(f"{d}/step/answers_q32.jsonl")]
assert sweeps and fresh, "Needs the finished runs of notebooks 11 and 17."
assert casc, "Needs the finished run of notebook 26 (with results/summary.md)."
OLD_SWEEP, FRESH_RUN, CASC_RUN = sweeps[-1], fresh[-1], casc[-1]
OLD_DATA = json.load(open(f"{OLD_SWEEP}/config.json"))["data_dir"]
CASC_CFG = json.load(open(f"{CASC_RUN}/config.json"))
ADAPTERS = CASC_CFG["adapters"]                      # r05 (notebook 26), r06 (25a), r08 (25b), r3 (23a)
LABEL_DIR = f"{PROJECT_DIR}/data/hotpotqa_train/label_rest_c20_hard_s7" + ("_dryrun" if DRY_RUN else "")
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_route05_dryrun" if DRY_RUN else "_route05"
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", f"{RUN_DIR}/step", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
TIMES = {}
print("Code files: current.")
print("Notebook 26 run: ", CASC_RUN)
print("Adapters:        ", json.dumps(ADAPTERS, indent=1))
print("Run folder:      ", RUN_DIR)'''),
    ("code", SET4_MODELS.split("def ensure_local")[0] + '''MODELS = {k: v for k, v in MODELS.items() if k in ("q3", "q7", "l8", "q32")}
BASES = {"r05": ("Qwen2.5-0.5B-Instruct", 0.49e9), "r06": ("Qwen3-0.6B", 0.6e9), "r08": ("Qwen3.5-0.8B", 0.8e9), "r3": ("Qwen2.5-3B-Instruct", 3.09e9)}
FTKEY = {r: f"{r}ft" for r in BASES}
for r, (bdir, params) in BASES.items():
    MODELS[FTKEY[r]] = {"id": f"{bdir} + LoRA ({ADAPTERS[r]})", "dir": FTKEY[r] + "_merged", "params": params, "nominal": params, "path": f"{LOCAL}/{FTKEY[r]}_merged"}
MODELS["r05route"] = {"id": "Qwen2.5 0.5B retriever + route LoRA trained in this run", "dir": "r05route_merged", "params": 0.49e9, "nominal": 0.49e9,
                      "path": f"{LOCAL}/r05route_merged"}
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "9 learned route + ensembles + calibrated escalation + Sure/Unsure", "casc_run": CASC_RUN, "adapters": ADAPTERS,
           "old_sweep": OLD_SWEEP, "fresh_run": FRESH_RUN, "old_data": OLD_DATA, "fresh_data": FRESH_DATA, "label_dir": LABEL_DIR, "n_calib": N_CALIB,
           "route_epochs": ROUTE_EPOCHS, "target": TARGET, "gpu": gpu, "dry_run": DRY_RUN, "code": EXPECTED_CODE, "started": SESSION},
          open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return max(0.0, TIME_LIMIT_MIN - (time.time() - T_START) / 60)

def run_stage(args, log_name, script="route9.py"):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/{script}"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ", "Warning", "warn", "Loading weights", "Writing model")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s, {minutes_left():.0f} min left of the limit")
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

BACKEND = "mock" if DRY_RUN else "vllm"
COMMON = ["--old_sweep", OLD_SWEEP, "--old_data", OLD_DATA, "--fresh_run", FRESH_RUN, "--fresh_data", FRESH_DATA, "--run_dir", RUN_DIR,
          "--models_json", f"{RUN_DIR}/models.json", "--label_dir", LABEL_DIR]
STEP = f"{RUN_DIR}/step"
ROUTE_ADAPTER = f"{RUN_DIR}/route_adapter"
def route_trained():
    s = f"{ROUTE_ADAPTER}/train_state.json"
    return os.path.exists(s) and "chosen" in json.load(open(s))'''),
    ("md", "### 1. Copy the answerers to local disk (background), install peft"),
    ("code", '''import importlib.util, threading
COPY = {"err": None, "done": set(), "seconds": {}}
def _copy_all(keys):
    try:
        for key in keys:
            m = MODELS[key]
            src = f"{PROJECT_DIR}/models/{m['dir']}"
            if not os.path.exists(f"{m['path']}/config.json"):
                assert os.path.exists(f"{src}/config.json"), f"{src} missing."
                t = time.time()
                shutil.rmtree(m["path"] + ".part", ignore_errors=True)
                shutil.copytree(src, m["path"] + ".part")
                os.rename(m["path"] + ".part", m["path"])
                COPY["seconds"][key] = round(time.time() - t)
            COPY["done"].add(key)
    except Exception as e:
        COPY["err"] = e

def wait_local(key):
    if DRY_RUN:
        return
    while key not in COPY["done"]:
        if COPY["err"]:
            raise COPY["err"]
        time.sleep(5)
    TIMES[f"copy_{key}"] = COPY["seconds"].get(key, 0)

if not DRY_RUN:
    os.makedirs(LOCAL, exist_ok=True)
    threading.Thread(target=_copy_all, args=(["l8", "q32"],), daemon=True).start()
if importlib.util.find_spec("peft") is None:
    print("installing peft ...")
    !pip install -q peft
if not DRY_RUN and importlib.util.find_spec("fla") is None:
    print("installing flash-linear-attention (for Qwen3.5 0.8B) ...")
    !pip install -q flash-linear-attention'''),
    ("md", "### 2. The labelling questions (HotpotQA training questions no retriever was trained on; first time only)"),
    ("code", '''if not os.path.exists(f"{LABEL_DIR}/calib.json"):
    if DRY_RUN:
        run_stage(["build_label", "--hotpot_train", f"{PROJECT_DIR}/AgentTTS/data/HotpotQA/hotpotqa-dev.json", "--skip_first", 7000, "--n_max", 150,
                   "--n_calib", 50] + COMMON, "build_label.log")
    else:
        run_stage(["build_label", "--n_calib", N_CALIB] + COMMON, "build_label.log")
cfg = json.load(open(f"{LABEL_DIR}/data_config.json"))
print(f"route-teaching questions: {cfg['route_train']}, calibration questions: {cfg['calib']}")'''),
    ("md", "### 3. Merge the four fine-tuned retrievers"),
    ("code", '''for r, (bdir, params) in BASES.items():
    run_stage(["merge", "--adapter_dir", ADAPTERS[r], "--merged_dir", MODELS[FTKEY[r]]["path"], "--base", f"{PROJECT_DIR}/models/{bdir}"]
              + (["--backend", "tiny"] if DRY_RUN else []), "merge.log", script="retft.py")'''),
    ("md", "### 4. Retrieval: all four retrievers on the labelling questions; 0.6B, 0.8B and 3B on the test questions (vLLM)"),
    ("code", '''if not DRY_RUN and importlib.util.find_spec("vllm") is None:
    t = time.time()
    !pip install -q vllm
    !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
    chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                         capture_output=True, text=True)
    print(chk.stdout or chk.stderr[-2000:])
    assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
    TIMES["install_vllm"] = round(time.time() - t)
run_stage(["retrieve", "--set", "label", "--backend", BACKEND, "--retrievers"] + [f"{r}:{FTKEY[r]}" for r in BASES] + COMMON, "retrieve.log")
run_stage(["retrieve", "--set", "test", "--backend", BACKEND, "--retrievers"] + [f"{r}:{FTKEY[r]}" for r in ("r06", "r08", "r3")] + COMMON, "retrieve.log")'''),
    ("md", "### 5. Labels: Llama 8B and Qwen 32B answer the route-teaching questions on the 0.5B shortlists"),
    ("code", '''run_stage(["reach", "--sets", "route_train"] + COMMON, "reach.log")
for key in ("l8", "q32"):
    wait_local(key)
run_stage(["answer", "--backend", BACKEND, "--shortlists", f"{STEP}/route_train_shortlists.jsonl"] + COMMON, "answer_labels.log")
run_stage(["route_data"] + COMMON, "route_data.log")'''),
    ("md", "### 6. Teach the 0.5B the two route lines (LoRA on notebook 26's merged 0.5B; resumable)"),
    ("code", '''if not route_trained() or not os.path.exists(f"{MODELS['r05route']['path']}/config.json"):
    run_stage(["train_route", "--out_dir", ROUTE_ADAPTER, "--merged_dir", MODELS["r05route"]["path"], "--base", MODELS[FTKEY["r05"]]["path"],
               "--epochs", ROUTE_EPOCHS, "--backend", "tiny" if DRY_RUN else "hf", "--time_limit_min", max(5.0, minutes_left() - 100)] + COMMON, "train_route.log")
ROUTE_DONE = route_trained()
print("Route lines taught." if ROUTE_DONE else "NOT COMPLETE YET - run this notebook again to continue.")'''),
    ("md", "### 7. The route model reads the calibration and test questions (its shortlist, its choices, its probabilities)"),
    ("code", '''if ROUTE_DONE:
    for st in ("calib", "test"):
        run_stage(["route", "--set", st, "--backend", "mock" if DRY_RUN else "hf"] + COMMON, "route.log")'''),
    ("md", "### 8. Answers and samples on every shortlist either ladder can reach (most test answers reused from notebook 26)"),
    ("code", '''ALL_DONE = False
if ROUTE_DONE:
    run_stage(["reach", "--sets", "calib", "test"] + COMMON, "reach.log")
    REACH = [f"{STEP}/calib_reach.jsonl", f"{STEP}/test_reach.jsonl"]
    run_stage(["answer", "--backend", BACKEND, "--require_logprob", "--shortlists"] + REACH + ["--cache_dirs", CASC_RUN] + COMMON, "answer.log")
    run_stage(["answer", "--backend", BACKEND, "--models", "q32", "--shortlists", f"{STEP}/test_ref_shortlists.jsonl", "--cache_dirs", CASC_RUN] + set7 + COMMON,
              "answer_ref.log")
    run_stage(["sample", "--backend", BACKEND, "--n", 4, "--shortlists"] + REACH + ["--cache_dirs", CASC_RUN] + COMMON, "sample.log")
    ALL_DONE = True'''),
    ("md", "### 9. Calibrate on the 1,000 calibration questions, then the test questions (CPU)"),
    ("code", '''if ALL_DONE:
    from IPython.display import Image, Markdown, display
    run_stage(["analyze", "--target", TARGET] + COMMON, "analyze.log")
    display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
    display(Image(f"{RUN_DIR}/results/charts/accuracy_vs_compute.png"))
    print("Excel workbook (incl. the per-question table with Sure/Unsure):", f"{RUN_DIR}/results/set9_learned_route.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "code": EXPECTED_CODE, "complete": ALL_DONE},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE." if ALL_DONE else "\\nNOT COMPLETE YET - run this notebook again to continue.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])

# ------------------------------------------------------------------------------ 28: Set 10, a self-learnt rulebook (soft rules, hard guardrails, micro-experiments)
CODE28 = CODE27 + ["rules10.py"]
EXPECTED28 = {f: _sha(f) for f in CODE28}
nb28 = nb([
    ("md", f"""# 28 · Set 10: a self-learnt rulebook for model switching (A100 for the planner)

Needs notebook 27 to be finished: this notebook uses only outputs it saved (every retriever's shortlist, Llama 8B and Qwen 32B
answers, Qwen samples, on 1,000 dev questions and the 400 test questions). No training. The only GPU work is the planner.

| Part | What it is |
|---|---|
| Soft rules (start) | R1: 0.5B unsure (lowest 20%) → escalate retrieval to the 0.8B, skipping the 0.6B, and ensemble. R2: still split and unsure → escalate again. A1: answer not yes/no and not fully found in the passages → Qwen 32B answers; agree → accept, else Qwen samples vote. A2–A4: keep. |
| Hard rules (locked) | H1: 0.5B in its lowest 10% → always escalate. H2: empty / "cannot be determined" answer → always escalate. H3: at most 60% reach Qwen 32B. H4: dev accuracy ≥ best fixed pipeline − 1 point (below it, the loop is in repair mode). H5: "Sure" needs a score ≥ 0.85. H6: ≤ 8 soft rules, each firing on ≥ 20 dev questions. H7: a threshold moves ≤ 0.05 per round. H8: an accuracy gain may cost ≤ 10% compute per point. |
| Micro-experiments (dev) | 4 rounds. Code tries fine threshold steps; the planner (Qwen 32B) reads evidence tables from dev half A and proposes up to 3 changes. A change is kept only if it passes on both halves, beats random escalation of the same size, and breaks no hard rule. |
| Test | The final rulebook, frozen, applied by code once to the 400 test questions; which rule decided is logged per question, with a Sure / Unsure flag |

**My estimate:** copying Qwen 32B and installing vLLM about 10 minutes; planner and micro-experiments about 10 minutes; test
a few minutes. **About 20–30 minutes.**

Output: `runs/{SET7_NAME}/<timestamp>_<GPU>_rules10/` with `rulebooks/` (every version), `planner_round*.txt`, and
`results/summary.md`, `set10_rulebook.xlsx`.

**Runtime → Change runtime type → A100 GPU.**"""),
    ("code", f'''# ==== settings ====
PROJECT_DIR = "{PROJECT}"
EXPERIMENT  = "{SET7_NAME}"
ROUNDS = 4
NEW_RUN = False                       # False = continue the latest run of this notebook
DRY_RUN = False                       # True = notebook 27's dry run and a stand-in planner on CPU

EXPECTED_CODE = {json.dumps(EXPECTED28, indent=1)}'''),
    ("code", GPU_CHECK),
    ("code", MOUNT),
    ("md", "### Check that Drive is serving the current code; find notebook 27's run"),
    ("code", '''import glob, hashlib
from datetime import datetime
from zoneinfo import ZoneInfo
def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()
CODE_DIR = "/content/set10code"
os.makedirs(CODE_DIR, exist_ok=True)
stale = []
for f, want in EXPECTED_CODE.items():
    shutil.copy(f"{PROJECT_DIR}/code/{f}", f"{CODE_DIR}/{f}")
    if sha(f"{CODE_DIR}/{f}") != want:
        stale.append(f)
assert not stale, (f"Drive is serving a different version of {stale} than this notebook was made for. Wait two minutes, then "
                   "Runtime -> Disconnect and delete runtime, and run again.")
routes = [d for d in sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*_route05" + ("_dryrun" if DRY_RUN else ""))) if os.path.exists(f"{d}/results/summary.md")]
assert routes, "Needs the finished run of notebook 27."
ROUTE_RUN = routes[-1]
CFG = json.load(open(f"{ROUTE_RUN}/config.json"))
SESSION = datetime.now(ZoneInfo("Asia/Singapore")).strftime("%Y%m%d-%H%M%S")
tag = "_rules10_dryrun" if DRY_RUN else "_rules10"
prev = sorted(glob.glob(f"{PROJECT_DIR}/runs/{EXPERIMENT}/*{tag}"))
RUN_DIR = prev[-1] if (prev and not NEW_RUN) else f"{PROJECT_DIR}/runs/{EXPERIMENT}/{SESSION}_{GPU_TAG}{tag}"
SESSION_DIR = f"{RUN_DIR}/sessions/{SESSION}"
for d in (f"{RUN_DIR}/code", f"{RUN_DIR}/logs", SESSION_DIR):
    os.makedirs(d, exist_ok=True)
for f in EXPECTED_CODE:
    shutil.copy(f"{CODE_DIR}/{f}", f"{RUN_DIR}/code/{f}")
TIMES = {}
print("Code files: current.")
print("Notebook 27 run:", ROUTE_RUN)
print("Run folder:     ", RUN_DIR)'''),
    ("code", SET4_MODELS.split("def ensure_local")[0] + '''MODELS = {k: v for k, v in MODELS.items() if k in ("q32",)}
json.dump(MODELS, open(f"{RUN_DIR}/models.json", "w"), indent=1)
json.dump({"experiment": EXPERIMENT, "set": "10 self-learnt rulebook", "route_run": ROUTE_RUN, "rounds": ROUNDS, "gpu": gpu, "dry_run": DRY_RUN,
           "code": EXPECTED_CODE, "started": SESSION}, open(f"{RUN_DIR}/config.json", "w"), indent=1)

def minutes_left():
    return 999.0

def run_stage(args, log_name):
    cmd = [sys.executable, "-u", f"{CODE_DIR}/rules10.py"] + [str(a) for a in args]
    t0 = time.time()
    with open(f"{RUN_DIR}/logs/{log_name}", "a") as log, open(f"{SESSION_DIR}/{log_name}", "a") as slog:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        for line in p.stdout:
            log.write(line); slog.write(line)
            if not any(k in line for k in ("it/s]", "s/it]", "Loading checkpoint", "INFO ", "(EngineCore", "WARNING ", "Warning", "warn", "Loading weights")):
                print(line, end="")
        p.wait()
    TIMES[log_name.replace(".log", "")] = TIMES.get(log_name.replace(".log", ""), 0) + round(time.time() - t0)
    json.dump(TIMES, open(f"{SESSION_DIR}/stage_times.json", "w"), indent=1)
    print(f"--- {log_name}: exit {p.returncode}, {time.time() - t0:.0f}s")
    assert p.returncode == 0, f"stage failed - see {RUN_DIR}/logs/{log_name}"

COMMON = ["--route_run", ROUTE_RUN, "--label_dir", CFG["label_dir"], "--old_data", CFG["old_data"], "--fresh_data", CFG["fresh_data"],
          "--old_sweep", CFG["old_sweep"], "--fresh_run", CFG["fresh_run"], "--out_dir", RUN_DIR, "--models_json", f"{RUN_DIR}/models.json"]
LEARNT = os.path.exists(f"{RUN_DIR}/rulebooks/rulebook_final.json")
print("rulebook already learnt:", LEARNT)'''),
    ("md", "### 1. The planner model (Qwen 32B) and vLLM"),
    ("code", '''import importlib.util
if not LEARNT and not DRY_RUN:
    m = MODELS["q32"]
    if not os.path.exists(f"{m['path']}/config.json"):
        t = time.time()
        print("copying Qwen 32B to local disk ...")
        shutil.copytree(f"{PROJECT_DIR}/models/{m['dir']}", m["path"] + ".part", dirs_exist_ok=True)
        os.rename(m["path"] + ".part", m["path"])
        TIMES["copy_q32"] = round(time.time() - t)
    if importlib.util.find_spec("vllm") is None:
        t = time.time()
        !pip install -q vllm
        !pip uninstall -y -q torchaudio   # Colab's torchaudio is built for another CUDA version and breaks imports
        chk = subprocess.run([sys.executable, "-c", "from vllm import LLM; import vllm, torch; print('vLLM', vllm.__version__, '| torch', torch.__version__)"],
                             capture_output=True, text=True)
        print(chk.stdout or chk.stderr[-2000:])
        assert chk.returncode == 0, "vLLM does not import - Runtime -> Disconnect and delete runtime, then run again."
        TIMES["install_vllm"] = round(time.time() - t)'''),
    ("md", "### 2. Micro-experiments on the dev questions: learn the rulebook"),
    ("code", '''if not LEARNT:
    run_stage(["learn", "--rounds", ROUNDS, "--planner", "mock" if DRY_RUN else "vllm"] + COMMON, "learn.log")
print(open(f"{RUN_DIR}/rulebooks/rulebook_final.json").read())'''),
    ("md", "### 3. Test: the frozen rulebook on the 400 test questions (CPU)"),
    ("code", '''from IPython.display import Markdown, display
run_stage(["test"] + COMMON, "test.log")
display(Markdown(open(f"{RUN_DIR}/results/summary.md").read()))
print("Excel workbook:", f"{RUN_DIR}/results/set10_rulebook.xlsx")
TIMES["total_session"] = round(time.time() - T_START)
json.dump({"session": SESSION, "run_dir": RUN_DIR, "gpu": gpu, "seconds": TIMES, "code": EXPECTED_CODE, "complete": True},
          open(f"{SESSION_DIR}/session.json", "w"), indent=1)
print("Wall-clock this session (s):", TIMES)
print("\\nALL DONE.")
try:
    drive.flush_and_unmount()
    print("Saved to Drive.")
except Exception as e:
    print("Drive flush skipped:", e)'''),
])

import copy
def nb11_variant(models, title_note):
    v = copy.deepcopy(nb11)
    old = 'ONLY_MODELS = []  '
    new = 'ONLY_MODELS = ' + json.dumps(models) + '  '
    hit = 0
    for c in v.cells:
        if old in c.source:
            c.source = c.source.replace(old, new); hit += 1
        if c.source.startswith("# 11 "):
            c.source = c.source.replace("# 11 ", "# " + title_note + " ", 1)
            c.source += "\n\n**This copy is preset to `ONLY_MODELS = " + json.dumps(models) + "`.**"
    assert hit == 1
    for c in v.cells:
        c.id = nbf.v4.new_code_cell().id
    return v
nb11a = nb11_variant(["q3", "q7", "l3", "l8"], "11a")
nb11b = nb11_variant(["q32"], "11b")

nb0.metadata.pop("accelerator", None)          # notebook 00 only downloads: CPU runtime
nb0.metadata["colab"].pop("gpuType", None)
os.makedirs(OUT, exist_ok=True)
for name, n in [("00_setup_download_qwen7b.ipynb", nb0), ("04_step0_results_table.ipynb", nb4),
                ("04b_add_cot_answers.ipynb", nb4b), ("05_script1_agenttts_global.ipynb", nb5),
                ("06_script2_switching.ipynb", nb6), ("07_set2b_switching_cot.ipynb", nb7),
                ("08_set3_gold_answering.ipynb", nb8), ("09_set3_retrieval_cot.ipynb", nb9),
                ("10_set4_setup_models.ipynb", nb10), ("11_set4_sweep.ipynb", nb11),
                ("11a_set4_sweep_q3-q7-l3-l8.ipynb", nb11a), ("11b_set4_sweep_q32.ipynb", nb11b), ("12_set4_agenttts_switching.ipynb", nb12),
                ("13_set5a_offline_policies.ipynb", nb13),
                ("14_set5d_structure_offline.ipynb", nb14),
                ("16_set5d_hidden_signals.ipynb", nb16), ("17_set5c_final_test.ipynb", nb17), ("18_set5e_retriever_gate.ipynb", nb18), ("19_set5f_cross_family_ensembles.ipynb", nb19), ("20_set5g_ideas_document.ipynb", nb20), ("21_set6_next_experiments.ipynb", nb21), ("22_set7_retriever_finetune_3b.ipynb", nb22), ("23a_set7b_retriever_finetune_3b_12k.ipynb", nb23a), ("23b_set7b_retriever_finetune_7b_8k.ipynb", nb23b), ("24_set7c_answer_gate_ft3b.ipynb", nb24), ("25a_set7d_retriever_finetune_qwen3_0.6b.ipynb", nb25a), ("25b_set7d_retriever_finetune_qwen3.5_0.8b.ipynb", nb25b), ("26_set8_cascade_0.5b_0.6b_0.8b.ipynb", nb26), ("27_set9_learned_route_sure_unsure.ipynb", nb27), ("28_set10_self_learnt_rulebook.ipynb", nb28)]:
    nbf.validate(n)
    nbf.write(n, f"{OUT}/{name}")
    print("wrote", name)
