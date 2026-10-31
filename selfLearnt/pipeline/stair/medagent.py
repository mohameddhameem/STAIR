"""MedAgentBench on the SMU cluster: the benchmark's agent loop, run on our models, graded by the authors' refsol.

Faithful to the authors' harness (stanfordmlgroup/MedAgentBench @ 9926011) and to Jonathan's settings (his reply of 31 Oct):
  - the prompt (MedAgentBench_prompt, copied from src/server/tasks/medagentbench/__init__.py), as the first user message;
    the model's turns are assistant messages; no system message (the chat template adds a model's default one, as the
    OpenAI-style server does)
  - each round: output stripped of ``` / ```tool_code; GET -> the authors' send_get_request(url + '&_format=json') and their
    follow-up message; POST -> JSON payload checked, acknowledged only (never sent); FINISH( -> done; anything else ends the
    episode (invalid action); max_round 8 (their configs/tasks/medagentbench.yaml)
  - decoding: temperature 0, max_tokens 2048, everything else from the model's generation_config (what vLLM's server applies
    when a request leaves it unset); a conversation that no longer fits max_model_len - 2048 ends as "agent context limit"
    (max_model_len 11264 for the 72B, as Jonathan served it; 16384 otherwise)
  - grading: their refsol.py (sha256 67d19da2...), task function by id prefix, True -> correct; needs the FHIR server too
  - cost: stair_common.cost.units(params_b, cumulative conversation tokens, reply tokens) per round, summed (as his table)
The FHIR server is the authors' image unpacked without Docker (tools/pull_image.py), started per job on a private copy of
its database. All 300 tasks advance round by round together, one vLLM call per round.

Paths (environment, defaults under ~/stair-data): STAIR_MAB_REPO (git clone of the repo at 9926011), STAIR_MAB_REFSOL
(refsol.py from the authors' Box link), FHIR_IMAGE (the unpacked image)."""
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time

from .common import append_jsonl, log, read_jsonl, write_json
from .config import variant
from .engines import free_gpu, make_engine

MAB_COMMIT = "9926011"
REFSOL_SHA = "67d19da263ac92c1b7075b3218ace3a4bd19b206a06fc7aebcc897dc39625419"
DATA_SHA = "b6e89b2ef82f1bef29c5778a644a8a66c7b25738bbd6a48d03ab3e5182df6ca8"
MAX_ROUND = 8
MAX_TOKENS = 2048
MAX_MODEL_LEN = {"q72": 11264}            # Jonathan's serving of the 72B (46 GB card); 16384 for the others
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# copied verbatim from src/server/tasks/medagentbench/__init__.py @ 9926011
MedAgentBench_prompt = """You are an expert in using FHIR functions to assist medical professionals. You are given a question and a set of possible functions. Based on the question, you will need to make one or more function/tool calls to achieve the purpose.

1. If you decide to invoke a GET function, you MUST put it in the format of
GET url?param_name1=param_value1&param_name2=param_value2...

2. If you decide to invoke a POST function, you MUST put it in the format of
POST url
[your payload data in JSON format]

3. If you have got answers for all the questions and finished all the requested tasks, you MUST call to finish the conversation in the format of (make sure the list is JSON loadable.)
FINISH([answer1, answer2, ...])

Your response must be in the format of one of the three cases, and you can call only one function each time. You SHOULD NOT include any other text in the response.

Here is a list of functions in JSON format that you can invoke. Note that you should use {api_base} as the api_base.
{functions}

Context: {context}
Question: {question}"""


def _p(env, default):
    return os.path.expanduser(os.environ.get(env) or default)


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def wdir(cfg):
    d = os.path.join(cfg["run_dir"], "medagent")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------- the authors' files ----------------
def harness(cfg):
    """(tasks, functions, utils module, refsol module): data checked against Jonathan's hashes; utils and refsol imported
    from a minimal copy of the authors' package path (src.server.tasks.medagentbench), without the AgentBench framework."""
    repo = _p("STAIR_MAB_REPO", "~/stair-data/MedAgentBench")
    refsol = _p("STAIR_MAB_REFSOL", "~/stair-data/refsol.py")
    dfile = os.path.join(repo, "data", "medagentbench", "test_data_v2.json")
    for p, what in ((dfile, f"the MedAgentBench repo at {MAB_COMMIT} (git clone ... {repo})"), (refsol, "refsol.py (authors' Box link)")):
        if not os.path.exists(p):
            raise SystemExit(f"{p} not found: {what}")
    if _sha(dfile) != DATA_SHA:
        raise SystemExit(f"{dfile}: sha256 differs from Jonathan's {DATA_SHA[:12]}...")
    if _sha(refsol) != REFSOL_SHA:
        raise SystemExit(f"{refsol}: sha256 differs from Jonathan's {REFSOL_SHA[:12]}... (not the same grader)")
    shim = os.path.join(wdir(cfg), "shim")
    pkg = os.path.join(shim, "src", "server", "tasks", "medagentbench")
    os.makedirs(pkg, exist_ok=True)
    for d in (os.path.join(shim, "src"), os.path.join(shim, "src", "server"), os.path.join(shim, "src", "server", "tasks"), pkg):
        open(os.path.join(d, "__init__.py"), "a").close()
    shutil.copy(os.path.join(repo, "src", "server", "tasks", "medagentbench", "utils.py"), os.path.join(pkg, "utils.py"))
    shutil.copy(refsol, os.path.join(pkg, "refsol.py"))
    if shim not in sys.path:
        sys.path.insert(0, shim)
    import importlib
    utils = importlib.import_module("src.server.tasks.medagentbench.utils")
    ref = importlib.import_module("src.server.tasks.medagentbench.refsol")
    tasks = json.load(open(dfile, encoding="utf-8"))
    funcs = json.load(open(os.path.join(repo, "data", "medagentbench", "funcs_v1.json"), encoding="utf-8"))
    return tasks, funcs, utils, ref


class FhirServer:
    """The authors' FHIR server from the unpacked image (jobs/fhir_server.sh does the same in bash): a private copy of the
    database on local disk, a free port, ready when /fhir/metadata answers."""

    def __init__(self, log_dir):
        self.image = _p("FHIR_IMAGE", "~/stair-data/medagentbench-image")
        self.log_dir = log_dir

    def __enter__(self):
        R = os.path.join(self.image, "rootfs")
        java = os.path.join(R, "usr/lib/jvm/java-17-openjdk-amd64/bin/java")
        if not os.access(java, os.X_OK):
            raise SystemExit(f"no Java in {R}: run tools/pull_image.py first")
        db = os.path.join(R, "data", "test_db.mv.db")
        base = os.environ.get("TMPDIR") or "/tmp"
        if shutil.disk_usage(base).free < os.path.getsize(db) + (1 << 30):
            base = os.path.expanduser("~/stair-data/tmp")
        self.db_dir = os.path.join(base, f"fhir-db-{os.environ.get('SLURM_JOB_ID', os.getpid())}")
        os.makedirs(self.db_dir, exist_ok=True)
        log(f"copying the patient database to {self.db_dir} ...")
        shutil.copy(db, self.db_dir)
        self.port = 8080                   # the authors' address, which is in every prompt; another port only if taken
        try:
            s = socket.socket()
            s.bind(("127.0.0.1", 8080))
            s.close()
        except OSError:
            s = socket.socket()
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
            s.close()
            log(f"port 8080 is taken on this node: using {self.port} (the prompts then differ from the authors' in the port)")
        self.url = f"http://localhost:{self.port}/fhir/"
        lf = open(os.path.join(self.log_dir, f"fhir-{os.environ.get('SLURM_JOB_ID', os.getpid())}.log"), "w")
        self.proc = subprocess.Popen(
            [java, f"-Xmx{os.environ.get('FHIR_HEAP', '12g')}", "--class-path", os.path.join(R, "app", "main.war"),
             f"-Dloader.path=main.war!/WEB-INF/classes/,main.war!/WEB-INF/,{os.path.join(R, 'app', 'extra-classes')}",
             "org.springframework.boot.loader.PropertiesLauncher", f"--spring.config.location=file://{R}/configs/application.yaml",
             f"--server.port={self.port}", f"--spring.datasource.url=jdbc:h2:{self.db_dir}/test_db;DB_CLOSE_DELAY=-1"],
            cwd=os.path.join(R, "app"), stdout=lf, stderr=subprocess.STDOUT)
        import urllib.request
        for i in range(180):
            try:
                urllib.request.urlopen(self.url + "metadata", timeout=10)
                log(f"FHIR server ready at {self.url} after {5 * i} s")
                return self
            except Exception:  # noqa: BLE001
                if self.proc.poll() is not None:
                    raise SystemExit(f"FHIR server stopped while starting (see {lf.name})")
                time.sleep(5)
        self.__exit__()
        raise SystemExit("FHIR server not ready after 15 minutes")

    def __exit__(self, *a):
        if getattr(self, "proc", None) and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(30)
            except Exception:  # noqa: BLE001
                self.proc.kill()
        shutil.rmtree(getattr(self, "db_dir", ""), ignore_errors=True)


# ---------------- the agent loop ----------------
class _Item(dict):
    """A chat history item / task output readable both as obj.attr and obj['attr'] (refsol's access style unknown)."""
    __getattr__ = dict.get


def _template(tok, messages, thinking=None):
    kw = {"enable_thinking": thinking} if thinking is not None else {}
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    return list(tok(text, add_special_tokens=False)["input_ids"])


def run_episodes(cfg, model_key, tids, fhir_url, utils, tasks, funcs):
    """Episodes of one model on the given task ids (resumes: tasks already in the store are skipped)."""
    from stair_common.cost import units
    out = os.path.join(wdir(cfg), f"episodes_{model_key}.jsonl")
    have = {r["tid"] for r in read_jsonl(out)}
    by_id = {t["id"]: t for t in tasks}
    todo = [t for t in tids if t not in have]
    log(f"{model_key}: {len(todo)} of {len(tids)} episodes to run")
    if not todo:
        return out
    m = variant(cfg, model_key)
    mml = MAX_MODEL_LEN.get(model_key, 16384)
    eng = make_engine(cfg, m, max_model_len=mml)
    from vllm import SamplingParams
    try:                                   # the model's generation_config defaults, as vLLM's server applies them
        sp = eng.llm.get_default_sampling_params()
        sp.temperature, sp.max_tokens = 0.0, MAX_TOKENS
        log(f"sampling: temperature 0, max_tokens {MAX_TOKENS}, from generation_config: repetition_penalty {sp.repetition_penalty}, "
            f"top_p {sp.top_p}, top_k {sp.top_k}")
    except Exception as e:  # noqa: BLE001
        sp = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
        log(f"no generation_config defaults ({e}): temperature 0, max_tokens {MAX_TOKENS} only")
    thinking = False if m.get("no_thinking") else None
    ep = {}
    for tid in todo:
        case = by_id[tid]
        ep[tid] = {"tid": tid, "model": model_key, "status": None, "result": None, "rounds": 0, "n_in": [], "n_out": [],
                   "history": [{"role": "user", "content": MedAgentBench_prompt.format(
                       api_base=fhir_url, functions=json.dumps(funcs), context=case["context"], question=case["instruction"])}]}
    for rnd in range(MAX_ROUND):
        act = [tid for tid in todo if ep[tid]["status"] is None]
        if not act:
            break
        prompts, ids = [], []
        for tid in act:
            msgs = [{"role": "assistant" if h["role"] == "agent" else "user", "content": h["content"]} for h in ep[tid]["history"]]
            p = _template(eng.tok, msgs, thinking)
            if len(p) + MAX_TOKENS > mml:      # the server would refuse it: AgentBench ends the episode
                ep[tid]["status"] = "agent context limit"
                continue
            prompts.append({"prompt_token_ids": p})
            ids.append(tid)
        outs = eng.llm.generate(prompts, [sp] * len(prompts), use_tqdm=False) if prompts else []
        for tid, pr, o in zip(ids, prompts, outs):
            e = ep[tid]
            text = o.outputs[0].text
            e["n_in"].append(len(pr["prompt_token_ids"]))
            e["n_out"].append(len(o.outputs[0].token_ids))
            e["rounds"] = rnd + 1
            e["history"].append({"role": "agent", "content": text})
            try:                                # the authors' handling of one action, line by line
                r = text.strip().replace('```tool_code', '').replace('```', '').strip()
                if r.startswith('GET'):
                    url = r[3:].strip() + '&_format=json'
                    get_res = utils.send_get_request(url)
                    if "data" in get_res:
                        e["history"].append({"role": "user", "content": f"Here is the response from the GET request:\n{get_res['data']}. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"})
                    else:
                        e["history"].append({"role": "user", "content": f"Error in sending the GET request: {get_res['error']}"})
                elif r.startswith('POST'):
                    try:
                        json.loads('\n'.join(r.split('\n')[1:]))
                    except Exception:  # noqa: BLE001
                        e["history"].append({"role": "user", "content": "Invalid POST request"})
                    else:
                        e["history"].append({"role": "user", "content": "POST request accepted and executed successfully. Please call FINISH if you have got answers for all the questions and finished all the requested tasks"})
                elif r.startswith('FINISH('):
                    e["status"], e["result"] = "completed", r[len('FINISH('):-1]
                else:
                    e["status"] = "agent invalid action"
            except Exception as ex:  # noqa: BLE001
                e["status"], e["result"] = "task error", {"error": str(ex)}
        log(f"{model_key} round {rnd + 1}: {len(act)} active, {sum(1 for t in todo if ep[t]['status'] == 'completed')} finished")
    params = float(m["params_b"])
    rows = []
    for tid in todo:
        e = ep[tid]
        e["status"] = e["status"] or "task limit reached"
        e["units"] = float(sum(units(params, a, b) for a, b in zip(e["n_in"], e["n_out"])))
        rows.append(e)
    append_jsonl(out, rows)
    free_gpu(eng)
    return out


def grade(cfg, model_key, fhir_url, ref, tasks):
    """refsol on every stored episode (the authors' eval(): a result is needed; grader True -> correct; errors -> wrong)."""
    by_id = {t["id"]: t for t in tasks}
    rows = read_jsonl(os.path.join(wdir(cfg), f"episodes_{model_key}.jsonl"))
    out = []
    for e in rows:
        ok = False
        if e.get("result") is not None:
            results = _Item(index=None, status=e["status"], result=e["result"], history=[_Item(h) for h in e["history"]])
            try:
                ok = getattr(ref, e["tid"].split("_")[0])(by_id[e["tid"]], results, fhir_url) is True
            except Exception as ex:  # noqa: BLE001
                log(f"{e['tid']}: grader error {ex}")
        out.append({"tid": e["tid"], "model": model_key, "correct": int(ok), "status": e["status"], "rounds": e["rounds"], "units": e["units"]})
    write_json(os.path.join(wdir(cfg), f"graded_{model_key}.json"), out)
    acc = sum(r["correct"] for r in out) / max(1, len(out))
    log(f"{model_key}: {sum(r['correct'] for r in out)}/{len(out)} correct ({100 * acc:.2f}%), {sum(r['units'] for r in out) / max(1, len(out)):.0f} units per task")
    return out


def accept(cfg, pairs=(("q7", "qw"), ("q72", "qs"))):
    """Our runs against Jonathan's replay table (results/medagentbench/training1_table.json on branch jonathan): per task type
    accuracy (his fingerprint) and per task agreement."""
    t = subprocess.run(["git", "-C", ROOT, "show", "origin/jonathan:results/medagentbench/training1_table.json"], capture_output=True, text=True)
    if t.returncode:
        raise SystemExit("run `git fetch origin jonathan` first")
    his = {r["tid"]: r for r in json.loads(t.stdout)}
    report = {}
    for mk, col in pairs:
        p = os.path.join(wdir(cfg), f"graded_{mk}.json")
        if not os.path.exists(p):
            log(f"{mk}: not graded yet")
            continue
        ours = {r["tid"]: r for r in json.load(open(p))}
        common = sorted(set(ours) & set(his), key=lambda x: (int(x.split("_")[0][4:]), int(x.split("_")[1])))
        types = sorted({c.split("_")[0] for c in common}, key=lambda x: int(x[4:]))
        line = []
        for ty in types:
            ids = [c for c in common if c.startswith(ty + "_")]
            line.append(f"{ty[4:]}: {100 * sum(ours[c]['correct'] for c in ids) / len(ids):.0f} / {100 * sum(his[c][col] for c in ids) / len(ids):.0f}")
        a = sum(ours[c]["correct"] for c in common) / len(common)
        b = sum(his[c][col] for c in common) / len(common)
        flips = sum(ours[c]["correct"] != his[c][col] for c in common)
        cu = sum(ours[c]["units"] for c in common) / len(common)
        hu = sum(his[c]["cw" if col == "qw" else "cs"] for c in common) / len(common)
        log(f"{mk} vs Jonathan's {col}: {100 * a:.2f} vs {100 * b:.2f} on {len(common)} tasks; {flips} tasks differ "
            f"({100 * flips / len(common):.1f}%); units {cu:.0f} vs {hu:.0f}")
        log(f"  by task type (ours / his, %): " + " | ".join(line))
        report[mk] = {"ours": a, "his": b, "n": len(common), "flips": flips, "units": cu, "his_units": hu}
    write_json(os.path.join(wdir(cfg), "acceptance.json"), report)


def run(cfg, part, model=""):
    tasks, funcs, utils, ref = harness(cfg)
    if part == "accept":
        accept(cfg)
        return
    tids = [t["id"] for t in tasks]                      # all 300 (the acceptance check compares with all of his table)
    with FhirServer(wdir(cfg)) as srv:
        if part == "episodes":
            run_episodes(cfg, model, tids, srv.url, utils, tasks, funcs)
            grade(cfg, model, srv.url, ref, tasks)
        elif part == "grade":
            grade(cfg, model, srv.url, ref, tasks)
