"""Submit the whole run as SLURM jobs with dependencies, in priority order:
  phase 1  WS1 + WS2 (AgentTTS baselines)           -> first results in about a day
  phase 2  WS4 (fine-tuning on AgentTTS's models + self-learnt rules)
  phase 3  WS5 (full model ladders from nano-jev 33M; reuses WS4's fine-tuned models, adds the small models)
  phase 4  WS6 (control: the same ladders from the fine-tuned Qwen 0.5B; reuses everything, no new training)
  phase 5  WS3 (hidden-state probes on the untrained models)
  report   comparison of everything (also rebuilt after every phase)

Usage (from STAIR/selfLearnt/pipeline, after jobs/install_env.sh and jobs/setup_login.sh):
  python jobs/submit_all.py                 # everything
  python jobs/submit_all.py --phases 1      # only WS1-2;  --phases 2,3 etc.
  python jobs/submit_all.py --dry           # print the jobs without submitting
  python jobs/submit_all.py --after 12345   # make the first jobs wait for job 12345
  python jobs/submit_all.py --seed 1        # another run seed (team standard: 0, 1, 2)
  python jobs/submit_all.py --smoke         # only the 1-GPU smoke test (run this first)
  python jobs/submit_all.py --continue logs/plan_<time>.json   # resume submitting after a disconnect
  python jobs/submit_all.py --as-job        # run the submitter itself as a small CPU job (no tmux needed)
  python jobs/submit_all.py --ss --as-job   # AL-02 replication only: greedy always-strong on our stack, then the comparison
  python jobs/submit_all.py --config configs/golden.yaml --twin q72   # a 2 x H100 copy of a waiting training job
      (whichever starts first trains; the other goes back to the queue, and the submitter withdraws the 4-GPU job once
      the model is trained; cancel a left-over copy by hand)
The cluster limits how many jobs one user may have queued (MAX_JOBS in cluster.env, default 15): this script submits
the jobs in order as earlier ones finish, so leave it running in tmux until it prints "all ... jobs submitted".
Every job resumes from its saved outputs, so re-submitting a phase after a failure is safe (finished work is skipped)."""
import argparse
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def read_env(path):
    env = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.replace("export ", "", 1)
        v = v.strip()
        if v.startswith('"'):
            v = v[1:v.index('"', 1)]
        else:
            v = "" if v.startswith("#") else re.split(r"\s+#", v)[0].strip()
        env[k.strip()] = os.path.expandvars(v)
    return env


class Plan:
    """Builds the job list first (placeholder ids J1, J2, ... and dependencies), then submits it in order, keeping at most
    MAX_JOBS of your jobs in the queue (the cluster limits submitted jobs per user). Later jobs are submitted as earlier
    ones finish: run it in tmux and leave it. Its state is saved, so `--continue` picks up after a disconnect."""

    def __init__(self, env, dry, after):
        self.env, self.dry, self.jobs = env, dry, []
        self.after = after
        self.seed_arg = ""
        self.big_bf16 = lambda m: False   # set in main(): is this model uncompressed and over 20B?
        self.prefix = ""            # job-name prefix of a non-default config (e.g. "pf-"), so its jobs never mix with the real run's

    def add(self, name, args, gpus, hours, deps=(), cpu_only=False, small=False):
        jid = f"J{len(self.jobs) + 1}"
        deps = [d for d in deps if d]
        self.jobs.append({"key": jid, "id": None, "name": name, "args": args + self.seed_arg, "gpus": gpus, "hours": hours,
                          "deps": deps, "cpu_only": cpu_only, "small": small, "no24": self.no24(args)})
        mins = int(hours * 60)
        print(f"{jid:>6}  {name:34s} gpus={gpus}{' (24GB ok)' if small and gpus else ''}  "
              f"time={mins // 1440}-{(mins % 1440) // 60:02d}:{mins % 60:02d}  after={','.join(deps) or '-'}")
        return jid

    def no24(self, args):
        """Jobs with a model over 20B go to cards of 40 GB+ only (see big_bf16 in main)."""
        t = args.split()
        m = t[t.index("--model") + 1] if "--model" in t else None
        if m is None and "--part" in t and "-" in t[t.index("--part") + 1]:
            m = t[t.index("--part") + 1].split("-", 1)[1]
        try:
            return bool(m and self.big_bf16(m))
        except Exception:  # noqa: BLE001
            return False

    def _cmd(self, j, real_deps):
        e, mins = self.env, int(j["hours"] * 60)
        t = f"{mins // 1440}-{(mins % 1440) // 60:02d}:{mins % 60:02d}:00"
        cmd = ["sbatch", "--parsable", f"--job-name=stair-{self.prefix}{j['name']}", f"--time={t}",
               f"--cpus-per-task={e.get('CPUS', '8') if not j['cpu_only'] else '4'}",
               f"--mem={j.get('mem') or (e.get('MEM', '120G') if not j['cpu_only'] else '48G')}"]
        for k, flag in (("PARTITION", "--partition"), ("ACCOUNT", "--account"), ("QOS", "--qos")):
            if e.get(k) and not e[k].startswith("EDIT"):
                cmd.append(f"{flag}={e[k]}")
        if e.get("MAIL"):
            cmd += [f"--mail-user={e['MAIL']}", "--mail-type=FAIL"]      # an e-mail only when a job fails
        gpus, con = j["gpus"], (e.get("CONSTRAINT_SMALL") if j["small"] and e.get("CONSTRAINT_SMALL") else e.get("CONSTRAINT"))
        if j.get("constraint"):                # a job pinned to particular cards (the --twin copy on H100s)
            con = j["constraint"]
        if gpus >= 2 and not j["name"].startswith("train-") and e.get("BIG_GPUS"):
            # models split over several GPUs (32B / 70B / 72B inference): the model is spread over however many GPUs the job
            # gets (vLLM tensor parallel = visible GPUs), so 4 smaller cards work as well as 2 large ones
            gpus, con = int(e["BIG_GPUS"]), e.get("CONSTRAINT_BIG") or con
            if j.get("no24") and con:
                con = "|".join(c for c in con.split("|") if c not in ("a5000", "3090"))
        if gpus:
            cmd.append(f"--gres=gpu:{gpus}")
            if con:
                cmd.append(f"--constraint={con}")
        if real_deps:
            cmd.append("--dependency=afterok:" + ":".join(real_deps))
        cmd += [f"--export=ALL,STAGE_ARGS={j['args']},TIME_LIMIT_MIN={max(30, mins - 120)},STAIR_CONFIG={e.get('CONFIG', 'configs/default.yaml')}",
                os.path.join("jobs", "job.sbatch")]
        return cmd


def _sh(cmd):
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    return r.returncode, r.stdout.strip(), r.stderr.strip()


def _my_queue():
    """{job id: state} of your jobs still in the queue (pending or running)."""
    rc, out, _ = _sh(["squeue", "--me", "-h", "-o", "%i %T"])
    return dict(l.split()[:2] for l in out.splitlines() if l.strip()) if rc == 0 else {}


def _never_satisfied():
    """Your queued jobs whose dependency failed: they would wait forever and hold queue places."""
    rc, out, _ = _sh(["squeue", "--me", "-h", "-o", "%i %j %r"])
    return [l.split()[:2] for l in out.splitlines() if "DependencyNeverSatisfied" in l] if rc == 0 else []


def _final_state(jid):
    rc, out, _ = _sh(["sacct", "-j", jid, "-X", "-n", "-P", "-o", "State"])
    return out.splitlines()[-1].split()[0] if rc == 0 and out.strip() else "UNKNOWN"


def _already_done(cfg, args):
    """True when the work of a job with these run.py arguments is finished and saved (so it needs no GPU allocation).
    Stages that always re-check for new inputs (answer, sample, reach, test, report) are never skipped."""
    from stair.progress import is_done
    t = args.split()
    get = lambda f: t[t.index(f) + 1] if f in t else None
    st, part, model, ws = get("--stage"), get("--part"), get("--model"), get("--ws")
    if st in ("ws1_grid", "ws1_test", "probe") and part:
        return is_done(cfg, st, part)
    if st in ("ws1_search", "data"):
        return is_done(cfg, st)
    if st in ("train", "retrieve") and model:
        return is_done(cfg, st, model)
    if st == "learn" and ws:
        return is_done(cfg, st, ws)
    return False


_NT_CACHE = {}


def _nothing_to_do(cfg, args):
    """An answer / sample job whose every shortlist is already answered (another experiment's job of the same model did
    them): True, so it is not left waiting days for big GPUs only to find nothing to do. Read from the saved outputs on
    CPU; only asked once the job's own dependencies have finished (so its experiment's shortlists are known)."""
    t = args.split()
    st = t[t.index("--stage") + 1] if "--stage" in t else None
    v = t[t.index("--model") + 1] if "--model" in t else None
    if st == "ws1_test" and "--part" in t:   # AgentTTS's test settings: a model none of them uses has nothing to run
        try:
            from stair import agenttts as AT
            kind, _, m = t[t.index("--part") + 1].partition("-")
            plan = AT.test_plan(cfg)
            return bool(plan) and not any((c[0] if kind == "retrieve" else c[2]) == m for c in plan)
        except Exception:  # noqa: BLE001
            return False
    if st not in ("answer", "sample") or not v:
        return False
    try:
        from stair import states as S
        from stair.common import key, read_jsonl
        need = S.needs(cfg, v, samples=(st == "sample"))
        path = S.f_ans(cfg, v) if st == "answer" else S.f_smp(cfg, v)
        if not need:
            return False                  # nothing known to be needed yet: let the job decide
        if not os.path.exists(path):
            return False
        ck = (path, os.path.getmtime(path))
        if ck not in _NT_CACHE:
            n = int(cfg["rules"]["answer_samples"])
            _NT_CACHE.clear()
            _NT_CACHE[ck] = {(r["qid"], key(r["sel"])) for r in read_jsonl(path) if st == "answer" or len(r.get("samples", [])) >= n}
        have = _NT_CACHE[ck]
        return all(k in have for k in need)
    except Exception:  # noqa: BLE001
        return False


def _repoint(P, real, q):
    """Queued jobs that waited on a job withdrawn because its work was already done: make them wait only on the jobs
    they need that are still queued (a cancelled dependency would otherwise never be satisfied)."""
    for x in P.jobs:
        if x["id"] in (None, "done", "blocked") or q.get(x["id"]) != "PENDING":
            continue
        if not any(real.get(d) == "done" for d in x["deps"]):
            continue
        keep = [real[d] for d in x["deps"] if real.get(d) not in (None, "done", "blocked") and real[d] in q]
        rc, _, err = _sh(["scontrol", "update", f"JobId={x['id']}", "Dependency=" + ("afterok:" + ":".join(keep) if keep else "")])
        print(f"{time.strftime('%H:%M')}  {x['name']} ({x['id']}) now waits on {', '.join(keep) or 'nothing'}"
              + (f" (scontrol: {err})" if rc else ""), flush=True)


def submit_loop(P, state_path, max_jobs, poll, cfg=None, wait_slots=6):
    """Submit P.jobs in order with real dependency ids, never more than max_jobs of yours in the queue."""
    # jobs of this plan already in the queue (from an earlier submitter): adopt them instead of submitting twice
    rc, out, _ = _sh(["squeue", "--me", "-h", "-o", "%i %j"])
    pre = "stair-" + P.prefix
    queued = {n[len(pre):]: i for i, n in (l.split()[:2] for l in out.splitlines() if l.strip()) if n.startswith(pre)} if rc == 0 else {}
    for j in P.jobs:
        if not j["id"] and j["name"] in queued:
            j["id"] = queued.pop(j["name"])
            print(f"{time.strftime('%H:%M')}  already queued: {j['name']} ({j['id']})", flush=True)
    real = {j["key"]: j["id"] for j in P.jobs if j["id"]}
    names = {j["key"]: j["name"] for j in P.jobs}
    first = True
    while True:
        q = _my_queue()
        free = max_jobs - len(q)
        # candidates: unsubmitted jobs whose dependencies are all submitted; those that can start now go first, so that
        # jobs waiting on long chains do not fill the queue while ready ones (e.g. training) wait outside it
        states = {}

        def st(rid):
            if rid == "done":
                return "COMPLETED"
            if rid == "blocked":
                return "BLOCKED"
            if rid in q:
                return "QUEUED"
            if rid not in states:
                states[rid] = _final_state(rid)
            return states[rid]

        withdrew = False
        if cfg is not None:
            for j in P.jobs:               # finished work: no job needed (also clears a queued or cancelled job that would only skip)
                if j["id"] in ("done", "blocked"):
                    continue
                if not _already_done(cfg, j["args"]):
                    deps_done = all(d in real and st(real[d]) == "COMPLETED" for d in j["deps"])
                    if not (deps_done and _nothing_to_do(cfg, j["args"])):
                        continue
                if j["id"] and j["id"] in q:
                    if q[j["id"]] != "PENDING":
                        continue
                    _sh(["scancel", j["id"]])
                print(f"{time.strftime('%H:%M')}  already done, not submitted: {j['name']}", flush=True)
                j["id"] = real[j["key"]] = "done"
                withdrew = True
        if withdrew:
            _repoint(P, real, q)
        cands = []
        for k, j in enumerate(P.jobs):
            if j["id"] or any(d not in real for d in j["deps"]):
                continue
            ok_states = ("COMPLETED", "QUEUED", "UNKNOWN", "PENDING", "RUNNING", "REQUEUED")
            # reports are chained only so two never write at once: a report still runs when an earlier one was set aside
            soft = {d for d in j["deps"] if j["name"].startswith("report-") and names[d].startswith("report-")}
            bad = [(d, real[d], st(real[d])) for d in j["deps"] if d not in soft and st(real[d]) not in ok_states]
            if bad:                        # a job it needs failed: set it aside, keep going with everything else
                d, rid, s_ = bad[0]
                j["id"] = real[j["key"]] = "blocked"
                print(f"{time.strftime('%H:%M')}  set aside {j['name']}: it needs {d} ({rid}), which ended {s_}", flush=True)
                continue
            waiting = sum(1 for d in j["deps"] if st(real[d]) != "COMPLETED")
            cands.append((waiting, k, j))
        # jobs waiting on others may hold at most WAIT_SLOTS queue places, so work that can start now always gets in;
        # if waiting jobs already hold more, the latest-submitted ones nothing else waits on are withdrawn (sent again later)
        rc_, out_, _ = _sh(["squeue", "--me", "-h", "-o", "%i %r"])
        waiting_q = [l.split()[0] for l in out_.splitlines() if l.strip() and "Dependency" in l] if rc_ == 0 else []
        ready = [c for c in cands if c[0] == 0]
        if ready and free <= 0 and len(waiting_q) > wait_slots:
            needed = {d for x in P.jobs if x["id"] not in (None, "done", "blocked") for d in x["deps"]}
            leaves = [x for x in P.jobs if x["id"] in waiting_q and x["key"] not in needed]
            for x in sorted(leaves, key=lambda x: -int(x["id"]))[:min(len(ready), len(waiting_q) - wait_slots)]:
                _sh(["scancel", x["id"]])
                print(f"{time.strftime('%H:%M')}  withdrew {x['name']} ({x['id']}) to make room for jobs that can start now; sent again later", flush=True)
                waiting_q.remove(x["id"])
                x["id"] = None
                real.pop(x["key"], None)
                free += 1
        n_wait = len(waiting_q)
        for waiting_, _, j in sorted(cands, key=lambda c: (c[0], c[1])):
            if free <= 0:
                break
            if waiting_ and n_wait >= wait_slots:
                continue
            n_wait += 1 if waiting_ else 0
            deps = [real[d] for d in j["deps"] if st(real[d]) not in ("COMPLETED", "BLOCKED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL")]
            if P.after and not deps and first:
                deps = [x for x in str(P.after).split(",") if x.strip()]     # --after 123,456: wait for these jobs
            rc, out, err = _sh(P._cmd(j, deps))
            if rc:
                if "QOSMaxSubmitJobPerUserLimit" in err or "job submit limit" in err:
                    break                  # the queue is full after all: try again later
                sys.exit(f"sbatch failed for {j['name']}: {err}")
            j["id"] = real[j["key"]] = out.split(";")[0]
            free -= 1
            print(f"{time.strftime('%H:%M')}  submitted {j['id']:>8}  {j['name']}" + (f"  after {','.join(deps)}" if deps else ""), flush=True)
        first = False
        for jid, name in _never_satisfied():   # queued jobs whose dependency failed: cancel, set aside
            for j in P.jobs:
                if j["id"] == jid:
                    failed = [d for d in j["deps"] if real.get(d) != "done" and real.get(d) not in q and st(real.get(d, "blocked")) != "COMPLETED"]
                    if not failed:             # what it waited on was withdrawn as already done, nothing failed
                        _repoint(P, real, q)
                        continue
                    _sh(["scancel", jid])
                    j["id"] = real[j["key"]] = "blocked"
                    print(f"{time.strftime('%H:%M')}  set aside {j['name']} ({jid}): a job it needs failed", flush=True)
        json.dump({"jobs": P.jobs, "after": P.after}, open(state_path, "w"), indent=1)
        left = sum(1 for j in P.jobs if not j["id"])
        aside = [j["name"] for j in P.jobs if j["id"] == "blocked"]
        if not left and any(j["id"] in q for j in P.jobs):
            print(f"{time.strftime('%H:%M')}  all jobs submitted; watching the {sum(1 for j in P.jobs if j['id'] in q)} still queued or running"
                  f" (checking again in {poll // 60} min)", flush=True)
            time.sleep(poll)
            continue
        if not left:
            failed = [f"{j['name']} ({j['id']})" for j in P.jobs if j["id"] not in (None, "done", "blocked") and j["id"] not in _my_queue()
                      and _final_state(j["id"]) not in ("COMPLETED", "UNKNOWN", "PENDING", "RUNNING", "REQUEUED")]
            if aside:
                print(f"\nall runnable jobs submitted; {len(aside)} set aside because a job they need failed: {', '.join(aside)}"
                      + (f"\nfailed: {', '.join(failed)}" if failed else "")
                      + f"\nAfter fixing the cause:  python jobs/submit_all.py --as-job --continue {state_path}"
                        "   (failed and set-aside jobs are submitted again; finished work is skipped)", flush=True)
                return             # ends normally: the lines above say what was set aside
            print(f"\nall {len(P.jobs)} jobs submitted (list in {state_path}). Follow with: squeue --me")
            return
        print(f"{time.strftime('%H:%M')}  {left} jobs still to submit" + (f" ({len(aside)} set aside after a failure)" if aside else "")
              + f"; {len(q)} of yours in the queue (limit {max_jobs}); checking again in {poll // 60} min", flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="submit only the 1-GPU smoke test (jobs/smoke.sbatch)")
    ap.add_argument("--config", default="", help="another config, e.g. configs/preflight.yaml (default: CONFIG in cluster.env)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--after", default="", help="job id(s), comma-separated, the first jobs wait for")
    ap.add_argument("--continue", dest="cont", default="", help="resume submitting from a saved plan (logs/plan_*.json)")
    ap.add_argument("--twin", default="", help="model key (e.g. q72): a copy of its training job on --twin-gpus x --twin-cards")
    ap.add_argument("--twin-gpus", type=int, default=2)
    ap.add_argument("--twin-cards", default="h100|h100nvl", help="Slurm features the copy may run on (80 GB+ cards)")
    ap.add_argument("--twin-mem", default="240G", help="CPU memory of the copy (loading a 72B model for QLoRA used ~119 GB)")
    ap.add_argument("--fhir-test", action="store_true", help="CPU job: start MedAgentBench's records server and check one known answer")
    ap.add_argument("--ss-jon", action="store_true", help="AL-02 replication on Jonathan's pools, retrievals and prompt (answer step)")
    ap.add_argument("--ss", action="store_true", help="AL-02 replication only: greedy Qwen 72B -> Llama 70B, then compare with the shared rows")
    ap.add_argument("--as-job", action="store_true", help="run this submitter as a small CPU job (survives logouts) instead of in tmux")
    a = ap.parse_args()
    env = read_env(os.path.join(HERE, "cluster.env"))
    if os.path.exists(os.path.join(HERE, "cluster.local.env")):      # your own settings (account, storage), never committed
        env.update(read_env(os.path.join(HERE, "cluster.local.env")))
    if a.config:
        env["CONFIG"] = a.config
    if a.as_job:
        args = [x for x in sys.argv[1:] if x != "--as-job"]
        cmd = ["sbatch", "--parsable"] + [f"{flag}={env[k]}" for k, flag in (("PARTITION", "--partition"), ("ACCOUNT", "--account"), ("QOS", "--qos"))
                                          if env.get(k) and not env[k].startswith("EDIT")]
        cmd += [f"--export=ALL,SUBMIT_ARGS={' '.join(args)}", os.path.join("jobs", "submitter.sbatch")]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        sys.exit(f"sbatch failed: {r.stderr.strip()}" if r.returncode else
                 f"submitter running as job {r.stdout.strip()}; follow it with:  tail -f logs/stair-submitter-{r.stdout.strip()}.out")
    for k, v in env.items():
        if k.startswith("STAIR_"):
            if v.startswith("EDIT"):
                sys.exit(f"jobs/cluster.env: set {k} first")
            os.environ.setdefault(k, v)
    from stair.config import ft_settings, load_config, variant
    cfg = load_config(os.path.join(ROOT, env.get("CONFIG", "configs/default.yaml")), seed=a.seed)
    sd = f" --seed {a.seed}" if a.seed is not None else ""
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    if a.fhir_test:
        cmd = ["sbatch", "--parsable"] + [f"{flag}={env[k]}" for k, flag in (("PARTITION", "--partition"), ("ACCOUNT", "--account"), ("QOS", "--qos"))
                                          if env.get(k) and not env[k].startswith("EDIT")] + [os.path.join("jobs", "fhir_test.sbatch")]
        r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        sys.exit(f"sbatch failed: {r.stderr.strip()}" if r.returncode else
                 f"submitted {r.stdout.strip()}: tail -f logs/stair-fhir-test-{r.stdout.strip()}.out")
    if a.smoke:
        cmd = ["sbatch", "--parsable"] + [f"{flag}={env[k]}" for k, flag in (("PARTITION", "--partition"), ("ACCOUNT", "--account"), ("QOS", "--qos"))
                                          if env.get(k) and not env[k].startswith("EDIT")]
        if env.get("CONSTRAINT_SMALL") or env.get("CONSTRAINT"):
            cmd.append(f"--constraint={env.get('CONSTRAINT_SMALL') or env['CONSTRAINT']}")   # small models only
        cmd.append(os.path.join("jobs", "smoke.sbatch"))
        print(" ".join(cmd))
        if not a.dry:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            sys.exit(f"sbatch failed: {r.stderr.strip()}" if r.returncode else f"smoke test submitted: job {r.stdout.strip()} -> logs/stair-smoke-{r.stdout.strip()}.out")
        return
    phases = {int(x) for x in a.phases.split(",") if x.strip()}
    P = Plan(env, a.dry, a.after)
    P.seed_arg = sd
    # every model over 20B: 4 cards of 40 GB+ (160 GB+, about twice what the largest job needs: Qwen 32B ~75 GB,
    # fine-tuned Llama 70B ~65 GB with its 49k-token context). 24 GB cards ran out of memory for 32B and fine-tuned 70B.
    P.big_bf16 = lambda v: (v in cfg["models"] or v.split(":")[0] in cfg["models"]) and float(variant(cfg, v)["params_b"]) > 20
    if os.path.basename(env.get("CONFIG", "configs/default.yaml")) != "default.yaml":
        P.prefix = os.path.splitext(os.path.basename(env["CONFIG"]))[0][:2] + "-"     # preflight.yaml -> "pr-"
    max_jobs, poll = int(env.get("MAX_JOBS", 15)), int(float(env.get("POLL_MIN", 5)) * 60)
    if a.cont:
        st = json.load(open(a.cont))
        P.jobs, P.after = st["jobs"], st.get("after", "")
        for j in P.jobs:                   # plans saved by an older version: recompute the GPU-type rule
            j["no24"] = P.no24(j["args"])
        retry = 0
        for j in P.jobs:                   # set-aside jobs, and submitted jobs that failed, are tried again
            if j["id"] == "blocked" or (j["id"] not in (None, "done") and j["id"] not in _my_queue()
                                        and _final_state(j["id"]) not in ("COMPLETED", "UNKNOWN", "PENDING", "RUNNING", "REQUEUED")):
                j["id"], retry = None, retry + 1
        print(f"continuing {a.cont}: {sum(1 for j in P.jobs if not j['id'])} of {len(P.jobs)} jobs still to submit ({retry} to try again)")
        submit_loop(P, a.cont, max_jobs, poll, cfg, int(env.get("WAIT_SLOTS", 6)))
        return
    W = cfg["workstreams"]
    tp = lambda v: int(variant(cfg, v).get("tensor_parallel", 1))
    if a.twin:
        j = {"key": "T", "id": None, "name": f"train-{a.twin}-{a.twin_cards.split('|')[0]}", "args": f"--stage train --model {a.twin}" + sd,
             "gpus": a.twin_gpus, "hours": 96, "deps": [], "cpu_only": False, "small": False, "no24": False, "constraint": a.twin_cards,
             "mem": a.twin_mem}
        if _already_done(cfg, j["args"]):
            sys.exit(f"{a.twin} is already trained: no copy needed")
        cmd = P._cmd(j, [])
        print(" ".join(cmd))
        if not a.dry:
            rc, out, err = _sh(cmd)
            sys.exit(f"sbatch failed: {err}" if rc else f"submitted {out.split(';')[0]}  stair-{P.prefix}{j['name']}: whichever of it and the "
                     f"4-GPU job starts first trains {a.twin}")
        return
    if a.ss_jon:                           # one GPU job, no submitter: the queue is nearly full; compare runs on the login node
        P.prefix = "jon-"
        P.add("ans-l70", "--stage ss_jon --part answer --model l70", tp("l70"), 6)   # --model: the big-GPU card rule
        cmd = P._cmd(P.jobs[0], [])
        if a.dry:
            print(" ".join(cmd))
            return
        rc, out, err = _sh(cmd)
        sys.exit(f"sbatch failed: {err}" if rc else f"submitted {out.split(';')[0]}  stair-jon-ans-l70; when it has finished:  "
                 "bash -c 'source jobs/cluster.env; source jobs/cluster.local.env; python run.py --config configs/default.yaml "
                 "--stage ss_jon --part compare'")
    if a.ss:
        rm, am = (W.get("ws2", {}).get("config") or ["q72", 1, "l70", 1])[0::2]
        P.prefix = "ss-"
        r = P.add(f"ret-{rm}", f"--stage ss_greedy --part retrieve-{rm}", tp(rm), 12)
        n = P.add(f"ans-{am}", f"--stage ss_greedy --part answer-{am}", tp(am), 12, [r])
        P.add("compare", "--stage ss_greedy --part compare", 0, 1, [n], cpu_only=True)
        if a.dry:
            print(f"\n{len(P.jobs)} jobs (dry run, nothing submitted)")
            return
        out = os.path.join(ROOT, "logs", f"plan_ss_{time.strftime('%Y%m%d_%H%M%S')}.json")
        submit_loop(P, out, max_jobs, poll, cfg, int(env.get("WAIT_SLOTS", 6)))
        return
    pb = lambda v: float(variant(cfg, v)["params_b"])
    sm_inf = lambda v: pb(v) <= 7          # vLLM inference fits a 24 GB GPU up to 7B (Llama 8B's 64k-token context does not)
    sm_tr = lambda k: pb(k) <= 3           # LoRA training with long retrieval prompts: up to 3B on 24 GB

    def train_gpus(k):
        _, s = ft_settings(cfg, k)
        return int(s.get("gpus", 1))

    if cfg.get("models_from_run"):             # inference only on another run's fine-tuned models: all must be trained
        from stair.progress import is_done as _isd
        src = dict(cfg, run_dir=os.path.join(cfg["paths"]["runs_dir"], cfg["models_from_run"], f"seed{cfg['seed']}"))
        need = sorted({v.split(":")[0] for w in cfg["workstreams"].values() if w.get("kind") == "rules"
                       for v in w["retrievers"] + w["answerers"] if v.endswith(":ft")})
        left = [k for k in need if not _isd(src, "train", k)]
        if left:
            sys.exit(f"run {cfg['models_from_run']} has not finished fine-tuning {', '.join(left)} yet: submit this run after it has")
    data = P.add("data", "--stage data", 0, 2, cpu_only=True)
    last = [data]
    # ---------------- phase 1: WS1 + WS2
    if 1 in phases and "ws1" in W:
        g = [P.add(f"ws1-grid-ret-{m}", f"--stage ws1_grid --part retrieve-{m}", tp(m), 24, [data], small=sm_inf(m)) for m in W["ws1"]["retrievers"]]
        ga = [P.add(f"ws1-grid-ans-{m}", f"--stage ws1_grid --part answer-{m}", tp(m), 48, g, small=sm_inf(m)) for m in W["ws1"]["answerers"]]
        s = P.add("ws1-search", "--stage ws1_search", tp(cfg["planner"]), 12, ga)
        t = [P.add(f"ws1-test-ret-{m}", f"--stage ws1_test --part retrieve-{m}", tp(m), 24, [s], small=sm_inf(m)) for m in W["ws1"]["retrievers"]]
        ta = [P.add(f"ws1-test-ans-{m}", f"--stage ws1_test --part answer-{m}", tp(m), 48, t, small=sm_inf(m)) for m in W["ws1"]["answerers"]]
        last = [P.add("report-phase1", "--stage report", 0, 1, ta, cpu_only=True)]
    # ---------------- phases 2-5: rule-based experiments, run side by side (the cluster runs up to 8 of your jobs at once).
    # Each job waits only for what it reads: a model's training, its shortlists, the experiment's earlier steps. Jobs that
    # write the same model's files (answers, samples) are chained, so two never write at once; reports take turns via a lock.
    train_job, ret_job, last_ans, last_smp = {}, {}, {}, {}

    def phase(ws):
        w = W[ws]
        base = lambda v: v.split(":")[0]
        for v in w["retrievers"] + w["answerers"]:
            b, _, tag = v.partition(":")
            if tag == "ft" and b not in train_job and cfg.get("models_from_run"):
                train_job[b] = None            # the models come from another run, which must have trained them
                continue
            if tag == "ft" and b not in train_job:
                train_job[b] = P.add(f"train-{b}", f"--stage train --model {b}", train_gpus(b), 96, [data], small=sm_tr(b))
        own = lambda v: [train_job[base(v)] or data] if v.endswith(":ft") else [data]
        for v in w["retrievers"]:
            if v not in ret_job:
                ret_job[v] = P.add(f"ret-{v}-{ws}", f"--stage retrieve --model {v}", tp(v) if variant(cfg, v).get("kind") != "ranker" else 1, 36,
                                   own(v), small=sm_inf(v))
        rc = P.add(f"reach-{ws}", f"--stage reach --ws {ws}", 0, 1, [ret_job[v] for v in w["retrievers"]], cpu_only=True)
        an = []
        for v in w["answerers"]:
            last_ans[v] = P.add(f"ans-{v}-{ws}", f"--stage answer --model {v}", tp(v), 36, [rc] + own(v) + ([last_ans[v]] if v in last_ans else []),
                                small=sm_inf(v))
            an.append(last_ans[v])
        dep = an
        sig = w.get("signals", "all")
        if isinstance(sig, list) and any("probe" in x for x in sig):
            pr = [P.add(f"probe-ret-{v}", f"--stage probe --part ret-{v}", tp(v), 24, an, small=sm_inf(v)) for v in w["retrievers"] if "ret_probe_pct" in sig]
            pr += [P.add(f"probe-ans-{v}", f"--stage probe --part ans-{v}", tp(v), 24, an, small=sm_inf(v)) for v in w["answerers"] if "ans_probe_pct" in sig]
            dep = [P.add(f"probe-fit-{ws}", "--stage probe --part fit", 0, 2, pr, cpu_only=True)]
        sm = []
        for v in w["answerers"][1:]:
            last_smp[v] = P.add(f"smp-{v}-{ws}", f"--stage sample --model {v}", tp(v), 24, an + ([last_smp[v]] if v in last_smp else []), small=sm_inf(v))
            sm.append(last_smp[v])
        ln = P.add(f"learn-{ws}", f"--stage learn --ws {ws}", tp(cfg["planner"]), 12, dep + sm)
        te = P.add(f"test-{ws}", f"--stage test --ws {ws}", 0, 2, [ln], cpu_only=True)
        P.add(f"report-{ws}", "--stage report", 0, 1, [te], cpu_only=True)   # reports take turns via a lock file, so no chain

    for ph, ws in ((2, "ws4"), (3, "ws5"), (4, "ws6"), (5, "ws3")):
        if ph in phases and ws in W:
            phase(ws)
    if a.dry:
        print(f"\n{len(P.jobs)} jobs (dry run, nothing submitted)")
        return
    out = os.path.join(ROOT, "logs", f"plan_{time.strftime('%Y%m%d_%H%M%S')}.json")
    print(f"\nsubmitting {len(P.jobs)} jobs, at most {max_jobs} of yours queued at a time (keep this running in tmux) ...")
    submit_loop(P, out, max_jobs, poll, cfg, int(env.get("WAIT_SLOTS", 6)))


if __name__ == "__main__":
    main()
