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
            v = re.split(r"\s+#", v)[0].strip()
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

    def add(self, name, args, gpus, hours, deps=(), cpu_only=False, small=False):
        jid = f"J{len(self.jobs) + 1}"
        deps = [d for d in deps if d]
        self.jobs.append({"key": jid, "id": None, "name": name, "args": args + self.seed_arg, "gpus": gpus, "hours": hours,
                          "deps": deps, "cpu_only": cpu_only, "small": small})
        mins = int(hours * 60)
        print(f"{jid:>6}  {name:34s} gpus={gpus}{' (24GB ok)' if small and gpus else ''}  "
              f"time={mins // 1440}-{(mins % 1440) // 60:02d}:{mins % 60:02d}  after={','.join(deps) or '-'}")
        return jid

    def _cmd(self, j, real_deps):
        e, mins = self.env, int(j["hours"] * 60)
        t = f"{mins // 1440}-{(mins % 1440) // 60:02d}:{mins % 60:02d}:00"
        cmd = ["sbatch", "--parsable", f"--job-name=stair-{j['name']}", f"--time={t}",
               f"--cpus-per-task={e.get('CPUS', '8') if not j['cpu_only'] else '4'}", f"--mem={e.get('MEM', '120G') if not j['cpu_only'] else '48G'}"]
        for k, flag in (("PARTITION", "--partition"), ("ACCOUNT", "--account"), ("QOS", "--qos")):
            if e.get(k) and not e[k].startswith("EDIT"):
                cmd.append(f"{flag}={e[k]}")
        if e.get("MAIL"):
            cmd += [f"--mail-user={e['MAIL']}", "--mail-type=END,FAIL"]
        if j["gpus"]:
            cmd.append(f"--gres=gpu:{j['gpus']}")
            con = e.get("CONSTRAINT_SMALL") if j["small"] and e.get("CONSTRAINT_SMALL") else e.get("CONSTRAINT")
            if con:
                cmd.append(f"--constraint={con}")
        if real_deps:
            cmd.append("--dependency=afterok:" + ":".join(real_deps))
        cmd += [f"--export=ALL,STAGE_ARGS={j['args']},TIME_LIMIT_MIN={max(30, mins - 120)}", os.path.join("jobs", "job.sbatch")]
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


def submit_loop(P, state_path, max_jobs, poll):
    """Submit P.jobs in order with real dependency ids, never more than max_jobs of yours in the queue."""
    real = {j["key"]: j["id"] for j in P.jobs if j["id"]}
    first = True
    while True:
        q = _my_queue()
        free = max_jobs - len(q)
        blocked = None
        for j in P.jobs:
            if j["id"]:
                continue
            if free <= 0:
                break
            if any(d not in real for d in j["deps"]):
                break                      # an earlier job is not submitted yet: keep the order
            deps = []
            for d in j["deps"]:
                rid = real[d]
                if rid in q:
                    deps.append(rid)
                    continue
                st = _final_state(rid)
                if st == "COMPLETED":
                    continue               # finished fine: nothing to wait for
                if st in ("UNKNOWN", "PENDING", "RUNNING", "REQUEUED"):
                    deps.append(rid)
                    continue
                blocked = (j["name"], d, rid, st)
                break
            if blocked:
                break
            if P.after and not deps and first:
                deps = [P.after]
            rc, out, err = _sh(P._cmd(j, deps))
            if rc:
                if "QOSMaxSubmitJobPerUserLimit" in err or "job submit limit" in err:
                    break                  # the queue is full after all: try again later
                sys.exit(f"sbatch failed for {j['name']}: {err}")
            j["id"] = real[j["key"]] = out.split(";")[0]
            free -= 1
            print(f"{time.strftime('%H:%M')}  submitted {j['id']:>8}  {j['name']}" + (f"  after {','.join(deps)}" if deps else ""), flush=True)
        first = False
        json.dump({"jobs": P.jobs, "after": P.after}, open(state_path, "w"), indent=1)
        stuck = _never_satisfied()
        if stuck:
            failed = [f"{j['name']} ({j['id']}: {st})" for j in P.jobs if j["id"] and j["id"] not in q
                      for st in [_final_state(j["id"])] if st not in ("COMPLETED", "UNKNOWN", "PENDING", "RUNNING", "REQUEUED")]
            sys.exit("\nstopped: a job failed, so jobs depending on it can never start: "
                     + ", ".join(n for _, n in stuck)
                     + (f"\nfailed: {', '.join(failed)}" if failed else "")
                     + "\nLook at the failed job's log (logs/stair-<name>-<id>.out) and fix the cause; then cancel the stuck jobs"
                       " (scancel --me) and submit again: python jobs/submit_all.py   (finished work is skipped)")
        left = sum(1 for j in P.jobs if not j["id"])
        if blocked:
            n, d, rid, st = blocked
            sys.exit(f"\nstopped: {n} needs job {rid} ({d}), which ended {st}. Look at its log (logs/stair-*-{rid}.out), fix the cause,"
                     f" then run:  python jobs/submit_all.py --continue {state_path}   (or resubmit the phase: finished work is skipped)")
        if not left:
            print(f"\nall {len(P.jobs)} jobs submitted (list in {state_path}). Follow with: squeue --me")
            return
        print(f"{time.strftime('%H:%M')}  {left} jobs still to submit; {len(q)} of yours in the queue (limit {max_jobs}); checking again in {poll // 60} min",
              flush=True)
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="submit only the 1-GPU smoke test (jobs/smoke.sbatch)")
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--after", default="")
    ap.add_argument("--continue", dest="cont", default="", help="resume submitting from a saved plan (logs/plan_*.json)")
    a = ap.parse_args()
    env = read_env(os.path.join(HERE, "cluster.env"))
    if os.path.exists(os.path.join(HERE, "cluster.local.env")):      # your own settings (account, storage), never committed
        env.update(read_env(os.path.join(HERE, "cluster.local.env")))
    for k, v in env.items():
        if k.startswith("STAIR_"):
            if v.startswith("EDIT"):
                sys.exit(f"jobs/cluster.env: set {k} first")
            os.environ.setdefault(k, v)
    from stair.config import ft_settings, load_config, variant
    cfg = load_config(os.path.join(ROOT, env.get("CONFIG", "configs/default.yaml")), seed=a.seed)
    sd = f" --seed {a.seed}" if a.seed is not None else ""
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
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
    max_jobs, poll = int(env.get("MAX_JOBS", 15)), int(float(env.get("POLL_MIN", 5)) * 60)
    if a.cont:
        st = json.load(open(a.cont))
        P.jobs, P.after = st["jobs"], st.get("after", "")
        print(f"continuing {a.cont}: {sum(1 for j in P.jobs if not j['id'])} of {len(P.jobs)} jobs still to submit")
        submit_loop(P, a.cont, max_jobs, poll)
        return
    W = cfg["workstreams"]
    tp = lambda v: int(variant(cfg, v).get("tensor_parallel", 1))
    pb = lambda v: float(variant(cfg, v)["params_b"])
    sm_inf = lambda v: pb(v) <= 7          # vLLM inference fits a 24 GB GPU up to 7B (Llama 8B's 64k-token context does not)
    sm_tr = lambda k: pb(k) <= 3           # LoRA training with long retrieval prompts: up to 3B on 24 GB

    def train_gpus(k):
        _, s = ft_settings(cfg, k)
        return int(s.get("gpus", 1))

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
    # ---------------- phases 2-4: rule-based workstreams
    trained = set()
    done_ret = set()   # a model's outputs are shared between experiments: run each once

    def phase(ws, after):
        w = W[ws]
        jobs = []
        tr = {}
        for v in w["retrievers"] + w["answerers"]:
            b, _, tag = v.partition(":")
            if tag == "ft" and b not in trained:
                trained.add(b)
                tr[b] = P.add(f"train-{b}", f"--stage train --model {b}", train_gpus(b), 96, after, small=sm_tr(b))
        own = lambda v: [tr[v.split(":")[0]]] if v.split(":")[0] in tr else list(after)
        rt = [P.add(f"ret-{v}-{ws}", f"--stage retrieve --model {v}", tp(v) if variant(cfg, v).get("kind") != "ranker" else 1, 36, own(v), small=sm_inf(v))
              for v in w["retrievers"] if v not in done_ret]
        done_ret.update(w["retrievers"])
        rt = rt or list(after)
        rc = P.add(f"reach-{ws}", f"--stage reach --ws {ws}", 0, 1, rt, cpu_only=True)
        an = [P.add(f"ans-{v}-{ws}", f"--stage answer --model {v}", tp(v), 36, [rc] + (own(v) if v.split(":")[0] in tr else []), small=sm_inf(v)) for v in w["answerers"]]
        dep = an
        sig = w.get("signals", "all")
        if isinstance(sig, list) and any("probe" in x for x in sig):
            pr = [P.add(f"probe-ret-{v}", f"--stage probe --part ret-{v}", tp(v), 24, an, small=sm_inf(v)) for v in w["retrievers"] if "ret_probe_pct" in sig]
            pr += [P.add(f"probe-ans-{v}", f"--stage probe --part ans-{v}", tp(v), 24, an, small=sm_inf(v)) for v in w["answerers"] if "ans_probe_pct" in sig]
            dep = [P.add(f"probe-fit-{ws}", "--stage probe --part fit", 0, 2, pr, cpu_only=True)]
        sm = [P.add(f"smp-{v}-{ws}", f"--stage sample --model {v}", tp(v), 24, an, small=sm_inf(v)) for v in w["answerers"][1:]]
        ln = P.add(f"learn-{ws}", f"--stage learn --ws {ws}", tp(cfg["planner"]), 12, dep + sm)
        te = P.add(f"test-{ws}", f"--stage test --ws {ws}", 0, 2, [ln], cpu_only=True)
        jobs.append(P.add(f"report-{ws}", "--stage report", 0, 1, [te], cpu_only=True))
        return jobs

    for ph, ws in ((2, "ws4"), (3, "ws5"), (4, "ws6"), (5, "ws3")):
        if ph in phases and ws in W:
            last = phase(ws, last)
    if a.dry:
        print(f"\n{len(P.jobs)} jobs (dry run, nothing submitted)")
        return
    out = os.path.join(ROOT, "logs", f"plan_{time.strftime('%Y%m%d_%H%M%S')}.json")
    print(f"\nsubmitting {len(P.jobs)} jobs, at most {max_jobs} of yours queued at a time (keep this running in tmux) ...")
    submit_loop(P, out, max_jobs, poll)


if __name__ == "__main__":
    main()
