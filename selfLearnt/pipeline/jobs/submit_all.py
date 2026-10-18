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
    def __init__(self, env, dry, after):
        self.env, self.dry, self.jobs, self.n = env, dry, [], 0
        self.after = after
        self.seed_arg = ""

    def add(self, name, args, gpus, hours, deps=(), cpu_only=False):
        args += self.seed_arg
        deps = [d for d in deps if d]
        if not deps and self.after:
            deps = [self.after]
        mins = int(hours * 60)
        t = f"{mins // 1440}-{(mins % 1440) // 60:02d}:{mins % 60:02d}:00"
        cmd = ["sbatch", "--parsable", f"--job-name=stair-{name}", f"--time={t}", f"--cpus-per-task={self.env.get('CPUS', '8') if not cpu_only else '4'}",
               f"--mem={self.env.get('MEM', '120G') if not cpu_only else '48G'}"]
        for k, flag in (("PARTITION", "--partition"), ("ACCOUNT", "--account"), ("QOS", "--qos")):
            if self.env.get(k) and not self.env[k].startswith("EDIT"):
                cmd.append(f"{flag}={self.env[k]}")
        if self.env.get("MAIL"):
            cmd += [f"--mail-user={self.env['MAIL']}", "--mail-type=END,FAIL"]
        if gpus:
            cmd.append(f"--gres=gpu:{gpus}")
            if self.env.get("CONSTRAINT"):
                cmd.append(f"--constraint={self.env['CONSTRAINT']}")
        if deps:
            cmd.append("--dependency=afterok:" + ":".join(deps))
        cmd += [f"--export=ALL,STAGE_ARGS={args},TIME_LIMIT_MIN={max(30, mins - 120)}", os.path.join("jobs", "job.sbatch")]
        if self.dry:
            self.n += 1
            jid = f"J{self.n}"
        else:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
            if r.returncode:
                sys.exit(f"sbatch failed for {name}: {r.stderr.strip()}")
            jid = r.stdout.strip().split(";")[0]
        self.jobs.append({"id": jid, "name": name, "args": args, "gpus": gpus, "hours": hours, "deps": deps})
        print(f"{jid:>8}  {name:34s} gpus={gpus}  time={t}  after={','.join(deps) or '-'}")
        return jid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phases", default="1,2,3,4,5")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--after", default="")
    a = ap.parse_args()
    env = read_env(os.path.join(HERE, "cluster.env"))
    for k, v in env.items():
        if k.startswith("STAIR_"):
            if v.startswith("EDIT"):
                sys.exit(f"jobs/cluster.env: set {k} first")
            os.environ.setdefault(k, v)
    from stair.config import ft_settings, load_config, variant
    cfg = load_config(os.path.join(ROOT, env.get("CONFIG", "configs/default.yaml")), seed=a.seed)
    sd = f" --seed {a.seed}" if a.seed is not None else ""
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    phases = {int(x) for x in a.phases.split(",") if x.strip()}
    P = Plan(env, a.dry, a.after)
    P.seed_arg = sd
    W = cfg["workstreams"]
    tp = lambda v: int(variant(cfg, v).get("tensor_parallel", 1))

    def train_gpus(k):
        _, s = ft_settings(cfg, k)
        return int(s.get("gpus", 1))

    data = P.add("data", "--stage data", 0, 2, cpu_only=True)
    last = [data]
    # ---------------- phase 1: WS1 + WS2
    if 1 in phases and "ws1" in W:
        g = [P.add(f"ws1-grid-ret-{m}", f"--stage ws1_grid --part retrieve-{m}", tp(m), 24, [data]) for m in W["ws1"]["retrievers"]]
        ga = [P.add(f"ws1-grid-ans-{m}", f"--stage ws1_grid --part answer-{m}", tp(m), 48, g) for m in W["ws1"]["answerers"]]
        s = P.add("ws1-search", "--stage ws1_search", tp(cfg["planner"]), 12, ga)
        t = [P.add(f"ws1-test-ret-{m}", f"--stage ws1_test --part retrieve-{m}", tp(m), 24, [s]) for m in W["ws1"]["retrievers"]]
        ta = [P.add(f"ws1-test-ans-{m}", f"--stage ws1_test --part answer-{m}", tp(m), 48, t) for m in W["ws1"]["answerers"]]
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
                tr[b] = P.add(f"train-{b}", f"--stage train --model {b}", train_gpus(b), 96, after)
        own = lambda v: [tr[v.split(":")[0]]] if v.split(":")[0] in tr else list(after)
        rt = [P.add(f"ret-{v}-{ws}", f"--stage retrieve --model {v}", tp(v) if variant(cfg, v).get("kind") != "ranker" else 1, 36, own(v))
              for v in w["retrievers"] if v not in done_ret]
        done_ret.update(w["retrievers"])
        rt = rt or list(after)
        rc = P.add(f"reach-{ws}", f"--stage reach --ws {ws}", 0, 1, rt, cpu_only=True)
        an = [P.add(f"ans-{v}-{ws}", f"--stage answer --model {v}", tp(v), 36, [rc] + (own(v) if v.split(":")[0] in tr else [])) for v in w["answerers"]]
        dep = an
        sig = w.get("signals", "all")
        if isinstance(sig, list) and any("probe" in x for x in sig):
            pr = [P.add(f"probe-ret-{v}", f"--stage probe --part ret-{v}", tp(v), 24, an) for v in w["retrievers"] if "ret_probe_pct" in sig]
            pr += [P.add(f"probe-ans-{v}", f"--stage probe --part ans-{v}", tp(v), 24, an) for v in w["answerers"] if "ans_probe_pct" in sig]
            dep = [P.add(f"probe-fit-{ws}", "--stage probe --part fit", 0, 2, pr, cpu_only=True)]
        sm = [P.add(f"smp-{v}-{ws}", f"--stage sample --model {v}", tp(v), 24, an) for v in w["answerers"][1:]]
        ln = P.add(f"learn-{ws}", f"--stage learn --ws {ws}", tp(cfg["planner"]), 12, dep + sm)
        te = P.add(f"test-{ws}", f"--stage test --ws {ws}", 0, 2, [ln], cpu_only=True)
        jobs.append(P.add(f"report-{ws}", "--stage report", 0, 1, [te], cpu_only=True))
        return jobs

    for ph, ws in ((2, "ws4"), (3, "ws5"), (4, "ws6"), (5, "ws3")):
        if ph in phases and ws in W:
            last = phase(ws, last)
    out = os.path.join(ROOT, "logs", f"submitted_{time.strftime('%Y%m%d_%H%M%S')}.json")
    if not a.dry:
        json.dump(P.jobs, open(out, "w"), indent=1)
        print(f"\n{len(P.jobs)} jobs submitted (list in {out}). Follow with: squeue --me ; tail -f logs/stair-*.out")
    else:
        print(f"\n{len(P.jobs)} jobs (dry run, nothing submitted)")


if __name__ == "__main__":
    main()
