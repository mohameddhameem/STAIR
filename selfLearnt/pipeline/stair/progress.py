"""Progress tracking and the GitHub status page.

Every long loop reports (done, total) to a small JSON file under runs/<run>/status/progress/. The status page (stage table,
estimated finish time, results so far, charts) is rebuilt from those files and pushed to a branch of the group repository as a
single commit that replaces the previous one, so the branch stays a few hundred KB however long the run.

Token: environment variable GITHUB_TOKEN, or the file ~/.stair_github_token (a classic token with the 'public_repo' scope).
If a job cannot reach GitHub, the push fails quietly; run jobs/status_loop.sh on the login node instead."""
import os
import shutil
import subprocess
import tempfile
import time

from .common import log, read_json, write_json

STAGES = ["setup", "data", "train", "ws1_grid", "ws1_search", "ws1_test", "retrieve", "probe", "answer", "sample", "learn", "test", "report"]
STAGE_NAMES = {"setup": "Download models and HotpotQA", "data": "Build question sets", "train": "Fine-tune retrievers and answerers",
               "ws1_grid": "WS1: AgentTTS settings on the search questions", "ws1_search": "WS1: AgentTTS planner search",
               "ws1_test": "WS1-2: fixed settings on the test questions", "retrieve": "WS3-5: shortlists (every retriever)",
               "reach": "WS3-5: list every reachable shortlist", "probe": "WS3: hidden-state probes", "answer": "WS3-5: answers (every answerer)", "sample": "WS3-5: sampled answers (votes)",
               "learn": "WS3-5: micro-experiments (rule learning)", "test": "WS3-5: learnt rules on the test questions",
               "report": "Comparison of all workstreams"}


def sdir(cfg, *parts):
    d = os.path.join(cfg["run_dir"], "status", *parts)
    os.makedirs(d if not os.path.splitext(d)[1] else os.path.dirname(d), exist_ok=True)
    return d


class Tracker:
    """Progress of one part of a stage (e.g. stage 'answer', part 'l70:ft')."""

    def __init__(self, cfg, stage, part, total, unit="items"):
        self.cfg, self.stage, self.part, self.total, self.unit = cfg, stage, part, max(1, int(total)), unit
        self.path = os.path.join(sdir(cfg, "progress"), f"{stage}__{part.replace(':', '-').replace('/', '-')}.json")
        old = read_json(self.path, {}) or {}
        self.t0 = time.time()
        self.d0 = 0
        self.rec = {"stage": stage, "part": part, "total": self.total, "unit": unit, "done": old.get("done", 0) if old.get("total") == self.total else 0,
                    "state": "running", "started": old.get("started") or time.strftime("%Y-%m-%d %H:%M"), "job": os.environ.get("SLURM_JOB_ID", "")}
        self.d0 = self.rec["done"]
        self._last = 0
        self._write()

    def _write(self):
        el = time.time() - self.t0
        dn = self.rec["done"] - self.d0
        if dn > 0 and el > 30:
            rate = dn / el
            self.rec["eta_min"] = round((self.total - self.rec["done"]) / rate / 60, 1)
            self.rec["rate_per_min"] = round(rate * 60, 2)
        self.rec["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        self.rec["updated_ts"] = time.time()
        write_json(self.path, self.rec)

    def tick(self, done, note=None):
        self.rec["done"] = int(min(done, self.total))
        if note:
            self.rec["note"] = note
        if time.time() - self._last > 20:
            self._last = time.time()
            self._write()
            maybe_push(self.cfg)

    def finish(self, note=None):
        self.rec["done"] = self.total
        self.rec["state"] = "done"
        if note:
            self.rec["note"] = note
        self._write()


def mark_done(cfg, stage, part="all"):
    write_json(os.path.join(sdir(cfg, "done"), f"{stage}__{part.replace(':', '-')}.json"),
               {"stage": stage, "part": part, "finished": time.strftime("%Y-%m-%d %H:%M"), "job": os.environ.get("SLURM_JOB_ID", "")})


def is_done(cfg, stage, part="all"):
    return os.path.exists(os.path.join(cfg["run_dir"], "status", "done", f"{stage}__{part.replace(':', '-')}.json"))


def stage_state(cfg, stage, parts):
    """(fraction done 0-1, state, eta minutes or None, running parts)."""
    if not parts:
        return 1.0, "n/a", None, []
    done = sum(is_done(cfg, stage, p) for p in parts)
    running, frac_run, etas = [], 0.0, []
    pdir = os.path.join(cfg["run_dir"], "status", "progress")
    for p in parts:
        if is_done(cfg, stage, p):
            continue
        r = read_json(os.path.join(pdir, f"{stage}__{p.replace(':', '-').replace('/', '-')}.json"))
        if r:
            frac_run += r["done"] / max(1, r["total"])
            fresh = time.time() - r.get("updated_ts", 0) < 3 * 3600
            if fresh and r.get("state") == "running":
                running.append(f"{p} {r['done']}/{r['total']} {r['unit']}" + (f", ~{_hm(r['eta_min'])} left" if r.get("eta_min") is not None else ""))
                if r.get("eta_min") is not None:
                    etas.append(r["eta_min"])
    f = (done + frac_run) / len(parts)
    state = "done" if done == len(parts) else ("running" if running else ("started" if f > 0 else "waiting"))
    return f, state, (max(etas) if etas else None), running


def _hm(minutes):
    minutes = float(minutes)
    return f"{minutes / 60:.1f} h" if minutes >= 90 else f"{minutes:.0f} min"


# ==================================================================================
# GitHub push
# ==================================================================================
def _token():
    t = os.environ.get("GITHUB_TOKEN")
    if t:
        return t.strip()
    p = os.path.expanduser("~/.stair_github_token")
    if os.path.exists(p):
        return open(p).read().strip()
    return None


def maybe_push(cfg, force=False):
    P = cfg.get("progress") or {}
    if not P.get("github") or cfg["backend"] == "mock" and not os.environ.get("STAIR_PUSH_IN_MOCK"):
        if force or _due(cfg, P):
            _build(cfg)
        return
    if not (force or _due(cfg, P)):
        return
    _build(cfg)
    push(cfg)


def _due(cfg, P):
    p = os.path.join(cfg["run_dir"], "status", "last_push.json")
    last = (read_json(p, {}) or {}).get("ts", 0)
    if time.time() - last < 60 * float(P.get("every_minutes", 20)):
        return False
    write_json(p, {"ts": time.time(), "at": time.strftime("%Y-%m-%d %H:%M:%S")})
    return True


def _build(cfg):
    try:
        from .report import build_status
        build_status(cfg)
    except Exception as e:  # noqa: BLE001  (a status page must never stop a run)
        log(f"status page not rebuilt: {type(e).__name__}: {e}")


def push(cfg):
    P = cfg["progress"]
    tok = _token()
    page = os.path.join(cfg["run_dir"], "status", "page")
    if not tok or not os.path.isdir(page):
        if not tok:
            _once(cfg, "no GitHub token (~/.stair_github_token): status page kept locally in " + page)
        return False
    url = P["repo"].replace("https://", f"https://x-access-token:{tok}@")
    tmp = tempfile.mkdtemp(prefix="stair_status_")
    try:
        shutil.copytree(page, os.path.join(tmp, "w"))
        w = os.path.join(tmp, "w")
        env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
        run = lambda *a: subprocess.run(["git", *a], cwd=w, env=env, capture_output=True, text=True, timeout=120)
        run("init", "-q")
        run("checkout", "-q", "-b", P["branch"])
        run("add", "-A")
        r = run("-c", f"user.name={P.get('author_name', 'stair')}", "-c", f"user.email={P.get('author_email', 'stair@localhost')}",
                "commit", "-q", "-m", f"Status {time.strftime('%Y-%m-%d %H:%M')} ({cfg['run_name']})")
        if r.returncode:
            raise RuntimeError(r.stderr.strip()[:200])
        r = run("push", "-q", "--force", url, f"HEAD:refs/heads/{P['branch']}")
        if r.returncode:
            raise RuntimeError(r.stderr.replace(tok, "***").strip()[:300])
        write_json(os.path.join(cfg["run_dir"], "status", "last_push_ok.json"), {"at": time.strftime("%Y-%m-%d %H:%M:%S")})
        return True
    except Exception as e:  # noqa: BLE001
        _once(cfg, f"status push failed ({type(e).__name__}: {str(e).replace(tok, '***')[:300]}); run jobs/status_loop.sh on the login node")
        return False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


_SAID = set()


def _once(cfg, msg):
    if msg not in _SAID:
        _SAID.add(msg)
        log(msg)
