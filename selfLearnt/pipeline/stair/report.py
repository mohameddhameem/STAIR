"""Status page and the comparison of all workstreams (markdown + CSV + PNG charts).

build_status(cfg) writes runs/<run>/status/page/ (pushed to GitHub by progress.py):
  README.md   stages, progress, estimated finish, results so far, training, AgentTTS search
  results.csv the test-question results of every workstream, one row per policy
  charts/     accuracy vs compute, accuracy vs AgentTTS budget, training curves, AgentTTS search, progress"""
import csv
import datetime as dt
import json
import os
import shutil
import time

from .common import read_json
from .config import ft_keys, workstreams
from .progress import STAGE_NAMES, stage_state

# reference categorical palette (validated order); workstreams keep their slot whatever subset is shown
WS_COLORS = {"ws1": "#2a78d6", "ws2": "#eb6834", "ws3": "#1baf7a", "ws4": "#6250d6", "ws5": "#e34948", "ws6": "#e87ba4"}
WS_MARKERS = {"ws1": "o", "ws2": "s", "ws3": "^", "ws4": "D", "ws5": "P", "ws6": "X"}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
ORDER = ["setup", "data", "ws1_grid", "ws1_search", "ws1_test", "train", "retrieve", "reach", "answer", "probe", "sample", "learn", "test", "report"]


def stage_parts(cfg, stage):
    from . import agenttts as AT
    from .probe import probe_variants
    from .states import ans_variants, ret_variants
    ws = cfg["workstreams"]
    rules = workstreams(cfg, "rules")
    if stage in ("ws1_grid", "ws1_test"):
        return AT.all_parts(cfg, stage) if "ws1" in ws else []
    if stage == "ws1_search":
        return ["all"] if "ws1" in ws else []
    if stage == "train":
        return ft_keys(cfg)
    if stage == "retrieve":
        return ret_variants(cfg) if rules else []
    if stage == "reach":
        return rules
    if stage == "answer":
        return ans_variants(cfg) if rules else []
    if stage == "probe":
        pv = probe_variants(cfg)
        return [f"{r}-{v}" for v, r in pv] + (["fit"] if pv else [])
    if stage == "sample":
        out = []
        for w in rules:
            for v in ws[w]["answerers"][1:]:
                if v not in out:
                    out.append(v)
        return out
    if stage in ("learn", "test"):
        return rules
    return ["all"]


def _now():
    try:
        from zoneinfo import ZoneInfo
        return dt.datetime.now(ZoneInfo("Asia/Singapore"))
    except Exception:  # noqa: BLE001
        return dt.datetime.utcnow() + dt.timedelta(hours=8)


def _bar(f, n=12):
    k = int(round(f * n))
    return "█" * k + "░" * (n - k) + f" {100 * f:.0f}%"


def overview(cfg):
    est = cfg.get("estimates_hours", {})
    rows, remaining, total = [], 0.0, 0.0
    for st in ORDER:
        parts = stage_parts(cfg, st)
        if not parts:
            continue
        f, state, eta, running = stage_state(cfg, st, parts)
        h = float(est.get(st, 0.5))
        total += h
        if state != "done":
            not_started = 1 - f
            remaining += max((eta or 0) / 60.0, h * not_started) if eta else h * not_started
        rows.append({"stage": st, "name": STAGE_NAMES.get(st, st), "state": state, "frac": f, "running": running, "parts": len(parts)})
    done_h = total - remaining
    return rows, (done_h / total if total else 0.0), remaining


def collect_results(cfg):
    """Test results of every workstream available so far -> list of rows."""
    out = []
    w1 = read_json(os.path.join(cfg["run_dir"], "ws1", "test_results.json")) or []
    for r in w1:
        for tag in (r["policy"] or "AgentTTS setting").split(" = "):
            ws = "ws2" if tag.startswith("Baseline 2") else "ws1"
            out.append({"ws": ws, "workstream": cfg["workstreams"].get(ws, {}).get("name", ws), "policy": f"{tag}: {r['setting']}",
                        "main": tag.startswith("Baseline"), "acc": r["acc"], "em": float(sum(x["EM"] for x in r["per_question"]) / max(1, r["n"])),
                        "units_per_q": r["units_per_q"], "n": r["n"]})
    for ws in workstreams(cfg, "rules"):
        R = read_json(os.path.join(cfg["run_dir"], ws, "results.json"))
        if not R:
            continue
        for p in R["policies"]:
            out.append({"ws": ws, "workstream": R["name"], "policy": p["policy"], "main": p["policy"].startswith("LEARNT"), "acc": p["acc"],
                        "em": p["EM"], "units_per_q": p["units_per_q"], "n": p["n"],
                        "sure_share": p.get("sure_share"), "right_when_sure": p.get("right_when_sure"), "vs_random_pts": p.get("vs_random_pts"),
                        "kind": p["kind"]})
    b1 = [r for r in out if r["policy"].startswith("Baseline 1")]
    if b1:
        a0, c0 = b1[0]["acc"], b1[0]["units_per_q"]
        for r in out:
            r["vs_b1_acc_pts"] = 100 * (r["acc"] - a0)
            r["vs_b1_compute"] = r["units_per_q"] / c0 - 1 if c0 else None
    return out


def _fmt(v, kind):
    if v is None:
        return "-"
    if kind == "pct":
        return f"{100 * v:.1f}%"
    if kind == "pts":
        return f"{v:+.1f}"
    if kind == "rel":
        return f"{100 * v:+.0f}%"
    if kind == "f1":
        return f"{v:.1f}"
    return f"{v:.2f}"


# ==================================================================================
def build_status(cfg):
    from .config import full_run
    cfg = full_run(cfg)
    page = os.path.join(cfg["run_dir"], "status", "page")
    os.makedirs(os.path.join(page, "charts"), exist_ok=True)
    rows, frac, rem_h = overview(cfg)
    now = _now()
    fin = now + dt.timedelta(hours=rem_h)
    L = [f"# STAIR: self-learnt switching, run `{cfg['run_name']}`", "",
         f"**Updated {now:%a %d %b %Y, %H:%M} (Singapore time).** Overall {_bar(frac)}. "
         + (f"**Estimated finish: {fin:%a %d %b, %H:%M}** (about {rem_h:.0f} h left; planned stage times, corrected by measured speed as stages run)." if rem_h > 0.05 else "**All stages done.**"),
         "", "Branch `selfLearnt-runs` holds only this status page and is replaced at every update. Code: branch `selfLearnt`.", "",
         "## Stages", "", "| # | Stage | State | Progress | Running now |", "|---|---|---|---|---|"]
    icons = {"done": "✅ done", "running": "🔄 running", "started": "⏸ started", "waiting": "⏳ waiting"}
    for i, r in enumerate(rows, 1):
        L.append(f"| {i} | {r['name']} | {icons.get(r['state'], r['state'])} | {_bar(r['frac'], 10)} | {'; '.join(r['running']) or ''} |")
    res = collect_results(cfg)
    L += ["", f"## Results so far ({max([r['n'] for r in res], default=0) or 'the'} AgentTTS test questions)", ""]
    if res:
        L += ["Accuracy = lenient (the team's standard ruler, stair_common.score). EM = official exact match. Cost = AgentTTS units per "
              "question, measured tokens (stair_common.cost). Baseline 1 = AgentTTS with the setting its own search chose.", "",
              "| Experiment | Policy | Lenient | EM | AgentTTS units / question | vs Baseline 1: accuracy (points) | compute |",
              "|---|---|---|---|---|---|---|"]
        for r in res:
            if r["ws"] in ("ws1", "ws2") or r.get("kind") in ("rulebook", "fixed"):
                b = "**" if r["main"] else ""
                L.append(f"| {cfg['workstreams'][r['ws']].get('exp_id', r['ws'])} | {b}{r['policy']}{b} | {b}{_fmt(r['acc'], 'pct')}{b} | {_fmt(r['em'], 'pct')} | "
                         f"{_fmt(r['units_per_q'], 'f1')} | {_fmt(r.get('vs_b1_acc_pts'), 'pts')} | {_fmt(r.get('vs_b1_compute'), 'rel')} |")
        L.append("")
        for ws in workstreams(cfg, "rules"):
            R = read_json(os.path.join(cfg["run_dir"], ws, "results.json"))
            if R:
                lr = [p for p in R["policies"] if p["policy"].startswith("LEARNT")][0]
                L.append(f"- **{ws.upper()}** learnt rulebook: {len(R['accepted'])} of {R['tried']} changes accepted; vs random escalation of the same size "
                         f"{lr['vs_random_pts']:+.1f} points ({lr['ci_random'][0]:+.1f} to {lr['ci_random'][1]:+.1f}); Sure on {100 * lr['sure_share']:.0f}% "
                         f"of questions, right {_fmt(lr['right_when_sure'], 'pct')} of the time when Sure.")
    else:
        L.append("No test results yet.")
    # AgentTTS search
    sr = read_json(os.path.join(cfg["run_dir"], "ws1", "search_result.json"))
    if sr:
        c = sr["chosen"]
        L += ["", "## AgentTTS search (AL-01)", "",
              f"{sr['trials']} trials within budget {sr['budget_limit'] if 'budget_limit' in sr else cfg['workstreams'].get('ws1', {}).get('budget', 900)}. Chosen: **{cfg['models'][c[0]]['label']} x{c[1]} retrieval samples, "
              f"{cfg['models'][c[2]]['label']} x{c[3]} answer samples** (Gen_EM {sr['score_on_search_questions']['Gen_EM']:.3f} on the 50 search questions, "
              f"budget {sr['budget']:.0f})."]
    # training
    tl = []
    for k in ft_keys(cfg):
        st = read_json(os.path.join(cfg["run_dir"], "adapters", k, "train_state.json"))
        if st and st.get("curve"):
            b = st.get("chosen") or st.get("best") or st["curve"][-1]
            tl.append(f"| {cfg['models'][k]['label']} | {st['curve'][-1]['epoch']:.2f} | {b['epoch']:.2f} | {b['val_loss']:.4f} | "
                      f"{'chosen' + (' (stopped early)' if b.get('stopped_early') else '') if st.get('chosen') else 'training'} |")
    if tl:
        L += ["", "## Fine-tuning (early stopping on validation loss)", "",
              "| Model | Epochs run | Best epoch | Best validation loss | State |", "|---|---|---|---|---|"] + tl
    charts = make_charts(cfg, res, rows, os.path.join(page, "charts"))
    if charts:
        L += ["", "## Charts", ""] + [f"![{t}](charts/{f})" for f, t in charts]
    open(os.path.join(page, "README.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    with open(os.path.join(page, "results.csv"), "w", newline="", encoding="utf-8") as f:
        cols = ["ws", "workstream", "policy", "acc", "em", "units_per_q", "vs_b1_acc_pts", "vs_b1_compute", "n"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in res:
            w.writerow(r)
    return page


# ==================================================================================
def make_charts(cfg, res, stages, out):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:  # noqa: BLE001
        return []
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                         "axes.titlecolor": INK, "axes.titlesize": 11, "axes.titleweight": "bold", "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    made = []

    def style(ax):
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)

    pts = [r for r in res if r["ws"] in ("ws1", "ws2") or r.get("kind") in ("rulebook",) or (r.get("kind") == "fixed" and "largest" in r["policy"])]
    for xk, xl, fn in (("units_per_q", "AgentTTS units per question, measured tokens (log scale)", "accuracy_vs_compute.png"),):
        if not pts:
            break
        fig, ax = plt.subplots(figsize=(7.5, 5.4), dpi=110)
        style(ax)
        seen = set()
        for r in pts:
            if not r.get(xk):
                continue
            ws = r["ws"]
            lab = cfg["workstreams"].get(ws, {}).get("name", ws) if ws not in seen else None
            seen.add(ws)
            ax.scatter(r[xk], 100 * r["acc"], s=70 if r["main"] else 36, color=WS_COLORS.get(ws, INK2), marker=WS_MARKERS.get(ws, "o"),
                       edgecolor="#fcfcfb", linewidth=1.5, label=lab, zorder=3)
            if r["main"]:
                ax.annotate(ws.upper(), (r[xk], 100 * r["acc"]), xytext=(6, 4), textcoords="offset points", color=INK, fontsize=8, fontweight="bold")
        ax.set_xscale("log")
        ax.set_xlabel(xl)
        ax.set_ylabel("Lenient accuracy on the test questions (%)")
        ax.set_title("Lenient accuracy against compute, all experiments", loc="left")
        ax.legend(frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, fn))
        plt.close(fig)
        made.append((fn, "Accuracy against compute"))
    # training curves
    curves = []
    for k in ft_keys(cfg):
        s = read_json(os.path.join(cfg["run_dir"], "adapters", k, "train_state.json"))
        if s and s.get("curve"):
            curves.append((k, s["curve"]))
    if curves:
        fig, ax = plt.subplots(figsize=(8, 3.8), dpi=110)
        style(ax)
        pal = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#e87ba4", "#008300", "#eda100"]
        for i, (k, c) in enumerate(curves):
            ax.plot([p["epoch"] for p in c], [p["val_loss"] for p in c], color=pal[i % len(pal)], linewidth=2, marker="o", markersize=3,
                    label=cfg["models"][k]["label"])
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation loss")
        ax.set_title("Fine-tuning: validation loss (early stopping keeps the minimum)", loc="left", fontsize=10)
        ax.legend(frameon=False, fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "training_curves.png"))
        plt.close(fig)
        made.append(("training_curves.png", "Fine-tuning: validation during training"))
    sr = read_json(os.path.join(cfg["run_dir"], "ws1", "search_result.json"))
    if sr and sr.get("history"):
        fig, ax = plt.subplots(figsize=(7, 3.4), dpi=110)
        style(ax)
        x = [h["trial"] for h in sr["history"]]
        y = [100 * h["score"]["Gen_EM"] for h in sr["history"]]
        best = [max(y[:i + 1]) for i in range(len(y))]
        ax.scatter(x, y, s=18, color="#2a78d6", alpha=0.6, label="trial")
        ax.step(x, best, where="post", color=INK, linewidth=2, label="best so far")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Gen_EM on search questions (%)")
        ax.set_title("AgentTTS planner search (workstream 1)", loc="left")
        ax.legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(out, "ws1_search.png"))
        plt.close(fig)
        made.append(("ws1_search.png", "AgentTTS search"))
    if stages:
        fig, ax = plt.subplots(figsize=(7.5, 0.32 * len(stages) + 0.8), dpi=110)
        names = [s["name"] for s in stages][::-1]
        fr = [100 * s["frac"] for s in stages][::-1]
        ax.barh(names, [100] * len(names), color=GRID, height=0.6)
        ax.barh(names, fr, color="#2a78d6", height=0.6)
        for i, f in enumerate(fr):
            ax.text(min(f + 1.5, 92), i, f"{f:.0f}%", va="center", color=INK, fontsize=8)
        ax.set_xlim(0, 100)
        ax.set_xlabel("Done (%)")
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.set_title("Progress by stage", loc="left")
        fig.tight_layout()
        fig.savefig(os.path.join(out, "progress.png"))
        plt.close(fig)
        made.insert(0, ("progress.png", "Progress by stage"))
    return made


# ==================================================================================
def final_report(cfg):
    """Stage 'report': the comparison table in results/ (and the status page)."""
    import fcntl                     # several report jobs may run at once (one per experiment): take turns
    os.makedirs(cfg["run_dir"], exist_ok=True)
    with open(os.path.join(cfg["run_dir"], "report.lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _final_report(cfg)


def _final_report(cfg):
    from .config import full_run
    cfg = full_run(cfg)
    from .progress import mark_done, maybe_push
    page = build_status(cfg)
    res_dir = os.path.join(cfg["run_dir"], "results")
    os.makedirs(res_dir, exist_ok=True)
    for f in ("README.md", "results.csv"):
        shutil.copy(os.path.join(page, f), os.path.join(res_dir, "comparison.md" if f == "README.md" else f))
    if os.path.isdir(os.path.join(res_dir, "charts")):
        shutil.rmtree(os.path.join(res_dir, "charts"))
    shutil.copytree(os.path.join(page, "charts"), os.path.join(res_dir, "charts"))
    json.dump({"finished": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(res_dir, "report_done.json"), "w"))
    from .export import export
    export(cfg)
    mark_done(cfg, "report")
    maybe_push(cfg, force=True)
