"""Workstream 3: single hidden-state probes (development Set 5: a linear probe on a model's hidden state).

For every untrained retriever / answerer of a workstream that uses probe signals: the final-layer hidden state at the last
token of (prompt + the model's own output) is read with vLLM's pooling runner; a logistic-regression probe predicts
  retrievers: the shortlist holds both gold passages;   answerers: the answer is right (AgentTTS metric).
Dev questions get out-of-fold scores (5-fold cross-fitting, so the micro-experiments never see a probe scored on its own
training rows); test questions are scored by a probe fitted on all dev questions."""
import os

import numpy as np

from . import data as D
from .common import QA_SYSTEM, RET_SYSTEM, key, log, read_jsonl, write_jsonl
from .config import serve_entry, variant, workstreams
from .engines import free_gpu, make_engine
from .progress import Tracker, mark_done
from .states import SPLITS, f_ans, f_reach, f_ret, fname, qa_prompt, ret_prompt, step_dir


def probe_variants(cfg):
    """[(variant, 'ret'|'ans')] needing probes."""
    out = []
    for ws in workstreams(cfg, "rules"):
        W = cfg["workstreams"][ws]
        sig = W.get("signals", "all")
        if sig == "all" or not isinstance(sig, list):
            continue
        if "ret_probe_pct" in sig:
            out += [(v, "ret") for v in W["retrievers"] if (v, "ret") not in out]
        if "ans_probe_pct" in sig:
            out += [(v, "ans") for v in W["answerers"] if (v, "ans") not in out]
    return out


def f_hidden(cfg, v, role):
    return os.path.join(step_dir(cfg), f"hidden_{role}_{fname(v)}.npz")


def f_probe(cfg, v, role):
    return os.path.join(step_dir(cfg), f"probe_{role}_{fname(v)}.jsonl")


def _items(cfg, v, role):
    qm = D.qmap(cfg, SPLITS)
    items = []
    if role == "ret":
        for sp in SPLITS:
            for r in read_jsonl(f_ret(cfg, sp, v)):
                items.append((f"{sp}|{r['qid']}|", sp, qm[r["qid"]], r, r["found_both"]))
    else:
        want = set()
        for ws in workstreams(cfg, "rules"):
            if v in cfg["workstreams"][ws]["answerers"]:
                for sp in SPLITS:
                    want |= {(sp, r["qid"], key(r["sel"])) for r in read_jsonl(f_reach(cfg, sp, ws))}
        A = {(r["qid"], key(r["sel"])): r for r in read_jsonl(f_ans(cfg, v))}
        for sp, q, s in sorted(want):
            a = A.get((q, s))
            if a:
                items.append((f"{sp}|{q}|{','.join(map(str, s))}", sp, qm[q], a, a["acc"]))
    return items


def extract(cfg, v, role):
    path = f_hidden(cfg, v, role)
    items = _items(cfg, v, role)
    done_ids, X = [], []
    if os.path.exists(path):
        z = np.load(path, allow_pickle=False)
        done_ids, X = list(z["ids"]), list(z["X"])
    have = set(done_ids)
    todo = [it for it in items if it[0] not in have]
    tr = Tracker(cfg, "probe", f"{role}-{v}", len(items), "outputs")
    tr.tick(len(items) - len(todo))
    if todo:
        m = variant(cfg, v)
        eng = make_engine(cfg, serve_entry(m), pooling=True)
        chunk = max(16, cfg["gpu"]["chunk"] // 2)
        for c in range(0, len(todo), chunk):
            part = todo[c:c + chunk]
            if role == "ret":
                toks = [eng.encode_full(RET_SYSTEM, ret_prompt(m, q), r["text"]) for _, _, q, r, _ in part]
            else:
                toks = [eng.encode_full(QA_SYSTEM, qa_prompt(m, q, tuple(r["sel"])), r["text"]) for _, _, q, r, _ in part]
            H = eng.hidden(toks, [{"label": y} for *_, y in part])
            done_ids += [it[0] for it in part]
            X += [np.asarray(h, np.float16) for h in H]
            np.savez(path + ".tmp.npz", ids=np.array(done_ids), X=np.stack(X))
            os.replace(path + ".tmp.npz", path)
            tr.tick(len(items) - len(todo) + c + len(part))
        free_gpu(eng)
    tr.finish()
    mark_done(cfg, "probe", f"{role}-{v}")


def fit(cfg):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    for v, role in probe_variants(cfg):
        z = np.load(f_hidden(cfg, v, role), allow_pickle=False)
        ids, X = list(z["ids"]), z["X"].astype(np.float32)
        lab = {it[0]: it[4] for it in _items(cfg, v, role)}
        keep = [i for i, x in enumerate(ids) if x in lab]
        ids, X = [ids[i] for i in keep], X[keep]
        y = np.array([lab[x] for x in ids], float)
        dev = np.array([x.startswith("dev|") for x in ids])
        mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.05, max_iter=3000))
        score = np.zeros(len(ids))
        Xd, yd, di = X[dev], y[dev], np.where(dev)[0]
        if len(set(yd)) < 2:
            score[:] = yd.mean() if len(yd) else 0.5
        else:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
            for trn, tst in skf.split(Xd, yd):
                if len(set(yd[trn])) < 2:
                    score[di[tst]] = yd[trn].mean()
                    continue
                score[di[tst]] = mk().fit(Xd[trn], yd[trn]).predict_proba(Xd[tst])[:, 1]
            score[~dev] = mk().fit(Xd, yd).predict_proba(X[~dev])[:, 1]
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(yd, score[dev]) if len(set(yd)) > 1 else float("nan")
        except Exception:  # noqa: BLE001
            auc = float("nan")
        rows = []
        for x, s in zip(ids, score):
            sp, q, sel = x.split("|")
            rows.append({"split": sp, "qid": q, "sel": [int(t) for t in sel.split(",")] if sel else None, "probe": float(s)})
        write_jsonl(f_probe(cfg, v, role), rows)
        log(f"probe {role} {v}: {len(ids)} outputs; out-of-fold AUC on dev {auc:.3f} (0.5 = no signal)")
    mark_done(cfg, "probe", "fit")
