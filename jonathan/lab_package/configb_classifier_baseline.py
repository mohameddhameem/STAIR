# configb_classifier_baseline.py -- NON-LLM routing baseline (CPU, read-only).
# Question: does the retrieval routing decision need an LLM at all?
# Logistic regression on simple runtime-computable features of the SAME router
# view (question + weak retriever's selections), same labels and splits as the
# LLM routers. Reports validation AUC for the capacity-curve figure.
import json, os, re, sys
sys.path.insert(0, "/mnt/hdd/jonathan/slr/stage-level-routing/lab_package")
import run_labels4 as R

B = "/mnt/hdd/jonathan/runs/config_b"
LABELS = f"{B}/ret_labels/ret_labels.jsonl"
DATA = "/mnt/hdd/jonathan/data/tts_test.json"

data = json.load(open(DATA))
retr_w = {}
for l in open(f"{B}/retr_weak.jsonl"):
    r = json.loads(l); retr_w[int(str(r["k"]).split("|")[0])] = r["v"]

WH = ("what","who","which","where","when","why","how")
def feats(i, n_sel):
    inst = data[i]
    q = inst["question"].lower()
    pool = R.trim_pool(inst)
    sel = R.parse_selection(retr_w[i], len(pool))
    txt = " ".join(pool[k-1] for k in sel).lower() if sel else ""
    qw = set(re.findall(r"[a-z0-9]+", q)) - {"the","a","an","of","is","was","are","in","and"}
    tw = set(re.findall(r"[a-z0-9]+", txt))
    ov = len(qw & tw) / max(len(qw), 1)
    return [len(q), len(sel), len(txt), ov,
            1.0 if any(q.startswith(w) for w in WH) else 0.0,
            1.0 if re.match(r"(is|are|was|were|do|does|did)\b", q) else 0.0,   # yes/no form
            1.0 if " or " in q else 0.0,                                        # comparison form
            sum(1 for w in qw if w not in tw)]                                  # question words MISSING from evidence
X, y, Xv, yv = [], [], [], []
for l in open(LABELS):
    r = json.loads(l)
    if r["split"] not in ("train","val"): continue
    f = feats(r["idx"], r["n_weak_sel"])
    t = 1 if r["ret_target"] == "W" else 0
    (X if r["split"]=="train" else Xv).append(f)
    (y if r["split"]=="train" else yv).append(t)
print(f"train {len(X)}  val {len(Xv)}")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
sc = StandardScaler().fit(X)
clf = LogisticRegression(max_iter=2000, class_weight="balanced").fit(sc.transform(X), y)
pv = clf.predict_proba(sc.transform(Xv))[:,1]
print(f"logistic-regression val AUC: {roc_auc_score(yv, pv):.4f}")
names = ["q_len","n_sel","sel_len","overlap","wh_form","yesno_form","or_form","missing_words"]
for n, c in sorted(zip(names, clf.coef_[0]), key=lambda t: -abs(t[1])):
    print(f"  {n}: {c:+.3f}")
