"""Three probes: non-linear on pool features, combined features, and a cascade."""
import json, re, math, os, numpy as np
from collections import Counter
import run_labels as R
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score

COMBOS = ["weak-weak", "weak-strong", "strong-weak", "strong-strong"]
COST = {"weak-weak": 2.0, "weak-strong": 5.7, "strong-weak": 5.7, "strong-strong": 9.4}
SS = COST["strong-strong"]

rows = {json.loads(l)["idx"]: json.loads(l) for l in open("router_data.jsonl")}
sp = json.load(open("splits.json"))["splits"]
data = json.load(open("/mnt/hdd/jonathan/data/tts_test.json"))
tr, te = sp["train"], sp["test"]

TOK = lambda s: re.findall(r"[a-z0-9]+", s.lower())
CAP = lambda s: set(re.findall(r"\b[A-Z][a-zA-Z]+", s))

def bm25(qt, docs, k1=1.5, b=0.75):
    N = len(docs); avgdl = sum(len(d) for d in docs) / N
    df = Counter()
    for d in docs: df.update(set(d))
    out = []
    for d in docs:
        tf = Counter(d); s = 0.0
        for t in qt:
            if t not in tf: continue
            idf = math.log(1 + (N - df[t] + 0.5) / (df[t] + 0.5))
            s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avgdl))
        out.append(s)
    return np.array(out)

def pool_feats(idx):
    inst = dict(data[idx]); pool = R.trim_pool(inst)
    q = inst["question"]; qt = TOK(q); qc = CAP(q)
    s = bm25(qt, [TOK(c) for c in pool]); ss = np.sort(s)[::-1]
    tot = s.sum() + 1e-9; p = s / tot
    ent = float(-np.sum(np.where(p > 0, p * np.log(p + 1e-12), 0)))
    ov = [len(qc & CAP(c)) for c in pool]
    return [ss[0], s.mean(), s.std(), ss[0]-ss[1], ss[0]-ss[min(4, len(ss)-1)],
            ss[0]/tot, ent, max(ov), float(np.mean(ov)), sum(1 for o in ov if o > 0),
            len(qt), len(qc),
            int(bool(re.search(r"\b(which|who|older|both|same|or)\b", q.lower())))]

CACHE = "probe_feats.npz"
if os.path.exists(CACHE):
    z = np.load(CACHE); Ptr, Pte, Etr, Ete = z["Ptr"], z["Pte"], z["Etr"], z["Ete"]
    print("loaded cached features")
else:
    print("computing pool features ...")
    Ptr = np.array([pool_feats(j) for j in tr]); Pte = np.array([pool_feats(j) for j in te])
    print("computing question embeddings ...")
    import torch
    from transformers import AutoTokenizer, AutoModel
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    mdl = AutoModel.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct",
            torch_dtype=torch.float16, device_map="cuda").eval()
    def embed(ix, bs=64):
        out = []
        for i in range(0, len(ix), bs):
            qs = [rows[j]["question"] for j in ix[i:i+bs]]
            enc = tok(qs, return_tensors="pt", padding=True, truncation=True,
                      max_length=64).to("cuda")
            with torch.no_grad(): h = mdl(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1)
            out.append(((h*m).sum(1)/m.sum(1)).float().cpu().numpy())
        return np.vstack(out)
    Etr, Ete = embed(tr), embed(te)
    np.savez(CACHE, Ptr=Ptr, Pte=Pte, Etr=Etr, Ete=Ete)
    print(f"cached -> {CACHE}")

y4tr = np.array([int(np.argmax(rows[j]["R"])) for j in tr])
y4te = np.array([int(np.argmax(rows[j]["R"])) for j in te])

def report(tag, pred):
    em = 100*np.mean([rows[j]["em"][p] for j, p in zip(te, pred)])
    cost = 100*np.mean([COST[COMBOS[p]] for p in pred])/SS
    mix = np.bincount(pred, minlength=4)/len(pred)
    base = sum(mix[k]*np.mean([rows[j]["em"][k] for j in te])*100 for k in range(4))
    print(f"  {tag:34s} EM {em:5.1f}%  cost {cost:5.1f}%  random {base:5.1f}%  signal {em-base:+.1f}")

print(f"\nreference:  always-ww EM 23.4 / cost 21.3   best-fixed(EM) 46.1 / 100   oracle 55.5 / 36.5")

print("\n--- 1. NON-LINEAR on pool features ---")
for cw in (None, "balanced"):
    m = HGB(max_iter=300, learning_rate=0.06,
            class_weight=cw).fit(Ptr, y4tr)
    report(f"HGB pool  cw={cw}", m.predict(Pte))

print("\n--- 2. COMBINED embeddings + pool ---")
Ctr = np.hstack([Etr, Ptr]); Cte = np.hstack([Ete, Pte])
for cw in (None, "balanced"):
    m = make_pipeline(StandardScaler(),
        LogisticRegression(max_iter=5000, C=0.1, class_weight=cw)).fit(Ctr, y4tr)
    report(f"LR combined  cw={cw}", m.predict(Cte))

print("\n--- 3. CASCADE: predict 'will weak-weak solve this?' ---")
btr = np.array([rows[j]["em"][0] for j in tr])
bte = np.array([rows[j]["em"][0] for j in te])
print(f"  positive rate: train {btr.mean():.3f}  test {bte.mean():.3f}")
for name, Xa, Xb in (("pool", Ptr, Pte), ("combined", Ctr, Cte)):
    m = HGB(max_iter=300, learning_rate=0.06).fit(Xa, btr)
    p = m.predict_proba(Xb)[:, 1]
    print(f"\n  [{name}]  AUC = {roc_auc_score(bte, p):.3f}")
    for t in (0.2, 0.3, 0.4, 0.5, 0.6):
        pred = np.where(p > t, 0, 3)          # ww if confident, else strong-strong
        report(f"    threshold {t}", pred)
