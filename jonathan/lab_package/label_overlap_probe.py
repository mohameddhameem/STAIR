#!/usr/bin/env python3
# =============================================================================
# label_overlap_probe.py — VERSION: label_overlap_probe v1.1 2026-09-15 (per-split AUC + judge-fusion test)
#
# AXIS-4 EXPERIMENT (label-design space): can a STAGE-LOCAL signal — the
# overlap between the weak and strong retrievers' selections — predict the
# routing labels WITHOUT any downstream rollout?  Hypothesis: when both
# retrievers select the same references, weak retrieval suffices (they made
# the same decision, so the cheap one was enough).
#
# Selection extraction handles BOTH evidence dialects mapped on 15 Sep:
#   - index citations  ("Selected references: [4], [26]")
#   - title mentions   (prose or "[k] Title: ..." chunks)
# A selection set = cited in-range indices ∪ pool-indices of mentioned titles.
#
# Outputs, on lenient-resolved rows:
#   - AUC of overlap (Jaccard) predicting ret_label = W   [benchmark: the
#     zero-shot 7B judge's 0.620; majority chance = 0.5]
#   - ret-label W-rate and cell WW-rate by overlap quintile (the shape)
#   - same AUC for qa_given_W label as a secondary read
#
# USAGE (lab, CPU, ~1 min):
#   python3 label_overlap_probe.py \
#     --data /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl
# =============================================================================
import json, re, argparse, collections

VERSION = "label_overlap_probe v1.1 2026-09-15 (per-split AUC + judge-fusion test)"


def pool_title_index(pool_text):
    """{title: index} for every '[k] Title:' passage header in the pool."""
    out = {}
    for m in re.finditer(r"^\[(\d+)\]\s*([^:\n]{2,80}):", pool_text, flags=re.M):
        out[m.group(2).strip()] = int(m.group(1))
    return out


def selection_set(evidence, titles, n_pool):
    """Indices selected by this retriever: cited [k]'s (in range) + mentioned titles."""
    blob = str(evidence)
    sel = set()
    part = blob.split("Selected references", 1)[-1]
    for m in re.findall(r"\[(\d+)\]", part):
        k = int(m)
        if 1 <= k <= (n_pool or 10**9):
            sel.add(k)
    for t, k in titles.items():
        if t in blob:
            sel.add(k)
    return sel


def jaccard(a, b):
    if not a and not b:
        return None
    u = a | b
    return len(a & b) / len(u) if u else None


def auc(pos, neg):
    """Rank AUC (Mann-Whitney), no dependencies."""
    if not pos or not neg:
        return float("nan")
    allv = sorted((v, 1, i) for i, v in enumerate(pos)) + sorted((v, 0, i) for i, v in enumerate(neg))
    allv.sort(key=lambda x: x[0])
    # average ranks with tie handling
    ranks = {}
    i = 0
    while i < len(allv):
        j = i
        while j < len(allv) and allv[j][0] == allv[i][0]:
            j += 1
        r = (i + j + 1) / 2  # average 1-based rank
        for k in range(i, j):
            ranks[(allv[k][1], allv[k][2])] = r
        i = j
    rp = sum(ranks[(1, i)] for i in range(len(pos)))
    return (rp - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--scores", default="", help="router_scores.jsonl for the fusion test (optional)")
    a = ap.parse_args()
    print(VERSION, flush=True)

    rows = [json.loads(l) for l in open(a.data)]
    scored = []
    skipped = 0
    for r in rows:
        titles = pool_title_index(r.get("pool_text", ""))
        sw = selection_set(r.get("evidence_W", ""), titles, r.get("n_pool"))
        ss = selection_set(r.get("evidence_S", ""), titles, r.get("n_pool"))
        ov = jaccard(sw, ss)
        if ov is None:
            skipped += 1
            continue
        scored.append((r, ov, len(sw), len(ss)))
    print(f"rows {len(rows)} | overlap computed {len(scored)} | skipped (no selections parsed) {skipped}")
    sizes_w = sum(s for _, _, s, _ in scored) / len(scored)
    sizes_s = sum(s for _, _, _, s in scored) / len(scored)
    print(f"mean selections: weak {sizes_w:.1f} | strong {sizes_s:.1f} | mean Jaccard {sum(o for _,o,_,_ in scored)/len(scored):.3f}")

    res = [(r, ov) for r, ov, _, _ in scored if r.get("ret_label")]
    pos = [ov for r, ov in res if r["ret_label"] == "W"]
    neg = [ov for r, ov in res if r["ret_label"] == "S"]
    print(f"\nresolved rows with ret label: {len(res)} (W {len(pos)} / S {len(neg)})")
    A = auc(pos, neg)
    print(f"AUC(overlap -> ret_label W): {A:.3f}   [chance 0.500 | zero-shot 7B judge benchmark 0.620]")

    res.sort(key=lambda x: x[1])
    q = max(1, len(res) // 5)
    print("\noverlap quintile -> W-rate (ret) | WW-cell rate:")
    for k in range(5):
        chunk = res[k * q: (k + 1) * q] if k < 4 else res[4 * q:]
        if not chunk:
            continue
        wr = sum(1 for r, _ in chunk if r["ret_label"] == "W") / len(chunk)
        wwr = sum((r.get("em_lenient") or [0])[0] for r, _ in chunk) / len(chunk)
        lo, hi = chunk[0][1], chunk[-1][1]
        print(f"  Q{k+1} [{lo:.2f}-{hi:.2f}] n={len(chunk):4d} | ret W-rate {100*wr:5.1f}% | WW success {100*wwr:5.1f}%")

    qres = [(r, ov) for r, ov, _, _ in scored if r.get("qa_label_W")]
    qpos = [ov for r, ov in qres if r["qa_label_W"] == "W"]
    qneg = [ov for r, ov in qres if r["qa_label_W"] == "S"]
    print(f"\nsecondary: AUC(overlap -> qa_given_W = W): {auc(qpos, qneg):.3f}  (n={len(qres)})")

    # ---- per-split protocol (the citable version) ----------------------------
    print("\nper-split AUC (protocol-clean):")
    for sp in ("train", "val", "test"):
        p2 = [ov for r, ov in res if r.get("split") == sp and r["ret_label"] == "W"]
        n2 = [ov for r, ov in res if r.get("split") == sp and r["ret_label"] == "S"]
        if p2 and n2:
            print(f"  {sp:5s}: AUC {auc(p2, n2):.3f}  (W {len(p2)} / S {len(n2)})")

    # ---- fusion test: does overlap ADD to the 7B judge? ----------------------
    if a.scores:
        J = {}
        for l in open(a.scores):
            j = json.loads(l)
            J[j["idx"]] = j.get("p_ret_W")
        both = [(r, ov, J[r["idx"]]) for r, ov in res if r["idx"] in J and J[r["idx"]] is not None]
        if both:
            def rankify(vals):
                order = sorted(range(len(vals)), key=lambda i: vals[i])
                rk = [0.0] * len(vals)
                i = 0
                while i < len(order):
                    j = i
                    while j < len(order) and vals[order[j]] == vals[order[i]]:
                        j += 1
                    r_avg = (i + j + 1) / 2
                    for k in range(i, j):
                        rk[order[k]] = r_avg
                    i = j
                return [x / len(vals) for x in rk]
            ovs = rankify([ov for _, ov, _ in both])
            jds = rankify([pj for _, _, pj in both])
            fused = [(o + jj) / 2 for o, jj in zip(ovs, jds)]
            lab = [r["ret_label"] for r, _, _ in both]
            def auc_of(scorevec):
                p3 = [s2 for s2, l3 in zip(scorevec, lab) if l3 == "W"]
                n3 = [s2 for s2, l3 in zip(scorevec, lab) if l3 == "S"]
                return auc(p3, n3)
            print(f"\nFUSION on rows with judge scores (n={len(both)}, val+test):")
            print(f"  overlap alone : AUC {auc_of(ovs):.3f}")
            print(f"  judge alone   : AUC {auc_of(jds):.3f}")
            print(f"  rank-mean fuse: AUC {auc_of(fused):.3f}   (fusion > both alones = the signals are complementary)")

    print("\nREADING: AUC ~0.5 = axis eliminated cheaply. AUC >= ~0.58 = a zero-rollout label signal exists —")
    print("a stage-local supervision source, measured today from data already on disk.")


if __name__ == "__main__":
    main()
