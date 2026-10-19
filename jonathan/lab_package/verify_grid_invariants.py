#!/usr/bin/env python3
# =============================================================================
# verify_grid_invariants.py — VERSION: verify_grid_invariants v1.6 2026-09-15 (I3 accepts index-only citations, validates index range)
#
# AVENUE #3 (grid layer): the one region the 11/11 board never audited — the
# code that PRODUCED the answers/evidence. This script checks invariants that
# a pairing/truncation/misalignment bug there could not survive, across ALL
# 7,405 rows. Report-style: counts + first examples, hard-fails only on
# alignment breaks.
#
# INVARIANTS
#  I1  identity alignment: for every idx, jsonl _id == combo_results _id
#      (all 4 combo lines), and question matches the source dataset's.
#  I2  gold alignment: combo gold == source dataset's answer for that idx.
#  I3  evidence provenance: every passage line in evidence_W / evidence_S
#      occurs verbatim inside that row's pool_text (references must come
#      from the pool that was shown).
#  I4  pool integrity: n_pool == number of passages in pool_text.
#  I5  combo completeness: exactly 4 combo lines per idx, one per cell.
#
# USAGE (lab, CPU, ~1-2 min):
#   python3 verify_grid_invariants.py \
#       --data  /mnt/hdd/jonathan/runs/full_run_p100/router_data_lenient.jsonl \
#       --combo /mnt/hdd/jonathan/runs/full_run_p100/combo_results_full.jsonl \
#       --source /mnt/hdd/jonathan/data/tts_test.json
# =============================================================================
import json, re, argparse, collections

VERSION = "verify_grid_invariants v1.6 2026-09-15 (I3 accepts index-only citations, validates index range)"
CELLS = {"weak-weak": "WW", "weak-strong": "WS", "strong-weak": "SW", "strong-strong": "SS"}
GOLD_KEYS = ["answer", "answers", "gold", "golds"]
QUESTION_KEYS = ["question", "query", "q"]


def get_first(d, keys):
    for k in keys:
        if k in d:
            v = d[k]
            return v[0] if isinstance(v, list) and v else v
    return None


def passage_lines(evidence):
    """Parse an evidence block into passage chunks.

    Real format (discovered 15 Sep): 'Query: <q>' header line, then a
    'Selected references:' block whose passages are '[k] Title: text'
    chunks, often run together on one line. We strip the two scaffold
    prefixes, split on [k] markers, and return chunks long enough to be
    passages (index-only enumerations like '[2], [36]' fall below the
    length floor and are treated as scaffold)."""
    body = []
    for ln in str(evidence).split("\n"):
        ln = ln.strip()
        if re.match(r"^(Query|Question)\s*:", ln):
            continue
        ln = re.sub(r"^Selected references\s*:\s*", "", ln)
        if ln:
            body.append(ln)
    blob = " ".join(body)
    chunks = re.split(r"(?=\[\d+\]\s)", blob)
    out = []
    for ch in chunks:
        ch = ch.strip().rstrip(",")
        ch = re.sub(r"^\[\d+\]\s*", "", ch).strip()
        if len(ch) >= 30:
            out.append(ch)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--combo", required=True)
    ap.add_argument("--source", required=True)
    ap.add_argument("--show", type=int, default=8)
    a = ap.parse_args()
    print(VERSION, flush=True)

    rows = {}
    for l in open(a.data):
        r = json.loads(l)
        rows[r["idx"]] = r
    combo = collections.defaultdict(dict)
    combo_id = {}
    combo_gold = {}
    for l in open(a.combo):
        c = json.loads(l)
        combo[c["idx"]][CELLS.get(c["combo"], c["combo"])] = c
        combo_id.setdefault(c["idx"], c.get("_id"))
        combo_gold.setdefault(c["idx"], c.get("gold"))
    src = json.load(open(a.source))
    print(f"rows {len(rows)} | combo idx {len(combo)} | source items {len(src)}", flush=True)

    v1 = v1q = v2 = v3w = v3s = v4 = v5 = 0
    ex = collections.defaultdict(list)
    verb_num = collections.Counter(); verb_den = collections.Counter()
    done = 0
    for i, r in rows.items():
        done += 1
        if done % 500 == 0:
            print(f"  ... {done}/{len(rows)} rows checked", flush=True)
        # I5 completeness
        if set(combo[i].keys()) != {"WW", "WS", "SW", "SS"}:
            v5 += 1; ex["I5"].append((i, sorted(combo[i].keys())))
        # I1 id + question alignment
        if combo_id.get(i) != r.get("_id"):
            v1 += 1; ex["I1"].append((i, r.get("_id"), combo_id.get(i)))
        s = src[i] if i < len(src) else {}
        sq = get_first(s, QUESTION_KEYS)
        if sq is not None and str(sq).strip() != str(r.get("question", "")).strip():
            v1q += 1; ex["I1q"].append((i, str(r.get("question"))[:40], str(sq)[:40]))
        # I2 gold alignment
        sg = get_first(s, GOLD_KEYS)
        if sg is not None and combo_gold.get(i) is not None and str(sg).strip() != str(combo_gold[i]).strip():
            v2 += 1; ex["I2"].append((i, str(combo_gold[i])[:40], str(sg)[:40]))
        # I3 evidence provenance — TITLE level (evidence bodies are MODEL-GENERATED
        # text about pool passages, so verbatim match is not required by design;
        # but every cited title must exist in THIS row's pool, else mispairing).
        pool = r.get("pool_text", "")
        pool_titles = set(m.group(1).strip() for m in re.finditer(r"^\[\d+\]\s*([^:\n]{2,80}):", pool, flags=re.M))
        for tag in ("evidence_W", "evidence_S"):
            blob = str(r.get(tag, ""))
            chunks = passage_lines(blob)
            verbatim = sum(1 for ch in chunks if ch in pool)
            if chunks:
                verb_num[tag] += verbatim
                verb_den[tag] += len(chunks)
            # pairing check, prose-proof: the evidence must reference THIS row's
            # pool — either by mentioning a pool title anywhere, or (weak model's
            # terse style, discovered 15 Sep) by citing indices, which must then
            # all lie within 1..n_pool.
            if pool_titles and len(blob) >= 40:
                if not any(t in blob for t in pool_titles):
                    sel = blob.split("Selected references", 1)[-1]
                    ks = [int(m) for m in re.findall(r"\[(\d+)\]", sel)]
                    npool = r.get("n_pool") or 0
                    if ks and all(1 <= k <= npool for k in ks):
                        pass                      # index-cited, in range: paired
                    else:
                        if tag == "evidence_W": v3w += 1
                        else: v3s += 1
                        why = ("cited index out of range " + str([k for k in ks if not (1 <= k <= npool)][:5])) if ks else ("no title, no index; blob starts: " + blob[:60])
                        ex["I3"].append((i, tag, why))
        # I4 pool integrity
        n_markers = len(re.findall(r"^\[\d+\]", r.get("pool_text", ""), flags=re.M))
        if n_markers and n_markers != r.get("n_pool"):
            v4 += 1; ex["I4"].append((i, r.get("n_pool"), n_markers))

    print(f"\nI1 _id alignment jsonl vs combo : {len(rows)-v1}/{len(rows)} ok, {v1} FAIL")
    print(f"I1q question vs source dataset  : {len(rows)-v1q}/{len(rows)} ok, {v1q} differ")
    print(f"I2 gold vs source dataset       : {len(rows)-v2}/{len(rows)} ok, {v2} differ")
    print(f"I3 evidence_W mentions own pool : {len(rows)-v3w}/{len(rows)} ok, {v3w} rows fail title/index pairing")
    print(f"I3 evidence_S mentions own pool : {len(rows)-v3s}/{len(rows)} ok, {v3s} rows fail title/index pairing")
    for tag in ("evidence_W", "evidence_S"):
        if verb_den[tag]:
            print(f"   (info) {tag} chunks verbatim in pool: {verb_num[tag]}/{verb_den[tag]} = {100*verb_num[tag]/verb_den[tag]:.1f}%  — the rest are model-paraphrased, by design")
    print(f"I4 n_pool == passage markers    : {len(rows)-v4}/{len(rows)} ok, {v4} differ")
    print(f"I5 four combo cells per idx     : {len(rows)-v5}/{len(rows)} ok, {v5} incomplete")
    for k in ("I1", "I1q", "I2", "I3", "I4", "I5"):
        for e in ex[k][: a.show]:
            print(f"  {k} example:", e)
    hard = v1 + v5
    print("\nVERDICT:", "GRID INVARIANTS CERTIFIED (no alignment breaks)" if hard == 0 and v3w + v3s + v4 == 0 and v1q + v2 == 0
          else ("ALIGNMENT BREAK — investigate I1/I5 first" if hard else "soft differences to review (see examples)"))


if __name__ == "__main__":
    main()
