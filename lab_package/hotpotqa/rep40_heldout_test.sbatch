#!/bin/bash
# rep40_heldout_test.sbatch -- HotpotQA "AgentTTS (reproduced)" on OUR held-out test (slr-test, n = 1,481), so the router
# table (deck slide 124) has a like-for-like AgentTTS row on the same questions as every router row; the size-matched
# n = 500 (seed 20261015) is a SUBSET of these 1,481 and is read from the same output at scoring time. DRAFT v1.0
# 2026-11-04 -- submit only after Jonathan's go. Decided 4 Nov 13:38 (the 105-question comparison is withdrawn).
# Method = rep40.py UNCHANGED (the 21 Oct re-implementation: their prompts, Llama-3.2-3B x40 at T 0.9 + fuser at T 0.3,
# on the 72B strong-retrieval selections). Inputs: ~/slr/data/tts_test.json (7,405 dev records in dev order,
# sha f65d5813...), slr-test.txt (1,481 _ids, STAIR registry), ~/slr/hotpotqa_stores/retr_strong.jsonl (all 7,405).
# rep40 loops over the keys of its retrieval store, so a FILTERED store (positions of the 1,481 ids) is written first.
# ~3x the 500-run: expect ~15 h on one H200. Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/hotpotqa/rep40_heldout_test.sbatch
#SBATCH --job-name=rep40_heldout
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=24:00:00
#SBATCH --output=rep40_heldout.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SRC_STORE=$HOME/slr/hotpotqa_stores/retr_strong.jsonl
DATA=$HOME/slr/data/tts_test.json
IDS=$HOME/slr/STAIR/common/stair_common/splits/hotpotqa/slr-test.txt
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/rep40_heldout_test
PY=$HOME/slr/slr-env/bin/python
REP=$HOME/slr/stage-level-routing/lab_package
[ -f "$SRC_STORE" ] || { echo "missing $SRC_STORE"; exit 1; }
[ -e "$OUT/rep40.jsonl" ] && { echo "REFUSING: $OUT/rep40.jsonl exists"; exit 1; }
mkdir -p "$OUT"; cd "$REP"
sha256sum "$SRC_STORE" "$DATA" "$IDS" rep40.py | tee "$OUT/inputs_sha256.txt"
# positions of the 1,481 test ids in tts_test.json -> filtered retrieval store (selections untouched)
$PY - "$DATA" "$IDS" "$SRC_STORE" "$OUT" <<'PYF'
import json, sys
data, ids_f, store, out = sys.argv[1:]
pos = {r["_id"]: i for i, r in enumerate(json.load(open(data)))}
ids = [l.strip() for l in open(ids_f) if l.strip()]
want = sorted(pos[i] for i in ids)                      # KeyError here = an id not in the data file -> stop
assert len(want) == 1481 and len(set(want)) == 1481
json.dump(want, open(f"{out}/positions_slr_test.json", "w"))
n = 0
with open(f"{out}/retr_strong.jsonl", "w") as f:
    for l in open(store):
        if not l.strip(): continue
        r = json.loads(l); i = int(str(r["k"]).split("|")[0])
        if i in set(want):
            f.write(l if l.endswith("\n") else l + "\n"); n += 1
print("filtered store entries:", n); assert n == 1481, "retrieval store does not cover all 1,481 test positions"
PYF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY rep40.py --out "$OUT" --save "$OUT/rep40.jsonl" 2>&1 | grep -v "max_new_tokens\|clean_up_tokenization\|torch_dtype"
$PY -c "import json;ids=sorted(json.loads(l)['idx'] for l in open('$OUT/rep40.jsonl'));want=json.load(open('$OUT/positions_slr_test.json'));print('rows',len(ids));assert ids==want"
echo "REP40 held-out test DONE -> $OUT   (score with the standard lenient ruler; the seed-20261015 500 is a subset read)"
