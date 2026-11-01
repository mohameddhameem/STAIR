#!/bin/bash
# twowiki_determinism.sbatch -- the §8 determinism spot-check for the 6,000 label job, run ALONGSIDE the array (decided by
# Jonathan 1 Nov 2026, option B). DRAFT v1.0 2026-11-01 -- submit only after Jonathan's go.
# What: the first 5 instances of chunk test600_00, all four phases, run TWICE into two separate folders with the same
# runner, env and settings as twowiki_labels6000.sbatch; then every store entry (key -> value) is compared between runs.
# PASS = identical outputs for all 5 x 4 calls. FAIL = list of differing keys -> stop the array before the train chunks.
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/twowiki/twowiki_determinism.sbatch
#SBATCH --job-name=twowiki_det5
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=twowiki_det5.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export SLR_POOL_K=110
export SLR_SHUFFLE=off
CHUNKS=$HOME/slr/twowiki_pools6000/chunks
ROOT=/common/scratch/users/d/dohyung.ahn.2025/runs/twowiki/determinism5
RUNNER=$HOME/slr/run_labels_agenttts.py
PY=$HOME/slr/slr-env/bin/python
source $HOME/slr/slr-env/bin/activate
if [ -e "$ROOT" ]; then echo "REFUSING: $ROOT exists"; exit 1; fi
mkdir -p "$ROOT"; cd $HOME/slr
# the 5 instances = first 5 records of the hash-checked chunk test600_00 (records copied unchanged)
WANT=$($PY -c "import json;print(json.load(open('$CHUNKS/chunks_manifest.json'))['chunks']['test600_00']['sha256'])")
GOT=$(sha256sum "$CHUNKS/test600_00.json" | cut -d' ' -f1)
[ "$WANT" = "$GOT" ] || { echo "CHUNK HASH MISMATCH: want $WANT got $GOT"; exit 1; }
$PY -c "import json;json.dump(json.load(open('$CHUNKS/test600_00.json'))[:5],open('$ROOT/det5.json','w'),ensure_ascii=False)"
DATA=$ROOT/det5.json
sha256sum "$RUNNER" "$DATA" | tee "$ROOT/inputs_sha256.txt"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

for RUN in run_a run_b; do
  mkdir -p "$ROOT/$RUN"
  for PH in retr_weak retr_strong qa_weak qa_strong; do
    echo "=== $RUN phase $PH $(date -Iseconds)"
    $PY "$RUNNER" --data "$DATA" --out "$ROOT/$RUN" --phase $PH
  done
done

# compare every store entry {"k": key, "v": value} between the two runs
$PY - "$ROOT" <<'EOF'
import glob, json, os, sys
root = sys.argv[1]
def store(p): return {json.loads(l)["k"]: json.loads(l)["v"] for l in open(p) if l.strip()}
names = sorted(os.path.basename(p) for p in glob.glob(os.path.join(root, "run_a", "*.jsonl")))
assert names, "no store files found in run_a"
total = diff = 0; bad = []
for n in names:
    a, b = store(os.path.join(root, "run_a", n)), store(os.path.join(root, "run_b", n))
    assert a.keys() == b.keys(), f"{n}: different keys"
    for k in a:
        total += 1
        if a[k] != b[k]:
            diff += 1; bad.append(f"{n}:{k}")
res = {"stores": names, "entries": total, "differing": diff, "differing_keys": bad, "verdict": "PASS" if diff == 0 else "FAIL"}
json.dump(res, open(os.path.join(root, "determinism_result.json"), "w"), indent=1)
print(json.dumps(res, indent=1))
EOF
