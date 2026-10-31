#!/usr/bin/env bash
# =============================================================================
# twowiki_repro_run.sh -- Row A: "AgentTTS (reproduced), their 100" on 2WikiMultiHopQA.
# VERSION v1.0 2026-10-31 -- DRAFT for Jonathan's line-by-line review. NOT to be launched before approval.
#
# WHAT IT RUNS. AgentTTS's own code_others/archon_main.py (commit 3f0c2df), unchanged, with the configuration from
# the command block at the bottom of that file ("tts", test budget 4):
#     --generate --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1
#                --generate_model_name meta-llama/Llama-3.2-3B-Instruct --generate_samples 50
# WITHOUT --test (their 500-instance 2Wiki test file was never released; without --test the script loads
# data/2wikimultihopqa/tts_test.json = their 100 pooled instances = official dev[0:100]) and WITH --examples 100
# (their default is the first 50 when --test is absent). Everything else -- prompts, temperatures, merge, fuser -- is
# theirs, inside their script.
#
# WHAT WE ADD (all outside their files; recorded in the registration §4):
#   * two import shims (shims/utils.py, shims/evaluation_2wiki.py) for modules they never published; they record
#     every output / gold list / retrieval selection to a sidecar JSONL and change no prompt, sample or model call
#   * local resolution of the gated id meta-llama/Llama-3.2-3B-Instruct to the cached unsloth re-upload of the same
#     weights (a symlink inside the HF cache) with HF_HUB_OFFLINE=1 -- their script tests `'llama' in name` and keys
#     model_size_mapping on the meta-llama id, so the id itself cannot be swapped on the command line
#   * wandb offline (their script calls wandb.init)
#   * --smoke: 2 instances x 2 samples into a separate folder, to prove the stack loads; its outputs are never reported
#
# Usage (lab, tmux):   bash twowiki_repro_run.sh --smoke        # first
#                      bash twowiki_repro_run.sh                # the run of record (refuses to overwrite)
# Scoring afterwards: twowiki_score_sidecar.py (certified rulers) -- separate script, separate review.
# =============================================================================
set -euo pipefail

ENV=/mnt/hdd/jonathan/archon-env
REPO=/mnt/hdd/jonathan/agenttts_verify
SHIMS=/mnt/hdd/jonathan/STAIR/lab_package/twowiki/shims          # this folder, pulled from branch jonathan (lab clone = STAIR)
DATA=$REPO/data/2wikimultihopqa/tts_test.json
export HF_HOME=/mnt/hdd/jonathan/hf_cache
HUB=$HF_HOME/hub
OUTROOT=/mnt/hdd/jonathan/runs/twowiki

SMOKE=0; [ "${1:-}" = "--smoke" ] && SMOKE=1
if [ $SMOKE = 1 ]; then OUT=$OUTROOT/repro_their100_SMOKE; N=2; G=2; else OUT=$OUTROOT/repro_their100; N=100; G=50; fi

# ---- 0. guards -------------------------------------------------------------------------------------------------
[ -x "$ENV/bin/python" ] || { echo "archon-env missing -- run twowiki_repro_env.sh first"; exit 1; }
[ -f "$DATA" ] || { echo "data file missing: $DATA"; exit 1; }
[ -f "$SHIMS/utils.py" ] && [ -f "$SHIMS/evaluation_2wiki.py" ] || { echo "shims missing in $SHIMS"; exit 1; }
[ -f "$REPO/code_others/archon_main.py" ] || { echo "archon_main.py missing"; exit 1; }
[ ! -e "$REPO/code_others/utils.py" ] && [ ! -e "$REPO/code_others/evaluation_2wiki.py" ] || { echo "their tree contains utils.py/evaluation_2wiki.py -- stop, provenance must be settled first"; exit 1; }
if [ $SMOKE = 0 ] && [ -e "$OUT/sidecar.jsonl" ]; then echo "REFUSING: $OUT/sidecar.jsonl exists -- the run of record has already been started"; exit 1; fi
mkdir -p "$OUT"

# their code must be exactly the GitHub commit (no local edits)
(cd "$REPO" && git rev-parse HEAD | grep -q 3f0c2df814c059d0df57a71056c3266ff02c8559) || { echo "agenttts_verify is not at commit 3f0c2df"; exit 1; }
[ -z "$(cd "$REPO" && git status --porcelain -- code_others data/2wikimultihopqa)" ] || { echo "local modifications in their code/data -- stop"; (cd "$REPO" && git status --short); exit 1; }

# the data file: 100 instances, first question = official dev[0] (checked against the registry split report)
"$ENV/bin/python" - "$DATA" <<'PY'
import json, sys, hashlib
p = sys.argv[1]; d = json.load(open(p, encoding="utf-8"))
assert len(d) == 100, f"expected 100 instances, got {len(d)}"
assert all(k in d[0] for k in ("question", "answers", "ctxs_candidate", "ctxs_gt")), d[0].keys()
print("data sha256", hashlib.sha256(open(p, "rb").read()).hexdigest(), "| n", len(d), "| pool sizes", min(len(x["ctxs_candidate"]) for x in d), "-", max(len(x["ctxs_candidate"]) for x in d))
PY

# models in the local cache (never downloaded by this script)
for m in models--Qwen--Qwen2.5-72B-Instruct-AWQ models--unsloth--Llama-3.2-3B-Instruct; do
  [ -d "$HUB/$m" ] || { echo "cached model missing: $HUB/$m"; exit 1; }
done
# resolve their gated id to the cached unsloth weights (same weights, ungated re-upload) -- symlink, offline only
if [ ! -e "$HUB/models--meta-llama--Llama-3.2-3B-Instruct" ]; then
  ln -s "$HUB/models--unsloth--Llama-3.2-3B-Instruct" "$HUB/models--meta-llama--Llama-3.2-3B-Instruct"
  echo "cache link created: meta-llama/Llama-3.2-3B-Instruct -> unsloth/Llama-3.2-3B-Instruct"
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=offline WANDB_DIR="$OUT"
export PYTHONPATH="$REPO/code_others:$SHIMS"
export STAIR_SIDECAR="$OUT/sidecar.jsonl"

# ---- 1. manifest BEFORE the run ------------------------------------------------------------------------------------
"$ENV/bin/python" -m pip freeze > "$OUT/env_freeze.txt"
cat > "$OUT/manifest_pre.json" <<EOF
{"row": "AgentTTS (reproduced), their 100", "dataset": "2wikimultihopqa", "smoke": $SMOKE,
 "agenttts_commit": "3f0c2df814c059d0df57a71056c3266ff02c8559", "script": "code_others/archon_main.py (unchanged)",
 "data": "$DATA", "data_sha256": "$(sha256sum "$DATA" | cut -d' ' -f1)",
 "shims_sha256": {"utils.py": "$(sha256sum "$SHIMS/utils.py" | cut -d' ' -f1)", "evaluation_2wiki.py": "$(sha256sum "$SHIMS/evaluation_2wiki.py" | cut -d' ' -f1)"},
 "config": {"retrieve_model": "Qwen/Qwen2.5-72B-Instruct-AWQ", "retrieve_samples": 1, "generate_model": "meta-llama/Llama-3.2-3B-Instruct (cache link -> unsloth re-upload)", "generate_samples": $G, "examples": $N, "test_flag": false},
 "env": "$ENV", "machine": "$(hostname)", "gpu": "$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1)",
 "started": "$(date -Iseconds)"}
EOF

# ---- 2. the run: their script, their arguments, from the repo root (their data path is relative) -------------------
cd "$REPO"
echo "launching archon_main.py | examples $N | generate_samples $G | log $OUT/run.log"
"$ENV/bin/python" code_others/archon_main.py --generate \
    --retrieve_model_name Qwen/Qwen2.5-72B-Instruct-AWQ --retrieve_samples 1 \
    --generate_model_name meta-llama/Llama-3.2-3B-Instruct --generate_samples $G \
    --examples $N 2>&1 | tee "$OUT/run.log"

# ---- 3. after the run ---------------------------------------------------------------------------------------------
"$ENV/bin/python" - "$STAIR_SIDECAR" "$N" <<'PY'
import json, sys, collections
rows = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")]
c = collections.Counter(r["kind"] for r in rows)
print("sidecar:", dict(c), "| expected answers:", sys.argv[2])
assert c["answer"] == int(sys.argv[2]) and c["retrieval"] == int(sys.argv[2]), "sidecar incomplete -- do not score"
PY
date -Iseconds > "$OUT/finished.txt"
echo "DONE -> $OUT   (score with twowiki_score_sidecar.py; nothing in this folder is a reportable number yet)"
