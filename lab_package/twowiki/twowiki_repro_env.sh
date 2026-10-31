#!/usr/bin/env bash
# =============================================================================
# twowiki_repro_env.sh -- build the environment for Row A (AgentTTS's own archon_main.py on 2Wiki).
# VERSION v1.0 2026-10-31 -- DRAFT for Jonathan's line-by-line review. Nothing here touches a model or the GPU.
#
# WHY A NEW ENVIRONMENT. No lab or cluster venv has `autoawq` (their 72B retriever is loaded as AWQ through
# transformers). Their requirements.txt leaves torch / transformers / autoawq unpinned and omits bitsandbytes
# (which their Llama loader needs), so "their versions" do not exist; we install their list, drop flash-attn (listed,
# never used by archon_main.py; hour-long build), add bitsandbytes, and pin the four libraries that decide whether
# AWQ + bnb loading works. The installed versions are written to env_freeze.txt and copied into the registration.
#
# PINS (proposal; the newest set in which transformers still loads AWQ through autoawq):
#   torch 2.6.0 (cu124 wheel, runs on the CUDA 12.9 driver) · transformers 4.51.3 · autoawq 0.2.9 (final release)
#   bitsandbytes 0.45.5 · accelerate 0.33.0 (their pin)
# Usage (lab):  bash twowiki_repro_env.sh          -> /mnt/hdd/jonathan/archon-env
# =============================================================================
set -euo pipefail
ENV=/mnt/hdd/jonathan/archon-env
REPO=/mnt/hdd/jonathan/agenttts_verify          # AgentTTS clone, commit 3f0c2df
REQ=$REPO/code_others/requirements.txt

[ -f "$REQ" ] || { echo "requirements.txt not found at $REQ"; exit 1; }
[ -e "$ENV" ] && { echo "$ENV already exists -- refusing to overwrite (delete it yourself if a rebuild is intended)"; exit 1; }

python3.10 -m venv "$ENV"
"$ENV/bin/pip" install -q --upgrade pip wheel setuptools

# 1. their requirements, minus flash-attn (unused here) -- unpinned entries get the pins in step 2
grep -v -i "^flash-attn" "$REQ" | grep -v -i "^torch$" | grep -v -i "^transformers$" | grep -v -i "^autoawq$" > /tmp/req_their.txt
"$ENV/bin/pip" install -q --index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
"$ENV/bin/pip" install -q -r /tmp/req_their.txt

# 2. the four pins + bitsandbytes (missing from their list) ; numpy<2 for the bnb/awq wheels of this era
"$ENV/bin/pip" install -q "transformers==4.51.3" "accelerate==0.33.0" "autoawq==0.2.9" "bitsandbytes==0.45.5" "numpy<2" tqdm

# 3. record and self-check (no model load)
"$ENV/bin/python" -m pip freeze > "$ENV/env_freeze.txt"
"$ENV/bin/python" - <<'PY'
import torch, transformers, awq, bitsandbytes, accelerate, wandb
print("torch", torch.__version__, "| cuda", torch.version.cuda, "| cuda available", torch.cuda.is_available())
print("transformers", transformers.__version__, "| autoawq", awq.__version__, "| bitsandbytes", bitsandbytes.__version__,
      "| accelerate", accelerate.__version__, "| wandb", wandb.__version__)
PY
echo "environment built: $ENV  (versions in $ENV/env_freeze.txt -- paste the two lines above into the registration §4)"
