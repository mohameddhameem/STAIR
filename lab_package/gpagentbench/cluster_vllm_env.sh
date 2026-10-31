#!/usr/bin/env bash
# cluster_vllm_env.sh -- build a vLLM serving environment on the SMU cluster for the GPAgentBench pilot (and any
# other vLLM job). DRAFT v1.0 2026-10-31 for Jonathan's review. Login node, no GPU; ~10 min.
# Pins = the lab's known-good serving stack (vllm-env2, used for the HotpotQA grid and the MedAgentBench 72B server):
#   vLLM 0.19.0 (pulls torch 2.10.0+cu128 and a matching transformers); the H200 driver supports CUDA 12.8.
# Usage:  bash cluster_vllm_env.sh        -> ~/slr/vllm-env   (refuses to overwrite)
set -euo pipefail
ENV=${VLLM_ENV:-$HOME/slr/vllm-env}
[ -e "$ENV" ] && { echo "$ENV exists -- refusing to overwrite"; exit 1; }
python3.10 -m venv "$ENV"
"$ENV/bin/pip" install -q --upgrade pip wheel setuptools
"$ENV/bin/pip" install -q "vllm==0.19.0" pandas openpyxl
"$ENV/bin/python" -m pip freeze > "$ENV/env_freeze.txt"
"$ENV/bin/python" - <<'PY'
import vllm, torch, transformers
print("vllm", vllm.__version__, "| torch", torch.__version__, "| cuda", torch.version.cuda, "| transformers", transformers.__version__)
PY
echo "built: $ENV  (versions above -> registration; cuda available=False is expected on the login node)"
