#!/bin/bash
# Create the Python environment once, on the login node (needs internet). Takes ~10-15 minutes.
#   cd STAIR/selfLearnt/pipeline && bash jobs/install_env.sh
set -e
cd "$(dirname "$0")/.."
source jobs/cluster.env
module purge 2>/dev/null || true
module load "$PY_MODULE"
module load "${CUDA_MODULE:-CUDA/12.4.0}" 2>/dev/null || true
python3 -m venv "$VENV"
source "$VENV/bin/activate"
pip install --upgrade pip wheel
# vLLM brings its matching PyTorch/CUDA build: install it first so later packages do not replace torch.
pip install "vllm>=0.10.2"
pip install -r requirements.txt
pip install -e ../../common          # stair_common: team splits, scoring, cost units, results schema
pip install flash-linear-attention || echo "flash-linear-attention not installed (only speeds up Qwen3.5 0.8B training)"
python - <<'EOF'
import torch, vllm, transformers, peft
print("torch", torch.__version__, "| CUDA build", torch.version.cuda, "| vLLM", vllm.__version__, "| transformers", transformers.__version__, "| peft", peft.__version__)
EOF
echo "Environment ready: $VENV"
