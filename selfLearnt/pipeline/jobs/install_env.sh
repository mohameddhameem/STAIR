#!/bin/bash
# Create the Python environment once, on the login node (needs internet). Takes ~10-15 minutes.
#   cd STAIR/selfLearnt/pipeline && bash jobs/install_env.sh
set -e
cd "$(dirname "$0")/.."
source jobs/cluster.env
[ -f jobs/cluster.local.env ] && source jobs/cluster.local.env   # your own settings, never committed
CONDA_BIN="${CONDA_EXE:-$(command -v conda || true)}"      # remembered before `module purge` may hide it
module purge 2>/dev/null || true
# A Python 3.10-3.12 for the environment, first found of: the cluster module, one already on PATH, a private conda copy.
PY=""
if module load "$PY_MODULE" 2>/dev/null; then
  PY=$(command -v python3)
  echo "Python from module $PY_MODULE"
else
  echo "module $PY_MODULE not found; looking for another Python 3.10-3.12"
  for c in python3.11 python3.12 python3.10; do
    if command -v "$c" >/dev/null 2>&1; then PY=$(command -v "$c"); break; fi
  done
  if [ -z "$PY" ] && [ -n "$CONDA_BIN" ]; then
    echo "creating Python 3.11 with conda in $HOME/stair-py311 (once)"
    [ -x "$HOME/stair-py311/bin/python" ] || "$CONDA_BIN" create -y -q -p "$HOME/stair-py311" python=3.11
    PY="$HOME/stair-py311/bin/python"
  fi
fi
[ -n "$PY" ] || { echo "No Python 3.10-3.12 found. Run: module spider Python ; then set PY_MODULE in jobs/cluster.local.env"; exit 1; }
"$PY" -c 'import sys; v=sys.version_info; assert (3,10)<=v[:2]<=(3,12), v; print("using Python", sys.version.split()[0], sys.executable)'
module load "${CUDA_MODULE:-CUDA/12.4.0}" 2>/dev/null || true
"$PY" -m venv "$VENV"
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
