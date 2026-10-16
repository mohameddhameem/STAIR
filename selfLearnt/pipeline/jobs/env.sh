# Sourced by every job (from the pipeline folder). EDIT the marked lines for the SMU cluster.
# module load cuda/12.4                         # EDIT if the cluster uses environment modules
source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || true    # EDIT: where conda lives
conda activate stair                            # EDIT: the environment created from environment.yml
export CONFIG="${CONFIG:-configs/default.yaml}"
export HF_HOME="${HF_HOME:-$(python -c 'import yaml,os; print(yaml.safe_load(open(os.environ["CONFIG"]))["paths"]["project_dir"] + "/hf_cache")')}"
# export HF_TOKEN=hf_...                        # EDIT: needed once to download Llama (gated). Never commit a token.
export TOKENIZERS_PARALLELISM=false
mkdir -p logs
