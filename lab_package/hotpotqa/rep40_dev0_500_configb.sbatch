#!/bin/bash
# rep40_dev0_500_configb.sbatch -- CONTROL: HotpotQA "AgentTTS (reproduced)" on dev[0:500] with the Config B store (full pool,
# file order) -- same questions as the 21 Oct 74.6, same retrieval setting as the 5 Nov 76.8 -> separates "questions" from "pool". DRAFT v1.0 2026-11-06 --
# submit only after Jonathan's go. Decided 4 Nov 12:55 (the 21 Oct run, 74.6, used dev[0:500]).
# Method = our rep40.py v1.1 (one guarded line, see below) (their prompts transcribed; Llama-3.2-3B x40 at T 0.9 + fuser at T 0.3, on the 72B
# strong-retrieval selections) -- the re-implementation, not their archon_main.py (2Wiki Row C ran their script).
# Inputs: ~/slr/data/tts_test.json (7,405; sha f65d5813...) and the HotpotQA grid's retr_strong.jsonl (all 7,405 instances;
# the cluster copy armC_72b is gone -> copied from the lab's config_b, see the scp step). rep40 loops over every key in the
# retrieval store, so a FILTERED store holding only keys 500-999 is written first; nothing else in rep40 changes.
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/hotpotqa/rep40_dev0_500_configb.sbatch
# v1.1 (5 Nov): the v1.0 run is VOID -- rep40 resolved the lab store's reference numbers against the CLUSTER pool
# (run_labels.py: 30, shuffled) while they index the LAB pool (run_labels_agenttts.py: 110, no shuffle); every
# question got the wrong passages (32.6 / 33.1). Fix: rep40_store_from_config_b.py rebuilds the lab pool, checks every
# selected title against the store text, and writes the PASSAGE TEXTS; rep40.py v1.1 uses a list value directly.
# Before resubmitting, move the void output aside once:  mv $OUT ${OUT}_VOID_pool_mismatch_4Nov
#SBATCH --job-name=rep40_0_500_cb
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=rep40_0_500_cb.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SRC_STORE=$HOME/slr/hotpotqa_stores/retr_strong.jsonl                      # scp'd from lab /mnt/hdd/jonathan/runs/config_b/retr_strong.jsonl
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/rep40_dev0_500_configb
PY=$HOME/slr/slr-env/bin/python
REP=$HOME/slr/stage-level-routing/lab_package
[ -f "$SRC_STORE" ] || { echo "missing $SRC_STORE -- copy config_b/retr_strong.jsonl from the lab first"; exit 1; }
[ -e "$OUT/rep40.jsonl" ] && { echo "REFUSING: $OUT/rep40.jsonl exists"; exit 1; }
mkdir -p "$OUT"; cd "$REP"
sha256sum "$SRC_STORE" $HOME/slr/data/tts_test.json rep40.py $HOME/slr/STAIR/lab_package/hotpotqa/rep40_store_from_config_b.py $HOME/slr/run_labels_agenttts.py | tee "$OUT/inputs_sha256.txt"
# retrieval store with passage texts (lab pool rebuilt + title-checked), see rep40_store_from_config_b.py
SLR_POOL_K=110 SLR_SHUFFLE=off $PY $HOME/slr/STAIR/lab_package/hotpotqa/rep40_store_from_config_b.py \
    --data $HOME/slr/data/tts_test.json --store "$SRC_STORE" --out "$OUT" --range 0 500
cat "$OUT/store_manifest.json"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY rep40.py --out "$OUT" --save "$OUT/rep40.jsonl" 2>&1 | grep -v "max_new_tokens\|clean_up_tokenization\|torch_dtype"
$PY -c "import json;ids=sorted(json.loads(l)['idx'] for l in open('$OUT/rep40.jsonl'));print('rows',len(ids),'idx',ids[0],'-',ids[-1]);assert ids==list(range(0,500))"
echo "REP40 dev[0:500] CONTROL DONE -> $OUT   (score with the standard lenient ruler, as the 21 Oct 74.6)"
