#!/bin/bash
# hotpotqa_samplek_lab_sweep.sh -- temperature ablation for the K = 40 weak-answerer cells, on the LAB GPU (tmux). 18 Nov 2026.
# Runs val then test for each temperature, back to back; resume-safe (rerun the same command). ~45-60 min per split per temperature.
# Usage:  source /mnt/hdd/jonathan/slr-env/bin/activate && bash lab_package/budget/hotpotqa_samplek_lab_sweep.sh 0.3 1.0 1.3
set -euo pipefail
export SLR_POOL_K=110 SLR_SHUFFLE=off
SCRIPT=/mnt/hdd/jonathan/STAIR/lab_package/budget/hotpotqa_samplek_cells.py
ROOT=/mnt/hdd/jonathan/runs/hotpotqa
for T in "$@"; do
  OUT=$ROOT/samplek_k40_t${T}; mkdir -p "$OUT"
  sha256sum "$SCRIPT" /mnt/hdd/jonathan/slr/stage-level-routing/lab_package/run_labels4.py /mnt/hdd/jonathan/data/tts_test.json \
            /mnt/hdd/jonathan/runs/config_b/retr_weak.jsonl /mnt/hdd/jonathan/runs/config_b/retr_strong.jsonl | tee -a "$OUT/inputs_sha256.txt"
  for S in val test; do
    echo "=== T=$T split=$S  $(date)"
    python3 "$SCRIPT" --runner lab --split "$S" --out "$OUT" --k 40 --temp "$T" --top-p 0.9 --seed 20261117 2>&1 | grep -v "max_new_tokens" | tee -a "$OUT/run_${S}.log"
  done
done
echo "SWEEP DONE $(date)"
