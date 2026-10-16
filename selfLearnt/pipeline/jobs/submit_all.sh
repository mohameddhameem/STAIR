#!/bin/bash
# Submits every stage with dependencies; the two answerers run side by side (one per A100).
# Usage (from the pipeline folder):  CONFIG=configs/default.yaml bash jobs/submit_all.sh
# Assumes setup (models + HotpotQA files) is done: run jobs/00_setup.sbatch first on a node with internet.
set -e
cd "$(dirname "$0")/.."
export CONFIG="${CONFIG:-configs/default.yaml}"
s() { sbatch --parsable --export=ALL,CONFIG="$CONFIG" "$@"; }
d=$(s jobs/01_data.sbatch)
t=$(s --dependency=afterok:$d jobs/02_train.sbatch)
r=$(s --dependency=afterok:$t jobs/03_retrieve.sbatch)
a1=$(s --dependency=afterok:$r jobs/04_answer_small.sbatch)
a2=$(s --dependency=afterok:$r jobs/05_answer_big.sbatch)
l=$(s --dependency=afterok:$a1:$a2 jobs/06_learn.sbatch)
x=$(s --dependency=afterok:$l jobs/07_test.sbatch)
echo "submitted: data $d, train $t, retrieve $r, answers $a1 $a2, learn $l, test $x"
echo "follow with: squeue -u \$USER ; tail -f logs/*.out"
