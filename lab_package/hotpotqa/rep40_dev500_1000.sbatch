#!/bin/bash
# rep40_dev500_1000.sbatch -- HotpotQA "AgentTTS (reproduced)" on dev[500:1000], the slice AgentTTS's released generator
# produces (start = sample_number), so the HotpotQA row uses the SAME rule as 2Wiki's Row C. DRAFT v1.0 2026-11-04 --
# submit only after Jonathan's go. Decided 4 Nov 12:55 (the 21 Oct run, 74.6, used dev[0:500]).
# Method = our rep40.py UNCHANGED (their prompts transcribed; Llama-3.2-3B x40 at T 0.9 + fuser at T 0.3, on the 72B
# strong-retrieval selections) -- the re-implementation, not their archon_main.py (2Wiki Row C ran their script).
# Inputs: ~/slr/data/tts_test.json (7,405; sha f65d5813...) and the HotpotQA grid's retr_strong.jsonl (all 7,405 instances;
# the cluster copy armC_72b is gone -> copied from the lab's config_b, see the scp step). rep40 loops over every key in the
# retrieval store, so a FILTERED store holding only keys 500-999 is written first; nothing else in rep40 changes.
# Submit:  cd ~/slr && sbatch ~/slr/STAIR/lab_package/hotpotqa/rep40_dev500_1000.sbatch
#SBATCH --job-name=rep40_500_1000
#SBATCH --account=zhoupanresearch
#SBATCH --partition=researchshort
#SBATCH --gres=gpu:h200:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=rep40_500_1000.%j.log
set -euo pipefail
export HF_HOME=/common/scratch/users/d/dohyung.ahn.2025/hf_cache
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
SRC_STORE=$HOME/slr/hotpotqa_stores/retr_strong.jsonl                      # scp'd from lab /mnt/hdd/jonathan/runs/config_b/retr_strong.jsonl
OUT=/common/scratch/users/d/dohyung.ahn.2025/runs/hotpotqa/rep40_dev500_1000
PY=$HOME/slr/slr-env/bin/python
REP=$HOME/slr/stage-level-routing/lab_package
[ -f "$SRC_STORE" ] || { echo "missing $SRC_STORE -- copy config_b/retr_strong.jsonl from the lab first"; exit 1; }
[ -e "$OUT/rep40.jsonl" ] && { echo "REFUSING: $OUT/rep40.jsonl exists"; exit 1; }
mkdir -p "$OUT"; cd "$REP"
sha256sum "$SRC_STORE" $HOME/slr/data/tts_test.json rep40.py | tee "$OUT/inputs_sha256.txt"
# filtered retrieval store: keys 500..999 only (retrieval selections untouched)
$PY - "$SRC_STORE" "$OUT/retr_strong.jsonl" <<'PYF'
import json, sys
n = 0
with open(sys.argv[2], "w") as f:
    for l in open(sys.argv[1]):
        if not l.strip(): continue
        r = json.loads(l); i = int(str(r["k"]).split("|")[0])
        if 500 <= i < 1000:
            f.write(l if l.endswith("\n") else l + "\n"); n += 1
print("filtered store entries:", n); assert n == 500, "expected 500 retrieval entries for 500..999"
PYF
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$PY rep40.py --out "$OUT" --save "$OUT/rep40.jsonl" 2>&1 | grep -v "max_new_tokens\|clean_up_tokenization\|torch_dtype"
$PY -c "import json;ids=sorted(json.loads(l)['idx'] for l in open('$OUT/rep40.jsonl'));print('rows',len(ids),'idx',ids[0],'-',ids[-1]);assert ids==list(range(500,1000))"
echo "REP40 dev[500:1000] DONE -> $OUT   (score with the standard lenient ruler, as the 21 Oct 74.6)"
