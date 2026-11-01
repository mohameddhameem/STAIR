#!/bin/bash
# CPU-only check that MedAgentBench's records server runs on the cluster without Docker (~20-30 min).
#   python jobs/submit_all.py --fhir-test
# The server is started twice: on the node's clock and on UTC (as in the authors' container). Date searches must find the
# same magnesium records as the records themselves show (task4_3: patient S6547257, last 24 hours before 2023-11-13T10:15Z).
#SBATCH --job-name=stair-fhir-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
cd "${SLURM_SUBMIT_DIR:-.}"
source jobs/fhir_server.sh
echo "node clock: $(date +'%Z %z')"
for FHIR_TZ in "" UTC; do
  export FHIR_TZ
  echo; echo "=========== server on ${FHIR_TZ:-the node clock}"
  fhir_start || exit 1
  echo "--- task1_1: MRN of Peter Stafford, born 1932-12-29 (expected S6534835):"
  curl -s "${FHIR_URL}Patient?given=Peter&family=Stafford&birthdate=1932-12-29" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ids = [i.get('value') for e in d.get('entry', []) for i in e['resource'].get('identifier', [])]
print('found', ids, '->', 'PASS' if 'S6534835' in ids else 'FAIL')"
  echo "--- task4_3: S6547257's magnesium records after 2023-11-11, as stored:"
  curl -s "${FHIR_URL}Observation?code=MG&patient=S6547257&_count=500&_format=json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = sorted(e['resource'].get('effectiveDateTime', '') for e in d.get('entry', []))
print(len(t), 'records in all; after 2023-11-11:', [x for x in t if x >= '2023-11-11'])"
  echo "--- the same through the date search the model used (date=ge2023-11-12T10:15:00+00:00):"
  curl -s "${FHIR_URL}Observation?code=MG&patient=S6547257&date=ge2023-11-12T10:15:00+00:00&_format=json" | python3 -c "
import sys, json
d = json.load(sys.stdin)
print('found', [e['resource'].get('effectiveDateTime') for e in d.get('entry', [])])"
  fhir_stop
done
