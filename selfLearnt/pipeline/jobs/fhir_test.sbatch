#!/bin/bash
# CPU-only check that MedAgentBench's records server runs on the cluster without Docker (~10-20 min).
#   python jobs/submit_all.py --fhir-test
#SBATCH --job-name=stair-fhir-test
#SBATCH --output=logs/%x-%j.out
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
cd "${SLURM_SUBMIT_DIR:-.}"
source jobs/fhir_server.sh
fhir_start || exit 1
echo "--- server capability statement:"
curl -s "${FHIR_URL}metadata" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('resourceType'), d.get('fhirVersion'), d.get('software',{}).get('name'), d.get('software',{}).get('version'))"
echo "--- number of patients:"
curl -s "${FHIR_URL}Patient?_summary=count" | python3 -c "import sys,json; print(json.load(sys.stdin).get('total'))"
echo "--- task1_1: MRN of Peter Stafford, born 1932-12-29 (expected S6534835):"
curl -s "${FHIR_URL}Patient?given=Peter&family=Stafford&birthdate=1932-12-29" | python3 -c "
import sys, json
d = json.load(sys.stdin)
ids = [i.get('value') for e in d.get('entry', []) for i in e['resource'].get('identifier', [])]
print('found', ids, '->', 'PASS' if 'S6534835' in ids else 'FAIL')"
fhir_stop
