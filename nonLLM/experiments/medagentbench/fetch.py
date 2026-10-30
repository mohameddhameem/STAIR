"""Fetch the pinned replay campaign without merging branches or touching GPUs."""
import argparse,hashlib,json,subprocess
from pathlib import Path
import requests

CAMPAIGN='3ac054d1d72994cab5e8927c6cdba1449371aba4'
DATASET='99260117137b09f04837a8c18d18a1107efa55ae'

def fetch(out):
    repo=Path(__file__).resolve().parents[3];ref=out/'reference';ref.mkdir(parents=True,exist_ok=True)
    subprocess.run(['git','cat-file','-e',CAMPAIGN+'^{commit}'],cwd=repo,check=True)
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',CAMPAIGN],cwd=repo,text=True).splitlines()
    selected=[p for p in paths if p.startswith(('results/medagentbench/','lab_package/medagentbench/','common/stair_common/splits/medagentbench/')) or p in ['common/TRAINING.md','common/stair_common/splits/MANIFEST.json','common/tools/medagentbench_patient_split.py']]
    hashes={}
    def save(path,raw):
        if path.exists():assert path.read_bytes()==raw, 'Refusing to replace different source artifact: '+str(path)
        else:path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw)
    for name in selected:
        raw=subprocess.check_output(['git','show',f'{CAMPAIGN}:{name}'],cwd=repo)
        save(ref/name,raw);hashes[name]=hashlib.sha256(raw).hexdigest()
    url=f'https://raw.githubusercontent.com/stanfordmlgroup/MedAgentBench/{DATASET}/data/medagentbench/test_data_v2.json'
    response=requests.get(url,timeout=60);response.raise_for_status()
    raw=response.content;save(ref/'test_data_v2.json',raw)
    sha=hashlib.sha256(raw).hexdigest()
    common=json.loads((ref/'common/stair_common/splits/MANIFEST.json').read_text())
    assert sha==common['sources']['medagentbench']['sha256']
    manifest={'jonathan_commit':CAMPAIGN,'source_hashes':hashes,'official_dataset_commit':DATASET,'official_dataset_url':url,'official_dataset_sha256':sha}
    dest=out/'download_manifest.json'
    if dest.exists():assert json.loads(dest.read_text())==manifest
    else:dest.write_text(json.dumps(manifest,indent=2))
    print('Pinned replay files:',len(selected),'official tasks:',len(response.json()),'hashes verified')

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();fetch(a.out)
