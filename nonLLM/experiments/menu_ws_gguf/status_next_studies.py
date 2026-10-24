import argparse,json
from pathlib import Path
import next_studies as study
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=study.runtime.WORK/'runs/nonllm_common_ws_next_v1');args=p.parse_args();out=args.out
print(json.dumps({'status':json.loads((out/'status.json').read_text()),'folds':[{'fold':i,'complete':(out/'folds'/str(i)/'complete.json').exists(),'progress':json.loads((out/'folds'/str(i)/'progress.json').read_text()) if (out/'folds'/str(i)/'progress.json').exists() else None} for i in range(5)]},indent=2))
