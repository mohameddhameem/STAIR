import argparse,json
from pathlib import Path
import alternatives as a
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=a.runtime.WORK/'runs/nonllm_common_ws_alternatives_v1');args=p.parse_args();root=args.out
print(json.dumps({'status':json.loads((root/'status.json').read_text()),'neural_complete':len(list((root/'training').glob('*/complete.json'))),'neural_total':len(a.tasks()),'trees_complete':len(list((root/'trees').glob('*/complete.json'))),'trees_total':len(a.tree_tasks()),'workers':[json.loads(p.read_text()) for p in sorted(root.glob('worker_*.json'))]},indent=2))
