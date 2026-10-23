"""Read-only progress for corrected-cache non-LLM GRPO."""
import json,argparse
from pathlib import Path
import grpo
p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=grpo.runtime.WORK/'runs/nonllm_common_ws_grpo_v1');a=p.parse_args()
status=json.loads((a.out/'status.json').read_text()) if (a.out/'status.json').exists() else {'phase':'preparing'}
complete=[]
for t in grpo.tasks():
 path=a.out/'training'/grpo.tag(t)/'complete.json'
 if path.exists():complete.append(json.loads(path.read_text()))
print(json.dumps({'status':status,'completed_streams':len(complete),'total_streams':len(grpo.tasks()),'completed_grpo':sum(r['task']['method']=='grpo' for r in complete),'total_grpo':24,'completed_exact_controls':sum(r['task']['method']=='exact' for r in complete),'total_exact_controls':12,'workers':[json.loads(p.read_text()) for p in sorted(a.out.glob('worker_*.json'))]},indent=2))
