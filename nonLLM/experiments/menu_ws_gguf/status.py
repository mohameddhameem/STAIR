"""Print committed cache counts without modifying generation or GPU processes."""
import argparse,json,time
from pathlib import Path
import run as runtime

p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=runtime.WORK/'runs/nonllm_common_ws_gguf_v2');a=p.parse_args()
settings=json.loads((a.out/'protocol.json').read_text());counts={}
for tag,m in settings['models'].items():
 records=runtime.model_cache(a.out,tag)
 counts[tag]={'committed_calls':len(records),'expected':7405 if m['stage']=='retrieval' else 14810}
state=json.loads((a.out/'status.json').read_text())
print(json.dumps({'phase':state['phase'],'updated':time.time(),'calls':counts,'total_committed':sum(x['committed_calls'] for x in counts.values()),'total_expected':44430,'training_started':state.get('training_started',False),'scope':settings['comparison_scope']},indent=2))
