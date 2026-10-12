"""CPU-only local relay: helper delta -> serverA -> final full evaluation."""
import fcntl,json,shlex,subprocess,time
from pathlib import Path
R=Path('/ssd1/zmcheng/cs707/runs/sampling40_helper_sync');R.mkdir(parents=True,exist_ok=True)
lock=open(R/'lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
RA='/data1/zmcheng/cs707_sampling40';RB='/data1/zmcheng/cs707_sampling40_helper';PY='/home/zmcheng/miniconda3/envs/sglang518/bin/python';state=R/'offsets.json'
after=json.loads(state.read_text()) if state.exists() else [0,0,0,0]
def call(host,args):return subprocess.check_output(['ssh','-o','ConnectTimeout=15',host,shlex.join(args)],text=True,timeout=1800)
while True:
 try:
  meta=json.loads(call('SMU_B',[PY,RB+'/code/export_delta.py','--root',RB,'--after',json.dumps(after)]).splitlines()[-1])
  if meta['rows']:
   subprocess.run(['scp','SMU_B:'+RB+'/runs/transfer_delta.jsonl',str(R/'delta.jsonl')],check=True,timeout=1200)
   subprocess.run(['scp',str(R/'delta.jsonl'),'SMU_A:'+RA+'/runs/helper_delta.jsonl'],check=True,timeout=1200)
   print(call('SMU_A',[PY,RA+'/code/merge_delta.py','--root',RA,'--delta',RA+'/runs/helper_delta.jsonl']),flush=True)
   after=meta['after'];tmp=R/'offsets.tmp';tmp.write_text(json.dumps(after));tmp.replace(state)
  # Only declare completion after merged coverage is complete, not merely local done markers.
  if meta['helper_done']:
   check="import sqlite3,json; from pathlib import Path; r=Path('/data1/zmcheng/cs707_sampling40/runs'); print(json.dumps([sqlite3.connect(f'file:{r}/answers_S_{g}.sqlite?mode=ro',uri=True).execute('SELECT count(*) FROM (SELECT substr(key,1,instr(key,\":\")-1),count(*) c FROM results GROUP BY 1 HAVING c=82)').fetchone()[0] for g in range(4)]))"
   counts=json.loads(call('SMU_A',[PY,'-c',check]).splitlines()[-1])
   if sum(counts)==7405:
    mark="from pathlib import Path; r=Path('/data1/zmcheng/cs707_sampling40/runs'); [(r/f'answers_S_{g}.done').write_text('merged complete\\n') for g in range(4)]"
    call('SMU_A',[PY,'-c',mark]);print(call('SMU_A',[PY,RA+'/code/analyze.py','--root',RA]),flush=True)
    for filename in ['report.json','per_question_oracle.csv']:
     subprocess.run(['scp','SMU_A:'+RA+'/runs/'+filename,str(R/filename)],check=True)
    (R/'COMPLETE').write_text('merged and analyzed\n');break
  print(json.dumps({'time':time.strftime('%Y-%m-%d %H:%M:%S'),'helper_offsets':after,'helper_done':meta['helper_done']}),flush=True)
 except Exception as e:print('SYNC ERROR',str(e),flush=True)
 time.sleep(300)
