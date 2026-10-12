"""Read-only exploratory analysis on completed TRAIN examples; leaves held-out data untouched."""
import ast,gzip,json,re,sqlite3,string,time
from collections import Counter
from pathlib import Path
import numpy as np
R=Path('/data1/zmcheng/cs707_sampling40')
env={'re':re,'string':string,'Counter':Counter}
tree=ast.parse((R/'assets/hotpot_evaluate_v1.py').read_text());nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_answer','exact_match_score','f1_score']];exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official>','exec'),env)
data=json.load(gzip.open(R/'assets/AgentTTS-HotpotQA.json.gz','rt'));train=set(json.load(open(R/'assets/splits.json'))['splits']['train'])
scores={pair:np.full((len(data),40,2),np.nan) for pair in ['WW','SW']};texts={}
for g in range(4):
 p=R/'runs'/f'answers_W_{g}.sqlite'
 if not p.exists():continue
 db=sqlite3.connect(f'file:{p}?mode=ro',uri=True)
 for key,payload in db.execute("SELECT key,payload FROM results WHERE key LIKE '%:fuse:%'"):
  i,_,pair,k=key.split(':');i=int(i);k=int(k)-1
  if i not in train:continue
  text=json.loads(payload)['outputs'][0]['text'];gold=data[i]['answers'];scores[pair][i,k]=[max(env['exact_match_score'](text,a) for a in gold),max(env['f1_score'](text,a)[0] for a in gold)]
 db.close()
 print('Scored shard',g,flush=True)
report={'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'scope':'completed frozen TRAIN questions only; exploratory incomplete cohort','metric':'official HotpotQA EM/F1 on unmodified fusion text','pairs':{}}
for pair,full in scores.items():
 ids=np.flatnonzero(np.isfinite(full).all(axis=(1,2)));s=full[ids];em=s[:,:,0];f1=s[:,:,1]
 if not len(ids):continue
 small=em[:,:5].max(axis=1);large=em[:,19:].max(axis=1);ever=em.max(axis=1);earliest=em.argmax(axis=1)+1
 firstbetter=np.flatnonzero((em[:,0]==1)&(em[:,39]==0));lastbetter=np.flatnonzero((em[:,0]==0)&(em[:,39]==1))
 smallonly=np.flatnonzero((small==1)&(large==0));largeonly=np.flatnonzero((small==0)&(large==1))
 best=int(em.mean(axis=0).argmax());bestf=int(f1.mean(axis=0).argmax())
 stats={'questions':len(ids),'fixed_k_em_percent':(100*em.mean(axis=0)).tolist(),'fixed_k_f1_percent':(100*f1.mean(axis=0)).tolist(),'best_fixed_em_k':best+1,'best_fixed_em_percent':float(100*em[:,best].mean()),'per_question_em_oracle_percent':float(100*ever.mean()),'best_fixed_f1_k':bestf+1,'best_fixed_f1_percent':float(100*f1[:,bestf].mean()),'per_question_f1_oracle_percent':float(100*f1.max(axis=1).mean()),'k1_correct_k40_wrong':len(firstbetter),'k1_wrong_k40_correct':len(lastbetter),'some_k1to5_correct_no_k20to40_correct':len(smallonly),'some_k20to40_correct_no_k1to5_correct':len(largeonly),'f1_small_best_exceeds_large_best_by_atleast005':int(((f1[:,:5].max(axis=1)-f1[:,19:].max(axis=1))>=.05).sum()),'f1_large_best_exceeds_small_best_by_atleast005':int(((f1[:,19:].max(axis=1)-f1[:,:5].max(axis=1))>=.05).sum()),'ever_em_correct':int(ever.sum()),'never_em_correct':int((ever==0).sum()),'earliest_correct_k_counts':dict(sorted(Counter(earliest[ever==1].tolist()).items())),'examples':{}}
 for name,which in [('k1_better',firstbetter),('k40_better',lastbetter),('small_only',smallonly),('large_only',largeonly)]:
  examples=[]
  for ix in which[:3]:
   i=int(ids[ix]);chosen=[1,40] if name.startswith('k') else [int(em[ix,:5].argmax()+1),int(em[ix,19:].argmax()+20)]
   db=sqlite3.connect(f'file:{R}/runs/answers_W_{i%4}.sqlite?mode=ro',uri=True);out={}
   for k in chosen:
    row=db.execute('SELECT payload FROM results WHERE key=?',(f'{i}:fuse:{pair}:{k}',)).fetchone();out[str(k)]=json.loads(row[0])['outputs'][0]['text']
   db.close();examples.append({'index':i,'question':data[i]['question'],'gold':data[i]['answers'],'correct_ks':(np.flatnonzero(em[ix])+1).tolist(),'outputs':out})
  stats['examples'][name]=examples
 report['pairs'][pair]=stats
p=R/'runs'/f'partial_training_{time.strftime("%Y%m%d_%H%M%S")}.json';p.write_text(json.dumps(report,indent=2));print('REPORT',p,flush=True)
for pair,v in report['pairs'].items():print(pair,json.dumps({k:val for k,val in v.items() if k not in ['examples','fixed_k_em_percent','fixed_k_f1_percent']}),flush=True)
