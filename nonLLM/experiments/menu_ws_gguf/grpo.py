"""Finite-action GRPO on the corrected four-model executor grid; no executor training."""
import argparse,copy,gzip,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.sparse import hstack,csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from stair_common.score import score,RULER_SHA256
from stair_common.cost import units
from stair_common.schema import validate
import run as runtime
import train as classical

RET=['qwen7','qwen72'];QA=['llama3','llama70']
SEEDS=[0,1,2];LAMBDAS=[0.,.05,.15,.4];GROUPS=[8,32]
PENALTIES=[0.,.05,.15,.4,.8,1.6]

def tasks():
 return [dict(method=m,group=g,lam=l,seed=s) for m,g in [('grpo',8),('grpo',32),('exact',0)] for l in LAMBDAS for s in SEEDS]
def tag(t):return f"{t['method']}_g{t['group']}_lambda{t['lam']}_seed{t['seed']}"
def config_tag(t):return f"{t['method']}_g{t['group']}_lambda{t['lam']}"
def direct_grid(ids,questions,rr,qq,protocol):
 correct=np.zeros((len(ids),2,2),np.float32);cost=np.zeros_like(correct)
 for i,qid in enumerate(ids):
  for r,rt in enumerate(RET):
   rec=rr[rt][qid];rcost=units(protocol['models'][rt]['params_b'],rec['prompt_tokens'],rec['generated_tokens'])
   for a,at in enumerate(QA):
    rec=qq[at][qid+'|'+rt];correct[i,r,a]=score(rec['text'],questions[qid]['answers'])['lenient']
    cost[i,r,a]=rcost+units(protocol['models'][at]['params_b'],rec['prompt_tokens'],rec['generated_tokens'])
 return correct,cost

def targets(em,cost):
 valid_ret=em.any((1,2));ret=np.where(em.astype(bool),cost,np.inf).reshape(-1,4).argmin(1)//2
 valid_qa=em.any(2);qa=np.where(em.astype(bool),cost,np.inf).argmin(2)
 return ret.astype(np.float32),valid_ret,qa.astype(np.float32),valid_qa

def transform(bundle,views):
 text=[r[0] for r in views];numeric=np.asarray([r[1] for r in views],np.float32)
 return hstack([bundle['vectorizer'].transform(text),csr_matrix(bundle['scaler'].transform(numeric))],format='csr',dtype=np.float32)

def load_source(source):
 p=json.loads((source/'protocol.json').read_text());assert p['ruler_sha256']==RULER_SHA256
 assert json.loads((source/'status.json').read_text())['phase']=='complete'
 assert set(p['models'])==set(RET+QA)
 with gzip.open(runtime.WORK/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:questions={q['_id']:q for q in json.load(f)}
 rr={m:runtime.model_cache(source,m) for m in RET};qq={m:runtime.model_cache(source,m) for m in QA}
 assert all(len(v)==7405 for v in rr.values()) and all(len(v)==14810 for v in qq.values())
 return p,questions,rr,qq

def prepare(source,out,max_epochs):
 assert not (out/'protocol.json').exists(),'Use a new output directory or --phase train/evaluate to resume'
 p,questions,rr,qq=load_source(source);out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir()
 splits={s:p['splits'][s] for s in ['train_extra','val','test']}
 assert not(set(splits['train_extra'])&set(splits['val']) or set(splits['train_extra'])&set(splits['test']) or set(splits['val'])&set(splits['test']))
 for part in ['train_extra','val']:
  em,cost=direct_grid(splits[part],questions,rr,qq,p)
  np.savez(out/'data'/f'grid_{part}.npz',correct=em,cost=cost)
 bundles={};dimensions={}
 for stage in ['retrieval','answering']:
  views={part:([classical.view(questions[i],stage) for i in splits[part]] if stage=='retrieval' else [classical.view(questions[i],stage,rr[r][i],r) for i in splits[part] for r in RET]) for part in ['train_extra','val']}
  vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=25000,sublinear_tf=True,dtype=np.float32)
  vec.fit([v[0] for v in views['train_extra']]);scaler=StandardScaler().fit(np.asarray([v[1] for v in views['train_extra']],np.float32))
  bundle={'vectorizer':vec,'scaler':scaler};bundles[stage]=bundle
  for part in views:
   x=transform(bundle,views[part]);np.save(out/'data'/f'{stage}_{part}.npy',x.toarray())
   dimensions[stage]=x.shape[1]
 joblib.dump(bundles,out/'features.joblib')
 tr=np.load(out/'data/grid_train_extra.npz');normalizer=float(tr['cost'][:,-1,-1].mean())
 retmean=np.array([np.mean([units(p['models'][r]['params_b'],rr[r][i]['prompt_tokens'],rr[r][i]['generated_tokens']) for i in splits['train_extra']]) for r in RET])
 qamean=np.array([[np.mean([units(p['models'][a]['params_b'],qq[a][i+'|'+r]['prompt_tokens'],qq[a][i+'|'+r]['generated_tokens']) for i in splits['train_extra']]) for a in QA] for r in RET])
 provenance={str(f.relative_to(source)):runtime.digest(f) for f in sorted((source/'cache').glob('*.jsonl'))}
 protocol={'source_run':str(source),'source_protocol_sha256':runtime.digest(source/'protocol.json'),'source_cache_hashes':provenance,'ruler_sha256':RULER_SHA256,'splits':splits,'models':p['models'],'dimensions':dimensions,'normalizer':normalizer,'training_ret_mean_units':retmean.tolist(),'training_qa_mean_units':qamean.tolist(),'generation':'Reuse corrected frozen K1/greedy/no-fusion GGUF executor cache; no new LLM sampling','scope':'common0.2 test500, GGUF exploratory; NOT common0.3 slr-* golden-set reproduction','actions':'Two-stage direct W/S: question+pool selects retrieval; question+paid evidence selects answering. No sunk weak calls, no unexecuted answers as input.','reward':'author_lenient - lambda * executed_route_token_units / training_mean_direct_SS_units','seeds':SEEDS,'lambdas':LAMBDAS,'groups':GROUPS,'maximum_epochs':max_epochs,'batch_size':512,'warm_start':'masked cheapest-correct BCE, <=30 epochs, best validation BCE, patience5','optimizer':'Adam lr0.002 weight_decay0.001; warm lr0.01','rl':'GRPO finite-action clipped PPO surrogate, clip0.2, two updates per rollout, KL0.01 to rollout policy, entropy0.002; group-centered/std-normalized rewards, zero variance -> zero advantage','exact_control':'Expected reward over all4 routes, same features, warm start, cost lambda and entropy regularizer','early_stopping':'Validation expected negative task reward, checked every10 epochs; patience6, min_delta1e-5; keep best. RL surrogate and validation task loss are logged separately.','selection':'Average3 seed stage probabilities; validation only operating-point selection for learned-only and fixed-fallback variants. Raw learned policies remain visible. All choices locked before test features/scoring.','test_disclosure':'Test has previously been inspected. Results descriptive; no fresh sealed-test claim.'}
 runtime.write(out/'protocol.json',protocol)
 (out/'code').mkdir()
 for name in ['grpo.py','train.py','run.py']:__import__('shutil').copy2(Path(__file__).with_name(name),out/'code'/name)
 runtime.write(out/'status.json',{'phase':'prepared','streams':len(tasks()),'maximum_epochs':max_epochs})
 print('Prepared',dimensions,'normalizer',normalizer,flush=True)

# Torch is imported only for training/tests, not for CPU-only report evaluation.
def joint_logprob(ret_logits,qa_logits):
 import torch
 r=torch.stack([torch.nn.functional.logsigmoid(-ret_logits),torch.nn.functional.logsigmoid(ret_logits)],1)
 q=torch.stack([torch.nn.functional.logsigmoid(-qa_logits),torch.nn.functional.logsigmoid(qa_logits)],2)
 return (r[:,:,None]+q).reshape(-1,4)
def normalized_advantages(rewards):
 std=rewards.std(1,keepdim=True,unbiased=False)
 return (rewards-rewards.mean(1,keepdim=True))/std.clamp_min(1e-6),std

def train_worker(out,shard,nshards):
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
 device='cuda' if torch.cuda.is_available() else 'cpu';p=json.loads((out/'protocol.json').read_text())
 x={s:{part:torch.tensor(np.load(out/'data'/f'{s}_{part}.npy',mmap_mode='r'),device=device) for part in ['train_extra','val']} for s in ['retrieval','answering']}
 grids={part:np.load(out/'data'/f'grid_{part}.npz') for part in ['train_extra','val']}
 em={part:torch.tensor(grids[part]['correct'].reshape(-1,4),device=device) for part in grids}
 costs={part:torch.tensor(grids[part]['cost'].reshape(-1,4)/p['normalizer'],device=device) for part in grids}
 yr,mr,yq,mq=targets(grids['train_extra']['correct'],grids['train_extra']['cost'])
 vy,vm,vqy,vqm=targets(grids['val']['correct'],grids['val']['cost'])
 labels={part:(torch.tensor(a,device=device),torch.tensor(b,device=device),torch.tensor(c,device=device),torch.tensor(d,device=device)) for part,(a,b,c,d) in [('train_extra',(yr,mr,yq,mq)),('val',(vy,vm,vqy,vqm))]}
 n=len(yr);work=[t for i,t in enumerate(tasks()) if i%nshards==shard];warm={}
 def model():return torch.nn.ModuleDict({'retrieval':torch.nn.Linear(p['dimensions']['retrieval'],1),'answering':torch.nn.Linear(p['dimensions']['answering'],1)}).to(device)
 def forward(net,part,b=None):
  xr=x['retrieval'][part];xq=x['answering'][part].reshape(-1,2,p['dimensions']['answering'])
  if b is not None:xr=xr[b];xq=xq[b]
  return net['retrieval'](xr).flatten(),net['answering'](xq).squeeze(-1)
 def ce(net,part,b=None):
  r,q=forward(net,part,b);y,mask,yqa,mqa=labels[part]
  if b is not None:y,mask,yqa,mqa=y[b],mask[b],yqa[b],mqa[b]
  lr=torch.nn.functional.binary_cross_entropy_with_logits(r,y,reduction='none');lq=torch.nn.functional.binary_cross_entropy_with_logits(q,yqa,reduction='none')
  # Equal weight per question; average valid QA branches within each question.
  count=mqa.sum(1).clamp_min(1);qloss=(lq*mqa).sum(1)/count
  return (lr*mask).sum()/mask.sum().clamp_min(1)+(qloss*mqa.any(1)).sum()/mqa.any(1).sum().clamp_min(1)
 def package(net):
  return {stage:{k:v.detach().cpu().numpy().copy() for k,v in head.state_dict().items()} for stage,head in net.items()}
 def state_cpu(net):return {k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
 def validation(net,lam):
  with torch.no_grad():
   r,q=forward(net,'val');probs=joint_logprob(r,q).exp();reward=em['val']-lam*costs['val']
   return {'val_loss':float(-(probs*reward).sum(1).mean()),'val_expected_accuracy':float((probs*em['val']).sum(1).mean()),'val_expected_cost_pct_ss':float((probs*costs['val']).sum(1).mean()*100)}
 def save_policy(dest,net,meta):joblib.dump({'heads':package(net),'meta':meta},dest)
 for number,t in enumerate(work):
  folder=out/'training'/tag(t);folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists():continue
  torch.manual_seed(t['seed']);gen=torch.Generator().manual_seed(t['seed']);net=model()
  if t['seed'] not in warm:
   opt=torch.optim.Adam(net.parameters(),lr=.01,weight_decay=.001);best=float('inf');bad=0;warm_curve=[]
   for ep in range(1,31):
    for b in torch.randperm(n,generator=gen).split(512):
     b=b.to(device);loss=ce(net,'train_extra',b);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
    with torch.no_grad():vl=float(ce(net,'val'));tl=float(ce(net,'train_extra'))
    warm_curve.append({'epoch':ep,'train_loss':tl,'val_loss':vl})
    if vl<best-1e-5:best=vl;best_state=state_cpu(net);best_epoch=ep;bad=0
    else:bad+=1
    if bad>=5:break
   warm[t['seed']]=(best_state,best_epoch,warm_curve)
  net.load_state_dict(warm[t['seed']][0]);warm_dir=out/'warm'/f"seed{t['seed']}";warm_dir.mkdir(parents=True,exist_ok=True)
  # Multiple shards can train a same-seed warm start deterministically; use shard-specific files.
  save_policy(warm_dir/f'shard{shard}.joblib',net,{'seed':t['seed'],'epoch':warm[t['seed']][1]})
  runtime.write(warm_dir/f'curve_shard{shard}.json',warm[t['seed']][2])
  opt=torch.optim.Adam(net.parameters(),lr=.002,weight_decay=.001);gen=torch.Generator().manual_seed(t['seed']+1000);torch.manual_seed(t['seed']+2000)
  best=validation(net,t['lam'])['val_loss'];best_epoch=0;bad=0;start=time.monotonic();resume=folder/'state_last.pt';start_epoch=1
  if not resume.exists():save_policy(folder/'best.joblib',net,{**t,'epoch':0})
  if resume.exists():
   checkpoint=torch.load(resume,map_location=device,weights_only=False);net.load_state_dict(checkpoint['model']);opt.load_state_dict(checkpoint['optimizer']);gen.set_state(checkpoint['batch_rng'].cpu());torch.set_rng_state(checkpoint['torch_rng'].cpu())
   if device=='cuda':torch.cuda.set_rng_state(checkpoint['cuda_rng'].cpu())
   best=checkpoint['best_loss'];best_epoch=checkpoint['best_epoch'];bad=checkpoint['bad_checks'];start_epoch=checkpoint['epoch']+1
  def snapshot(epoch):
   torch.save({'task':t,'epoch':epoch,'model':net.state_dict(),'optimizer':opt.state_dict(),'batch_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None,'best_loss':best,'best_epoch':best_epoch,'bad_checks':bad},resume)
  with (folder/'curve.jsonl').open('a' if start_epoch>1 else 'w') as curve:
   for ep in range(start_epoch,p['maximum_epochs']+1):
    losses=[];zeros=[]
    for b in torch.randperm(n,generator=gen).split(512):
     b=b.to(device);reward=em['train_extra'][b]-t['lam']*costs['train_extra'][b]
     def lp():
      r,q=forward(net,'train_extra',b);return joint_logprob(r,q)
     logp=lp()
     if t['method']=='exact':
      prob=logp.exp();loss=-(prob*reward).sum(1).mean()+.002*(prob*logp).sum(1).mean()
      opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
     else:
      old=logp.detach();oldp=old.exp();actions=torch.multinomial(oldp,t['group'],replacement=True);rewards=reward.gather(1,actions);adv,std=normalized_advantages(rewards);adv=adv.detach();zeros.append((std<1e-6).float().mean().detach())
      for update in range(2):
       if update:logp=lp()
       ratio=(logp.gather(1,actions)-old.gather(1,actions)).clamp(-20,20).exp();surrogate=torch.minimum(ratio*adv,ratio.clamp(.8,1.2)*adv).mean();kl=(oldp*(old-logp)).sum(1).mean();entropy=-(logp.exp()*logp).sum(1).mean();loss=-surrogate+.01*kl-.002*entropy
       opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step()
     if not torch.isfinite(loss):raise RuntimeError(f'Nonfinite loss {tag(t)}')
     losses.append(loss.detach())
    if ep%10==0 or ep==1 or ep==p['maximum_epochs']:
     values=validation(net,t['lam'])
     with torch.no_grad():
      r,q=forward(net,'train_extra');pr=joint_logprob(r,q).exp();train_loss=float(-(pr*(em['train_extra']-t['lam']*costs['train_extra'])).sum(1).mean())
     record={'epoch':ep,'train_loss':train_loss,**values,'surrogate_loss':float(torch.stack(losses).mean()),'zero_advantage_fraction':float(torch.stack(zeros).mean()) if zeros else None,'elapsed_seconds':time.monotonic()-start}
     if values['val_loss']<best-1e-5:best=values['val_loss'];best_epoch=ep;bad=0;save_policy(folder/'best.joblib',net,{**t,'epoch':ep})
     else:bad+=1
     curve.write(json.dumps(record)+'\n');curve.flush();snapshot(ep)
     runtime.write(out/f'worker_{shard}.json',{'phase':'training','completed_streams':number,'streams':len(work),'task':t,'epoch':ep,'best_epoch':best_epoch,'last':record})
     print(tag(t),ep,'val_loss',values['val_loss'],'best',best_epoch,flush=True)
     if bad>=6 and ep>=60:break
  runtime.write(folder/'complete.json',{'task':t,'best_epoch':best_epoch,'last_epoch':ep,'best_val_loss':best,'seconds':time.monotonic()-start,'early_stopped':ep<p['maximum_epochs']})
 runtime.write(out/f'worker_{shard}.json',{'phase':'complete','streams':len(work)})

def sigmoid(z):return 1./(1.+np.exp(-np.clip(z,-60,60)))
def predict_heads(policies,xr,xq):
 r=[];q=[]
 for policy in policies:
  heads=policy['heads'];r.append(sigmoid(np.asarray(xr@heads['retrieval']['weight'].T).ravel()+heads['retrieval']['bias'][0]))
  q.append(sigmoid(np.asarray(xq@heads['answering']['weight'].T).ravel()+heads['answering']['bias'][0]).reshape(-1,2))
 return np.mean(r,axis=0),np.mean(q,axis=0)
def choices(pr,pq,lr,la,retmean,qamean,normalizer):
 retp=np.stack([1-pr,pr],1);qap=np.stack([1-pq,pq],2)
 r=np.argmax(retp-lr*np.asarray(retmean)[None]/normalizer,1);a=np.argmax(qap[np.arange(len(r)),r]-la*np.asarray(qamean)[r]/normalizer,1)
 return r,a

def select(candidates,objective,ss_accuracy,ss_cost):
 eligible=[c for c in candidates if objective=='max_accuracy' or (c['units']<=ss_cost+1e-8 if objective=='ss_budget' else c['accuracy']>=.95*ss_accuracy-1e-10)]
 if not eligible:return None
 return min(eligible,key=(lambda c:(c['units'],-c['accuracy'])) if objective=='cheap95' else (lambda c:(-c['accuracy'],c['units'])))

def evaluate(out):
 p=json.loads((out/'protocol.json').read_text());source=Path(p['source_run']);source_p,questions,rr,qq=load_source(source)
 assert runtime.digest(source/'protocol.json')==p['source_protocol_sha256']
 for name,digest in p['source_cache_hashes'].items():assert runtime.digest(source/name)==digest,'Source cache changed'
 grid=np.load(out/'data/grid_val.npz');em=grid['correct'];cost=grid['cost'];xr=np.load(out/'data/retrieval_val.npy',mmap_mode='r');xq=np.load(out/'data/answering_val.npy',mmap_mode='r')
 configurations={};policies={}
 for t in tasks():
  assert (out/'training'/tag(t)/'complete.json').exists()
  name=config_tag(t);configurations.setdefault(name,[]).append(out/'training'/tag(t)/'best.joblib')
 for seed in SEEDS:
  path=sorted((out/'warm'/f'seed{seed}').glob('shard*.joblib'))[0];configurations.setdefault('sft',[]).append(path)
 candidates=[]
 for name,paths in configurations.items():
  bundle=[joblib.load(path) for path in paths];assert len(bundle)==3
  policies[name]=bundle;pr,pq=predict_heads(bundle,xr,xq)
  for lr in PENALTIES:
   for la in PENALTIES:
    r,a=choices(pr,pq,lr,la,p['training_ret_mean_units'],p['training_qa_mean_units'],p['normalizer'])
    candidates.append({'type':'learned','config':name,'method':name.split('_')[0],'lambda_ret':lr,'lambda_qa':la,'accuracy':float(em[np.arange(len(r)),r,a].mean()),'units':float(cost[np.arange(len(r)),r,a].mean())})
 fixed=[{'type':'fixed','r':r,'a':a,'accuracy':float(em[:,r,a].mean()),'units':float(cost[:,r,a].mean())} for r in range(2) for a in range(2)]
 selections=[]
 for method in ['sft','grpo','exact']:
  learned=[c for c in candidates if c['method']==method]
  for objective in ['max_accuracy','ss_budget','cheap95']:
   for mode in ['learned_only','with_fixed_fallback']:
    winner=select(learned+(fixed if mode=='with_fixed_fallback' else []),objective,float(em[:,1,1].mean()),float(cost[:,1,1].mean()))
    selections.append({'method':method,'objective':objective,'mode':mode,'selected':winner,'note':'No learned configuration satisfies the validation constraint' if winner is None else ''})
 runtime.write(out/'validation_lock.json',{'selections':selections,'candidates':candidates,'fixed_validation':fixed,'note':'All policies/hyperparameters locked here before test feature construction/scoring.'})
 # Only after the lock: construct test outcomes/features and measure chosen-policy CPU decisions.
 ids=p['splits']['test'];tem,tcost=direct_grid(ids,questions,rr,qq,source_p);features=joblib.load(out/'features.joblib');rows=[];summaries=[]
 def append(name,chosen):
  routes=[];latencies=[]
  for qid in ids:
   if chosen['type']=='fixed':routes.append((chosen['r'],chosen['a']));latencies.append(0.);continue
   start=time.perf_counter();retview=classical.view(questions[qid],'retrieval');xx=transform(features['retrieval'],[retview]);models=policies[chosen['config']]
   retprob=np.mean([sigmoid(np.asarray(xx@m['heads']['retrieval']['weight'].T).ravel()+m['heads']['retrieval']['bias'][0])[0] for m in models]);r=int(np.argmax(np.array([1-retprob,retprob])-chosen['lambda_ret']*np.asarray(p['training_ret_mean_units'])/p['normalizer']))
   qa_view=classical.view(questions[qid],'answering',rr[RET[r]][qid],RET[r]);xx=transform(features['answering'],[qa_view]);qa_prob=np.mean([sigmoid(np.asarray(xx@m['heads']['answering']['weight'].T).ravel()+m['heads']['answering']['bias'][0])[0] for m in models]);a=int(np.argmax(np.array([1-qa_prob,qa_prob])-chosen['lambda_qa']*np.asarray(p['training_qa_mean_units'])[r]/p['normalizer']))
   latencies.append((time.perf_counter()-start)*1000);routes.append((r,a))
  records=[]
  for j,(qid,(r,a)) in enumerate(zip(ids,routes)):
   rt,at=RET[r],QA[a];retr=rr[rt][qid];ans=qq[at][qid+'|'+rt];ru=units(source_p['models'][rt]['params_b'],retr['prompt_tokens'],retr['generated_tokens']);au=units(source_p['models'][at]['params_b'],ans['prompt_tokens'],ans['generated_tokens'])
   records.append({'dataset':'hotpotqa','split':'test','id':qid,'track':'nonLLM','policy':name,'menu':'A','menu_variant':'WS2x2','seed':0,'seed_description':'ensemble0/1/2' if chosen['type']=='learned' else 'fixed executor seed0','answer':ans['text'],**score(ans['text'],questions[qid]['answers']),'units':ru+au,'units_retrieve':ru,'units_answer':au,'units_fuse':0.,'units_router':0.,'ms_router':latencies[j],'models':f"retriever={source_p['models'][rt]['repo']}@{source_p['models'][rt]['revision']};answerer={source_p['models'][at]['repo']}@{source_p['models'][at]['revision']}",'comparison_scope':p['scope'],'ret_choice':r,'qa_choice':a})
  rows.extend(records);summaries.append({'policy':name,'accuracy':float(np.mean([r['lenient'] for r in records])),'units':float(np.mean([r['units'] for r in records])),'ms_router':float(np.mean(latencies)),'routes':{f'{RET[r]}-{QA[a]}':routes.count((r,a)) for r in range(2) for a in range(2)},'selected':chosen})
 for r in range(2):
  for a in range(2):append(f'grpo-control-{RET[r]}-{QA[a]}',{'type':'fixed','r':r,'a':a})
 for s in selections:
  if s['selected'] is not None:append(f"grpo-v2-{s['method']}-{s['objective']}-{s['mode']}",s['selected'])
 df=pd.DataFrame(rows);validate(df);df.to_csv(out/'results.csv',index=False)
 ss=next(r for r in summaries if r['policy']=='grpo-control-qwen72-llama70');assert abs(ss['accuracy']-.718)<1e-10
 runtime.write(out/'results.json',{'scope':p['scope'],'ss':ss,'summaries':summaries,'selections_without_feasible_policy':[s for s in selections if s['selected'] is None]})
 lines=['# Corrected four-model finite-action GRPO','','Source: corrected GGUF K1 cache; common0.2 train_extra6405/val500/test500. Frozen executors, author lenient score, actual executed-call token units. Not common0.3 SLR golden set. Test previously inspected; results descriptive.','','Reward = lenient - lambda * route units / training mean direct SS. Direct routing pays only selected retrieval+QA, no weak-first sunk calls. Three-seed stage-probability ensembles. Learned-only and fixed-fallback selections reported separately.','',f"SS: {100*ss['accuracy']:.2f}% lenient; {ss['units']:.3f} units.",'','| Policy | Lenient % | Cost % SS | ms/router |','|---|---:|---:|---:|']
 for row in summaries:lines.append(f"| {row['policy']} | {100*row['accuracy']:.2f} | {100*row['units']/ss['units']:.2f} | {row['ms_router']:.2f} |")
 for missing in [s for s in selections if s['selected'] is None]:lines.append(f"\nNo validation-feasible learned policy: {missing['method']} / {missing['objective']}.")
 lines+=['','GRPO samples discrete routing actions from the cached four-action grid. It does not generate extra LLM answers. Exact control enumerates expected reward over these same actions. KL is to the rollout policy, not a frozen reference LLM. CPU feature/decision time is reported separately; CPU/training work is excluded from the FLOP proxy. All operating points were validation-selected before test scoring.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')


def orchestrate(source,out,max_epochs):
 if not (out/'protocol.json').exists():prepare(source,out,max_epochs)
 else:
  existing=json.loads((out/'protocol.json').read_text());assert existing['maximum_epochs']==max_epochs and Path(existing['source_run'])==source
  for name in ['grpo.py','train.py','run.py']:assert runtime.digest(out/'code'/name)==runtime.digest(Path(__file__).with_name(name)),'Source changed since preparation; explicit migration required'
 used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True)
 free=[i for i,v in enumerate(used.splitlines()) if int(v.strip())<1024]
 if not free:raise RuntimeError('No free GPUs; prepared experiment retained, no external GPU work interrupted')
 devices=free[:4];runtime.write(out/'status.json',{'phase':'training','streams':len(tasks()),'physical_gpus':devices})
 jobs=[]
 for shard,gpu in enumerate(devices):
  log=(out/f'gpu_{gpu}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
  proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','train','--out',str(out),'--shard',str(shard),'--shards',str(len(devices))],env=env,stdout=log,stderr=subprocess.STDOUT);jobs.append((proc,log))
 codes=[proc.wait() for proc,_ in jobs]
 for _,log in jobs:log.close()
 if any(codes):raise RuntimeError(f'Training workers failed: {codes}')
 runtime.write(out/'status.json',{'phase':'validation_selection_and_evaluation'})
 evaluate(out)
 runtime.write(out/'status.json',{'phase':'complete','streams':len(tasks()),'source':str(source)})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['all','prepare','train','evaluate'],default='all');ap.add_argument('--source',type=Path,default=runtime.WORK/'runs/nonllm_common_ws_gguf_v2');ap.add_argument('--out',type=Path,default=runtime.WORK/'runs/nonllm_common_ws_grpo_v1');ap.add_argument('--max-epochs',type=int,default=300);ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);args=ap.parse_args();out=args.out.resolve()
 try:
  if args.phase=='all':orchestrate(args.source.resolve(),out,args.max_epochs)
  elif args.phase=='prepare':prepare(args.source.resolve(),out,args.max_epochs)
  elif args.phase=='train':train_worker(out,args.shard,args.shards)
  else:evaluate(out)
 except BaseException as error:
  if args.phase in ['all','prepare','evaluate']:runtime.write(out/'status.json',{'phase':'failed','error_type':type(error).__name__,'error':str(error)})
  raise
