"""Outcome/gain prediction and unscaled policy-gradient experiments on frozen caches."""
import argparse,copy,gzip,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from stair_common.score import score
from stair_common.cost import units
from stair_common.schema import validate
import grpo as base
runtime=base.runtime
SEEDS=[0,1,2];LAMBDAS=[0.,.05,.15,.4]
PENALTIES=[0.,.005,.01,.02,.05,.1,.15,.25,.4,.8,1.6]
OBJECTIVES=['max_accuracy','ss_budget','cheap95','match_ss_accuracy']
STOPWORDS=set('what which is the a an of in on was are were and or to by for does did do with who when where how many that this from as'.split())

def extras(question,rec=None):
 q=question['question'];qt=re.findall(r'[a-z0-9]+',q.lower());qw=set(qt)-STOPWORDS
 docs=runtime.pool(question) if rec is None else [runtime.pool(question)[i] for i in rec['selected']]
 tokens=[set(re.findall(r'[a-z0-9]+',d.lower())) for d in docs]
 coverage=np.array([len(qw&t)/max(1,len(qw)) for t in tokens],float)
 title=np.array([len(qw&set(re.findall(r'[a-z0-9]+',d.split(':',1)[0].lower())))/max(1,len(qw)) for d in docs],float)
 cov=sorted(coverage,reverse=True)+[0.,0.,0.];length=np.array([len(d) for d in docs],float)
 first=(qt[0] if qt else '')
 values=[float(first==w) for w in ['what','which','who','when','where','how','is','was','are','were','did','does']]
 values += [len(q),len(qt),len(qw),len(set(qt))/max(1,len(qt)),len(re.findall(r'\b[A-Z][A-Za-z]+\b',q)),sum(c.isdigit() for c in q),q.count('"'),float(' or ' in q.lower()),float(' and ' in q.lower()),float(any(w in qt for w in ['more','less','before','after','longer','earlier','same'])),len(docs)]
 values += [float(length.sum()),float(length.mean()) if len(length) else 0.,float(length.std()) if len(length) else 0.,float(length.max()) if len(length) else 0.]
 values += [cov[0],cov[1],cov[2],cov[0]-cov[1],float(coverage.mean()) if len(coverage) else 0.,float(coverage.std()) if len(coverage) else 0.]
 values += [float(np.quantile(coverage,p)) if len(coverage) else 0. for p in [.25,.5,.75,.9]]
 values += [float((coverage>0).sum()),float((coverage>=.5).sum()),float((coverage>=1).sum()),float(title.max()) if len(title) else 0.,float(title.mean()) if len(title) else 0.,float((title>0).sum())]
 if len(tokens)>=2:
  rank=np.argsort(-coverage);a,b=tokens[rank[0]],tokens[rank[1]];values += [len(a&b)/max(1,len(a|b)),len(a&b),len((a|b)&qw)/max(1,len(qw))]
 else:values += [0.,0.,float(coverage.max()) if len(coverage) else 0.]
 return np.array(values,np.float32)

def tasks():
 tasks=[]
 for method in ['outcome_linear','outcome_residual','gain_linear']:
  for wd in [.001,.01]:
   for seed in SEEDS:tasks.append(dict(method=method,wd=wd,lam=0.,seed=seed))
 for method in ['group_centered','expected_baseline','exact_augmented']:
  for lam in LAMBDAS:
   for seed in SEEDS:tasks.append(dict(method=method,wd=.001,lam=lam,seed=seed))
 return tasks

def task_tag(t):return f"{t['method']}_wd{t['wd']}_lambda{t['lam']}_seed{t['seed']}"
def config_tag(t):return f"{t['method']}_wd{t['wd']}_lambda{t['lam']}"
def tree_tasks():return [dict(method=m,depth=d,leaf=l,seed=s) for m in ['outcome_trees','gain_trees'] for d in [8,None] for l in [10,30] for s in SEEDS]
def tree_tag(t):return f"{t['method']}_depth{t['depth']}_leaf{t['leaf']}_seed{t['seed']}"
def tree_config(t):return f"{t['method']}_depth{t['depth']}_leaf{t['leaf']}"

def prepare(source,out):
 assert not (out/'protocol.json').exists()
 old=json.loads((source/'protocol.json').read_text());assert json.loads((source/'status.json').read_text())['phase']=='complete'
 raw,questions,rr,qq=base.load_source(Path(old['source_run']))
 assert runtime.digest(runtime.WORK/'extracted/Dataset/AgentTTS-HotpotQA.json.gz')==raw['data_sha256']
 assert runtime.digest(Path(old['source_run'])/'protocol.json')==old['source_protocol_sha256']
 out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir();scalers={};dimensions={}
 for stage in ['retrieval','answering']:
  values={part:np.asarray(([extras(questions[i]) for i in old['splits'][part]] if stage=='retrieval' else [extras(questions[i],rr[r][i]) for i in old['splits'][part] for r in base.RET]),np.float32) for part in ['train_extra','val']}
  scaler=StandardScaler().fit(values['train_extra']);scalers[stage]=scaler
  old_numeric=11 if stage=='retrieval' else 13
  for part,v in values.items():
   x=np.load(source/'data'/f'{stage}_{part}.npy',mmap_mode='r');extra=scaler.transform(v).astype(np.float32)
   full=np.concatenate([x,extra],1);numeric=np.concatenate([x[:,-old_numeric:],extra],1)
   np.save(out/'data'/f'{stage}_{part}.npy',full);np.save(out/'data'/f'{stage}_numeric_{part}.npy',numeric);dimensions[stage]=full.shape[1]
 joblib.dump(scalers,out/'extra_scalers.joblib')
 for part in ['train_extra','val']:(out/'data'/f'grid_{part}.npz').symlink_to((source/'data'/f'grid_{part}.npz').resolve())
 protocol={**{k:old[k] for k in ['source_run', 'source_protocol_sha256', 'source_cache_hashes', 'ruler_sha256', 'splits', 'models', 'normalizer', 'training_ret_mean_units', 'training_qa_mean_units', 'scope', 'generation', 'actions']},'prior_grpo_run':str(source),'run':'corrected-cache-router-alternatives-v1','dimensions':dimensions,'base_dimensions':old['dimensions'],'prior_feature_sha256':runtime.digest(source/'features.joblib'),'question_source_sha256':raw['data_sha256'],'seeds':SEEDS,'policy_group':8,'optimizer':'Adam lr0.002; weight decay per task. Policy heads load original per-seed SFT weights, extra-feature coefficients initially zero. Supervised outcome/gain heads start fresh.','early_stopping':'BCE for outcome, MSE for gain, negative expected task reward for policy; check every10 epochs plus1, patience6 after at least60, maximum300.','features':'Existing training-only TFIDF/numeric features plus training-only scaled lexical candidate/evidence statistics; no gold/answers/unselected outputs as inputs.','neural_tasks':tasks(),'tree_tasks':tree_tasks(),'maximum_epochs':300,'selection':'Predeclared eight families, three seeds, validation-only checkpoints and operating points. Learned-only and fixed-fallback separated. No test data used before validation lock.','outcome':'Predict correctness of all4 routes from pool, then2 QA outcomes from selected evidence; BCE on all outcomes, no cheapest-correct mask.','gain':'Predict SS minus weak-retrieval/strong-QA correctness; QA fixed strong. MSE includes ties; threshold selected on validation.','group_centered':'Group8 PPO, advantages r minus sampled group mean, no std normalization; same warm start, clip0.2, KL0.01, entropy0.002.','expected_baseline':'Group8 PPO, advantages r minus exact current-policy expected reward from the cached four-route train grid; no std normalization.','exact_augmented':'Exact expected reward with appended lexical features; same warm start and regularizers.','outcome_architectures':'Linear and linear plus64-unit ReLU residual; output residual initialized zero. Weight decay0.001/0.01.','tree_configs':'ExtraTrees192 trees, depth8/None, leaf10/30, three seeds; numeric-only extended features.','operating_points':{'stage_penalties':PENALTIES,'gain_thresholds':'all unique validation predicted gains plus extremes','objectives':OBJECTIVES},'test_disclosure':'Repeatedly inspected test; these are exploratory method comparisons, not fresh confirmatory evidence.'}
 runtime.write(out/'protocol.json',protocol);(out/'code').mkdir()
 for name in ['alternatives.py','grpo.py','train.py','run.py']:shutil.copy2(Path(__file__).with_name(name),out/'code'/name)
 runtime.write(out/'status.json',{'phase':'prepared','neural_streams':len(tasks()),'tree_streams':len(tree_tasks())});print('Prepared',dimensions,flush=True)

def neural_worker(out,shard,shards):
 import torch
 torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False
 p=json.loads((out/'protocol.json').read_text());device='cuda' if torch.cuda.is_available() else 'cpu'
 x={stage:{part:torch.tensor(np.load(out/'data'/f'{stage}_{part}.npy'),device=device) for part in ['train_extra','val']} for stage in ['retrieval','answering']}
 grids={part:np.load(out/'data'/f'grid_{part}.npz') for part in ['train_extra','val']}
 em={part:torch.tensor(grids[part]['correct'],device=device) for part in grids};cost={part:torch.tensor(grids[part]['cost']/p['normalizer'],device=device) for part in grids}
 n=len(em['train_extra']);work=[t for i,t in enumerate(tasks()) if i%shards==shard]
 class Head(torch.nn.Module):
  def __init__(self,d,k,residual):
   super().__init__();self.base=torch.nn.Linear(d,k);self.residual=residual
   if residual:self.hidden=torch.nn.Linear(d,64);self.output=torch.nn.Linear(64,k);torch.nn.init.zeros_(self.output.weight);torch.nn.init.zeros_(self.output.bias)
  def forward(self,a):return self.base(a)+(self.output(torch.relu(self.hidden(a))) if self.residual else 0.)
 def package(net):return {name:{key:value.detach().cpu().numpy().copy() for key,value in head.state_dict().items()} for name,head in net.items()}
 for index,t in enumerate(work):
  folder=out/'training'/task_tag(t);folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists():continue
  torch.manual_seed(t['seed']);gen=torch.Generator().manual_seed(t['seed']+1000)
  outcome=t['method'].startswith('outcome');gain=t['method']=='gain_linear';policy=not outcome and not gain
  heads={'retrieval':Head(p['dimensions']['retrieval'],4 if outcome else 1,t['method']=='outcome_residual')}
  if not gain:heads['answering']=Head(p['dimensions']['answering'],2 if outcome else 1,t['method']=='outcome_residual')
  net=torch.nn.ModuleDict(heads).to(device)
  if policy:
   prior=Path(p['prior_grpo_run']);warm=joblib.load(sorted((prior/'warm'/f"seed{t['seed']}").glob('*.joblib'))[0])
   with torch.no_grad():
    for stage in heads:
     w=warm['heads'][stage];net[stage].base.weight.zero_();net[stage].base.weight[:,:w['weight'].shape[1]].copy_(torch.tensor(w['weight'],device=device));net[stage].base.bias.copy_(torch.tensor(w['bias'],device=device))
  optimizer=torch.optim.Adam(net.parameters(),lr=.002,weight_decay=t['wd'])
  def forward(part,b=None):
   xr=x['retrieval'][part];xq=x['answering'][part].reshape(-1,2,p['dimensions']['answering'])
   if b is not None:xr=xr[b];xq=xq[b]
   r=net['retrieval'](xr)
   if gain:return r.flatten(),None
   q=net['answering'](xq)
   return (r.reshape(-1,2,2),q) if outcome else (r.flatten(),q.squeeze(-1))
  def loss_on(part,b=None):
   r,q=forward(part,b);y=em[part] if b is None else em[part][b]
   if outcome:return .5*(torch.nn.functional.binary_cross_entropy_with_logits(r,y)+torch.nn.functional.binary_cross_entropy_with_logits(q,y))
   if gain:return torch.nn.functional.mse_loss(r,y[:,1,1]-y[:,0,1])
   c=cost[part] if b is None else cost[part][b];prob=base.joint_logprob(r,q).exp();return -(prob*(y-t['lam']*c).reshape(-1,4)).sum(1).mean()
  best=float('inf');best_epoch=0;bad=0;start=time.monotonic()
  if policy:
   with torch.no_grad():best=float(loss_on('val'))
   joblib.dump({'task':t,'heads':package(net),'kind':'policy','numeric':False,'epoch':0},folder/'best.joblib')
  with (folder/'curve.jsonl').open('w') as curve:
   for epoch in range(1,p['maximum_epochs']+1):
    losses=[]
    for batch in torch.randperm(n,generator=gen).split(512):
     batch=batch.to(device)
     if not policy:loss=loss_on('train_extra',batch);optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
     else:
      reward=(em['train_extra'][batch]-t['lam']*cost['train_extra'][batch]).reshape(-1,4)
      def lp():
       r,q=forward('train_extra',batch);return base.joint_logprob(r,q)
      logp=lp();prob=logp.exp()
      if t['method']=='exact_augmented':
       loss=-(prob*reward).sum(1).mean()+.002*(prob*logp).sum(1).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
      else:
       old=logp.detach();oldp=old.exp();actions=torch.multinomial(oldp,8,replacement=True);rewards=reward.gather(1,actions)
       advantage=unscaled_advantages(rewards,oldp,reward,t['method']).detach()
       for update in range(2):
        if update:logp=lp()
        ratio=(logp.gather(1,actions)-old.gather(1,actions)).clamp(-20,20).exp();surrogate=torch.minimum(ratio*advantage,ratio.clamp(.8,1.2)*advantage).mean();kl=(oldp*(old-logp)).sum(1).mean();entropy=-(logp.exp()*logp).sum(1).mean();loss=-surrogate+.01*kl-.002*entropy
        optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1.);optimizer.step()
     if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss '+task_tag(t))
     losses.append(loss.detach())
    if epoch==1 or epoch%10==0:
     with torch.no_grad():vl=float(loss_on('val'));tl=float(loss_on('train_extra'))
     if vl<best-1e-5:
      best=vl;best_epoch=epoch;bad=0;joblib.dump({'task':t,'heads':package(net),'kind':'outcome' if outcome else 'gain' if gain else 'policy','numeric':False,'epoch':epoch},folder/'best.joblib')
     else:bad+=1
     rec={'epoch':epoch,'train_loss':tl,'val_loss':vl,'surrogate_loss':float(torch.stack(losses).mean()),'seconds':time.monotonic()-start};curve.write(json.dumps(rec)+'\n');curve.flush()
     torch.save({'task':t,'epoch':epoch,'model':net.state_dict(),'optimizer':optimizer.state_dict(),'batch_rng':gen.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if device=='cuda' else None},folder/'state_last.pt')
     runtime.write(out/f'worker_{shard}.json',{'phase':'training','completed':index,'total':len(work),'task':t,'epoch':epoch,'best_epoch':best_epoch})
     if bad>=6 and epoch>=60:break
  runtime.write(folder/'complete.json',{'task':t,'best_epoch':best_epoch,'last_epoch':epoch,'val_loss':best,'seconds':time.monotonic()-start})
 runtime.write(out/f'worker_{shard}.json',{'phase':'complete','total':len(work)})

def unscaled_advantages(sample_rewards,old_probs,all_rewards,method):
 baseline=sample_rewards.mean(1,keepdim=True) if method=='group_centered' else (old_probs*all_rewards).sum(1,keepdim=True)
 return sample_rewards-baseline

def tree_worker(out):
 p=json.loads((out/'protocol.json').read_text());x={stage:{part:np.load(out/'data'/f'{stage}_numeric_{part}.npy') for part in ['train_extra','val']} for stage in ['retrieval','answering']};grid={part:np.load(out/'data'/f'grid_{part}.npz')['correct'] for part in ['train_extra','val']}
 for idx,t in enumerate(tree_tasks()):
  folder=out/'trees'/tree_tag(t);folder.mkdir(parents=True,exist_ok=True)
  if (folder/'complete.json').exists():continue
  start=time.monotonic();outcome=t['method']=='outcome_trees';models={}
  for stage in (['retrieval','answering'] if outcome else ['retrieval']):
   target=grid['train_extra'].reshape(-1,4) if stage=='retrieval' else grid['train_extra'].reshape(-1,2)
   if not outcome:target=grid['train_extra'][:,1,1]-grid['train_extra'][:,0,1]
   model=ExtraTreesRegressor(n_estimators=192,max_depth=t['depth'],min_samples_leaf=t['leaf'],max_features=1.,random_state=t['seed'],n_jobs=2);model.fit(x[stage]['train_extra'],target);models[stage]=model
  bundle={'task':t,'kind':'outcome' if outcome else 'gain','numeric':True,'models':models};joblib.dump(bundle,folder/'best.joblib')
  runtime.write(folder/'complete.json',{'task':t,'seconds':time.monotonic()-start});runtime.write(out/'trees_progress.json',{'completed':idx+1,'total':len(tree_tasks())})

def forward_np(head,x):
 value=np.asarray(x@head['base.weight'].T)+head['base.bias']
 if 'hidden.weight' in head:value+=np.maximum(0,np.asarray(x@head['hidden.weight'].T)+head['hidden.bias'])@head['output.weight'].T+head['output.bias']
 return value

def predict(models,xr,xq):
 kind=models[0]['kind'];rp=[];qp=[]
 for model in models:
  if model['numeric']:
   r=model['models']['retrieval'].predict(xr);q=model['models']['answering'].predict(xq) if kind=='outcome' else None
  else:
   r=forward_np(model['heads']['retrieval'],xr);q=forward_np(model['heads']['answering'],xq) if kind!='gain' else None
   if kind!='gain':r=base.sigmoid(r);q=base.sigmoid(q)
  rp.append(np.asarray(r));
  if q is not None:qp.append(np.asarray(q))
 r=np.mean(rp,0)
 if kind=='gain':return np.clip(r.reshape(-1),-1,1),None
 q=np.mean(qp,0)
 return (r.reshape(-1,2,2),q.reshape(-1,2,2)) if kind=='outcome' else (r.reshape(-1),q.reshape(-1,2))

def choose(kind,rscore,qscore,candidate,p):
 if kind=='gain':return (rscore>candidate['threshold']).astype(int),np.ones(len(rscore),int)
 lr,la=candidate['lambda_ret'],candidate['lambda_qa'];retmean=np.asarray(p['training_ret_mean_units'])/p['normalizer'];qamean=np.asarray(p['training_qa_mean_units'])/p['normalizer']
 if kind=='outcome':
  # Retrieval planning uses only pool-based outcome estimates, never other-branch evidence.
  plan=(rscore-la*qamean[None]).max(2)-lr*retmean[None];r=np.argmax(plan,1);a=np.argmax(qscore[np.arange(len(r)),r]-la*qamean[r],1)
  return r,a
 return base.choices(rscore,qscore,lr,la,p['training_ret_mean_units'],p['training_qa_mean_units'],p['normalizer'])

def select(candidates,objective,ss_acc,ss_cost):
 eligible=[c for c in candidates if objective=='max_accuracy' or (c['units']<=ss_cost+1e-8 if objective=='ss_budget' else c['accuracy']>=ss_acc*(.95 if objective=='cheap95' else 1.)-1e-10)]
 if not eligible:return None
 return min(eligible,key=(lambda c:(c['units'],-c['accuracy'])) if objective in ['cheap95','match_ss_accuracy'] else (lambda c:(-c['accuracy'],c['units'])))

def evaluate(out):
 p=json.loads((out/'protocol.json').read_text());source=Path(p['source_run']);old=Path(p['prior_grpo_run'])
 assert runtime.digest(source/'protocol.json')==p['source_protocol_sha256']
 for name,digest in p['source_cache_hashes'].items():assert runtime.digest(source/name)==digest
 configs={}
 for t in tasks():
  path=out/'training'/task_tag(t);assert (path/'complete.json').exists();configs.setdefault(config_tag(t),[]).append(path/'best.joblib')
 for t in tree_tasks():
  path=out/'trees'/tree_tag(t);assert (path/'complete.json').exists();configs.setdefault(tree_config(t),[]).append(path/'best.joblib')
 models={name:[joblib.load(path) for path in paths] for name,paths in configs.items()};assert all(len(m)==3 for m in models.values())
 x={stage:{numeric:np.load(out/'data'/f"{stage}_{'numeric_' if numeric else ''}val.npy",mmap_mode='r') for numeric in [False,True]} for stage in ['retrieval','answering']}
 grid=np.load(out/'data/grid_val.npz');em,cost=grid['correct'],grid['cost'];candidate_rows=[]
 for name,bundle in models.items():
  numeric=bundle[0]['numeric'];kind=bundle[0]['kind'];method=bundle[0]['task']['method'];rscore,qscore=predict(bundle,x['retrieval'][numeric],x['answering'][numeric])
  options=([{'threshold':float(t)} for t in np.r_[rscore.min()-1e-6,np.unique(rscore),rscore.max()+1e-6]] if kind=='gain' else [{'lambda_ret':lr,'lambda_qa':la} for lr in PENALTIES for la in PENALTIES])
  for option in options:
   choice={'type':'learned','config':name,'method':method,'kind':kind,**option};r,a=choose(kind,rscore,qscore,choice,p);candidate_rows.append({**choice,'accuracy':float(em[np.arange(len(r)),r,a].mean()),'units':float(cost[np.arange(len(r)),r,a].mean())})
 fixed=[{'type':'fixed','r':r,'a':a,'accuracy':float(em[:,r,a].mean()),'units':float(cost[:,r,a].mean())} for r in range(2) for a in range(2)]
 selections=[]
 for method in sorted(set(c['method'] for c in candidate_rows)):
  pool=[c for c in candidate_rows if c['method']==method]
  for objective in OBJECTIVES:
   for mode in ['learned_only','with_fixed_fallback']:
    selections.append({'method':method,'objective':objective,'mode':mode,'selected':select(pool+(fixed if mode=='with_fixed_fallback' else []),objective,float(em[:,1,1].mean()),float(cost[:,1,1].mean()))})
 # Also predeclare global family choice on validation, never on test rankings.
 for objective in OBJECTIVES:
  for mode in ['learned_only','with_fixed_fallback']:
   selections.append({'method':'global','objective':objective,'mode':mode,'selected':select(candidate_rows+(fixed if mode=='with_fixed_fallback' else []),objective,float(em[:,1,1].mean()),float(cost[:,1,1].mean()))})
 pd.DataFrame(candidate_rows).to_csv(out/'validation_candidates.csv.gz',index=False,compression='gzip')
 runtime.write(out/'validation_lock.json',{'created_at':time.time(),'selections':selections,'fixed_validation':fixed,'candidate_count':len(candidate_rows),'note':'All configurations locked before test features/outcomes are constructed. Test was previously examined in older runs.'})
 # Test feature creation only after the complete validation lock.
 raw,questions,rr,qq=base.load_source(source);assert runtime.digest(old/'features.joblib')==p['prior_feature_sha256'];ids=p['splits']['test'];features=joblib.load(old/'features.joblib');scalers=joblib.load(out/'extra_scalers.joblib');xt={};nt={}
 for stage in ['retrieval','answering']:
  views=([base.classical.view(questions[i],stage) for i in ids] if stage=='retrieval' else [base.classical.view(questions[i],stage,rr[r][i],r) for i in ids for r in base.RET])
  additional=np.asarray(([extras(questions[i]) for i in ids] if stage=='retrieval' else [extras(questions[i],rr[r][i]) for i in ids for r in base.RET]))
  ex=scalers[stage].transform(additional).astype(np.float32);core=base.transform(features[stage],views);xt[stage]=np.concatenate([core.toarray(),ex],1);nt[stage]=np.concatenate([core.toarray()[:,-(11 if stage=='retrieval' else 13):],ex],1)
 predicted={};rows=[];summaries=[]
 def append(name,choice):
  if choice['type']=='fixed':r=np.full(len(ids),choice['r']);a=np.full(len(ids),choice['a'])
  else:
   config=choice['config'];bundle=models[config];numeric=bundle[0]['numeric']
   if config not in predicted:predicted[config]=predict(bundle,(nt if numeric else xt)['retrieval'],(nt if numeric else xt)['answering'])
   r,a=choose(bundle[0]['kind'],*predicted[config],choice,p)
  records=[]
  for j,qid in enumerate(ids):
   rt,at=base.RET[r[j]],base.QA[a[j]];retr=rr[rt][qid];ans=qq[at][qid+'|'+rt];ru=units(raw['models'][rt]['params_b'],retr['prompt_tokens'],retr['generated_tokens']);au=units(raw['models'][at]['params_b'],ans['prompt_tokens'],ans['generated_tokens'])
   records.append({'dataset':'hotpotqa','split':'test','id':qid,'track':'nonLLM','policy':name,'menu':'A','menu_variant':'WS2x2','seed':0,'seed_description':'three-seed ensemble for learned models','answer':ans['text'],**score(ans['text'],questions[qid]['answers']),'units':ru+au,'units_retrieve':ru,'units_answer':au,'units_fuse':0.,'units_router':0.,'ms_router':None,'models':f"retriever={raw['models'][rt]['repo']}@{raw['models'][rt]['revision']};answerer={raw['models'][at]['repo']}@{raw['models'][at]['revision']}",'ret_choice':int(r[j]),'qa_choice':int(a[j]),'comparison_scope':p['scope']})
  rows.extend(records);summaries.append({'policy':name,'accuracy':float(np.mean([r['lenient'] for r in records])),'units':float(np.mean([r['units'] for r in records])),'routes':{f'{base.RET[ri]}-{base.QA[ai]}':int(((r==ri)&(a==ai)).sum()) for ri in range(2) for ai in range(2)},'selected':choice})
 for r in range(2):
  for a in range(2):append(f'alternative-control-{base.RET[r]}-{base.QA[a]}',{'type':'fixed','r':r,'a':a})
 for selected in selections:
  if selected['selected'] is not None:append('alternative-'+selected['method']+'-'+selected['objective']+'-'+selected['mode'],selected['selected'])
 df=pd.DataFrame(rows);validate(df);df.to_csv(out/'results.csv',index=False);ss=next(r for r in summaries if r['policy']=='alternative-control-qwen72-llama70');assert abs(ss['accuracy']-.718)<1e-10
 runtime.write(out/'results.json',{'scope':p['scope'],'ss':ss,'summaries':summaries,'no_feasible_policy':[s for s in selections if s['selected'] is None],'latency_note':'Offline vectorized replay; per-question ms_router unmeasured, left empty rather than treating precomputed features as free.'})
 lines=['# Additional corrected-cache non-LLM routers','','Same frozen executors, common0.2 train_extra6405/val500/test500, author lenient and actual direct-call token units. Not the common0.3 golden set. Repeatedly inspected test: exploratory comparisons.','', 'All models are non-LLM TFIDF/numeric predictors. New lexical statistics use only candidate pool before retrieval or selected paid evidence after retrieval. No fresh executor generation. Learned-only and fixed-fallback selections are separate. Offline vectorized evaluation does not measure full routing latency; ms_router is left empty.','','| Policy | Lenient % | Cost % SS | SS choices |','|---|---:|---:|---:|']
 for row in summaries:lines.append(f"| {row['policy']} | {100*row['accuracy']:.2f} | {100*row['units']/ss['units']:.2f} | {row['routes']['qwen72-llama70']} |")
 lines+=['','Validation constraints do not guarantee the same test constraint. Thresholds, regularization, architectures and checkpoints were locked before test construction. Exact denotes expected-reward enumeration, not exact-match grading. Statistical comparisons after repeated exploration are descriptive.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

def orchestrate(source,out):
 if not (out/'protocol.json').exists():prepare(source,out)
 else:
  for name in ['alternatives.py','grpo.py','train.py','run.py']:assert runtime.digest(Path(__file__).with_name(name))==runtime.digest(out/'code'/name),'Source changed; explicit migration required'
 used=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],text=True);gpus=[i for i,v in enumerate(used.splitlines()) if int(v.strip())<1024][:4]
 if not gpus:raise RuntimeError('No free local GPUs; no external work interrupted')
 runtime.write(out/'status.json',{'phase':'training','gpus':gpus,'neural_streams':len(tasks()),'tree_streams':len(tree_tasks())});workers=[]
 for shard,gpu in enumerate(gpus):
  log=(out/f'gpu_{gpu}.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
  proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','neural','--out',str(out),'--shard',str(shard),'--shards',str(len(gpus))],env=env,stdout=log,stderr=subprocess.STDOUT);workers.append((proc,log))
 log=(out/'trees.log').open('w');proc=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','trees','--out',str(out)],env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2'),stdout=log,stderr=subprocess.STDOUT);workers.append((proc,log))
 codes=[proc.wait() for proc,_ in workers]
 for _,log in workers:log.close()
 if any(codes):raise RuntimeError(f'Worker exit codes {codes}')
 runtime.write(out/'status.json',{'phase':'validation_selection_and_evaluation'});evaluate(out);runtime.write(out/'status.json',{'phase':'complete','neural_streams':len(tasks()),'tree_streams':len(tree_tasks())})

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['all','prepare','neural','trees','evaluate'],default='all');ap.add_argument('--source',type=Path,default=runtime.WORK/'runs/nonllm_common_ws_grpo_v1');ap.add_argument('--out',type=Path,default=runtime.WORK/'runs/nonllm_common_ws_alternatives_v1');ap.add_argument('--shard',type=int,default=0);ap.add_argument('--shards',type=int,default=4);args=ap.parse_args();out=args.out.resolve()
 try:
  if args.phase=='all':orchestrate(args.source.resolve(),out)
  elif args.phase=='prepare':prepare(args.source.resolve(),out)
  elif args.phase=='neural':neural_worker(out,args.shard,args.shards)
  elif args.phase=='trees':tree_worker(out)
  else:evaluate(out)
 except BaseException as e:
  if args.phase in ['all','prepare','evaluate']:runtime.write(out/'status.json',{'phase':'failed','error':str(e),'error_type':type(e).__name__})
  raise
