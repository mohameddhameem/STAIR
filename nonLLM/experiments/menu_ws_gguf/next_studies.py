"""Pairwise rescue/harm, nested OOF stacking, held-out risk gating and paid cascades."""
import argparse,json,os,re,shutil,subprocess,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.ensemble import ExtraTreesClassifier,ExtraTreesRegressor
from sklearn.model_selection import KFold
from stair_common.score import score
from stair_common.cost import units
from stair_common.schema import validate
import grpo as base
import alternatives as previous
runtime=base.runtime
SEEDS=[0,1,2];FOLDS=5;INNER_FOLDS=3;TREES=96
WEIGHTS=[0.,1e-5,2e-5,5e-5,1e-4,2e-4,5e-4,1e-3]
OBJECTIVES=['max_accuracy','ss_budget','cheap95','match_ss_accuracy']
FAMILIES=['outcome_joint','outcome_staged','gain_joint','gain_staged','pair_joint','pair_staged','stack_joint','stack_staged','cascade']

def raw_features(q,rec=None,branch=None):
 stage='retrieval' if rec is None else 'answering'
 _,numeric=base.classical.view(q,stage,rec,branch)
 return np.r_[np.asarray(numeric,np.float32),previous.extras(q,rec)].astype(np.float32)

def cascade_retrieval_features(q,weak_retrieval,r_features=None,weak_features=None):
 # No weak QA or strong retrieval is observed yet.
 return np.r_[raw_features(q) if r_features is None else r_features,raw_features(q,weak_retrieval,'qwen7') if weak_features is None else weak_features,weak_retrieval['prompt_tokens'],weak_retrieval['generated_tokens'],weak_retrieval['finish_reason']=='length',len(weak_retrieval['selected']),len(weak_retrieval['text'])].astype(np.float32)

def weak_answer_features(question,evidence,weak):
 text=weak['text'];words=set(re.findall(r'[a-z0-9]+',text.lower()));q=set(re.findall(r'[a-z0-9]+',question.lower()));e=set(re.findall(r'[a-z0-9]+',evidence.lower()))
 return np.asarray([len(text),len(text.split()),len(words),text.count('\n'),text.count('?'),text.count('['),text.count('"'),sum(c.isdigit() for c in text)/max(1,len(text)),len(words&q)/max(1,len(q)),len(words&e)/max(1,len(words)),len(words&e)/max(1,len(e)),float('yes' in words),float('no' in words),float('yes' in words and 'no' in words),sum(phrase in text.lower() for phrase in ['cannot','insufficient','not enough','not provided',"don't know",'unable']),weak['prompt_tokens'],weak['generated_tokens'],float(weak['finish_reason']=='length')],np.float32)

def cascade_qa_features(q,retrieval,weak,branch,q_features=None):
 return np.r_[raw_features(q,retrieval,branch) if q_features is None else q_features,weak_answer_features(q['question'],runtime.evidence(q,retrieval),weak)].astype(np.float32)

def cascade_costs(ret,qa):
 n=len(ret);result=np.zeros((n,2,2),np.float64)
 for r in range(2):
  for a in range(2):result[:,r,a]=ret[:,0]+(ret[:,1] if r else 0)+qa[:,r,0]+(qa[:,r,1] if a else 0)
 return result

def data_for(ids,questions,rr,qq,p):
 xr=[];xq=[];cr=[];cq=[];y=np.zeros((len(ids),2,2),np.float32);ret=np.zeros((len(ids),2));qa=np.zeros((len(ids),2,2))
 for i,qid in enumerate(ids):
  q=questions[qid];rf=raw_features(q);wf=raw_features(q,rr['qwen7'][qid],'qwen7');xr.append(rf);cr.append(cascade_retrieval_features(q,rr['qwen7'][qid],rf,wf))
  for r,rt in enumerate(base.RET):
   rec=rr[rt][qid];ret[i,r]=units(p['models'][rt]['params_b'],rec['prompt_tokens'],rec['generated_tokens']);qf=wf if r==0 else raw_features(q,rec,rt);xq.append(qf);weak=qq['llama3'][qid+'|'+rt];cq.append(cascade_qa_features(q,rec,weak,rt,qf))
   for a,at in enumerate(base.QA):
    answer=qq[at][qid+'|'+rt];y[i,r,a]=score(answer['text'],q['answers'])['lenient'];qa[i,r,a]=units(p['models'][at]['params_b'],answer['prompt_tokens'],answer['generated_tokens'])
 return {'r':np.asarray(xr,np.float32),'q':np.asarray(xq,np.float32),'cr':np.asarray(cr,np.float32),'cq':np.asarray(cq,np.float32),'y':y,'ret':ret,'qa':qa,'direct':ret[:,:,None]+qa,'cascade':cascade_costs(ret,qa)}

def prepare(source,out):
 assert not (out/'protocol.json').exists()
 p,questions,rr,qq=base.load_source(source);assert runtime.digest(runtime.WORK/'extracted/Dataset/AgentTTS-HotpotQA.json.gz')==p['data_sha256']
 all_ids=p['splits']['train_extra'];perm=np.random.default_rng(20261024).permutation(len(all_ids));calibration_indices=sorted(perm[:1280].tolist());fit_indices=sorted(perm[1280:].tolist())
 splits={'fit':[all_ids[i] for i in fit_indices],'calibration':[all_ids[i] for i in calibration_indices],'val':p['splits']['val'],'test':p['splits']['test']}
 assert len(splits['fit'])==5125 and len(splits['calibration'])==1280
 assert sum(map(len,splits.values()))==len(set(sum(splits.values(),[])))==7405
 out.mkdir(parents=True,exist_ok=True);(out/'data').mkdir()
 for part in ['fit','calibration','val']:
  np.savez(out/'data'/f'{part}.npz',**data_for(splits[part],questions,rr,qq,p));print('Prepared',part,len(splits[part]),flush=True)
 protocol={'source_run':str(source),'source_protocol_sha256':runtime.digest(source/'protocol.json'),'source_cache_hashes':{str(f.relative_to(source)):runtime.digest(f) for f in sorted((source/'cache').glob('*.jsonl'))},'question_source_sha256':p['data_sha256'],'ruler_sha256':p['ruler_sha256'],'models':p['models'],'splits':splits,'split_seed':20261024,'seeds':SEEDS,'outer_folds':FOLDS,'inner_folds':INNER_FOLDS,'trees':TREES,'features':'Raw deterministic vocabulary-free numerical lexical statistics. No TFIDF vocabulary or scaler is fitted, shared across folds, or reused from previous routers. QA branches of a question always share a fold.','core_model_selection':'OOF family+weight selected only on outer-fold held-out predictions. Stacked outer predictions use inner OOF meta-training, never other-fold predictions that saw the outer-heldout labels.','stacking':'Three base experts: outcome regression, signed-gain regression, rescue/harm/tie classification. Meta outcome regressors fit OOF predictions plus raw features. Three-seed averaging.','pairwise':'For each route vs SS, classify harm/tie/rescue. QA weak-vs-strong classifier uses only selected evidence. Full four-path joint and sequential variants reported.','cost_forecast':'Separate train-only regressors predict route/QA cost from available features. No per-query unexecuted token counts or actual counterfactual costs enter decisions. Cost penalty is per measured AgentTTS unit; no fitted normalization.','cascade':'Always weak retrieval, optional strong retrieval, then weak QA on selected evidence, optional strong QA. Retrieval reviewer cannot see weak QA/strong retrieval; QA reviewer sees only selected evidence and executed weak answer. All sunk calls are charged.','risk':'Joint pair/stack proposers freeze on val before using held-out calibration labels. Nested acceptance on predicted harm, extra-error loss=1{switch & SS correct & alternative wrong}. CRC-style threshold requires (calibration_harm+1)/(n+1)<=alpha, or exact no-switch fallback.','risk_alphas':[.005,.01,.02],'risk_guarantee':'Exploratory only: calibration is held out from all NEW models, but was used in previous research. Test was inspected before; no certified fresh finite-sample guarantee or per-test accuracy guarantee claimed.','weights':WEIGHTS,'objectives':OBJECTIVES,'scope':'GGUF common0.2 test500, not common0.3 SLR golden. New fit5125/calibration1280 partition within original6405. Older routers used more fitting labels.','test_protocol':'All OOF/val method choices and all calibration thresholds locked before test construction. No new LLM generation; fixed executors unchanged.'}
 runtime.write(out/'protocol.json',protocol);(out/'code').mkdir()
 for name in ['next_studies.py','alternatives.py','grpo.py','train.py','run.py']:shutil.copy2(Path(__file__).with_name(name),out/'code'/name)
 runtime.write(out/'status.json',{'phase':'prepared','fit':5125,'calibration':1280,'val':500,'test':500})

def reg(seed,meta=False):return ExtraTreesRegressor(n_estimators=TREES,max_depth=6 if meta else 8,min_samples_leaf=30 if meta else 20,random_state=seed,n_jobs=2)
def clf(seed):return ExtraTreesClassifier(n_estimators=TREES,max_depth=8,min_samples_leaf=20,random_state=seed,n_jobs=2)
def pair_labels(delta):return np.where(delta<0,0,np.where(delta>0,2,1)).astype(int)
def qa_indices(indices):return (indices[:,None]*2+np.arange(2)[None]).reshape(-1)

def fit_bundle(d,indices,experts_only=False):
 xr=d['r'][indices];qi=qa_indices(indices);xq=d['q'][qi];yr=d['y'][indices].reshape(-1,4);yq=d['y'][indices].reshape(-1,2)
 dr=yr-yr[:,3,None];dq=yq-yq[:,1,None];models=[]
 for seed in SEEDS:
  m={}
  for name,targets in [('outcome',(yr,yq)),('gain',(dr,dq)),('pair',(pair_labels(dr),pair_labels(dq)))]:
   for stage,x,target in [('r',xr,targets[0]),('q',xq,targets[1])]:
    model=clf(seed) if name=='pair' else reg(seed);model.fit(x,target);m[name+'_'+stage]=model
  if not experts_only:
   route_targets=np.c_[d['direct'][indices].reshape(-1,4),d['ret'][indices]]
   target_specs=[('cost_r',xr,route_targets),('cost_q',xq,d['qa'][indices].reshape(-1,2)),('cascade_r',d['cr'][indices],yr),('cascade_q',d['cq'][qi],yq),('cascade_cost_r',d['cr'][indices],d['cascade'][indices].reshape(-1,4)),('cascade_cost_q',d['cq'][qi],np.c_[np.zeros(len(qi)),d['qa'][indices].reshape(-1,2)[:,1]])]
   for name,x,target in target_specs:m[name]=reg(seed).fit(x,target)
  models.append(m)
 return models

def pair_prediction(model,x):
 outputs=model.predict_proba(x);classes=model.classes_;gain=[];harm=[]
 for probabilities,labels in zip(outputs,classes):
  h=probabilities[:,list(labels).index(0)] if 0 in labels else np.zeros(len(x));g=probabilities[:,list(labels).index(2)] if 2 in labels else np.zeros(len(x));gain.append(g-h);harm.append(h)
 return np.stack(gain,1),np.stack(harm,1)

def predict_bundle(models,d,indices,experts_only=False):
 qi=qa_indices(indices);result={}
 for stage,x in [('r',d['r'][indices]),('q',d['q'][qi])]:
  for name in ['outcome','gain']:
   value=np.mean([m[name+'_'+stage].predict(x) for m in models],0);result[name+'_'+stage]=value if stage=='r' else value.reshape(-1,2,2)
  values=[pair_prediction(m['pair_'+stage],x) for m in models]
  for position,name in [(0,'pair'),(1,'harm')]:
   value=np.mean([v[position] for v in values],0);result[name+'_'+stage]=value if stage=='r' else value.reshape(-1,2,2)
 if not experts_only:
  for name,feature in [('cost_r','r'),('cost_q','q'),('cascade_r','cr'),('cascade_q','cq'),('cascade_cost_r','cr'),('cascade_cost_q','cq')]:
   x=d[feature][indices if feature in ['r','cr'] else qi];v=np.mean([m[name].predict(x) for m in models],0)
   result[name]=v if feature in ['r','cr'] else v.reshape(-1,2,2)
 return result

def meta_features(d,indices,pred):
 qi=qa_indices(indices)
 return {'r':np.c_[d['r'][indices],pred['outcome_r'],pred['gain_r'],pred['pair_r'],pred['harm_r']],'q':np.c_[d['q'][qi],pred['outcome_q'].reshape(-1,2),pred['gain_q'].reshape(-1,2),pred['pair_q'].reshape(-1,2),pred['harm_q'].reshape(-1,2)]}

def fit_meta(features,y):
 return [{'r':reg(seed,True).fit(features['r'],y.reshape(-1,4)),'q':reg(seed,True).fit(features['q'],y.reshape(-1,2))} for seed in SEEDS]
def add_meta(pred,meta,d,indices):
 features=meta_features(d,indices,pred);pred['stack_r']=np.mean([m['r'].predict(features['r']) for m in meta],0);pred['stack_q']=np.mean([m['q'].predict(features['q']) for m in meta],0).reshape(-1,2,2)
 return pred

def concatenate_predictions(parts,n):
 result={key:np.zeros((n,*next(iter(parts))[1][key].shape[1:])) for key in parts[0][1]}
 for indices,pred in parts:
  for key,value in pred.items():result[key][indices]=value
 return result

def fold_worker(out,fold):
 d=dict(np.load(out/'data/fit.npz'));outer=list(KFold(FOLDS,shuffle=True,random_state=20261024).split(d['r']));train,held=outer[fold];dest=out/'folds'/str(fold);dest.mkdir(parents=True,exist_ok=True)
 if (dest/'complete.json').exists():return
 models=fit_bundle(d,train);pred=predict_bundle(models,d,held);inner_parts=[];inner_audit=[]
 for inner,(subtrain,subheld) in enumerate(KFold(INNER_FOLDS,shuffle=True,random_state=20261100+fold).split(train)):
  ti,vi=train[subtrain],train[subheld];assert not set(ti)&set(held) and not set(vi)&set(held)
  bundle=fit_bundle(d,ti,True);ip=predict_bundle(bundle,d,vi,True);inner_parts.append((subheld,ip));inner_audit.append({'train':ti.tolist(),'heldout':vi.tolist()});runtime.write(dest/'progress.json',{'phase':'inner_crossfit','completed_inner':inner+1,'total_inner':INNER_FOLDS})
 oof=concatenate_predictions(inner_parts,len(train));features=meta_features(d,train,oof);meta=fit_meta(features,d['y'][train]);pred=add_meta(pred,meta,d,held)
 np.savez(dest/'predictions.npz',heldout=held,**pred);runtime.write(dest/'complete.json',{'outer_fold':fold,'train':train.tolist(),'heldout':held.tolist(),'inner_folds':inner_audit,'strictly_excludes_outer_labels':True})


def decisions(pred,family,weight):
 if family=='cascade':
  utility=pred['cascade_r']-weight*pred['cascade_cost_r'];r=utility.argmax(1)//2;a=(pred['cascade_q'][np.arange(len(r)),r]-weight*pred['cascade_cost_q'][np.arange(len(r)),r]).argmax(1)
  return r,a
 name,mode=family.rsplit('_',1);utility=pred[name+'_r']-weight*pred['cost_r'][:,:4];route=utility.argmax(1);r=route//2;a=route%2
 if mode=='staged':a=(pred[name+'_q'][np.arange(len(r)),r]-weight*pred['cost_q'][np.arange(len(r)),r]).argmax(1)
 return r,a

def select(candidates,objective,ss_accuracy,ss_units):
 eligible=[c for c in candidates if objective=='max_accuracy' or (c['units']<=ss_units+1e-8 if objective=='ss_budget' else c['accuracy']>=ss_accuracy*(.95 if objective=='cheap95' else 1.)-1e-10)]
 if not eligible:return None
 return min(eligible,key=(lambda c:(c['units'],-c['accuracy'])) if objective in ['cheap95','match_ss_accuracy'] else (lambda c:(-c['accuracy'],c['units'])))

def candidate_table(pred,d):
 rows=[];y=d['y'];n=len(y)
 for family in FAMILIES:
  for weight in WEIGHTS:
   r,a=decisions(pred,family,weight);c=d['cascade'] if family=='cascade' else d['direct'];rows.append({'type':'model','family':family,'weight':weight,'accuracy':float(y[np.arange(n),r,a].mean()),'units':float(c[np.arange(n),r,a].mean())})
 return rows

def fixed_table(d):return [{'type':'fixed','r':r,'a':a,'accuracy':float(d['y'][:,r,a].mean()),'units':float(d['direct'][:,r,a].mean())} for r in range(2) for a in range(2)]

def calibrated_threshold(confidence,eligible,harm,alpha):
 # Nested switch sets; no score/model selection is performed on calibration.
 n=len(confidence)
 for threshold in np.r_[-1.,np.unique(confidence[eligible])]:
  switch=eligible&(confidence>threshold);errors=int((harm&switch).sum());bound=(errors+1)/(n+1)
  if bound<=alpha and switch.any():return {'threshold':float(threshold),'calibration_harm':errors,'criterion_bound':bound,'switches':int(switch.sum()),'alpha':alpha,'n':n,'no_switch':False}
 return {'threshold':None,'calibration_harm':0,'criterion_bound':0.,'switches':0,'alpha':alpha,'n':n,'no_switch':True}

def apply_gate(r,a,confidence,calibration):
 switch=((r*2+a)!=3)&(confidence>calibration['threshold']) if not calibration['no_switch'] else np.zeros(len(r),bool)
 return np.where(switch,r,1),np.where(switch,a,1)


def finish(out):
 p=json.loads((out/'protocol.json').read_text());fit=dict(np.load(out/'data/fit.npz'));val=dict(np.load(out/'data/val.npz'));parts=[]
 for fold in range(FOLDS):
  folder=out/'folds'/str(fold);assert (folder/'complete.json').exists();data=np.load(folder/'predictions.npz');parts.append((data['heldout'],{k:data[k] for k in data.files if k!='heldout'}))
 oof=concatenate_predictions(parts,len(fit['r']));assert sorted(np.concatenate([i for i,_ in parts]).tolist())==list(range(len(fit['r'])))
 oof_candidates=candidate_table(oof,fit);pd.DataFrame(oof_candidates).to_csv(out/'oof_candidates.csv',index=False,lineterminator='\n')
 # Final models fit only5125; the1280 calibration questions remain entirely held out.
 all_fit=np.arange(len(fit['r']));inner_parts=[];final_meta_audit=[]
 # Match the same inner3-fold stacking procedure used in each outer evaluation.
 for inner,(train,held) in enumerate(KFold(INNER_FOLDS,shuffle=True,random_state=20261200).split(all_fit)):
  bundle=fit_bundle(fit,train,True);inner_parts.append((held,predict_bundle(bundle,fit,held,True)));final_meta_audit.append({'train':train.tolist(),'heldout':held.tolist()})
 final_oof=concatenate_predictions(inner_parts,len(all_fit));models=fit_bundle(fit,all_fit);meta=fit_meta(meta_features(fit,all_fit,final_oof),fit['y']);joblib.dump({'models':models,'meta':meta},out/'models.joblib')
 runtime.write(out/'final_meta_crossfit.json',{'folds':final_meta_audit,'inner_folds':INNER_FOLDS,'calibration_used':False})
 pv=add_meta(predict_bundle(models,val,np.arange(len(val['r']))),meta,val,np.arange(len(val['r'])));val_candidates=candidate_table(pv,val);fixed_val=fixed_table(val);selections=[]
 for family in FAMILIES:
  pool=[c for c in val_candidates if c['family']==family]
  for objective in OBJECTIVES:
   for mode in ['learned_only','with_fixed_fallback']:
    selections.append({'name':f'{family}-{objective}-{mode}','selection_source':'validation','selected':select(pool+(fixed_val if mode=='with_fixed_fallback' else []),objective,float(val['y'][:,1,1].mean()),float(val['direct'][:,1,1].mean()))})
 for source,candidates,d in [('global_val',val_candidates,val),('global_oof',oof_candidates,fit)]:
  for objective in OBJECTIVES:
   for mode in ['learned_only','with_fixed_fallback']:
    selections.append({'name':f'{source}-{objective}-{mode}','selection_source':'validation' if source=='global_val' else 'nested_outer_oof','selected':select(candidates+(fixed_table(d) if mode=='with_fixed_fallback' else []),objective,float(d['y'][:,1,1].mean()),float(d['direct'][:,1,1].mean()))})
 # Freeze the complete proposal/model selection before loading calibration labels.
 proposers=[]
 for family in ['pair_joint','stack_joint']:
  for objective in OBJECTIVES:
   selected=select([c for c in val_candidates if c['family']==family],objective,float(val['y'][:,1,1].mean()),float(val['direct'][:,1,1].mean()))
   if selected is not None:proposers.append({'name':f'{family}-{objective}','selected':selected})
 runtime.write(out/'proposal_lock.json',{'created_at':time.time(),'selections':selections,'risk_proposers':proposers,'note':'Created before calibration-label use; no old trained router/OOF model is reused.'})
 calibration=dict(np.load(out/'data/calibration.npz'));ci=np.arange(len(calibration['r']));pc=add_meta(predict_bundle(models,calibration,ci),meta,calibration,ci);risk=[]
 for proposal in proposers:
  chosen=proposal['selected'];r,a=decisions(pc,chosen['family'],chosen['weight']);route=r*2+a;confidence=1-pc['harm_r'][ci,route];eligible=route!=3;harm=(calibration['y'][:,1,1]==1)&(calibration['y'][ci,r,a]==0)
  for alpha in p['risk_alphas']:risk.append({'name':proposal['name']+f'-alpha{alpha}','selected':chosen,'calibration':calibrated_threshold(confidence,eligible,harm,alpha)})
 runtime.write(out/'validation_lock.json',{'created_at':time.time(),'selections':selections,'risk':risk,'calibration_label_use':'Threshold only after proposal_lock, fit/core models never train on calibration','test_started':False,'risk_certified':False})
 # Only now construct test features/outcomes, including the hypothetical branch caches.
 source=Path(p['source_run']);assert runtime.digest(source/'protocol.json')==p['source_protocol_sha256']
 for name,digest in p['source_cache_hashes'].items():assert runtime.digest(source/name)==digest
 raw,questions,rr,qq=base.load_source(source);assert runtime.digest(runtime.WORK/'extracted/Dataset/AgentTTS-HotpotQA.json.gz')==p['question_source_sha256']
 ids=p['splits']['test'];test=data_for(ids,questions,rr,qq,raw);ti=np.arange(len(ids));pt=add_meta(predict_bundle(models,test,ti),meta,test,ti);rows=[];summaries=[]
 def append(name,r,a,cascade=False,selection=None,risk_info=None):
  records=[]
  for j,qid in enumerate(ids):
   rt,at=base.RET[r[j]],base.QA[a[j]];answer=qq[at][qid+'|'+rt]['text']
   if cascade:ru=test['ret'][j,0]+(test['ret'][j,1] if r[j] else 0);au=test['qa'][j,r[j],0]+(test['qa'][j,r[j],1] if a[j] else 0)
   else:ru=test['ret'][j,r[j]];au=test['qa'][j,r[j],a[j]]
   called=['qwen7']+(['qwen72'] if r[j] else [])+['llama3']+(['llama70'] if a[j] else []) if cascade else [rt,at]
   records.append({'dataset':'hotpotqa','split':'test','id':qid,'track':'nonLLM','policy':name,'menu':'A','menu_variant':'WS2x2','seed':0,'answer':answer,**score(answer,questions[qid]['answers']),'units':float(ru+au),'units_retrieve':float(ru),'units_answer':float(au),'units_fuse':0.,'units_router':0.,'ms_router':None,'models':';'.join(t+'='+raw['models'][t]['repo']+'@'+raw['models'][t]['revision'] for t in called),'ret_choice':int(r[j]),'qa_choice':int(a[j]),'execution':'weak_first_cascade' if cascade else 'direct','comparison_scope':p['scope']})
  rows.extend(records);summary={'policy':name,'accuracy':float(np.mean([z['lenient'] for z in records])),'units':float(np.mean([z['units'] for z in records])),'cascade':cascade,'routes':{f'{base.RET[ri]}-{base.QA[ai]}':int(((r==ri)&(a==ai)).sum()) for ri in range(2) for ai in range(2)},'selected':selection}
  if risk_info is not None:summary['calibration']=risk_info;summary['test_extra_harm']=int(((test['y'][:,1,1]==1)&(np.array([z['lenient'] for z in records])==0)).sum());summary['test_rescues']=int(((test['y'][:,1,1]==0)&(np.array([z['lenient'] for z in records])==1)).sum())
  summaries.append(summary)
 for r in range(2):
  for a in range(2):append(f'next-control-{base.RET[r]}-{base.QA[a]}',np.full(len(ids),r),np.full(len(ids),a))
 append('next-cascade-always-escalate',np.ones(len(ids),int),np.ones(len(ids),int),True)
 for choice in selections:
  chosen=choice['selected']
  if chosen is None:continue
  if chosen['type']=='fixed':r=np.full(len(ids),chosen['r']);a=np.full(len(ids),chosen['a']);cascade=False
  else:r,a=decisions(pt,chosen['family'],chosen['weight']);cascade=chosen['family']=='cascade'
  append('next-'+choice['name'],r,a,cascade,chosen)
 for choice in risk:
  chosen=choice['selected'];r,a=decisions(pt,chosen['family'],chosen['weight']);confidence=1-pt['harm_r'][ti,r*2+a];r,a=apply_gate(r,a,confidence,choice['calibration']);append('next-risk-'+choice['name'],r,a,False,chosen,choice['calibration'])
 df=pd.DataFrame(rows);validate(df);df.to_csv(out/'results.csv',index=False);ss=next(s for s in summaries if s['policy']=='next-control-qwen72-llama70');assert abs(ss['accuracy']-.718)<1e-10
 runtime.write(out/'results.json',{'scope':p['scope'],'ss':ss,'summaries':summaries,'infeasible':[s for s in selections if s['selected'] is None],'risk_certified':False,'latency':'Vectorized cache replay, full online latency unmeasured; ms_router empty; CPU work excluded from executor units.'})
 lines=['# Pairwise routing, nested cross-fitting, calibration and weak-first cascades','','New core5125/calibration1280 partition inside old6405; validation500 and previously inspected test500 unchanged. Three-seed tree ensembles; vocabulary-free features. Frozen four-model GGUF cache. Not common0.3 golden-set reproduction.','', 'Outer5/inner3 strict grouped cross-fitting supplies base and stacked OOF predictions. OOF family selection is separate from validation family selection. Calibration labels are used only after all proposers freeze. CRC-style thresholds are exploratory because these samples were used by previous experiments; no certified or per-test accuracy guarantee.','', 'Cascade costs include weak retrieval plus optional strong retrieval, weak QA on the selected branch plus optional strong QA. Direct SS remains the denominator. Vectorized replay does not measure online router latency.','','| Policy | Test lenient % | Cost % direct SS | Execution |','|---|---:|---:|---|']
 for row in summaries:lines.append(f"| {row['policy']} | {100*row['accuracy']:.2f} | {100*row['units']/ss['units']:.2f} | {'cascade' if row['cascade'] else 'direct'} |")
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n');runtime.write(out/'status.json',{'phase':'complete','outer_folds':FOLDS,'inner_folds':INNER_FOLDS,'policies':len(summaries),'rows':len(rows)})


def orchestrate(source,out):
 if not (out/'protocol.json').exists():prepare(source,out)
 else:
  for name in ['next_studies.py','alternatives.py','grpo.py','train.py','run.py']:assert runtime.digest(Path(__file__).with_name(name))==runtime.digest(out/'code'/name),'Source changed since preparation'
 runtime.write(out/'status.json',{'phase':'outer_nested_crossfit','outer_folds':FOLDS});pending=list(range(FOLDS));running=[]
 while pending or running:
  while pending and len(running)<4:
   fold=pending.pop(0);log=(out/f'fold_{fold}.log').open('w');env=dict(os.environ,OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',MKL_NUM_THREADS='2')
   process=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fold','--out',str(out),'--fold',str(fold)],env=env,stdout=log,stderr=subprocess.STDOUT);running.append((fold,process,log))
  for item in list(running):
   fold,process,log=item;code=process.poll()
   if code is not None:
    log.close();running.remove(item)
    if code:raise RuntimeError(f'Outer fold{fold} failed with exit{code}')
  if running:time.sleep(1)
 runtime.write(out/'status.json',{'phase':'final_fit_selection_calibration_evaluation'});finish(out)

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['all','prepare','fold','finish'],default='all');ap.add_argument('--source',type=Path,default=runtime.WORK/'runs/nonllm_common_ws_gguf_v2');ap.add_argument('--out',type=Path,default=runtime.WORK/'runs/nonllm_common_ws_next_v1');ap.add_argument('--fold',type=int,default=0);args=ap.parse_args();out=args.out.resolve()
 try:
  if args.phase=='all':orchestrate(args.source.resolve(),out)
  elif args.phase=='prepare':prepare(args.source.resolve(),out)
  elif args.phase=='fold':fold_worker(out,args.fold)
  else:finish(out)
 except BaseException as error:
  if args.phase in ['all','prepare','finish']:runtime.write(out/'status.json',{'phase':'failed','error':str(error),'error_type':type(error).__name__})
  raise
