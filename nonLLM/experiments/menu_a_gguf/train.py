"""Train grouped, three-way non-LLM routers after a complete GGUF grid exists."""
import argparse,gzip,json,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from stair_common.cost import units
from stair_common.score import score,RULER_SHA256
from stair_common.schema import COLUMNS,validate
import run as runtime

FAMILIES=['majority','logreg','linear_svm','multinomial_nb','random_forest','extra_trees']
RET=['qwen7','qwen32','qwen72'];QA=['llama3','llama8','llama70']
LAMBDAS=[0.,.05,.15,.4,.8,1.6]

def view(q,stage,rec=None,branch=None):
 ctx='\n'.join(runtime.pool(q)) if stage=='retrieval' else runtime.evidence(q,rec)
 text=(q['question']+'\n')*3+ctx
 if branch is not None:text+='\nRetrieval model: '+branch
 words=set(q['question'].lower().split());cw=set(ctx.lower().split());tokens=ctx.split()
 nums=[len(q['question']),len(words),len(ctx),len(tokens),len(cw),len(cw&words)/max(1,len(words)),len(cw&words)/max(1,len(cw|words)),ctx.count('\n'),sum(c.isdigit() for c in ctx)/max(1,len(ctx)),int(' or ' in q['question'].lower()),int('both' in words)]
 if stage=='answering':nums += [int(branch==r) for r in RET]
 return text,nums

def proba(bundle,text,num):
 xx=np.asarray(num) if bundle['numeric'] else bundle['vectorizer'].transform(text)
 model=bundle['model'];raw=model.predict_proba(xx);p=np.zeros((len(text),3))
 p[:,model.classes_.astype(int)]=raw
 return p

def candidates(family,seed,folds):
 if family=='majority':return [(None,DummyClassifier(strategy='most_frequent'))]
 if family=='logreg':return [(c,LogisticRegression(C=c,max_iter=1000,random_state=seed)) for c in [.1,1.,10.]]
 if family=='linear_svm':return [(c,CalibratedClassifierCV(LinearSVC(C=c,random_state=seed,max_iter=10000),cv=folds,method='sigmoid',ensemble=False)) for c in [.1,1.]]
 if family=='multinomial_nb':return [(a,MultinomialNB(alpha=a)) for a in [.1,1.,10.]]
 cls=RandomForestClassifier if family=='random_forest' else ExtraTreesClassifier
 return [(d,cls(n_estimators=200,max_depth=d,min_samples_leaf=5,class_weight='balanced',random_state=seed,n_jobs=8)) for d in [8,None]]

def main(out):
 protocol=json.loads((out/'protocol.json').read_text());assert protocol['ruler_sha256']==RULER_SHA256
 with gzip.open(runtime.WORK/'extracted/Dataset/AgentTTS-HotpotQA.json.gz','rt') as f:questions={q['_id']:q for q in json.load(f)}
 rr={m:runtime.model_cache(out,m) for m in RET};qq={m:runtime.model_cache(out,m) for m in QA}
 assert all(len(v)==7405 for v in rr.values());assert all(len(v)==22215 for v in qq.values())
 def grid(ids):
  correct=[];cost=[]
  for qid in ids:
   q=questions[qid];em=np.zeros((3,3));c=np.zeros((3,3))
   for r,rtag in enumerate(RET):
    retr=rr[rtag][qid];rcost=units(protocol['models'][rtag]['params_b'],retr['prompt_tokens'],retr['generated_tokens'])
    for a,atag in enumerate(QA):
     rec=qq[atag][qid+'|'+rtag];em[r,a]=score(rec['text'],q['answers'])['lenient'];c[r,a]=rcost+units(protocol['models'][atag]['params_b'],rec['prompt_tokens'],rec['generated_tokens'])
   correct.append(em);cost.append(c)
  return np.array(correct),np.array(cost)
 ids={s:protocol['splits'][s] for s in ['train_extra','val','test']}
 train_em,train_cost=grid(ids['train_extra']);val_em,val_cost=grid(ids['val'])
 # Counterfactual costs are used for training/validation, never as deployment features.
 retmean=np.array([np.mean([units(protocol['models'][r]['params_b'],rr[r][i]['prompt_tokens'],rr[r][i]['generated_tokens']) for i in ids['train_extra']]) for r in RET])
 qamean=np.array([[np.mean([units(protocol['models'][a]['params_b'],qq[a][i+'|'+r]['prompt_tokens'],qq[a][i+'|'+r]['generated_tokens']) for i in ids['train_extra']]) for a in QA] for r in RET])
 normalizer=retmean[2]+qamean[2,2]
 features={}
 for stage in ['retrieval','answering']:
  features[stage]={}
  for part in ['train_extra','val']:
   rows=[view(questions[qid],stage,None) for qid in ids[part]] if stage=='retrieval' else [view(questions[qid],stage,rr[r][qid],r) for qid in ids[part] for r in RET]
   features[stage][part]=([v[0] for v in rows],np.array([v[1] for v in rows]))
 for stage in features:
  vec=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_df=.98,max_features=25000,sublinear_tf=True,dtype=np.float32)
  vec.fit(features[stage]['train_extra'][0]);features[stage]['vectorizer']=vec
 def targets(em,c,stage):
  if stage=='retrieval':
   masked=np.where(em.astype(bool),c,np.inf).reshape(len(em),9);valid=em.any(axis=(1,2));y=masked.argmin(1)//3
  else:
   valid=em.any(axis=2).reshape(-1);y=np.where(em.astype(bool),c,np.inf).argmin(2).reshape(-1)
  return y.astype(int),valid
 models={};validation=[];fit_records=[]
 for seed in [0,1,2]:
  for family in FAMILIES:
   bundles={};preds={}
   for stage in ['retrieval','answering']:
    yt,mask=targets(train_em,train_cost,stage);yv,vmask=targets(val_em,val_cost,stage)
    trtext,trnum=features[stage]['train_extra'];vtext,vnum=features[stage]['val']
    numeric=family in ['random_forest','extra_trees'];vec=features[stage]['vectorizer']
    xt=trnum if numeric else vec.transform(trtext);xv=vnum if numeric else vec.transform(vtext)
    groups=np.arange(len(yt)) if stage=='retrieval' else np.repeat(np.arange(len(ids['train_extra'])),3)
    valid_groups=groups[mask];counts=np.bincount(valid_groups,minlength=len(ids['train_extra']));weights=1./counts[valid_groups]
    folds=list(StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=seed).split(np.zeros(mask.sum()),yt[mask],valid_groups))
    best=None
    for param,model in candidates(family,seed,folds):
     t=time.monotonic();model.fit(xt[mask],yt[mask],sample_weight=weights)
     bundle={'model':model,'numeric':numeric,'vectorizer':vec};pv=proba(bundle,vtext,vnum)
     loss=log_loss(yv[vmask],pv[vmask],labels=[0,1,2]);pt=proba(bundle,[trtext[j] for j in np.flatnonzero(mask)],trnum[mask])
     fit_records.append({'family':family,'stage':stage,'seed':seed,'parameter':param,'train_log_loss':log_loss(yt[mask],pt,labels=[0,1,2],sample_weight=weights),'val_log_loss':loss,'seconds':time.monotonic()-t,'training_rows':int(mask.sum()),'validation_rows':int(vmask.sum()),'selection':'best validation log loss; classical fit has no neural epoch'})
     if best is None or loss<best[0]:best=(loss,param,bundle,pv)
    bundles[stage]=best[2];preds[stage]=best[3]
   name=f'{family}_seed{seed}';models[name]=bundles
   dest=out/'routers'/f'{name}.joblib';dest.parent.mkdir(exist_ok=True);joblib.dump(bundles,dest)
   pr=preds['retrieval'];pq=preds['answering'].reshape(-1,3,3);n=len(pr);candidates_v=[]
   for lr in LAMBDAS:
    for la in LAMBDAS:
     r=np.argmax(pr-lr*retmean[None]/normalizer,axis=1);a=np.argmax(pq[np.arange(n),r]-la*qamean[r]/normalizer,axis=1)
     candidates_v.append({'type':'router','lambda_ret':lr,'lambda_qa':la,'accuracy':float(val_em[np.arange(n),r,a].mean()),'units':float(val_cost[np.arange(n),r,a].mean())})
   for r in range(3):
    for a in range(3):candidates_v.append({'type':'fixed','r':r,'a':a,'accuracy':float(val_em[:,r,a].mean()),'units':float(val_cost[:,r,a].mean())})
   for objective in ['max_accuracy','ss_budget','cheap95']:
    eligible=[c for c in candidates_v if objective=='max_accuracy' or (c['units']<=val_cost[:,2,2].mean()+1e-10 if objective=='ss_budget' else c['accuracy']>=.95*val_em[:,2,2].mean()-1e-10)]
    key=(lambda c:(c['units'],-c['accuracy'])) if objective=='cheap95' else (lambda c:(-c['accuracy'],c['units']))
    validation.append({'name':name,'family':family,'seed':seed,'objective':objective,'selected':min(eligible,key=key)})
   runtime.write(out/'training_progress.json',{'phase':'training','completed_model_pairs':len(models),'total_model_pairs':18,'last':name})
 runtime.write(out/'fit_records.json',fit_records)
 runtime.write(out/'validation_lock.json',{'selections':validation,'training_mean_ret_units':retmean.tolist(),'training_mean_qa_units':qamean.tolist(),'note':'Written before any test feature construction or test scoring; historical test was previously examined.'})
 # Test processing starts only after the complete validation lock.
 rows=[];summaries=[]
 def append_policy(policy,seed,r,a,latencies):
  records=[]
  for j,qid in enumerate(ids['test']):
   rt=RET[int(r[j])];at=QA[int(a[j])];retr=rr[rt][qid];ans=qq[at][qid+'|'+rt]
   ru=units(protocol['models'][rt]['params_b'],retr['prompt_tokens'],retr['generated_tokens']);au=units(protocol['models'][at]['params_b'],ans['prompt_tokens'],ans['generated_tokens'])
   value={'dataset':'hotpotqa','split':'test','id':qid,'track':'nonLLM','policy':'gguf-'+policy,'menu':'A','seed':seed,'answer':ans['text'],**score(ans['text'],questions[qid]['answers']),'units':ru+au,'units_retrieve':ru,'units_answer':au,'units_fuse':0.,'units_router':0.,'ms_router':latencies[j],'models':f"retriever={protocol['models'][rt]['repo']}@{protocol['models'][rt]['revision']};answerer={protocol['models'][at]['repo']}@{protocol['models'][at]['revision']}",'comparison_scope':'GGUF exploratory, not team pooled bf16/AWQ','weights_quant':'Q4_K_M','kv_quant':'q8_0'}
   records.append(value)
  rows.extend(records);summaries.append({'policy':'gguf-'+policy,'seed':seed,'accuracy':float(np.mean([r['lenient'] for r in records])),'units':float(np.mean([r['units'] for r in records])),'ms_router':float(np.mean(latencies))})
 for r in range(3):
  for a in range(3):append_policy(f'fixed-{RET[r]}-{QA[a]}',0,np.full(500,r),np.full(500,a),np.zeros(500))
 for config in validation:
  selected=config['selected'];rlist=[];alist=[];latency=[];bundle=models[config['name']]
  for qid in ids['test']:
   if selected['type']=='fixed':rlist.append(selected['r']);alist.append(selected['a']);latency.append(0.);continue
   t=time.perf_counter();q=questions[qid];text,num=view(q,'retrieval');pr=proba(bundle['retrieval'],[text],[num])[0]
   r=int(np.argmax(pr-selected['lambda_ret']*retmean/normalizer));text,num=view(q,'answering',rr[RET[r]][qid],RET[r]);pq=proba(bundle['answering'],[text],[num])[0]
   a=int(np.argmax(pq-selected['lambda_qa']*qamean[r]/normalizer));latency.append((time.perf_counter()-t)*1000);rlist.append(r);alist.append(a)
  append_policy(config['family']+'-'+config['objective'],config['seed'],rlist,alist,latency)
 df=pd.DataFrame(rows);validate(df);df.to_csv(out/'results.csv',index=False)
 runtime.write(out/'results.json',{'scope':protocol['comparison_scope'],'summaries':summaries,'router_seeds':[0,1,2],'generation_seed':0})
 strongest=next(r for r in summaries if r['policy']=='gguf-fixed-qwen72-llama70')
 lines=['# Menu A GGUF direct non-LLM routing','',protocol['comparison_scope'],'','Standard common test IDs and author lenient ruler; Q4_K_M weights, q8_0 KV, K1/greedy/no fusion. Cached generation seed0 is shared by all router seeds; only router fitting varies. No bf16/AWQ pooling claim.','',f"Strongest: {100*strongest['accuracy']:.2f}% lenient, {strongest['units']:.3f} AgentTTS units.",'','| Policy | Router seed | Lenient % | Units | Cost % strongest | ms/router |','|---|---:|---:|---:|---:|---:|']
 for r in summaries:lines.append(f"| {r['policy']} | {r['seed']} | {100*r['accuracy']:.2f} | {r['units']:.3f} | {100*r['units']/strongest['units']:.2f} | {r['ms_router']:.3f} |")
 lines+=['','Router cost reports CPU time separately; the parameter-FLOP proxy excludes classical CPU feature/model work. Classical fits are selected by validation log loss, not a fabricated neural epoch. Hyperparameters and all operating points were locked before test features/scoring. Previously examined test content means results are descriptive.']
 (out/'REPORT.md').write_text('\n'.join(lines)+'\n')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
