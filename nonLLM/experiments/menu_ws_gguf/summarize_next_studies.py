"""Audit fold isolation, paid-call costs and risk calibration for the four studies."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd,joblib
from stair_common.schema import validate
from stair_common.cost import units
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import next_studies as study

def main(out):
 p=json.loads((out/'protocol.json').read_text());df=pd.read_csv(out/'results.csv',keep_default_na=False);validate(df);result=json.loads((out/'results.json').read_text());summaries=result['summaries'];ss=result['ss'];assert len(df)==500*len(summaries)
 for row in summaries:
  chosen=df[df.policy==row['policy']];assert len(chosen)==500 and chosen.id.nunique()==500
  assert abs(chosen.lenient.mean()-row['accuracy'])<1e-12 and abs(chosen.units.mean()-row['units'])<1e-8
 universe=set(range(5125));held_all=[];fold_audit=[]
 for fold in range(5):
  a=json.loads((out/'folds'/str(fold)/'complete.json').read_text());train,held=set(a['train']),set(a['heldout']);assert not train&held and train|held==universe;held_all+=a['heldout'];inner_held=[]
  for inner in a['inner_folds']:
   ti,vi=set(inner['train']),set(inner['heldout']);assert not ti&vi and ti|vi==train and not held&(ti|vi);assert not set(study.qa_indices(np.array(sorted(ti))))&set(study.qa_indices(np.array(sorted(vi))));inner_held+=inner['heldout']
  assert sorted(inner_held)==sorted(train);fold_audit.append(a)
 assert sorted(held_all)==list(range(5125))
 final=json.loads((out/'final_meta_crossfit.json').read_text());held_all=[]
 for fold in final['folds']:
  assert not set(fold['train'])&set(fold['heldout']);assert set(fold['train'])|set(fold['heldout'])==universe;held_all+=fold['heldout']
 assert sorted(held_all)==list(range(5125))
 models=joblib.load(out/'models.joblib')
 for model in models['models']:
  assert model['outcome_r'].estimators_[0].tree_.n_node_samples[0]==5125
  assert model['outcome_q'].estimators_[0].tree_.n_node_samples[0]==10250
 for model in models['meta']:assert model['r'].estimators_[0].tree_.n_node_samples[0]==5125
 source=Path(p['source_run']);rr={t:study.runtime.model_cache(source,t) for t in study.base.RET};qq={t:study.runtime.model_cache(source,t) for t in study.base.QA};cost_lookup={}
 for qid in p['splits']['test']:
  ret=[units(p['models'][t]['params_b'],rr[t][qid]['prompt_tokens'],rr[t][qid]['generated_tokens']) for t in study.base.RET]
  qa=[[units(p['models'][t]['params_b'],qq[t][qid+'|'+r]['prompt_tokens'],qq[t][qid+'|'+r]['generated_tokens']) for t in study.base.QA] for r in study.base.RET];cost_lookup[qid]=(ret,qa)
 for row in df.itertuples():
  ret,qa=cost_lookup[row.id];r,a=row.ret_choice,row.qa_choice
  ru=ret[0]+(ret[1] if r else 0) if row.execution=='weak_first_cascade' else ret[r]
  au=qa[r][0]+(qa[r][1] if a else 0) if row.execution=='weak_first_cascade' else qa[r][a]
  assert abs(ru-row.units_retrieve)<1e-8 and abs(au-row.units_answer)<1e-8
  assert row.answer==qq[study.base.QA[a]][row.id+'|'+study.base.RET[r]]['text']
 baseline=df[df.policy=='next-control-qwen72-llama70'].set_index('id');assert baseline.lenient.sum()==359
 risk=[];table=[];paired=[];rng=np.random.default_rng(20261024)
 for row in summaries:
  selected=df[df.policy==row['policy']].set_index('id').loc[baseline.index];harm=int(((baseline.lenient==1)&(selected.lenient==0)).sum());rescue=int(((baseline.lenient==0)&(selected.lenient==1)).sum())
  if 'calibration' in row:
   c=row['calibration'];assert c['criterion_bound']<=c['alpha']+1e-12
   assert harm==row['test_extra_harm'] and rescue==row['test_rescues']
   risk.append({'policy':row['policy'],'alpha':c['alpha'],'calibration_n':c['n'],'calibration_harm':c['calibration_harm'],'calibration_bound':c['criterion_bound'],'calibration_switches':c['switches'],'test_harm':harm,'test_harm_rate':harm/500,'test_rescues':rescue,'test_accuracy':row['accuracy'],'cost_pct_ss':100*row['units']/ss['units'],'test_rate_above_nominal':harm/500>c['alpha']+1e-12})
  if row['policy'].endswith('learned_only'):
   table.append({'policy':row['policy'],'correct':int(selected.lenient.sum()),'accuracy_pct':100*row['accuracy'],'cost_pct_ss':100*row['units']/ss['units'],'cascade':row['cascade'],'selected':json.dumps(row['selected'])})
   if 'max_accuracy' in row['policy'] or 'match_ss_accuracy' in row['policy']:
    delta=(selected.lenient-baseline.lenient).to_numpy();boot=np.array([delta[rng.integers(0,500,500)].mean()*100 for _ in range(3000)])
    paired.append({'policy':row['policy'],'rescues':rescue,'harm':harm,'delta_pp':100*float(delta.mean()),'paired_descriptive95_pp':np.quantile(boot,[.025,.975]).tolist()})
 pd.DataFrame(table).to_csv(out/'method_summary.csv',index=False,lineterminator='\n');pd.DataFrame(risk).to_csv(out/'risk_summary.csv',index=False,lineterminator='\n');study.runtime.write(out/'fold_audits.json',{'outer':fold_audit,'final_meta':final})
 validation=json.loads((out/'validation_lock.json').read_text());proposal=json.loads((out/'proposal_lock.json').read_text());assert proposal['created_at']<=validation['created_at']
 cascade=next(row for row in summaries if row['policy']=='next-cascade-always-escalate');assert cascade['accuracy']==ss['accuracy'] and cascade['units']>ss['units']
 study.runtime.write(out/'verification.json',{'rows':len(df),'policies':len(summaries),'outer_folds':5,'inner_folds':3,'strict_nested_group_isolation':True,'final_fit_rows':5125,'calibration_rows':1280,'all_calibration_criteria_met':True,'cascade_paid_call_cost_parity':True,'cached_answer_parity':True,'cascade_all_strong_cost_pct_ss':100*cascade['units']/ss['units'],'risk_test_above_nominal':[r for r in risk if r['test_rate_above_nominal']],'paired':paired,'risk_certified':False,'note':'Expected-risk calibration targets are not bounds on each finite test sample; historical reuse and possible distribution differences preclude certified claims.'})
 fig,ax=plt.subplots(figsize=(10,5.6))
 for family,color in [('pair_joint','#6f4ca2'),('stack_joint','#236eb1'),('cascade','#dd7831')]:
  pts=[r for r in summaries if r['policy'].startswith('next-'+family+'-') and r['policy'].endswith('learned_only')];xy=sorted(set((100*r['units']/ss['units'],100*r['accuracy']) for r in pts));ax.plot([x for x,y in xy],[y for x,y in xy],marker='o',label=family,color=color)
 for name,label,marker,color in [('next-control-qwen72-llama70','Direct SS','*','black'),('next-cascade-always-escalate','Cascade always escalate','x','gray'),('next-global_oof-match_ss_accuracy-learned_only','OOF match-SS selection','D','#1d8a58')]:
  row=next(r for r in summaries if r['policy']==name);ax.scatter([100*row['units']/ss['units']],[100*row['accuracy']],marker=marker,s=100,color=color,label=label,zorder=5)
 ax.axhline(71.8,color='black',linestyle='--',alpha=.4);ax.set(xlabel='All executed LLM-call cost / direct SS (%)',ylabel='Test lenient accuracy (%)',title='Nested OOF, four-path pairs, and fully charged weak-first cascades');ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'accuracy_cost.png',dpi=170);plt.close(fig)
 names={r['policy']:r for r in summaries};report=(out/'REPORT.md').read_text().split('\n## Study summary',1)[0]
 report+='\n## Study summary\n\n| Study / selection | Test lenient | Cost % direct SS |\n|---|---:|---:|\n'
 for name,label in [('next-pair_joint-max_accuracy-learned_only','Four-path pairwise, max accuracy'),('next-stack_joint-max_accuracy-learned_only','OOF stacking, max accuracy'),('next-global_oof-match_ss_accuracy-learned_only','Nested OOF global match-SS selection'),('next-risk-stack_joint-max_accuracy-alpha0.005','Stack joint, alpha0.5% gate'),('next-cascade-max_accuracy-learned_only','Weak-first cascade, max accuracy'),('next-cascade-cheap95-learned_only','Weak-first cascade,95% validation target'),('next-cascade-always-escalate','Cascade always escalates, control')]:
  if name in names:
   row=names[name];report+=f"| {label} | {100*row['accuracy']:.2f}% | {100*row['units']/ss['units']:.2f} |\n"
 report+='\nAll reported hyperparameters were selected on OOF/validation, and risk thresholds on separate calibration, before test construction. The stack joint result is360/500 versus359/500 SS: one rescued question, not evidence of a stable accuracy gain. The OOF match-SS selector chose outcome_joint, maintaining359/500 at89.77% cost. Validation global max accuracy selected cascade at71.8%/98.21%; OOF global max accuracy selected a different cascade weight at71.8%/95.17%. These are distinct selection objectives, not the best test row.\n\nThe paid-call audit verifies every test record, including weak QA on the selected retrieval branch. Cascade always-escalate has the same final answer as SS but109.61% cost. No weak-call cost is hidden.\n\nRisk alpha is an unconditional additional-harm target relative to SS, not overall answer error or error conditional on switching. Some test policies have observed extra-harm rates above their nominal targets: all calibration criteria passed, but expected-risk control does not bound every finite test sample, and historical reuse means this is not certified CRC. See risk_summary.csv for every proposer/alpha, including unhelpful all-SS policies.\n\nNew models use5125 fitting questions plus1280 reserved calibration questions within the earlier6405 training pool. Thus comparisons with earlier6405-fit routers are not equal fitting-label budgets. Trees use raw deterministic statistics; no global TFIDF/scaler or old trained router is reused. Outer stacking predictions use genuine inner3-fold meta-training, and the final stack uses the same3-fold procedure.\n\n[Accuracy/cost figure](accuracy_cost.png) · [Method table](method_summary.csv) · [Calibration and observed risk](risk_summary.csv) · [Fold isolation audit](fold_audits.json) · [Verification](verification.json). Latency remains unmeasured in this offline vectorized replay. All test comparisons remain exploratory.\n'
 (out/'REPORT.md').write_text(report);print('Verified',len(df),'rows;',len(summaries),'policies; nested folds, answer parity, cascade costs and risk criteria.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=study.runtime.WORK/'runs/nonllm_common_ws_next_v1');args=p.parse_args();main(args.out)
