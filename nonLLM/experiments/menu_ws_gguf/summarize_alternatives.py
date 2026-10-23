"""Verify and summarize validation-selected alternative routers; never selects on test."""
import argparse,json
from pathlib import Path
import numpy as np,pandas as pd
from stair_common.schema import validate
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import alternatives as experiment

def main(out):
 df=pd.read_csv(out/'results.csv');validate(df);results=json.loads((out/'results.json').read_text());summaries=results['summaries'];ss=results['ss']
 assert len(summaries)==76 and len(df)==38000
 for r in summaries:
  chosen=df[df.policy==r['policy']];assert len(chosen)==500 and chosen.id.nunique()==500
  assert abs(chosen.lenient.mean()-r['accuracy'])<1e-12 and abs(chosen.units.mean()-r['units'])<1e-8
 neural=[json.loads(p.read_text()) for p in (out/'training').glob('*/complete.json')];trees=[json.loads(p.read_text()) for p in (out/'trees').glob('*/complete.json')];assert len(neural)==54 and len(trees)==24
 experiment.runtime.write(out/'training_summary.json',{'neural':neural,'trees':trees})
 baseline=df[df.policy=='alternative-control-qwen72-llama70'].set_index('id');paired=[];rows=[];rng=np.random.default_rng(20261024)
 for r in summaries:
  if not r['policy'].endswith('learned_only'):continue
  chosen=df[df.policy==r['policy']].set_index('id').loc[baseline.index];delta=(chosen.lenient-baseline.lenient).to_numpy()
  bootstrap=np.array([delta[rng.integers(0,500,500)].mean()*100 for _ in range(4000)])
  gain=int(((chosen.lenient==1)&(baseline.lenient==0)).sum());loss=int(((chosen.lenient==0)&(baseline.lenient==1)).sum())
  paired.append({'policy':r['policy'],'gain':gain,'loss':loss,'delta_pp':float(delta.mean()*100),'paired_descriptive95_pp':np.quantile(bootstrap,[.025,.975]).tolist()})
  rows.append({'policy':r['policy'],'accuracy_pct':100*r['accuracy'],'correct':int(chosen.lenient.sum()),'cost_pct_ss':100*r['units']/ss['units'],'ss_routes':r['routes']['qwen72-llama70'],'selected_config':r['selected']['config']})
 pd.DataFrame(rows).to_csv(out/'method_summary.csv',index=False,lineterminator='\n');experiment.runtime.write(out/'verification.json',{'rows':len(df),'policies':len(summaries),'schema_and_aggregate_parity':True,'neural_streams':len(neural),'tree_streams':len(trees),'ss_correct':359,'test_construction_after_validation_lock':True,'paired':paired,'note':'Intervals are descriptive, unadjusted for multiple validation/test explorations. This previously examined test is not a fresh confirmation set.'})
 fig,ax=plt.subplots(figsize=(10,6));palette=plt.get_cmap('tab10');methods=sorted({r['selected']['method'] for r in summaries if r['policy'].endswith('max_accuracy-learned_only') and '-global-' not in r['policy']})
 for i,method in enumerate(methods):
  selected=[r for r in summaries if r['policy'].startswith('alternative-'+method+'-') and r['policy'].endswith('learned_only')]
  xy=sorted(set((100*r['units']/ss['units'],100*r['accuracy']) for r in selected))
  ax.plot([v[0] for v in xy],[v[1] for v in xy],marker='o',linewidth=.8,alpha=.8,color=palette(i),label=method.replace('_',' '))
 ax.axhline(71.8,color='black',linestyle='--',alpha=.5);ax.scatter([100],[71.8],marker='*',s=180,color='black',label='SS / prior GRPO')
 ax.scatter([83.79],[71.4],marker='^',s=65,color='gray',label='Prior exact reward')
 global_winner=next(r for r in summaries if r['policy']=='alternative-global-max_accuracy-learned_only')
 ax.scatter([100*global_winner['units']/ss['units']],[100*global_winner['accuracy']],marker='D',s=85,facecolors='white',edgecolors='black',label='Global validation winner',zorder=6)
 ax.set(xlabel='Measured executor cost / SS (%)',ylabel='Test lenient accuracy (%)',title='Exploratory alternatives: per-family validation-selected operating points')
 ax.grid(alpha=.2);ax.legend(loc='lower right',fontsize=8);fig.tight_layout();fig.savefig(out/'accuracy_cost.png',dpi=170);plt.close(fig)
 names={r['policy']:r for r in summaries};report=(out/'REPORT.md').read_text().split('\n## Per-family maximum-accuracy selections',1)[0]
 report+='\n## Per-family maximum-accuracy selections\n\nThese configurations were chosen within each family on validation, not chosen by test accuracy. Selecting the family itself globally on validation gives a different result below.\n\n| Method | Test lenient | Cost % SS | Correct /500 |\n|---|---:|---:|---:|\n'
 for method in methods:
  r=names['alternative-'+method+'-max_accuracy-learned_only'];report+=f"| {method} | {100*r['accuracy']:.2f}% | {100*r['units']/ss['units']:.2f} | {round(500*r['accuracy'])} |\n"
 report+='\n## Global family selection on validation\n\n| Validation objective | Chosen family | Test lenient | Cost % SS |\n|---|---|---:|---:|\n'
 for objective in experiment.OBJECTIVES:
  r=names['alternative-global-'+objective+'-learned_only'];report+=f"| {objective} | {r['selected']['method']} | {100*r['accuracy']:.2f}% | {100*r['units']/ss['units']:.2f} |\n"
 report+='\nThe per-family outcome-tree result is362/500 versus SS359/500, with cost96.77% SS. The expected-baseline policy is360/500 at96.54% SS. These are small observed gains, not evidence of a stable improvement. The global validation maximum-accuracy winner is outcome_linear (356/500,85.61% SS cost); do not relabel the best test family as the preselected global winner.\n\nThe augmented exact-reward family maximum-accuracy policy preserves359/500 at89.25% SS cost; its per-family match-validation-SS policy is356/500 at79.04% SS cost. Matching validation SS does not guarantee matching test SS.\n\n[Accuracy/cost plot](accuracy_cost.png) · [Compact method table](method_summary.csv) · [Paired descriptive differences](verification.json). Test reuse and the many validation comparisons prevent confirmatory significance/equivalence claims.\n'
 (out/'REPORT.md').write_text(report)
 print('Verified38000 rows,76 policies,54 neural streams and24 tree configurations; summaries and plot written.')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=experiment.runtime.WORK/'runs/nonllm_common_ws_alternatives_v1');args=p.parse_args();main(args.out)
