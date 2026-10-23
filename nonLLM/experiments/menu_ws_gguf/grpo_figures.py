"""Training diagnostics for the corrected-cache finite-action GRPO experiment."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import grpo

def main(out):
 records=[json.loads(p.read_text()) for p in sorted((out/'training').glob('*/complete.json'))]
 assert len(records)==36
 grpo.runtime.write(out/'training_summary.json',{'streams':records,'warm':[{'seed':seed,'epochs':json.loads(sorted((out/'warm'/f'seed{seed}').glob('curve*.json'))[0].read_text())} for seed in grpo.SEEDS]})
 fig,axes=plt.subplots(1,2,figsize=(11,4.2))
 for method,group,color in [('grpo',8,'#3159b8'),('grpo',32,'#be5930'),('exact',0,'#24835b')]:
  series=[]
  for seed in grpo.SEEDS:
   folder=out/'training'/grpo.tag({'method':method,'group':group,'lam':0.,'seed':seed})
   curve=[json.loads(line) for line in (folder/'curve.jsonl').read_text().splitlines()];series.append({r['epoch']:r for r in curve})
   for ax,key,factor in [(axes[0],'val_expected_accuracy',100),(axes[1],'val_expected_cost_pct_ss',1)]:ax.plot([r['epoch'] for r in curve],[factor*r[key] for r in curve],color=color,alpha=.18,linewidth=.8)
  epochs=sorted(set.intersection(*(set(s) for s in series)))
  label=f'GRPO G={group}' if method=='grpo' else 'Exact expectation'
  for ax,key,factor in [(axes[0],'val_expected_accuracy',100),(axes[1],'val_expected_cost_pct_ss',1)]:ax.plot(epochs,[np.mean([s[e][key] for s in series])*factor for e in epochs],color=color,label=label,linewidth=2)
 for ax in axes:ax.set_xlabel('RL epoch');ax.grid(alpha=.2);ax.legend(frameon=False,fontsize=9)
 axes[0].set_ylabel('Validation expected lenient accuracy (%)');axes[1].set_ylabel('Expected cost / training-mean SS (%)')
 fig.suptitle('Four-model non-LLM policy training: lambda = 0, three seeds');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=170);plt.close(fig)
 print('Training summary and learning_curves.png written')

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=grpo.runtime.WORK/'runs/nonllm_common_ws_grpo_v1');a=p.parse_args();main(a.out)
