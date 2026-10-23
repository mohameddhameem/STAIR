"""Post-run analysis with both direct and cascade fixed-policy cost baselines.

This never selects methods on test: the primary method is selected using the
locked validation operating point. Test mixture frontiers are descriptive.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parent))
from suite import fixed_mixture_accuracy


def analyze(base,out):
    rows=[json.load(open(p)) for p in (out/'results').glob('*/results.json')]
    if not all('test' in r for r in rows):raise ValueError('Pending test comparisons')
    source=json.load(open(base/'router/results.json'))
    c=json.load(open(base/'token_costs.json'))['gate_eval_constants']
    direct=np.array([c['COST_RET_W']+c['COST_QA_W_GIVEN_W'],c['COST_RET_W']+c['COST_QA_S_GIVEN_W'],
                     c['COST_RET_S']+c['COST_QA_W_GIVEN_S'],c['COST_RET_S']+c['COST_QA_S_GIVEN_S']])
    cascade=np.array([source['fixed_test'][k]['cascade_cost'] for k in ['WW','WS','SW','SS']])
    accuracy=np.array([source['fixed_test'][k]['accuracy'] for k in ['WW','WS','SW','SS']])
    selected=min(rows,key=lambda r:(r['val']['mean_cost'],-r['val']['lenient_accuracy'],r['name']))
    baseline=next(r for r in rows if r['name']=='frozen_baseline')
    status=json.load(open(out/'status.json')) if (out/'status.json').exists() else {}
    complete=status.get('phase')=='complete'
    text=['# Non-LLM router improvements: '+('complete' if complete else 'partial')+' comparison','',
        f"Completed comparisons: {len(rows)} / 27. Run phase: {status.get('phase','unknown')}.",'',
        f"Primary method selected on validation: **{selected['name']}**.",'',
        'Every configuration and its operating thresholds were fixed using train/validation. The previously used test set is descriptive; no new confirmatory holdout was collected.',
        'All cost percentages use direct SS as denominator. Costs count sunk weak calls and any extra sampling, using parameter-weighted token proxies; CPU routing overhead is excluded.',
        'Two fixed-policy frontiers are reported. Direct policies skip unused weak calls when their model choices are fixed; cascade policies still run weak first. The direct frontier is the stronger cost-matched comparator.',
        'The previously quoted +1.44 pp for the frozen baseline applies only to the cascade fixed-policy frontier. Against direct fixed policies its descriptive gain is about -0.14 pp.',
        '', '| Method | Val cost % SS | Test lenient % | Test strict % | Test cost % SS | Gain vs direct mixture (pp) | Gain vs cascade mixture (pp) |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in sorted(rows,key=lambda r:r['name']):
        t=r['test'];dg=100*(t['lenient_accuracy']-fixed_mixture_accuracy(t['mean_cost'],direct,accuracy))
        cg=100*(t['lenient_accuracy']-fixed_mixture_accuracy(t['mean_cost'],cascade,accuracy))
        r['descriptive_gain_vs_direct_mixture_pp']=dg;r['descriptive_gain_vs_cascade_mixture_pp']=cg
        text.append(f"| {r['name']} | {r['val']['cost_pct_of_direct_SS']:.2f} | {100*t['lenient_accuracy']:.2f} | {100*t['strict_accuracy']:.2f} | {t['cost_pct_of_direct_SS']:.2f} | {dg:+.2f} | {cg:+.2f} |")
    text += ['','![Accuracy and token-cost comparison](comparison.png)','',
        'GRPO is a finite-action group-relative PPO adaptation of linear TF-IDF/numeric policies, with three fixed seeds averaged. No LLM router or Qwen fine-tuning was performed.',
        'Augmentation policies consume one greedy answer at deployment. Consensus policies consume one greedy plus four sampled weak answers and pay for all five.',
        'Confidence statistics were collected with teacher forcing for cached greedy answers. The reported deployment cost assumes those statistics can be extracted from the ordinary weak generation logits. If a separate model forward pass is required in deployment, its cost must be added.',
        'Paired accuracy intervals and raw model artifacts are available in REPORT.md and per-method folders. Differences across the full exploratory suite are not corrected for multiple comparisons.']
    (out/'results/COMPARISON.md').write_text('\n'.join(text)+'\n')
    (out/'results/comparison.json').write_text(json.dumps({'selected_on_val':selected['name'],'direct_prices':direct.tolist(),
        'cascade_prices':cascade.tolist(),'fixed_accuracies':accuracy.tolist(),'results':rows},indent=2))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.7))
    seen=set()
    for r in rows:
        n=r['name'];group='GRPO' if n.startswith('grpo') else 'Expected reward' if n.startswith('exact') else 'Consensus (5 calls)' if 'consensus' in n else 'Training augmentation' if 'augmentation' in n else 'Confidence' if n.startswith('confidence') else 'Baseline' if n=='frozen_baseline' else 'Features / labels'
        colors={'GRPO':'#e58606','Expected reward':'#8e6c8a','Consensus (5 calls)':'#e45756','Training augmentation':'#54a24b','Confidence':'#4c78a8','Baseline':'black','Features / labels':'#72b7b2'}
        for j,key in enumerate(['lenient_accuracy','strict_accuracy']):
            axes[j].scatter(r['test']['cost_pct_of_direct_SS'],100*r['test'][key],c=colors[group],s=40,
                            marker='*' if n==selected['name'] else 'o',label=group if j==0 and group not in seen else None)
        seen.add(group)
    allcost=np.array([r['test']['mean_cost'] for r in rows])
    xs=np.linspace(allcost.min()*.97,allcost.max()*1.03,100)
    axes[0].plot(100*xs/source['direct_SS_cost'],[100*fixed_mixture_accuracy(x,direct,accuracy) for x in xs],c='gray',ls='--',label='Direct fixed mixture')
    for ax,title in zip(axes,['Lenient answer accuracy','Strict answer accuracy']):
        ax.set_xlabel('Token-cost proxy (% of direct SS)');ax.set_ylabel('Accuracy (%)');ax.set_title(title);ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'results/comparison.png',dpi=180);plt.close(fig)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.wait:
        while True:
            f=a.out/'status.json';status=json.load(open(f)) if f.exists() else {}
            if status.get('phase')=='failed':raise RuntimeError(status)
            if status.get('phase')=='complete':break
            time.sleep(10)
    analyze(a.base,a.out)
