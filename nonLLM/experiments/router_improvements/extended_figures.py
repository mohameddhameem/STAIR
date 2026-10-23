"""Render training diagnostics and final fixed-policy comparisons after completion."""
import argparse
import json
from pathlib import Path
import time

import numpy as np


def render(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    modes=['base','confidence','aug5','aug9'];seeds=[17,29,43]
    baseline=100*np.load(out/'data/base/em_val.npy')[:,3].mean()
    fig,axes=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    figz,axesz=plt.subplots(2,2,figsize=(11,8),sharex=True,sharey=True)
    for mode,ax,az in zip(modes,axes.flat,axesz.flat):
        for method,g in [('exact',0),('grpo',8),('grpo',16),('grpo',32)]:
            runs=[]
            for seed in seeds:
                f=out/f'training/{mode}_{method}_g{g}_lambda0.0_seed{seed}/curve.jsonl'
                runs.append([json.loads(l) for l in f.open()])
            epochs=np.array([r['epoch'] for r in runs[0]])
            y=np.array([[100*r['val_expected_accuracy'] for r in run] for run in runs])
            label='Exact reward' if method=='exact' else f'GRPO G={g}'
            ax.plot(epochs,y.mean(0),label=label);ax.fill_between(epochs,y.min(0),y.max(0),alpha=.08)
            if method=='grpo':
                z=np.array([[100*r['zero_advantage_fraction'] for r in run] for run in runs])
                az.plot(epochs,z.mean(0),label=label)
        ax.axhline(baseline,c='black',ls='--',lw=1,label='SS validation');ax.set_title(mode);ax.grid(alpha=.2)
        az.set_title(mode);az.grid(alpha=.2)
    for ax in axes[-1]:ax.set_xlabel('Policy training epoch')
    for ax in axes[:,0]:ax.set_ylabel('Expected validation accuracy (%)')
    for ax in axesz[-1]:ax.set_xlabel('Policy training epoch')
    for ax in axesz[:,0]:ax.set_ylabel('Zero-advantage groups (%)')
    axes[0,0].legend(fontsize=8);axesz[0,0].legend(fontsize=8)
    fig.suptitle('Accuracy-focused training (lambda=0), three seeds');fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=180)
    figz.suptitle('GRPO reward variation (lambda=0), mean across three seeds');figz.tight_layout();figz.savefig(out/'zero_advantage.png',dpi=180)
    plt.close(fig);plt.close(figz)
    result=json.load(open(out/'results.json'))
    fig,ax=plt.subplots(figsize=(8,5))
    colors={'base':'#72b7b2','confidence':'#4c78a8','aug5':'#54a24b','aug9':'#e45756'}
    seen=set()
    for r in result['results']:
        m=r['selected_config']['mode'];key=(m,r['selected_config']['method'])
        ax.scatter(r['test']['cost_pct_of_direct_SS'],100*r['test']['lenient_accuracy'],
                   c=colors[m],marker='o' if key[1]=='exact' else '^',s=50,label=f'{m} / {key[1]}' if key not in seen else None)
        seen.add(key)
    ax.scatter([100],[100*result['ss_test_accuracy']],c='black',marker='*',s=160,label='Direct SS')
    ax.set_xlabel('Token-cost proxy (% of direct SS)');ax.set_ylabel('Test lenient accuracy (%)');ax.grid(alpha=.2);ax.legend(fontsize=8)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=180);plt.close(fig)
    block='\n\n![Training curves](learning_curves.png)\n\n![Zero-advantage groups](zero_advantage.png)\n\n![Final comparison](comparison.png)\n'
    report=out/'REPORT.md'
    if '![Training curves]' not in report.read_text():report.write_text(report.read_text()+block)
    (out/'figures_complete.json').write_text(json.dumps({'phase':'complete','figures':['learning_curves.png','zero_advantage.png','comparison.png']}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--wait',action='store_true');a=p.parse_args()
    if a.wait:
        while True:
            status=json.load(open(a.out/'status.json'))
            if status['phase']=='failed':raise RuntimeError(status)
            if status['phase']=='complete':break
            time.sleep(10)
    render(a.out)
