"""Static comparison of validation-selected operating points."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[4]

def main():
    old=json.load(open(ROOT/'runs/nonllm_grpo_extended_v1/results.json'))
    a=json.load(open(ROOT/'runs/nonllm_architecture_v1/results.json'))
    c=json.load(open(ROOT/'runs/nonllm_architecture_combined_v1/results.json'))
    fig,ax=plt.subplots(figsize=(9,5.5))
    points=[('Direct SS',100,100*a['SS_accuracy']),
        ('Previous GRPO',114.6674866,58.338960),
        ('Pre-routing / SS budget',a['results']['direct4_ss_budget']['test']['cost_pct_SS'],100*a['results']['direct4_ss_budget']['test']['accuracy']),
        ('Pre-routing / cheap95',a['results']['direct4_cheap95']['test']['cost_pct_SS'],100*a['results']['direct4_cheap95']['test']['accuracy']),
        ('Answer chooser',a['results']['post_answer_chooser_max_accuracy']['test']['cost_pct_SS'],100*a['results']['post_answer_chooser_max_accuracy']['test']['accuracy'])]
    for obj,r in c['results'].items():points.append(('Combined / '+obj,r['test']['cost_pct_SS'],100*r['test']['accuracy']))
    for j,(label,cost,acc) in enumerate(points):
        ax.scatter(cost,acc,s=65,marker='*' if j==0 else 'o',label=f'{label}: {acc:.2f}% / {cost:.2f}%')
    ax.axvline(100,color='gray',linestyle='--',alpha=.5);ax.axhline(100*a['SS_accuracy'],color='gray',linestyle='--',alpha=.5)
    ax.set_xlabel('Inference cost (% of direct SS; parameter-weighted token proxy)')
    ax.set_ylabel('Test lenient accuracy (%)')
    ax.set_title('Non-LLM routers: architecture changes and paid answer selection')
    ax.grid(alpha=.15);ax.legend(loc='best',fontsize=8)
    fig.text(.01,.01,'Operating points selected on validation. Previously examined test set; exploratory results.',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1))
    fig.savefig(ROOT/'runs/nonllm_architecture_combined_v1/comparison.png',dpi=160)

if __name__=='__main__':main()
