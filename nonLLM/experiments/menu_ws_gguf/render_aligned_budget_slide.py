"""English report figure: two operating points, paid weak calls, and training recipes."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from render_aligned_model_slide import card,NAVY,MUTED,BORDER,BG,COLORS

def render(run,out):
    assert json.loads((run/'status.json').read_text())['phase']=='complete'
    rows=json.loads((run/'results.json').read_text());assert len(rows)==18
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(16,9),facecolor=BG);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
    ax.text(.55,8.58,'Non-LLM Routing: Accuracy–Cost Trade-offs',fontsize=25,fontweight='bold',color=NAVY)
    ax.text(.57,8.19,'Jonathan-aligned outputs · HotpotQA: 4,814 train / 1,110 validation / 1,481 test · Shared standard lenient',fontsize=10.8,color=MUTED)
    ax.text(.57,7.94,'Qwen2.5-7B / 72B retrieval + Llama-3.2-3B / Llama-3.1-70B answering · Three-seed probability ensembles',fontsize=10,color=MUTED)
    card(ax,.55,7.35,14.9,.40,face=NAVY,edge=NAVY,r=.07)
    ax.text(.76,7.55,'SS BASELINE',fontsize=10.5,fontweight='bold',color='white',va='center')
    ax.text(3.0,7.55,'76.77% accuracy',fontsize=12,fontweight='bold',color='white',va='center')
    ax.text(6.5,7.55,'100.00% cost  =  929 Table 5 budget units',fontsize=10.7,color='white',va='center')
    ax.text(15.21,7.55,'Direct strong retrieval + strong QA',fontsize=9.6,color='#DCE8F8',ha='right',va='center')
    starts=[.55,5.6,10.65];width=4.8
    arches=[('linear','Linear Softmax','TF-IDF + statistics → 2-way linear head'),
            ('mlp64','MLP · 64 hidden units','TF-IDF + statistics → 64 ReLU → 2'),
            ('mlp128_64','MLP · 128 → 64','TF-IDF + statistics → 128 → 64 → 2')]
    for x,(arch,title,subtitle) in zip(starts,arches):
        card(ax,x,4.47,width,2.70)
        ax.text(x+.2,6.84,title,fontsize=17,fontweight='bold',color=NAVY)
        ax.text(x+.2,6.59,subtitle,fontsize=8.5,color=MUTED)
        ax.text(x+2.48,6.27,'Accuracy-first',fontsize=9.4,fontweight='bold',color=NAVY,ha='center')
        ax.text(x+4.12,6.27,'SS-budget*',fontsize=9.4,fontweight='bold',color=NAVY,ha='center')
        for pos,title in [(.2,'Training'),(2.14,'Acc. %'),(2.95,'Cost %'),(3.82,'Acc. %'),(4.61,'Cost %')]:
            ax.text(x+pos,6.04,title,fontsize=8,color=MUTED,ha='left' if pos==.2 else 'right')
        ax.plot([x+.2,x+4.6],[5.93,5.93],color=BORDER,lw=.8)
        ax.plot([x+3.13,x+3.13],[4.90,6.32],color=BORDER,lw=.8)
        for y,method in zip([5.68,5.29,4.90],COLORS):
            ax.scatter([x+.23],[y+.01],s=21,color=COLORS[method],zorder=3)
            ax.text(x+.40,y,method,fontsize=9.7,color=NAVY,va='center')
            for objective,ap,cp in [('max_accuracy',2.14,2.95),('ss_budget',3.82,4.61)]:
                r=next(r for r in rows if r['architecture']==arch and r['training']==method and r['objective']==objective)
                ax.text(x+ap,y,f"{r['accuracy_pct']:.2f}",fontsize=11.8,fontweight='bold',color=NAVY,ha='right',va='center')
                color='#A65B16' if r['cost_pct_ss']>100 else MUTED
                ax.text(x+cp,y,f"{r['cost_pct_ss']:.2f}",fontsize=10.9,color=color,ha='right',va='center')
        counts=[next(r for r in rows if r['architecture']==arch and r['training']==m and r['objective']=='max_accuracy')['route_counts']['SS'] for m in COLORS]
        ax.text(x+.2,4.66,f'Accuracy-first SS:  SFT {counts[0]:,}  ·  Reward {counts[1]:,}  ·  GRPO {counts[2]:,}',fontsize=7.9,color=MUTED)
    ax.text(.58,4.29,'Accuracy-first: highest validation accuracy, cost tie-break.   SS-budget*: highest validation accuracy with validation cost ≤ 100% SS.',fontsize=9.3,color=MUTED)
    card(ax,.55,3.29,14.9,.82,face='#EAF0F8',edge='#D3DFEF',r=.08)
    ax.text(.75,3.87,'WHY CAN COST EXCEED 100%?',fontsize=10.4,fontweight='bold',color=NAVY)
    ax.text(.75,3.62,'Direct SS: 814 + 115 = 929  (100%)',fontsize=10.2,color=NAVY)
    ax.text(6.3,3.62,'Weak-first SS: 77 + 814 + 3 + 115 = 1009  (108.61%)',fontsize=10.2,fontweight='bold',color=NAVY)
    ax.text(.75,3.40,'Accuracy-first uses SS on 96–100% of questions and also pays for weak calls. Budgeted MLP-128→64 SFT matches SS accuracy at 98.18% cost.',fontsize=9.0,color=MUTED)
    methods=[
      ('01','SFT',[
       'Learn W/S labels with weighted cross-entropy.',
       'Retrieval: coverage in weak selected evidence.',
       'QA: keep correct weak answers; upgrade if rescued.',
       'Both-wrong cases map to W.',
       '3 epochs · lr 1e−4 · checkpoint by val AUC']),
      ('02','SFT + Reward',[
       'Start from SFT; score both cached W/S actions.',
       'R = token F1 − 0.1 × C_train / 929',
       'Maximize exact expected R + 0.03 × entropy.',
       'Refine both stages; RET uses best downstream QA R.',
       '3 epochs · lr 1e−5 · checkpoint by val reward']),
      ('03','GRPO',[
       'Start from SFT; sample 8 actions per row.',
       'Group-normalized advantages + policy gradient.',
       'Add 0.02 KL to frozen SFT reference; no PPO clip.',
       'Stage adaptation of Jonathan’s public code.',
       '2 epochs · lr 1e−4 · checkpoint by val reward'])]
    for x,(number,method,lines) in zip(starts,methods):
        card(ax,x,1.19,width,1.91)
        ax.text(x+.2,2.80,number,fontsize=11.5,fontweight='bold',color=COLORS[method])
        ax.text(x+.65,2.80,method,fontsize=15,fontweight='bold',color=COLORS[method])
        for y,line in zip([2.44,2.20,1.96,1.72,1.40],lines):
            ax.text(x+.2,y,line,fontsize=8.5 if y!=1.40 else 8.2,color=NAVY if y!=1.40 else MUTED)
    foot=[
      '* Budget constraints apply on validation; test cost is observed, not guaranteed. Both operating points use the same trained model. Amber costs exceed SS.',
      'Cost excludes decision-model compute. Reward prices follow the author: [80,195,894,929]; actual weak-first evaluation: [80,195,894,1009]. GRPO is an adaptation, not a verified slide reproduction.',
      'Previously inspected test: descriptive results. 76.84% is only one additional correct answer vs SS. Train-fitted features only; no LLM encoder or generation.']
    for y,line in zip([.89,.61,.33],foot):ax.text(.58,y,line,fontsize=7.9,color=MUTED)
    stem=out/'nonllm_jonathan_aligned_comparison_v2'
    for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=240,facecolor=BG)
    plt.close(fig)
    stem.with_suffix('.json').write_text(json.dumps({'source':str(run.resolve()),'source_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),'data':rows,'size':'3840x2160'},indent=2))
    print(stem.with_suffix('.png'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.run,a.out)
