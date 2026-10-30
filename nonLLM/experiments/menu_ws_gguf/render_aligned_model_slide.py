"""Presentation figure with exact experiment numbers and editable vector exports."""
import argparse,json,hashlib
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY='#14263D';MUTED='#56657A';BORDER='#DCE3ED';BG='#F5F7FB'
COLORS={'SFT':'#2563EB','SFT + Reward':'#087F8C','GRPO':'#7C3AED'}

def card(ax,x,y,w,h,face='white',edge=BORDER,r=.12):
    shape=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',linewidth=.8,edgecolor=edge,facecolor=face)
    ax.add_patch(shape);return shape

def render(run,out):
    assert json.loads((run/'status.json').read_text())['phase']=='complete'
    data=json.loads((run/'results.json').read_text());assert len(data)==9
    verification=json.loads((run/'verification.json').read_text());assert verification['audited_prediction_rows']==9*1481
    out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(16,9),facecolor=BG);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,9));ax.axis('off')
    ax.text(.55,8.52,'Non-LLM Routing: Three Training Recipes',fontsize=25,fontweight='bold',color=NAVY)
    ax.text(.57,8.13,'Jonathan-aligned Config B outputs  |  HotpotQA: 4,814 train / 1,110 validation / 1,481 test',fontsize=11.1,color=MUTED)
    ax.text(.57,7.84,'Qwen2.5-7B / 72B retrieval  +  Llama-3.2-3B / Llama-3.1-70B answering  |  Shared standard lenient accuracy',fontsize=10,color=MUTED)
    card(ax,.55,7.21,14.9,.42,face=NAVY,edge=NAVY,r=.07)
    ax.text(.77,7.42,'SS BASELINE',fontsize=10.5,fontweight='bold',color='white',va='center')
    ax.text(3.0,7.42,'76.77% accuracy',fontsize=12,fontweight='bold',color='white',va='center')
    ax.text(6.5,7.42,'100.00% cost  (929 Table 5 budget units)',fontsize=10.8,color='white',va='center')
    ax.text(15.2,7.42,'Direct strong + strong',fontsize=10,color='#DCE8F8',ha='right',va='center')
    names=[('linear','Linear Softmax','TF-IDF + statistics → 2-way linear head'),
           ('mlp64','MLP · 64 hidden units','TF-IDF + statistics → 64 ReLU → 2'),
           ('mlp128_64','MLP · 128 → 64','TF-IDF + statistics → 128 → 64 → 2')]
    starts=[.55,5.6,10.65];width=4.8
    for x,(arch,title,subtitle) in zip(starts,names):
        card(ax,x,4.13,width,2.85)
        ax.text(x+.2,6.62,title,fontsize=17,fontweight='bold',color=NAVY)
        ax.text(x+.2,6.34,subtitle,fontsize=8.6,color=MUTED)
        ax.text(x+.2,5.98,'Training',fontsize=9,color=MUTED)
        ax.text(x+3.16,5.98,'Accuracy',fontsize=9,color=MUTED,ha='right')
        ax.text(x+4.6,5.98,'Cost / SS',fontsize=9,color=MUTED,ha='right')
        ax.plot([x+.2,x+4.6],[5.84,5.84],color=BORDER,lw=.8)
        for y,method in zip([5.51,5.0,4.49],COLORS):
            row=next(r for r in data if r['architecture']==arch and r['training']==method)
            ax.scatter([x+.24],[y+.015],s=25,color=COLORS[method],zorder=3)
            ax.text(x+.43,y,method,fontsize=11.1,color=NAVY,va='center')
            ax.text(x+3.16,y,f"{row['accuracy_pct']:.2f}%",fontsize=14,fontweight='bold',color=NAVY,ha='right',va='center')
            ax.text(x+4.6,y,f"{row['cost_pct_ss']:.2f}%",fontsize=13,color=MUTED,ha='right',va='center')
    ax.text(.58,3.88,'All rows: two stage routers · three-seed probability ensemble · thresholds selected for maximum validation accuracy',fontsize=9.7,color=MUTED)
    explanations=[
      ('01','SFT',[
       'Learn W/S labels with weighted cross-entropy.',
       'Retrieval labels: coverage in weak evidence.',
       'QA labels: keep a correct weak answer;',
       'otherwise upgrade if the strong answer works.',
       'Unrecoverable cases map to W.',
       '3 epochs · lr 1e−4 · checkpoint by val AUC']),
      ('02','SFT + Reward',[
       'Start from SFT; evaluate both cached actions.',
       'R = token F1 − 0.1 × C_train / 929',
       'Optimize exact expected R + 0.03 × entropy.',
       'Retrieval uses the best downstream QA reward.',
       'Both stage routers are refined.',
       '3 epochs · lr 1e−5 · checkpoint by val reward']),
      ('03','GRPO',[
       'Start from SFT; sample 8 actions per row.',
       'Normalize rewards within each sampled group.',
       'Policy-gradient update + 0.02 reference KL.',
       'Frozen SFT reference; no PPO clipping.',
       'Stage adaptation of Jonathan’s public code.',
       '2 epochs · lr 1e−4 · checkpoint by val reward'])]
    for x,(num,method,lines) in zip(starts,explanations):
        card(ax,x,1.35,width,2.3)
        ax.text(x+.2,3.3,num,fontsize=12,fontweight='bold',color=COLORS[method])
        ax.text(x+.65,3.3,method,fontsize=15,fontweight='bold',color=COLORS[method])
        for y,line in zip([2.91,2.66,2.41,2.16,1.91,1.6],lines):
            ax.text(x+.2,y,line,fontsize=9.0 if y!=1.6 else 8.4,color=NAVY if y!=1.6 else MUTED)
    foot=[
      'Cost: Table 5 prices, excluding decision-model compute. Every executed weak call is charged; fully escalated SS costs 1009 / 929 = 108.61%.',
      'Reward uses the author’s training prices [80, 195, 894, 929]; evaluation uses [80, 195, 894, 1009]. GRPO is an adaptation, not a verified reproduction of the slide’s GRPO row.',
      'Same frozen executor outputs and split across models. Previously inspected test: descriptive results. Models use train-fitted features; no LLM encoder or generation.']
    for y,line in zip([.95,.65,.35],foot):ax.text(.58,y,line,fontsize=8.1,color=MUTED)
    stem=out/'nonllm_jonathan_aligned_comparison'
    fig.savefig(stem.with_suffix('.png'),dpi=240,facecolor=BG)
    fig.savefig(stem.with_suffix('.pdf'),facecolor=BG)
    fig.savefig(stem.with_suffix('.svg'),facecolor=BG)
    plt.close(fig)
    manifest={'title':'Non-LLM Routing: Three Training Recipes','source_results':str((run/'results.json').resolve()),
              'source_results_sha256':hashlib.sha256((run/'results.json').read_bytes()).hexdigest(),
              'rows':data,'exports':['PNG 3840×2160','PDF vector','SVG editable text'],
              'rendering':'Deterministic matplotlib; no generative alteration of numerical results'}
    stem.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
    print(stem.with_suffix('.png'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.run,a.out)
