"""English reporting figure: all tested non-LLM MedAgentBench router families.

Exact numbers are read from verified artifacts. No new fitting or test inference.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

NAVY='#14263D';MUTED='#56657A';BORDER='#DCE3ED';BG='#F5F7FB';PURPLE='#7940CA';TEAL='#087F8C'

def card(ax,x,y,w,h,face='white',edge=BORDER):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=.10',linewidth=.8,facecolor=face,edgecolor=edge))

def build_data(grpo,classic):
    for run in [grpo,classic]:assert json.loads((run/'status.json').read_text())['phase']=='complete'
    g=json.loads((grpo/'results.json').read_text());c=json.loads((classic/'results.json').read_text());ref=json.loads((grpo/'llm_reference.json').read_text())
    assert json.loads((classic/'verification.json').read_text())['all_336_aggregate_rows_verified']
    specs=[
      ('linear','Linear Softmax','GRPO (A1)','Linear softmax on TF-IDF + view statistics.'),
      ('mlp64','MLP · 64','GRPO (A1)','One ReLU hidden layer with 64 units.'),
      ('mlp128_64','MLP · 128 → 64','GRPO (A1)','Two ReLU hidden layers: 128 then 64 units.'),
      ('random_forest','Random Forest','Supervised','Average 300 bootstrapped decision trees.'),
      ('extra_trees','Extra Trees','Supervised','Average 300 trees with randomized splits.'),
      ('decision_tree','Decision Tree','Supervised','One tree of learned feature-split rules.'),
      ('linear_svm','Linear SVM','Supervised','Maximum-margin linear decision boundary.'),
      ('rbf_svm','RBF SVM','Supervised','Maximum-margin boundary with an RBF kernel.'),
      ('logistic_regression','Logistic Regression','Supervised','Linear classifier fitted by weighted logistic loss.'),
      ('multinomial_nb','Naive Bayes','Supervised','Class-conditional token likelihoods; text only.')]
    rows=[]
    rr=[next(v for v in ref if v['setting']==s and v['point']=='OP_A_fixed_050' and v['risk_gate']) for s in [1,3]]
    rows.append({'key':'jonathan','model':'Jonathan LLM','recipe':'GRPO (A1)','description':'Qwen2.5-1.5B + LoRA; pretrained backbone.',
                 't1_accuracy':rr[0]['accuracy']*100,'t1_cost':rr[0]['cost_pct_always_escalate'],
                 't3_accuracy':rr[1]['accuracy']*100,'t3_cost':rr[1]['cost_pct_always_escalate'],'repeat_range':None})
    for key,label,recipe,desc in specs:
        source=g if recipe.startswith('GRPO') else c;field='architecture' if recipe.startswith('GRPO') else 'family'
        values=[next(v for v in source if v['setting']==s and v[field]==key and v['seed']==17 and v['point']=='val_reward_selected' and v['risk_gate']) for s in [1,3]]
        repeats=[v['accuracy']*100 for v in source if v['setting']==3 and v[field]==key and v['seed'] in [0,1,2] and v['point']=='val_reward_selected' and v['risk_gate']]
        assert len(repeats)==3
        rows.append({'key':key,'model':label,'recipe':recipe,'description':desc,
             't1_accuracy':values[0]['accuracy']*100,'t1_cost':values[0]['cost_pct_always_escalate'],
             't3_accuracy':values[1]['accuracy']*100,'t3_cost':values[1]['cost_pct_always_escalate'],
             't1_threshold':values[0]['threshold'],'t3_threshold':values[1]['threshold'],
             'repeat_range':[min(repeats),max(repeats)]})
    published=[next(v for v in ref if v['setting']==s and v['point']=='OP_A_fixed_050' and not v['risk_gate']) for s in [1,3]]
    return rows,published

def render(grpo,classic,out):
    rows,published=build_data(grpo,classic);out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','pdf.fonttype':42,'svg.fonttype':'none'})
    fig=plt.figure(figsize=(16,12),facecolor=BG);ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,16),ylim=(0,12));ax.axis('off')
    ax.text(.55,11.56,'MedAgentBench: Non-LLM Routers vs Jonathan’s LLM',fontsize=23.5,fontweight='bold',color=NAVY)
    ax.text(.58,11.17,'Same replay views, outcomes and costs · Patient-disjoint split: 90 train / 60 validation / 150 test · Benchmark pass/fail accuracy',fontsize=10.6,color=MUTED)
    ax.text(.58,10.88,'T1: decide after the full weak attempt     |     T3: decide at round 2     |     Executors: Qwen2.5-7B (weak) / Qwen2.5-72B-AWQ (strong)',fontsize=10.1,color=MUTED)
    card(ax,.55,10.26,14.9,.40,face=NAVY,edge=NAVY)
    ax.text(.77,10.46,'PUBLISHED LLM · NO RISK GATE',fontsize=9.7,fontweight='bold',color='white',va='center')
    ax.text(5.45,10.46,f"T1  {published[0]['accuracy']*100:.2f}% accuracy / {published[0]['cost_pct_always_escalate']:.2f}% cost",fontsize=11,color='white',va='center')
    ax.text(10.43,10.46,f"T3  {published[1]['accuracy']*100:.2f}% accuracy / {published[1]['cost_pct_always_escalate']:.2f}% cost",fontsize=11,color='white',va='center')
    ax.text(.58,9.99,'MAIN TABLE: same high-risk escalation gate for every router · Seed 17 · Non-LLM thresholds chosen on validation; Jonathan uses θ = 0.50',fontsize=10.1,fontweight='bold',color=NAVY)
    card(ax,.55,4.72,14.9,5.02)
    ax.text(.77,9.38,'Router',fontsize=10,fontweight='bold',color=NAVY)
    ax.text(2.98,9.38,'Training',fontsize=10,fontweight='bold',color=NAVY)
    ax.text(4.65,9.38,'How the model works',fontsize=10,fontweight='bold',color=NAVY)
    ax.text(10.03,9.50,'Full attempt · T1',fontsize=10,fontweight='bold',color=NAVY,ha='center')
    ax.text(12.30,9.50,'Round 2 · T3',fontsize=10,fontweight='bold',color=NAVY,ha='center')
    ax.text(14.42,9.49,'T3 accuracy range',fontsize=9.3,fontweight='bold',color=NAVY,ha='center')
    ax.text(14.42,9.28,'seeds 0 / 1 / 2',fontsize=8.2,color=MUTED,ha='center')
    for x,text in [(9.67,'Acc. %'),(10.78,'Cost %'),(11.96,'Acc. %'),(13.07,'Cost %')]:ax.text(x,9.24,text,fontsize=8.6,color=MUTED,ha='right')
    ax.plot([.76,15.23],[9.12,9.12],lw=.8,color=BORDER)
    for x in [8.82,11.14,13.45]:ax.plot([x,x],[4.95,9.58],lw=.65,color=BORDER)
    for index,row in enumerate(rows):
        y=8.85-index*.36
        if index==0:card(ax,.67,y-.15,14.66,.34,face='#EAF0F8',edge='#EAF0F8')
        elif index%2==0:card(ax,.67,y-.15,14.66,.34,face='#F8FAFD',edge='#F8FAFD')
        color=PURPLE if row['recipe'].startswith('GRPO') else TEAL
        ax.text(.78,y,row['model'],fontsize=10.3,fontweight='bold' if index==0 else 'normal',color=NAVY,va='center')
        ax.text(2.98,y,row['recipe'],fontsize=9.5,fontweight='bold',color=color,va='center')
        ax.text(4.65,y,row['description'],fontsize=8.5,color=MUTED,va='center')
        for x,key in [(9.67,'t1_accuracy'),(10.78,'t1_cost'),(11.96,'t3_accuracy'),(13.07,'t3_cost')]:
            value=row[key];highlight=row['key'] in ['linear_svm','rbf_svm','logistic_regression'] and key in ['t3_accuracy','t3_cost']
            ax.text(x,y,f'{value:.2f}',fontsize=10.8,fontweight='bold' if highlight or index==0 else 'normal',color=TEAL if highlight else NAVY,ha='right',va='center')
        text='Not reported' if row['repeat_range'] is None else f"{row['repeat_range'][0]:.2f}–{row['repeat_range'][1]:.2f}"
        ax.text(15.11,y,text,fontsize=9,color=MUTED,ha='right',va='center')
        if index==3:ax.plot([.76,15.23],[y-.18,y-.18],lw=.8,color=BORDER)
    ax.text(.78,4.88,'Cost denominator: full weak + strong on the same 150 tasks = 100%.  Full always-escalate: 54.00% accuracy; two-action oracle: 63.33%.',fontsize=8.8,color=MUTED)
    explanations=[
      (.55,'GRPO (A1) · neural classifiers',PURPLE,[
       'Linear / MLP: policy-gradient training, as in the LLM recipe.',
       'Fresh classifier; sample 8 actions per task.',
       'Advantage = reward − group mean; entropy bonus 0.03.',
       '40 epochs · lr 1e−5 · accumulation 8 · gradient clip 1.',
       'No std normalization, KL term or PPO clipping.',
       'Best checkpoint selected by validation delivered reward.']),
      (5.60,'Supervised · classical classifiers',TEAL,[
       'Label = argmax(R_W, R_S); weight = |R_W − R_S|.',
       'Fit the classifier to reward-optimal routing actions.',
       'Hyperparameters and thresholds selected on validation.',
       'SVM: patient-grouped out-of-fold score calibration.',
       'This is supervised training, not GRPO.',
       'Same view features; Naive Bayes uses text only.']),
      (10.65,'Shared objective and readout',NAVY,[
       'R = Q − 0.1 × (C / c_s) − max(0, C / c_s − 1).',
       'Q is the benchmark’s pass/fail, not QA lenient match.',
       'T1 costs: deliver c_w; escalate c_w + c_s.',
       'T3 costs: deliver c_w; escalate c_w2 + c_s.',
       'Common gate: τ ≥ 0.8 forces escalation.',
       'Inputs: task text/context + observed weak attempt.'])]
    for x,title,color,lines in explanations:
        card(ax,x,2.07,4.8,2.38)
        ax.text(x+.20,4.10,title,fontsize=12.1,fontweight='bold',color=color)
        for y,line in zip([3.72,3.45,3.18,2.91,2.64,2.35],lines):ax.text(x+.2,y,line,fontsize=8.5,color=NAVY)
    card(ax,.55,.86,14.9,.98,face='#EAF4F2',edge='#D3E7E2')
    ax.text(.77,1.54,'SVM / logistic regression: 63.33% accuracy at 77.68% cost',fontsize=13.2,fontweight='bold',color=TEAL)
    ax.text(.77,1.20,'Same success rate as Jonathan’s T3; 3.79 cost points less. Repeats 0/1/2 agree.',fontsize=9.5,color=NAVY)
    ax.text(10.1,1.55,'Why? Avoid 15 ineffective strong upgrades.',fontsize=10.1,fontweight='bold',color=NAVY)
    ax.text(10.1,1.29,'Both models fail on those task7 cases; failures remain.',fontsize=8.9,color=MUTED)
    ax.text(10.1,1.04,'Lower compute does not mean improved clinical outcomes.',fontsize=8.7,color=MUTED)
    foot=[
      'Main table applies the same cached τ ≥ 0.8 gate to all rows. It changes the LLM’s T1 cost from 54.11% to 80.99%; T3 is unchanged. No test-based threshold selection.',
      'Routers actually run; executor outcomes are replayed, not rerun. Cost excludes router / risk-scorer compute. Previously inspected test; small, homogeneous task types; GRPO repeats vary.']
    for y,line in zip([.55,.28],foot):ax.text(.58,y,line,fontsize=7.8,color=MUTED)
    stem=out/'medagentbench_nonllm_vs_jonathan'
    for ext in ['png','pdf','svg']:fig.savefig(stem.with_suffix('.'+ext),dpi=240,facecolor=BG)
    plt.close(fig)
    manifest={'comparison_point':'seed17; nonLLM validation-selected threshold; LLM fixed.50; all common tau>=.8 gate',
        'data':rows,'published_llm_ungated':published,
        'source_hashes':{str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in [grpo/'results.json',grpo/'llm_reference.json',classic/'results.json']},
        'rendering':'Deterministic matplotlib, exact source metrics','size':'3840x2880'}
    stem.with_suffix('.json').write_text(json.dumps(manifest,indent=2))
    print(stem.with_suffix('.png'))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--grpo',type=Path,required=True);p.add_argument('--classical',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();render(a.grpo,a.classical,a.out)
