"""Verify all recent published metrics without models, raw records or inference."""
import json,math
from pathlib import Path
import numpy as np
import pandas as pd

def verify_recent(root):
    if not (root/'recent_publication.json').exists():return 0
    manifest=json.loads((root/'recent_publication.json').read_text());indexed=0
    truth=pd.read_csv(root/'hotpotqa_jonathan_aligned_v1/outcomes_test.csv.gz').set_index('id')
    assert len(truth)==1481 and truth.index.is_unique and truth.SS.sum()==1137
    price=np.array([80.,195.,894.,1009.]);direct=np.array([80.,192.,817.,929.]);names=['WW','WS','SW','SS']
    configs=[('hotpotqa_jonathan_aligned_v1',['method','objective'],30),
             ('hotpotqa_model_comparison_v1',['architecture','method'],9),
             ('hotpotqa_budget_points_v2',['architecture','method','objective'],18)]
    for folder,keys,expected in configs:
        data=pd.read_csv(root/folder/'per_question.csv.gz');summary=json.loads((root/folder/'results.json').read_text())
        assert len(summary)==expected and len(data)==expected*1481;indexed+=len(summary)
        for row in data.itertuples():
            a=names.index(row.route) if isinstance(row.route,str) else row.route
            correct=int(truth.loc[row.id,names[a]])
            assert row.lenient==correct
            wanted=direct[a] if folder=='hotpotqa_jonathan_aligned_v1' and row.method.startswith('fixed_') else price[a]
            assert row.cost==wanted
        for item in summary:
            chosen=data
            for key in keys:chosen=chosen[chosen[key]==item[key]]
            assert len(chosen)==1481 and chosen.id.nunique()==1481
            acc=item['accuracy'] if 'accuracy' in item else item['accuracy_pct']/100
            assert math.isclose(chosen.lenient.mean(),acc,abs_tol=1e-12)
            assert math.isclose(chosen.cost.mean(),item['cost'],abs_tol=1e-8)
            assert math.isclose(chosen.cost.mean()/929*100,item['cost_pct_ss'],abs_tol=1e-10)
    scores=pd.read_csv(root/'hotpotqa_jonathan_aligned_v1/author_router_scores_test.csv.gz').fillna({'ret_choice':''})
    reference=json.loads((root/'hotpotqa_jonathan_aligned_v1/author_four_pairs.json').read_text());indexed+=len(reference)
    original=truth.reset_index().set_index('idx')
    for item in reference:
        rf='ret_label' if item['pair'] in ['label','reward_qa'] else 'ret_reward'
        qf='qa_label' if item['pair'] in ['label','reward_ret'] else 'qa_reward'
        ret=scores[scores.role==rf].set_index('idx').pW
        qa=scores[scores.role==qf].set_index(['idx','ret_choice']).pW
        correct=cost=0;tr,tw,ts=item['threshold']
        for idx,t in original.iterrows():
            r=int(ret[idx]<tr);a=int(qa[idx,'S' if r else 'W']<(ts if r else tw));route=2*r+a
            correct+=t[names[route]];cost+=price[route]
        assert math.isclose(correct/1481,item['test_accuracy'],abs_tol=1e-12)
        assert math.isclose(cost/1481/929*100,item['test_cost_pct_ss'],abs_tol=1e-10)
    minimal=root/'medagentbench_replay_inputs';outcomes={}
    for setting in [1,3]:
        d=pd.read_csv(minimal/f'outcomes_costs_t{setting}.csv.gz')
        assert len(d)==300 and d.tid.nunique()==300 and set(d.split)=={'train','val','test'}
        assert set(d.columns)==set(['tid','idx','split','qw','qs','cw','cs']+(['cw2'] if setting==3 else []))
        if setting==1:d['cw2']=d.cw
        outcomes[setting]=d.set_index('tid')
    groups=pd.read_csv(minimal/'split_groups.csv.gz');assert len(groups)==300
    assert groups.groupby('anonymous_group').split.nunique().eq(1).all()
    assert groups.split.value_counts().to_dict()=={'test':150,'train':90,'val':60}
    for part,count in [('train',30),('val',19),('test',49)]:
        chosen=groups[(groups.split==part)&groups.has_patient]
        assert chosen.anonymous_group.nunique()==count
        expected={'train':9,'val':6,'test':15}[part]
        assert groups[groups.split==part].tid.str.split('_').str[0].value_counts().eq(expected).all()
    risk=pd.read_csv(minimal/'risk_scores.csv.gz').set_index('tid').tau
    for folder,family,expected,models in [('medagentbench_grpo_v1','architecture',144,24),('medagentbench_classical_v1','family',336,56)]:
        d=pd.read_csv(root/folder/'per_task.csv.gz');pred=pd.read_csv(root/folder/'test_scores.csv.gz',float_precision='round_trip');summary=json.loads((root/folder/'results.json').read_text())
        assert len(summary)==expected and len(d)==expected*150 and len(pred)==models*150
        assert set(pred.columns)<=set(['setting',family,'seed','tid','idx','pDeliver'])
        indexed+=len(summary);keys=['setting',family,'seed','point','risk_gate']
        meta=pd.DataFrame(summary)[keys+['threshold']]
        joined=d.merge(meta,on=keys,validate='many_to_one').merge(pred[['setting',family,'seed','tid','pDeliver']],on=['setting',family,'seed','tid'],validate='many_to_one')
        for setting in [1,3]:
            sub=joined[joined.setting==setting].copy();base=outcomes[setting].loc[sub.tid]
            action=(sub.pDeliver.to_numpy()<sub.threshold.to_numpy()).astype(int)
            high=risk.loc[sub.tid].to_numpy()>=.8;action[high&sub.risk_gate.to_numpy()]=1
            np.testing.assert_array_equal(sub.action,action)
            q=np.where(action==0,base.qw,base.qs)
            c=np.where(action==0,base.cw,base.cw+base.cs if setting==1 else base.cw2+base.cs)
            reward=q-.1*c/base.cs.to_numpy()-np.maximum(0,c/base.cs.to_numpy()-1)
            np.testing.assert_array_equal(sub.correct,q);np.testing.assert_allclose(sub.cost,c,rtol=0,atol=1e-7);np.testing.assert_allclose(sub.reward,reward,rtol=0,atol=1e-7)
        for row in summary:
            chosen=d
            for key in keys:chosen=chosen[chosen[key]==row[key]]
            assert len(chosen)==150 and chosen.tid.nunique()==150
            assert math.isclose(chosen.correct.mean(),row['accuracy'],abs_tol=1e-12)
            assert math.isclose(chosen.cost.mean(),row['mean_cost'],abs_tol=1e-8)
            base=outcomes[row['setting']];base=base[base.split=='test'];denom=(base.cw+base.cs).sum()
            assert math.isclose(chosen.cost.sum()/denom*100,row['cost_pct_always_escalate'],abs_tol=1e-8)
    refs=json.loads((root/'medagentbench_grpo_v1/llm_reference.json').read_text());indexed+=len(refs)
    assert len(refs)==14
    assert indexed==manifest['new_index_rows']==559
    # All figure numbers must match the same gated, validation-selected rows.
    figure=json.loads((root/'figures/medagentbench/medagentbench_nonllm_vs_jonathan.json').read_text())
    g=json.loads((root/'medagentbench_grpo_v1/results.json').read_text());c=json.loads((root/'medagentbench_classical_v1/results.json').read_text())
    for row in figure['data']:
        for setting in [1,3]:
            if row['key']=='jonathan':record=next(r for r in refs if r['setting']==setting and r['point']=='OP_A_fixed_050' and r['risk_gate'])
            else:
                table,field=(g,'architecture') if row['recipe'].startswith('GRPO') else (c,'family')
                record=next(r for r in table if r['setting']==setting and r[field]==row['key'] and r['seed']==17 and r['point']=='val_reward_selected' and r['risk_gate'])
            assert math.isclose(row[f't{setting}_accuracy'],100*record['accuracy'],abs_tol=1e-12)
            assert math.isclose(row[f't{setting}_cost'],record['cost_pct_always_escalate'],abs_tol=1e-12)
    print(f'Verified {indexed} recent index rows, minimal replay inputs, anonymous patient splits, probabilities/actions, all grades/costs, references and figure values.')
    return indexed

if __name__=='__main__':verify_recent(Path(__file__).resolve().parent)
