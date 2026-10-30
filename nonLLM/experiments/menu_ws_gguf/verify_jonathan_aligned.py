"""Audit published aggregates against frozen reference scores and call accounting."""
import argparse, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import jonathan_aligned as experiment

def main(out):
    p=json.loads((out/'protocol.json').read_text())
    for name,digest in p['reference_sha256'].items():
        assert hashlib.sha256((out/'reference'/Path(name).name).read_bytes()).hexdigest()==digest
    sets=[set(p['ids'][s]) for s in ['train','val','test']]
    assert not (sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
    em=np.load(out/'data/grid_test.npz')['correct'].reshape(-1,4)
    ids={qid:i for i,qid in enumerate(p['ids']['test'])};routes=['WW','WS','SW','SS']
    detail=pd.read_csv(out/'per_question.csv.gz');summary=json.loads((out/'results.json').read_text())
    for row in detail.itertuples():
        a=routes.index(row.route)
        assert row.lenient==em[ids[row.id],a]
        expected=([80.,192.,817.,929.][a] if row.method.startswith('fixed_') else experiment.PRICE[a])
        assert row.cost==expected
    for row in summary:
        subset=detail[(detail.method==row['method'])&(detail.objective==row['objective'])]
        assert len(subset)==len(ids) and subset.id.nunique()==len(ids)
        assert row['correct']==subset.lenient.sum()
        assert abs(row['accuracy']-subset.lenient.mean())<1e-10
        assert abs(row['cost']-subset.cost.mean())<1e-10
    training=[]
    for done in sorted((out/'training').glob('seed*/*/complete.json')):
        info=json.loads(done.read_text());curve=json.loads(done.with_name('curve.json').read_text())
        weights=joblib.load(done.with_name('best.joblib'))
        assert info['best_epoch']==weights['meta']['epoch']
        assert info['last_epoch']==curve[-1]['epoch']
        kind=weights['meta']['kind'];stage=weights['meta']['stage']
        criterion=('val_auc' if stage!='query' else 'val_ce') if kind=='sft' else 'val_expected_reward'
        best=(min if criterion=='val_ce' else max)(r[criterion] for r in curve)
        actual=curve[info['best_epoch']-1][criterion]
        assert abs(actual-best)<1e-7
        training.append({'run':str(done.parent.relative_to(out/'training')),**info})
    assert len(training)==42
    # Independently check the author's four-pair readout, including the 76.23%
    # candidate for the slide, from committed scores rather than copied numbers.
    ref=out/'reference'
    original=[json.loads(l) for l in (ref/'router_data_standard.jsonl').read_text().splitlines()]
    pairs={'label':('sft_router5_ret','sft_router5_qa'),
           'reward_qa':('sft_router5_ret','reward_train_qa'),
           'reward_ret':('reward_train_ret','sft_router5_qa'),
           'reward_both':('reward_train_ret','reward_train_qa')}
    for item in json.loads((out/'author_four_pairs.json').read_text()):
        rs,qs=pairs[item['pair']]
        for part in ['val','test']:
            rows=[r for r in original if r['split']==part]
            rf=ref/rs/'val_scores.jsonl' if part=='val' else ref/'test_scores'/('ret_label.jsonl' if rs.startswith('sft') else 'ret_reward.jsonl')
            qf=ref/qs/'val_scores.jsonl' if part=='val' else ref/'test_scores'/('qa_label.jsonl' if qs.startswith('sft') else 'qa_reward.jsonl')
            rp={r['idx']:r['pW'] for r in map(json.loads,rf.read_text().splitlines())}
            qp={(r['idx'],r['ret_choice']):r['pW'] for r in map(json.loads,qf.read_text().splitlines())}
            tr,tw,ts=item['threshold'];correct=cost=0
            for r in rows:
                i=r['idx'];branch='W' if rp[i]>=tr else 'S'
                a=0 if qp[i,branch]>=(tw if branch=='W' else ts) else 1
                route=2*(branch=='S')+a
                correct+=r['em_standard'][route];cost+=experiment.PRICE[route]
            assert abs(correct/len(rows)-item[part+'_accuracy'])<1e-10
            assert abs(cost/len(rows)/929*100-item[part+'_cost_pct_ss'])<1e-9
    verification=json.loads((out/'verification.json').read_text())
    verification.update({'completed_training_fits':len(training),'audited_policy_rows':len(detail),
                         'aggregate_parity':True,'per_question_reference_correctness_parity':True,
                         'all_executed_weak_calls_charged':True,'checkpoint_validation_criterion_verified':True,
                         'author_four_pair_readout_verified':True,
                         'all_reference_files_sha256':{str(f.relative_to(ref)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(ref.rglob('*')) if f.is_file()}})
    experiment.write(out/'verification.json',verification)
    experiment.write(out/'training_summary.json',training)
    print(json.dumps(verification,indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();main(a.out)
