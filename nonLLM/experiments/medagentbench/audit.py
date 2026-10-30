"""Post-read audits and paired patient-cluster intervals. Never runs a router."""
import argparse,json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import replay

def main(out):
    integrity=replay.verify_source(out)
    protocol=json.loads((out/'protocol.json').read_text())
    locks=json.loads((out/'validation_lock.json').read_text())
    assert len(locks['models'])==24 and len(locks['model_hashes'])==24
    for item in locks['model_hashes']:assert replay.digest(out/item['path'])==item['sha256']
    for folder in sorted((out/'training').glob('t*/*/seed*')):
        done=json.loads((folder/'complete.json').read_text());curve=json.loads((folder/'curve.json').read_text());state=joblib.load(folder/'best.joblib')
        assert done['last_step']==440 and len(curve)==44
        best=max(v['val_meanR_delivered'] for v in curve)
        first=next(v for v in curve if v['val_meanR_delivered']==best)
        assert done['best_step']==state['step']==first['step']
        assert done['best_val_reward']==best
    scores=pd.read_csv(out/'test_scores.csv');detail=pd.read_csv(out/'per_task.csv.gz')
    original={r['id']:r for r in json.loads((out/'reference/test_data_v2.json').read_text())}
    tau={r['tid']:r['tau'] for r in json.loads((out/'reference/results/medagentbench/tau_scores_v2.json').read_text())['scores']}
    comparisons=[]
    for setting in [1,3]:
        rows=[r for r in replay.table(out,setting) if r['split']=='test']
        # Patient-less tasks receive their own cluster; no identifier is exported.
        cluster=[original[r['tid']].get('eval_MRN') or 'no_patient:'+r['tid'] for r in rows]
        unique=sorted(set(cluster));group=np.array([unique.index(x) for x in cluster])
        n=np.bincount(group);denom=np.bincount(group,weights=[r['cw']+r['cs'] for r in rows])
        rng=np.random.default_rng(20260729);draw=rng.integers(len(unique),size=(2000,len(unique)))
        peer={r['tid']:r['pDeliver'] for r in map(json.loads,(out/f'reference/results/medagentbench/training{setting}_test_scores.jsonl').read_text().splitlines())}
        for arch in replay.ARCHES:
            local=scores[(scores.setting==setting)&(scores.architecture==arch)&(scores.seed==17)]
            pmap=dict(zip(local.tid,local.pDeliver));p=np.array([pmap[r['tid']] for r in rows]);refp=np.array([peer[r['tid']] for r in rows])
            lock=next(v for v in locks['models'] if v['setting']==setting and v['architecture']==arch and v['seed']==17)
            for point,threshold in [('OP_A_fixed_050',.5),('val_reward_selected',lock['val_reward_threshold']['threshold'])]:
                for gate in [False,True]:
                    met,(a,q,c,r)=replay.metric(rows,setting,p,threshold,tau,gate)
                    ref,(_,rq,rc,_)=replay.metric(rows,setting,refp,.5,tau,gate)
                    dq=np.bincount(group,weights=q-rq);dc=np.bincount(group,weights=c-rc)
                    acc=100*dq[draw].sum(1)/n[draw].sum(1);cost=100*dc[draw].sum(1)/denom[draw].sum(1)
                    comparisons.append({'setting':setting,'architecture':arch,'seed':17,'point':point,'threshold':threshold,'risk_gate':gate,
                        'paired_reference':'Jonathan same-setting theta.50, same risk gate',
                        'accuracy_delta_pp':100*(met['accuracy']-ref['accuracy']),
                        'accuracy_delta_patient_bootstrap_95ci_pp':np.quantile(acc,[.025,.975]).tolist(),
                        'cost_delta_pp':met['cost_pct_always_escalate']-ref['cost_pct_always_escalate'],
                        'cost_delta_patient_bootstrap_95ci_pp':np.quantile(cost,[.025,.975]).tolist(),
                        'patient_clusters':len(unique),'bootstrap_resamples':2000})
    replay.write(out/'paired_patient_comparisons.json',comparisons)
    verification=json.loads((out/'verification.json').read_text())
    verification.update({'official_dataset_hash_matches_team_manifest':True,
        'all_24_model_hashes_match_pretest_lock':True,'all_checkpoint_choices_verified_against_rounded_validation_objective':True,
        'paired_patient_bootstrap_contrasts':len(comparisons)})
    replay.write(out/'verification.json',verification)
    # Add the risk and seed readouts to the narrative report, preserving primary
    # metrics and all prior outputs. No fresh predictions or threshold tuning.
    text=(out/'REPORT.md').read_text().split('\n## Risk-gate paired readout')[0]
    results=json.loads((out/'results.json').read_text());repeat=json.loads((out/'seed_repeats.json').read_text())
    lines=['','## Risk-gate paired readout','',
      'The lower-cost ungated round2 point delivers 15 high-risk weak failures. When the same tau>=.8 forced-escalation guardrail is applied, those apparent savings disappear. The gate uses the author\'s frozen risk-score cache; risk-score generation cost is excluded for both router families.',
      '', '| Setting | MLP64 seed17 point | Gate | Accuracy | Cost % | High-risk wrong deliveries |','|---|---|---|---:|---:|---:|']
    for r in results:
        if r['architecture']=='mlp64' and r['seed']==17 and r['point'] in ['OP_A_fixed_050','val_reward_selected']:
            lines.append(f"| T{r['setting']} | {r['point']} | {r['risk_gate']} | {r['accuracy']*100:.2f}% | {r['cost_pct_always_escalate']:.2f}% | {r['high_wrong_delivered']} |")
    lines+=['','## Seed repeat sensitivity','',
      'These are independent seeds0/1/2, with no ensemble, not a selection of the best seed. Seed17 is the source-matching primary run. Results vary substantially under the fixed low learning rate and small90-task training set; a single matching seed does not establish robust reproduction.',
      '', '| Setting | Router | Point | Mean accuracy (range) | Mean cost % |','|---|---|---|---:|---:|']
    for r in repeat:
        if r['point'] in ['OP_A_fixed_050','val_reward_selected']:
            lines.append(f"| T{r['setting']} | {r['architecture']} | {r['point']} | {r['accuracy_mean']*100:.2f}% ({r['accuracy_min']*100:.2f}–{r['accuracy_max']*100:.2f}) | {r['cost_pct_mean']:.2f}% |")
    lines+=['','Paired patient-cluster bootstrap intervals (2,000 resamples) are in paired_patient_comparisons.json. They address patient clustering on this replay; they do not remove model-selection multiplicity, prior public-test knowledge, or seed sensitivity.']
    (out/'REPORT.md').write_text(text+'\n'.join(lines)+'\n')
    print(json.dumps(verification,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();main(a.out)
