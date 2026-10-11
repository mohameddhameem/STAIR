"""Strict official HotpotQA answer EM/F1 and retrospective prefix oracle."""
import argparse,ast,csv,gzip,json,sqlite3,string,re
from pathlib import Path
import numpy as np

def main(root):
    root=Path(root);env={'re':re,'string':string,'Counter':__import__('collections').Counter}
    tree=ast.parse((root/'assets/hotpot_evaluate_v1.py').read_text())
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['normalize_answer','f1_score','exact_match_score']]
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<official-hotpot>','exec'),env)
    data=json.load(gzip.open(root/'assets/AgentTTS-HotpotQA.json.gz','rt'));splits=json.load(open(root/'assets/splits.json'))['splits']
    scores=np.full((len(data),160,2),np.nan); costs=np.full((len(data),160),np.nan)
    for shard in range(4):
        for am in ['W','S']:
            path=root/'runs'/f'answers_{am}_{shard}.sqlite'
            if not path.exists():continue
            db=sqlite3.connect(path);rows={k:json.loads(v) for k,v in db.execute('SELECT key,payload FROM results')};db.close()
            for rm in ['W','S']:
                path=root/'runs'/f'retrieval_{rm}_{shard}.sqlite'
                if not path.exists():continue
                db=sqlite3.connect(path);retr={k:json.loads(v) for k,v in db.execute('SELECT key,payload FROM results')};db.close()
                pair=rm+am;group=['WW','WS','SW','SS'].index(pair)
                for i in range(shard,len(data),4):
                    qa=rows.get(f'{i}:qa:{pair}');rt=retr.get(f'{i}:ret:{rm}')
                    if not qa or not rt:continue
                    emitted=0
                    for k in range(1,41):
                        emitted+=qa['outputs'][k-1]['meta_info']['completion_tokens']
                        fused=rows.get(f'{i}:fuse:{pair}:{k}')
                        if not fused:continue
                        text=fused['outputs'][0]['text'];gold=data[i]['answers']
                        scores[i,group*40+k-1]=[max(env['exact_match_score'](text,g) for g in gold),max(env['f1_score'](text,g)[0] for g in gold)]
                        # Approximate model-weighted token work, not measured FLOPs.
                        rp=1.5 if rm=='W' else 7;ap=1.5 if am=='W' else 7
                        costs[i,group*40+k-1]=rp*(rt['input_tokens']+rt['outputs'][0]['meta_info']['completion_tokens'])+ap*(qa['input_tokens']+emitted+fused['input_tokens']+fused['outputs'][0]['meta_info']['completion_tokens'])
    complete=np.isfinite(scores).all(axis=(1,2));report={'complete_questions':int(complete.sum()),'total_questions':len(data),'metric':'official HotpotQA, raw fusion text; no heuristic answer extraction','cost':'parameter billions times prompt/output token counts; approximate compute proxy; includes retrieval, candidates and fusion'}
    if not complete.all():
        (root/'runs/progress_metrics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    val=splits['val'];test=splits['test'];means=scores[val,:,0].mean(axis=0);cost=costs[val].mean(axis=0)
    best=min(range(160),key=lambda j:(-means[j],cost[j],j))
    fixed=scores[test,best,0];oracle=scores[test,:,0].max(axis=1)
    report.update(fixed_config={'pair':['WW','WS','SW','SS'][best//40],'k':best%40+1,'test_em':float(fixed.mean()),'test_f1':float(scores[test,best,1].mean()),'cost':float(costs[test,best].mean())},joint_oracle={'test_em':float(oracle.mean()),'em_gain':float((oracle-fixed).mean()),'qualification':'gold-assisted retrospective upper bound, selected over 160 stochastic outputs; not a deployable policy'})
    per_pair={}
    for g,pair in enumerate(['WW','WS','SW','SS']):
        b=min(range(g*40,(g+1)*40),key=lambda j:(-means[j],cost[j],j));per_pair[pair]={'fixed_k':b%40+1,'fixed_test_em':float(scores[test,b,0].mean()),'budget_oracle_test_em':float(scores[test,g*40:(g+1)*40,0].max(axis=1).mean())}
    report['per_pair']=per_pair
    rng=np.random.default_rng(20261011)
    gain=oracle-fixed
    boot=np.array([gain[rng.integers(0,len(gain),len(gain))].mean() for _ in range(2000)])
    report['joint_oracle']['paired_bootstrap_gain_ci95']=np.quantile(boot,[.025,.975]).tolist()
    with open(root/'runs/per_question_oracle.csv','w') as f:
        writer=csv.writer(f);writer.writerow(['index','id','split','oracle_pair','oracle_k','oracle_em','oracle_cost','fixed_em','fixed_k'])
        membership={i:s for s,ids in splits.items() for i in ids}
        for i in range(len(data)):
            high=scores[i,:,0].max();eligible=np.flatnonzero(scores[i,:,0]==high);j=min(eligible,key=lambda j:(costs[i,j],j))
            writer.writerow([i,data[i]['_id'],membership[i],['WW','WS','SW','SS'][j//40],j%40+1,high,costs[i,j],scores[i,best,0],best%40+1])
    # Lagrangian oracle frontier; all winners include the globally fixed action.
    frontier=[]
    for lam in np.r_[0,np.logspace(-8,-3,40)]:
        ix=np.argmax(scores[test,:,0]-lam*costs[test],axis=1);rr=np.arange(len(test));frontier.append({'lambda':float(lam),'em':float(scores[test,:,0][rr,ix].mean()),'cost':float(costs[test][rr,ix].mean())})
    report['cost_aware_oracle_frontier']=frontier
    baseline_cost=float(costs[test,best].mean())
    feasible=[r for r in frontier if r['cost']<=baseline_cost]+[{'em':float(fixed.mean()),'cost':baseline_cost,'lambda':None}]
    report['evaluated_oracle_at_fixed_cost']=max(feasible,key=lambda r:r['em'])
    report['evaluated_oracle_at_fixed_cost']['qualification']='Best evaluated gold-assisted policy below baseline average cost; not an exact constrained optimum'
    report['test_fixed_action_grid']=[{'pair':['WW','WS','SW','SS'][j//40],'k':j%40+1,'validation_em':float(means[j]),'test_em':float(scores[test,j,0].mean()),'test_f1':float(scores[test,j,1].mean()),'cost':float(costs[test,j].mean())} for j in range(160)]
    (root/'runs/report.json').write_text(json.dumps(report,indent=2));np.savez_compressed(root/'runs/scores.npz',scores=scores,costs=costs);print(json.dumps(report,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(Path(p.parse_args().root))
