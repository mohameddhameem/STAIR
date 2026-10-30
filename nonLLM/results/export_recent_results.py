"""Publish recent replay results using an explicit artifact/column allowlist.

No raw views, answers, patient identifiers, checkpoints or feature vocabularies.
"""
import gzip,hashlib,json,shutil
from pathlib import Path
import numpy as np
import pandas as pd

RECENT_RUNS={'nonllm_jonathan_aligned_v1','nonllm_jonathan_model_comparison_v1',
             'nonllm_medagentbench_replay_v1','nonllm_medagentbench_classical_v1'}
TEXT={'REPORT.md','protocol.json','verification.json','validation_lock.json','status.json',
      'results.json','results.csv','training_summary.json','seed_repeats.json',
      'paired_patient_comparisons.json','linear_svm_t3_paired_changes.json',
      'author_four_pairs.json','author_readout.txt','patient_split_check.txt',
      'download_manifest.json','llm_reference.json'}
BINARY={'per_question.csv.gz','per_task.csv.gz'}
CURVES={'curve.json','candidates.json','complete.json','calibration_folds.json'}

def export_recent(work,dest,rows,inventory,files,copy_text):
    repo=dest.parents[1];published=[];before=len(rows)
    def record(src,target):
        item={'source':str(src.relative_to(work)),'published':str(target.relative_to(repo)),
              'source_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),
              'published_sha256':hashlib.sha256(target.read_bytes()).hexdigest()}
        files.append(item);return item
    def binary(src,target):
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,target);record(src,target)
    def dataframe(frame,target,source):
        target.parent.mkdir(parents=True,exist_ok=True)
        with target.open('wb') as stream:
            with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as zipped:zipped.write(frame.to_csv(index=False).encode())
        record(source,target)
    def run(source,name):
        target=dest/name;copied=[]
        for src in sorted(source.iterdir()):
            if not src.is_file():continue
            if src.name in TEXT:
                item=copy_text(src,target/src.name);files.append(item);copied.append(item['published'])
            elif src.name in BINARY:
                binary(src,target/src.name);copied.append(str((target/src.name).relative_to(repo)))
            elif src.name=='test_scores.csv':
                # Only task ID, setting, method, seed and decision probabilities.
                frame=pd.read_csv(src);allowed={'setting','architecture','family','seed','tid','idx','pDeliver'}
                assert set(frame.columns)<=allowed
                zipped=target/'test_scores.csv.gz';zipped.parent.mkdir(parents=True,exist_ok=True)
                with zipped.open('wb') as stream:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=stream,mtime=0) as compressed:compressed.write(src.read_bytes())
                record(src,zipped);copied.append(str(zipped.relative_to(repo)))
        for src in sorted((source/'training').rglob('*.json')):
            if src.name in CURVES:
                target_file=target/src.relative_to(source);item=copy_text(src,target_file);files.append(item);copied.append(item['published'])
        inventory.append({'run':str(source.relative_to(work/'runs')),'source_phase':'complete',
            'published_artifacts':copied,'classification':'separate replay cohort; no raw views/answers/patient identifiers/weights'})
        published.append({'source_run':str(source.relative_to(work/'runs')),'published':name,'artifacts':len(copied)})
        return target
    def index(cohort,method,seed,n,accuracy,cost,source,metric,convention,units='',status='descriptive replay; selections on validation'):
        rows.append({'cohort':cohort,'method':method,'seed':seed,'n':n,'accuracy_pct':accuracy,'cost_pct_ss':cost,
             'units':units,'status':status,'source':source,'dataset':'medagentbench' if metric=='benchmark_pass_fail' else 'hotpotqa',
             'accuracy_metric':metric,'cost_convention':convention})
    hp=work/'runs/nonllm_jonathan_aligned_v1';hpt=run(hp,'hotpotqa_jonathan_aligned_v1')
    for r in json.loads((hp/'results.json').read_text()):
        index('HotpotQA-Jonathan-cache-slr1481',r['method']+':'+r['objective'],'ensemble17/18/19',r['n'],100*r['accuracy'],r['cost_pct_ss'],
              'hotpotqa_jonathan_aligned_v1/results.json','shared_standard_lenient','Table5 budget / direct SS929; weak-first SS1009')
    for r in json.loads((hp/'author_four_pairs.json').read_text()):
        index('HotpotQA-Jonathan-LLM-reference-slr1481',r['pair']+':'+r['point'],17,1481,100*r['test_accuracy'],r['test_cost_pct_ss'],
              'hotpotqa_jonathan_aligned_v1/author_four_pairs.json','shared_standard_lenient','Table5 budget / direct SS929; router excluded',status='reference scores and val thresholds replayed; not a new LLM run')
    protocol=json.loads((hp/'protocol.json').read_text());truth=np.load(hp/'data/grid_test.npz')['correct'].reshape(-1,4)
    original_grid=[json.loads(line) for line in (hp/'reference/router_data_standard.jsonl').read_text().splitlines()]
    indices=[r['idx'] for r in original_grid if r['split']=='test']
    frame=pd.DataFrame({'id':protocol['ids']['test'],'idx':indices,**{k:truth[:,i].astype(int) for i,k in enumerate(['WW','WS','SW','SS'])}})
    dataframe(frame,hpt/'outcomes_test.csv.gz',hp/'data/grid_test.npz')
    author_scores=[]
    for role in ['ret_label','ret_reward','qa_label','qa_reward']:
        score_file=hp/'reference/test_scores'/f'{role}.jsonl'
        for item in map(json.loads,score_file.read_text().splitlines()):
            author_scores.append({'role':role,'idx':item['idx'],'ret_choice':item.get('ret_choice',''),'pW':item['pW']})
    dataframe(pd.DataFrame(author_scores),hpt/'author_router_scores_test.csv.gz',hp/'author_four_pairs.json')
    comp=work/'runs/nonllm_jonathan_model_comparison_v1';run(comp,'hotpotqa_model_comparison_v1')
    for r in json.loads((comp/'results.json').read_text()):
        index('HotpotQA-Jonathan-cache-models-slr1481',r['architecture']+':'+r['method'],'ensemble17/18/19',r['n'],r['accuracy_pct'],r['cost_pct_ss'],
              'hotpotqa_model_comparison_v1/results.json','shared_standard_lenient','Table5 budget / direct SS929')
    budget=comp/'operating_points_v2';run(budget,'hotpotqa_budget_points_v2')
    for r in json.loads((budget/'results.json').read_text()):
        index('HotpotQA-Jonathan-cache-budget-slr1481',r['architecture']+':'+r['method']+':'+r['objective'],'ensemble17/18/19',r['test_n'],r['accuracy_pct'],r['cost_pct_ss'],
              'hotpotqa_budget_points_v2/results.json','shared_standard_lenient','Table5 budget / direct SS929; budget selected on val')
    for source_name,target,family in [('nonllm_medagentbench_replay_v1','medagentbench_grpo_v1','architecture'),
                                      ('nonllm_medagentbench_classical_v1','medagentbench_classical_v1','family')]:
        source=work/'runs'/source_name;run(source,target)
        for r in json.loads((source/'results.json').read_text()):
            method=f"T{r['setting']}:{r[family]}:{r['point']}:gate{int(r['risk_gate'])}"
            index('MedAgentBench-patient-disjoint-'+('GRPO' if family=='architecture' else 'classical'),method,r['seed'],r['n'],100*r['accuracy'],r['cost_pct_always_escalate'],
                  target+'/results.json','benchmark_pass_fail','measured FLOP units / full weak-first sum(cw+cs)',r['mean_cost'])
    med=work/'runs/nonllm_medagentbench_replay_v1'
    for r in json.loads((med/'llm_reference.json').read_text()):
        index('MedAgentBench-Jonathan-LLM-reference',f"T{r['setting']}:{r['point']}:gate{int(r['risk_gate'])}",17,r['n'],100*r['accuracy'],r['cost_pct_always_escalate'],
              'medagentbench_grpo_v1/llm_reference.json','benchmark_pass_fail','measured FLOP units / full weak-first sum(cw+cs)',r['mean_cost'],status='reference cached probabilities; includes fixed/oracle controls')
    # Minimal replay lookup permits public cost/grade checks without patient views.
    shared=dest/'medagentbench_replay_inputs';shared.mkdir(parents=True,exist_ok=True)
    for setting in [1,3]:
        src=med/f'reference/results/medagentbench/training{setting}_table.json';data=json.loads(src.read_text())
        allowed=['tid','idx','split','qw','qs','cw','cs']+(['cw2'] if setting==3 else [])
        dataframe(pd.DataFrame([{k:t[k] for k in allowed} for t in data]),shared/f'outcomes_costs_t{setting}.csv.gz',src)
    tau=med/'reference/results/medagentbench/tau_scores_v2.json'
    dataframe(pd.DataFrame([{'tid':t['tid'],'tau':t['tau']} for t in json.loads(tau.read_text())['scores']]),shared/'risk_scores.csv.gz',tau)
    # Anonymous group labels expose split membership only, never actual MRNs.
    tasks=json.loads((med/'reference/test_data_v2.json').read_text());groups={};audit=[]
    split_by_id={t['tid']:t['split'] for t in json.loads((med/'reference/results/medagentbench/training1_table.json').read_text())}
    for task in tasks:
        patient=task.get('eval_MRN');key=patient or 'patientless:'+task['id']
        if key not in groups:groups[key]='group_'+str(len(groups)).zfill(3)
        audit.append({'tid':task['id'],'split':split_by_id[task['id']],'anonymous_group':groups[key],'has_patient':patient is not None})
    dataframe(pd.DataFrame(audit),shared/'split_groups.csv.gz',med/'reference/test_data_v2.json')
    (shared/'README.md').write_text('# Minimal replay audit inputs\n\nNumeric outcomes/costs, frozen risk scores and anonymous split-group labels only. No task views, answers, actual patient identifiers, demographics or medical records. Original table hashes and official dataset revision are in ../medagentbench_grpo_v1/download_manifest.json. Group labels are arbitrary local integers, including separate patient-less tasks. These files allow grade, cost and patient-disjoint split checks without publishing clinical content.\n')
    # Presentation exports: exact deterministic figures plus editable PDF/SVG.
    assets=[('nonllm_jonathan_aligned_slides_v1','hotpotqa_v1'),('nonllm_jonathan_aligned_slides_v2','hotpotqa_v2'),
            ('medagentbench_nonllm_summary_v1','medagentbench')]
    for folder,name in assets:
        for src in sorted((work/'reports'/folder).iterdir()):
            if src.name=='preview.png' or src.suffix not in ['.png','.pdf','.svg','.json']:continue
            target=dest/'figures'/name/src.name
            if src.suffix=='.json':files.append(copy_text(src,target))
            elif src.suffix=='.svg':
                target.parent.mkdir(parents=True,exist_ok=True)
                target.write_text('\n'.join(line.rstrip() for line in src.read_text().splitlines())+'\n')
                record(src,target)
            else:binary(src,target)
    for row in rows:
        row.setdefault('dataset','hotpotqa');row.setdefault('accuracy_metric','cohort-specific lenient; see report')
        row.setdefault('cost_convention','cohort-specific; see report')
    metadata={'exported_on':'2026-10-31','new_index_rows':len(rows)-before,'total_index_rows':len(rows),'runs':published,
              'privacy':'Explicit allowlist; no raw views/answers/actual patient identifiers/checkpoints/feature vocabularies',
              'minimal_replay_inputs':'medagentbench_replay_inputs','figures':'figures'}
    (dest/'recent_publication.json').write_text(json.dumps(metadata,indent=2)+'\n')
    return len(rows)-before
