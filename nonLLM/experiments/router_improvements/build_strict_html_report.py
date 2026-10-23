"""Render the completed official-EM rerun without mixing historical metrics."""
import argparse
from datetime import date
from html import escape
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[4]

def main(run):
    def read(path):return json.loads((run/path).read_text())
    protocol=read('protocol.json');summary=read('results.json')
    initial=read('initial/results/summary.json');ext=read('extended/results.json')
    arch=read('architecture/results.json');combined=read('combined/results.json')
    records=[]
    def add(name,t,source,note):
        records.append({'name':name,'accuracy':100*t.get('strict_accuracy',t.get('accuracy',0)),
            'cost':t.get('cost_pct_of_direct_SS',t.get('cost_pct_SS',t.get('cost_pct_of_SS',0))),
            'source':str((run/source).relative_to(ROOT)),'note':note})
    original=read('original/results.json')
    for name,t in original['composed'].items():add('Original / '+name,t,Path('original/REPORT.md'),'Three classifiers; unresolved EM labels skipped.')
    original_lines=['# Original-style non-LLM classifiers: corrected EM','',
        'Three independent classifiers. All score-dependent labels use official normalized EM. Unresolved labels are skipped in fitting; all test questions are evaluated.',
        'Costs use the common mean parameter-weighted token ruler for direct routes; CPU router overhead is excluded. No weak-first sunk calls are charged for direct routing.',
        '', '| Classifier | Test exact match % | Cost % direct SS |','|---|---:|---:|']
    original_lines += [f"| {name} | {100*t['strict_accuracy']:.2f} | {t['cost_pct_of_SS']:.2f} |" for name,t in original['composed'].items()]
    (run/'original/REPORT.md').write_text('\n'.join(original_lines)+'\n')
    for folder,name in [('base/router','October-style full cascade'),('pilot','No-weak-answer ablation')]:
        add(name,read(folder+'/results.json')['test'],Path(folder+'/REPORT.md'),'Retrieval coverage labels; answering labels rebuilt with exact match.')
    initial_map={r['name']:r for r in initial['results']}
    for key,name in [('gain_trees','Gain regression'),('cost_sensitive_logreg','Cost-sensitive logistic regression'),
                     ('confidence_outcome_logreg','Confidence + outcome targets'),('sampled_augmentation_logreg','Greedy + 4 training samples'),
                     ('sampled_consensus_logreg','Five-answer consensus'),('exact_reward_lambda0.0','Initial exact reward'),('grpo_lambda0.0','Initial GRPO')]:
        add(name,initial_map[key]['test'],Path('initial/results/REPORT.md'),'Same predefined variant as the earlier suite; retrained with EM.')
    ext_map={r['name']:r for r in ext['results']}
    for objective,label in [('max_accuracy','Extended / max accuracy'),('ss_budget','Extended / SS budget'),('cheap95','Extended / cheapest ≥95% SS')]:
        name=ext['global_selected'][objective];add(label,ext_map[name]['test'],Path('extended/REPORT.md'),'Validation selection: '+name)
    for key,label in [('direct4_ss_budget','Direct pre-routing / SS budget'),('direct4_cheap95','Direct pre-routing / cheap95'),
                      ('fixed_S_qa_max_accuracy','Fixed strong retrieval + QA review'),('post_answer_chooser_max_accuracy','Post-answer chooser')]:
        add(label,arch['results'][key]['test'],Path('architecture/REPORT.md'),'Family/objective selected using validation; train-only grouped CV selects model.')
    for obj,label in [('ss_budget','Combined / SS budget'),('cheap95','Combined / cheap95')]:
        add(label,combined['results'][obj]['test'],Path('combined/REPORT.md'),'Validation selection: '+combined['results'][obj]['lock']['key'])
    ss=100*protocol['SS']['test']['accuracy'];val=100*protocol['SS']['val']['accuracy']
    combo=combined['results']['ss_budget']['test']
    combo_note='Direct-SS fallback selected; no budget-constrained gain.' if combined['results']['ss_budget']['lock']['key']=='direct_SS' else 'Selected on validation; exploratory test result.'
    def table(rs,notes=False):
        body=[]
        for r in rs:
            name=escape(r.get('name',r.get('method','')))
            if 'source' in r:name=f'<a href="../{escape(r["source"],quote=True)}">{name}</a>'
            acc=r['accuracy'] if 'name' in r else 100*r['accuracy']
            cost=r.get('cost',r.get('cost_pct_SS'))
            extra='<td class="note">'+escape(r['note'])+'</td>' if notes else ''
            body.append(f'<tr><td>{name}</td><td class="num">{acc:.2f}%</td><td class="num">{cost:.2f}%</td>{extra}</tr>')
        return '<div class="scroll"><table><thead><tr><th>Method</th><th>Test exact match</th><th>Cost / SS</th>'+('<th>Selection / change</th>' if notes else '')+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    # Inline SVG works offline and includes exact values in native hover tooltips.
    w,h=960,310;left,right,top,bottom=64,25,22,52
    xmin=min(r['cost'] for r in records)-3;xmax=max(105,max(r['cost'] for r in records)+3)
    ymin=min(r['accuracy'] for r in records)-1;ymax=max(ss,max(r['accuracy'] for r in records))+1
    x=lambda v:left+(v-xmin)/(xmax-xmin)*(w-left-right)
    y=lambda v:h-bottom-(v-ymin)/(ymax-ymin)*(h-top-bottom)
    svg=[f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="Exact-match accuracy versus inference cost">']
    for tick in range(int(xmin//10)*10,int(xmax)+1,10):
        if tick<xmin:continue
        svg.append(f'<line x1="{x(tick)}" y1="{top}" x2="{x(tick)}" y2="{h-bottom}" stroke="#e4e9f0"/><text x="{x(tick)}" y="{h-bottom+22}" text-anchor="middle">{tick}%</text>')
    for tick in range(int(ymin),int(ymax)+1,2):svg.append(f'<line x1="{left}" y1="{y(tick)}" x2="{w-right}" y2="{y(tick)}" stroke="#e4e9f0"/><text x="{left-10}" y="{y(tick)+4}" text-anchor="end">{tick}%</text>')
    svg.append(f'<line x1="{x(100)}" y1="{top}" x2="{x(100)}" y2="{h-bottom}" stroke="#8290a3" stroke-dasharray="5 5"/><line x1="{left}" y1="{y(ss)}" x2="{w-right}" y2="{y(ss)}" stroke="#8290a3" stroke-dasharray="5 5"/>')
    for r in records:
        color='#138166' if r['name'].startswith('Combined') else '#245bc4'
        svg.append(f'<circle tabindex="0" cx="{x(r["cost"])}" cy="{y(r["accuracy"])}" r="5" fill="{color}" stroke="white"><title>{escape(r["name"])}: {r["accuracy"]:.2f}% EM / {r["cost"]:.2f}% SS cost</title></circle>')
    svg.append(f'<circle cx="{x(100)}" cy="{y(ss)}" r="6" fill="#182338"><title>Direct SS: {ss:.2f}% EM / 100% cost</title></circle><text x="{w/2}" y="{h-9}" text-anchor="middle">Inference cost (% of direct SS)</text></svg>')
    baseline={'method':'Direct SS baseline','accuracy':ss/100,'cost_pct_SS':100}
    tables=table([baseline])
    references='''<div class="scroll"><table><thead><tr><th>Reference design</th><th>Before October</th><th>October Config A</th></tr></thead><tbody>
<tr><td>LLM router</td><td>One shared Qwen1.5B-Instruct LoRA adapter; stage prompts.</td><td>Separate retrieval and QA adapters; one QA adapter shared across branches.</td></tr>
<tr><td>Retrieval</td><td>Question + candidate pool → direct W/S choice.</td><td>Run weak retrieval → review selected evidence → optional strong retrieval.</td></tr>
<tr><td>Answering</td><td>Evidence → direct W/S answering choice.</td><td>Evidence + weak answer → retain or escalate.</td></tr>
<tr><td>Our counterpart</td><td>Three classical classifiers: retrieval, QA-W, QA-S.</td><td>Two classical classifiers: retrieval and shared QA.</td></tr></tbody></table></div><p class="small muted"><b>Author-reported lenient results, separate from the EM table:</b> before-October mixed-stage LLM router 41.8% accuracy / 55% cost; October Config A LLM cascade 50.10% / 43%. These were not locally rerun and use different scoring / answer grids.</p>'''
    now=date.today().isoformat()
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Corrected Non-LLM Router Results — Exact Match</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#f7f9fc;color:#182338;font:15px/1.6 system-ui,sans-serif}}main{{max-width:1120px;margin:auto;padding:35px 25px}}h1{{font-size:32px;line-height:1.2;margin:8px 0 12px}}h2{{font-size:22px;margin:0 0 12px}}p{{margin:8px 0 14px}}.muted,.note{{color:#58677c}}.eyebrow{{font-size:12px;color:#245bc4;letter-spacing:1.4px;font-weight:700}}section{{background:white;border:1px solid #dce3eb;border-radius:12px;padding:22px;margin:20px 0}}.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{border:1px solid #dce3eb;border-radius:12px;padding:18px;background:white}}.card.green{{background:#f0faf5;border-color:#a3d1bf}}.label{{font-size:12px;font-weight:700;color:#58677c}}.value{{font-size:32px;font-weight:750;margin:8px 0}}.card p{{font-size:13px;margin:0}}.notice{{padding:13px 16px;border-left:3px solid #d4a148;background:#fff8e9;font-size:13px;margin:18px 0}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:13px}}th{{background:#f2f5f9;text-align:left;font-size:11px;color:#58677c;text-transform:uppercase;white-space:nowrap}}th,td{{padding:11px 10px;border-bottom:1px solid #e8edf3;vertical-align:top}}td:first-child{{min-width:190px;font-weight:600}}.num{{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}}a{{color:#245bc4;text-decoration:none}}a:hover{{text-decoration:underline}}summary{{cursor:pointer;font-weight:650}}details{{padding:15px 0;border-top:1px solid #dce3eb}}details .scroll{{margin-top:14px}}svg{{width:100%;display:block;background:#fbfcfe;border-radius:8px}}svg text{{font:12px system-ui;fill:#58677c}}.small{{font-size:13px}}footer{{font-size:12px;color:#58677c}}@media(max-width:700px){{main{{padding:20px 12px}}.cards{{grid-template-columns:1fr}}section{{padding:16px}}h1{{font-size:27px}}td.note{{min-width:220px}}}}@media print{{section,.cards{{break-inside:avoid}}body{{background:white}}}}
</style></head><body><main><div class="eyebrow">CS707 · CORRECTED EXPERIMENTS · {now}</div><h1>Non-LLM routers: official exact match</h1>
<p class="muted">All previously tested routing families retrained with a single metric, the same Instruct answer grid, and one cost ruler. Historical lenient results are kept separately.</p>
<div class="cards"><div class="card"><div class="label">DIRECT SS · TEST EM</div><div class="value">{ss:.2f}%</div><p>{protocol['SS']['test']['correct']} / 1,481 correct · 100% cost</p></div><div class="card green"><div class="label">COMBINED ROUTER · SS-BUDGET CHOICE</div><div class="value">{100*combo['strict_accuracy']:.2f}%</div><p>{combo['cost_pct_SS']:.2f}% of SS cost · selected on validation</p></div><div class="card"><div class="label">DIRECT SS · VALIDATION EM</div><div class="value">{val:.2f}%</div><p>{protocol['SS']['val']['correct']} / 1,110 correct · model selection only</p></div></div>
<div class="notice"><b>52.79% is the supplied reference validation lenient score.</b> It is not the exact-match baseline. On that original grid, validation strict EM is 46.85%. On the regenerated Instruct grid used here, validation strict EM is {val:.2f}% and test strict EM is {ss:.2f}%.</div>
<section><h2>1. Reference methods and our counterparts</h2>{references}<p class="small muted">Reference routers use LoRA on Instruct LLMs; our routers use classical classifiers or linear policies. The early Base/GGUF live runs are excluded. Author-reported lenient values remain in the <a href="router_methods_results_exploratory.html">historical report</a>.</p></section>
<section><h2>2. Retrained results</h2><p class="small muted">Train 4,814 · validation 1,110 · test 1,481. Every reported accuracy below is normalized answer exact match. Cost includes every executed weak / strong call and excludes CPU router overhead. Each listed operating point uses validation, not test, for selection.</p>{tables}{table(records,True)}<h2 style="margin-top:24px">Accuracy versus cost</h2>{''.join(svg)}<p class="small muted">Dashed lines: direct SS. Green: combined router. Hover or focus on a point for values. Higher accuracy and lower cost are preferable.</p></section>
<section><h2>3. What was corrected</h2><ul><li>Reused FP16 Qwen2.5-1.5B-Instruct / 7B-Instruct outputs. No Base generations are included.</li><li>Regraded all 29,620 greedy QA answers with official HotpotQA answer EM; regraded sampled answers before rebuilding training targets and rewards.</li><li>Retrained original classifiers, October-style cascades, confidence / sampling variants, exact reward, initial and extended GRPO, architecture models and the combined router.</li><li>Re-selected checkpoints, model families and thresholds on validation. Extra training samples retain equal total weight per question.</li><li>Used the same six token-based model-call prices. Direct methods skip unused weak calls; cascades and consensus pay for all their calls.</li></ul><p class="small muted">Legacy JSON keys named <code>em_lenient</code> / <code>lenient_accuracy</code> are retained only as compatibility aliases of strict EM in this isolated run. Both columns equal strict EM; no containment rule is used for training or selection.</p></section>
<section><h2>4. Full results — {len(summary['results'])} configurations</h2><p class="small muted">Some objectives select identical models. All detailed outputs and validation locks are retained.</p><details><summary>Show all retrained configurations</summary>{table(summary['results'])}</details></section>
<section><h2>5. Limits and artifacts</h2><p class="small">These are exploratory results on an already examined test split. Small differences do not establish stable superiority. Exact match is verified against official HotpotQA code; exact equivalence to AgentTTS’s missing <code>evaluation_2wiki.py</code> or the reference’s missing <code>mint_labels.py</code> is not claimed.</p><p class="small"><a href="../{run.relative_to(ROOT)}/REPORT.md">Complete report</a> · <a href="../{run.relative_to(ROOT)}/protocol.json">Protocol and generation provenance</a> · <a href="../{run.relative_to(ROOT)}/results.json">Machine-readable results</a> · <a href="router_methods_results_exploratory.html">Historical lenient experiments</a></p></section><footer>Single-file offline HTML. Model weights and prior experiment outputs are unchanged. Built from the completed corrected run.</footer></main></body></html>'''
    html=html.replace('cost · selected on validation</p>','cost<br>'+combo_note+'</p>')
    reportdir=ROOT/'reports';old=reportdir/'router_methods_results.html';archive=reportdir/'router_methods_results_exploratory.html'
    if not archive.exists():shutil.copy2(old,archive)
    target=reportdir/'router_methods_results_strict_em.html';target.write_text(html)
    old.write_text(html)
    print(target)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
