"""English summary of the completed review-validated lenient rerun."""
import argparse
from datetime import date
from html import escape
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]

def main(run):
    def read(p):return json.loads((run/p).read_text())
    summary=read('results.json');protocol=read('protocol.json')
    assert len(summary['results'])==73
    initial=read('initial/results/summary.json');extended=read('extended/results.json')
    architecture=read('architecture/results.json');combined=read('combined/results.json')
    ss=100*protocol['SS']['test']['accuracy'];val=100*protocol['SS']['val']['accuracy']
    data=[]
    def add(name,t,group,note,source):
        data.append({'name':name,'accuracy':100*t.get('lenient_accuracy',t.get('accuracy',0)),
            'cost':t.get('cost_pct_of_direct_SS',t.get('cost_pct_SS',0)),
            'strict':100*t['strict_accuracy'],'group':group,'note':note,
            'source':str((run/source).relative_to(ROOT))})
    data.append({'name':'Direct SS baseline','accuracy':ss,'cost':100.,'strict':100*protocol['SS']['test']['strict_accuracy'],
        'group':'baseline','note':'Strong retrieval + strong answering; no weak calls.',
        'source':str((run/'REPORT.md').relative_to(ROOT))})
    add('October-style non-LLM cascade',read('base/router/results.json')['test'],'classical',
        'Weak-first review; classifier family and thresholds selected on validation.',Path('base/router/REPORT.md'))
    im={r['name']:r for r in initial['results']}
    for key,name,group,note in [
        ('gain_trees','Gain regression','classical','ExtraTrees predicts strong-minus-weak correctness gain.'),
        ('cost_sensitive_logreg','Cost-sensitive logistic regression','classical','Weighted supervision balances correctness gain and model-call cost.'),
        ('confidence_outcome_logreg','Confidence + outcome targets','classical','Weak-answer confidence; retrieval outcome targets.'),
        ('sampled_augmentation_logreg','Multi-sample training','sampling','Greedy + 4 random training answers; one weak answer at deployment.'),
        ('sampled_consensus_logreg','Five-answer consensus','sampling','Greedy + 4 random weak answers at deployment; all calls charged.'),
        ('exact_reward_lambda0.0','Initial exact expected reward','rl','Exact expectation over four lenient-reward actions; not exact-match grading.'),
        ('grpo_lambda0.0','Initial GRPO','rl','Linear non-LLM policy; group size 8; three-seed mean probabilities.')]:
        add(name,im[key]['test'],group,note,Path('initial/results/REPORT.md'))
    em={r['name']:r for r in extended['results']}
    for obj,name in [('max_accuracy','Extended policies / maximum accuracy'),('ss_budget','Extended policies / SS budget'),('cheap95','Extended policies / cheapest ≥95% SS')]:
        selected=extended['global_selected'][obj]
        add(name,em[selected]['test'],'rl','Global validation choice: '+selected,Path('extended/REPORT.md'))
    for key,name,note in [
        ('direct4_cheap95','Pre-retrieval direct routing','Chooses WW/WS/SW/SS from question + candidate pool; cheap95 objective.'),
        ('fixed_S_qa_max_accuracy','Fixed strong retrieval + QA review','Maximum-accuracy validation choice; includes direct-SS fallback.'),
        ('post_answer_chooser_max_accuracy','Post-answer chooser','Strong retrieval and both answers paid; ExtraTrees retains one answer.')]:
        add(name,architecture['results'][key]['test'],'architecture',note,Path('architecture/REPORT.md'))
    add('Combined / SS-budget choice',combined['results']['ss_budget']['test'],'architecture',
        'Validation operating point: '+combined['results']['ss_budget']['lock']['key'],Path('combined/REPORT.md'))
    combo=combined['results']['ss_budget']['test']
    combo_note='Validation selected the direct-SS fallback.' if combined['results']['ss_budget']['lock']['key']=='direct_SS' else 'Validation selected the combined routing policy.'
    post=architecture['results']['post_answer_chooser_max_accuracy']['test']
    def table(records,kind='summary'):
        body=[]
        for r in records:
            name=r.get('name',r.get('method',''));acc=r['accuracy'] if 'name' in r else 100*r['accuracy'];cost=r.get('cost',r.get('cost_pct_SS'))
            label=escape(name)
            if r.get('source'):label=f'<a href="../{escape(r["source"],quote=True)}">{label}</a>'
            tail='<td class="note">'+escape(r.get('note',''))+'</td>' if kind=='summary' else '<td class="num">'+f"{r['strict'] if 'strict' in r else 100*r['strict_accuracy']:.2f}%"+'</td>'
            body.append(f'<tr><td>{label}</td><td class="num">{acc:.2f}%</td><td class="num">{cost:.2f}%</td>{tail}</tr>')
        return '<div class="scroll"><table><thead><tr><th>Method / operating point</th><th>Lenient accuracy</th><th>Cost / SS</th><th>'+('What changed' if kind=='summary' else 'Strict EM diagnostic')+'</th></tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'
    # Original-reference cohort remains authoritative and uses its own flat cost ruler.
    old=json.loads((ROOT/'runs/baselines_real_v1/results.json').read_text());old_rows=[]
    for name,t in old['composed'].items():old_rows.append({'name':name,'accuracy':100*t['lenient_accuracy'],'strict':100*t['strict_accuracy'],'cost':t['cost_pct_of_SS']})
    pilot=json.loads((ROOT/'runs/nonllm_cascade_configA_ablation_v1/results.json').read_text())['test']
    old_rows.append({'name':'October input ablation (no weak answer)','accuracy':100*pilot['lenient_accuracy'],'strict':100*pilot['strict_accuracy'],'cost':pilot['cost_pct_of_direct_SS']})
    pre=json.loads((ROOT/'runs/nonllm_reference_prerouter_v1/results.json').read_text())
    for obj,r in pre['results'].items():
        t=r['test'];old_rows.append({'name':'Pre-router / '+obj,'accuracy':100*t['accuracy'],'strict':100*t['strict_accuracy'],'cost':t['cost_pct_SS']})
    groups=[('Original classifiers',lambda r:r['method'].startswith('Original / ')),
        ('October full cascade and input ablation',lambda r:r['method'].startswith('October ')),
        ('Initial improvements — 27 comparisons',lambda r:r['method'].startswith('Initial / ')),
        ('Extended reward / GRPO — 24 operating points',lambda r:r['method'].startswith('Extended / ')),
        ('Architecture — 12 operating points',lambda r:r['method'].startswith('Architecture / ')),
        ('Combined — 3 objectives',lambda r:r['method'].startswith('Combined / '))]
    details=''.join('<details><summary>'+escape(label)+'</summary>'+table([r for r in summary['results'] if predicate(r)],'inventory')+'</details>' for label,predicate in groups)
    original_report=run/'original/REPORT.md'
    original_report.write_text('# Original-style classifiers — regenerated cohort, lenient rerun\n\nPrimary metric: review-validated reference-lenient reconstruction. All QA/route labels rebuilt. Cost: common token-price ruler; CPU overhead excluded.\n\n| Classifier | Test lenient % | Strict diagnostic % | Cost % SS |\n|---|---:|---:|---:|\n'+''.join(f"| {k} | {100*t['lenient_accuracy']:.2f} | {100*t['strict_accuracy']:.2f} | {t['cost_pct_of_SS']:.2f} |\n" for k,t in read('original/results.json')['composed'].items()))
    css=(ROOT/'reports/router_methods_results_strict_em.html').read_text().split('<style>',1)[1].split('</style>',1)[0]
    css+='''.filters{display:flex;gap:7px;flex-wrap:wrap;margin:15px 0}button{border:1px solid #dce3eb;border-radius:18px;background:white;padding:6px 12px;color:#58677c;font:inherit;font-size:12px;cursor:pointer}button[aria-pressed="true"]{background:#182338;color:white}.done{color:#16634b;font-weight:700}.legend{font-size:12px;color:#58677c}.legend span{margin-right:15px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.highlight td{background:#f0faf5}summary:focus-visible,button:focus-visible,a:focus-visible{outline:3px solid #d4a148;outline-offset:3px}'''
    html=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Non-LLM Routers — Completed Lenient Evaluation</title><style>{css}</style></head><body><main>
<div class="eyebrow">CS707 · LENIENT EVALUATION · {date.today().isoformat()}</div><h1>Non-LLM routers: completed lenient experiments</h1><p class="muted">All tested non-LLM families retrained and evaluated with one frozen, review-validated lenient rule. Exact match is a secondary diagnostic. Original-reference results and regenerated-answer results are separate cohorts.</p>
<div class="cards"><div class="card"><div class="label">ORIGINAL REFERENCE SS · VALIDATION</div><div class="value">52.79%</div><p>586 / 1,110 · authoritative supplied labels<br>Original test SS: 53.54%</p></div><div class="card"><div class="label">REGENERATED SS · LENIENT TEST</div><div class="value">{ss:.2f}%</div><p>{protocol['SS']['test']['correct']} / 1,481 · 100% cost<br>Validation SS: {val:.2f}%</p></div><div class="card green"><div class="label">COMPLETED COMPARISONS</div><div class="value">73</div><p>Including 144 extended training streams<br>Plus original-grid reference replays</p></div></div>
<div class="notice"><b>Scorer provenance:</b> the original scoring source was not recovered. The reconstructed lenient rule agrees with all <b>4,267</b> available original untruncated answer-review judgments. This supports the reconstruction, but does not certify every unobserved original case. No regenerated answer is paired with an original answer’s outcome label.</div>
<section><h2>1. Reference methods and non-LLM counterparts</h2><div class="scroll"><table><thead><tr><th>Design</th><th>Before October</th><th>October Config A</th></tr></thead><tbody><tr><td>LLM router</td><td>One shared Qwen2.5-1.5B-Instruct LoRA adapter; stage-specific prompts.</td><td>Separate retrieval and answering adapters; one QA adapter shared across branches.</td></tr><tr><td>Retrieval decision</td><td>Question + candidate pool → direct W/S choice.</td><td>Weak retrieval → review weak evidence → keep or escalate.</td></tr><tr><td>Answer decision</td><td>Question + evidence → direct W/S choice.</td><td>Weak answer → review evidence + answer → keep or escalate.</td></tr><tr><td>Our non-LLM counterpart</td><td>Three classifiers: retrieval, QA-W and QA-S.</td><td>Two classifiers: retrieval and shared QA.</td></tr><tr><td>Author-reported lenient result</td><td>41.8% accuracy / 55% cost, mixed-stage training.</td><td>50.10% / 43% cost.</td></tr></tbody></table></div><p class="small muted">Author-reported LLM-router results were not locally rerun and are not ranked against the regenerated cohort. W/S denote weak/strong retrieval or QA; models are Qwen2.5-1.5B/7B Instruct.</p></section>
<section><h2>2. Completed results: regenerated Instruct cohort</h2><p class="small muted"><b class="done">All training, reward rebuilding, validation selection and test replay finished.</b> Train / validation / test = 4,814 / 1,110 / 1,481. Every row here uses the same lenient scorer and token-price cost ruler. All checkpoint/threshold selection uses validation. Previously examined test is exploratory.</p><div class="filters" aria-label="Filter results"><button data-filter="all" aria-pressed="true">All</button><button data-filter="classical" aria-pressed="false">Classical</button><button data-filter="sampling" aria-pressed="false">Sampling</button><button data-filter="rl" aria-pressed="false">Reward / GRPO</button><button data-filter="architecture" aria-pressed="false">Architecture</button></div><div id="main-results">{table(data)}</div><p class="small muted">Selected predefined variants are shown. Extended-policy rows are global validation selections for each objective; architecture rows are selected within their workflow. The combined SS-budget choice uses <code>{escape(combined['results']['ss_budget']['lock']['key'])}</code>. {combo_note}</p><h2 style="margin-top:22px">Accuracy versus cost</h2><div id="chart"></div><div class="legend"><span>Gray: direct SS</span><span>Blue: classical</span><span>Purple: sampling</span><span>Orange: reward / GRPO</span><span>Green: architecture</span></div><p class="small muted">Higher accuracy and lower cost are preferable. Hover or focus on points for exact values. Costs include every generated weak/strong answer and exclude CPU router overhead; these are token proxies, not wall-clock measurements.</p></section>
<section><h2>3. All 73 completed method / operating-point records</h2><p class="small muted">Multiple objectives may choose the same model. Strict EM is shown only as a diagnostic; lenient outcomes drive all training targets, policy rewards, checkpoints and thresholds. Feature mode <code>base</code> means no-confidence features, not a Base language model: all generations use Instruct.</p>{details}</section>
<section><h2>4. Original-reference cohort — actual supplied labels</h2><p class="small muted">This cohort directly replays the original <code>em_lenient</code> judgments: validation SS <b>52.79%</b>, test SS <b>53.54%</b>. No reconstructed grader is needed for these recorded outcomes. Its cost ruler is the original flat proxy, not the regenerated cohort’s measured token-price ruler.</p>{table(old_rows,'inventory')}<p class="small muted">The October input ablation lacks original weak answers and is explicitly not the complete October method. Complete weak-answer/sampling experiments above use regenerated answers and their own freshly matching labels.</p></section>
<section><h2>5. Models, training and grading</h2><p><b>Router models:</b> majority baseline; logistic regression; linear SVM; multinomial Naive Bayes; random forest; ExtraTrees; histogram gradient boosting; Ridge regression; PyTorch linear policies.</p><p class="small">Confidence, cost-sensitive targets, multi-sample augmentation, voting, exact expected reward and GRPO are feature / training / decision variants. GRPO trains the linear router, not Qwen. Exact expected reward means summing the four action rewards; it does not mean exact-match answer scoring.</p><p class="small"><b>Frozen lenient rule:</b> lowercase, remove Unicode punctuation and English articles, collapse whitespace; accept normalized equality or a complete gold word sequence inside the prediction. Reverse containment is rejected. Contradictory yes/no answers are rejected for boolean golds. All 4,267 available original review judgments agree.</p><p class="small"><b>Protocol:</b> no gold-derived prediction features. Training samples have equal total weight per question. Greedy + 4 / 8 sample variants are training-only unless explicitly called consensus. Upgrading pays for sunk weak calls; answer choice pays for both answers. Reused features are target-free; all labels, rewards and router weights were rebuilt.</p><p class="small"><b>Limits:</b> scorer reconstruction is not a recovered original source file; all test results are exploratory after repeated inspection. Small gains and unadjusted paired intervals do not establish stable superiority.</p></section>
<section><h2>6. Artifacts</h2><p class="small"><a href="../{run.relative_to(ROOT)}/REPORT.md">Full lenient report</a> · <a href="../{run.relative_to(ROOT)}/protocol.json">Protocol and scorer provenance</a> · <a href="../{run.relative_to(ROOT)}/results.json">All results as JSON</a> · <a href="reference_lenient_audit.json">Original-answer audit</a> · <a href="router_methods_results_strict_em.html">Separate exact-match diagnostic</a> · <a href="router_methods_results_exploratory.html">Earlier local bidirectional experiments</a></p></section><footer>Single-file offline HTML. Historical outputs and original labels remain unchanged.</footer></main><script type="application/json" id="data">{json.dumps(data,ensure_ascii=True,allow_nan=False)}</script>
<script>
const data=JSON.parse(document.getElementById('data').textContent);const colors={{baseline:'#64748b',classical:'#245bc4',sampling:'#9865bd',rl:'#c17825',architecture:'#16816b'}};const esc=s=>s.replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
function draw(rows){{const W=960,H=330,L=65,R=25,T=25,B=52;const xmin=Math.floor(Math.min(...data.map(r=>r.cost))/10)*10-2,xmax=Math.ceil(Math.max(105,...data.map(r=>r.cost))/10)*10+2;const ymin=Math.floor(Math.min(...data.map(r=>r.accuracy)))-1,ymax=Math.ceil(Math.max(...data.map(r=>r.accuracy)))+1;const x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);let s=`<svg viewBox="0 0 ${{W}} ${{H}}" role="img" aria-label="Lenient accuracy versus model-call cost">`;
for(let v=Math.ceil(xmin/10)*10;v<=xmax;v+=10)s+=`<line x1="${{x(v)}}" y1="${{T}}" x2="${{x(v)}}" y2="${{H-B}}" stroke="#e4e9f0"/><text x="${{x(v)}}" y="${{H-B+23}}" text-anchor="middle">${{v}}%</text>`;
for(let v=ymin;v<=ymax;v++)s+=`<line x1="${{L}}" y1="${{y(v)}}" x2="${{W-R}}" y2="${{y(v)}}" stroke="#e4e9f0"/><text x="${{L-10}}" y="${{y(v)+4}}" text-anchor="end">${{v}}%</text>`;
const ss=data.find(r=>r.group==='baseline').accuracy;s+=`<line x1="${{x(100)}}" y1="${{T}}" x2="${{x(100)}}" y2="${{H-B}}" stroke="#8290a3" stroke-dasharray="5 5"/><line x1="${{L}}" y1="${{y(ss)}}" x2="${{W-R}}" y2="${{y(ss)}}" stroke="#8290a3" stroke-dasharray="5 5"/>`;
for(const r of rows){{const label=esc(`${{r.name}}: ${{r.accuracy.toFixed(2)}}% lenient / ${{r.cost.toFixed(2)}}% SS cost`);s+=`<circle tabindex="0" role="img" aria-label="${{label}}" cx="${{x(r.cost)}}" cy="${{y(r.accuracy)}}" r="${{r.name.startsWith('Combined')?7:5}}" fill="${{colors[r.group]}}" stroke="white"><title>${{label}}</title></circle>`;}}s+=`<text x="${{W/2}}" y="${{H-9}}" text-anchor="middle">Inference cost (% of this cohort's direct SS)</text></svg>`;document.getElementById('chart').innerHTML=s;}}
function render(filter){{const rows=data.filter(r=>filter==='all'||r.group===filter||r.group==='baseline');document.querySelector('#main-results tbody').innerHTML=rows.map(r=>`<tr class="${{r.name.startsWith('Combined')?'highlight':''}}"><td><a href="../${{encodeURI(r.source)}}">${{esc(r.name)}}</a></td><td class="num">${{r.accuracy.toFixed(2)}}%</td><td class="num">${{r.cost.toFixed(2)}}%</td><td class="note">${{esc(r.note)}}</td></tr>`).join('');draw(rows);}}
document.querySelectorAll('[data-filter]').forEach(b=>b.addEventListener('click',()=>{{document.querySelectorAll('[data-filter]').forEach(a=>a.setAttribute('aria-pressed',String(a===b)));render(b.dataset.filter);}}));render('all');
</script></body></html>'''
    path=ROOT/'reports/router_methods_results_lenient.html';path.write_text(html)
    (ROOT/'reports/router_methods_results.html').write_text(html)
    print(path)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);main(p.parse_args().run.resolve())
