"""Build a self-contained English report from completed experiment artifacts."""
from html import escape
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[4]
OUT=ROOT/'reports/router_methods_results.html'

def read(path):return json.loads((ROOT/path).read_text())
def link(path,label):return f'<a href="../{escape(path,quote=True)}">{escape(label)}</a>'
def rec(name,test,group,note='',source='',selected=False):
    return {'name':name,'accuracy':100*test.get('lenient_accuracy',test.get('accuracy',0)),
        'strict':100*test.get('strict_accuracy',0),
        'cost':test.get('cost_pct_of_direct_SS',test.get('cost_pct_SS',0)),
        'group':group,'note':note,'source':source,'selected':selected}
def table(rows,strict=False):
    cols='<th scope="col">Method / configuration</th><th scope="col">Accuracy</th>'
    if strict:cols+='<th scope="col">Strict EM</th>'
    cols+='<th scope="col">Cost / SS</th>'
    if not strict:cols+='<th scope="col">What changed</th>'
    body=[]
    for r in rows:
        title=escape(r['name'])
        if r.get('source'):title=link(r['source'],r['name'])
        s=f'<tr><td>{title}</td><td class="num">{r["accuracy"]:.2f}%</td>'
        if strict:s+=f'<td class="num">{r["strict"]:.2f}%</td>'
        s+=f'<td class="num">{r["cost"]:.2f}%</td>'
        if not strict:s+=f'<td class="description">{escape(r["note"])}</td>'
        body.append(s+'</tr>')
    return '<div class="table-scroll"><table><thead><tr>'+cols+'</tr></thead><tbody>'+''.join(body)+'</tbody></table></div>'

def main():
    initial=read('runs/nonllm_router_improvements_v2/results/comparison.json')
    extended=read('runs/nonllm_grpo_extended_v1/results.json')
    arch=read('runs/nonllm_architecture_v1/results.json')
    combined=read('runs/nonllm_architecture_combined_v1/results.json')
    legacy=read('runs/baselines_real_v1/results.json')
    pilot=read('runs/nonllm_cascade_configA_ablation_v1/results.json')
    cost_audit=read('runs/nonllm_architecture_combined_v1/token_cost_audit.json')
    initial_map={r['name']:r for r in initial['results']}
    ext_map={r['name']:r for r in extended['results']}
    current=[]
    grid=[json.loads(s) for s in (ROOT/'runs/nonllm_cascade_four_gpu_full_v1/labels/full_run_p100/router_data_lenient.jsonl').read_text().splitlines()]
    test=[g for g in grid if g['split']=='test']
    assert len(test)==1481
    fixedstrict=[100*sum(g['em_strict'][j] for g in test)/len(test) for j in range(4)]
    current.append({'name':'Direct SS baseline','accuracy':100*initial['fixed_accuracies'][3],
        'strict':fixedstrict[3],'cost':100.,'group':'baseline','note':'Strong retrieval + strong answer; no weak calls.','source':'runs/nonllm_cascade_four_gpu_full_v1/router/REPORT.md','selected':False})
    choices=[
        ('frozen_baseline','October-style non-LLM cascade','classical','Multinomial NB; weak-first review, then escalation.'),
        ('gain_trees','Gain regression','classical','ExtraTrees learns strong-minus-weak correctness gain.'),
        ('cost_sensitive_logreg','Cost-sensitive supervision','classical','Weighted logistic regression balances correctness gain and cost.'),
        ('confidence_outcome_logreg','Confidence + outcome labels','classical','Adds weak-answer confidence and outcome-based retrieval targets.'),
        ('sampled_augmentation_logreg','Multi-sample training','sampling','Greedy + 4 sampled weak answers in training; one answer at deployment.'),
        ('sampled_consensus_logreg','Five-answer consensus','sampling','Votes over greedy + 4 sampled weak answers at deployment; all calls paid.'),
        ('exact_reward_lambda0.0','Exact expected-reward training','rl','Optimizes all four cached action rewards; initial suite validation selection.'),
        ('grpo_lambda0.0','Initial GRPO','rl','Linear non-LLM policy; 8 action samples per group.')]
    for key,name,group,note in choices:
        r=initial_map[key];current.append(rec(name,r['test'],group,note,'runs/nonllm_router_improvements_v2/results/COMPARISON.md'))
    for obj,label in [('max_accuracy','Extended GRPO: maximum accuracy'),('ss_budget','Extended policies: SS budget'),('cheap95','Extended policies: lowest cost')]:
        name=extended['global_selected'][obj];r=ext_map[name]
        note={'max_accuracy':'120 epochs; group size 16; greedy + 4 training samples; no cost penalty.',
              'ss_budget':'Exact reward, no-confidence features; validation mean cost capped at SS.',
              'cheap95':'Exact reward + confidence; cheapest validation policy reaching 95% of SS accuracy.'}[obj]
        current.append(rec(label,r['test'],'rl',note,'runs/nonllm_grpo_extended_v1/REPORT.md'))
    for key,label,note in [
        ('direct4_cheap95','Pre-retrieval direct routing','Choose WW / WS / SW / SS before any model retrieval; cost penalty 0.05.'),
        ('pre_ret_qa_max_accuracy','Pre-retrieval routing + QA review','Validation maximum-accuracy choice falls back to direct SS.'),
        ('fixed_S_qa_max_accuracy','Fixed strong retrieval + QA review','Validation maximum-accuracy choice falls back to direct SS.'),
        ('post_answer_chooser_max_accuracy','Post-answer chooser','Strong retrieval + both answers; ExtraTrees chooses which answer to retain.')]:
        current.append(rec(label,arch['results'][key]['test'],'architecture',note,'runs/nonllm_architecture_v1/REPORT.md'))
    current.append(rec('Combined: pre-routing + answer chooser',combined['results']['ss_budget']['test'],'architecture',
        'Validation SS-budget selection: direct pre-routing, then answer choice on initially-SS questions.',
        'runs/nonllm_architecture_combined_v1/REPORT.md',True))

    inventory_initial=[rec(r['name'],r['test'],'initial') for r in initial['results']]
    inventory_extended=[rec(r['name'],r['test'],'extended') for r in extended['results']]
    inventory_arch=[rec(k,r['test'],'architecture') for k,r in arch['results'].items()]
    inventory_combined=[rec(k,r['test'],'combined') for k,r in combined['results'].items()]
    legacyrows=[]
    for k,t in legacy['composed'].items():
        # The original file uses its own SS-relative cost field.
        legacyrows.append({'name':k,'accuracy':100*t['lenient_accuracy'],'strict':100*t['strict_accuracy'],
            'cost':t['cost_pct_of_SS'],
            'note':'Three independent classifiers: retrieval, QA-W, QA-S.'})
    pilotrow=rec('October-style pilot: random forest',pilot['test'],'pilot',
        'Weak-answer input absent; flat parameter-ratio cost. This is an ablation.','runs/nonllm_cascade_configA_ablation_v1/REPORT.md')
    familytable='''<div class="table-scroll"><table class="method-table"><thead><tr><th>Design</th><th>Before October</th><th>October Config A</th></tr></thead><tbody>
      <tr><td>LLM router</td><td>One shared Qwen2.5-1.5B LoRA adapter; stage-specific prompts.</td><td>Separate retrieval and QA LoRA adapters; QA adapter shared across both retrieval branches.</td></tr>
      <tr><td>Retrieval decision</td><td>Question + full candidate pool → choose weak or strong retrieval before running it.</td><td>Run weak retrieval first; question + weak selected evidence → keep it or escalate.</td></tr>
      <tr><td>Answer decision</td><td>Question + selected evidence → choose the answering model before generating an answer.</td><td>Generate a weak answer first; evidence + weak answer → keep it or escalate.</td></tr>
      <tr><td>Training / selection</td><td>Weighted SFT; sequential or mixed-stage training. Best checkpoint by mean stage probe accuracy.</td><td>Weighted SFT; best checkpoint by validation AUC. Joint validation thresholds choose cheapest ≥95% of SS accuracy.</td></tr>
      <tr><td>Unresolved labels</td><td>Default: skip unresolved labels in classifier training.</td><td>Map X to weak, avoiding escalation when neither route helps.</td></tr>
      <tr><td>Our non-LLM counterpart</td><td>Three classical classifiers: retrieval, QA-W and QA-S.</td><td>Two classical classifiers: retrieval and one shared QA classifier; weak-answer features and paid weak-first calls.</td></tr>
    </tbody></table></div>'''
    reference='''<div class="reference-results"><strong>Results reported in the supplied reference documentation</strong>
      <p>Before October: mixed-stage LLM router <b>41.8% accuracy / 55% cost</b>; sequential training 42.4% / 65%; fixed-strong-retrieval QA router 43.89% / 68.2%. The documented direct-SS baseline is 53.54%.</p>
      <p>October Config A: LLM cascade <b>50.10% accuracy / 43% cost</b>; retrieval / QA validation AUC 0.771 / 0.707.</p>
      <p class="muted">In the supplied original / October label grid, direct SS is <b>52.79% on validation</b> (586/1,110) and <b>53.54% on test</b> (793/1,481). These author-reported router results are not locally rerun LLM-router results and must not be ranked against the new-grid results below.</p>
    </div>'''
    sources=[('extracted/Python Codes/sft_router.py','Before-October reference trainer'),
        ('extracted/Python Codes/RUN_COMMANDS.md','Before-October reported results'),
        ('October 2026/slr_code_configA/sft_router5.py','October reference trainer'),
        ('October 2026/slr_code_configA/README.md','October method and reported results'),
        ('STAIR/nonLLM/experiments/router_improvements/ARCHITECTURE.md','New architecture protocol'),
        ('runs/nonllm_architecture_combined_v1/token_cost_audit.json','Recorded-token cost audit')]
    combos=combined['results']['ss_budget']['test'];post=arch['results']['post_answer_chooser_max_accuracy']['test']
    html=TEMPLATE.replace('@@METHODS@@',familytable).replace('@@REFERENCE@@',reference)
    html=html.replace('@@CURRENT_ROWS@@',table(current))
    html=html.replace('@@LEGACY@@',table(legacyrows,True)).replace('@@PILOT@@',table([pilotrow]))
    html=html.replace('@@INVENTORY_INITIAL@@',table(inventory_initial,True))
    html=html.replace('@@INVENTORY_EXTENDED@@',table(inventory_extended,True))
    html=html.replace('@@INVENTORY_ARCH@@',table(inventory_arch,True))
    html=html.replace('@@INVENTORY_COMBINED@@',table(inventory_combined,True))
    html=html.replace('@@SOURCES@@',' · '.join(link(p,label) for p,label in sources))
    html=html.replace('@@DATA@@',json.dumps(current,ensure_ascii=True,allow_nan=False))
    html=html.replace('@@COMBO_ACC@@',f"{100*combos['accuracy']:.2f}").replace('@@COMBO_COST@@',f"{combos['cost_pct_SS']:.2f}")
    html=html.replace('@@ACTUAL_COST@@',f"{cost_audit['combined_ss_budget']['actual_cached_token_cost_pct_test_SS']:.2f}")
    assert '@@' not in html
    assert abs(current[-1]['accuracy']-59.41931127616475)<1e-8
    OUT.parent.mkdir(parents=True,exist_ok=True);OUT.write_text(html)
    print(OUT)

TEMPLATE='''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Router Methods &amp; Results — CS707</title>
<style>
:root{--ink:#17243b;--muted:#59677c;--line:#dce3eb;--blue:#245bc4;--green:#16634b;--soft:#f1f5fa}
*{box-sizing:border-box}body{margin:0;background:#f7f9fc;color:var(--ink);font:15px/1.6 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1100px;margin:0 auto;padding:42px 28px 50px}h1{font-size:34px;line-height:1.2;letter-spacing:-1px;margin:8px 0 12px}h2{font-size:23px;margin:0 0 12px;letter-spacing:-.4px}h3{font-size:17px;margin:0 0 8px}p{margin:8px 0 14px}.eyebrow{color:var(--blue);font-size:12px;font-weight:750;letter-spacing:1.8px;text-transform:uppercase}.muted,.description{color:var(--muted)}
header{margin-bottom:26px}.intro{max-width:850px;color:var(--muted)}a{color:var(--blue);text-decoration:none}a:hover{text-decoration:underline}a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #eaa32f;outline-offset:3px}
nav{display:flex;gap:18px;flex-wrap:wrap;font-size:13px}.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}.card{padding:20px;background:white;border:1px solid var(--line);border-radius:12px}.card.best{border-color:#8fc8b4;background:#f0faf5}.card .label{font-size:12px;font-weight:700;text-transform:uppercase;color:var(--muted)}.metric{font-size:32px;font-weight:750;line-height:1.35;margin:8px 0}.card p{font-size:13px;margin:0;color:var(--muted)}
section{background:white;border:1px solid var(--line);border-radius:14px;padding:25px;margin-top:20px;scroll-margin-top:18px}.section-note{font-size:13px;color:var(--muted)}.table-scroll{overflow:auto}table{width:100%;border-collapse:collapse;font-size:13px}th{text-align:left;color:var(--muted);font-size:11px;text-transform:uppercase;letter-spacing:.5px;background:#f5f7fa;padding:12px 10px;white-space:nowrap}td{padding:12px 10px;border-bottom:1px solid #edf0f4;vertical-align:top}td:first-child{font-weight:600;min-width:180px}td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}th:nth-child(2),th:nth-child(3){text-align:right}.method-table th{text-align:left!important}.method-table td:first-child{width:19%;min-width:140px}.method-table td:not(:first-child){width:40.5%}tr:last-child td{border-bottom:0}
.reference-results{background:var(--soft);border-radius:10px;padding:16px 18px;margin-top:16px;font-size:13px}.reference-results p:last-child{margin-bottom:0}.notice{background:#fff8e9;border-left:3px solid #dba344;padding:12px 16px;font-size:13px;margin:16px 0}.badge{display:inline-block;font-size:10px;white-space:nowrap;font-weight:700;background:#e2f2e9;color:var(--green);padding:2px 7px;border-radius:20px;margin-left:7px}.highlight td{background:#f0faf5}.filters{display:flex;gap:7px;flex-wrap:wrap;margin:16px 0}button{border:1px solid var(--line);border-radius:20px;background:white;color:var(--muted);padding:6px 12px;font:inherit;font-size:12px;cursor:pointer}button[aria-pressed="true"]{background:var(--ink);color:white;border-color:var(--ink)}
.chart{width:100%;height:auto;display:block;background:#fbfcfe;border-radius:10px;border:1px solid #edf0f4}.chart-note{font-size:12px;color:var(--muted);margin-top:8px}.takeaways{display:grid;grid-template-columns:repeat(2,1fr);gap:12px}.takeaway{padding:15px;background:var(--soft);border-radius:10px}.takeaway p{font-size:13px;margin-bottom:0}.pipeline{font-size:13px;line-height:2;padding:15px 18px;background:#f0faf5;border:1px solid #b7dccb;border-radius:10px;margin:16px 0}.pipeline strong{color:var(--green)}details{border-top:1px solid var(--line);padding:14px 0}details:first-of-type{border-top:0}summary{font-weight:650;cursor:pointer;font-size:14px}details p{font-size:13px;color:var(--muted)}details .table-scroll{margin-top:14px}footer{margin-top:24px;color:var(--muted);font-size:12px}.legend{font-size:12px;color:var(--muted)}.legend span{display:inline-block;margin-right:15px}.dot{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:5px}.source-list{font-size:12px;line-height:2}
@media(max-width:700px){main{padding:24px 14px}h1{font-size:28px}.cards{grid-template-columns:1fr}.metric{font-size:28px}section{padding:18px}.takeaways{grid-template-columns:1fr}.description{min-width:220px}.method-table{min-width:650px}nav{gap:12px}.card{padding:15px}.cards{gap:9px}}
@media print{body{background:white}main{padding:0;max-width:none}section,.card{break-inside:avoid;box-shadow:none}.filters,nav{display:none}details:not([open]){display:none}a{color:inherit}h1{font-size:26px}.chart{max-height:250px}}
</style></head><body><main>
<header><div class="eyebrow">CS707 · Experiment summary · 13 October 2026</div><h1>Router methods &amp; results</h1>
<p class="intro">Two reference LLM-router designs, our non-LLM adaptations, and the improvements tested so far. The supplied reference SS validation baseline is <b>52.79% lenient EM</b>. Locally regenerated answers form a separate experiment, not a reproduction of that baseline.</p>
<nav aria-label="Page sections"><a href="#reference">Reference methods</a><a href="#results">Current results</a><a href="#findings">What we learned</a><a href="#inventory">All experiments</a></nav></header>
<div class="cards">
 <div class="card"><div class="label">Reference direct SS · Validation</div><div class="metric">52.79%</div><p>Lenient EM · supplied labels<br>586 correct / 1,110 questions</p></div>
 <div class="card best"><div class="label">Combined router · New-grid test</div><div class="metric">@@COMBO_ACC@@%</div><p>@@COMBO_COST@@% of new-grid SS cost<br>Compared with new-grid test SS = 59.08%</p></div>
 <div class="card"><div class="label">Direct pre-routing · New-grid test</div><div class="metric">58.74%</div><p>93.28% of new-grid SS cost<br>Results cannot use 52.79% as their comparator</p></div>
</div>
<p class="section-note"><b>Notation:</b> W = Qwen2.5-1.5B-Instruct; S = Qwen2.5-7B-Instruct. In WW / WS / SW / SS, the first letter is retrieval and the second is answering. Accuracy is end-to-end lenient EM, not router classification accuracy.</p>

<section id="reference"><h2>1. The two reference methods</h2><p class="section-note">The reference routers are LLMs trained with LoRA. Our adaptations replace the routing model with classical machine learning.</p>
@@METHODS@@@@REFERENCE@@
<details><summary>Our early non-LLM adaptations — separate benchmark</summary>
<p><b>Before-October counterpart:</b> TF-IDF logistic regression, linear SVM, multinomial NB, majority, or numerical-feature random forest. Three classifiers independently route retrieval, QA-W and QA-S.</p>
<p>These original-grid experiments have <b>SS = 53.54%</b> and fixed FLOP-style costs. They are not directly comparable to the current regenerated-grid results.</p>@@LEGACY@@
<p><b>October pilot:</b> shared QA classifier and weak-first cascade, but missing weak-answer inputs. Uses the supplied old outcome grid (SS = 53.54%) and a flat parameter-ratio cost proxy.</p>@@PILOT@@
</details></section>

<section id="results"><h2>2. Current non-LLM results</h2><p class="section-note">Regenerated Instruct grid · 4,814 train / 1,110 validation / 1,481 test questions. All rows below use the same direct-SS denominator. Router models and thresholds use train/validation; the test set has been examined repeatedly.</p>
<div class="filters" aria-label="Filter current results">
<button type="button" data-filter="all" aria-pressed="true">All</button><button type="button" data-filter="classical" aria-pressed="false">Classical</button><button type="button" data-filter="sampling" aria-pressed="false">Sampling</button><button type="button" data-filter="rl" aria-pressed="false">Reward / GRPO</button><button type="button" data-filter="architecture" aria-pressed="false">Architecture</button>
</div><div id="main-results">@@CURRENT_ROWS@@</div>
<p class="section-note">Selected configurations are shown, rather than every hyperparameter. Extended-policy rows are the global validation selections for their stated objectives; architecture rows are selected within each workflow. The combined row is selected under the validation SS-budget objective.</p>
<h3>Accuracy versus inference cost</h3><div id="chart" aria-label="Accuracy versus cost chart"></div>
<div class="legend"><span><i class="dot" style="background:#64748b"></i>Baseline</span><span><i class="dot" style="background:#245bc4"></i>Classical</span><span><i class="dot" style="background:#9865bd"></i>Sampling</span><span><i class="dot" style="background:#c17825"></i>Reward / GRPO</span><span><i class="dot" style="background:#16816b"></i>Architecture</span></div>
<p class="chart-note">Higher accuracy and lower cost are preferable. Hover or focus on a point for its values. Dashed lines mark direct SS. Reference LLM results use different grids/costs and are excluded.</p>
</section>

<section id="findings"><h2>3. What changed, and what worked</h2>
<div class="pipeline"><strong>Current combined workflow</strong><br>Question + candidate pool → non-LLM pre-router chooses a direct route.<br>If it initially chooses SS → run strong retrieval and both QA models → non-LLM chooser retains one answer.<br>Other routes → run only their selected retrieval and QA models. Both answers are paid for whenever generated.</div>
<div class="takeaways">
 <div class="takeaway"><h3>Remove expensive weak-first retrieval</h3><p>In the old cascade, choosing SS still pays for weak retrieval and weak QA. Always-SS cascade costs <b>121.44%</b> of direct SS. The unconstrained extended GRPO mostly escalates, so it costs <b>114.67%</b> without beating SS.</p></div>
 <div class="takeaway"><h3>Choose between answers after escalation</h3><p>The standalone ExtraTrees chooser corrects <b>9</b> SS errors and introduces <b>2</b>, reaching <b>59.55%</b> at <b>100.56%</b> cost. The combined system corrects <b>10</b> and introduces <b>5</b>, reaching <b>59.42%</b> at <b>98.69%</b> cost.</p></div>
 <div class="takeaway"><h3>More samples did not solve routing</h3><p>Increasing random training answers from 4 to 8 keeps the selected maximum-accuracy GRPO result at <b>58.34%</b>. At 120 epochs and group size 32, about <b>91%</b> of λ=0 groups have zero relative advantage. Repeated action sampling often adds little learning signal.</p></div>
 <div class="takeaway"><h3>Learn comparative gains directly</h3><p>The new models use correctness differences instead of retrieval-coverage labels. Three-fold grouped training CV selects <b>Ridge</b> for pre-routing / QA gains and <b>ExtraTrees</b> for answer choice. Qwen weights are unchanged.</p></div>
</div>
<div class="notice"><b>Small gains, not yet confirmed.</b> Combined accuracy gain: +0.34 pp; unadjusted paired 95% interval <b>[−0.14, +0.88] pp</b>, which includes zero. Standalone chooser: +0.47 pp, interval [+0.07, +0.95] pp. Repeated test inspection and multiple comparisons make these exploratory observations; an independent holdout is needed to establish stable superiority.</div>
<p class="section-note"><b>Cost audit:</b> the tables retain the prior mean parameter-weighted token cost ruler. Recalculation using each question’s recorded token counts gives <b>@@ACTUAL_COST@@%</b> of test direct-SS cost for the combined system. CPU feature extraction and routing overhead are excluded from both estimates; these are not wall-clock measurements.</p>
</section>

<section id="inventory"><h2>4. Full experiment inventory</h2><p class="section-note">Expand for all evaluated family/configuration results. Identical outcomes may appear under different validation objectives. Accuracy = lenient EM; Strict EM = normalized exact match.</p>
<details><summary>Initial improvement suite — 27 comparisons</summary><p>Classical features, gain / cost-sensitive targets, confidence, four-sample augmentation, five-answer consensus, exact reward, and initial GRPO (G=8). GRPO policies are linear TF-IDF/numeric policies, averaged over three seeds; no LLM router is trained.</p>@@INVENTORY_INITIAL@@</details>
<details><summary>Extended reward / GRPO suite — 24 selected operating points</summary><p>144 seed-specific training streams; 144 validation candidates. Base, confidence, greedy + 4 samples, and greedy + 8 samples; 30/60/120 epochs; GRPO groups 8/16/32. Three objectives per family: maximum accuracy, SS budget, and cheapest ≥95% SS validation accuracy. Augmentation is training-only; deployment uses one greedy weak answer.</p>@@INVENTORY_EXTENDED@@</details>
<details><summary>Architecture experiments — 12 selected operating points</summary><p>Four workflows × three objectives. Model families are chosen by three-fold grouped CV on training questions. TF-IDF, scaling and SVD fit within each fold. A direct-SS fallback skips weak calls. The post-answer chooser always pays for both QA calls.</p>@@INVENTORY_ARCH@@</details>
<details><summary>Combined pre-routing + answer choice — 3 objectives</summary><p>Reuses the train-CV-selected models. Seven pre-routing cost penalties and five chooser thresholds are compared on validation, with uncombined routes and direct SS also available. Validation choices are saved before test replay.</p>@@INVENTORY_COMBINED@@</details>
</section>

<section id="protocol"><h2>5. Reading the results correctly</h2>
<div class="notice"><b>Scorer audit:</b> the local bidirectional lenient rule adds 81 reverse-only matches on validation. On the same regenerated SS answers, validation scores are 59.73% bidirectional versus 52.43% forward-only (582/1,110); the supplied reference is 52.79% (586/1,110). On test, the local scores are 59.08% bidirectional versus 52.60% forward-only. The missing original scorer and inconsistent archive descriptions prevent certifying an exact reproduction. Current router results were trained and evaluated using the local bidirectional rule. <a href="ss_baseline_audit.json">Read the baseline audit</a>.</div>
<p class="section-note"><b>Metric:</b> main results use lenient matching, not strict exact match. Strict EM requires equality after normalization; the local lenient scorer also accepts guarded whole-word containment in either direction. The reference's original final scoring script was not supplied. For context, original-grid test SS strict EM is 48.14%; regenerated-grid test SS strict EM is 47.20%. The higher regenerated lenient score therefore does not establish a stronger model or reproduce the reference baseline.</p>
<div class="table-scroll"><table><thead><tr><th>Direct-SS baseline / answer grid</th><th>Validation (1,110)</th><th>Test (1,481)</th></tr></thead><tbody><tr><td>Supplied original / October labels</td><td class="num">52.79% (586 correct)</td><td class="num">53.54% (793 correct)</td></tr><tr><td>Locally regenerated Instruct answers</td><td class="num">59.73% (663 correct)</td><td class="num">59.08% (875 correct)</td></tr></tbody></table></div>
<p class="section-note"><b>Local versus reported:</b> LLM reference numbers come from the supplied documentation; the local experiments train non-LLM routers. The October pilot lacks complete inputs. The complete local grid uses regenerated answers and an explicitly implemented lenient scorer, so it is not a numerical reproduction of the reference 50.10% / 43% result.</p>
<p class="section-note"><b>Features and labels:</b> gold answers and observed correctness are training targets or evaluation data, never prediction features. The pre-router sees only the question and supplied candidates. QA review uses already-paid evidence and weak answers; post-answer choice also sees the paid strong answer. The supplied retrieval labels are coverage-based despite an outcome-based description in the archive README.</p>
<p class="section-note"><b>Costs and selection:</b> escalation includes sunk weak calls; consensus includes all five weak answers. Confidence costs assume extraction from ordinary generation logits. An additional confidence forward pass would need to be charged. “SS budget” constrains mean validation cost, not every question or every future batch. Cost-matched fixed-policy mixtures are a stronger baseline than cascade-only mixtures; the original local cascade is about −0.14 pp versus the direct-policy mixture at its cost.</p>
<div class="source-list">@@SOURCES@@</div>
</section><footer>Built from local experiment artifacts. Single-file HTML; works offline. Source links refer to files in the same project folder. No new training or inference is launched by this page.</footer>
</main>
<script type="application/json" id="result-data">@@DATA@@</script>
<script>
const data=JSON.parse(document.getElementById('result-data').textContent);
const colors={baseline:'#64748b',classical:'#245bc4',sampling:'#9865bd',rl:'#c17825',architecture:'#16816b'};
const escapeHTML=s=>s.replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
function renderRows(filter){const rows=data.filter(r=>filter==='all'||r.group===filter||r.group==='baseline');const body=document.querySelector('#main-results tbody');
body.innerHTML=rows.map(r=>`<tr class="${r.selected?'highlight':''}"><td><a href="../${encodeURI(r.source)}">${escapeHTML(r.name)}</a>${r.selected?'<span class="badge">COMBINED</span>':''}</td><td class="num">${r.accuracy.toFixed(2)}%</td><td class="num">${r.cost.toFixed(2)}%</td><td class="description">${escapeHTML(r.note)}</td></tr>`).join('');drawChart(rows);}
function drawChart(rows){const W=960,H=350,L=66,R=28,T=28,B=54;const xmin=65,xmax=125,ymin=54,ymax=61;
const x=v=>L+(v-xmin)/(xmax-xmin)*(W-L-R),y=v=>H-B-(v-ymin)/(ymax-ymin)*(H-T-B);
let s=`<svg class="chart" viewBox="0 0 ${W} ${H}" role="img" aria-labelledby="chart-title chart-desc"><title id="chart-title">Current-grid accuracy and cost</title><desc id="chart-desc">Each point is a validation-selected configuration. Direct SS has 59.08 percent accuracy at 100 percent cost. The combined router has 59.42 percent accuracy at 98.69 percent cost.</desc>`;
for(let v=70;v<=120;v+=10)s+=`<line x1="${x(v)}" y1="${T}" x2="${x(v)}" y2="${H-B}" stroke="#e4e9f0"/><text x="${x(v)}" y="${H-B+23}" text-anchor="middle" fill="#59677c" font-size="12">${v}%</text>`;
for(let v=54;v<=61;v++)s+=`<line x1="${L}" y1="${y(v)}" x2="${W-R}" y2="${y(v)}" stroke="#e4e9f0"/><text x="${L-12}" y="${y(v)+4}" text-anchor="end" fill="#59677c" font-size="12">${v}%</text>`;
s+=`<line x1="${x(100)}" y1="${T}" x2="${x(100)}" y2="${H-B}" stroke="#8995a7" stroke-dasharray="5 5"/><line x1="${L}" y1="${y(59.081701553)}" x2="${W-R}" y2="${y(59.081701553)}" stroke="#8995a7" stroke-dasharray="5 5"/>`;
for(const r of rows){const title=escapeHTML(`${r.name}: ${r.accuracy.toFixed(2)}% accuracy; ${r.cost.toFixed(2)}% SS cost`);s+=`<circle tabindex="0" role="img" aria-label="${title}" cx="${x(r.cost)}" cy="${y(r.accuracy)}" r="${r.selected?8:5.5}" fill="${colors[r.group]}" stroke="${r.selected?'#103a30':'white'}" stroke-width="${r.selected?2:1.5}" opacity=".9"><title>${title}</title></circle>`;}
s+=`<text x="${(W+L-R)/2}" y="${H-9}" text-anchor="middle" fill="#59677c" font-size="12">Inference cost (% of direct SS)</text><text transform="translate(17 ${(H+T-B)/2}) rotate(-90)" text-anchor="middle" fill="#59677c" font-size="12">Test accuracy</text></svg>`;document.getElementById('chart').innerHTML=s;}
document.querySelectorAll('[data-filter]').forEach(button=>button.addEventListener('click',()=>{document.querySelectorAll('[data-filter]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));renderRows(button.dataset.filter);}));
renderRows('all');
</script></body></html>'''

if __name__=='__main__':main()
