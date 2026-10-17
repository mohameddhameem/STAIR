"""Compare SS x40 with gold-assisted minimum-budget and minimum-cost policies."""
import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sqlite3


def main(root, cohort, output):
    ids = json.loads((cohort / 'config.json').read_text())['indices']
    sets = {}
    with (cohort / 'lenient_ranges/per_question.csv').open() as handle:
        for row in csv.DictReader(handle):
            if row['metric'] == 'lenient_em':
                sets[int(row['index']), row['pair']] = json.loads(row['correct_ks'])
    assert len(ids) == 500 and len(sets) == 2000
    dbs = {}
    for shard in {i % 4 for i in ids}:
        for stage, models in [('retrieval', 'WS'), ('answers', 'WS'), ('extraction', 'S')]:
            for model in models:
                base = cohort / 'evaluation_v2' if stage == 'extraction' else root
                path = base / 'runs' / f'{stage}_{model}_{shard}.sqlite'
                dbs[stage, model, shard] = sqlite3.connect(f'file:{path}?mode=ro', uri=True)

    def read(stage, model, i, key):
        row = dbs[stage, model, i % 4].execute('SELECT payload FROM results WHERE key=?', (key,)).fetchone()
        assert row is not None, key
        return json.loads(row[0])

    collected = {name: [] for name in ['ss40', 'min_k_then_cost', 'min_cost_then_k']}
    per_question = []
    try:
        for i in ids:
            retrieval = {m: read('retrieval', m, i, f'{i}:ret:{m}') for m in 'WS'}
            qa = {p: read('answers', p[1], i, f'{i}:qa:{p}') for p in ['WW', 'WS', 'SW', 'SS']}
            for v in qa.values():
                assert len(v['outputs']) == 40
            cost_cache = {}

            def cost(pair, k):
                if (pair, k) in cost_cache:
                    return cost_cache[pair, k]
                rt, batch = retrieval[pair[0]], qa[pair]
                key = f'{i}:fuse:{pair}:{k}'
                fused = read('answers', pair[1], i, key)
                ex = read('extraction', 'S', i, key)
                assert hashlib.sha256(fused['outputs'][0]['text'].encode()).hexdigest() == ex['source_sha256']
                rp, ap = [1.5 if m == 'W' else 7 for m in pair]
                parts = {
                    'retrieval': rp * (rt['input_tokens'] + rt['outputs'][0]['meta_info']['completion_tokens']),
                    'answer': ap * (batch['input_tokens'] + sum(o['meta_info']['completion_tokens'] for o in batch['outputs'][:k])),
                    'fusion': ap * (fused['input_tokens'] + fused['outputs'][0]['meta_info']['completion_tokens']),
                }
                usage = ex['extraction'].get('usage', {})
                parts['reader'] = 7 * (usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0))
                value = {'pair': pair, 'k': k, 'correct': int(k in sets[i, pair]),
                         'generation_cost': parts['retrieval'] + parts['answer'] + parts['fusion'],
                         'total_cost': sum(parts.values()), **parts}
                cost_cache[pair, k] = value
                return value

            baseline = cost('SS', 40)
            successful = [cost(pair, k) for pair in ['WW', 'WS', 'SW', 'SS'] for k in sets[i, pair]]
            selected = {
                'ss40': baseline,
                'min_k_then_cost': min(successful, key=lambda v: (v['k'], v['total_cost'], v['pair'])) if successful else baseline,
                'min_cost_then_k': min(successful, key=lambda v: (v['total_cost'], v['k'], v['pair'])) if successful else baseline,
            }
            for policy, value in selected.items():
                collected[policy].append(value)
                per_question.append({'index': i, 'policy': policy, 'fallback': int(not successful), **value})
    finally:
        for db in dbs.values():
            db.close()
    summary = {
        'questions': len(ids), 'metric': 'extracted_lenient_em',
        'cost_units': 'Billion model parameters times logical prompt/output tokens; same proxy as gguf_analyze.py. QA shared prompt counted once. Includes one selected retrieval, k answer completions, one selected fusion and one selected extraction.',
        'qualification': 'Gold-assisted retrospective oracle on a frozen availability-based training cohort. No search/controller cost. Assumes correct choice is known before execution; not the cost of exhaustively searching all configurations. Not measured GPU time, FLOPs or dollars. Fusion prompt tokens counted in full, independent of incidental cache reuse.',
        'failure_policy': 'If every pair/budget is incorrect, use SS x40, the same as baseline (86 questions). Never drop failed questions.',
        'policy_definitions': {
            'ss40': 'Always SS, 40 answer samples, one fusion and one extraction.',
            'min_k_then_cost': 'Among successful configurations choose the smallest k, then lowest total-cost model pair. Failed questions fall back to SS x40.',
            'min_cost_then_k': 'Among successful configurations choose the lowest total-cost pair/k, then smallest k. Failed questions fall back to SS x40.',
        }, 'policies': {},
    }
    for name, rows in collected.items():
        value = {'correct': sum(v['correct'] for v in rows),
                 'mean_k': sum(v['k'] for v in rows) / len(rows),
                 'pair_counts': dict(Counter(v['pair'] for v in rows)),
                 'k_histogram': dict(sorted(Counter(v['k'] for v in rows).items()))}
        for key in ['generation_cost', 'total_cost', 'retrieval', 'answer', 'fusion', 'reader']:
            value[key] = sum(v[key] for v in rows)
        value['em'] = value['correct'] / len(rows)
        summary['policies'][name] = value
    baseline = summary['policies']['ss40']
    for value in summary['policies'].values():
        value['em_gain_percentage_points'] = 100 * (value['em'] - baseline['em'])
        value['relative_em_gain_percent'] = 100 * (value['em'] / baseline['em'] - 1)
        for key in ['generation_cost', 'total_cost']:
            value[key + '_reduction_percent'] = 100 * (1 - value[key] / baseline[key])
        value['sample_count_reduction_percent'] = 100 * (1 - value['mean_k'] / 40)
    assert summary['policies']['min_k_then_cost']['correct'] == 414
    assert sum(row['fallback'] for row in per_question if row['policy'] == 'ss40') == 86
    output.mkdir(exist_ok=True, parents=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (output / 'per_question.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_question[0]))
        writer.writeheader(); writer.writerows(per_question)
    lines = ['# Retrospective oracle versus SS x40', '', summary['cost_units'], '', summary['qualification'], '', summary['failure_policy'], '',
             '| Policy | Correct / 500 | Lenient EM | Gain (pp) | Total cost reduction | Generation cost reduction | Mean k |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for name, v in summary['policies'].items():
        lines.append(f'| {name} | {v["correct"]} | {100*v["em"]:.1f}% | {v["em_gain_percentage_points"]:.1f} | {v["total_cost_reduction_percent"]:.2f}% | {v["generation_cost_reduction_percent"]:.2f}% | {v["mean_k"]:.3f} |')
    (output / 'comparison.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'cohort', 'output']:
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.cohort, args.output)
