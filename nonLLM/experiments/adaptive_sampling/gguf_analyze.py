"""Fixed validation baselines and retrospective test oracles, with reader costs."""
import argparse
import ast
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import re
import sqlite3
import string

import numpy as np

PAIRS = ['WW', 'WS', 'SW', 'SS']
METRICS = ['raw_em', 'raw_f1', 'raw_lenient', 'extracted_em', 'extracted_f1', 'extracted_lenient']


def official(root):
    env = {'re': re, 'string': string, 'Counter': Counter}
    tree = ast.parse((root / 'assets/hotpot_evaluate_v1.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in
             ['normalize_answer', 'f1_score', 'exact_match_score']]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<official-hotpot>', 'exec'), env)
    return env


def loose(text, gold, normalize):
    p = normalize(text)
    if not p:
        return 0.0
    guarded = bool(re.match(r'^(more than|less than|at least|at most|over|under)\b', p))
    return float(any(p == normalize(g) or (not guarded and normalize(g) and (
        re.search(r'(?<!\w)' + re.escape(normalize(g)) + r'(?!\w)', p) or
        re.search(r'(?<!\w)' + re.escape(p) + r'(?!\w)', normalize(g)))) for g in gold))


def main(root, output_name, pairs=None, reader_root=None):
    root = Path(root)
    pairs = list(pairs or PAIRS)
    if not pairs or len(set(pairs)) != len(pairs) or any(p not in PAIRS for p in pairs):
        raise ValueError('Invalid model-pair selection')
    reader_root = Path(reader_root) if reader_root else root
    configurations = len(pairs) * 40
    out = root / output_name
    out.mkdir(exist_ok=True)
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    splits = json.loads((root / 'assets/splits.json').read_text())['splits']
    env = official(root)
    scores = np.full((len(data), configurations, 6), np.nan, dtype=np.float32)
    generation_cost = np.full((len(data), configurations), np.nan, dtype=np.float64)
    total_cost = generation_cost.copy()
    extracted_available = np.zeros((len(data), configurations), dtype=bool)

    def open_db(name):
        path = (reader_root if name.startswith('extraction_') else root) / 'runs' / name
        return sqlite3.connect(f'file:{path}?mode=ro', uri=True) if path.exists() else None

    def get(db, key):
        if db is None:
            return None
        row = db.execute('SELECT payload FROM results WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    for shard in range(4):
        ret = {m: open_db(f'retrieval_{m}_{shard}.sqlite') for m in ['W', 'S']}
        ans = {m: open_db(f'answers_{m}_{shard}.sqlite') for m in ['W', 'S']}
        reader = open_db(f'extraction_S_{shard}.sqlite')
        for i in range(shard, len(data), 4):
            gold = data[i]['answers']
            for group, pair in enumerate(pairs):
                rm, am = pair
                rt, qa = get(ret[rm], f'{i}:ret:{rm}'), get(ans[am], f'{i}:qa:{pair}')
                if not rt or not qa:
                    continue
                emitted = 0
                for k in range(1, 41):
                    emitted += qa['outputs'][k - 1]['meta_info']['completion_tokens']
                    key = f'{i}:fuse:{pair}:{k}'
                    fused = get(ans[am], key)
                    if not fused:
                        continue
                    text = fused['outputs'][0]['text']
                    j = group * 40 + k - 1
                    scores[i, j, :3] = [max(env['exact_match_score'](text, g) for g in gold),
                                        max(env['f1_score'](text, g)[0] for g in gold),
                                        loose(text, gold, env['normalize_answer'])]
                    rp, ap = (1.5 if rm == 'W' else 7), (1.5 if am == 'W' else 7)
                    cost = rp * (rt['input_tokens'] + rt['outputs'][0]['meta_info']['completion_tokens'])
                    cost += ap * (qa['input_tokens'] + emitted + fused['input_tokens'] + fused['outputs'][0]['meta_info']['completion_tokens'])
                    generation_cost[i, j] = cost
                    ex = get(reader, key)
                    if ex is None:
                        continue
                    ex = ex['extraction']
                    predicted = ex['answer'] or ''
                    scores[i, j, 3:] = [max(env['exact_match_score'](predicted, g) for g in gold),
                                        max(env['f1_score'](predicted, g)[0] for g in gold),
                                        loose(predicted, gold, env['normalize_answer'])]
                    usage = ex.get('usage', {})
                    total_cost[i, j] = cost + 7 * (usage.get('prompt_tokens', 0) + usage.get('completion_tokens', 0))
                    extracted_available[i, j] = ex['answer'] is not None
        for db in [*ret.values(), *ans.values(), reader]:
            if db:
                db.close()
        print(json.dumps({'analyzed_shards': shard + 1}), flush=True)

    raw_complete = np.isfinite(scores[:, :, :3]).all(axis=(1, 2))
    complete = np.isfinite(scores).all(axis=(1, 2))
    report = {'total_questions': len(data), 'pairs': pairs, 'reader_root': str(reader_root),
              'raw_complete_questions': int(raw_complete.sum()),
              'complete_questions': int(complete.sum()), 'metric_order': METRICS,
              'metric_notes': 'Official normalized EM/F1. Lenient is a locally defined guarded bidirectional word containment, not certified identical to the missing upstream scorer. Reader is gold-independent; abstentions are incorrect.',
              'cost_notes': 'Parameter billions times prompt/output token counts, not measured FLOPs or wall time. QA prefix prompt counted once for shared KV. Extracted scores include the fixed 7B GGUF reader tokens.',
              'qualification': 'Retrospective gold-assisted oracle over stochastic prefix/fusion outcomes; not a deployable controller or an unbiased estimate of true optimal budgets.',
              'extraction_asserted_answer_rate': float(extracted_available[complete].mean()) if complete.any() else None}
    if not complete.all():
        (out / 'progress_metrics.json').write_text(json.dumps(report, indent=2))
        print(json.dumps(report), flush=True)
        return
    val, test = np.array(splits['val']), np.array(splits['test'])
    rng = np.random.default_rng(20261015)
    for metric_index in [0, 2, 3, 5]:
        metric = METRICS[metric_index]
        costs = generation_cost if metric_index < 3 else total_cost
        means = scores[val, :, metric_index].mean(axis=0)
        mean_costs = costs[val].mean(axis=0)
        best = min(range(configurations), key=lambda j: (-float(means[j]), mean_costs[j], j))
        fixed = scores[test, best, metric_index]
        oracle = scores[test, :, metric_index].max(axis=1)
        gain = oracle - fixed
        boot = np.array([gain[rng.integers(0, len(test), len(test))].mean() for _ in range(2000)])
        per_pair = {}
        for g, pair in enumerate(pairs):
            start = g * 40
            b = min(range(start, start + 40), key=lambda j: (-float(means[j]), mean_costs[j], j))
            matrix = scores[test, start:start + 40, metric_index]
            per_pair[pair] = {
                'fixed_k_selected_on_validation': b % 40 + 1,
                'fixed_test_score': float(scores[test, b, metric_index].mean()),
                'budget_oracle_test_score': float(matrix.max(axis=1).mean()),
                'k1_correct_k40_wrong': int(((matrix[:, 0] == 1) & (matrix[:, 39] == 0)).sum()),
                'k40_correct_k1_wrong': int(((matrix[:, 39] == 1) & (matrix[:, 0] == 0)).sum()),
                'small_1to5_success_but_all_6to40_fail': int(((matrix[:, :5].max(axis=1) == 1) & (matrix[:, 5:].max(axis=1) == 0)).sum()),
                'large_6to40_success_but_all_1to5_fail': int(((matrix[:, :5].max(axis=1) == 0) & (matrix[:, 5:].max(axis=1) == 1)).sum()),
            }
        section = {'fixed_pair': pairs[best // 40], 'fixed_k': best % 40 + 1,
                   'fixed_test_score': float(fixed.mean()), 'joint_oracle_test_score': float(oracle.mean()),
                   'fixed_test_f1': float(scores[test, best, 1 if metric_index < 3 else 4].mean()),
                   'gain': float(gain.mean()), 'paired_bootstrap_gain_ci95': np.quantile(boot, [.025, .975]).tolist(),
                   'per_pair': per_pair}
        frontier = []
        for lam in np.r_[0, np.logspace(-8, -3, 40)]:
            winner = np.argmax(scores[test, :, metric_index] - lam * costs[test], axis=1)
            ix = np.arange(len(test))
            frontier.append({'lambda': float(lam), 'score': float(scores[test, :, metric_index][ix, winner].mean()),
                             'cost': float(costs[test][ix, winner].mean())})
        fixed_cost = float(costs[test, best].mean())
        feasible = [f for f in frontier if f['cost'] <= fixed_cost] + [{'score': float(fixed.mean()), 'cost': fixed_cost, 'lambda': None}]
        section['evaluated_oracle_below_fixed_average_cost'] = max(feasible, key=lambda f: f['score'])
        section['cost_constraint_note'] = 'Best evaluated Lagrangian policy below the baseline average cost, not an exact constrained optimum.'
        section['cost_frontier'] = frontier
        section['grid'] = [{'pair': pairs[j // 40], 'k': j % 40 + 1, 'validation_score': float(means[j]),
                            'test_score': float(scores[test, j, metric_index].mean()),
                            'test_mean_cost': float(costs[test, j].mean())} for j in range(configurations)]
        report[metric] = section
    membership = {i: split for split, ids in splits.items() for i in ids}
    with (out / 'per_question_oracle.csv').open('w') as file:
        writer = csv.writer(file)
        writer.writerow(['index', 'id', 'split', 'oracle_pair', 'oracle_k', 'oracle_extracted_em', 'oracle_cost'])
        for i in range(len(data)):
            high = scores[i, :, 3].max()
            eligible = np.flatnonzero(scores[i, :, 3] == high)
            j = min(eligible, key=lambda j: (total_cost[i, j], j))
            writer.writerow([i, data[i]['_id'], membership[i], pairs[j // 40], j % 40 + 1, high, total_cost[i, j]])
    np.savez_compressed(out / 'scores.npz', scores=scores, generation_cost=generation_cost, total_cost=total_cost)
    (out / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k not in METRICS}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', default='analysis')
    parser.add_argument('--pairs', nargs='+', choices=PAIRS, default=PAIRS)
    parser.add_argument('--reader-root')
    args = parser.parse_args()
    main(args.root, args.output, args.pairs, args.reader_root)
