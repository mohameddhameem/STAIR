"""Read-only scoring of a frozen cohort's answer-budget success sets."""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import sqlite3
import statistics

from gguf_analyze import loose, official


def intervals(values):
    runs = []
    for k in values:
        if runs and k == runs[-1][1] + 1:
            runs[-1][1] = k
        else:
            runs.append([k, k])
    return runs


def bins(values):
    return {label: sum(lo <= k <= hi for k in values)
            for label, lo, hi in [('1', 1, 1), ('2-5', 2, 5), ('6-10', 6, 10),
                                  ('11-20', 11, 20), ('21-40', 21, 40)]}


def describe(sets, n):
    successful = [ks for ks in sets if ks]
    first = [ks[0] for ks in successful]
    last = [ks[-1] for ks in successful]
    hits = [sum(k in ks for ks in sets) for k in range(1, 41)]
    best = max(hits)
    return {
        'questions': n, 'ever_correct': len(successful), 'never_correct': n - len(successful),
        'first_success_histogram': dict(sorted(Counter(first).items())),
        'last_success_histogram': dict(sorted(Counter(last).items())),
        'first_success_bins': bins(first), 'last_success_bins': bins(last),
        'best_shared_ks': [k + 1 for k, count in enumerate(hits) if count == best],
        'best_shared_hits': best, 'hits_by_k': hits,
        'median_first_success': statistics.median(first) if first else None,
        'median_last_success': statistics.median(last) if last else None,
        'median_correct_budget_count': statistics.median(map(len, successful)) if successful else None,
        'non_contiguous_success_sets': sum(len(intervals(ks)) > 1 for ks in successful),
        'only_one_correct_budget': sum(len(ks) == 1 for ks in successful),
        'k1_correct_k40_wrong': sum(1 in ks and 40 not in ks for ks in sets),
        'k40_correct_k1_wrong': sum(40 in ks and 1 not in ks for ks in sets),
        'min_max_ranges': [{'min_k': lo, 'max_k': hi, 'questions': count}
                           for (lo, hi), count in sorted(Counter(zip(first, last)).items())],
    }


def main(root, cohort, output):
    config = json.loads((cohort / 'config.json').read_text())
    ids = config['indices']
    assert len(ids) == len(set(ids)) == 500
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    metric = official(root)
    pairs = ['WW', 'WS', 'SW', 'SS']
    all_sets = {name: {pair: {} for pair in pairs} for name in ['strict_em', 'lenient_em']}
    dbs = {shard: sqlite3.connect(
        f'file:{cohort}/evaluation_v2/runs/extraction_S_{shard}.sqlite?mode=ro', uri=True)
        for shard in {i % 4 for i in ids}}
    try:
        for i in ids:
            for pair in pairs:
                success = {name: [] for name in all_sets}
                for k in range(1, 41):
                    key = f'{i}:fuse:{pair}:{k}'
                    row = dbs[i % 4].execute('SELECT payload FROM results WHERE key=?', (key,)).fetchone()
                    assert row is not None, key
                    answer = json.loads(row[0])['extraction']['answer'] or ''
                    gold = data[i]['answers']
                    strict = max(metric['exact_match_score'](answer, g) for g in gold)
                    lenient = loose(answer, gold, metric['normalize_answer'])
                    assert not strict or lenient, key
                    if strict:
                        success['strict_em'].append(k)
                    if lenient:
                        success['lenient_em'].append(k)
                for name in all_sets:
                    all_sets[name][pair][i] = success[name]
    finally:
        for db in dbs.values():
            db.close()
    output.mkdir(parents=True, exist_ok=True)
    summary = {
        'questions': len(ids),
        'metric_notes': 'Scores use the same gold-independent v2 extracted answers. Lenient EM reuses gguf_analyze.loose: official normalization, guarded bidirectional word-boundary containment. Locally defined; not certified identical to the missing upstream scorer.',
        'qualification': 'Frozen availability-based training cohort, not held-out accuracy. Success sets and ranges use gold retrospectively. Min/max bounds do not imply every intervening budget succeeds. Each k uses the first k of the same forty candidates and its own fusion.',
        'metrics': {},
    }
    rows = []
    for name, by_pair in all_sets.items():
        values = {}
        for pair, by_id in by_pair.items():
            values[pair] = describe(list(by_id.values()), len(ids))
            for i, ks in by_id.items():
                rows.append([name, i, data[i]['question'], pair, ks[0] if ks else '',
                             ks[-1] if ks else '', len(ks), json.dumps(ks), json.dumps(intervals(ks))])
        union = [sorted(set().union(*(by_pair[pair][i] for pair in pairs))) for i in ids]
        values['any_pair'] = describe(union, len(ids))
        summary['metrics'][name] = values
    # The strict calculation must reproduce the previously published cohort.
    old = json.loads((cohort / 'summary.json').read_text())
    for pair in pairs:
        new = summary['metrics']['strict_em'][pair]
        assert new['ever_correct'] == old['pairs'][pair]['ever_correct_questions']
        assert new['best_shared_hits'] == old['pairs'][pair]['best_fixed_hits']
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    with (output / 'per_question.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'index', 'question', 'pair', 'min_correct_k', 'max_correct_k',
                         'correct_budget_count', 'correct_ks', 'contiguous_correct_intervals'])
        writer.writerows(rows)
    with (output / 'histograms.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['metric', 'pair', 'k', 'first_success_questions', 'last_success_questions', 'correct_questions'])
        for name, values in summary['metrics'].items():
            for pair, value in values.items():
                for k in range(1, 41):
                    writer.writerow([name, pair, k, value['first_success_histogram'].get(k, 0),
                                     value['last_success_histogram'].get(k, 0), value['hits_by_k'][k - 1]])
    lines = ['# Lenient EM budget distribution', '', summary['metric_notes'], '', summary['qualification'], '',
             'W = 1.5B, S = 7B. First letter retrieval; second answer/fusion.', '',
             '| Pair | First 1 | First 2–5 | First 6–10 | First 11–20 | First 21–40 | Never correct |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for pair, value in summary['metrics']['lenient_em'].items():
        lines.append('| ' + ' | '.join(map(str, [pair, *value['first_success_bins'].values(), value['never_correct']])) + ' |')
    lines += ['', '`any_pair` permits switching both model pair and budget after seeing gold.', '',
              '| Pair | Last 1 | Last 2–5 | Last 6–10 | Last 11–20 | Last 21–40 |',
              '|---|---:|---:|---:|---:|---:|']
    for pair, value in summary['metrics']['lenient_em'].items():
        lines.append('| ' + ' | '.join(map(str, [pair, *value['last_success_bins'].values()])) + ' |')
    lines += ['',
              '| Pair | Best shared k | Shared hits | Any-k hits | Median [min, max] among successful questions | Non-contiguous sets |',
              '|---|---|---:|---:|---|---:|']
    for pair in pairs:
        v = summary['metrics']['lenient_em'][pair]
        lines.append(f'| {pair} | {v["best_shared_ks"]} | {v["best_shared_hits"]} | {v["ever_correct"]} | [{v["median_first_success"]}, {v["median_last_success"]}] | {v["non_contiguous_success_sets"]} |')
    lines += ['', 'Median min and max are separate marginal medians, not the success interval of a representative question. Per-question exact sets and contiguous intervals are in per_question.csv; full 1–40 histograms are in histograms.csv.']
    (output / 'table.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(summary['metrics']['lenient_em'], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--cohort', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    main(args.root, args.cohort, args.output)
