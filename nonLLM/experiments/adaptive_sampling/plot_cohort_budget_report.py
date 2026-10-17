"""Render the frozen 500-question lenient-EM results as an English figure."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np


def main():
    repo = Path(__file__).resolve().parents[3]
    folder = repo / 'nonLLM/results/adaptive_sampling40_instruct_gguf/cohort_500/lenient_ranges'
    summary = json.loads((folder / 'summary.json').read_text())
    values = summary['metrics']['lenient_em']
    pairs = ['WW', 'WS', 'SW', 'SS']
    labels = ['1', '2–5', '6–10', '11–20', '21–40']
    colors = ['#176B70', '#49A6A4', '#8CCBC1', '#EDC76C', '#D98352', '#DEE3E8']
    ink, muted = '#172C40', '#526579'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
                         'text.color': ink, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': ink,
                         'axes.edgecolor': '#CBD4DD', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(18, 16), facecolor='#F6F8FB')
    fig.text(.055, .959, 'How many samples does each question need?', size=29, weight='bold')
    fig.text(.055, .931, 'HotpotQA  •  500 matched training questions  •  Four weak/strong combinations  •  Lenient EM',
             size=15, color=muted)

    # Protocol band: every prefix reuses the same forty generated candidates.
    band = fig.add_axes([.045, .798, .91, .112]); band.axis('off')
    band.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle='round,pad=0.008,rounding_size=0.03',
                                 facecolor='#E7EEF4', edgecolor='none'))
    blocks = [(.018, '01  RETRIEVE', 'Once with W and once with S\nW = Qwen2.5-1.5B Instruct; S = 7B Instruct'),
              (.357, '02  SAMPLE + FUSE', '40 answer samples for each model pair\nFuse the first k samples for every k = 1…40'),
              (.718, '03  SCORE', 'Fixed, gold-blind answer extraction\nCompare each result with gold answers')]
    for x, title, body in blocks:
        band.text(x, .75, title, weight='bold', size=12, color='#176B70')
        band.text(x, .26, body, size=10.5, linespacing=1.65)
    fig.text(.055, .781, 'GGUF Q4_K_M on serverA  •  AgentTTS public HotpotQA prompts  •  Temperature: retrieval 0.9 / answer 0.7 / fusion 0.3  •  Max output 512 tokens',
             size=10, color=muted)
    fig.text(.055, .766, 'Pair code: retrieval model → answer/fusion model. For example, WS = weak retrieval + strong answer/fusion.',
             size=10, color=muted)

    # A: complete, mutually exclusive earliest-success counts.
    a = fig.add_axes([.09, .546, .405, .176], facecolor='none')
    fig.text(.055, .749, 'A   Earliest successful sampling budget', size=17, weight='bold')
    fig.text(.055, .728, 'Question counts; each row totals 500.', size=11, color=muted)
    left = np.zeros(4)
    for j, label in enumerate(labels + ['Never correct']):
        counts = np.array([values[p]['first_success_bins'][label.replace('–', '-')] if j < 5 else values[p]['never_correct'] for p in pairs])
        a.barh(np.arange(4), counts, left=left, height=.65, color=colors[j], label=label)
        for y, c, start in zip(range(4), counts, left):
            a.text(start + c / 2, y, str(c), ha='center', va='center', size=7 if c < 15 else 10,
                   color='white' if j < 2 else ink, weight='bold')
        left += counts
    a.set(yticks=np.arange(4), yticklabels=pairs, xlim=(0, 500), xticks=np.arange(0, 501, 100))
    a.invert_yaxis(); a.set_xlabel('Questions', fontsize=11)
    a.spines[['top', 'right', 'left']].set_visible(False); a.tick_params(axis='y', length=0)
    a.legend(ncol=3, bbox_to_anchor=(-.08, -.21), loc='upper left', frameon=False,
             fontsize=10, handlelength=1.1, columnspacing=1.4)

    # B: same-cohort fixed-k peak versus retrospective per-question peak.
    b = fig.add_axes([.60, .546, .34, .176], facecolor='none')
    fig.text(.56, .749, 'B   Fixed versus per-question budgets', size=17, weight='bold')
    fig.text(.56, .728, 'Both selected retrospectively on the same 500 questions.', size=11, color=muted)
    for offset, name, color in [(-.18, 'Best shared k', '#8BA1B5'), (.18, 'Per-question oracle', '#176B70')]:
        data = [100 * values[p]['best_shared_hits' if offset < 0 else 'ever_correct'] / 500 for p in pairs]
        b.barh(np.arange(4) + offset, data, height=.31, color=color, label=name)
        for y, v, p in zip(np.arange(4) + offset, data, pairs):
            suffix = '  k=' + '/'.join(map(str, values[p]['best_shared_ks'])) if offset < 0 else ''
            b.text(v + 1, y, f'{v:.1f}%{suffix}', va='center', size=10)
    b.set(yticks=np.arange(4), yticklabels=pairs, xlim=(0, 88), xticks=[0, 20, 40, 60, 80])
    b.invert_yaxis(); b.set_xlabel('Lenient EM (%)', fontsize=11)
    b.spines[['top', 'right', 'left']].set_visible(False); b.tick_params(axis='y', length=0)
    b.legend(ncol=2, bbox_to_anchor=(-.08, -.21), loc='upper left', frameon=False, fontsize=10)

    # C: joint min/max distributions; zero cells remain visible and annotated.
    fig.text(.055, .451, 'C   Where do correct sampling budgets begin and end?', size=17, weight='bold')
    fig.text(.055, .429, 'Each heatmap counts questions by their minimum and maximum successful k. Only questions correct at least once are included.',
             size=11, color=muted)
    def bucket(k):
        return next(j for j, (lo, hi) in enumerate([(1, 1), (2, 5), (6, 10), (11, 20), (21, 40)]) if lo <= k <= hi)
    matrices = []
    for p in pairs:
        matrix = np.zeros((5, 5), dtype=int)
        for row in values[p]['min_max_ranges']:
            matrix[bucket(row['min_k']), bucket(row['max_k'])] += row['questions']
        assert matrix.sum() == values[p]['ever_correct']
        matrices.append(matrix)
    maximum = max(m.max() for m in matrices)
    for j, (p, matrix) in enumerate(zip(pairs, matrices)):
        ax = fig.add_axes([.095 + .222*j, .235, .178, .166], facecolor='none')
        ax.imshow(matrix, cmap='Blues', vmin=0, vmax=maximum, aspect='auto')
        for y in range(5):
            for x in range(5):
                if x >= y:
                    v = matrix[y, x]
                    ax.text(x, y, str(v) if v else '·', ha='center', va='center', size=11,
                            color='white' if v > maximum * .58 else ink)
        ax.set(xticks=range(5), xticklabels=labels, yticks=range(5), yticklabels=labels)
        ax.tick_params(length=0, labelsize=9)
        ax.set_title(f'{p}  |  n = {matrix.sum()}', fontsize=13, weight='bold', pad=9)
        ax.set_xlabel('Maximum successful k', fontsize=10)
        if j == 0: ax.set_ylabel('Minimum successful k', fontsize=10)
        for spine in ax.spines.values(): spine.set_visible(False)
    fig.text(.095, .198, 'Correctness can have gaps within these bounds:  WW 228/283  •  WS 264/342  •  SW 263/313  •  SS 294/346 questions have non-contiguous success sets.',
             size=11, color=muted)

    # Real observed example; exact success set rather than a fictional interval.
    fig.text(.055, .164, 'Example   One SS question succeeds only at k = 10, 13, 24, 31, 34', size=14, weight='bold')
    ex = fig.add_axes([.095, .116, .82, .031]); ex.axis('off')
    correct = {10, 13, 24, 31, 34}
    with (folder / 'per_question.csv').open() as handle:
        example = next(row for row in csv.DictReader(handle)
                       if row['metric'] == 'lenient_em' and row['index'] == '24' and row['pair'] == 'SS')
    assert set(json.loads(example['correct_ks'])) == correct
    for k in range(1, 41):
        ex.add_patch(plt.Rectangle((k-1, .15), .86, .65,
                                  facecolor='#176B70' if k in correct else '#DEE3E8'))
        if k in correct or k in {1, 20, 40}:
            ex.text(k-.57, -.16, str(k), ha='center', va='top', size=9)
    ex.set_xlim(0, 40); ex.set_ylim(0, 1)
    fig.text(.055, .081, 'Reading the result: extra samples sometimes rescue an answer, but can also change a correct fusion into an incorrect one.',
             size=12, weight='bold', color='#176B70')
    fig.text(.055, .052, 'Scope: availability-based training cohort, not a held-out benchmark or an AgentTTS reproduction score. Oracle uses gold answers and is not a deployable or cost-matched policy.',
             size=9, color=muted)
    fig.text(.055, .037, 'Lenient EM: normalized bidirectional whole-word containment with inequality guards, applied to extracted answers. Local scoring rule; may accept partial matches and differ from the original scorer.',
             size=9, color=muted)
    fig.text(.055, .022, 'Retrieval is sampled once per model here. Sampling budgets k vary only for answer generation and fusion. Frozen cohort results completed 17 Oct 2026.',
             size=9, color=muted)
    for suffix in ['png', 'pdf']:
        path = folder / f'budget_report_en.{suffix}'
        fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
        print(path)
    plt.close(fig)


if __name__ == '__main__':
    main()
