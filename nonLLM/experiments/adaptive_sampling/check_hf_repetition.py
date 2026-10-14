"""CPU-only conformance checks against the installed Transformers processor."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers.generation.logits_process import RepetitionPenaltyLogitsProcessor
from hf_repetition import HFRepetitionPenalty


def main(output):
    checks = []
    for penalty in [1.0, 1.05, 1.1, 2.0]:
        for prompt, generated in [([0, 1, 1, 3], []),
                                  ([0, 1, 1, 3], [1, 2, 2]),
                                  ([4], [0, 4]), ([], [])]:
            scores = torch.tensor([[-3., 0., 2., 6., -9., 8.]])
            expected = RepetitionPenaltyLogitsProcessor(penalty)(
                torch.tensor([prompt + generated], dtype=torch.long), scores.clone())
            request = SimpleNamespace(origin_input_ids=prompt, output_ids=generated)
            actual = HFRepetitionPenalty()(scores.clone(), [{
                'hf_repetition_penalty': penalty, '__req__': request}])
            checks.append({'penalty': penalty, 'prompt': prompt, 'output': generated,
                           'passed': torch.equal(expected, actual)})
    report = {'passed': all(c['passed'] for c in checks), 'checks': checks,
              'reference': 'installed Transformers RepetitionPenaltyLogitsProcessor'}
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    if not report['passed']:
        raise SystemExit('HF repetition penalty conformance failed')
    print(f'All {len(checks)} repetition-penalty checks passed')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    main(parser.parse_args().output)
