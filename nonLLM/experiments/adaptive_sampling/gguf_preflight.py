"""CPU-only context bounds across all questions, without scoring gold answers."""
import argparse
import gzip
import json
from pathlib import Path
import sqlite3

from transformers import AutoTokenizer
from protocol import load_protocol


def main(root, retrievals=False):
    root = Path(root)
    profile = json.loads((root / 'profile.json').read_text())
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    tokenizer = AutoTokenizer.from_pretrained(root / 'tokenizers/S')
    prompt = load_protocol(root / 'upstream')
    maximum = (0, None)
    if not retrievals:
        for offset in range(0, len(data), 32):
            examples = data[offset:offset + 32]
            messages = [prompt('fusion', e, answers=[''] * 40) for e in examples]
            texts = [tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
            lengths = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)['input_ids']]
            for j, count in enumerate(lengths):
                # Include 40 maximum outputs, token-boundary slack, and fusion output.
                bound = count + 40 * (512 + 2) + 512
                if bound > maximum[0]:
                    maximum = (bound, offset + j)
            if offset % 512 == 0:
                print(json.dumps({'checked': offset + len(examples), 'upper_bound': maximum[0]}), flush=True)
    else:
        for model in ['W', 'S']:
            for shard in range(4):
                db = sqlite3.connect(f'file:{root / "runs" / f"retrieval_{model}_{shard}.sqlite"}?mode=ro', uri=True)
                for key, payload in db.execute('SELECT key,payload FROM results'):
                    i = int(key.split(':')[0])
                    text = json.loads(payload)['outputs'][0]['text']
                    messages = prompt('qa', data[i], retrieval=text)
                    count = len(tokenizer.encode(tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True), add_special_tokens=False)) + 512
                    if count > maximum[0]:
                        maximum = (count, key)
                db.close()
    report = {'passed': maximum[0] <= profile['context_limit'], 'questions': len(data),
              'scope': 'actual retrieval QA inputs' if retrievals else 'conservative full-pool 40-candidate fusion bound',
              'max_tokens_including_output': maximum[0], 'maximum_at': maximum[1],
              'context_limit': profile['context_limit'], 'gold_answers_used': False}
    name = 'qa_context' if retrievals else 'all_context'
    (root / 'gates' / (name + '.json')).write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(f'Context preflight failed: {report}')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--retrievals', action='store_true')
    args = parser.parse_args()
    main(args.root, args.retrievals)
