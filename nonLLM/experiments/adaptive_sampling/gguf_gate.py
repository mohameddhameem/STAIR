"""Serving, long-context, source-default and train-only integration gates."""
import argparse
import asyncio
import gzip
import json
import time
from pathlib import Path

import aiohttp
from transformers import AutoTokenizer
from gguf_client import Client
from gguf_extract import extract
from protocol import load_protocol


async def main(root):
    root = Path(root)
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    train = json.loads((root / 'assets/splits.json').read_text())['splits']['train']
    prompt = load_protocol(root / 'upstream')
    tokens = {m: AutoTokenizer.from_pretrained(root / 'tokenizers' / m) for m in ['W', 'S']}
    clients = {m: Client(root, m, p, tokens[m]) for m, p in [('W', 30280), ('S', 30281)]}
    report = {'passed': False, 'training_indices': train[:2], 'controls': {}, 'pilot': []}

    def save():
        (root / 'gates/report.json').write_text(json.dumps(report, indent=2))

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1800), connector=aiohttp.TCPConnector(force_close=True)) as session:
        for client in clients.values():
            deadline = time.time() + 600
            while True:
                try:
                    async with session.get(client.endpoint + '/health') as response:
                        if response.status == 200:
                            break
                except aiohttp.ClientError:
                    pass
                if time.time() > deadline:
                    raise RuntimeError('GGUF service readiness timeout')
                await asyncio.sleep(2)
        for model, client in clients.items():
            controls = []
            for j, (question, expected) in enumerate([
                ('What is the capital of France? Answer with the city name only.', 'paris'),
                ('What is 2 + 2? Answer with the number only.', '4'),
                ('Reply with exactly the word READY and nothing else.', 'ready'),
            ]):
                messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
                            {'role': 'user', 'content': question}]
                r = await client.generate(session, f'gate:{model}:{j}', messages, 0, maximum=32)
                text = r['outputs'][0]['text']
                passed = text.strip(' .!\n\t').lower() == expected
                controls.append({'question': question, 'expected': expected, 'output': text, 'passed': passed})
                if not passed:
                    report['controls'][model] = controls
                    save()
                    raise RuntimeError(f'{model} failed short instruction quality gate')
            # Native n_cmpl shares the prompt KV and returns stable indexed RNG draws.
            r = await client.generate(session, f'gate:{model}:batch40', messages, .7, n=40, maximum=16)
            if len({o['seed'] for o in r['outputs']}) != 40:
                raise RuntimeError('Repeated candidate RNG seeds')
            report['controls'][model] = controls
            report['controls'][model + '_batch40'] = {'outputs': 40, 'unique_seeds': 40, 'seconds': r['seconds']}
            print('SHORT GATE PASSED', model, flush=True)
            save()
        checks = []
        for j, (question, response, expected) in enumerate([
            ('What is the capital of France?', 'The answer is Berlin.', 'Berlin'),
            ('What is 2 + 2?', 'The answer is 5.', '5'),
            ('What is the capital of France?', 'Berlin is the capital. Paris is not the answer.', 'Berlin'),
            ('What is 2 + 2?', 'The sum given in this response is five.', 'five'),
            ('Who wrote this book?', 'I cannot determine the author.', None),
            ('What is the capital of France?', 'What is the capital of France?', None),
            ('Are they of the same nationality?', 'No, they are not of the same nationality.', 'no'),
            ('Are J and K of the same nationality?', 'Both J and K are British.', 'yes'),
            ('Are J and K of the same nationality?', 'J is British, while K is German.', 'no'),
            ('Is 2 + 2 equal to 4?', 'Is 2 + 2 equal to 4?', None),
        ]):
            result = await extract(session, clients['S'], f'gate:extract:{j}', question, response)
            passed = result['answer'] == expected or (expected is not None and result['answer'] is None)
            checks.append({'expected': expected, 'result': result, 'passed': passed})
        report['extractor_checks'] = checks
        save()
        if not all(c['passed'] for c in checks) or sum(c['result']['answer'] is not None for c in checks) < 4:
            raise RuntimeError('Reader corrected a wrong assertion or invented an answer')
        # Public QA prompt and varied training articles, with a needle before the final 32K.
        pieces, seen, chars = [], set(), 0
        for i in train:
            for ref in data[i]['ctxs_candidate']:
                if ref not in seen:
                    pieces.append(ref.replace('[', '(').replace(']', ')'))
                    seen.add(ref)
                    chars += len(ref)
            if chars >= 240000:
                break
        example = {'question': 'What is the verification code assigned to project Bluebird?',
                   'ctxs_candidate': ['Project Bluebird: The verification code assigned to project Bluebird is K7P3. ' + ' '.join(pieces)]}
        messages = prompt('qa', example, retrieval='Query: control\nSelected references: [1]')
        for model, client in clients.items():
            result = await client.generate(session, f'gate:{model}:long', messages, 0, maximum=128)
            passed = result['input_tokens'] > 32768 and 'K7P3' in result['outputs'][0]['text']
            report['controls'][model + '_long'] = {'passed': passed, **result}
            save()
            if not passed:
                raise RuntimeError(f'{model} failed the long-context gate')
            print('LONG GATE PASSED', model, result['input_tokens'], flush=True)
        # Real retrieval and real QA candidates; no gold-evidence substitution.
        for i in train[:2]:
            retrieval = {}
            for model in ['W', 'S']:
                retrieval[model] = (await clients[model].generate(session, f'gate:{i}:ret:{model}', prompt('retrieval', data[i]), .9))['outputs'][0]['text']
            for pair in ['WW', 'WS', 'SW', 'SS']:
                rm, am = pair
                qa = await clients[am].generate(session, f'gate:{i}:qa:{pair}', prompt('qa', data[i], retrieval=retrieval[rm]), .7, n=40)
                answers = [r['text'] for r in qa['outputs']]
                for k in [40, 20, 10, 1]:
                    fused = await clients[am].generate(session, f'gate:{i}:fuse:{pair}:{k}', prompt('fusion', data[i], answers=answers[:k]), .3)
                    text = fused['outputs'][0]['text']
                    prediction = await extract(session, clients['S'], f'gate:{i}:extract:{pair}:{k}', data[i]['question'], text)
                    report['pilot'].append({'index': i, 'pair': pair, 'k': k, 'qa_seconds': qa['seconds'],
                                            'input_tokens': fused['input_tokens'], 'raw_text': text, 'extraction': prediction})
                    save()
                print('PILOT PASSED', i, pair, flush=True)
        if sum(r['extraction']['answer'] is not None for r in report['pilot']) < 8:
            raise RuntimeError('Too few asserted answers in integration pilot; inspect output quality')
        report['passed'] = True
        save()
        (root / 'gates/PASSED').write_text('GGUF controls and training integration passed\n')
        print('ALL GGUF GATES PASSED', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    asyncio.run(main(parser.parse_args().root))
