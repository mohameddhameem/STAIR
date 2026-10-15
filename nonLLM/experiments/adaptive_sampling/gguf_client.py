"""Native llama.cpp client with frozen public-protocol sampling parameters."""
import hashlib
import asyncio
import aiohttp
import json
import math
import time
from pathlib import Path


def seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') % 2147483000


class Client:
    def __init__(self, root, model, port, tokenizer):
        self.root = Path(root)
        self.model = model
        self.endpoint = f'http://127.0.0.1:{port}'
        self.tokenizer = tokenizer
        self.profile = json.loads((self.root / 'profile.json').read_text())
        self.defaults = json.loads((self.root / 'tokenizers' / model / 'generation_config.json').read_text())

    def encode(self, messages):
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.tokenizer.encode(text, add_special_tokens=False)

    async def generate(self, session, key, messages, temperature, n=1, maximum=512, schema=None):
        ids = self.encode(messages)
        if len(ids) + maximum > self.profile['context_limit']:
            raise RuntimeError(f'Context overflow, no truncation allowed: {key}, {len(ids)}')
        if n > self.profile['slots']:
            raise RuntimeError('Not enough native parallel slots for the requested candidate batch')
        body = {
            'prompt': ids, 'temperature': temperature, 'n_predict': maximum,
            'top_k': self.defaults.get('top_k', 50), 'top_p': self.defaults.get('top_p', 1),
            'min_p': 0, 'typical_p': 1, 'repeat_penalty': self.defaults.get('repetition_penalty', 1),
            'repeat_last_n': self.profile['context_limit'], 'presence_penalty': 0,
            'frequency_penalty': 0, 'dry_multiplier': 0, 'mirostat': 0,
            # HF applies penalties, temperature, top-k, then nucleus filtering.
            'samplers': ['penalties', 'temperature', 'top_k', 'top_p'],
            'n_cmpl': n, 'seed': seed(key), 'cache_prompt': True, 'stream': False,
            'stop': ['<|im_end|>', '<|endoftext|>'], 'return_tokens': False,
        }
        if schema is not None:
            body.update(json_schema=schema, temperature=0, top_k=50, top_p=1, repeat_penalty=1)
        start = time.time()
        for attempt in range(3):
            try:
                async with session.post(self.endpoint + '/completion', json=body) as response:
                    raw = await response.json()
                    if response.status != 200:
                        raise RuntimeError(f'{key}: HTTP {response.status}: {str(raw)[:800]}')
                break
            except (aiohttp.ClientError, asyncio.TimeoutError):
                if attempt == 2:
                    raise
                await asyncio.sleep(.5 * (attempt + 1))
        rows = raw if isinstance(raw, list) else [raw]
        if len(rows) != n:
            raise RuntimeError(f'{key}: expected {n} candidates, got {len(rows)}')
        # The native reader returns results in task-index order, not finish order.
        if [r['index'] for r in rows] != list(range(n)):
            raise RuntimeError(f'Unexpected candidate ordering for {key}')
        outputs = []
        for j, row in enumerate(rows):
            if row.get('truncated') or row['tokens_evaluated'] != len(ids):
                raise RuntimeError(f'Truncated or modified input for {key}: {row["tokens_evaluated"]}/{len(ids)}')
            settings = row.get('generation_settings', {})
            for name in ['temperature', 'top_k', 'top_p', 'repeat_penalty', 'repeat_last_n']:
                if name in settings and not math.isclose(settings[name], body[name], rel_tol=1e-6, abs_tol=1e-6):
                    raise RuntimeError(f'Server ignored {name}: {key}')
            if settings.get('samplers') != body['samplers']:
                raise RuntimeError(f'Sampling order changed: {key}: {settings.get("samplers")}')
            outputs.append({
                'text': row['content'], 'sample_index': j,
                'seed': settings.get('seed'),
                'meta_info': {'completion_tokens': row['tokens_predicted'], 'prompt_tokens': len(ids)},
                'llama': {k: row.get(k) for k in ['id_slot', 'stop_type', 'stopping_word', 'truncated', 'tokens_cached', 'timings']},
            })
        return {'outputs': outputs, 'input_tokens': len(ids), 'truncated_tokens': 0,
                'seconds': time.time() - start, 'request_seed': seed(key),
                'sampling': {k: v for k, v in body.items() if k not in ['prompt', 'json_schema']}}
