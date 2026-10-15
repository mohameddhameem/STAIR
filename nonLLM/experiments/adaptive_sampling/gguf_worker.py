"""ServerA-only resumable retrieval, 40-candidate fusion, and answer reading."""
import argparse
import asyncio
import gzip
import hashlib
import json
from pathlib import Path
import sqlite3
import time

import aiohttp
from transformers import AutoTokenizer
from gguf_client import Client
from gguf_extract import extract
from protocol import load_protocol


class Admission:
    """Bound active KV tokens and slots, including native shared-prefix forks."""
    def __init__(self, tokens, slots):
        self.maximum = tokens
        self.tokens = tokens
        self.slots = slots
        self.condition = asyncio.Condition()

    async def acquire(self, tokens, slots):
        if tokens > self.maximum:
            raise RuntimeError('A single request exceeds the admitted KV working set')
        async with self.condition:
            await self.condition.wait_for(lambda: self.tokens >= tokens and self.slots >= slots)
            self.tokens -= tokens
            self.slots -= slots

    async def release(self, tokens, slots):
        async with self.condition:
            self.tokens += tokens
            self.slots += slots
            self.condition.notify_all()


def database(path, fingerprint):
    db = sqlite3.connect(path, timeout=60)
    db.execute('PRAGMA journal_mode=WAL')
    db.execute('CREATE TABLE IF NOT EXISTS results (key TEXT PRIMARY KEY,payload TEXT NOT NULL)')
    db.execute('CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT NOT NULL)')
    old = db.execute("SELECT value FROM metadata WHERE key='profile'").fetchone()
    if old and old[0] != fingerprint:
        raise RuntimeError(f'Cache profile changed: {path}; use a separate run root')
    if not old and db.execute('SELECT COUNT(*) FROM results').fetchone()[0]:
        raise RuntimeError(f'Refusing an unversioned cache: {path}')
    db.execute('INSERT OR REPLACE INTO metadata VALUES (?,?)', ('profile', fingerprint))
    db.commit()
    return db


async def main(a):
    root = Path(a.root)
    profile_text = (root / 'profile.json').read_text()
    profile = json.loads(profile_text)
    # Bind inference/extraction code as well as model and serving configuration.
    h = hashlib.sha256(profile_text.encode())
    for name in ['protocol.py', 'gguf_client.py', 'gguf_extract.py', 'answer_parser.py', 'grounded_answer.py']:
        h.update((root / 'code' / name).read_bytes())
    fingerprint = h.hexdigest()
    db = database(root / 'runs' / f'{a.stage}_{a.model}_{a.shard}.sqlite', fingerprint)
    data = json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt'))
    tok = AutoTokenizer.from_pretrained(root / 'tokenizers' / a.model)
    client = Client(root, a.model, a.port, tok)
    prompt = load_protocol(root / 'upstream')
    admission = Admission(int(profile['kv_pool_tokens'] * .75), profile['slots'])
    # Scheduling is separate from the frozen model/prompt/sampling profile.
    schedule_path = root / 'runtime_schedule.json'
    schedule = json.loads(schedule_path.read_text()) if schedule_path.exists() else {}
    settings = schedule.get('models', {}).get(a.model, {})
    fusion_limit = int(settings.get('fusion_lanes', 2))
    concurrency = int(settings.get('question_lanes', 2))
    if not (1 <= fusion_limit <= profile['slots'] and 1 <= concurrency <= profile['slots']):
        raise RuntimeError('Invalid runtime scheduling limits')
    # Long fusions replicate reference-pool prefill. More simultaneous requests
    # can reduce prefix reuse and throughput even when GPU utilization rises.
    fusion_lanes = asyncio.Semaphore(fusion_limit)
    print(json.dumps({'runtime_schedule': {'fusion_lanes': fusion_limit,
                                          'question_lanes': concurrency},
                      'stage': a.stage, 'model': a.model, 'shard': a.shard}), flush=True)

    def get(key):
        row = db.execute('SELECT payload FROM results WHERE key=?', (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def save(key, value):
        db.execute('INSERT OR REPLACE INTO results VALUES (?,?)', (key, json.dumps(value)))
        db.commit()

    def retrieval(i, model):
        path = root / 'runs' / f'retrieval_{model}_{a.shard}.sqlite'
        source = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
        row = source.execute('SELECT payload FROM results WHERE key=?', (f'{i}:ret:{model}',)).fetchone()
        source.close()
        if not row:
            raise RuntimeError(f'Missing retrieval {i}/{model}')
        return json.loads(row[0])['outputs'][0]['text']

    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600), connector=aiohttp.TCPConnector(force_close=True)) as session:
        async def generate(key, messages, temperature, n=1):
            cached = get(key)
            if cached is not None:
                return cached
            # llama.cpp n_cmpl copies sequence IDs over the existing prompt KV.
            is_fusion = ':fuse:' in key
            if is_fusion:
                await fusion_lanes.acquire()
            reservation = len(client.encode(messages)) + n * 512 + 256
            await admission.acquire(reservation, n)
            try:
                for attempt in range(3):
                    try:
                        result = await client.generate(session, key, messages, temperature, n=n)
                        save(key, result)
                        return result
                    except (aiohttp.ClientError, asyncio.TimeoutError):
                        if attempt == 2:
                            raise
                        await asyncio.sleep(2 * (attempt + 1))
            finally:
                await admission.release(reservation, n)
                if is_fusion:
                    fusion_lanes.release()

        async def question(i):
            start = time.time()
            if a.stage == 'retrieval':
                await generate(f'{i}:ret:{a.model}', prompt('retrieval', data[i]), .9)
            else:
                for rm in ['W', 'S']:
                    pair = rm + a.model
                    qa = await generate(f'{i}:qa:{pair}', prompt('qa', data[i], retrieval=retrieval(i, rm)), .7, n=40)
                    answers = [o['text'] for o in qa['outputs']]
                    # Bound longest fusion first; subsequent prefixes reuse its prompt cache.
                    await generate(f'{i}:fuse:{pair}:40', prompt('fusion', data[i], answers=answers), .3)
                    await asyncio.gather(*(generate(f'{i}:fuse:{pair}:{k}', prompt('fusion', data[i], answers=answers[:k]), .3) for k in range(39, 0, -1)))
            print(json.dumps({'stage': a.stage, 'model': a.model, 'shard': a.shard,
                              'index': i, 'seconds': time.time() - start,
                              'saved_requests': db.execute('SELECT COUNT(*) FROM results').fetchone()[0]}), flush=True)

        if a.stage == 'extraction':
            queue = asyncio.Queue(maxsize=64)
            allowed_reader_indices = set(json.loads(Path(a.indices_file).read_text())) if a.indices_file else None

            async def reader():
                while True:
                    item = await queue.get()
                    if item is None:
                        queue.task_done()
                        return
                    key, value = item
                    i = int(key.split(':')[0])
                    text = value['outputs'][0]['text']
                    source_sha = hashlib.sha256(text.encode()).hexdigest()
                    cached = get(key)
                    if cached is None:
                        result = await extract(session, client, key + ':reader', data[i]['question'], text)
                        save(key, {'source_sha256': source_sha, 'extraction': result})
                    elif cached['source_sha256'] != source_sha:
                        raise RuntimeError('Extraction source changed')
                    queue.task_done()

            async with asyncio.TaskGroup() as group:
                tasks = [group.create_task(reader()) for _ in range(32)]
                for am in ['W', 'S']:
                    path = root / 'runs' / f'answers_{am}_{a.shard}.sqlite'
                    source = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
                    for key, text in source.execute("SELECT key,payload FROM results WHERE key LIKE '%:fuse:%'"):
                        if allowed_reader_indices is not None and int(key.split(':')[0]) not in allowed_reader_indices:
                            continue
                        await queue.put((key, json.loads(text)))
                    source.close()
                for _ in tasks:
                    await queue.put(None)
        else:
            indices = [i for i in range(len(data)) if i % 4 == a.shard]
            if a.indices_file:
                allowed = set(json.loads(Path(a.indices_file).read_text()))
                indices = [i for i in indices if i in allowed]
            if a.limit:
                indices = indices[:a.limit]
            # Independent long retrieval prompts do not share KV. Large batches
            # multiply attention over the unified pool; use two question lanes.
            for offset in range(0, len(indices), concurrency):
                await asyncio.gather(*(question(i) for i in indices[offset:offset + concurrency]))
    marker = 'pilot.done' if a.limit or a.indices_file else 'done'
    (root / 'runs' / f'{a.stage}_{a.model}_{a.shard}.{marker}').write_text('complete\n')
    db.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', choices=['retrieval', 'answers', 'extraction'], required=True)
    parser.add_argument('--model', choices=['W', 'S'], required=True)
    parser.add_argument('--shard', type=int, required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--indices-file')
    args = parser.parse_args()
    if args.stage == 'extraction' and args.model != 'S':
        parser.error('The frozen answer reader always uses S')
    asyncio.run(main(args))
