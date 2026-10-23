"""Four independent GPU workers: weak-answer samples and raw-token confidence.

Retrieval and strong answering remain the frozen Instruct grid. Greedy confidence
is scored without using gold labels. Sample seeds are stable per question/branch.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / 'STAIR/nonLLM'))
from cascade import load
sys.path.insert(0, str(ROOT / 'STAIR/nonLLM/deployment'))
from cascade_four_gpu import core, write_json


def seed(key):
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], 'big') % 2147483647


def confidence(logits, targets):
    import torch
    if not len(targets): return [0.] * 6 + [1.]
    z = logits.float()
    lse = torch.logsumexp(z, -1)
    lp = z.gather(-1, targets[:, None]).squeeze(-1) - lse
    top = torch.topk(z, 2, dim=-1).values
    return [float(lp.mean()), float(lp.min()), float(lp.std(unbiased=False)),
            float(lp[0]), float(torch.exp(top[:, 0] - lse).mean()),
            float((top[:, 0] - top[:, 1]).mean()), 0.]


def main(a):
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    torch.set_num_threads(2)
    grid = load(a.base / 'labels/full_run_p100/router_data_lenient.jsonl')
    qa = {r['k']: r['v'] for r in load(a.base / 'labels/full_run_p100/qa_weak.jsonl')}
    tok = AutoTokenizer.from_pretrained(ROOT / 'models/Qwen2.5-1.5B-Instruct-HF', local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(ROOT / 'models/Qwen2.5-1.5B-Instruct-HF',
        dtype=torch.float16, device_map='cuda', attn_implementation='sdpa', local_files_only=True).eval()
    R = core()
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f'{a.phase}_{a.shard}.jsonl'
    done = {r['key'] for r in load(path)} if path.exists() else set()
    special = set(tok.all_special_ids)
    rows = [r for r in grid if r['idx'] % 4 == a.shard]
    sampling_splits=set(a.sampling_splits.split(','))
    if a.skip_greedy and a.phase=='trainval':
        rows=[r for r in rows if r['split'] in sampling_splits]
    if a.limit: rows = rows[:a.limit]
    start = time.monotonic(); calls = 0
    with path.open('a') as f:
        for g in rows:
            for branch, suffix in [('W', 'weak'), ('S', 'strong')]:
                user = R.qa_prompt({'question': g['question']}, g['evidence_' + branch])
                text = tok.apply_chat_template([{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': user}], tokenize=False, add_generation_prompt=True)
                ids = tok(text, return_tensors='pt', add_special_tokens=False).input_ids.to('cuda')
                if ids.shape[1] > 16384: raise ValueError('Input overflow')
                # Phase 1 scores greedy outputs for every split, samples train/val only.
                samples = ([-1] if a.phase == 'trainval' and not a.skip_greedy else [])
                if (a.phase == 'trainval' and g['split'] in sampling_splits) or (a.phase == 'test' and g['split'] == 'test'):
                    samples += list(range(a.sample_start,a.sample_end))
                for k in samples:
                    key = f"{g['idx']}|{branch}|{k}"
                    if key in done: continue
                    before = time.monotonic()
                    with torch.inference_mode():
                        if k == -1:
                            answer = qa[f"{g['idx']}|{suffix}"]
                            target = tok(answer, add_special_tokens=False, return_tensors='pt').input_ids.to('cuda')[0]
                            if len(target):
                                inp = torch.cat([ids, target[None]], -1)
                                z = model(input_ids=inp, logits_to_keep=len(target) + 1).logits[0, :-1]
                                keep = torch.tensor([i for i, t in enumerate(target.tolist()) if t not in special], device='cuda')
                                conf = confidence(z[keep], target[keep]) if len(keep) else [0.] * 6 + [1.]
                                del z, inp
                            else: conf = [0.] * 6 + [1.]
                            nout = len(target)
                        else:
                            torch.manual_seed(seed(key))
                            output = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=128, do_sample=True,
                                temperature=.7, top_p=.95, top_k=0, pad_token_id=tok.eos_token_id,
                                return_dict_in_generate=True, output_logits=True)
                            target = output.sequences[0, ids.shape[1]:]
                            answer = tok.decode(target, skip_special_tokens=True)
                            keep = [i for i, t in enumerate(target.tolist()) if t not in special]
                            conf = confidence(torch.stack([output.logits[i][0] for i in keep]), target[keep]) if keep else [0.] * 6 + [1.]
                            nout = len(target); del output
                    record = {'key': key, 'idx': g['idx'], 'split': g['split'], 'branch': branch,
                              'sample': k, 'text': answer, 'confidence': conf, 'n_in': ids.shape[1],
                              'n_out': nout, 'seed': seed(key) if k >= 0 else None,
                              'seconds': time.monotonic() - before}
                    f.write(json.dumps(record, ensure_ascii=False) + '\n'); f.flush()
                    done.add(key); calls += 1
                    if calls % 100 == 0:
                        print(f'{a.phase} shard={a.shard} completed={len(done)} seconds={time.monotonic()-start:.1f}', flush=True)
                        write_json(a.out / f'{a.phase}_{a.shard}.status.json', {'calls': len(done), 'phase': 'generation', 'last_key': key})
    write_json(a.out / f'{a.phase}_{a.shard}.status.json', {'calls': len(done), 'phase': 'complete'})


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--base', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True); p.add_argument('--shard', type=int, choices=range(4), required=True)
    p.add_argument('--phase', choices=['trainval', 'test'], required=True); p.add_argument('--limit', type=int, default=0)
    p.add_argument('--sample-start',type=int,default=0);p.add_argument('--sample-end',type=int,default=4)
    p.add_argument('--skip-greedy',action='store_true');p.add_argument('--sampling-splits',default='train,val')
    main(p.parse_args())
