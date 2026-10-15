"""Disjoint logical-shard subsets for a reduced number of healthy GPUs."""
import argparse
import gzip
import json
from pathlib import Path
import subprocess
from gguf_runtime import PYTHON


def main(a):
    root = Path(a.root)
    n = len(json.load(gzip.open(root / 'assets/AgentTTS-HotpotQA.json.gz', 'rt')))
    for shard in range(4):
        ids = [i for i in range(n) if i % 4 == shard and (i // 4) % a.ranks == a.rank]
        assignment = root / 'gates' / f'assignment_{a.ranks}_{a.rank}_{shard}.json'
        assignment.write_text(json.dumps(ids))
        print(json.dumps({'physical_rank': a.rank, 'logical_shard': shard, 'questions': len(ids), 'stage': a.stage}), flush=True)
        subprocess.run([PYTHON, str(root / 'code/gguf_worker.py'), '--root', str(root),
                        '--stage', a.stage, '--model', a.model, '--port', str(a.port),
                        '--shard', str(shard), '--indices-file', str(assignment)], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--port', type=int, required=True)
    parser.add_argument('--rank', type=int, required=True)
    parser.add_argument('--ranks', type=int, required=True)
    main(parser.parse_args())
