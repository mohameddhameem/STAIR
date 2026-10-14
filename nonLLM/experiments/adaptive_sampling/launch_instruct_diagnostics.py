"""Launch two isolated long-context diagnostic services on SMU_A.

Does not start dataset inference. Check device availability before invoking.
Transformers 5 uses rope_parameters; a rope_scaling-only override is ignored
by SGLang's get_rope_config when rope_parameters is present.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path('/data1/zmcheng/cs707_sampling40')
PYTHON = '/home/zmcheng/miniconda3/envs/sglang518/bin/python'


def main(selected, weak_device, strong_device):
    for device, size, port, memory in [(weak_device, '1.5B', 30186, '.50'),
                                       (strong_device, '7B', 30187, '.65')]:
        role = 'W' if size == '1.5B' else 'S'
        if selected and role != selected:
            continue
        used = int(subprocess.check_output([
            'nvidia-smi', '--id', str(device), '--query-gpu=memory.used',
            '--format=csv,noheader,nounits'], text=True).strip())
        if used > 1024:
            raise RuntimeError(f'GPU {device} still has {used} MiB allocated; do not overlap launches')
        override = {'max_position_embeddings': 131072, 'rope_parameters': {
            'rope_type': 'yarn', 'rope_theta': 1000000.0,
            'factor': 4.0, 'original_max_position_embeddings': 32768,
        }}
        command = [PYTHON, '-m', 'sglang.launch_server', '--model-path',
                   str(ROOT / f'diagnostics/models/Qwen2.5-{size}-Instruct-HF'),
                   '--context-length', '131072', '--json-model-override-args',
                   json.dumps(override), '--mem-fraction-static', memory,
                   '--max-total-tokens', '150000', '--max-running-requests', '8',
                   '--disable-cuda-graph', '--disable-piecewise-cuda-graph',
                   '--enable-custom-logit-processor',
                   '--host', '127.0.0.1', '--port', str(port)]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), OMP_NUM_THREADS='8',
                   CUDA_HOME='/usr/local/cuda-13.0',
                   SGLANG_CACHE_DIR=str(ROOT / 'runtime_cache'))
        env['PATH'] = str(Path(PYTHON).parent) + ':/usr/local/cuda-13.0/bin:' + env['PATH']
        env['PYTHONPATH'] = str(ROOT / 'code') + os.pathsep + env.get('PYTHONPATH', '')
        log = ROOT / f'diagnostics/logs/sglang_instruct{size}_yarn_v5_{time.time_ns()}.log'
        with log.open('w') as output:
            process = subprocess.Popen(command, env=env, stdout=output,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps({'pid': process.pid, 'device': device, 'port': port,
                          'command': command, 'log': str(log)}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['W', 'S'])
    parser.add_argument('--weak-device', type=int, default=1)
    parser.add_argument('--strong-device', type=int, default=2)
    args = parser.parse_args()
    main(args.model, args.weak_device, args.strong_device)
