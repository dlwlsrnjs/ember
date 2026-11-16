"""Run a new matched panel on one or two GPUs; preserve every first attempt."""
from pathlib import Path
import argparse
import json
import os
import subprocess
import sys
import time
from verify_inputs import verify

ROOT = Path(__file__).resolve().parent
P = ROOT / 'www_short/wsdm_recenter_20261116/owned_r18_gpu_allocation_factorial'
WRAPPER = ROOT / 'www_short/two_sided_certificates/single_thread.py'


def command(program, stage, receipt, *extra):
    args = [sys.executable, str(WRAPPER), '--receipt', str(P / receipt), str(P / program)]
    if stage:
        args.append(stage)
    return [*args, *extra]


def gpu_available(devices):
    output = subprocess.check_output([
        'nvidia-smi', '--query-gpu=index,memory.free,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True)
    info = {int(row[0]): (int(row[1]), int(row[2]))
            for line in output.splitlines() if (row := line.split(','))}
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible:
        mapping = [int(value.strip()) for value in visible.split(',')]
        if max(devices) >= len(mapping):
            raise ValueError('Requested logical GPU index exceeds CUDA_VISIBLE_DEVICES.')
        devices = [mapping[d] for d in devices]
    missing = set(devices) - info.keys()
    if missing:
        raise ValueError(f'GPU indices not visible to nvidia-smi: {sorted(missing)}')
    return all(info[d][0] >= 8192 and info[d][1] <= 30 for d in devices)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu-a', type=int, default=0)
    parser.add_argument('--gpu-b', type=int, default=1,
                        help='Use 0 with --gpu-a 0 for two small workers on one GPU.')
    parser.add_argument('--wait-for-gpu', action='store_true')
    args = parser.parse_args()
    if min(args.gpu_a, args.gpu_b) < 0:
        parser.error('GPU indices must be nonnegative.')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').strip()
    if visible and not all(value.strip().isdigit() for value in visible.split(',')):
        parser.error('The availability check supports full GPUs and numeric CUDA_VISIBLE_DEVICES; UUID/MIG masks require an adapted scheduler check.')
    if not verify():
        raise SystemExit('Transfer the exact private input archive before launching.')
    # Claim before starting preparation or any worker; duplicate launches fail.
    with (P / 'LAUNCH_CLAIM.json').open('x') as out:
        json.dump({'created_unix': time.time(), 'pid': os.getpid(),
                   'gpus': [args.gpu_a, args.gpu_b], 'automatic_retry': False}, out)
    stable = 0
    while stable < 3:
        if gpu_available((args.gpu_a, args.gpu_b)):
            stable += 1
        else:
            stable = 0
            if not args.wait_for_gpu:
                raise SystemExit('Requested GPUs are busy. No model started; preserve this launch claim.')
            print('Waiting for requested GPUs: at least 8GiB free and utilization <=30%.', flush=True)
        if stable < 3:
            time.sleep(15)
    env = dict(os.environ, PYTHONINTMAXSTRDIGITS='0', ALLOCATION_DEVICE=f'cuda:{args.gpu_a}')
    for program, stage, receipt in [
        ('allocation_factorial.py', 'prepare', 'prepare_receipt01.json'),
        ('collect_factorial.py', 'pin', 'collector_pin_receipt01.json'),
        ('eligible_ranking.py', 'prepare', 'eligible_prepare_receipt01.json')]:
        subprocess.run(command(program, stage, receipt), cwd=ROOT, env=env, check=True)
    workers = []
    for name, device in [('a', args.gpu_a), ('b', args.gpu_b)]:
        log = (P / f'worker_gpu_{name}.log').open('x')
        worker_env = dict(env, ALLOCATION_DEVICE=f'cuda:{device}')
        worker = subprocess.Popen(
            command('allocation_factorial.py', 'worker', f'worker_gpu_{name}_receipt01.json',
                    '--label', f'external_gpu_{name}'), cwd=ROOT, env=worker_env,
            stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        workers.append(worker.pid)
    driver_env = dict(env, ALLOCATION_GPU_A=str(args.gpu_a), ALLOCATION_GPU_B=str(args.gpu_b))
    with (P / 'pipeline_driver.log').open('x') as log:
        driver = subprocess.Popen(command('finish_pipeline.py', '', 'pipeline_driver_receipt01.json'),
                                  cwd=ROOT, env=driver_env, stdout=log,
                                  stderr=subprocess.STDOUT, start_new_session=True)
    with (P / 'LAUNCHED.json').open('x') as out:
        json.dump({'created_unix': time.time(), 'worker_pids': workers,
                   'driver_pid': driver.pid, 'fit_count': 45}, out, indent=2)
    print('Launched 45 first fits, followed automatically by selected-checkpoint evaluation and collection.')
    print('Progress: python status.py; numerical results: run_v1/COLLECTED.json after completion.')


if __name__ == '__main__':
    main()
