"""One-shot phase orchestration; never retry a fit or inspect numerical outcomes."""
from pathlib import Path
import json
import os
import subprocess
import sys
import time
import hashlib

P = Path(__file__).resolve().parent
R = P.parent
ROOT = R.parents[1]
RUN = P / 'run_v1'
WRAPPER = ROOT / 'www_short/two_sided_certificates/single_thread.py'


def put(path, value):
    with path.open('x') as out: out.write(json.dumps(value, indent=2) + '\n')


def phase_command(program, stage, receipt, *extra):
    return [sys.executable, str(WRAPPER), '--receipt', str(receipt), str(P/program), stage, *extra]


def main():
    put(P/'PIPELINE_STARTED.json', {'created_unix': time.time(), 'pid': os.getpid(),
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope': 'wait for existing fixed45 fits; eligible evaluation; pinned collector',
        'automatic_retry': False})
    deadline = time.time() + 14*3600
    while True:
        if list((RUN/'claims').glob('*FAILURE*')):
            raise RuntimeError('A first-fit failure exists; preserve and stop, no retry')
        done = len(list((RUN/'fits').glob('*/RESULT.json')))
        receipts = [P/f'worker_gpu_{name}_receipt01.json' for name in ('a','b')]
        finished = all(path.exists() and json.loads(path.read_text()).get('completed') for path in receipts)
        if done == 45 and finished: break
        if time.time() > deadline: raise TimeoutError('Fixed panel not complete within orchestration window')
        time.sleep(15)
    put(P/'TRAINING_PHASE_COMPLETE.json', {'created_unix': time.time(), 'complete_first_fits': 45, 'new_metrics_read_by_driver': False})
    children, streams = [], []
    try:
        for name in ('a','b'):
            stream = (P/f'eligible_worker_gpu_{name}.log').open('x')
            streams.append(stream)
            command = phase_command('eligible_ranking.py','worker',P/f'eligible_worker_gpu_{name}_receipt01.json','--label','r18_eligible_'+name)
            env = dict(os.environ, ALLOCATION_DEVICE='cuda:' + os.environ.get('ALLOCATION_GPU_' + name.upper(), '0' if name == 'a' else '1'))
            children.append(subprocess.Popen(command,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT,env=env))
        exits = [child.wait() for child in children]
        if exits != [0,0]: raise RuntimeError('Eligible evaluator failed; no automatic retry: '+str(exits))
    finally:
        for stream in streams: stream.close()
    assert len(list((RUN/'fits').glob('*/eligible_eval/RESULT.json'))) == 45
    assert not list((RUN/'eligible_claims').glob('*FAILURE*'))
    put(P/'ELIGIBLE_PHASE_COMPLETE.json', {'created_unix': time.time(), 'checkpoints': 45, 'variants': 4, 'retraining': False})
    with (P/'collector.log').open('x') as stream:
        result = subprocess.run(phase_command('collect_factorial.py','collect',P/'collector_receipt01.json'),
            cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode: raise RuntimeError('Pinned collector failed; no automatic retry')
    assert (RUN/'COLLECTED.json').exists()
    put(P/'PIPELINE_COMPLETE.json', {'created_unix': time.time(), 'new_gpu_first_fits': 45,
        'selected_checkpoints_evaluated': 45, 'ranking_variants': 4, 'new_teacher_queries': 0,
        'automatic_retry': False, 'outcomes_collected': True, 'paper_not_yet_finalized': True})
    print('FIXED R18 GPU EXPERIMENT PIPELINE COMPLETE', flush=True)


if __name__ == '__main__':
    try: main()
    except BaseException as error:
        path=P/'PIPELINE_FAILURE.json'
        if not path.exists(): put(path,{'created_unix':time.time(),'type':type(error).__name__,'error':str(error),'automatic_retry':False})
        raise
