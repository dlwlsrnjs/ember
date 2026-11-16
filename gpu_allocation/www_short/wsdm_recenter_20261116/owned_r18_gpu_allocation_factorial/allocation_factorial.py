"""GPU factorial panel; distinct first fits from preserved CPU attempts for allocation and conditional pair selection."""
from pathlib import Path
from collections import Counter, defaultdict
from dataclasses import asdict
import argparse
import hashlib
import json
import os
import sys
import time
import traceback
import numpy as np
import torch

P = Path(__file__).resolve().parent
R = P.parent
sys.path.insert(0, str(R / 'owned_positive_controls'))
import positive_control_extension as base
sel = base.sel
ROOT = sel.ROOT
from ember2.dataset import Cache
from ember2.train import train, evaluate, warm_support
RUN = P / 'run_v1'
DEVICE = os.environ.get('ALLOCATION_DEVICE', 'cuda:0')
assert DEVICE.startswith('cuda:')
ARMS = ('NoOp', 'O_T', 'O_R', 'U_T', 'U_R')
CELLS = ('O_T', 'O_R', 'U_T', 'U_R')
JOBS = [(s, r, a) for r in range(3) for s in range(3) for a in ARMS]
VARIANTS = ('sampled_original', 'sampled_id_ties', 'eligible_fixed_targets', 'eligible_all_targets')
CONTRASTS = {
    'allocation_T': {'U_T': 1, 'O_T': -1},
    'allocation_R': {'U_R': 1, 'O_R': -1},
    'selector_O': {'O_T': 1, 'O_R': -1},
    'selector_U': {'U_T': 1, 'U_R': -1},
    'interaction': {'U_T': 1, 'O_T': -1, 'U_R': -1, 'O_R': 1},
    **{a + '_minus_NoOp': {a: 1, 'NoOp': -1} for a in CELLS},
    'policy_minus_NoOp': {'ValidationPolicy': 1, 'NoOp': -1},
}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
read = lambda path: json.loads(Path(path).read_text())


def put(path, value):
    with Path(path).open('x') as out:
        out.write(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def bucket(row):
    return row['d'], int(row['i'] if row['d'] == 'cold_item' else row['u'])


def priority(s, b):
    return hashlib.sha256(f'r17-allocation|{s}|{b[0]}|{b[1]}'.encode()).digest(), b


def quotas(s, old, capacity):
    result = {}
    for direction in ('cold_item', 'cold_user'):
        left = sum(v for b, v in old.items() if b[0] == direction)
        keys = sorted((b for b, v in capacity.items() if b[0] == direction and v), key=lambda b: priority(s, b))
        current = {b: 0 for b in keys}
        assert left <= sum(capacity[b] for b in keys)
        while left:
            changed = False
            for b in keys:
                if left and current[b] < capacity[b]:
                    current[b] += 1
                    left -= 1
                    changed = True
            assert changed
        result.update(current)
    return {b: n for b, n in result.items() if n}


def source_paths(s):
    critical = R / 'owned_r02_critical_controls/run_v1' / f'seed{s}'
    return {'O_T': critical / 'base_llm_cache.npz',
            'O_R': critical / 'base_qrandom_cache.npz',
            'NoOp': R / 'owned_positive_controls/run_v1' / f'seed{s}/none_cache.npz'}


def build_alternative(raw, s, original, full, retained, alternative, mode):
    pos_keys = {(row['u'], row['i']) for row in retained if row['b'][0] > 0}
    slots = defaultdict(list)
    for i, row in enumerate(retained):
        if row['b'][0] == 0:
            slots[bucket(row)].append(i)
    assignments, kept_by_bucket = {}, {}
    free = defaultdict(list)
    for b, positions in slots.items():
        n = min(len(positions), alternative.get(b, 0))
        kept_by_bucket[b] = positions[:n]
        for i in positions[:n]: assignments[i] = b
        free[b[0]].extend(positions[n:])
    for direction in ('cold_item', 'cold_user'):
        deficits = []
        for b in sorted(alternative, key=lambda b: priority(s, b)):
            if b[0] == direction:
                deficits.extend([b] * (alternative[b] - len(kept_by_bucket.get(b, []))))
        assert len(deficits) == len(free[direction])
        assignments.update(zip(sorted(free[direction]), deficits))
    assert Counter(assignments.values()) == alternative
    pool = defaultdict(list)
    for row in full:
        if ((mode == 'T' and row['b'][0] == 0) or
                (mode == 'R' and (row['u'], row['i']) not in pos_keys)):
            pool[bucket(row)].append((int(row['u']), int(row['i'])))
    rng = np.random.default_rng(s + 170771)
    pairs = [(int(row['u']), int(row['i'])) for row in retained]
    teacher_positive = {(row['u'], row['i']) for row in full if row['b'][0] > 0}
    for b in sorted(alternative, key=lambda b: priority(s, b)):
        keep = kept_by_bucket.get(b, [])
        used = {pairs[i] for i in keep}
        need = sorted(i for i, target in assignments.items() if target == b and i not in keep)
        candidates = sorted(set(pool[b]) - used)
        assert len(candidates) >= len(need), (s, mode, b, len(need))
        selected = rng.choice(len(candidates), len(need), replace=False)
        for i, j in zip(need, selected): pairs[i] = candidates[int(j)]
    aug = np.flatnonzero(original.train_is_aug > 0)
    neg_positions = np.array(sorted(assignments), dtype=np.int64)
    cache = base.clone_training(original)
    users = np.array([pairs[i][0] for i in neg_positions])
    items = np.array([pairs[i][1] for i in neg_positions])
    cache.train_feature_ids[aug[neg_positions]] = raw.compose_rows(
        users, items, original.train_feature_ids[aug[neg_positions], -raw.field_spec().n_ctx:])
    old_pos = original.train_binary[:, 0] > 0
    warm = original.train_is_aug <= 0
    assert np.array_equal(cache.train_feature_ids[old_pos | warm], original.train_feature_ids[old_pos | warm])
    for name in base.TRAIN_ARRAYS[1:]:
        assert np.array_equal(getattr(cache, name), getattr(original, name)), name
    assert np.array_equal(cache.train_feature_ids[:, -2:], original.train_feature_ids[:, -2:])
    assert len(set(pairs)) == len(pairs)
    assert all(pairs[i] not in pos_keys for i in neg_positions)
    assert all(bucket({'d': b[0], 'u': pairs[i][0], 'i': pairs[i][1]}) == b for i, b in assignments.items())
    if mode == 'T': assert all(pairs[i] not in teacher_positive for i in neg_positions)
    base.assert_same_evaluation(cache, original)
    return cache, {'minimum_slot_reassignments': sum(bucket(retained[i]) != b for i, b in assignments.items()),
        'negative_pair_changes': sum(pairs[i] != (retained[i]['u'], retained[i]['i']) for i in neg_positions),
        'teacher_positive_negative_slots': sum(pairs[i] in teacher_positive for i in neg_positions),
        'negative_buckets': len(alternative), 'unique_aug_pairs': len(set(pairs)),
        'all_positive_warm_targets_contexts_fixed': True,
        'same_direction_counts': True, 'capacity_feasible': True}


def prepare():
    assert not RUN.exists()
    RUN.mkdir()
    put(RUN / 'PREPARE_STARTED.json', {'created_unix': time.time(), 'new_outcomes_seen': False})
    sources, source_pins, audits = {}, {}, {}
    for s in range(3):
        raw = sel.raw_data(s)
        qpath = ROOT / f'ember2/runs/ar_fresh_ml1m_s{s}/dllm_rows.json'
        full = read(qpath)
        rowpaths = {a: R / f'selector_run_v1/seed{s}/{name}_rows.json' for a, name in [('O_T', 'llm'), ('O_R', 'random')]}
        rows = {a: read(path) for a, path in rowpaths.items()}
        for path in [qpath, *rowpaths.values()]: source_pins[str(path)] = sha(path)
        caches = {a: Cache.load(str(path)) for a, path in source_paths(s).items()}
        original = caches['O_T']
        for a, c in caches.items():
            base.assert_same_evaluation(c, original)
            source_pins[str(source_paths(s)[a])] = sha(source_paths(s)[a])
        for a in ('O_T', 'O_R'):
            aug = np.flatnonzero(caches[a].train_is_aug > 0)
            assert len(aug) == len(rows[a])
            reconstructed = raw.compose_rows(np.array([x['u'] for x in rows[a]]),
                np.array([x['i'] for x in rows[a]]), caches[a].train_feature_ids[aug, -2:])
            assert np.array_equal(reconstructed, caches[a].train_feature_ids[aug])
        for name in base.TRAIN_ARRAYS[1:]:
            assert np.array_equal(getattr(caches['O_T'], name), getattr(caches['O_R'], name))
        unchanged = ((caches['O_T'].train_is_aug <= 0) | (caches['O_T'].train_binary[:, 0] > 0))
        assert np.array_equal(caches['O_T'].train_feature_ids[unchanged], caches['O_R'].train_feature_ids[unchanged])
        assert np.array_equal(caches['O_T'].train_feature_ids[:, -2:], caches['O_R'].train_feature_ids[:, -2:])
        assert all(bucket(t) == bucket(r) for t, r in zip(rows['O_T'], rows['O_R']))
        old = Counter(bucket(row) for row in rows['O_T'] if row['b'][0] == 0)
        assert old == Counter(bucket(row) for row in rows['O_R'] if row['b'][0] == 0)
        cap = Counter(bucket(row) for row in full if row['b'][0] == 0)
        allocation = quotas(s, old, cap)
        audits[str(s)] = {'old_negative_buckets': len(old), 'new_negative_buckets': len(allocation),
            'B': original.n_aug, 'N': sum(old.values()), 'quota_policy': 'hashed-priority capped integer max-min',
            'positive_records': int(((original.train_is_aug > 0) & (original.train_binary[:, 0] > 0)).sum())}
        for a, mode in [('U_T', 'T'), ('U_R', 'R')]:
            caches[a], audits[str(s)][a] = build_alternative(raw, s, caches['O_' + mode], full, rows['O_' + mode], allocation, mode)
        dest = RUN / f'seed{s}'
        dest.mkdir()
        sources[str(s)] = {}
        for a, c in caches.items():
            path = dest / (a + '_cache.npz')
            c.save(str(path))
            sources[str(s)][a] = {'cache_path': str(path), 'cache_sha256': sha(path),
                'train_array_sha256': base.train_pins(c), 'validation_sha256': base.split_pins(c.val),
                'historical_test_sha256': base.split_pins(c.test), 'n_warm': c.n_warm, 'n_aug': c.n_aug}
        # Store quotas locally for exact reconstruction, not for external sharing.
        put(dest / 'QUOTAS.json', [{'direction': b[0], 'cold_id': b[1], 'O': old.get(b, 0), 'U': allocation.get(b, 0), 'teacherN_capacity': cap[b]}
                                    for b in sorted(set(old) | set(allocation))])
        del raw, caches, original
    local = [P / name for name in ('allocation_factorial.py', 'eligible_ranking.py', 'collect_factorial.py', 'PRE_EXECUTION_PROTOCOL_KO.txt')]
    local += [Path(base.__file__), Path(sel.__file__)]
    local += sorted((ROOT / 'ember2/ember2').glob('*.py'))
    local += [ROOT / f'data_kuairand/ml-1m/{name}.dat' for name in ('ratings', 'users', 'movies')]
    source_pins.update({str(f): sha(f) for f in local})
    put(RUN / 'INPUT_INVARIANTS.json', {'passed': True, 'settings': audits, 'new_fit_results_read': False})
    put(RUN / 'FROZEN_PROTOCOL.json', {'created_unix': time.time(), 'jobs': JOBS, 'arms': ARMS,
        'cfg': asdict(sel.config()), 'device': 'cuda', 'fit_count': 45,
        'initialization_rule': 'setting if repeat0 else17000+100*setting+repeat',
        'sources': sources, 'source_sha256': source_pins, 'contrasts': CONTRASTS,
        'evaluation_variants': VARIANTS, 'bootstrap_draws': 20000, 'bootstrap_rng': 2026111701,
        'scope': 'Post-hoc exploratory GPU factorial on consumed historical outcomes; not independent confirmation',
        'policy_selection': 'highest validation overall Recall20; tie NoOp,O_T,O_R,U_T,U_R',
        'automatic_retry': False, 'max_heavy_workers': 2})
    print(json.dumps({'prepared': True, 'first_fits': 45, 'device': 'cuda', 'protocol_sha256': sha(RUN / 'FROZEN_PROTOCOL.json'), 'input_invariants': True}))


def verify():
    protocol = read(RUN / 'FROZEN_PROTOCOL.json')
    assert protocol['jobs'] == [list(j) for j in JOBS]
    assert protocol['cfg'] == base.plain(asdict(sel.config()))
    for path, digest in protocol['source_sha256'].items():
        assert sha(path) == digest, ('Frozen source changed', path)
    return protocol


def fit(s, r, arm, label, protocol):
    spec = protocol['sources'][str(s)][arm]
    assert sha(spec['cache_path']) == spec['cache_sha256']
    cache = Cache.load(spec['cache_path'])
    assert base.train_pins(cache) == spec['train_array_sha256']
    assert base.split_pins(cache.val) == spec['validation_sha256']
    assert base.split_pins(cache.test) == spec['historical_test_sha256']
    historical, cache.test = cache.test, None
    init = s if r == 0 else 17000 + 100*s + r
    dest = RUN / 'fits' / f's{s}_r{r}_{arm}'
    dest.mkdir(parents=True)
    put(dest / 'INPUTS.json', {'setting': s, 'repeat': r, 'arm': arm, 'seed': init,
        'worker': label, 'device': DEVICE, 'started_unix': time.time(),
        'protocol_sha256': sha(RUN / 'FROZEN_PROTOCOL.json'), 'cache_sha256': spec['cache_sha256'],
        'torch_num_threads': torch.get_num_threads(), 'torch_num_interop_threads': torch.get_num_interop_threads()})
    started = time.monotonic()
    print(f'START s{s}/r{r}/{arm} GPU init{init}', flush=True)
    # train() only sees validation, never the historical test object.
    result = train(cache, sel.config(), device=DEVICE, seed=init, verbose=False, return_model=True)
    model = result.pop('model')
    chosen = max(result['history'], key=lambda x: x['val']['overall']['recall@20'])
    assert chosen['val']['overall']['recall@20'] == result['val']['ranking']['overall']['recall@20']
    su, si = warm_support(cache)
    torch.save({'state_dict': {n: q.detach().cpu() for n, q in model.state_dict().items()},
        'cfg': asdict(sel.config()), 'fields': asdict(cache.fields), 'tasks': asdict(cache.tasks),
        'core_mean': cache.core_mean, 'core_std': cache.core_std,
        'warm_support_user': su, 'warm_support_item': si,
        'setting': s, 'repeat': r, 'arm': arm, 'seed': init, 'cache_sha256': spec['cache_sha256'],
        'training_device': DEVICE, 'inference_tau': model.tau}, dest / 'selected.pt')
    selection = {'criterion': 'validation overall Recall@20; earliest maximum',
        'chosen_epoch': chosen['epoch'], 'val': result['val'],
        'checkpoint_sha256': sha(dest / 'selected.pt'), 'frozen_before_test_unix': time.time(),
        'test_used_in_fit_or_selection': False}
    put(dest / 'SELECTION.json', selection)
    put(dest / 'VALIDATION_ONLY.json', result)
    tested = evaluate(model, historical, cache, torch.device(DEVICE))
    per = sel.per_list_metrics(model, historical, DEVICE, dest / 'test_per_list.npz')
    for population in ('overall', 'cold', 'warm'):
        for metric in ('recall@20', 'ndcg@20'):
            assert abs(per[population][metric] - tested.ranking[population][metric]) < 1e-10
    result.update(historical_test=asdict(tested), selection=selection,
        fit_and_eval_wall_seconds=time.monotonic() - started,
        test_scope=protocol['scope'])
    put(dest / 'RESULT.json', result)
    print(f'DONE s{s}/r{r}/{arm} epoch{chosen["epoch"]} elapsed{time.monotonic()-started:.1f}s; test metrics retained for frozen collector', flush=True)


def worker(label):
    os.nice(19)
    protocol = verify()
    claims = RUN / 'claims'
    claims.mkdir(exist_ok=True)
    for s, r, a in JOBS:
        claim = claims / f's{s}_r{r}_{a}.json'
        try:
            put(claim, {'setting': s, 'repeat': r, 'arm': a, 'worker': label,
                        'pid': os.getpid(), 'claimed_unix': time.time()})
        except FileExistsError:
            continue
        try:
            fit(s, r, a, label, protocol)
        except BaseException as error:
            put(claim.with_name(claim.stem + '_FAILURE.json'), {'type': type(error).__name__,
                'error': str(error), 'traceback': traceback.format_exc(), 'automatic_retry': False})
            raise
    print('GPU PANEL WORKER COMPLETE ' + label, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'worker'))
    parser.add_argument('--label', default='gpu_worker')
    args = parser.parse_args()
    prepare() if args.stage == 'prepare' else worker(args.label)
