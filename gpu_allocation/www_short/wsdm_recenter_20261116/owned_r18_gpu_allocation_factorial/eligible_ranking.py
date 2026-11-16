"""Expand fixed historical case pools without fitting or model reselection."""
from pathlib import Path
from collections import defaultdict
import argparse
import os
import time
import traceback
import numpy as np
import torch
import allocation_factorial as panel
from ember2.dataset import Cache
from ember2.learner import EmberLearner
from ember2.rawdata import TEST
from ember2.metrics import recall_ndcg_at_k

P, R, RUN = panel.P, panel.R, panel.RUN
DEVICE = panel.DEVICE
sha, read, put = panel.sha, panel.read, panel.put


def prepare_features():
    protocol = panel.verify()
    root = RUN / 'eligible_inputs'
    root.mkdir()
    manifest = {'created_unix': time.time(), 'protocol_sha256': sha(RUN / 'FROZEN_PROTOCOL.json'),
                'selected_checkpoints_or_new_scores_read': False, 'settings': {}}
    for s in range(3):
        raw = panel.sel.raw_data(s)
        split = Cache.load(protocol['sources'][str(s)]['NoOp']['cache_path']).test
        seen, observed_positive = defaultdict(set), defaultdict(set)
        for j in np.flatnonzero(raw.period == TEST):
            u, i = int(raw.user[j]), int(raw.item[j])
            seen[u].add(i)
            if raw.binary[j, 0] > 0: observed_positive[u].add(i)
        starts = np.r_[0, np.flatnonzero(np.diff(split.group_id)) + 1]
        ends = np.r_[starts[1:], len(split)]
        cold = raw.is_cold_item
        pools = {False: np.flatnonzero(~cold), True: np.flatnonzero(cold)}
        groups = []
        offsets = [0]
        for lo, hi in zip(starts, ends):
            u = int(split.feature_ids[lo, 0] - 1)
            case = int(split.case[lo])
            pool = pools[case in (1, 3)]
            allpos = np.array(sorted(observed_positive[u]), np.int64)
            items = pool[(~np.isin(pool, sorted(seen[u]))) | np.isin(pool, allpos)]
            orig_items = split.feature_ids[lo:hi, raw.field_spec().n_user] - 1
            orig = np.searchsorted(items, orig_items)
            assert np.array_equal(items[orig], orig_items)
            fixed = orig[split.binary[lo:hi, 0] > 0]
            labels = np.isin(items, allpos)
            assert np.array_equal(labels[orig], split.binary[lo:hi, 0].astype(bool))
            groups.append((u, case, items, orig, fixed, labels, int(split.group_id[lo]), split.feature_ids[lo, -2:]))
            offsets.append(offsets[-1] + len(items))
        target = root / f'seed{s}'
        target.mkdir()
        shape = (offsets[-1], split.feature_ids.shape[1])
        feat = np.lib.format.open_memmap(target / 'features.npy', mode='w+', dtype=np.int32, shape=shape)
        all_labels = np.zeros(shape[0], bool)
        fixed_labels = np.zeros(shape[0], bool)
        original_lookup = []
        for g, (u, case, items, orig, fixed, labels, gid, context) in enumerate(groups):
            lo, hi = offsets[g:g+2]
            f = raw.compose_rows(np.full(len(items), u), items, np.tile(context, (len(items), 1)))
            assert f.min() >= 0 and f.max() < np.iinfo(np.int32).max
            feat[lo:hi] = f
            assert np.array_equal(f[orig], split.feature_ids[starts[g]:ends[g]])
            all_labels[lo:hi] = labels
            fixed_labels[lo + fixed] = True
            original_lookup.extend((lo + orig).tolist())
        feat.flush()
        del feat
        np.savez_compressed(target / 'metadata.npz', offsets=np.asarray(offsets), item_field=np.array(raw.field_spec().n_user),
            group_id=np.array([g[6] for g in groups]), user=np.array([g[0] for g in groups]),
            case=np.array([g[1] for g in groups]), cold=np.array([g[1] != 0 for g in groups]),
            all_labels=all_labels, fixed_labels=fixed_labels,
            original_lookup=np.asarray(original_lookup), original_starts=starts, original_ends=ends)
        record = {'feature_path': str(target / 'features.npy'), 'metadata_path': str(target / 'metadata.npz'),
            'features_sha256': sha(target / 'features.npy'), 'metadata_sha256': sha(target / 'metadata.npz'),
            'shape': shape, 'n_groups': len(groups), 'all_original_features_exactly_reproduced': True,
            'candidate_counts': {'minimum': int(np.diff(offsets).min()), 'median': float(np.median(np.diff(offsets))), 'maximum': int(np.diff(offsets).max())}}
        manifest['settings'][str(s)] = record
    put(root / 'FROZEN_FEATURES.json', manifest)
    print('Eligible case-pool features frozen; no checkpoint scores inspected.', flush=True)


@torch.inference_mode()
def predict(model, features):
    scores = np.empty(len(features), np.float32)
    for lo in range(0, len(features), 4096):
        ids = torch.as_tensor(np.array(features[lo:lo+4096], dtype=np.int64))
        scores[lo:lo+len(ids)] = model(ids.to(DEVICE))['binary_logits'][:, 0].cpu().numpy()
    assert np.isfinite(scores).all()
    return scores


def metric(scores, labels):
    recall, ndcg = recall_ndcg_at_k(scores, labels, 20)
    assert recall is not None and np.isfinite(ndcg)
    k = min(20, len(scores))
    order = np.argsort(-scores, kind='stable')
    threshold = scores[order[k-1]]
    tie = scores == threshold
    return recall, ndcg, len(scores), int(labels.sum()), int(tie.sum()), bool(labels[tie].any() and (~labels[tie]).any()), k/len(scores)


def per_variant(scores_full, scores_sampled, features, meta, split, variant):
    arrays = {k: [] for k in ('recall20', 'ndcg20', 'candidate_count', 'target_count',
                              'boundary_tie_size', 'boundary_tie_mixed_labels', 'random_expected_recall20')}
    for g, (lo, hi) in enumerate(zip(meta['offsets'][:-1], meta['offsets'][1:])):
        ol, oh = meta['original_starts'][g], meta['original_ends'][g]
        if variant.startswith('sampled'):
            scores, labels = scores_sampled[ol:oh], split.binary[ol:oh, 0].astype(bool)
            if variant == 'sampled_id_ties':
                order = np.argsort(split.feature_ids[ol:oh, int(meta['item_field'])], kind='stable')
                scores, labels = scores[order], labels[order]
        else:
            scores = scores_full[lo:hi]
            labels = meta['all_labels'][lo:hi] if variant == 'eligible_all_targets' else meta['fixed_labels'][lo:hi]
            if variant == 'eligible_fixed_targets':
                keep = (~meta['all_labels'][lo:hi]) | labels
                scores, labels = scores[keep], labels[keep]
        values = metric(scores, labels)
        for key, value in zip(arrays, values): arrays[key].append(value)
    result = {k: np.asarray(v) for k, v in arrays.items()}
    result.update({k: meta[k] for k in ('group_id', 'user', 'case', 'cold')})
    return result


def evaluate_checkpoint(s, repeat, arm, protocol, manifest):
    src = RUN / 'fits' / f's{s}_r{repeat}_{arm}'
    selection = read(src / 'SELECTION.json')
    assert sha(src / 'selected.pt') == selection['checkpoint_sha256']
    assert not selection['test_used_in_fit_or_selection']
    spec = manifest['settings'][str(s)]
    assert sha(spec['feature_path']) == spec['features_sha256']
    assert sha(spec['metadata_path']) == spec['metadata_sha256']
    cache = Cache.load(protocol['sources'][str(s)][arm]['cache_path'])
    state = torch.load(src / 'selected.pt', map_location='cpu', weights_only=False)
    model = EmberLearner(cache.fields, cache.tasks, panel.sel.config()).to(DEVICE)
    model.load_state_dict(state['state_dict'])
    model.set_support(state['warm_support_user'].to(DEVICE), state['warm_support_item'].to(DEVICE))
    model.set_tau(state['inference_tau'])
    model.eval()
    dest = src / 'eligible_eval'
    dest.mkdir()
    print(f'ELIGIBLE START s{s}/r{repeat}/{arm}', flush=True)
    started = time.monotonic()
    features = np.load(spec['feature_path'], mmap_mode='r')
    with np.load(spec['metadata_path']) as z: meta = {k: z[k] for k in z.files}
    full = predict(model, features)
    sampled = predict(model, cache.test.feature_ids)
    np.save(dest / 'scores_full.npy', full)
    np.save(dest / 'scores_sampled.npy', sampled)
    report = {'checkpoint_sha256': selection['checkpoint_sha256'], 'new_training': False,
        'checkpoint_reselection': False, 'variants': {}, 'original_GPU_replay': {}}
    old = dict(np.load(src / 'test_per_list.npz'))
    for variant in panel.VARIANTS:
        per = per_variant(full, sampled, features, meta, cache.test, variant)
        np.savez_compressed(dest / (variant + '_per_list.npz'), **per)
        report['variants'][variant] = {}
        masks = {'overall': np.ones(len(per['user']), bool), 'cold': per['cold'], 'warm': ~per['cold']}
        masks.update({name: per['case'] == code for code, name in enumerate(('warm', 'cold_item', 'cold_user', 'cold_both'))})
        for pop, mask in masks.items():
            report['variants'][variant][pop] = {k: float(per[k][mask].mean()) for k in
                ('recall20', 'ndcg20', 'candidate_count', 'target_count', 'random_expected_recall20')}
        if variant == 'sampled_original':
            for key in ('user', 'group_id', 'case', 'cold'): assert np.array_equal(old[key], per[key])
            for key in ('recall20', 'ndcg20'):
                report['original_GPU_replay'][key + '_max_abs_difference'] = float(np.max(np.abs(old[key] - per[key])))
    report.update(elapsed_seconds=time.monotonic()-started,
        candidate_inputs_sha256=spec, source_sha256=sha(__file__), inference_device=DEVICE)
    put(dest / 'RESULT.json', report)
    print(f'ELIGIBLE DONE s{s}/r{repeat}/{arm} elapsed{time.monotonic()-started:.1f}s; metrics retained for collector', flush=True)


def worker(label):
    os.nice(19)
    protocol = panel.verify()
    assert len(list((RUN / 'fits').glob('*/RESULT.json'))) == 45, 'Complete training panel first'
    assert not list((RUN / 'claims').glob('*FAILURE*'))
    manifest = read(RUN / 'eligible_inputs/FROZEN_FEATURES.json')
    assert manifest['protocol_sha256'] == sha(RUN / 'FROZEN_PROTOCOL.json')
    claims = RUN / 'eligible_claims'
    claims.mkdir(exist_ok=True)
    for s, r, a in panel.JOBS:
        path = claims / f's{s}_r{r}_{a}.json'
        try: put(path, {'setting': s, 'repeat': r, 'arm': a, 'label': label, 'pid': os.getpid(), 'claimed_unix': time.time()})
        except FileExistsError: continue
        try: evaluate_checkpoint(s, r, a, protocol, manifest)
        except BaseException as error:
            put(path.with_name(path.stem + '_FAILURE.json'), {'type': type(error).__name__,
                'error': str(error), 'traceback': traceback.format_exc(), 'automatic_retry': False})
            raise
    print('GPU ELIGIBLE WORKER COMPLETE ' + label, flush=True)


if __name__ == '__main__':
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('prepare', 'worker'))
    parser.add_argument('--label', default='eligible_gpu')
    args = parser.parse_args()
    prepare_features() if args.stage == 'prepare' else worker(args.label)
