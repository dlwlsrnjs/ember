"""Pinned collection of the complete GPU panel; no outcome-driven model choice."""
from pathlib import Path
import argparse
import csv
import math
import time
import numpy as np
from scipy.stats import t
import allocation_factorial as run

P, R, RUN = run.P, run.R, run.RUN
sha, read, put = run.sha, run.read, run.put
POPS = ('overall', 'cold', 'warm', 'cold_item', 'cold_user', 'cold_both')
METRICS = ('recall20', 'ndcg20')
OUTCOMES = [(pop, m) for pop in POPS for m in METRICS]
CORE = [('cold', 'recall20'), ('cold', 'ndcg20'), ('warm', 'recall20')]


def masks(per):
    result = {'overall': np.ones(len(per['user']), bool), 'cold': per['cold'], 'warm': ~per['cold']}
    result.update({name: per['case'] == code for code, name in enumerate(('warm', 'cold_item', 'cold_user', 'cold_both'))})
    return result


def load_per(path):
    with np.load(path) as z: return {k: z[k] for k in z.files}


def aligned(a, b):
    for name in ('user', 'group_id', 'cold', 'case'):
        assert np.array_equal(a[name], b[name]), name


def ratio(sums, counts, repeat_weights, user_weights):
    # Same user resample across the selected paired initializations and arms.
    combined = np.einsum('nr,cru->ncu', repeat_weights, sums, optimize=True)
    numerator = np.einsum('ncu,nu->nc', combined, user_weights, optimize=True)
    return numerator / (user_weights @ counts)[:, None]


def sanity():
    # Unequal list counts must be weighted once, not averaged as user means.
    counts = np.array([1., 3.])
    sums = np.array([[[2., 12.], [4., 24.], [6., 36.]]])
    rw = np.array([[1., 0., 0.], [1/3, 1/3, 1/3], [0., 0., 1.]])
    uw = np.array([[1, 1], [1, 1], [2, 0]])
    actual = ratio(sums, counts, rw, uw)[:, 0]
    assert np.allclose(actual, [3.5, 7., 6.], atol=1e-12)
    # Factorial interaction is a contrast of four arms, not additive attribution.
    cells = {'O_T': 1., 'O_R': 2., 'U_T': 4., 'U_R': 8.}
    contrast = sum(weight*cells[arm] for arm, weight in run.CONTRASTS['interaction'].items())
    assert contrast == (4-1)-(8-2) == -3
    return {'unequal_list_count_ratio': actual.tolist(), 'factorial_interaction_identity': True}


def pin():
    protocol = run.verify()
    result = sanity()
    assert not list((RUN / 'fits').glob('*/RESULT.json'))
    put(P / 'COLLECTOR_PIN.json', {'created_unix': time.time(), 'source_sha256': sha(__file__),
        'protocol_sha256': sha(RUN / 'FROZEN_PROTOCOL.json'), 'sanity': result,
        'real_new_outcomes_inspected': False, 'contrasts': run.CONTRASTS,
        'outcomes': OUTCOMES, 'core_bootstrap_outcomes': CORE,
        'draws': protocol['bootstrap_draws'], 'rng': protocol['bootstrap_rng'],
        'selection': protocol['policy_selection']})
    print('Factorial collector pinned before first new outcome; weighting sanity passed.', flush=True)


def checked(s, r, a, protocol):
    dest = RUN / 'fits' / f's{s}_r{r}_{a}'
    inp, selection, result = [read(dest / name) for name in ('INPUTS.json', 'SELECTION.json', 'RESULT.json')]
    assert inp['device'].startswith('cuda:')
    assert inp['seed'] == (s if r == 0 else 17000+100*s+r)
    assert inp['protocol_sha256'] == sha(RUN / 'FROZEN_PROTOCOL.json')
    assert inp['cache_sha256'] == protocol['sources'][str(s)][a]['cache_sha256']
    assert inp['torch_num_threads'] == inp['torch_num_interop_threads'] == 1
    assert not selection['test_used_in_fit_or_selection']
    assert selection['checkpoint_sha256'] == sha(dest / 'selected.pt')
    best = max(result['history'], key=lambda h: h['val']['overall']['recall@20'])
    assert best['epoch'] == selection['chosen_epoch']
    assert best['val']['overall']['recall@20'] == selection['val']['ranking']['overall']['recall@20']
    eval_result = read(dest / 'eligible_eval/RESULT.json')
    assert eval_result['checkpoint_sha256'] == selection['checkpoint_sha256']
    assert not eval_result['checkpoint_reselection'] and not eval_result['new_training']
    pins = {name: sha(dest/name) for name in ('INPUTS.json', 'SELECTION.json', 'RESULT.json', 'selected.pt', 'test_per_list.npz', 'eligible_eval/RESULT.json')}
    return result, eval_result, pins


def collect():
    protocol = run.verify()
    pin_record = read(P / 'COLLECTOR_PIN.json')
    assert pin_record['source_sha256'] == sha(__file__)
    assert pin_record['protocol_sha256'] == sha(RUN / 'FROZEN_PROTOCOL.json')
    assert not (RUN / 'COLLECTED.json').exists()
    assert len(list((RUN / 'fits').glob('*/RESULT.json'))) == 45
    assert len(list((RUN / 'fits').glob('*/eligible_eval/RESULT.json'))) == 45
    assert not list((RUN / 'claims').glob('*FAILURE*')) and not list((RUN / 'eligible_claims').glob('*FAILURE*'))
    results, eval_results, fit_pins, selections = {}, {}, {}, {}
    for s in range(3):
        for r in range(3):
            scores = []
            for a in run.ARMS:
                result, evaluated, pins = checked(s, r, a, protocol)
                results[s, r, a], eval_results[s, r, a], fit_pins[f's{s}/r{r}/{a}'] = result, evaluated, pins
                scores.append(result['val']['ranking']['overall']['recall@20'])
            selected = run.ARMS[int(np.argmax(scores))]
            selections[s, r] = selected
    put(RUN / 'VALIDATION_POLICY.json', {'created_unix': time.time(),
        'rule_prespecified_before_fits': protocol['policy_selection'], 'test_used': False,
        'choices': {f's{s}/r{r}': a for (s,r),a in selections.items()},
        'selection_scores': {f's{s}/r{r}': {a: results[s,r,a]['val']['ranking']['overall']['recall@20'] for a in run.ARMS} for s in range(3) for r in range(3)},
        'note': 'Choices computed from validation only; rule pinned before test. Historical test inference already existed; not a prospectively new cohort.'})
    output = {'created_unix': time.time(), 'all45_first_fits_complete': True,
        'training_device': 'cuda', 'scope': protocol['scope'], 'protocol_sha256': sha(RUN/'FROZEN_PROTOCOL.json'),
        'collector_sha256': sha(__file__), 'collector_pin_sha256': sha(P/'COLLECTOR_PIN.json'),
        'fit_pins': fit_pins, 'policy_selection': {f's{s}/r{r}': a for (s,r),a in selections.items()},
        'bootstrap': {'draws': 20000, 'rng': 2026111701, 'levels': 'setting -> paired learning initialization -> paired user cluster', 'unadjusted': True},
        'variants': {}, 'GPU_original_replay': {f's{s}/r{r}/{a}': eval_results[s,r,a]['original_GPU_replay'] for s,r,a in run.JOBS}}
    contrast_names = tuple(run.CONTRASTS)
    for v, variant in enumerate(run.VARIANTS):
        per = {}
        for s, r, a in run.JOBS:
            per[s,r,a] = load_per(RUN/'fits'/f's{s}_r{r}_{a}'/'eligible_eval'/(variant+'_per_list.npz'))
            aligned(per[s,r,a], per.get((s,0,'NoOp'), per[s,r,a]))
        for s in range(3):
            for r in range(3): per[s,r,'ValidationPolicy'] = per[s,r,selections[s,r]]
        C, O = len(contrast_names), len(OUTCOMES)
        effects = np.empty((3,3,C,O))
        summaries = {}
        for a in (*run.ARMS, 'ValidationPolicy'):
            summaries[a] = {}
            for o, (pop, metric) in enumerate(OUTCOMES):
                values = np.array([[per[s,r,a][metric][masks(per[s,r,a])[pop]].mean() for r in range(3)] for s in range(3)])
                summaries[a][pop+'/'+metric] = {'mean': float(values.mean()), 'setting_means': values.mean(axis=1).tolist(),
                    'setting_mean_sd': float(values.mean(axis=1).std(ddof=1)), 'scores_setting_by_repeat': values.tolist()}
        aggregate = {}
        for s in range(3):
            reference = per[s,0,'NoOp']
            aggregate[s] = {}
            for o, (pop, metric) in enumerate(OUTCOMES):
                mask = masks(reference)[pop]
                uids, inverse = np.unique(reference['user'][mask], return_inverse=True)
                counts = np.bincount(inverse).astype(float)
                sums = np.empty((C,3,len(uids)))
                for c, name in enumerate(contrast_names):
                    for r in range(3):
                        delta = sum(weight*per[s,r,a][metric][mask] for a, weight in run.CONTRASTS[name].items())
                        effects[s,r,c,o] = delta.mean()
                        sums[c,r] = np.bincount(inverse, weights=delta, minlength=len(uids))
                if (pop,metric) in CORE: aggregate[s][CORE.index((pop,metric))] = sums, counts
        B = protocol['bootstrap_draws']
        rng = np.random.default_rng(protocol['bootstrap_rng'] + 100*v)
        boot = np.zeros((B,C,len(CORE)))
        for lo in range(0,B,128):
            size = min(128,B-lo)
            sampled_settings = rng.integers(0,3,(size,3))
            for position in range(3):
                for s in range(3):
                    ids = np.flatnonzero(sampled_settings[:,position] == s)
                    if not len(ids): continue
                    rw = rng.multinomial(3,[1/3]*3,size=len(ids))/3
                    for o in range(len(CORE)):
                        sums, counts = aggregate[s][o]
                        n = len(counts)
                        uw = rng.multinomial(n,np.full(n,1/n),size=len(ids))
                        boot[lo+ids,:,o] += ratio(sums,counts,rw,uw)/3
            if lo % 4096 == 0: print(f'COLLECT {variant}: bootstrap{lo}/{B}', flush=True)
        paired = {}
        for c, name in enumerate(contrast_names):
            for o,(pop,metric) in enumerate(OUTCOMES):
                setting = effects[:,:,c,o].mean(axis=1)
                half = float(t.ppf(.975,2)*setting.std(ddof=1)/math.sqrt(3))
                value = {'mean': float(setting.mean()), 'setting_means': setting.tolist(),
                    'paired_effects_setting_by_repeat': effects[:,:,c,o].tolist(),
                    'within_setting_sd': effects[:,:,c,o].std(axis=1,ddof=1).tolist(),
                    'positive_learning_pairs': int((effects[:,:,c,o]>0).sum()),
                    'exploratory_setting_t2_95': [float(setting.mean()-half),float(setting.mean()+half)]}
                if (pop,metric) in CORE:
                    value['exploratory_hierarchical_bootstrap_95'] = np.quantile(boot[:,c,CORE.index((pop,metric))],[.025,.975]).tolist()
                paired[name+'/'+pop+'/'+metric] = value
        reference = per[0,0,'NoOp']
        composition = {str(s): {pop: {name: float(per[s,0,'NoOp'][name][masks(per[s,0,'NoOp'])[pop]].mean()) for name in
            ('candidate_count','target_count','random_expected_recall20')} for pop in POPS} for s in range(3)}
        output['variants'][variant] = {'summary': summaries, 'paired': paired, 'composition': composition,
            'bootstrap_rng': protocol['bootstrap_rng'] + 100*v}
    # New panel inventory remains separate from the historical 129-run inventory.
    rows = []
    for s,r,a in run.JOBS:
        result = results[s,r,a]
        rows.append({'setting': s, 'repeat': r, 'seed': s if r==0 else 17000+100*s+r, 'arm': a,
            'device': 'cuda', 'selected_epoch': result['selection']['chosen_epoch'],
            'validation_overall_R20': result['val']['ranking']['overall']['recall@20'],
            'sampled_cold_R20': result['historical_test']['ranking']['cold']['recall@20'],
            'sampled_warm_R20': result['historical_test']['ranking']['warm']['recall@20'],
            'result_sha256': fit_pins[f's{s}/r{r}/{a}']['RESULT.json'],
            'checkpoint_sha256': fit_pins[f's{s}/r{r}/{a}']['selected.pt']})
    with (RUN/'ALL45_GPU_FIRST_FITS.csv').open('x') as f:
        writer = csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    put(RUN/'COLLECTED.json',output)
    print('ALL45 GPU panel,4rankingvariants,allpredeclaredcontrasts collected.',flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('pin','collect','sanity'))
    args = parser.parse_args()
    if args.stage == 'pin': pin()
    elif args.stage == 'collect': collect()
    else: print(sanity())
