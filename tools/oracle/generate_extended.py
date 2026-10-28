"""Exercise all pinned CPU objective and metric factories, without substitute kernels."""
import json
import shlex
import subprocess
import time

from common import OUT, ROOT, SOURCE, check_resources, load_reference, save_json, sha

OBJECTIVES = ['regression', 'regression_l1', 'quantile', 'huber', 'fair', 'poisson', 'binary',
              'lambdarank', 'rank_xendcg', 'multiclass', 'multiclassova', 'cross_entropy',
              'cross_entropy_lambda', 'mape', 'gamma', 'tweedie']
METRICS = ['l2', 'rmse', 'l1', 'quantile', 'huber', 'fair', 'poisson', 'binary_logloss',
           'binary_error', 'auc', 'average_precision', 'auc_mu', 'ndcg', 'map', 'multi_logloss',
           'multi_error', 'cross_entropy', 'cross_entropy_lambda', 'kullback_leibler',
           'mape', 'gamma', 'gamma_deviance', 'tweedie', 'r2']


def case(name, objective, weighted=False, **params):
    labels = [-2.25, -0.125, 0., 1.125, 3.75, 7.]
    scores = [-3., -0.125, 0., 0.3, 2., 10.]
    if objective in ('poisson', 'gamma', 'tweedie'):
        labels = [0.25, 1., 2., 0.5, 3.75, 7.]
    if objective in ('binary', 'multiclass', 'multiclassova', 'lambdarank', 'rank_xendcg'):
        labels = [0., 1., 0., 1., 1., 0.]
    if objective in ('cross_entropy', 'cross_entropy_lambda'):
        labels = [0., 0.125, 0.25, 0.75, 0.875, 1.]
    config = dict(objective=objective, deterministic=True, num_threads=1, verbosity=-1, **params)
    if objective in ('multiclass', 'multiclassova'):
        config['num_class'] = 3
        scores = [-1., 0., 2., 0.5, 0.5, -2., 1., 0., -2., 0.5, 1., 2., 0., 0., 0., 0.5, -1., 0.]
    c = dict(name=name, objective=objective, params=config, labels=labels, scores=scores)
    if weighted:
        c['weights'] = [1., 0.125, 2., 0.5, 3., 1.25]
    if objective in ('lambdarank', 'rank_xendcg'):
        c.update(labels=[3., 1., 0., 2., 0., 0.], groups=[3, 2, 1], sequential=True)
    return c


def specifications():
    objectives = [case(o + ('_weighted' if w else ''), o, w) for o in OBJECTIVES for w in (False, True)]
    variants = [('regression_sqrt', 'regression', dict(reg_sqrt=True)),
                ('quantile_alpha02', 'quantile', dict(alpha=0.2)),
                ('quantile_alpha08', 'quantile', dict(alpha=0.8)),
                ('huber_alpha025', 'huber', dict(alpha=0.25)),
                ('fair_c025', 'fair', dict(fair_c=0.25)),
                ('poisson_delta02', 'poisson', dict(poisson_max_delta_step=0.2)),
                ('tweedie_power12', 'tweedie', dict(tweedie_variance_power=1.2)),
                ('tweedie_power18', 'tweedie', dict(tweedie_variance_power=1.8)),
                ('binary_sigmoid2', 'binary', dict(sigmoid=2.)),
                ('binary_unbalanced', 'binary', dict(is_unbalance=True)),
                ('binary_scale3', 'binary', dict(scale_pos_weight=3.)),
                ('ova_sigmoid2', 'multiclassova', dict(sigmoid=2.)),
                ('lambda_custom', 'lambdarank', dict(sigmoid=2., lambdarank_norm=False, lambdarank_truncation_level=2, label_gain=[0., 1., 3., 5.])),
                ('xe_seed42', 'rank_xendcg', dict(objective_seed=42))]
    objectives += [case(n, o, True, **p) for n, o, p in variants]
    positions = case('lambda_positions', 'lambdarank', True, learning_rate=0.2, lambdarank_position_bias_regularization=0.5)
    positions['positions'] = [7, 3, 7, 3, 7, 3]
    objectives.append(positions)
    balanced = case('binary_unbalanced_counts', 'binary', True, is_unbalance=True)
    balanced['labels'] = [0., 0., 0., 0., 1., 1.]
    objectives.append(balanced)
    for objective in ('lambdarank', 'rank_xendcg'):
        c = case(objective + '_tied_scores', objective, True)
        c['scores'] = [0.] * 6
        objectives.append(c)
    for label in (0., 1.):
        c = case('binary_single' + str(int(label)), 'binary')
        c['labels'] = [label] * 6
        objectives.append(c)
    metrics = []
    binary = {'binary_logloss', 'binary_error', 'auc', 'average_precision'}
    multi = {'auc_mu', 'multi_logloss', 'multi_error'}
    for m in METRICS:
        objective = 'binary' if m in binary else 'multiclass' if m in multi else 'lambdarank' if m in ('ndcg', 'map') else 'gamma' if m == 'gamma_deviance' else m if m in ('cross_entropy', 'cross_entropy_lambda', 'poisson', 'gamma', 'tweedie') else 'regression'
        for weighted in (False, True):
            c = case(m + ('_weighted' if weighted else ''), objective, weighted)
            c.pop('sequential', None)
            c['metric'] = m
            if m in multi:
                c['labels'] = [0., 1., 2., 1., 2., 0.]
            if m in ('ndcg', 'map'):
                c['params']['eval_at'] = [1, 2, 4]
            if m == 'kullback_leibler':
                c['objective'] = c['params']['objective'] = 'cross_entropy'
                c['labels'] = [0., 0.125, 0.25, 0.75, 0.875, 1.]
            metrics.append(c)
    for m in ('auc', 'average_precision'):
        for label in (0., 1.):
            c = case(f'{m}_single{int(label)}', 'binary', True)
            c.update(metric=m, labels=[label]*6, scores=[0.]*6)
            metrics.append(c)
        c = case(m + '_ties', 'binary', True)
        c.update(metric=m, scores=[0., 0., 1., 1., -1., -1.])
        metrics.append(c)
    for m in ('ndcg', 'map'):
        c = case(m + '_zero_relevance', 'lambdarank', True)
        c.pop('sequential')
        c.update(metric=m, labels=[0.]*6, scores=[0.]*6)
        c['params']['eval_at'] = [1, 2, 4]
        metrics.append(c)
    for perfect in (False, True):
        c = case('r2_constant' + ('_perfect' if perfect else ''), 'regression', True)
        c.update(metric='r2', labels=[2.]*6, scores=[2. if perfect else 0.]*6)
        metrics.append(c)
    c = case('auc_mu_ties', 'multiclass', True)
    c.update(metric='auc_mu', labels=[0., 1., 2., 0., 1., 2.], scores=[0.]*18)
    metrics.append(c)
    c = case('multi_error_top2', 'multiclass', True, multi_error_top_k=2)
    c.update(metric='multi_error', labels=[0., 1., 2., 0., 1., 2.])
    metrics.append(c)
    for m in ('multi_error', 'multi_logloss'):
        c = case(m + '_ova_absent', 'multiclassova', True, sigmoid=2.)
        c['metric'] = m
        metrics.append(c)
    c = case('auc_mu_costs', 'multiclass', True, auc_mu_weights=[0., 2., 1., 3., 0., 4., 2., 5., 0.])
    c.update(metric='auc_mu', labels=[0., 1., 2., 0., 1., 2.])
    metrics.append(c)
    for m, objective, params in [('quantile', 'quantile', dict(alpha=0.2)), ('huber', 'huber', dict(alpha=0.25)),
                                 ('fair', 'fair', dict(fair_c=0.25)), ('tweedie', 'tweedie', dict(tweedie_variance_power=1.8)),
                                 ('l2', 'regression', dict(reg_sqrt=True))]:
        c = case(m + '_variant', objective, True, **params)
        c['metric'] = m
        metrics.append(c)
    for o in ('multiclass', 'multiclassova'):
        c = case(o + '_all_classes', o, True)
        c['labels'] = [0., 1., 2., 0., 1., 2.]
        objectives.append(c)
    # Wrong-class saturation exercises std::log(float kEpsilon), not log(double).
    for metric, objective in [('binary_logloss', 'binary'), ('multi_logloss', 'multiclass'),
                              ('multi_logloss', 'multiclassova')]:
        for weighted in (False, True):
            c = case(f'{metric}_{objective}_saturated' + ('_weighted' if weighted else ''), objective, weighted)
            c['metric'] = metric
            if objective == 'binary':
                c.update(labels=[0., 1., 0., 1., 0., 1.], scores=[1000., -1000., -1000., 1000., 0., 0.])
            else:
                c.update(labels=[0., 1., 2., 0., 1., 2.],
                         scores=[-1000., 1000., 1000., 1000., 0., 0.,
                                 1000., -1000., -1000., -1000., 0., 0.,
                                 0., 0., -1000., -1000., 0., 0.])
            if weighted:
                c['weights'] = [1., 2., 0., 0.5, 0., 3.]
            c['record_objective_metadata'] = True
            metrics.append(c)
    metadata = []
    for name, weighted, classes, grouped, positioned in [
            ('labels_only', False, 1, False, False),
            ('weighted_scalar_init', True, 1, False, False),
            ('weighted_multiclass_init', True, 3, False, False),
            ('ranking_groups_positions', True, 1, True, True),
            ('ranking_unweighted', False, 1, True, False),
            ('negative_position_codes', True, 1, True, True)]:
        c = dict(name=name, labels=[-2.25, -0.125, 0., 1.125, 3.75, 7.], num_class=classes)
        if weighted:
            c['weights'] = [1., 0.125, 0., 0.5, 3., 1.25]
        if name != 'labels_only':
            c['initial_scores'] = [float(i) / 8 - 1. for i in range(6 * classes)]
        if grouped:
            c.update(labels=[3., 1., 0., 2., 0., 0.], groups=[3, 2, 1])
        if positioned:
            c['positions'] = [-5, 7, -5, 42, 7, 42] if name == 'negative_position_codes' else [7, 3, 7, 3, 7, 3]
        metadata.append(c)
    for zero in (False, True):
        c = dict(name='shared_dataset_bins' + ('_zero_missing' if zero else ''), labels=[0., 1., 2., 3., 4., 5.], num_class=1,
                 binning=dict(rows=[[-3., 10.], [-1., 0.], [0., 20.], [1., 30.], [3., 40.], [None, None]],
                              validation_rows=[[-1000., 1000.], [-1.5, 25.], [2.5, -1000.], [0., 0.], [None, None]],
                              forced_upper_bounds=[[-1.5, 2.5], [15., 35.]], max_bin_by_feature=[4, 7], zero_as_missing=zero))
        metadata.append(c)
    forced_bins = []
    for name, max_bin, forced, zero, use, missing in [
            ('forced_two_cuts', 8, [-1.5, 2.5], False, True, False),
            ('forced_outside_sample', 8, [-100., 100.], False, True, False),
            ('forced_max_bin3', 3, [-1.5, 2.5], False, True, False),
            ('forced_nan', 8, [-1.5, 2.5], False, True, True),
            ('forced_zero_missing', 8, [-1.5, 2.5], True, True, True),
            ('forced_missing_disabled', 8, [-1.5, 2.5], False, False, True),
            ('feature0_maxbin4', 4, [], False, True, False),
            ('feature1_maxbin7', 7, [], False, True, False),
            ('forced_zero_cut', 5, [0.], False, True, False)]:
        sample = [-3., -2., -1., 1., 2., 3., 4., 5.] + ([None] if missing else [])
        probes = [-1000., -100., -3., -1.5000000000000002, -1.5, -1.4999999999999998, -1e-36, 0., 1e-36, 2.4999999999999996, 2.5, 2.5000000000000004, 5., 100., 1000., None]
        forced_bins.append(dict(name=name, sample_values=sample, total_sample_count=len(sample)+2,
            probe_values=probes, forced_upper_bounds=forced, is_categorical=False,
            params=dict(max_bin=max_bin, min_data_in_bin=1, min_split_data=1, pre_filter=False,
                        use_missing=use, zero_as_missing=zero, bin_type='NumericalBin')))
    return dict(extended_objectives=objectives, metrics=metrics, dataset_metadata=metadata, forced_bins=forced_bins)


def generate(fixtures=ROOT / 'tests/fixtures'):
    started = time.monotonic()
    _, record = load_reference()
    build = OUT / 'build'
    commands = json.loads((build / 'compile_commands.json').read_text())
    command = shlex.split(commands[0]['command'])
    command[command.index('-o') + 1] = str(OUT / 'extended_driver.o')
    command[command.index('-c') + 1] = str(ROOT / 'tools/oracle/extended_driver.cpp')
    command.extend([f'-I{SOURCE}', f'-I{OUT}'])
    subprocess.run(command, cwd=build, check=True)
    objects = sorted((build / 'CMakeFiles/lightgbm_objs.dir').rglob('*.o'))
    link = [*command[:command.index('-o')], str(OUT / 'extended_driver.o'), *map(str, objects),
            str(build / 'external_libs/nanoarrow/libnanoarrow_static.a'), '-o', str(OUT / 'extended_driver')]
    subprocess.run(link, check=True)
    request = OUT / 'extended-request.json'
    save_json(request, specifications())
    fixtures.mkdir(parents=True, exist_ok=True)
    subprocess.run([str(OUT / 'extended_driver'), str(request), str(fixtures)], check=True, timeout=300)
    for family in ('extended_objectives', 'metrics', 'dataset_metadata', 'forced_bins'):
        path = fixtures / (family + '.json')
        save_json(path, json.loads(path.read_text()))
    save_json(OUT / 'extended-driver-record.json', dict(compile_command=command, link_command=link,
              source_sha256=sha(ROOT / 'tools/oracle/extended_driver.cpp'),
              shared_header_sha256=sha(ROOT / 'tools/oracle/driver_common.hpp'), executable_sha256=sha(OUT / 'extended_driver'),
              loaded_library_sha256=record['library']['loaded_sha256']))
    print('Generated native extended objectives, metrics, metadata and shared/forced bins',
          check_resources(started), check_resources(started, children=True))


if __name__ == '__main__':
    generate()
