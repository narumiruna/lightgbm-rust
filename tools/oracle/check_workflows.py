"""Validate native workflow shapes, callback control flow, and CV aggregation."""
import math
from common import ROOT
from generate_predictions import BASE


def close(actual, expected):
    assert abs(actual - expected) <= 1e-12 + 1e-10 * abs(expected), (actual, expected)


def model_shape(model, rows, classes):
    n = len(rows)
    for key in ('raw', 'transformed'):
        assert len(model[key]) == n and all(len(row) == classes for row in model[key])
        assert all(math.isfinite(v) for row in model[key] for v in row)
    assert len(model['leaves']) == n and all(len(row) == model['num_trees'] for row in model['leaves'])
    assert all(isinstance(v, int) or int(v) == v for row in model['leaves'] for v in row)
    assert model['current_iteration'] * classes == model['num_trees']
    if 'model_file' in model:
        path = ROOT / 'tests/fixtures' / model['model_file']
        assert path.resolve().is_relative_to(ROOT / 'tests/fixtures/models') and path.is_file()
        assert sum(line.startswith('Tree=') for line in path.read_text().splitlines()) == model['num_trees']


def validate(family, data, manifest=None):
    p = data['provenance']
    if manifest:
        assert p['loaded_library_sha256'] == manifest['loaded_library']['loaded_sha256']
        assert p['dependencies']['numpy'] == '2.2.6'
        assert p['dependencies']['scikit-learn'] == '1.6.1'
        hashes = {entry['path']: entry['sha256'] for entry in manifest['reference_sources']}
        assert all(hashes['python-package/lightgbm/' + name] == value for name, value in p['python_sources'].items())
    for c in data['cases']:
        rows, labels, params = c['rows'], c['labels'], c['params']
        n, features = len(rows), len(rows[0])
        classes = params.get('num_class', 1)
        assert len(labels) == n and all(len(row) == features for row in rows)
        assert all(math.isfinite(v) for row in rows for v in row)
        assert all(params[key] == BASE[key] for key in ('device_type', 'tree_learner', 'num_threads', 'deterministic', 'force_col_wise',
                                                       'seed', 'data_random_seed', 'feature_fraction_seed', 'bagging_seed', 'drop_seed',
                                                       'extra_seed', 'objective_seed'))
        for key in ('initial_model', 'final_model', 'full_model', 'builtin_reference', 'live_model'):
            if key in c:
                model_shape(c[key], rows, classes)
        if family == 'workflow':
            workflow(c, rows, classes)
        elif family == 'cv':
            cv(c, rows, classes)
        else:
            operations(c, rows, classes)


def workflow(c, rows, classes):
    n = len(rows)
    for valid in c['valid_sets']:
        assert len(valid['labels']) == n
    before = sorted((cb for cb in c['callback_order'] if cb['before_iteration']), key=lambda cb: cb['order'])
    after = sorted((cb for cb in c['callback_order'] if not cb['before_iteration']), key=lambda cb: cb['order'])
    assert len({cb['order'] for cb in before}) == len(before)
    assert len({cb['order'] for cb in after}) == len(after)
    by_iteration = {}
    for event in c['events']:
        assert event['begin_iteration'] == 0 and event['end_iteration'] == c['num_boost_round']
        assert 0 <= event['iteration'] < c['num_boost_round']
        assert event['model_best_iteration'] == 0
        by_iteration.setdefault(event['iteration'], []).append(event)
        evaluation = event['evaluation']
        if evaluation:
            assert len({(v['dataset_name'], v['metric_name']) for v in evaluation}) == len(evaluation)
            assert all(v['metric_std_dev'] is None for v in evaluation)
            for v in evaluation:
                if v['metric_name'] in ('l2', 'l1', 'binary_logloss', 'custom_l2', 'custom_binary_logloss'):
                    assert not v['maximize']
                elif v['metric_name'] == 'auc':
                    assert v['maximize']
    assert list(by_iteration) == list(range(len(by_iteration)))
    for iteration, events in by_iteration.items():
        order = [cb['name'] for cb in before + ([] if c['name'] == 'invalid_custom_length' else after)]
        raised = next((e['callback'] for e in events if e['phase'] == 'raise'), None)
        if raised is not None:
            order = order[:order.index(raised) + 1]
        assert [e['callback'] for e in events if e['phase'] == 'enter'] == order
        assert len(events) == 2 * len(order)
        for enter, leave in zip(events[::2], events[1::2]):
            assert enter['phase'] == 'enter' and leave['phase'] in ('exit', 'raise')
            assert enter['callback'] == leave['callback']
            assert (enter['evaluation'] is None) == (enter['callback'] in [cb['name'] for cb in before])
        for event in events:
            if event['callback'] == 'after_history':
                for ds, metric_values in c['history'].items():
                    for name in metric_values:
                        assert event['history_lengths'][ds][name] == iteration + 1
    for histories in c['history'].values():
        for values in histories.values():
            assert len(values) == len(by_iteration)
    if 'final_model' in c:
        assert c['final_model']['current_iteration'] == (c['best_iteration'] or c['live_model']['current_iteration'])
        assert c['full_model']['num_trees'] == c['live_model']['num_trees']
    if c['name'] == 'no_split':
        assert len(by_iteration) == c['num_boost_round'] and c['full_model']['num_trees'] == 1
    if c['name'].startswith('custom_'):
        assert c['final_model']['raw'] == c['builtin_reference']['raw']
        assert c['params']['boost_from_average'] is False
        assert c['derivatives'][0]['raw_input'] == [0.] * n
        custom_metric = 'custom_binary_logloss' if c['name'] == 'custom_binary' else 'custom_l2'
        close(c['history']['train_alias'][custom_metric][-1], c['builtin_evaluation'][0]['metric_value'])
    if c['learning_rate_schedule']:
        for iteration, events in by_iteration.items():
            applied = next(e for e in events if e['callback'] == 'reset_parameter' and e['phase'] == 'exit')
            assert applied['learning_rate'] == c['learning_rate_schedule'][iteration]


def cv(c, rows, classes):
    n, folds = len(rows), c['folds']
    assert len(folds) == len(c['models']) == c['nfold'] == len(c['source_folds'])
    all_valid = []
    for fold, source, models in zip(folds, c['source_folds'], c['models']):
        train, valid = fold['train_indices'], fold['valid_indices']
        assert train == sorted(set(train)) and valid == sorted(set(valid))
        assert train == sorted(source['train_indices']) and valid == sorted(source['valid_indices'])
        assert not set(train) & set(valid) and sorted(train + valid) == list(range(n))
        all_valid += valid
        if c['groups']:
            boundary = 0
            for group in c['groups']:
                assert len(set(range(boundary, boundary + group)) & set(valid)) in (0, group)
                boundary += group
            assert boundary == n
        assert models['best_iteration'] == c['best_iteration']
        model_shape(models['final_model'], rows, classes)
        model_shape(models['full_model'], rows, classes)
        if c['best_iteration'] > 0:
            assert models['final_model']['current_iteration'] == c['best_iteration']
    assert sorted(all_valid) == list(range(n))
    if c['permutation'] is not None:
        assert sorted(c['permutation']) == list(range(n))
    assert [s['iteration'] for s in c['steps']] == list(range(len(c['steps'])))
    history_size = c['best_iteration'] if c['best_iteration'] > 0 else c['num_boost_round']
    assert all(len(v) == history_size for v in c['history'].values())
    for step in c['steps']:
        assert len(step['folds']) == c['nfold']
        for i, aggregated in enumerate(step['aggregate']):
            values = []
            for fold in step['folds']:
                metric = fold['evaluation'][i]
                assert (metric['dataset_name'], metric['metric_name'], metric['maximize']) == (aggregated['dataset_name'], aggregated['metric_name'], aggregated['maximize'])
                values.append(metric['metric_value'])
            mean = sum(values) / len(values)
            close(aggregated['metric_value'], mean)
            close(aggregated['metric_std_dev'], math.sqrt(sum((v - mean) ** 2 for v in values) / len(values)))
            if step['iteration'] < history_size:
                prefix = aggregated['dataset_name'] + ' ' + aggregated['metric_name']
                assert c['history'][prefix + '-mean'][step['iteration']] == aggregated['metric_value']
                assert c['history'][prefix + '-stdv'][step['iteration']] == aggregated['metric_std_dev']
    cv_path = ROOT / 'tests/fixtures' / c['cvbooster_file']
    assert cv_path.is_file()


def operations(c, rows, classes):
    n = len(rows)
    params = c['params']
    operation = c['operation']
    if operation == 'continuation':
        if 'error' in c:
            assert c['name'] == 'continue_rf' and 'init_score' in c['error']
        else:
            assert c['initial_model']['current_iteration'] == 4 and c['final_model']['current_iteration'] == 7
            assert [step['iteration'] for step in c['steps']] == [4, 5, 6]
            for step in c['steps']:
                assert step['begin_iteration'] == 4 and step['end_iteration'] == 7
                assert len(step['training_score_buffer']) == n * classes
                model_shape(step['model'], rows, classes)
            for score, expected in zip(c['steps'][0]['training_score_buffer'], c['initial_model']['raw']):
                close(score, expected[0])
    elif operation == 'initial_scores':
        for score, margin, initial in zip(c['training_score_buffer'], c['final_model']['raw'], c['initial_scores']):
            close(score, margin[0] + initial)
    elif operation == 'refit':
        assert len(c['weights']) == n and all(v >= 0 for v in c['weights'])
        assert len(c['leaf_pred']) == n and all(len(row) == c['initial_model']['num_trees'] for row in c['leaf_pred'])
        assert c['positions'] is None  # original Python refit API supplies no position metadata
        if c['groups']:
            assert sum(c['groups']) == n
        if 'error' in c:
            assert c['name'].startswith('refit_regression_domain_')
            assert c['decay_rate'] < 0 or c['decay_rate'] > 1
            assert c['error_type'] == 'LightGBMError' and 'refit_decay_rate' in c['error']
        if 'backend_params' in c:
            backend = c['backend_params']
            assert backend['boosting'] == params.get('boosting', 'gbdt')
            assert backend['objective'] == params['objective']
            assert backend['data_sample_strategy'] == params.get('data_sample_strategy', 'bagging')
            assert backend['refit_decay_rate'] == c['decay_rate']
            assert backend['bagging_by_query'] == (params.get('bagging_by_query', False) and backend['data_sample_strategy'] == 'bagging')
        if 'final_model' in c:
            assert 0 <= c['decay_rate'] <= 1
            assert c['final_model']['num_trees'] == c['initial_model']['num_trees']
            assert len(c['refit_steps']) == c['initial_model']['current_iteration']
            for iteration, step in enumerate(c['refit_steps']):
                assert step['iteration'] == iteration
                assert all(len(step[key]) == n * classes for key in ('training_scores_before', 'gradients', 'hessians'))
            assert c['refit_steps'][0]['training_scores_before'] == [0.] * (n * classes)
            if params.get('boosting') == 'rf':
                for step in c['refit_steps']:
                    assert step['gradients'] == c['refit_steps'][0]['gradients']
                    assert step['hessians'] == c['refit_steps'][0]['hessians']
            assert len(c['training_scores_after']) == n * classes
            for k in range(classes):
                for row in range(n):
                    close(c['training_scores_after'][k * n + row], c['final_model']['raw'][row][k])
    elif operation == 'rollback':
        for i, step in enumerate(c['steps']):
            assert step['call'] == i
            assert step['before_iteration'] == max(0, 4 - i)
            assert step['model']['current_iteration'] == max(0, 3 - i)
            model_shape(step['model'], rows, classes)
            if step['model']['num_trees'] == 0:
                assert step['model']['raw'] == [[0.]] * n and step['model']['leaves'] == [[]] * n
    elif operation == 'set_leaf_output':
        assert c['set_output'] == c['get_output'] == 3.25
        assert c['leaves_before'] == c['final_model']['leaves']
        delta = c['set_output'] - c['old_output']
        for leaf, before, after in zip(c['leaves_before'], c['initial_model']['raw'], c['final_model']['raw']):
            close(after[0], before[0] + (delta if leaf[c['tree_id']] == c['leaf_id'] else 0.))
    elif operation == 'prediction_early_stop':
        for probe in c['probes']:
            if 'error' in probe:
                assert c['name'] == 'predict_early_linear' and 'linear trees' in probe['error']
            else:
                assert len(probe['raw']) == len(probe['transformed']) == len(probe['leaves']) == n
                assert all(len(v) == classes for v in probe['raw'])
                assert probe['leaves'] == c['initial_model']['leaves']
                frequency = probe['params']['pred_early_stop_freq']
                margin = probe['params']['pred_early_stop_margin']
                for row, actual in enumerate(probe['raw']):
                    expected = c['prefix_raw'][-1][row]
                    for count in range(frequency, len(c['prefix_raw']) + 1, frequency):
                        candidate = c['prefix_raw'][count - 1][row]
                        difference = 2. * abs(candidate[0]) if c['name'] == 'predict_early_binary' else (
                            sorted(candidate, reverse=True)[0] - sorted(candidate, reverse=True)[1] if classes > 1 else -1.)
                        if difference > margin:
                            expected = candidate
                            break
                    for a, e in zip(actual, expected):
                        close(a, e)
    else:
        assert operation == 'extreme_gain_diagnostic'
        assert c['importance_gain'] == ['Infinity']
        assert 'split_gain=inf' in (ROOT / 'tests/fixtures' / c['final_model']['model_file']).read_text()
        assert all(abs(v) < 3.4028234663852886e38 for v in c['labels'])
        # JSON model inspection clamps f32 infinities through Common::AvoidInf.
        assert all(t['tree_structure']['split_gain'] == 9.999999680285692e37 for t in c['model_dump']['tree_info'])
        for raw, contrib in zip(c['final_model']['raw'], c['contributions']):
            assert all(math.isfinite(v) for v in contrib)
            close(sum(contrib), raw[0])
