"""Capture original Python train/callback behavior over native CPU models."""
import copy
import math
import time
import warnings

import numpy as np

from common import COMMIT, ROOT, check_resources, load_reference, save_json
from generate_predictions import BASE
from workflow_common import integer_data, metrics, predict, provenance, save_model, snapshot

NAMES = {'ordering_schedule', 'early_multival', 'early_first_metric_only', 'early_all_metrics',
         'early_delta_scalar', 'early_delta_vector', 'early_final_round', 'training_excluded',
         'training_only', 'no_validation', 'dart_disabled', 'no_split',
         'custom_l2', 'custom_binary', 'invalid_custom_length'}


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, record = load_reference()
    (fixtures / 'models').mkdir(parents=True, exist_ok=True)
    x, original = integer_data()
    cases = []
    for name in sorted(NAMES):
        started = time.monotonic()
        binary = name in {'early_first_metric_only', 'early_all_metrics', 'custom_binary'}
        labels = [int(v > 0) for v in original] if binary else original
        rows = np.zeros_like(x) if name == 'no_split' else x
        labels = [2] * len(x) if name == 'no_split' else labels
        params = dict(BASE, objective='binary' if binary else 'regression', metric=['binary_logloss', 'auc'] if binary else ['l2', 'l1'])
        rounds = 7
        if name == 'dart_disabled':
            params.update(boosting='dart', drop_rate=0.5, skip_drop=0.)
        custom = name.startswith('custom_') or name == 'invalid_custom_length'
        if custom:
            params.update(boost_from_average=False, metric='None')
        train = lgb.Dataset(rows, label=np.asarray(labels, dtype=np.float32), params=params, free_raw_data=False)
        valid_labels = [-v for v in labels] if name in {'early_multival', 'training_excluded'} else labels
        valid = lgb.Dataset(rows, label=np.asarray(valid_labels, dtype=np.float32), reference=train)
        second = lgb.Dataset(rows, label=np.asarray(labels[::-1], dtype=np.float32), reference=train)
        datasets, names = [train, valid], ['train_alias', 'valid_a']
        if name == 'early_multival':
            datasets.append(second)
            names.append('valid_b')
        if name == 'training_only':
            params['metric'] = ['l2']
            datasets, names = [train], ['train_alias']
        if name == 'no_validation':
            datasets, names = None, None
        if name in {'early_delta_scalar', 'early_delta_vector'}:
            datasets, names = [train, valid, second], ['train_alias', 'valid_a', 'valid_b']
        events, history, derivatives = [], {}, []
        early = None

        def state(env):
            return dict(iteration=env.iteration, begin_iteration=env.begin_iteration, end_iteration=env.end_iteration,
                        model_current_iteration=env.model.current_iteration(), model_best_iteration=env.model.best_iteration,
                        learning_rate=float(env.params['learning_rate']), evaluation=metrics(env.evaluation_result_list),
                        history_lengths={ds: {m: len(v) for m, v in ms.items()} for ds, ms in history.items()})

        def wrap(label, callback, order, before):
            def invoke(env):
                events.append(dict(callback=label, phase='enter', **state(env)))
                try:
                    callback(env)
                except Exception as error:
                    item = dict(callback=label, phase='raise', error_type=type(error).__name__, **state(env))
                    if isinstance(error, lgb.callback.EarlyStopException):
                        item.update(exception_best_iteration_zero=error.best_iteration, exception_best_score=metrics(error.best_score))
                    else:
                        item['error'] = str(error)
                    events.append(item)
                    raise
                item = dict(callback=label, phase='exit', **state(env))
                if callback is early:
                    item.update(enabled=early.enabled, best_iterations_zero=list(early.best_iter))
                events.append(item)
            invoke.order, invoke.before_iteration = order, before
            return invoke

        callbacks = [wrap('before_first', lambda env: None, -20, True),
                     wrap('before_last', lambda env: None, 15, True),
                     wrap('after_first', lambda env: None, -10, False),
                     wrap('record_evaluation', lgb.record_evaluation(history), 20, False),
                     wrap('after_history', lambda env: None, 25, False),
                     wrap('after_stopping', lambda env: None, 40, False)]
        schedule = None
        if name == 'ordering_schedule':
            schedule = [0.5, 0.25, 0.125, 0.25, 0.5, 0.125, 0.25]
            callbacks.append(wrap('reset_parameter', lgb.reset_parameter(learning_rate=schedule), 10, True))
        stopping = None
        if name not in {'ordering_schedule', 'no_split', 'custom_l2', 'custom_binary', 'invalid_custom_length'}:
            stopping = dict(stopping_rounds=1 if binary else 2, first_metric_only=name == 'early_first_metric_only', verbose=False,
                            min_delta=[50., 1.] if name == 'early_delta_vector' else 50. if name == 'early_delta_scalar' else 0.)
            early = lgb.early_stopping(**stopping)
            callbacks.append(wrap('early_stopping', early, 30, False))

        def objective(scores, data):
            g, h = [], []
            for score, label in zip(scores.tolist(), data.get_label().tolist()):
                if binary:
                    p = 1. / (1. + math.exp(-score))
                    g.append(p - label)
                    h.append(p * (1. - p))
                else:
                    g.append(score - label)
                    h.append(1.)
            if name == 'invalid_custom_length':
                g = g[:-1]
            derivatives.append(dict(raw_input=scores.tolist(), gradients=np.asarray(g, dtype=np.float32).tolist(),
                                    hessians=np.asarray(h, dtype=np.float32).tolist()))
            return np.asarray(g, dtype=np.float32), np.asarray(h, dtype=np.float32)

        def evaluation(scores, data):
            values = list(zip(scores.tolist(), data.get_label().tolist()))
            if binary:
                # The objective is custom, so feval receives raw margins.
                value = sum(-y * math.log(max(1e-15, 1. / (1. + math.exp(-s))))
                            - (1. - y) * math.log(max(1e-15, 1. - 1. / (1. + math.exp(-s)))) for s, y in values) / len(values)
                return 'custom_binary_logloss', value, False
            return 'custom_l2', sum((s - y) ** 2 for s, y in values) / len(values), False

        public_params = dict(params)
        if custom:
            params['objective'] = objective
            public_params['objective'] = 'custom_binary' if binary else 'custom_l2'
        item = dict(name=name, params=public_params, rows=rows.tolist(), labels=labels,
                    valid_sets=[] if datasets is None else [dict(name=n, is_training=d is train, labels=d.get_label().tolist() if isinstance(d.get_label(), np.ndarray) else list(d.get_label())) for n, d in zip(names, datasets)],
                    num_boost_round=rounds, stopping=stopping, learning_rate_schedule=schedule,
                    callback_order=[dict(name=label, order=order, before_iteration=before) for label, order, before in
                                    [('before_first', -20, True), ('before_last', 15, True), ('after_first', -10, False),
                                     ('record_evaluation', 20, False), ('after_history', 25, False), ('after_stopping', 40, False)]])
        if schedule is not None:
            item['callback_order'].append(dict(name='reset_parameter', order=10, before_iteration=True))
        if early is not None:
            item['callback_order'].append(dict(name='early_stopping', order=30, before_iteration=False))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                model = lgb.train(params, train, num_boost_round=rounds, valid_sets=datasets, valid_names=names,
                                  callbacks=callbacks, feval=evaluation if custom else None, keep_training_booster=True)
            except (ValueError, lgb.basic.LightGBMError) as error:
                item.update(error_type=type(error).__name__, error=str(error), error_layer='original Python validation before native learner access')
            else:
                item.update(best_iteration=model.best_iteration, best_score=dict(model.best_score),
                            live_model=snapshot(model, rows), final_model=save_model(model, fixtures, 'workflow_' + name, rows),
                            full_model=save_model(model, fixtures, 'workflow_' + name + '_full', rows, num_iteration=-1))
                if custom:
                    builtin_params = dict(BASE, objective='binary' if binary else 'regression', boost_from_average=False)
                    builtin = lgb.train(builtin_params, lgb.Dataset(rows, label=np.asarray(labels, dtype=np.float32), params=builtin_params), num_boost_round=rounds, keep_training_booster=True)
                    item['builtin_evaluation'] = metrics(builtin.eval_train())
                    custom_metric = 'custom_binary_logloss' if binary else 'custom_l2'
                    native_value = item['builtin_evaluation'][0]['metric_value']
                    assert abs(history['train_alias'][custom_metric][-1] - native_value) <= 1e-12 + 1e-10 * abs(native_value)
                    item['builtin_reference'] = save_model(builtin, fixtures, 'workflow_' + name + '_builtin', rows)
                    assert predict(model, rows, raw_score=True) == predict(builtin, rows, raw_score=True)
            item['warnings'] = [str(w.message) for w in caught]
        item.update(events=events, history=copy.deepcopy(history), derivatives=derivatives)
        cases.append(item)
        check_resources(started)
    save_json(fixtures / 'workflow.json', dict(schema_version=1, upstream_commit=COMMIT, provenance=provenance(record), cases=cases))
    print(f'Generated {len(cases)} actual engine/callback workflow cases')


if __name__ == '__main__':
    generate()
