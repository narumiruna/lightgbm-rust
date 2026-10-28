"""Capture remaining native DART and coordinated custom-CV workflows.

Uses only the source-identity-checked Python engine and loaded native library.
Regenerate: uv run --project tools/oracle --locked python tools/oracle/generate_workflow_extended.py
"""
import time

import numpy as np

from common import COMMIT, ROOT, check_resources, load_reference, save_json
from generate_predictions import BASE
from workflow_common import integer_data, metrics, provenance, save_model

NAMES = {
    'dart_schedule_weighted', 'dart_schedule_uniform', 'dart_continue_weighted',
    'dart_continue_xgboost', 'dart_continue_original_three', 'dart_schedule_after',
    'goss_schedule', 'cv_custom_schedule', 'cv_builtin_schedule', 'cv_custom_stop',
    'cv_builtin_stop', 'cv_custom_schedule_after', 'gbdt_schedule_features',
    'dart_schedule_features', 'goss_schedule_features',
}


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, record = load_reference()
    x, labels = integer_data()
    y = np.asarray(labels, dtype=np.float32)
    cases = []
    rates = [0.25, 0.25, 0.4, 0.4, 0.15, 0.15, 0.3, 0.3]
    for name, warm, uniform, xg in [
        ('dart_schedule_weighted', False, False, False),
        ('dart_schedule_uniform', False, True, False),
        ('dart_continue_weighted', True, False, False),
        ('dart_continue_xgboost', True, False, True),
        ('dart_continue_original_three', True, False, False),
        ('dart_schedule_after', False, False, False),
        ('goss_schedule', False, False, False),
        ('gbdt_schedule_features', False, False, False),
        ('dart_schedule_features', False, False, False),
        ('goss_schedule_features', False, False, False),
    ]:
        started = time.monotonic()
        print(f'Capturing {name}', flush=True)
        params = dict(BASE, boosting='dart', objective='regression', metric=['l2', 'l1'],
                      drop_rate=0.7, skip_drop=0.2, max_drop=3, drop_seed=7,
                      uniform_drop=uniform, xgboost_dart_mode=xg)
        if name.startswith('goss_schedule'):
            params.update(boosting='gbdt', data_sample_strategy='goss', top_rate=0.6, other_rate=0.2)
        if name.startswith('gbdt_schedule'):
            params.update(boosting='gbdt')
        if name.endswith('_features'):
            params['feature_fraction'] = 0.5
        case_labels, case_rates = labels, rates
        initial = None
        if name == 'dart_continue_original_three':
            # The older operations fixture reads _inner_predict BEFORE the first
            # update. In DART that read calls DroppingTrees and consumes RNG.
            # Capture the same saved-prefix / changed-label workflow here without
            # that mutating probe; Rust callback environments are read-only.
            params.update(skip_drop=0., drop_rate=.5, drop_seed=0, max_drop=50)
            case_labels = [v + (2 if i % 2 else -1) for i, v in enumerate(labels)]
            case_rates = [params['learning_rate']] * 3
            prefix = lgb.train(params, lgb.Dataset(x, label=y, params=params), num_boost_round=4)
            initial = save_model(prefix, fixtures, f'workflow_extra_{name}_initial', x, num_iteration=-1)
        elif warm:
            prefix = lgb.train(params, lgb.Dataset(x, label=y, params=params), num_boost_round=5)
            initial = save_model(prefix, fixtures, f'workflow_extra_{name}_initial', x, num_iteration=-1)
        case_y = np.asarray(case_labels, dtype=np.float32)
        train = lgb.Dataset(x, label=case_y, params=params, free_raw_data=False)
        valid = lgb.Dataset(x[::-1].copy(), label=case_y[::-1].copy(), reference=train, free_raw_data=False)
        steps = []
        def capture(env):
            steps.append(dict(iteration=env.iteration, begin_iteration=env.begin_iteration,
                              end_iteration=env.end_iteration, learning_rate=env.params['learning_rate'],
                              evaluation=metrics(env.evaluation_result_list),
                              training_scores=env.model._Booster__inner_predict(data_idx=0).tolist(),
                              raw=env.model.predict(x, raw_score=True, num_iteration=-1).tolist()))
        capture.order = 25
        capture.before_iteration = False
        schedule = lgb.reset_parameter(learning_rate=case_rates)
        if name.endswith('_after'):
            schedule.before_iteration = False
            schedule.order = 35
        model = lgb.train(params, train, num_boost_round=len(case_rates), valid_sets=[train, valid],
                          valid_names=['train', 'valid'], init_model=None if initial is None else fixtures / initial['model_file'],
                          callbacks=[schedule, capture], keep_training_booster=True)
        cases.append(dict(name=name, kind='train', params=params, rows=x.tolist(), labels=case_labels,
                          learning_rates=case_rates, schedule_after=name.endswith('_after'), initial_model=initial, steps=steps,
                          final_model=save_model(model, fixtures, f'workflow_extra_{name}', x, num_iteration=-1)))
        check_resources(started)
    folds = [(np.asarray([i for i in range(len(x)) if i % 3 != k]),
              np.asarray([i for i in range(len(x)) if i % 3 == k])) for k in range(3)]
    for name, custom, stopping in [('cv_custom_schedule', True, False), ('cv_builtin_schedule', False, False),
                                    ('cv_custom_stop', True, True), ('cv_builtin_stop', False, True),
                                    ('cv_custom_schedule_after', True, False)]:
        started = time.monotonic()
        params = dict(BASE, objective='regression', metric='None', boost_from_average=False)
        objective_calls = []
        def objective(predictions, data):
            objective_calls.append(dict(rows=data.num_data(), predictions=predictions.tolist()))
            return predictions - data.get_label(), np.ones_like(predictions)
        if custom:
            params['objective'] = objective
        else:
            params['metric'] = 'l2'
        def metric(predictions, data):
            return 'custom_l2', float(np.mean((predictions - data.get_label()) ** 2)), False
        data = lgb.Dataset(x, label=y, params={k: v for k, v in params.items() if k != 'objective'}, free_raw_data=False)
        steps, trace, history = [], [], {}
        def observe_before(env):
            trace.append(dict(phase='before', iteration=env.iteration, history_length=len(history.get('valid', {}).get('custom_l2-mean' if custom else 'l2-mean', []))))
        observe_before.order, observe_before.before_iteration = 15, True
        def observe_early(env):
            trace.append(dict(phase='after_early', iteration=env.iteration, history_length=len(history.get('valid', {}).get('custom_l2-mean' if custom else 'l2-mean', []))))
        observe_early.order = 15
        def capture(env):
            trace.append(dict(phase='after_record', iteration=env.iteration, history_length=len(history['valid']['custom_l2-mean' if custom else 'l2-mean'])))
            steps.append(dict(iteration=env.iteration, learning_rate=env.params['learning_rate'],
                              evaluation=metrics(env.evaluation_result_list),
                              folds=[dict(evaluation=metrics(b.eval_valid(metric if custom else None)),
                                          raw=b.predict(x, raw_score=True, num_iteration=-1).tolist()) for b in env.model.boosters]))
        capture.order = 25
        schedule = lgb.reset_parameter(learning_rate=rates)
        if name.endswith('_after'):
            schedule.before_iteration = False
            schedule.order = 35
        callbacks = [schedule, observe_before, observe_early, lgb.record_evaluation(history), capture]
        if stopping:
            callbacks.append(lgb.early_stopping(2, verbose=False, min_delta=1e6))
        result = lgb.cv(params, data, num_boost_round=len(rates), folds=folds, stratified=False,
                        feval=metric if custom else None, eval_train_metric=True,
                        callbacks=callbacks, return_cvbooster=True)
        cv = result.pop('cvbooster')
        stored_params = dict(params, objective='custom' if custom else 'regression')
        cases.append(dict(name=name, kind='cv', params=stored_params, rows=x.tolist(), labels=labels,
                          learning_rates=rates, schedule_after=name.endswith('_after'), custom=custom, stopping=stopping,
                          folds=[dict(train_indices=a.tolist(), valid_indices=b.tolist()) for a, b in folds],
                          objective_calls=objective_calls, trace=trace, steps=steps, history=result,
                          best_iteration=cv.best_iteration,
                          models=[save_model(b, fixtures, f'workflow_extra_{name}_fold{k}', x, num_iteration=-1) for k, b in enumerate(cv.boosters)]))
        check_resources(started)
    assert {case['name'] for case in cases} == NAMES
    save_json(fixtures / 'workflow_extended.json', dict(schema_version=1, upstream_commit=COMMIT,
                                                      provenance=provenance(record), cases=cases))
    print(f'Generated {len(cases)} native DART/custom-CV cases')


if __name__ == '__main__':
    generate()
