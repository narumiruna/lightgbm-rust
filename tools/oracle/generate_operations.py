"""Native continuation, refit, rollback, leaf mutation, early prediction, and gain limits."""
import json
import subprocess
import time
import warnings

import numpy as np

from common import COMMIT, OUT, ROOT, check_resources, load_reference, save_json
from native import compile_helper
from generate_predictions import BASE
from workflow_common import exceptional, integer_data, predict, provenance, save_model, snapshot

LEGACY_REFIT_KINDS = ('regression', 'binary', 'multiclass', 'categorical', 'ranking', 'linear')
EXTRA_REFIT_PARAMS = {
    'rf': dict(boosting='rf', bagging_fraction=.8, bagging_freq=1, feature_fraction=.8),
    'dart': dict(boosting='dart', skip_drop=0., drop_rate=.5),
    'l1': dict(objective='regression_l1'),
    'quantile': dict(objective='quantile', alpha=.7),
    'mape': dict(objective='mape'),
    'gbdt_querybag': dict(objective='lambdarank', bagging_by_query=True, bagging_fraction=.8, bagging_freq=1),
    'goss_querybag': dict(objective='lambdarank', data_sample_strategy='goss', bagging_by_query=True,
                          top_rate=.25, other_rate=.25),
    'regression_domain': {},
}
REFIT_KINDS = (*LEGACY_REFIT_KINDS, *EXTRA_REFIT_PARAMS)


def refit_decays(kind):
    return (-.1, 1.1) if kind == 'regression_domain' else (0., .5, .9, 1.)


NAMES = ({'continue_' + k for k in ('gbdt', 'dart', 'goss', 'rf')} |
         {'initial_scores_only', 'continue_supplied_initial', 'rollback', 'leaf_output', 'extreme_f32_gain'} |
         {'refit_' + k + '_decay' + str(d).replace('.', '') for k in REFIT_KINDS for d in refit_decays(k)} |
         {'predict_early_' + k for k in ('binary', 'multiclass', 'linear')})


def generate(fixtures=ROOT / 'tests/fixtures'):
    lgb, record = load_reference()
    (fixtures / 'models').mkdir(parents=True, exist_ok=True)
    x, labels = integer_data()
    cases = []
    def fit(params, rows=x, y=labels, **metadata):
        data = lgb.Dataset(rows, label=np.asarray(y, dtype=np.float32), params=params, free_raw_data=False, **metadata)
        return lgb.train(params, data, num_boost_round=4, keep_training_booster=True)
    for kind in ('gbdt', 'dart', 'goss', 'rf', 'supplied_initial'):
        started = time.monotonic()
        name = 'continue_supplied_initial' if kind == 'supplied_initial' else 'continue_' + kind
        params = dict(BASE, objective='regression')
        if kind in ('dart', 'rf'):
            params.update(boosting=kind)
        if kind == 'dart':
            params.update(skip_drop=0., drop_rate=.5)
        if kind == 'goss':
            params.update(data_sample_strategy='goss', top_rate=.25, other_rate=.25)
        if kind == 'rf':
            params.update(bagging_fraction=.8, bagging_freq=1, feature_fraction=.8)
        initial = fit(params)
        source = save_model(initial, fixtures, name + '_initial', x)
        supplied = [7.] * len(x) if kind == 'supplied_initial' else None
        new_y = [v + (2 if i % 2 else -1) for i, v in enumerate(labels)]
        data = lgb.Dataset(x, label=np.asarray(new_y, dtype=np.float32), init_score=supplied, params=params, free_raw_data=False)
        steps = []
        def capture(env):
            steps.append(dict(iteration=env.iteration, begin_iteration=env.begin_iteration, end_iteration=env.end_iteration,
                              training_score_buffer=env.model._Booster__inner_predict(data_idx=0).tolist(),
                              model=snapshot(env.model, x)))
        capture.before_iteration, capture.order = True, 0
        item = dict(name=name, operation='continuation', params=params, rows=x.tolist(), labels=new_y,
                    original_training=dict(rows=x.tolist(), labels=labels, num_boost_round=4), initial_model=source,
                    supplied_initial_scores=supplied, num_boost_round=3)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            try:
                model = lgb.train(params, data, num_boost_round=3, init_model=fixtures / source['model_file'],
                                  callbacks=[capture], keep_training_booster=True)
            except (ValueError, lgb.basic.LightGBMError) as error:
                item.update(error_type=type(error).__name__, error=str(error))
            else:
                item['effective_params'] = dict(model.params)
                item['final_model'] = save_model(model, fixtures, name + '_final', x)
            item['warnings'] = [str(w.message) for w in caught]
        item['steps'] = steps
        cases.append(item)
        check_resources(started)
    params = dict(BASE, objective='regression')
    scores = [7.] * len(x)
    model = fit(params, init_score=scores)
    cases.append(dict(name='initial_scores_only', operation='initial_scores', params=params, rows=x.tolist(), labels=labels,
                      initial_scores=scores, num_boost_round=4, training_score_buffer=model._Booster__inner_predict(data_idx=0).tolist(),
                      final_model=save_model(model, fixtures, 'operations_initial_scores_only', x)))

    for kind in REFIT_KINDS:
        params = dict(BASE, objective={'binary': 'binary', 'multiclass': 'multiclass', 'ranking': 'lambdarank'}.get(kind, 'regression'))
        params.update(EXTRA_REFIT_PARAMS.get(kind, {}))
        ranking = params['objective'] == 'lambdarank'
        y = [int(v > 0) for v in labels] if kind == 'binary' else [i % 3 for i in range(len(x))] if kind == 'multiclass' else [3, 0, 2, 1] * 6 if ranking else labels
        rows = np.asarray([[i % 6, (i * 5) % 11 - 5] for i in range(len(x))], dtype=np.float64) if kind == 'categorical' else x
        if kind == 'multiclass':
            params['num_class'] = 3
        if kind == 'linear':
            params.update(linear_tree=True, linear_lambda=.5, num_leaves=3)
        if kind == 'categorical':
            params.update(categorical_feature=[0], max_cat_to_onehot=4, cat_smooth=0., cat_l2=0.)
        weights = [1., 2., .5, 1.] * 6
        metadata = dict(weight=np.asarray(weights, dtype=np.float32))
        if ranking:
            metadata.update(group=[4] * 6, position=np.asarray([7, 3, 7, 3] * 6, dtype=np.int32))
        initial = fit(params, rows, y, **metadata)
        source = save_model(initial, fixtures, 'refit_' + kind + '_initial', rows)
        new_rows = np.asarray([[float(a + (1 if kind != 'categorical' and i % 3 == 0 else 0)), float(b)] for i, (a, b) in enumerate(rows)])
        new_y = y[::-1] if kind in ('binary', 'multiclass') or ranking else [v + (3 if i % 2 else -2) for i, v in enumerate(y)]
        for decay in refit_decays(kind):
            started = time.monotonic()
            name = 'refit_' + kind + '_decay' + str(decay).replace('.', '')
            loaded = lgb.Booster(model_file=str(fixtures / source['model_file']))
            item = dict(name=name, operation='refit', params=params, initial_model=source,
                        original_training=dict(rows=rows.tolist(), labels=y, weights=weights, groups=[4] * 6 if ranking else None,
                                               positions=[7, 3, 7, 3] * 6 if ranking else None, num_boost_round=4),
                        rows=new_rows.tolist(), labels=new_y, weights=weights, groups=[4] * 6 if ranking else None,
                        positions=None, initial_scores=None, decay_rate=decay,
                        leaf_pred=predict(loaded, new_rows, pred_leaf=True))
            try:
                result = loaded.refit(new_rows, np.asarray(new_y, dtype=np.float32), decay_rate=decay,
                                      weight=np.asarray(weights, dtype=np.float32), group=item['groups'], dataset_params=params,
                                      categorical_feature=[0] if kind == 'categorical' else 'auto')
            except (ValueError, lgb.basic.LightGBMError) as error:
                item.update(error_type=type(error).__name__, error=str(error))
            else:
                item['effective_params'] = dict(result.params)
                item['final_model'] = save_model(result, fixtures, name, new_rows)
                if kind in EXTRA_REFIT_PARAMS:
                    reloaded = lgb.Booster(model_file=str(fixtures / item['final_model']['model_file']))
                    item['backend_params'] = {key: reloaded.params[key] for key in
                                              ('boosting', 'objective', 'data_sample_strategy', 'bagging_by_query', 'refit_decay_rate')}
            cases.append(item)
            check_resources(started)
    params = dict(BASE, objective='regression')
    model = fit(params)
    initial = save_model(model, fixtures, 'operations_rollback_initial', x)
    steps = []
    for i in range(6):
        before = model.current_iteration()
        model.rollback_one_iter()
        steps.append(dict(call=i, before_iteration=before, model=save_model(model, fixtures, f'operations_rollback_{i}', x, num_iteration=-1)))
    cases.append(dict(name='rollback', operation='rollback', params=params, rows=x.tolist(), labels=labels,
                      num_boost_round=4, initial_model=initial, steps=steps))
    model = fit(params)
    source = save_model(model, fixtures, 'operations_leaf_initial', x)
    old = model.get_leaf_output(0, 0)
    leaf_before = predict(model, x, pred_leaf=True)
    model.set_leaf_output(0, 0, 3.25)
    cases.append(dict(name='leaf_output', operation='set_leaf_output', params=params, rows=x.tolist(), labels=labels,
                      num_boost_round=4, initial_model=source, tree_id=0, leaf_id=0, old_output=old, set_output=3.25,
                      get_output=model.get_leaf_output(0, 0), leaves_before=leaf_before,
                      final_model=save_model(model, fixtures, 'operations_leaf_final', x)))
    for kind in ('binary', 'multiclass', 'linear'):
        params = dict(BASE, objective='binary' if kind == 'binary' else 'multiclass' if kind == 'multiclass' else 'regression')
        y = [int(v > 0) for v in labels] if kind == 'binary' else [i % 3 for i in range(len(x))] if kind == 'multiclass' else labels
        if kind == 'multiclass':
            params['num_class'] = 3
        if kind == 'linear':
            params['linear_tree'] = True
        model = fit(params, y=y)
        source = save_model(model, fixtures, 'operations_predict_early_' + kind, x)
        loaded = lgb.Booster(model_file=str(fixtures / source['model_file']))
        probes = []
        for frequency, margin in ((1, 0.), (2, .5), (3, 10.)):
            options = dict(pred_early_stop=True, pred_early_stop_freq=frequency, pred_early_stop_margin=margin)
            try:
                result = dict(raw=predict(loaded, x, raw_score=True, **options), transformed=predict(loaded, x, **options),
                              leaves=predict(loaded, x, pred_leaf=True, **options))
            except (ValueError, lgb.basic.LightGBMError) as error:
                result = dict(error_type=type(error).__name__, error=str(error))
            probes.append(dict(params=options, **result))
        if kind == 'linear':
            try:
                result = dict(contributions=predict(loaded, x, pred_contrib=True))
            except lgb.basic.LightGBMError as error:
                result = dict(error_type=type(error).__name__, error=str(error))
            probes.append(dict(params=dict(pred_contrib=True), **result))
        cases.append(dict(name='predict_early_' + kind, operation='prediction_early_stop', params=params, rows=x.tolist(),
                          labels=y, num_boost_round=4, initial_model=source, probes=probes,
                          prefix_raw=[predict(loaded, x, raw_score=True, num_iteration=count) for count in range(1, 5)]))
    started = time.monotonic()
    params = dict(BASE, objective='regression', num_leaves=2)
    rows = np.asarray([[-2.], [-1.], [1.], [2.]])
    extreme_labels = np.asarray([-1e20, -1e20, 1e20, 1e20], dtype=np.float32)
    model = fit(params, rows, extreme_labels.tolist())
    source = save_model(model, fixtures, 'operations_extreme_gain', rows)
    loaded = lgb.Booster(model_file=str(fixtures / source['model_file']))
    cases.append(exceptional(dict(name='extreme_f32_gain', operation='extreme_gain_diagnostic', params=params,
                                  rows=rows.tolist(), labels=extreme_labels.tolist(), num_boost_round=4, final_model=source,
                                  model_dump=loaded.dump_model(), contributions=predict(loaded, rows, pred_contrib=True),
                                  importance_split=loaded.feature_importance(importance_type='split').tolist(),
                                  importance_gain=loaded.feature_importance(importance_type='gain').tolist(),
                                  nonfinite_encoding='Infinity, -Infinity, NaN strings represent actual nonfinite native outputs')))
    check_resources(started)
    # Preserve the original 36 cases and their order; append new native evidence.
    extra_names = {'refit_' + k + '_decay' + str(d).replace('.', '') for k in EXTRA_REFIT_PARAMS for d in refit_decays(k)}
    cases = [c for c in cases if c['name'] not in extra_names] + [c for c in cases if c['name'] in extra_names]
    helper, _ = compile_helper('refit_driver')
    request = []
    for c in cases:
        if c['operation'] != 'refit' or 'final_model' not in c:
            continue
        request.append(dict(name=c['name'], params=dict(c['params'], refit_decay_rate=c['decay_rate']),
                            rows=c['rows'], labels=c['labels'], weights=c['weights'], groups=c['groups'],
                            leaf_pred=c['leaf_pred'], initial_model_path=str(fixtures / c['initial_model']['model_file']),
                            output_model_path=str(OUT / (c['name'] + '.txt'))))
    save_json(OUT / 'refit-request.json', dict(cases=request))
    subprocess.run([str(helper), str(OUT / 'refit-request.json'), str(OUT)], check=True, timeout=300)
    observed = json.loads((OUT / 'refit_internal.json').read_text())
    by_name = {c['name']: c for c in observed['cases']}
    for c in cases:
        if c['name'] not in by_name:
            continue
        native_model = lgb.Booster(model_file=str(OUT / (c['name'] + '.txt')))
        assert predict(native_model, np.asarray(c['rows']), raw_score=True) == c['final_model']['raw'], c['name']
        c['refit_steps'] = by_name[c['name']]['steps']
        c['training_scores_after'] = by_name[c['name']]['training_scores_after']
    check_resources(started, children=True)
    assert {c['name'] for c in cases} == NAMES
    save_json(fixtures / 'prediction_operations.json', dict(schema_version=1, upstream_commit=COMMIT, provenance=provenance(record), cases=cases))
    print(f'Generated {len(cases)} actual native model-operation cases')


if __name__ == '__main__':
    generate()
